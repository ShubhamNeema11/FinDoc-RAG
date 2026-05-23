"""
Retrieval pipeline — manual implementation, no broken LangChain 1.x dependencies.

Stage 1  Dense retrieval   Chroma MMR (passage) or similarity (tabular)
Stage 2  Sparse retrieval  BM25Retriever
Stage 3  RRF fusion        Reciprocal Rank Fusion, k=60
Stage 4  Best-chunk pick   BM25Okapi scores per candidate doc
Stage 5  Reranking         Cross-encoder on (query, best_chunk) pairs
"""

import logging

import nltk
from langchain_community.retrievers import BM25Retriever
from rank_bm25 import BM25Okapi

from .chunking import ChunkResult

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

logger = logging.getLogger(__name__)


# ── BM25 indexes ─────────────────────────────────────────────────────────────

def build_bm25_retriever(lc_docs, fetch_k: int) -> BM25Retriever:
    """LangChain BM25Retriever for Stage 2 retrieval."""
    return BM25Retriever.from_documents(lc_docs, k=fetch_k)


def build_bm25_okapi(texts: list[str]) -> BM25Okapi:
    """Raw BM25Okapi used in Stage 4 to select the best chunk per candidate doc."""
    return BM25Okapi([t.lower().split() for t in texts])


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

def rrf_fuse(
    ranked_lists: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    Merge multiple ranked doc-ID lists with Reciprocal Rank Fusion.

    RRF score for document d:
        score(d) = Σ  1 / (k + rank_r(d))   over all retrievers r

    k=60 is the standard constant — robust across datasets with no tuning.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ── Optional: MultiQuery expansion ───────────────────────────────────────────

def expand_query(query: str, llm) -> list[str]:
    """
    Use an LLM to generate 3 alternative phrasings of the query.
    Returns [original] + [variants] so retrieval always includes the original.
    Falls back gracefully if the LLM fails.
    """
    prompt = (
        "Generate 3 alternative search queries for the following financial question. "
        "Output only the queries, one per line, no numbering or explanation.\n\n"
        f"Original: {query}"
    )
    try:
        response = llm.invoke(prompt)
        variants = [
            line.strip()
            for line in response.content.strip().split("\n")
            if line.strip() and line.strip() != query
        ][:3]
        queries = [query] + variants
        logger.debug("MultiQuery expanded to %d variants.", len(queries))
        return queries
    except Exception as exc:
        logger.warning("MultiQuery expansion failed (%s) — using original query.", exc)
        return [query]


# ── Core retrieval function ───────────────────────────────────────────────────

def retrieve_and_rerank(
    vectorstore,
    bm25_retriever: BM25Retriever,
    bm25_okapi: BM25Okapi,
    chunk_result: ChunkResult,
    query: str,
    reranker,
    dataset_type: str,
    fetch_k: int,
    llm=None,
    k: int = 10,
    rerank_top_n: int = 30,
) -> list[tuple[str, float]]:
    """
    Full per-query retrieval pipeline.

    Parameters
    ----------
    vectorstore   : ChromaDB store for this dataset
    bm25_retriever: LangChain BM25Retriever (Stage 2)
    bm25_okapi    : BM25Okapi index for best-chunk selection (Stage 4)
    chunk_result  : output of split_documents
    query         : raw query string
    reranker      : CrossEncoder instance
    dataset_type  : "passage" | "tabular"
    fetch_k       : candidates to fetch from each retriever
    llm           : optional LLM for MultiQuery expansion (None = disabled)
    k             : final top-k to return
    rerank_top_n  : how many unique doc IDs to pass to the cross-encoder

    Returns
    -------
    List of (corpus_id, reranker_score) sorted descending.
    """
    # ── Stage 1 — Optional MultiQuery expansion ───────────────────────────
    queries = expand_query(query, llm) if llm is not None else [query]

    # ── Stage 2 — Dense retrieval (per query variant) ─────────────────────
    dense_ranked: list[list[str]] = []
    for q in queries:
        if dataset_type == "passage":
            docs = vectorstore.max_marginal_relevance_search(
                q, k=fetch_k, fetch_k=fetch_k * 3, lambda_mult=0.7
            )
        else:
            docs = vectorstore.similarity_search(q, k=fetch_k)
        dense_ranked.append([d.metadata["id"] for d in docs])

    # ── Stage 3 — BM25 retrieval ──────────────────────────────────────────
    bm25_docs   = bm25_retriever.invoke(query)          # always use original query
    bm25_ranked = [d.metadata["id"] for d in bm25_docs]

    # ── Stage 4 — RRF fusion ──────────────────────────────────────────────
    all_ranked_lists = dense_ranked + [bm25_ranked]
    fused = rrf_fuse(all_ranked_lists, k=60)

    # Deduplicate and cap at rerank_top_n
    # Build a page_content map from dense results for Stage 5 fallback
    seen: dict[str, str] = {}
    for doc_list in dense_ranked:
        pass   # doc_list has IDs only; content stored below via bm25_okapi
    # Collect content from BM25 docs (always have page_content)
    for d in bm25_docs:
        seen[d.metadata["id"]] = d.page_content

    candidate_ids: list[str] = []
    for doc_id, _ in fused:
        if doc_id not in [c for c in candidate_ids]:
            candidate_ids.append(doc_id)
        if len(candidate_ids) >= rerank_top_n:
            break

    if not candidate_ids:
        return []

    # ── Stage 5 — Best chunk per doc via BM25Okapi ────────────────────────
    chunk_scores  = bm25_okapi.get_scores(query.lower().split())
    candidate_set = set(candidate_ids)
    best_chunk: dict[str, tuple[float, str]] = {}

    for chunk_text, orig_id, score in zip(
        chunk_result.texts, chunk_result.original_ids, chunk_scores
    ):
        if orig_id not in candidate_set:
            continue
        s = float(score)
        if orig_id not in best_chunk or s > best_chunk[orig_id][0]:
            best_chunk[orig_id] = (s, chunk_text)

    for oid in candidate_ids:
        if oid not in best_chunk:
            best_chunk[oid] = (0.0, seen.get(oid, ""))

    # ── Stage 6 — Cross-encoder reranking ────────────────────────────────
    pairs  = [(query, best_chunk[oid][1]) for oid in candidate_ids]
    scores = reranker.predict(pairs, batch_size=32, show_progress_bar=False)

    ranked = sorted(
        zip(candidate_ids, scores.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked[:k]
