"""
End-to-end retrieval pipeline orchestrator.

Usage
-----
from finrag.pipeline import run_dataset, run_all

# Single dataset
result = run_dataset("financebench")
print(f"NDCG@10: {result.ndcg:.4f}")

# All datasets
results = run_all()
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv
from tqdm import tqdm

from .chunking import split_documents
from .config import DATASET_CONFIGS, DATASET_DIR, DatasetConfig
from .data import load_jsonl, load_qrels, make_corpus_lookup
from .evaluation import compute_ndcg
from .generation import GenerationResult, generate_answer
from .models import empty_cache, get_embedding_model, get_llm, get_reranker
from .retrieval import build_bm25_okapi, build_bm25_retriever, retrieve_and_rerank
from .vectorstore import get_vectorstore

load_dotenv()
logger = logging.getLogger(__name__)


@dataclass
class DatasetResult:
    """Retrieval scores and run metadata for one dataset."""
    name:         str
    ndcg:         float                           # NDCG@10 (NaN if no qrels)
    results:      dict[str, dict[str, float]]     # query_id → {corpus_id: score}
    num_queries:  int
    num_chunks:   int
    elapsed_sec:  float
    errors:       list[str] = field(default_factory=list)


def run_dataset(
    dataset_name: str,
    *,
    use_multiquery: bool = True,
    force_rebuild: bool = False,
    top_k: int = 10,
) -> DatasetResult:
    """
    Run the full retrieval pipeline on a single dataset.

    Parameters
    ----------
    dataset_name   : key from DATASET_CONFIGS (e.g. "financebench")
    use_multiquery : use LLM query expansion if GROQ_API_KEY is available
    force_rebuild  : re-embed corpus even if a ChromaDB cache exists
    top_k          : documents to return per query
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Valid options: {list(DATASET_CONFIGS)}"
        )

    cfg: DatasetConfig = DATASET_CONFIGS[dataset_name]
    t_start = time.time()

    logger.info("=" * 60)
    logger.info(
        "Dataset: %s  |  type=%s  |  chunk=%d",
        cfg.name, cfg.dataset_type, cfg.chunk_size,
    )
    logger.info("=" * 60)

    # ── Load ───────────────────────────────────────────────────────────────
    corpus  = load_jsonl(DATASET_DIR / cfg.corpus_file)
    queries = load_jsonl(DATASET_DIR / cfg.query_file)
    qrels   = load_qrels(DATASET_DIR / cfg.qrels_file)

    logger.info(
        "Corpus: %d docs | Queries: %d | Labeled: %d",
        len(corpus), len(queries), len(qrels),
    )

    # ── Chunk ──────────────────────────────────────────────────────────────
    chunks = split_documents(
        corpus,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        dataset_type=cfg.dataset_type,
    )

    # ── Models (lazy singletons — loaded once, shared across datasets) ─────
    embedding_model = get_embedding_model()
    reranker        = get_reranker()
    llm             = get_llm() if use_multiquery else None

    # ── Vector store ───────────────────────────────────────────────────────
    vectorstore = get_vectorstore(
        dataset_name=cfg.name,
        chunk_result=chunks,
        embedding_model=embedding_model,
        force_rebuild=force_rebuild,
    )

    # ── BM25 (built once per dataset) ─────────────────────────────────────
    logger.info("Building BM25 indexes …")
    bm25_retriever = build_bm25_retriever(chunks.lc_docs, fetch_k=cfg.fetch_k)
    bm25_okapi     = build_bm25_okapi(chunks.texts)

    # ── Per-query retrieval ────────────────────────────────────────────────
    logger.info(
        "Retrieving %d queries  (fetch_k=%d, rerank_top_n=%d) …",
        len(queries), cfg.fetch_k, cfg.rerank_top_n,
    )

    dataset_results: dict[str, dict[str, float]] = {}
    errors: list[str] = []

    for q in tqdm(queries, desc=f"  {cfg.name}", unit="query", leave=True):
        try:
            ranked = retrieve_and_rerank(
                vectorstore=vectorstore,
                bm25_retriever=bm25_retriever,
                bm25_okapi=bm25_okapi,
                chunk_result=chunks,
                query=q["text"],
                reranker=reranker,
                dataset_type=cfg.dataset_type,
                fetch_k=cfg.fetch_k,
                llm=llm,
                k=top_k,
                rerank_top_n=cfg.rerank_top_n,
            )
            dataset_results[q["_id"]] = {doc_id: float(s) for doc_id, s in ranked}
        except Exception as exc:
            logger.warning("Query %s failed: %s", q["_id"], exc)
            errors.append(f"{q['_id']}: {exc}")
            dataset_results[q["_id"]] = {}

    # ── Evaluate ──────────────────────────────────────────────────────────
    ndcg = float("nan")
    if qrels:
        ndcg = compute_ndcg(qrels, dataset_results, k=top_k)
        logger.info("NDCG@%d: %.4f", top_k, ndcg)
    else:
        logger.info("No qrels available — skipping NDCG evaluation.")

    elapsed = time.time() - t_start
    logger.info("Finished '%s' in %.1f s", cfg.name, elapsed)

    empty_cache()

    return DatasetResult(
        name=cfg.name,
        ndcg=ndcg,
        results=dataset_results,
        num_queries=len(queries),
        num_chunks=len(chunks.texts),
        elapsed_sec=elapsed,
        errors=errors,
    )


def run_all(
    datasets: Optional[list[str]] = None,
    **kwargs,
) -> dict[str, DatasetResult]:
    """
    Run the pipeline over multiple (or all) datasets sequentially.

    Parameters
    ----------
    datasets : subset of DATASET_CONFIGS keys — defaults to all 7.
    **kwargs : forwarded to run_dataset.
    """
    names = datasets or list(DATASET_CONFIGS)
    all_results: dict[str, DatasetResult] = {}

    for name in names:
        all_results[name] = run_dataset(name, **kwargs)

    # ── Summary table ──────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 55)
    logger.info("%-16s  %8s  %8s  %10s", "Dataset", "NDCG@10", "Queries", "Time (s)")
    logger.info("-" * 55)
    for name, res in all_results.items():
        ndcg_str = f"{res.ndcg:.4f}" if res.ndcg == res.ndcg else "   N/A  "
        logger.info(
            "%-16s  %8s  %8d  %10.1f",
            name, ndcg_str, res.num_queries, res.elapsed_sec,
        )
    logger.info("=" * 55)

    return all_results


# ── Interactive single-query RAG ──────────────────────────────────────────────

def run_rag_query(
    query:         str,
    dataset_name:  str  = "financebench",
    *,
    top_k:         int  = 5,
    use_multiquery: bool = True,
    force_rebuild: bool = False,
) -> GenerationResult:
    """
    Full RAG for a single query: retrieve → generate → cite.

    This is the primary demo / MCP entry point. Reuses cached ChromaDB and
    model singletons so repeated calls are fast (no re-embedding).

    Parameters
    ----------
    query          : natural language financial question
    dataset_name   : which corpus to search (default: "financebench")
    top_k          : docs to retrieve and pass to the LLM (default: 5)
    use_multiquery : use LLM query expansion if GROQ_API_KEY is set
    force_rebuild  : rebuild ChromaDB index even if cache exists

    Returns
    -------
    GenerationResult with .answer (str) and .sources (list[CitedSource])

    Example
    -------
    >>> from finrag.pipeline import run_rag_query
    >>> result = run_rag_query("What was Apple's total revenue in FY2022?")
    >>> print(result.pretty())
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Valid options: {list(DATASET_CONFIGS)}"
        )

    cfg = DATASET_CONFIGS[dataset_name]

    # ── Load corpus + build lookup ─────────────────────────────────────────
    corpus        = load_jsonl(DATASET_DIR / cfg.corpus_file)
    corpus_lookup = make_corpus_lookup(corpus)

    # ── Chunk + models (cached after first call) ───────────────────────────
    chunks          = split_documents(corpus, cfg.chunk_size, cfg.chunk_overlap, cfg.dataset_type)
    embedding_model = get_embedding_model()
    reranker        = get_reranker()
    llm             = get_llm() if use_multiquery else None

    # ── Vector store + BM25 (vector store cached on disk) ─────────────────
    vectorstore    = get_vectorstore(cfg.name, chunks, embedding_model, force_rebuild=force_rebuild)
    bm25_retriever = build_bm25_retriever(chunks.lc_docs, fetch_k=cfg.fetch_k)
    bm25_okapi     = build_bm25_okapi(chunks.texts)

    # ── Retrieve ───────────────────────────────────────────────────────────
    retrieved = retrieve_and_rerank(
        vectorstore=vectorstore,
        bm25_retriever=bm25_retriever,
        bm25_okapi=bm25_okapi,
        chunk_result=chunks,
        query=query,
        reranker=reranker,
        dataset_type=cfg.dataset_type,
        fetch_k=cfg.fetch_k,
        llm=llm,
        k=top_k,
        rerank_top_n=cfg.rerank_top_n,
    )

    used_multiquery = llm is not None

    # ── Generate ───────────────────────────────────────────────────────────
    gen_llm = get_llm()   # always use LLM for generation (separate from retrieval)
    if gen_llm is None:
        raise RuntimeError(
            "GROQ_API_KEY not set — generation requires an LLM. "
            "Add it to .env and retry."
        )

    return generate_answer(
        query=query,
        retrieved=retrieved,
        corpus_lookup=corpus_lookup,
        llm=gen_llm,
        top_k=top_k,
        multiquery=used_multiquery,
    )
