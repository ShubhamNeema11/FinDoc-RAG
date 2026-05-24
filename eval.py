"""
FinRAG RAGAS Evaluation CLI
============================

Runs faithfulness, answer_relevancy, and context_precision on a random
sample of queries, records the result in the regression CSV, and optionally
plots a comparison chart between configs.

Examples
--------
# Baseline eval — 20 queries, no MultiQuery expansion
python eval.py --dataset financebench --config baseline --n 20 --no-multiquery

# MultiQuery eval — same 20 queries (same seed → same sample)
python eval.py --dataset financebench --config multiquery_k10 --n 20

# Just compare recorded configs (no new eval)
python eval.py --compare --dataset financebench --chart results/ragas_comparison.png

# Show historical trend for one config
python eval.py --trend --dataset financebench --config baseline

Flags
-----
--dataset        DATASET_CONFIGS key (default: financebench)
--config         Run label saved to CSV (default: baseline)
--n              Queries to evaluate (default: 20, ~5-8 min with Groq)
--seed           Random seed for sampling (default: 42 — reproducible)
--top-k          Docs retrieved per query (default: 5)
--no-multiquery  Disable LLM query expansion
--rebuild        Force re-embed corpus (ignore ChromaDB cache)
--chart          PNG path for comparison bar chart
--compare        Print config comparison table (no new eval)
--trend          Print trend for --config (no new eval)
--log-level      Logging verbosity (default: INFO)
--csv            Override regression CSV path
"""

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="eval",
        description="Run RAGAS evaluation and track regression.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Target ────────────────────────────────────────────────────────────
    from finrag.config import DATASET_CONFIGS  # imported here to avoid top-level cost
    p.add_argument(
        "--dataset", "-d",
        default="financebench",
        choices=list(DATASET_CONFIGS),
        metavar="NAME",
        help=f"Dataset to evaluate. Choices: {', '.join(DATASET_CONFIGS)}  (default: financebench)",
    )
    p.add_argument(
        "--config", "-c",
        default="baseline",
        metavar="LABEL",
        help="Config label saved to regression CSV (default: baseline)",
    )

    # ── Eval options ──────────────────────────────────────────────────────
    p.add_argument("--n", type=int, default=20, metavar="N",
                   help="Number of queries to evaluate (default: 20)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for query sampling (default: 42)")
    p.add_argument("--top-k", type=int, default=5, metavar="K",
                   help="Docs retrieved per query (default: 5)")
    p.add_argument("--no-multiquery", action="store_true", default=False,
                   help="Disable LLM query expansion (faster, no extra LLM calls)")
    p.add_argument("--rebuild", action="store_true", default=False,
                   help="Force re-embed corpus even if ChromaDB cache exists")
    p.add_argument(
        "--provider",
        choices=["groq", "ollama"],
        default="groq",
        help="LLM provider for generation and RAGAS judge (default: groq)",
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        metavar="MODEL",
        help=(
            "Model name override. "
            "Defaults: groq=llama-3.3-70b-versatile, ollama=llama3.1:8b. "
            "Example: --provider ollama --model mistral:7b"
        ),
    )

    # ── Reporting ─────────────────────────────────────────────────────────
    p.add_argument("--chart", type=Path, default=None, metavar="PATH",
                   help="Save comparison bar chart to this PNG path")
    p.add_argument("--compare", action="store_true", default=False,
                   help="Print config comparison table (no new evaluation)")
    p.add_argument("--trend", action="store_true", default=False,
                   help="Print metric trend for --config (no new evaluation)")
    p.add_argument("--csv", type=Path, default=None, metavar="PATH",
                   help="Override regression CSV path")

    # ── General ───────────────────────────────────────────────────────────
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING"],
                   help="Logging verbosity (default: INFO)")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    # Suppress noisy third-party loggers
    for lib in ("httpx", "httpcore", "urllib3", "chromadb",
                "sentence_transformers", "huggingface_hub"):
        logging.getLogger(lib).setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)

    # ── Regression tracker ─────────────────────────────────────────────────
    from finrag.regression import RegressionTracker
    tracker = RegressionTracker(csv_path=args.csv) if args.csv else RegressionTracker()

    # ── Report-only modes ─────────────────────────────────────────────────
    if args.compare:
        tracker.compare_configs(
            dataset=args.dataset,
            output_path=args.chart,
            show=False,
        )
        return

    if args.trend:
        tracker.trend(
            dataset=args.dataset,
            config_name=args.config,
            output_path=args.chart,
            show=False,
        )
        return

    # ── Full evaluation run ───────────────────────────────────────────────
    from finrag.config import DATASET_CONFIGS
    from finrag.data import load_jsonl, load_qrels, make_corpus_lookup
    from finrag.chunking import split_documents
    from finrag.models import (
        GROQ_DEFAULT_MODEL,
        OLLAMA_DEFAULT_MODEL,
        empty_cache,
        get_embedding_model,
        get_provider_eval_llm,
        get_provider_llm,
        get_reranker,
    )
    from finrag.retrieval import build_bm25_okapi, build_bm25_retriever, retrieve_and_rerank
    from finrag.vectorstore import get_vectorstore
    from finrag.generation import generate_answer

    cfg = DATASET_CONFIGS[args.dataset]
    use_multiquery = not args.no_multiquery

    logger.info("=" * 60)
    logger.info(
        "RAGAS Eval  |  dataset=%s  config=%s  n=%d  multiquery=%s",
        args.dataset, args.config, args.n, use_multiquery,
    )
    logger.info("=" * 60)

    # ── Load data ──────────────────────────────────────────────────────────
    from finrag.config import DATASET_DIR
    corpus  = load_jsonl(DATASET_DIR / cfg.corpus_file)
    queries = load_jsonl(DATASET_DIR / cfg.query_file)
    corpus_lookup = make_corpus_lookup(corpus)

    logger.info("Corpus: %d docs | Queries: %d", len(corpus), len(queries))

    # ── Build pipeline components ──────────────────────────────────────────
    chunks          = split_documents(corpus, cfg.chunk_size, cfg.chunk_overlap, cfg.dataset_type)
    embedding_model = get_embedding_model()
    reranker        = get_reranker()

    gen_llm = get_provider_llm(args.provider, args.model)
    if gen_llm is None:
        logger.error(
            "LLM unavailable — for Groq: set GROQ_API_KEY in .env; "
            "for Ollama: run `ollama serve` and `ollama pull <model>`."
        )
        sys.exit(1)

    eval_llm = get_provider_eval_llm(args.provider, args.model)
    if eval_llm is None:
        logger.error("RAGAS judge LLM unavailable.")
        sys.exit(1)

    # MultiQuery reuses gen_llm — avoids loading two separate models
    retrieval_llm = gen_llm if use_multiquery else None

    actual_model = args.model or (
        OLLAMA_DEFAULT_MODEL if args.provider == "ollama" else GROQ_DEFAULT_MODEL
    )

    logger.info(
        "Provider: %s  |  model: %s  |  multiquery: %s",
        args.provider, actual_model, use_multiquery,
    )

    vectorstore    = get_vectorstore(cfg.name, chunks, embedding_model,
                                     force_rebuild=args.rebuild)
    bm25_retriever = build_bm25_retriever(chunks.lc_docs, fetch_k=cfg.fetch_k)
    bm25_okapi     = build_bm25_okapi(chunks.texts)

    # ── Sample queries ─────────────────────────────────────────────────────
    import random
    random.seed(args.seed)

    eligible = queries
    sample   = (
        random.sample(eligible, min(args.n, len(eligible)))
        if len(eligible) > args.n
        else eligible
    )

    logger.info("Retrieving + generating for %d sampled queries …", len(sample))

    retrieved_map: dict[str, list[tuple[str, float]]] = {}
    answers_map:   dict[str, str]                     = {}

    from tqdm import tqdm

    for q in tqdm(sample, desc="  retrieve+gen", unit="query"):
        qid = q["_id"]
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
                llm=retrieval_llm,
                k=args.top_k,
                rerank_top_n=cfg.rerank_top_n,
            )
            retrieved_map[qid] = ranked

            gen_result = generate_answer(
                query=q["text"],
                retrieved=ranked,
                corpus_lookup=corpus_lookup,
                llm=gen_llm,
                top_k=args.top_k,
                multiquery=retrieval_llm is not None,
                model_name=actual_model,
            )
            answers_map[qid] = gen_result.answer

        except Exception as exc:
            logger.warning("Query %s failed: %s", qid, exc)

    empty_cache()

    # ── RAGAS evaluation ───────────────────────────────────────────────────
    # Import here — compat patch is applied inside ragas_eval module
    from finrag.ragas_eval import run_ragas_eval

    ragas_result = run_ragas_eval(
        queries=sample,
        retrieved_map=retrieved_map,
        answers_map=answers_map,
        corpus_lookup=corpus_lookup,
        llm=eval_llm,
        dataset_name=args.dataset,
        config_name=args.config,
        model_name=f"{args.provider}/{actual_model}",  # e.g. "groq/llama-3.3-70b-versatile"
        n_samples=args.n,
        sample_seed=args.seed,
    )

    # ── Print summary ──────────────────────────────────────────────────────
    print("\n" + ragas_result.summary())

    # ── Record in regression CSV ───────────────────────────────────────────
    tracker.record(ragas_result)

    # ── Optional comparison chart ──────────────────────────────────────────
    if args.chart:
        tracker.compare_configs(
            dataset=args.dataset,
            output_path=args.chart,
            show=False,
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
