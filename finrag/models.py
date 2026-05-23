"""
Model singletons — embedding model, cross-encoder reranker, and LLM.

All models are loaded lazily on first access and cached for reuse across
datasets. Device detection prioritises: CUDA → MPS (Apple Silicon) → CPU.
"""

import logging
import os
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# ── Device detection ─────────────────────────────────────────────────────────

def get_device() -> str:
    """Return the best available compute device string."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def empty_cache() -> None:
    """Release GPU / MPS memory cache."""
    device = get_device()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


# ── Lazy singletons ───────────────────────────────────────────────────────────
_embedding_model = None
_reranker        = None
_llm             = None


def get_embedding_model():
    """
    Load FinLang/finance-embeddings-investopedia (finance-domain finetuned).
    Cached after first call.
    """
    global _embedding_model
    if _embedding_model is None:
        from langchain_huggingface import HuggingFaceEmbeddings

        device = get_device()
        logger.info("Loading embedding model on device=%s …", device)
        _embedding_model = HuggingFaceEmbeddings(
            model_name="FinLang/finance-embeddings-investopedia",
            model_kwargs={"device": device},
            encode_kwargs={"batch_size": 32, "normalize_embeddings": True},
        )
        logger.info("Embedding model ready.")
    return _embedding_model


def get_reranker():
    """
    Load BAAI/bge-reranker-v2-m3 cross-encoder (2.27 GB).
    Cached after first call.
    """
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder

        device = get_device()
        logger.info("Loading reranker on device=%s …", device)
        _reranker = CrossEncoder(
            "BAAI/bge-reranker-v2-m3",
            device=device,
            max_length=512,
        )
        logger.info("Reranker ready.")
    return _reranker


def get_llm() -> Optional[object]:
    """
    Load Groq LLM for MultiQueryRetriever (requires GROQ_API_KEY in .env).
    Returns None if the key is absent — pipeline continues without MultiQuery.
    """
    global _llm
    if _llm is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            logger.warning(
                "GROQ_API_KEY not set — MultiQuery expansion disabled. "
                "Add it to .env to enable."
            )
            return None

        from langchain_groq import ChatGroq

        logger.info("Loading Groq LLM (llama-3.3-70b-versatile) …")
        _llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            max_tokens=256,
        )
        logger.info("LLM ready.")
    return _llm


def get_eval_llm() -> Optional[object]:
    """
    Rate-limited Groq LLM for RAGAS evaluation (judge calls).

    RAGAS issues one LLM call per metric per sample — with 20 samples × 3
    metrics = 60 calls.  Groq free tier allows 30 req/min, so we cap at
    0.4 req/s (24/min) leaving headroom for MultiQuery calls running
    concurrently.

    Returns None if GROQ_API_KEY is absent.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY not set — RAGAS evaluation requires Groq.")
        return None

    from langchain_core.rate_limiters import InMemoryRateLimiter
    from langchain_groq import ChatGroq

    rate_limiter = InMemoryRateLimiter(
        requests_per_second=0.4,   # 24 req/min — safe under 30 req/min free-tier cap
        check_every_n_seconds=0.1,
        max_bucket_size=5,
    )

    logger.info(
        "Loading rate-limited Groq eval LLM (0.4 req/s — ~5-8 min for 20 samples) …"
    )
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=1024,           # larger budget for judge reasoning
        rate_limiter=rate_limiter,
    )
