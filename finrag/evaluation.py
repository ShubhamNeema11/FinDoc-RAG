"""
Retrieval evaluation — NDCG@10 with coverage reporting.
"""

import logging

import numpy as np
from sklearn.metrics import ndcg_score as sklearn_ndcg

logger = logging.getLogger(__name__)


def compute_ndcg(
    qrels: dict[str, dict[str, int]],
    results: dict[str, dict[str, float]],
    k: int = 10,
) -> float:
    """
    Compute mean NDCG@k over all queries that have ground-truth labels.

    Parameters
    ----------
    qrels   : query_id → {corpus_id: relevance_score}   (from qrels TSV)
    results : query_id → {corpus_id: retrieval_score}   (from pipeline)
    k       : rank cutoff

    Returns
    -------
    Mean NDCG@k (float). NaN if no scoreable queries exist.
    """
    ndcg_scores: list[float] = []
    total_labeled = sum(1 for qid in qrels if qid in results)

    for qid, doc_scores in results.items():
        if qid not in qrels:
            continue

        relevant    = qrels[qid]
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:k]

        if not sorted_docs:
            continue

        pred_scores = [s for _, s in sorted_docs]
        true_rel    = [relevant.get(doc_id, 0) for doc_id, _ in sorted_docs]

        # Skip queries where no relevant doc appeared in top-k
        if sum(true_rel) == 0:
            continue

        ndcg_scores.append(
            float(sklearn_ndcg([true_rel], [pred_scores], k=k))
        )

    if not ndcg_scores:
        logger.warning("No scoreable queries found — check qrels alignment.")
        return float("nan")

    covered = len(ndcg_scores)
    logger.info(
        "Coverage: %d / %d queries with ≥1 relevant doc in top-%d  (%.1f%%)",
        covered, total_labeled, k, 100 * covered / max(total_labeled, 1),
    )
    return float(np.mean(ndcg_scores))
