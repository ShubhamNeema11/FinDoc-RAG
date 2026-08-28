"""
Qdrant vector store — single shared collection, multi-tenant via a
`tenant_id` payload field (Qdrant's recommended multi-tenancy pattern).

Each of the 7 benchmark datasets is one tenant (tenant_id="financebench", …).
Each user-uploaded document gets its own tenant_id (a session UUID), so
ad-hoc uploads never mix with the benchmark corpora or with each other,
without needing a new collection per dataset/upload.
"""

import logging
import os
import uuid

from qdrant_client import QdrantClient, models
from tqdm import tqdm

from .chunking import ChunkResult

logger = logging.getLogger(__name__)

_BATCH_SIZE  = 64
_COLLECTION  = "finrag"

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        url     = os.getenv("QDRANT_URL")
        api_key = os.getenv("QDRANT_API_KEY")
        if not url or not api_key:
            raise RuntimeError("QDRANT_URL / QDRANT_API_KEY not set — add them to .env")
        _client = QdrantClient(url=url, api_key=api_key)
    return _client


def _ensure_collection(client: QdrantClient, dimension: int) -> None:
    if not client.collection_exists(_COLLECTION):
        logger.info("Creating Qdrant collection '%s' (dim=%d) …", _COLLECTION, dimension)
        client.create_collection(
            collection_name=_COLLECTION,
            vectors_config=models.VectorParams(
                size=dimension, distance=models.Distance.COSINE,
            ),
        )
        # Payload index on tenant_id — required for efficient multi-tenant filtering.
        client.create_payload_index(
            collection_name=_COLLECTION,
            field_name="tenant_id",
            field_schema=models.KeywordIndexParams(
                type=models.KeywordIndexType.KEYWORD,
                is_tenant=True,
            ),
        )


class QdrantStore:
    """Adapter exposing the similarity_search / max_marginal_relevance_search
    surface retrieval.py expects, scoped to one tenant_id."""

    def __init__(self, client: QdrantClient, embedding_model, tenant_id: str):
        self._client    = client
        self._embed     = embedding_model
        self._tenant_id = tenant_id

    def _tenant_filter(self) -> models.Filter:
        return models.Filter(
            must=[models.FieldCondition(
                key="tenant_id", match=models.MatchValue(value=self._tenant_id),
            )]
        )

    def similarity_search(self, query: str, k: int = 10) -> list["_Doc"]:
        vec = self._embed.embed_query(query)
        hits = self._client.query_points(
            collection_name=_COLLECTION,
            query=vec,
            query_filter=self._tenant_filter(),
            limit=k,
        ).points
        return [_Doc(h.payload["text"], {"id": h.payload["id"]}) for h in hits]

    def max_marginal_relevance_search(
        self, query: str, k: int = 10, fetch_k: int = 30, lambda_mult: float = 0.7,
    ) -> list["_Doc"]:
        import numpy as np

        query_vec = np.array(self._embed.embed_query(query))
        hits = self._client.query_points(
            collection_name=_COLLECTION,
            query=query_vec.tolist(),
            query_filter=self._tenant_filter(),
            limit=fetch_k,
            with_vectors=True,
        ).points
        if not hits:
            return []

        vecs = np.array([h.vector for h in hits])
        sims_to_query = vecs @ query_vec / (
            np.linalg.norm(vecs, axis=1) * np.linalg.norm(query_vec) + 1e-10
        )

        selected: list[int] = []
        candidates = list(range(len(hits)))

        while candidates and len(selected) < k:
            if not selected:
                best = max(candidates, key=lambda i: sims_to_query[i])
            else:
                sel_vecs = vecs[selected]
                redundancy = np.max(
                    (vecs[candidates] @ sel_vecs.T) / (
                        np.linalg.norm(vecs[candidates], axis=1, keepdims=True)
                        * np.linalg.norm(sel_vecs, axis=1) + 1e-10
                    ),
                    axis=1,
                )
                mmr_scores = (
                    lambda_mult * sims_to_query[candidates]
                    - (1 - lambda_mult) * redundancy
                )
                best = candidates[int(np.argmax(mmr_scores))]
            selected.append(best)
            candidates.remove(best)

        return [_Doc(hits[i].payload["text"], {"id": hits[i].payload["id"]}) for i in selected]

    def add_texts(self, texts: list[str], metadatas: list[dict], ids: list[str]) -> None:
        vectors = self._embed.embed_documents(texts)
        points = [
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{self._tenant_id}:{ids[i]}")),
                vector=vectors[i],
                payload={"id": metadatas[i]["id"], "text": texts[i], "tenant_id": self._tenant_id},
            )
            for i in range(len(texts))
        ]
        self._client.upsert(collection_name=_COLLECTION, points=points)

    def tenant_has_vectors(self) -> bool:
        count = self._client.count(
            collection_name=_COLLECTION, count_filter=self._tenant_filter(), exact=False,
        ).count
        return count > 0

    def clear_tenant(self) -> None:
        self._client.delete(
            collection_name=_COLLECTION,
            points_selector=models.FilterSelector(filter=self._tenant_filter()),
        )


class _Doc:
    """Minimal stand-in for a langchain_core.documents.Document."""
    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata     = metadata


def get_vectorstore(
    dataset_name: str,
    chunk_result: ChunkResult,
    embedding_model,
    *,
    force_rebuild: bool = False,
) -> QdrantStore:
    """
    Return a Qdrant-backed vectorstore scoped to tenant_id=dataset_name.

    If that tenant already has vectors (and force_rebuild is False), the
    existing data is reused — skipping re-embedding entirely.
    """
    client = get_client()
    dimension = len(embedding_model.embed_query("dimension probe"))
    _ensure_collection(client, dimension)

    store = QdrantStore(client, embedding_model, tenant_id=dataset_name)

    if store.tenant_has_vectors():
        if force_rebuild:
            logger.info("Clearing existing vectors for tenant '%s' …", dataset_name)
            store.clear_tenant()
        else:
            logger.info("Reusing existing Qdrant data for tenant '%s'.", dataset_name)
            return store

    logger.info(
        "Indexing '%s' (%d chunks) into Qdrant …",
        dataset_name, len(chunk_result.texts),
    )

    texts   = chunk_result.texts
    ids     = chunk_result.chroma_ids
    orig    = chunk_result.original_ids
    batches = range(0, len(texts), _BATCH_SIZE)

    for i in tqdm(batches, desc=f"  Indexing {dataset_name}", unit="batch"):
        store.add_texts(
            texts=texts[i : i + _BATCH_SIZE],
            metadatas=[{"id": oid} for oid in orig[i : i + _BATCH_SIZE]],
            ids=ids[i : i + _BATCH_SIZE],
        )

    logger.info("Vector store for '%s' built in Qdrant.", dataset_name)
    return store


def get_upload_vectorstore(
    session_id: str,
    embedding_model,
) -> QdrantStore:
    """
    Return a Qdrant-backed vectorstore scoped to an ad-hoc upload session
    (tenant_id=session_id). Caller is responsible for populating it via
    store.add_texts() and, when the session ends, store.clear_tenant().
    """
    client = get_client()
    dimension = len(embedding_model.embed_query("dimension probe"))
    _ensure_collection(client, dimension)
    return QdrantStore(client, embedding_model, tenant_id=session_id)
