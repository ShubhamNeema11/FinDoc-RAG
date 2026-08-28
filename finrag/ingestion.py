"""
Ad-hoc document ingestion — turns a user-uploaded report (PDF or plain
text) into the same corpus shape the rest of the pipeline already expects
([{"_id", "title", "text"}, ...]), then indexes it into a session-scoped
Qdrant tenant so it never mixes with the preloaded benchmark corpora.
"""

import logging
import uuid
from io import BytesIO

from .chunking import split_documents
from .vectorstore import get_upload_vectorstore

logger = logging.getLogger(__name__)


def extract_text(file_bytes: bytes, filename: str) -> str:
    """Extract raw text from an uploaded PDF or plain-text file."""
    if filename.lower().endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(file_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n\n".join(pages)
    else:
        text = file_bytes.decode("utf-8", errors="ignore")

    if not text.strip():
        raise ValueError(f"No extractable text found in '{filename}'.")
    return text


def new_session_id() -> str:
    """A fresh tenant_id for one uploaded-document session."""
    return f"upload-{uuid.uuid4().hex[:12]}"


def ingest_document(
    file_bytes: bytes,
    filename: str,
    embedding_model,
    *,
    session_id: str | None = None,
    chunk_size: int = 1024,
    chunk_overlap: int = 128,
):
    """
    Extract, chunk, embed, and index an uploaded report.

    Returns
    -------
    (session_id, corpus_lookup, vectorstore, chunk_result)
    """
    session_id = session_id or new_session_id()
    text = extract_text(file_bytes, filename)

    corpus = [{"_id": f"{session_id}_doc0", "title": filename, "text": text}]
    corpus_lookup = {d["_id"]: d for d in corpus}

    chunks = split_documents(
        corpus, chunk_size=chunk_size, chunk_overlap=chunk_overlap, dataset_type="tabular",
    )

    vectorstore = get_upload_vectorstore(session_id, embedding_model)
    vectorstore.add_texts(
        texts=chunks.texts,
        metadatas=[{"id": oid} for oid in chunks.original_ids],
        ids=chunks.chroma_ids,
    )

    logger.info(
        "Ingested '%s' → session '%s' (%d chunks).",
        filename, session_id, len(chunks.texts),
    )
    return session_id, corpus_lookup, vectorstore, chunks
