"""
Ad-hoc document ingestion — turns a user-uploaded report (PDF or plain
text) into the same corpus shape the rest of the pipeline already expects
([{"_id", "title", "text"}, ...]), then indexes it into a session-scoped
Qdrant tenant so it never mixes with the preloaded benchmark corpora.

PDF tables are extracted separately from body text (via pdfplumber, which
detects table geometry) and serialised into header-value sentences, e.g.
"Revenue was $4.2 billion, Fiscal Year was 2023." — this preserves which
number belongs to which column, which plain text extraction destroys.
"""

import logging
import uuid
from io import BytesIO

from .chunking import split_documents
from .vectorstore import get_upload_vectorstore

logger = logging.getLogger(__name__)


def _extract_pdf(file_bytes: bytes) -> tuple[str, list[str]]:
    """Return (body_text, table_sentence_chunks) from a PDF."""
    import pdfplumber

    body_pages: list[str] = []
    table_chunks: list[str] = []

    with pdfplumber.open(BytesIO(file_bytes)) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                body_pages.append(text)

            try:
                tables = page.extract_tables()
            except Exception:
                tables = []

            for table in tables:
                if not table or len(table) < 2:
                    continue
                headers = table[0]
                sentences = []
                for row in table[1:]:
                    if not row:
                        continue
                    parts = [
                        f"{str(h).strip()} was {str(v).strip()}"
                        for h, v in zip(headers, row)
                        if h and v
                    ]
                    if parts:
                        sentences.append(", ".join(parts) + ".")
                if sentences:
                    table_chunks.append(
                        f"Table on page {page_no}:\n" + "\n".join(sentences)
                    )

    return "\n\n".join(body_pages), table_chunks


def extract_text(file_bytes: bytes, filename: str) -> tuple[str, list[str]]:
    """Extract (body_text, table_chunks) from an uploaded PDF or plain-text file."""
    if filename.lower().endswith(".pdf"):
        body_text, table_chunks = _extract_pdf(file_bytes)
    else:
        body_text, table_chunks = file_bytes.decode("utf-8", errors="ignore"), []

    if not body_text.strip() and not table_chunks:
        raise ValueError(f"No extractable text found in '{filename}'.")
    return body_text, table_chunks


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

    Body text and detected tables are each turned into their own corpus
    "documents" (tables tagged in the title) so table-derived chunks stay
    citable and distinguishable from body-text chunks in generated answers.

    Returns
    -------
    (session_id, corpus_lookup, vectorstore, chunk_result)
    """
    session_id = session_id or new_session_id()
    body_text, table_chunks = extract_text(file_bytes, filename)

    corpus = []
    if body_text.strip():
        corpus.append({"_id": f"{session_id}_doc0", "title": filename, "text": body_text})
    for i, table_text in enumerate(table_chunks):
        corpus.append({
            "_id": f"{session_id}_table{i}",
            "title": f"{filename} — table {i + 1}",
            "text": table_text,
        })

    corpus_lookup = {d["_id"]: d for d in corpus}

    chunks = split_documents(
        corpus, chunk_size=chunk_size, chunk_overlap=chunk_overlap, dataset_type="passage",
    )

    vectorstore = get_upload_vectorstore(session_id, embedding_model)
    vectorstore.add_texts(
        texts=chunks.texts,
        metadatas=[{"id": oid} for oid in chunks.original_ids],
        ids=chunks.chroma_ids,
    )

    logger.info(
        "Ingested '%s' → session '%s' (%d chunks, %d tables).",
        filename, session_id, len(chunks.texts), len(table_chunks),
    )
    return session_id, corpus_lookup, vectorstore, chunks
