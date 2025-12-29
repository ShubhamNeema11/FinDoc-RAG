# rag.py
import chromadb
import requests
import re
from extractor import numeric_pipeline
from table_parser import load_tables_metadata

# -----------------------
# Configuration
# -----------------------
chroma_client = chromadb.Client()
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma:7b"

UNIFIED_PROMPT = """
You are answering a question about a SINGLE financial document.

CRITICAL RULES:
- All context comes from the SAME document
- Synthesize a unified, document-level answer
- Use ONLY the provided context
- Do NOT invent facts or numbers
- If the document does not support the question, say "Not found in document"
- If the text is mostly compliance or audit boilerplate, say so clearly
- Be conservative and professional

Context:
{context}

Question:
{question}

Answer in clear, concise bullet points.
"""

def call_ollama(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=90)
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except Exception as e:
        return f"LLM error: {e}"

def build_context_from_results(results):
    docs = results["documents"][0]
    metadatas = results.get("metadatas", [[]])[0]

    context_blocks = []
    sources = []

    for i, doc in enumerate(docs):
        meta = metadatas[i] if i < len(metadatas) else None
        if meta and isinstance(meta, dict):
            source = meta.get("source", f"Source_{i}")
            section = meta.get("section", "other")
        else:
            source = f"Source_{i}"
            section = "other"

        context_blocks.append(doc)
        sources.append({
            "source": source,
            "chunk_index": i,
            "section": section,
            "text_snippet": doc[:400]
        })

    context = "\n\n".join(context_blocks)
    return context, sources

def analyze_context(context: str):
    has_numbers = bool(re.search(r"\d", context))

    boilerplate_markers = [
        "internal control over financial reporting",
        "audited",
        "coso",
        "xbrl",
        "form 10-k",
        "opinion of",
        "independent registered public accounting firm"
    ]

    narrative_markers = [
        "business",
        "strategy",
        "operations",
        "management",
        "risk",
        "performance",
        "growth",
        "segment",
        "md&a",
        "results of operations"
    ]

    lower_ctx = context.lower()
    is_boilerplate = any(m in lower_ctx for m in boilerplate_markers)
    has_narrative = any(m in lower_ctx for m in narrative_markers)

    return {
        "has_numbers": has_numbers,
        "is_boilerplate": is_boilerplate,
        "has_narrative": has_narrative
    }

def ask_question(collection_name: str, question: str, k: int = 8):
    collection = chroma_client.get_collection(collection_name)
    q_lower = question.lower()
    want_summary = any(x in q_lower for x in ["summarize", "summary", "overview", "tl;dr", "summarize the"])

    if want_summary:
        try:
            results = collection.query(
                query_texts=[question],
                n_results=k,
                where={"section": {"$in": ["business", "mdna"]}}
            )
            if not results["documents"][0] or all(not d for d in results["documents"][0]):
                results = collection.query(query_texts=[question], n_results=k)
        except Exception:
            results = collection.query(query_texts=[question], n_results=k)
    else:
        results = collection.query(query_texts=[question], n_results=k)

    context, sources = build_context_from_results(results)
    evidence = analyze_context(context)

    # If numeric-style question: try table-first numeric pipeline (deterministic)
    numeric_indicators = ["how much", "what is", "what was", "exact", "figure", "amount", "$",
                          "net sales", "free cash flow", "revenue", "operating income", "net income", "eps", "ebitda", "cash"]
    if any(x in q_lower for x in numeric_indicators):
        answer = numeric_pipeline(collection_name, question, context)
    elif evidence["is_boilerplate"] and not evidence["has_narrative"]:
        answer = (
            "The retrieved sections primarily contain audit/ compliance disclosures. "
            "A business or performance-focused summary is not present in the retrieved content."
        )
    else:
        prompt = UNIFIED_PROMPT.format(context=context, question=question)
        answer = call_ollama(prompt)

    # Safety: if model returned numbers but evidence says none -> refuse
    if not evidence["has_numbers"] and any(ch.isdigit() for ch in answer):
        answer = (
            "The document discusses this topic qualitatively. "
            "Exact numerical values are not explicitly stated in the retrieved context."
        )

    # Also attach readiness info: how many tables exist for this collection
    tables_meta = load_tables_metadata(collection_name)
    readiness = {
        "num_tables": len(tables_meta),
        "num_retrieved_chunks": len(sources)
    }

    return {
        "answer": answer,
        "sources": sources,
        "readiness": readiness
    }
