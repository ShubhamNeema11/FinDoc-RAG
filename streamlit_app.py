"""
FinRAG — Streamlit demo

Ask questions against one of 7 preloaded financial benchmark corpora, or
upload your own company report (PDF/txt) and ask questions grounded in
that document instead.
"""

import streamlit as st
from dotenv import load_dotenv

from finrag.config import DATASET_CONFIGS
from finrag.ingestion import ingest_document, new_session_id
from finrag.models import get_embedding_model
from finrag.pipeline import run_rag_query, run_rag_query_on_upload

load_dotenv()

st.set_page_config(page_title="FinRAG", page_icon="💰", layout="centered")
st.title("💰 FinRAG — Financial Document Q&A")
st.caption("Hybrid retrieval (dense + BM25 + RRF + reranking) → grounded, cited answers.")


@st.cache_resource(show_spinner="Loading embedding model …")
def _embedding_model():
    return get_embedding_model()


mode = st.radio(
    "Data source",
    ["Preloaded benchmark dataset", "Upload a company report"],
    horizontal=True,
)

provider = st.sidebar.selectbox("LLM provider", ["groq", "ollama"], index=0)
model_override = st.sidebar.text_input("Model override (optional)", value="")
top_k = st.sidebar.slider("Documents to retrieve", 3, 10, 5)

if mode == "Preloaded benchmark dataset":
    dataset_name = st.selectbox("Dataset", list(DATASET_CONFIGS))
    use_multiquery = st.sidebar.checkbox("Use MultiQuery expansion (slower)", value=False)

    query = st.text_input("Ask a financial question", placeholder="What was FY2022 net revenue?")
    if st.button("Ask", type="primary") and query:
        with st.spinner("Retrieving and generating an answer …"):
            result = run_rag_query(
                query=query,
                dataset_name=dataset_name,
                top_k=top_k,
                use_multiquery=use_multiquery,
                provider=provider,
                model=model_override or None,
            )
        st.markdown("### Answer")
        st.write(result.answer)
        if result.sources:
            st.markdown("### Sources")
            for i, src in enumerate(result.sources, 1):
                with st.expander(f"[{i}] {src.corpus_id} — {src.title or 'untitled'}  (score={src.score:.3f})"):
                    st.write(src.excerpt)

else:
    uploaded = st.file_uploader("Upload a report (PDF or .txt)", type=["pdf", "txt"])

    if uploaded is not None:
        cache_key = f"{uploaded.name}:{uploaded.size}"
        if st.session_state.get("upload_key") != cache_key:
            with st.spinner(f"Indexing '{uploaded.name}' …"):
                embedding_model = _embedding_model()
                session_id, corpus_lookup, _vs, chunks = ingest_document(
                    uploaded.getvalue(), uploaded.name, embedding_model,
                    session_id=new_session_id(),
                )
            st.session_state["upload_key"]      = cache_key
            st.session_state["upload_session"]  = session_id
            st.session_state["upload_lookup"]   = corpus_lookup
            st.session_state["upload_chunks"]   = chunks
            st.success(f"Indexed '{uploaded.name}' ({len(chunks.texts)} chunks).")

        query = st.text_input("Ask a question about this report", placeholder="What was total revenue?")
        if st.button("Ask", type="primary") and query:
            with st.spinner("Retrieving and generating an answer …"):
                result = run_rag_query_on_upload(
                    query=query,
                    session_id=st.session_state["upload_session"],
                    corpus_lookup=st.session_state["upload_lookup"],
                    chunk_result=st.session_state["upload_chunks"],
                    top_k=top_k,
                    provider=provider,
                    model=model_override or None,
                )
            st.markdown("### Answer")
            st.write(result.answer)
            if result.sources:
                st.markdown("### Sources")
                for i, src in enumerate(result.sources, 1):
                    with st.expander(f"[{i}] {src.corpus_id} — {src.title or 'untitled'}  (score={src.score:.3f})"):
                        st.write(src.excerpt)
    else:
        st.info("Upload a PDF or text report to ask questions about it.")
