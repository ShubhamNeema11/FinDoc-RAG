# app.py
import streamlit as st
from ingest import ingest_pdf_return_collection
from rag import ask_question
from table_parser import load_tables_metadata
import os
import pandas as pd

st.set_page_config(page_title="Financial Document Intelligence", layout="wide")
st.title("📈 Financial Document Intelligence — Practical Version")

if "collection_name" not in st.session_state:
    st.session_state["collection_name"] = None
if "filename_hint" not in st.session_state:
    st.session_state["filename_hint"] = None

uploaded_file = st.file_uploader("Upload a financial PDF", type=["pdf"])
col1, col2 = st.columns([2,1])

with col1:
    if uploaded_file:
        st.write(f"Uploaded file: {uploaded_file.name}")
        with st.spinner("Saving and ingesting document (may take a moment)..."):
            try:
                os.makedirs("data/uploads", exist_ok=True)
                saved_path = os.path.join("data/uploads", uploaded_file.name)
                uploaded_file.seek(0)
                with open(saved_path, "wb") as f:
                    f.write(uploaded_file.read())

                collection_name, n_chunks = ingest_pdf_return_collection(saved_path, filename_hint=uploaded_file.name)
                st.session_state["collection_name"] = collection_name
                st.session_state["filename_hint"] = uploaded_file.name
                st.success(f"Ingested {n_chunks} chunks into collection: {collection_name}")

            except Exception as e:
                st.error(f"Ingestion failed: {e}")

with col2:
    st.markdown("**Session**")
    if st.session_state["collection_name"]:
        st.write(f"Active collection: `{st.session_state['collection_name']}`")
    else:
        st.write("_No document ingested yet_")

st.markdown("---")

if not st.session_state.get("collection_name"):
    st.info("Upload a PDF to begin.")
else:
    # Show document readiness
    st.subheader("Document Readiness Report")
    tables_meta = load_tables_metadata(st.session_state["collection_name"])
    st.write(f"- Chunks ingested: (see collection name; typical range: 100–200 for 10-Ks)")
    st.write(f"- Tables extracted: {len(tables_meta)}")
    if len(tables_meta) > 0:
        st.markdown("**Preview of extracted tables:**")
        for m in tables_meta[:5]:
            st.markdown(f"- Table `{m['table_id']}` (page {m.get('page')}) — preview:")
            try:
                if m["csv_path"].endswith(".json"):
                    with open(m["csv_path"], "r", encoding="utf-8") as f:
                        preview = json.load(f)
                    st.write(preview)
                else:
                    df = pd.read_csv(m["csv_path"])
                    st.dataframe(df.head(5))
            except Exception:
                st.write(m.get("preview", {}))

    st.markdown("---")
    question = st.text_input("Ask a question about the uploaded document", key="question_input")
    if st.button("Ask") and question:
        with st.spinner("Retrieving and generating..."):
            res = ask_question(st.session_state["collection_name"], question, k=8)

        st.subheader("Answer")
        st.write(res["answer"])

        st.subheader("Retrieved source snippets (transparency)")
        for s in res["sources"]:
            st.markdown(f"**{s['source']} | chunk {s['chunk_index']} | section: {s.get('section','other')}**")
            st.write(s["text_snippet"])
            st.divider()

        st.subheader("Readiness")
        st.write(res["readiness"])

        st.info("Design note: numeric extractions use deterministic table lookup first; if not possible, verbatim numeric matches from text are returned.")
