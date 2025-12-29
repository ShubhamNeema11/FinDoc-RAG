# ingest.py
import os
import time
import json
import pdfplumber
import chromadb
from chromadb.utils import embedding_functions
from utils import chunk_text
import pandas as pd

embedding_function = embedding_functions.DefaultEmbeddingFunction()
chroma_client = chromadb.Client()

def extract_text_from_pdf_path(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def extract_tables_from_pdf_path(path):

    tables = []
    with pdfplumber.open(path) as pdf:
        for pageno, page in enumerate(pdf.pages, start=1):
            # try page.extract_tables() first (returns list of table rows)
            try:
                raw_tables = page.extract_tables()
            except Exception:
                raw_tables = []
            if not raw_tables:
                # try find_tables
                try:
                    found = page.find_tables()
                    raw_tables = [t.extract() for t in found] if found else []
                except Exception:
                    raw_tables = []

            for ti, tbl in enumerate(raw_tables):
                # normalize: get list of lists, convert to pandas DataFrame
                try:
                    df = pd.DataFrame(tbl)
                except Exception:
                    continue
                # drop empty columns/rows heuristically
                df = df.dropna(how='all')
                df = df.loc[:, df.notna().any()]
                if df.empty:
                    continue
                tables.append({
                    "page": pageno,
                    "table_id": f"p{pageno}_t{ti}",
                    "df": df,
                    "preview": df.head(5).to_dict(orient="list")
                })
    return tables

def classify_section(text: str) -> str:
    t = text.lower()
    if any(x in t for x in [
        "independent registered public accounting firm",
        "internal control over financial reporting",
        "audited",
        "coso",
        "xbrl",
        "opinion of",
        "independent auditors"
    ]):
        return "audit"

    if any(x in t for x in [
        "management’s discussion",
        "managements discussion",
        "md&a",
        "results of operations",
        "analysis of results",
        "financial performance"
    ]):
        return "mdna"

    if any(x in t for x in [
        "business", "business model", "our business", "segments", "products", "services", "we operate"
    ]):
        return "business"

    if any(x in t for x in [
        "risk factors", "risks", "uncertain", "uncertainties", "forward-looking statements"
    ]):
        return "risk"

    return "other"

def ingest_pdf_return_collection(file_path: str, filename_hint: str = "doc"):
    raw_text = extract_text_from_pdf_path(file_path)
    if not raw_text or raw_text.strip() == "":
        raise ValueError("No text could be extracted from the PDF.")

    chunks = chunk_text(raw_text, chunk_size=500, overlap=100)

    safe_name = filename_hint.replace(" ", "_").replace(".", "_")
    collection_name = f"financial_docs_{safe_name}_{int(time.time())}"

    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function
    )

    # Add text chunks with section metadata
    for i, chunk in enumerate(chunks):
        section = classify_section(chunk)
        meta = {"chunk_index": i, "source": filename_hint, "section": section}
        collection.add(
            documents=[chunk],
            ids=[f"{collection_name}_chunk_{i}"],
            metadatas=[meta]
        )

    # Extract tables and save them to disk for deterministic queries
    tables = extract_tables_from_pdf_path(file_path)
    tables_dir = os.path.join("data", "tables", collection_name)
    os.makedirs(tables_dir, exist_ok=True)
    tables_meta = []
    for t in tables:
        # save CSV
        csv_path = os.path.join(tables_dir, f"{t['table_id']}.csv")
        try:
            t['df'].to_csv(csv_path, index=False, header=True)
        except Exception:
            # fallback: write as JSON preview
            with open(csv_path + ".json", "w", encoding="utf-8") as f:
                json.dump(t['preview'], f)
            csv_path = csv_path + ".json"

        tables_meta.append({
            "table_id": t["table_id"],
            "page": t["page"],
            "csv_path": csv_path,
            "preview": t["preview"]
        })

    # Save tables metadata
    meta_path = os.path.join(tables_dir, "tables_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(tables_meta, f, indent=2)

    return collection_name, len(chunks)
