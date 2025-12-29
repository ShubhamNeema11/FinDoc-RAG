# table_parser.py
import os
import json
import pandas as pd
import re
from rapidfuzz import fuzz, process

TABLES_ROOT = os.path.join("data", "tables")

def load_tables_metadata(collection_name: str):
    dir_path = os.path.join(TABLES_ROOT, collection_name)
    meta_path = os.path.join(dir_path, "tables_meta.json")
    if not os.path.exists(meta_path):
        return []
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    # read CSVs into DataFrames lazily when needed
    for m in meta:
        m["dir"] = dir_path
    return meta

def find_best_table_and_column(collection_name: str, question: str, top_k_tables=3, header_score_threshold=55):
    """
    Returns best match result or None:
    {
      "meta": table_meta,
      "df": df,
      "best_header": header,
      "score": score
    }
    """
    meta = load_tables_metadata(collection_name)
    if not meta:
        return None

    q = question.lower()

    # helper to compute header match score
    def score_table(meta_item):
        # load df
        path = meta_item["csv_path"]
        try:
            if path.endswith(".json"):
                # preview only
                headers = list(meta_item.get("preview", {}).keys())
                df = None
            else:
                df = pd.read_csv(path)
                headers = list(df.columns.astype(str))
        except Exception:
            df = None
            headers = list(meta_item.get("preview", {}).keys())

        # compute best header match to question
        best_h = None
        best_score = 0
        for h in headers:
            s = fuzz.token_set_ratio(q, str(h))
            if s > best_score:
                best_score = s
                best_h = h
        return {"meta": meta_item, "df": df, "best_header": best_h, "score": best_score, "headers": headers}

    scored = [score_table(m) for m in meta]
    scored_sorted = sorted(scored, key=lambda x: x["score"], reverse=True)
    best = scored_sorted[0] if scored_sorted else None
    if best and best["score"] >= header_score_threshold:
        return best
    # No strong header match — return top candidate if we have any
    return scored_sorted[0] if scored_sorted else None

def find_year_in_question(question: str):
    yrs = re.findall(r"(?<!\d)(20\d{2})(?!\d)", question)
    return yrs  # list of years as strings, e.g., ['2024']

def lookup_value_in_table(best_table_info, question: str):
    """
    Tries to find a numeric value in the matched table for the question.
    Strategy:
      - if question contains a year, find the row with that year (in any column), then fetch value from best_header column.
      - else, if header contains 'total' or 'Q' or year-like columns, pick the first numeric column.
      - return formatted answer with source.
    """
    meta = best_table_info["meta"]
    df = best_table_info["df"]
    header = best_table_info["best_header"]
    headers = best_table_info["headers"]

    # if only preview (no df), we can't compute
    if df is None:
        return None

    # standardize columns to string
    df = df.astype(str)

    years = find_year_in_question(question)
    if years:
        year = years[0]
        # find any row that contains the year in any cell
        mask_rows = df.apply(lambda row: row.astype(str).str.contains(year, na=False).any(), axis=1)
        if mask_rows.any():
            row_idx = df[mask_rows].index[0]
            # prefer header column if header exists in df
            if header in df.columns:
                val = df.at[row_idx, header]
                return {"value": val, "row": df.loc[row_idx].to_dict(), "meta": meta, "header": header}
            else:
                # choose numeric-like column in that row
                for col in df.columns:
                    cell = df.at[row_idx, col]
                    if re.search(r"\d", str(cell)):
                        return {"value": cell, "row": df.loc[row_idx].to_dict(), "meta": meta, "header": col}
    # no year found: attempt to find the row which best matches question using fuzzy matching
    # compute best row by matching concatenated row text to question
    row_scores = []
    for idx, row in df.iterrows():
        text = " ".join(row.astype(str).tolist())
        s = fuzz.token_set_ratio(question, text)
        row_scores.append((s, idx))
    row_scores.sort(reverse=True)
    if row_scores and row_scores[0][0] > 60:
        best_idx = row_scores[0][1]
        # get numeric in header column or first numeric column
        if header and header in df.columns:
            val = df.at[best_idx, header]
            if re.search(r"\d", str(val)):
                return {"value": val, "row": df.loc[best_idx].to_dict(), "meta": meta, "header": header}
        for col in df.columns:
            cell = df.at[best_idx, col]
            if re.search(r"\d", str(cell)):
                return {"value": cell, "row": df.loc[best_idx].to_dict(), "meta": meta, "header": col}
    # nothing found
    return None

def answer_numeric_question_from_tables(collection_name: str, question: str):
    """
    High-level function: find best table, then lookup value.
    Returns dict with answer_text and evidence or None
    """
    best = find_best_table_and_column(collection_name, question)
    if not best:
        return None
    lookup = lookup_value_in_table(best, question)
    if not lookup:
        return None
    # format the answer
    meta = lookup["meta"]
    header = lookup["header"]
    value = lookup["value"]
    page = meta.get("page", "unknown")
    table_id = meta.get("table_id", "unknown")
    answer_text = f"Found value for '{header}' from table {table_id} on page {page}: {value} (verbatim from table)."
    evidence = {"table_id": table_id, "page": page, "csv_path": meta.get("csv_path")}
    return {"answer_text": answer_text, "evidence": evidence, "row": lookup.get("row")}
