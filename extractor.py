# extractor.py
from utils import find_numeric_strings
from table_parser import answer_numeric_question_from_tables

def extract_numbers_from_context(context, max_results=20):
    numbers = find_numeric_strings(context)
    seen = set()
    out = []
    for n in numbers:
        if n not in seen:
            seen.add(n)
            out.append(n)
        if len(out) >= max_results:
            break
    return out

def numeric_pipeline(collection_name, question, context):
    try:
        table_result = answer_numeric_question_from_tables(collection_name, question)
    except Exception:
        table_result = None

    if table_result:
        return table_result["answer_text"] + f"\nSource: table {table_result['evidence']}"
    # fallback: regex
    found = extract_numbers_from_context(context, max_results=50)
    if not found:
        return "Not found in document."
    lines = ["Explicit numeric values found in the retrieved context (verbatim):"]
    for n in found:
        lines.append(f"- {n}")
    lines.append("\n(These are verbatim matches from extracted text. Do not infer or calculate.)")
    return "\n".join(lines)
