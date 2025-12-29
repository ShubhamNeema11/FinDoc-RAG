# utils.py
import tiktoken
import re

def chunk_text(text, chunk_size=500, overlap=50):
    """
    Token-aware chunking using tiktoken.
    Returns list of text chunks.
    """
    try:
        enc = tiktoken.get_encoding("cl100k_base")
    except Exception:
        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunks.append(" ".join(words[i:i+chunk_size]))
            i += chunk_size - overlap
        return chunks

    tokens = enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunks.append(enc.decode(chunk_tokens))
        start += chunk_size - overlap
    return chunks

# numeric finder
NUMERIC_RE = re.compile(
    r"(\$?\s?[\d]{1,3}(?:[\d,]{0,})(?:\.\d+)?\s?(?:million|billion|MM|B|k|thousand|bn|m)?|\d{1,3}(?:,\d{3})+(?:\.\d+)?)",
    flags=re.IGNORECASE
)

def find_numeric_strings(text):
    matches = NUMERIC_RE.findall(text)
    cleaned = [m.strip() for m in matches if m and m.strip()]
    return cleaned
