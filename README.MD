# FinDoc-Intelligence

A **financial document intelligence system** that enables safe, evidence-based question answering over long financial reports (e.g., SEC 10-K filings).

Unlike generic RAG chatbots, this project is designed to **minimize hallucinations** by separating:
- qualitative narrative analysis
- deterministic numeric extraction from tables

----

## 🔍 What Problem Does This Solve?

Financial reports are:
- long
- table-heavy
- compliance-driven
- sensitive to numeric hallucinations

Naive RAG systems fail by:
- breaking tables into text
- guessing numbers
- mixing audit boilerplate with business insights

**FinDoc-Intelligence** addresses this by using a **hybrid pipeline**:
- vector-based retrieval for narrative questions
- table-aware extraction for numeric questions

---

## 🧠 Key Features

- 📄 Upload financial PDFs (10-Ks, annual reports)
- 🧩 Token-aware chunking with section metadata
- 🔍 Evidence-driven retrieval (business, MD&A, risk)
- 📊 Table-first numeric extraction (no LLM math)
- 🚫 Hallucination-safe fallbacks
- 🔎 Transparent source snippets
- 🧪 Designed for analyst-style questions

---

## 🏗️ Architecture Overview

**Two-Lane Design**

1. **Narrative Lane**
   - Business overview
   - Risks
   - Strategy
   - Governance
   - Uses vector search + conservative LLM synthesis

2. **Numeric Lane**
   - Revenue
   - Cash flow
   - Financial metrics
   - Uses table parsing + deterministic lookup (no guessing)

---


## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
