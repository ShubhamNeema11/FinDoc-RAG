# Financial RAG System

Hybrid retrieval-augmented generation pipeline for financial document Q&A, built on the [ICAIF-24 Finance RAG Challenge](https://www.kaggle.com/competitions/icaif-24-finance-rag-challenge) datasets.

Dense + BM25 retrieval → RRF fusion → cross-encoder reranking → grounded generation with source citations.

---

## Stack

| | |
|---|---|
| Embeddings | `FinLang/finance-embeddings-investopedia` |
| Vector store | ChromaDB (local, persistent) |
| Sparse retrieval | BM25Okapi |
| Reranker | `BAAI/bge-reranker-v2-m3` |
| LLM | Groq — Llama 3.3-70B (free tier) **or** Ollama (fully local) |
| Evaluation | RAGAS · NDCG@10 |

**Cost: $0**

---

## Datasets

Seven benchmarks from the ICAIF-24 Finance RAG Challenge. Each dataset ships as `corpus.jsonl`, `queries.jsonl`, and `qrels.tsv`.

| Dataset | Docs | Queries | Domain |
|---|---|---|---|
| FinanceBench | 180 | 150 | 10-K annual reports |
| FinQABench | 92 | 100 | 10-K hallucination-aware |
| FinDER | 13,867 | 216 | 10-K domain jargon |
| TATQA | 2,756 | 1,663 | Hybrid table + text |
| FinQA | 2,789 | 1,147 | Multi-step numerical reasoning |
| ConvFinQA | 2,066 | 421 | Multi-turn conversational |
| MultiHiertt | 10,475 | 974 | Multi-hop hierarchical tables |

Download from Kaggle and place under `Dataset/`:

```bash
kaggle competitions download -c icaif-24-finance-rag-challenge
unzip icaif-24-finance-rag-challenge.zip -d Dataset/
```

---

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/shivam1423/Financial-RAG-System
cd Financial-RAG-System
uv sync
```

### Option A — Groq (cloud, free tier)

Create a `.env` file:

```env
GROQ_API_KEY=your_key_here

# Model selection — uncomment one (default: llama-3.3-70b-versatile)
# GROQ_MODEL=llama-3.3-70b-versatile   # best quality — 100K tokens/day
GROQ_MODEL=llama-3.1-8b-instant        # faster — 500K tokens/day, 20K TPM
```

Free key at [console.groq.com](https://console.groq.com) — no credit card required.

### Option B — Ollama (local, no limits)

```bash
brew install ollama        # macOS
ollama serve               # start the server
ollama pull llama3.1:8b    # default model (~5 GB)
# ollama pull mistral:7b   # or any other model
```

No `.env` needed. Override the server URL if not running on localhost:

```env
OLLAMA_BASE_URL=http://localhost:11434
```

---

## Commands

**Ask a question (full RAG — retrieve + generate + cite)**

```bash
# Groq (default)
python main.py --dataset financebench --query "What was Apple's revenue in FY2022?"

# Ollama
python main.py --dataset financebench --query "What was Apple's revenue in FY2022?" --provider ollama

# Specific model
python main.py --dataset financebench --query "..." --provider ollama --model mistral:7b
python main.py --dataset financebench --query "..." --provider groq --model llama-3.1-8b-instant
```

**Retrieval benchmark — single dataset**

```bash
python main.py --dataset financebench
```

**Retrieval benchmark — all 7 datasets**

```bash
python main.py --all
```

**Without MultiQuery expansion (faster, no extra LLM calls)**

```bash
python main.py --dataset financebench --no-multiquery
```

**Force re-embed corpus (ignore ChromaDB cache)**

```bash
python main.py --dataset financebench --rebuild
```

**RAGAS evaluation — faithfulness, answer relevancy, context utilization**

```bash
# Baseline — Groq (no MultiQuery)
python eval.py --dataset financebench --config baseline_k5 --n 20 --no-multiquery

# With MultiQuery expansion
python eval.py --dataset financebench --config multiquery_k5 --n 20

# Ollama — local, no API limits
python eval.py --dataset financebench --config ollama_baseline --n 20 --provider ollama

# Compare configs + save chart
python eval.py --compare --dataset financebench --chart results/ragas_comparison.png

# Historical trend for one config
python eval.py --trend --dataset financebench --config baseline_k5
```

---

## Retrieval Results (NDCG@10)

| Dataset | NDCG@10 |
|---|---|
| FinQABench | 0.9419 |
| FinanceBench | 0.8220 |
| FinQA | 0.7859 |
| TATQA | 0.7835 |
| MultiHiertt | 0.7763 |
| ConvFinQA | 0.7587 |
| FinDER | 0.7180 |

---

## RAGAS Results (FinanceBench)

Reference-free evaluation across configs. Each metric ∈ [0, 1]; higher is better.

| Config | Model | Faithful | Relevancy | Ctx Util |
|---|---|---|---|---|
| baseline (no MultiQuery) | groq/llama-3.3-70b-versatile | 86% | 90% | 72% |
| multiquery_k5 | groq/llama-3.3-70b-versatile | 87% | 92% | 76% |
| baseline (no MultiQuery) | ollama/llama3.1:8b | 76% | 84% | 66% |
| multiquery_k5 | ollama/llama3.1:8b | 79% | 89% | 69% |
*Run `python eval.py --compare --dataset financebench` to populate this table after completing evaluations with n≥20.*

**Metric definitions:**
- **Faithfulness** — fraction of answer claims directly supported by retrieved context (hallucination detector)
- **Answer Relevancy** — cosine similarity of back-generated questions to the original query
- **Context Utilization** — whether the retrieved context was actually used to form the answer

---

## Project Structure

```
Financial-RAG-System/
├── finrag/
│   ├── config.py          # Dataset configs and paths
│   ├── data.py            # JSONL and qrels loaders
│   ├── chunking.py        # Dataset-aware text splitting
│   ├── models.py          # Embedding, reranker, LLM singletons
│   ├── vectorstore.py     # ChromaDB build / load
│   ├── retrieval.py       # Dense + BM25 + RRF + reranking
│   ├── generation.py      # Grounded answer generation with citations
│   ├── evaluation.py      # NDCG@10
│   ├── ragas_eval.py      # RAGAS evaluation harness
│   ├── regression.py      # CSV regression tracker + charts
│   └── compat.py          # ragas import patch
├── main.py                # Retrieval benchmark + interactive RAG CLI
├── eval.py                # RAGAS evaluation CLI
├── Dataset/               # Kaggle competition datasets
└── results/               # Regression CSV + charts
```
