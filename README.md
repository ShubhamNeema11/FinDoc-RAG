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
| LLM | Groq — Llama 3.3-70B (free tier) |
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
git clone https://github.com/your-username/Financial-RAG-System
cd Financial-RAG-System
uv sync
```

Create a `.env` file:

```env
GROQ_API_KEY=your_key_here
```

Free key at [console.groq.com](https://console.groq.com) — no credit card required.

---

## Commands

**Ask a question (full RAG — retrieve + generate + cite)**

```bash
python main.py --dataset financebench --query "What was Apple's revenue in FY2022?"
```

**Retrieval benchmark — single dataset**

```bash
python main.py --dataset financebench
```

**Retrieval benchmark — all 7 datasets**

```bash
python main.py --all
```

**Without MultiQuery expansion (faster, no extra Groq calls)**

```bash
python main.py --dataset financebench --no-multiquery
```

**Force re-embed corpus (ignore ChromaDB cache)**

```bash
python main.py --dataset financebench --rebuild
```

**RAGAS evaluation — faithfulness, answer relevancy, context utilization**

```bash
# Baseline (no MultiQuery)
python eval.py --dataset financebench --config baseline_k5 --n 20 --no-multiquery

# With MultiQuery expansion
python eval.py --dataset financebench --config multiquery_k5 --n 20

# Compare configs + save chart
python eval.py --compare --dataset financebench --chart results/ragas_comparison.png
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
