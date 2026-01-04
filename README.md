# MedMem0 - Longitudinal Patient Memory with Mem0

> Memory-augmented retrieval for patient health records using [Mem0](https://mem0.ai)

🔗 **Live Demo:** [medical-mem0.vercel.app](https://medical-mem0.vercel.app)

## What is this?

A full-stack POC demonstrating how Mem0 can power longitudinal patient memory in healthcare applications. Clinicians can query patient history in natural language, with relevant memories retrieved and surfaced automatically.

**Example queries:**
- "What medications is this patient on?"
- "Given their current medications, are there any potential drug interactions?"
- "Any mental health concerns?"

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (Next.js)                       │
│  - Patient selector    - Strategy toggle    - Chat interface    │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Backend (FastAPI)                          │
│  /chat  /patients  /search                                      │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
        ┌───────────────────┐       ┌───────────────────┐
        │   Memory Service  │       │   LLM Orchestrator│
        │   (Mem0 + Pinecone)│       │   (GPT-4o-mini)   │
        └───────────────────┘       └───────────────────┘
```

## Key Features

| Feature | Description |
|---------|-------------|
| **Mem0 Integration** | Patient memories stored and retrieved via Mem0's memory layer |
| **Multiple Retrieval Strategies** | Vanilla dense vs Enhanced (LLM query expansion) |
| **Eval Harness** | 50 gold cases with Recall@K, MRR, latency metrics |
| **Clinical Reasoning** | Drug interaction checks, monitoring recommendations |
| **Graceful Handling** | Missing data, edge cases, malformed queries |

## Retrieval Strategies

### Vanilla Dense
Basic semantic search using Mem0's default vector retrieval.

### Enhanced (Query Expansion)
Medical queries often suffer from a **semantic gap** - clinicians ask questions using different terminology than what's stored in patient records. For example:
- Query: *"Does the patient have high blood pressure?"*
- Records contain: *"Systolic: 145 mmHg, Diastolic: 92 mmHg, hypertension diagnosed"*

Enhanced strategy uses an LLM to expand queries with clinical synonyms, abbreviations, and related terms before retrieval:

```
Original: "blood pressure concerns"
Expanded: "blood pressure hypertension systolic diastolic mmHg BP vital signs cardiovascular"
```

This bridges the vocabulary mismatch between natural language questions and structured medical data, improving recall by ~14%.

## Project Structure

```
medical_mem0/
├── backend/
│   ├── core/
│   │   ├── memory_service.py    # Mem0 integration
│   │   └── llm_orchestrator.py  # Response generation
│   ├── api/routes/              # FastAPI endpoints
│   └── models/                  # Pydantic schemas
├── frontend/                    # Next.js app
├── eval/
│   ├── harness.py              # Evaluation runner
│   ├── metrics.py              # Recall@K, MRR, latency
│   └── gold_dataset/           # 50 test cases
├── experiments/
│   ├── vanilla_dense.py
│   ├── hybrid_bm25.py          # BM25 + dense fusion
│   ├── temporal_decay.py       # Recency weighting
│   └── entity_filtered.py      # Medical NER filtering
├── research/
│   ├── rag_fusion.py           # Multi-query fusion
│   ├── colbert_retrieval.py    # Late interaction
│   └── advanced_retrieval.py   # Experimental methods
├── scripts/
│   ├── process_synthea.py      # Data pipeline
│   └── seed_patients.py        # Memory seeding
├── Dockerfile
└── requirements.txt
```

## Evaluation

Eval harness with 50 gold cases measuring:
- **Recall@K** (K=3, 5)
- **MRR** (Mean Reciprocal Rank)
- **Keyword Precision**
- **Latency** (mean, p50, p95, p99)

Run benchmarks:
```bash
python -m eval.harness
```

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- Mem0 API key
- OpenAI API key
- Pinecone API key

### Backend
```bash
cd backend
pip install -r requirements.txt

export OPENAI_API_KEY=your_key
export MEM0_API_KEY=your_key
export PINECONE_API_KEY=your_key

uvicorn main:app --reload
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

### Run Evaluation
```bash
python -m eval.harness --strategy vanilla
python -m eval.harness --strategy enhanced
python -m eval.harness  # all strategies
```

## Deployment

- **Frontend:** [Vercel](https://medical-mem0.vercel.app)
- **Backend:** Render

## Experiments

The `experiments/` folder contains retrieval strategies explored during development:
- `vanilla_dense.py` - Basic semantic search
- `hybrid_bm25.py` - BM25 + dense fusion
- `temporal_decay.py` - Recency weighting
- `entity_filtered.py` - Medical entity filtering
- `with_reranker.py` - Cross-encoder reranking

Currently, the production app uses `vanilla` and `enhanced` strategies via the memory service.

## Tech Stack

- **Frontend:** Next.js 14, TypeScript, Tailwind CSS
- **Backend:** FastAPI, Python 3.11
- **Memory:** Mem0, Pinecone (vector store)
- **LLM:** OpenAI GPT-4o-mini
- **Embeddings:** text-embedding-3-small


