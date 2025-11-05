# Education/Automation Track — Summary

## Context and Goals
- Replace manual search across ~400 Q&A with an intelligent conversational agent.
- Input: Excel (questions, answers, topics, metadata).
- Output: HTML-formatted answers via web/mobile.
- Capabilities: semantic retrieval, context analysis, fallback to forms/email, self-learning from interactions.
- Constraints: integrate with existing Vue.js + MariaDB; emphasize intelligence over UI; reliability and security.

---


## Roadmap 1: More detailed

Architecture Overview

- Data Processing Pipeline:
  - Parse Excel file to extract Q&A pairs, topics, and metadata
  - Create document embeddings (e.g., sentence transformers)
  - Store vectors in a vector database (ChromaDB, Pinecone, or Weaviate)

- Core Components:
  - Vector Search Engine for semantic similarity matching
  - LLM Integration (e.g., OpenAI GPT-4, Anthropic Claude, or Llama-2/3)
  - Context Assembly to retrieve top-k relevant documents
  - Response Generation using retrieved knowledge to produce HTML answers

- Implementation Stack:
  - Backend: FastAPI or Flask with LangChain
  - Vector DB: ChromaDB (local) or Pinecone (cloud)
  - Frontend: Vue.js (integrates with existing infrastructure)
  - Database: MariaDB for conversation logs and analytics

- Self-Learning Mechanism:
  - Log user queries and feedback
  - Continuously update embeddings based on new data/feedback
  - Run A/B tests to improve response quality over time


## Roadmap 1: RAG-based Solution (Retrieval-Augmented Generation)

Architecture highlights
- Ingest Excel, normalize Q&A, create embeddings, store in vector DB.
- Retrieve top-k results, optionally rerank; LLM generates HTML answers grounded in retrieved context.
- Log interactions and feedback; periodically update embeddings.

Implementation stack
- Backend: FastAPI/Flask (+ LangChain or LlamaIndex optional).
- Vector DB: Chroma/FAISS (local) or Pinecone/Weaviate (cloud).
- LLMs: OpenAI/Anthropic or open-source (Llama).
- Frontend: Vue.js; DB: MariaDB for logs/analytics.

Pros
- High answer quality, natural responses.
- Easy updates by upserting new content.
- Scales with growing knowledge base.

Cons
- LLM cost/latency; hallucination risk.
- Vector infra and embedding pipeline add complexity.

---

## Roadmap 1: More detailed

Architecture Overview

- Data Processing Pipeline:
  - Parse Excel file to extract Q&A pairs, topics, and metadata
  - Create document embeddings (e.g., sentence transformers)
  - Store vectors in a vector database (ChromaDB, Pinecone, or Weaviate)

- Core Components:
  - Vector Search Engine for semantic similarity matching
  - LLM Integration (e.g., OpenAI GPT-4, Anthropic Claude, or Llama-2/3)
  - Context Assembly to retrieve top-k relevant documents
  - Response Generation using retrieved knowledge to produce HTML answers

- Implementation Stack:
  - Backend: FastAPI or Flask with LangChain
  - Vector DB: ChromaDB (local) or Pinecone (cloud)
  - Frontend: Vue.js (integrates with existing infrastructure)
  - Database: MariaDB for conversation logs and analytics

- Self-Learning Mechanism:
  - Log user queries and feedback
  - Continuously update embeddings based on new data/feedback
  - Run A/B tests to improve response quality over time
 
  <img width="768" height="821" alt="image" src="https://github.com/user-attachments/assets/1b59dae4-bd06-4e13-b14a-3ef184365630" />

## Roadmap 2: NLP Classification + Similarity

Architecture highlights
- Clean and label Q&A; intent classification + entity extraction.
- TF-IDF/BERT embeddings + cosine similarity to pick best answer.
- Template-based HTML responses; confidence thresholds with fallbacks.

Implementation stack
- Python (scikit-learn, spaCy, Transformers), Elasticsearch (optional).
- Vue.js frontend; MariaDB for storage and logging.

Pros
- Low cost, fast, predictable behavior.
- Simple to integrate and operate.

Cons
- Rigid responses; struggles with novel queries.
- Requires retraining/retuning as content evolves.

---


## Recommendation
- Primary: Roadmap 1 (RAG) for best balance of quality and feasibility.
- Fast hackathon demo: start with Roadmap 2, evolve to RAG.
- Research/long-term innovation: Roadmap 3.

---

## Roadmap 1 Project Structures

Option A — Monorepo Monolith (FastAPI + local vector DB)
- Structure:
    .
    ├─ frontend/                # Vue app (chat UI, feedback)
    ├─ backend/
    │  ├─ app/
    │  │  ├─ main.py            # FastAPI entrypoint
    │  │  ├─ routers/           # chat, ingest, health
    │  │  ├─ rag/               # ingest, embed, store, retrieve, rerank, generate
    │  │  ├─ core/              # settings, deps, logging
    │  │  └─ models/            # pydantic DTOs
    │  └─ tests/
    ├─ data/                    # raw, processed, index
    ├─ infra/                   # docker-compose, Dockerfile
    └─ README.md
- Pros: simple local dev, fast iteration, minimal DevOps.
- Cons: coupled scaling; ingestion vs serving contention; refactor needed to adopt managed vector DB.

Option B — Services + Queue (API + Worker + managed vector DB)
- Structure:
    .
    ├─ services/
    │  ├─ api/                  # chat, feedback, status
    │  ├─ ingest-worker/        # Celery/RQ/Arq tasks
    │  └─ eval-dashboard/       # admin/evaluation UI
    ├─ shared/                  # schemas, prompts, utils
    ├─ deploy/                  # docker, k8s/helm, envs
    └─ infra/                   # Redis/RabbitMQ, Pinecone, observability
- Pros: independent scaling; resilient ingestion; team parallelism.
- Cons: higher ops complexity; coordinated releases; heavier local setup.

Option C — Serverless/Edge API + Static Frontend
- Structure:
    .
    ├─ frontend/                # Vue SPA (CDN)
    ├─ api/                     # serverless functions (query, ingest, feedback)
    ├─ lib/                     # rag core, adapters, schemas
    └─ tests/                   # unit, e2e
- Pros: minimal ops, pay-per-use, global latency, quick to ship.
- Cons: cold starts/time limits; background jobs need schedulers/queues; vendor coupling.

Quick pick
- Small team/demo speed: Option C.
- Stable local dev/offline-friendly: Option A.
- Production scale/high traffic: Option B.
