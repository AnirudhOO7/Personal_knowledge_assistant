# Personal Knowledge Assistant

> A self-correcting RAG service over your own documents it retrieves context, reflects on whether that context is actually enough to answer, re-retrieves what's missing, and only then generates a cited answer.

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-RAG-1C3C3C)
![Qdrant](https://img.shields.io/badge/Qdrant-vector_store-DC244C)
![Anthropic](https://img.shields.io/badge/Anthropic-Claude-D4A27F)

Most RAG pipelines retrieve once and hope the context is good enough. This one adds a **reflection loop**: after retrieving, the model judges whether the retrieved chunks actually answer the question, and if not, it identifies what's missing, re-retrieves with that gap appended to the query, and tries again before generating. The result is answers that are better grounded and that come with their sources.

---

## Why this project

- **Self-correcting retrieval** - a `retrieve → reflect → re-retrieve → generate` loop (bounded at 2 attempts) that detects insufficient context instead of confidently answering from it.
- **Cited answers** - every response includes deduplicated source citations (file + page), so answers are traceable to the documents they came from.
- **Pluggable vector store** - retrieval sits behind an abstract `BaseVectorStore` interface (`add_documents` / `similarity_search`), with a Qdrant implementation; swapping backends doesn't touch the pipeline.
- **Clean ingestion pipeline** - separate load → parse → chunk → embed → store stages, each independently testable.

---

## How the reflection loop works

```text
question
   │
   ▼
retrieve (top-k from Qdrant)
   │
   ▼
reflect ──► sufficient? ──yes──► generate ──► cited answer
   │              
   no (identify what's missing)
   │
   ▼
re-retrieve with "question + missing info"  ──► (loop, max 2 attempts)
```

The reflect step asks the LLM to return a structured judgment - `is_sufficient`, a `reason`, and the `missing` information which drives whether the loop re-retrieves or generates. The response reports how many reflection attempts it took, so the loop's behavior is observable.

---

## Architecture

```text
app/
  api/            FastAPI routes: /query, /ingest, /health
  ingestion/      loader · parser · chunker  (document -> chunks)
  embeddings/     sentence-transformers embedder (all-MiniLM-L6-v2)
  vectorstore/    BaseVectorStore (abstract) + Qdrant implementation
  retrieval/      top-k retrieval
  llm/            generator + reflect (self-correction) + prompts
  services/       ingest_service · query_service (the reflection loop)
  schemas/        Pydantic request/response + document models
  core/           config + logging
scripts/          ingest.py · rebuild_index.py
tests/            chunking · generator · retrieval
```

**Stack:** Python · FastAPI · LangChain · LangGraph · Qdrant · HuggingFace `sentence-transformers` (all-MiniLM-L6-v2) · Anthropic Claude (Haiku) · Pydantic Settings · PyMuPDF.

**Defaults** (`app/core/config.py`): chunk size 800 / overlap 200, top-k 5, embedding model `all-MiniLM-L6-v2`, LLM `claude-haiku-4-5` at temperature 0.

---

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/query` | Ask a question, runs the full retrieve/reflect/generate loop, returns answer + sources + attempt count |
| `POST` | `/ingest` | Upload a document (PDF) to ingest into the vector store |
| `GET`  | `/health` | Health check |

---

## Setup

Requires Python 3.12+ and a running Qdrant instance.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# start Qdrant (Docker)
docker run -p 6333:6333 qdrant/qdrant
```

Create a `.env`:

```bash
ANTHROPIC_API_KEY=your_anthropic_key
# optional overrides: QDRANT_URL, COLLECTION_NAME, LLM_MODEL, TOP_K, CHUNK_SIZE ...
```

Ingest documents and run:

```bash
python scripts/ingest.py            # ingest files from data/raw
uvicorn app.main:app --reload       # serve the API
```

Then query:

```bash
curl -X POST localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What does the document say about X?"}'
```

---

## Testing

```bash
python -m pytest
```

Unit tests cover chunking, generation, and retrieval.

---

## Roadmap

- Evaluation harness: a labeled Q→answer dataset with precision@k / recall@k and an A/B of the reflection loop (1 vs 2 attempts) to quantify its lift.
- `/health` that pings Qdrant; request logging with latency.
- Containerization (Dockerfile + compose for app + Qdrant).
