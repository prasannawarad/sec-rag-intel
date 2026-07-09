# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

SEC Filing Intelligence Pipeline: a RAG system that lets analysts query 10-K/10-Q filings
from SEC EDGAR in natural language and receive grounded, cited answers. Portfolio project
for data/ML engineering job applications — every architectural decision should be
explainable in an interview (see Design Decisions below). Deploys to HuggingFace Spaces.

## Commands

All workflows go through the Makefile (`make help` lists everything):

```bash
make install          # pip install -r requirements.txt
make dev-install      # install + pre-commit hooks
make lint             # ruff check + mypy (src only for mypy)
make format           # ruff format + ruff check --fix
make test             # pytest -m "not integration" (fast, no network/model load)
make test-all         # pytest including integration tests (hits SEC EDGAR, Groq, loads embedding model)
make ingest           # download SEC filings
make index            # full pipeline: download → parse → chunk → embed (Chroma)
make index-pinecone   # embed all chunks into Pinecone (needs PINECONE_API_KEY)
make eval             # RAGAS evaluation over all 20 golden questions
make eval-subset N=5  # RAGAS on first N questions (saves Groq tokens)
make api              # FastAPI dev server
make ui               # Streamlit app
make clean            # ⚠ also deletes data/processed/ and chroma_db/ — re-run make index after
```

Single test: `pytest tests/test_chunker.py -k <name>` (unit tests need no network or keys).
Pytest markers (`--strict-markers` is on): `integration` (external services / embedding
model load), `slow`. Ruff: line length 100, target py311, rules E/F/I/B/UP/SIM/C4.

## Architecture

```
src/
├── config.py              ← pydantic-settings Settings; ALL tunables live here (env-overridable)
├── logging_setup.py
├── ingest/                ← downloader (sec-edgar-downloader), sgml + parser (HTML→text),
│                            chunker, manifest (tracks what's ingested), quality checks
├── embeddings/vectorstore.py  ← Chroma (local) / Pinecone (prod) behind VECTOR_STORE_MODE
├── retrieval/retriever.py     ← MMR retriever + metadata filters (ticker, year, filing_type)
├── chain/                 ← prompts.py, rag_chain.py (LangChain LCEL), types.py,
│                            quota.py (free-tier budget guard), cache.py (disk answer cache)
├── evaluation/            ← golden_dataset.json ({version, description, questions[20]}),
│                            evaluate.py (RAGAS runner + pre-flight budget check)
└── api/main.py            ← FastAPI: POST /query, DELETE /session/{id}, GET /companies,
                             GET /metrics, GET /quota, GET /health
app/streamlit_app.py       ← Streamlit UI (HF Spaces entry point); theme in .streamlit/config.toml
scripts/                   ← build_index.py, index_pinecone.py, gen_ground_truth.py,
                             smoke_test_retrieval.py
```

### Key facts (verified against code — keep in sync with `src/config.py`)

- **Embeddings are LOCAL** — sentence-transformers `BAAI/bge-small-en-v1.5`, no API key.
  Loaded as a singleton (~130 MB, once per process). `EMBEDDING_DEVICE=mps` on Apple
  Silicon gives ~2–3× speedup. The same model runs in dev and on HF Spaces.
  (An earlier design used OpenAI `text-embedding-3-small` — that is gone.)
- **LLM:** Groq `llama-3.3-70b-versatile` (`GROQ_API_KEY` required).
- **Vector store:** ChromaDB persisted to `chroma_db/` locally; Pinecone in prod.
  Switch via `VECTOR_STORE_MODE=local|pinecone` — no code changes.
- **Chunking:** 512 tokens, 50 overlap. **Retrieval:** MMR, k=5, fetch_k=20.
- **SEC EDGAR requires a User-Agent** — set `SEC_USER_AGENT` to a real name/email.
- **Free-tier guardrails wrap every Groq call** (`src/chain/quota.py` + `cache.py`):
  persisted daily budget (90k tokens / 900 req, state in `data/quota_state.json`,
  resets midnight UTC), 25 RPM throttle, `max_tokens=1024` cap, disk answer cache
  (`data/cache/`, first-turn questions only), and a retrieval-only degraded answer
  when the budget is spent — the chain returns `{answer, sources, cached, degraded}`.
  `make eval` pre-checks the budget and refuses runs it can't afford (`--force` overrides);
  eval passes `use_cache=False` and a per-question session id.

## Environment variables

Loaded from `.env` via `src/config.py` (pydantic-settings; every Settings field is
overridable by env var). Never commit `.env`.

```
GROQ_API_KEY=            # required — LLM inference
VECTOR_STORE_MODE=local  # local | pinecone
PINECONE_API_KEY=        # only for pinecone mode
SEC_USER_AGENT=          # "name email" — SEC EDGAR blocks anonymous clients
EMBEDDING_DEVICE=cpu     # "mps" on Apple Silicon
```

`OPENAI_API_KEY` is **no longer used** — embeddings are local.

## Evaluation (the differentiator)

- `src/evaluation/golden_dataset.json` — 20 Q&A pairs with ground truth (factual,
  comparative, multi-hop).
- RAGAS metrics: faithfulness, answer_relevancy, context_recall.
- Results go to `eval_results/` (currently empty — regenerate with `make eval` before
  citing scores in the README).
- Groq free-tier tokens are the constraint — use `make eval-subset N=5` while iterating.

## CI / deployment

- `.github/workflows/ci.yml` — every push/PR to main runs `ruff check`,
  `ruff format --check` (unformatted code **fails CI** — run `make format` first),
  and unit tests with coverage.
- `.github/workflows/hf_sync.yml` — every push to main **force-pushes the repo to the
  HuggingFace Space** (`prasannawarad/sec-rag-intel`) if the `HF_TOKEN` repo secret is
  set; skipped silently when it isn't. Pushing to main is deploying.
- `README.md` starts with **HF Spaces YAML frontmatter** (`sdk: streamlit`,
  `app_file: app/streamlit_app.py`) — do not remove or reorder it; Spaces parses it.

## Coding conventions

- Python 3.11+, type hints on all function signatures
- Pydantic models for all FastAPI request/response schemas
- No hardcoded credentials — everything through `Settings` in `src/config.py`
  (not raw `os.getenv()` scattered around)
- Logging via `logging` (configured in `src/logging_setup.py`), never `print()`
- `src/` is pure logic, `app/` is pure UI — no business logic in the Streamlit file
- Run `make lint` before committing; pre-commit hooks enforce ruff

## Key design decisions (be ready to explain these in interviews)

1. **Why MMR over plain similarity search?** Reduces redundancy — without it the top-5
   chunks often come from the same paragraph repeated across filing sections.
2. **Why local BGE embeddings?** Zero API cost, no key management, and the identical
   model runs in dev and on HF Spaces — no dev/prod embedding drift. Query and chunk
   vectors must always come from the same model; re-index if the model changes.
3. **Why ChromaDB locally, Pinecone in prod?** Chroma needs zero infra for development;
   Pinecone demonstrates cloud vector DBs. `VECTOR_STORE_MODE` switches without code changes.
4. **Why RAGAS instead of manual eval?** Reproducible, quantitative scores. Faithfulness
   catches hallucinations — critical for financial data.
5. **Why source attribution?** SEC filings are legal documents; analysts must verify the
   source. Compliance argument, not just UX.
6. **Why Groq?** Cost + latency at competitive quality; consistent with resume.
7. **Why a quota guard + answer cache?** Free-tier quotas are the project's real
   production constraint. Budget/throttle/cache/degrade are the same patterns used
   against a paid API bill — and the app stays up (retrieval-only) when the LLM
   budget is spent. Caching is safe because the corpus is static and temperature=0.
