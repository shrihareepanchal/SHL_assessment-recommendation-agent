# SHL Assessment Recommendation Agent

A stateless, conversational agent that turns a vague hiring need into a
grounded shortlist of SHL Individual Test Solutions - built for the SHL
Labs AI Intern take-home assignment.

```
"We need a solution for senior leadership."
  -> clarifies seniority and use case
  -> recommends OPQ32r + supporting reports
  -> accepts refinement ("also add personality tests")
  -> compares assessments on request
  -> refuses anything outside SHL assessment selection
```

## Contents

- [Architecture](#architecture)
- [Setup](#setup)
- [Running locally](#running-locally)
- [API](#api)
- [Design decisions](#design-decisions)
- [Evaluation methodology](#evaluation-methodology)
- [Testing](#testing)
- [Deployment](#deployment)
- [Limitations & future improvements](#limitations--future-improvements)

## Architecture

```
app/
  api/          FastAPI routes + dependency injection
  agent/        Conversation Analyzer, Intent Detection, Missing-Info Detector,
                Decision Engine, Validator - the explicit control flow
  services/     LLM client adapter + one service per behavior
                (recommend/refine, compare, clarify, refuse)
  retriever/    Embedder (Sentence Transformers), VectorStore (ChromaDB),
                KeywordIndex (BM25), HybridRetriever (fuses both)
  ranking/      Metadata-fit scoring + weighted rank fusion
  prompts/      One LangChain PromptTemplate per behavior
  catalog/      Loads the processed SHL catalog into typed Assessment objects
  models/       Pydantic v2 models (API schema + internal catalog schema)
  config/       Environment-driven settings (pydantic-settings)
  evaluation/   Recall@K, trace parser, replay harness, behavior probes
  utils/        Structured logging

data/           catalog_raw.json, catalog_processed.json, chroma_db/,
                sample_conversations/ 
scripts/        build_catalog.py, ingest_catalog.py, run_eval.py
tests/          pytest suite (health, schema, catalog, retrieval, decision
                engine, intent detection, prompts, conversation flows)
```

### Request flow

Every `POST /chat` call runs one linear, explicit pipeline
(`app/agent/orchestrator.py`) - no agent framework, no autonomous loop:

```
Conversation Analyzer -> Intent Detection -> Missing Information Detector
    -> Decision Engine -> {Retriever -> Recommendation Generator} | Compare | Clarify | Refuse
    -> Validator -> JSON Formatter
```

1. **Conversation Analyzer** reconstructs conversation state from the full
   stateless message history (the API stores nothing between calls).
2. **Intent Detection** classifies the latest turn as refuse / compare /
   refine / recommend-or-clarify using explicit regex-based rules (fast,
   deterministic, auditable - not an LLM call).
3. **Missing Information Detector** extracts a structured `ConstraintProfile`
   (job level, test type, language, duration, adaptive/remote) via an LLM
   extraction prompt, with a safe empty-profile fallback if parsing fails.
4. **Decision Engine** (`app/agent/decision_engine.py`) is pure, dependency-light
   Python that picks exactly one action and computes `end_of_conversation`,
   respecting the 8-turn cap.
5. **Hybrid Retriever** combines ChromaDB semantic search, BM25 keyword
   search, and metadata-fit scoring via weighted rank fusion.
6. **Recommendation Generator** (an LLM call, grounded in the retrieved
   catalog snippets only) writes the natural-language reply.
7. **Validator** cross-checks every recommended URL against the catalog
   repository and drops anything unverifiable - hallucination prevention
   enforced in code, not just in the prompt.
8. **JSON Formatter** returns the exact `{reply, recommendations,
   end_of_conversation}` schema.

## Setup

Requires Python 3.11+.

```bash
git clone <this-repo>
cd shl-assessment-agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then set GOOGLE_API_KEY
```

Build the catalog and vector index once:

```bash
python scripts/build_catalog.py     # data/catalog_raw.json -> catalog_processed.json
python scripts/ingest_catalog.py    # builds the persisted Chroma index
```

(`app/main.py` also builds the index lazily on startup if it's empty, so
this step is an optimization, not a hard requirement.)

## Running locally

```bash
uvicorn app.main:app --reload
```

```bash
curl -s http://localhost:8000/health

curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hiring a Java developer who works with stakeholders"},
      {"role": "assistant", "content": "Sure. What is the seniority level?"},
      {"role": "user", "content": "Mid-level, around 4 years"}
    ]
  }'
```

```json
{
  "reply": "Got it - here are assessments that fit a mid-level Java developer who collaborates with stakeholders.",
  "recommendations": [
    {"name": "Core Java (Entry Level) (New)", "url": "https://www.shl.com/products/product-catalog/view/core-java-entry-level-new/", "test_type": "K"},
    {"name": "Occupational Personality Questionnaire OPQ32r", "url": "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/", "test_type": "P"}
  ],
  "end_of_conversation": false
}
```

## API

| Endpoint      | Method | Notes                                                   |
|---------------|--------|----------------------------------------------------------|
| `/health`     | GET    | Returns `{"status": "ok"}`. Cold starts may take ~2 min. |
| `/chat`       | POST   | Stateless; send the full `messages` history every call.  |

`recommendations` is `[]` while clarifying/refusing/comparing, or an array
of 1-10 `{name, url, test_type}` objects once a shortlist is committed.
`end_of_conversation` is `true` only once the agent considers the task
complete (the user confirmed the shortlist, or the 8-turn cap is reached).

Full request/response schema: `app/models/api_models.py`.

## Design decisions

- **Hybrid retrieval, weighted-sum fusion.** Dense embeddings alone
  under-perform on exact product codes ("OPQ32r", ".NET MVC") and
  acronyms; BM25 alone misses paraphrased/vague queries ("someone who can
  lead a team under pressure" -> personality assessments). Both signals are
  normalized to `[0, 1]` and combined with a metadata-fit term so the
  combination is interpretable and tunable against the labeled traces,
  rather than opaque RRF re-ranking. See `app/ranking/rank_fusion.py`.
- **Explicit decision logic, not an LLM router.** The assignment's own
  failure-mode list calls out "vibe-coding without understanding" and
  "conversational incoherence" - a non-deterministic conversation is
  exactly where implicit LLM-decided control flow breaks down under a hard
  8-turn cap and 30s timeout. `app/agent/decision_engine.py` is pure
  Python, unit-tested without mocking the LLM at all.
- **Validator as the real hallucination guard.** Prompts instruct the LLM
  not to invent names/URLs, but the actual guarantee is code: every
  recommended URL is checked against `CatalogRepository.is_valid_url`
  before it can reach the client. This is the layer the hard-eval scoring
  ("items from catalog only") is graded against, so it cannot be
  advisory-only.
- **One prompt per behavior.** Clarify / Recommend / Refine / Compare /
  Refuse / Extract are separate `PromptTemplate`s (`app/prompts/templates.py`)
  sharing one grounding-rules block, not one mega-prompt. Each is
  independently testable (`tests/test_prompts.py`) and can be iterated on
  without risking regressions in unrelated behaviors.
- **Stateless by construction.** No per-conversation state is stored
  anywhere (no session cache, no DB row). `ConversationState` is rebuilt
  from `messages` on every call, matching the assignment's explicit
  requirement and making horizontal scaling trivial.
- **Swappable LLM provider.** All call sites depend on the `LLMClient`
  protocol (`app/services/llm_service.py`), never on
  `langchain_google_genai` directly. A second provider is a new adapter
  class + a settings flag, not a rewrite. A dependency-free
  `EchoFallbackLLMClient` keeps the retrieval/decision/API layers fully
  testable without a live LLM call or API key, and keeps the service
  *degraded, not down* if the LLM provider has an outage.
- **LangChain used narrowly.** `PromptTemplate`, the chat model wrapper,
  and (implicitly) the Document/Retriever shape are used because they
  genuinely help; LangChain Agents and LangGraph are deliberately not
  used, per the assignment - the conversational control flow is explicit
  Python.

## Evaluation methodology

`scripts/run_eval.py` replays the 10 supplied labeled traces
(`data/sample_conversations/`) against a running instance and reports:

- **Recall@10** per trace and the mean across traces, computed exactly per
  the assignment's appendix definition (`app/evaluation/recall.py`). The
  "expected shortlist" for each trace is parsed from the final
  `end_of_conversation: true` turn's markdown table
  (`app/evaluation/trace_parser.py`).
- **Schema compliance** on every response in every replayed conversation.
- **Turn-cap adherence** (conversations must resolve within 8 turns).
- **Behavior probes** (`app/evaluation/probes.py`): no recommendation on a
  vague turn 1, off-topic refusal, prompt-injection resistance, and
  refinement honored without restarting the shortlist.

```bash
uvicorn app.main:app &
python scripts/run_eval.py --base-url http://localhost:8000
```

## Testing

```bash
pytest
```

Covers: health endpoint, `/chat` schema compliance (including the 1-10
bound, `extra=forbid` on `Recommendation`, and empty-message/non-user-last
validation), catalog loading and URL validation, hybrid retrieval and
metadata scoring, decision engine branch coverage, intent detection regexes
against real trace-style utterances, prompt template rendering, and
multi-turn conversation flows (turn-cap adherence, refinement continuity).

## Deployment

**Docker:**

```bash
docker build -t shl-agent .
docker run -p 8000:8000 --env-file .env shl-agent
```

**Render:** connect the repo, choose "Docker" as the environment, set the
environment variables from `.env.example` (at minimum `GOOGLE_API_KEY`),
and deploy. `GET /health` is used as the readiness check; the assignment
allows up to 2 minutes for a cold start to respond.

