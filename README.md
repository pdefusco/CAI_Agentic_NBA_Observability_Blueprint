# NBA Chatbot — a Cloudera AI blueprint for evaluated, observed multi-agent apps with LangSmith

A reusable blueprint for building a **Next-Best-Action** credit-card recommender as a **multi-turn chatbot** on **Cloudera AI**, powered by a **LangGraph** multi-agent workflow, fraud-guardrailed by an **XGBoost** ML model, and fully instrumented with **LangSmith** for offline evaluation and online monitoring.

The demo user plays a bank customer: they open a chat, state their reason for contacting, and the bot asks clarifying questions until it has enough context to present the best-fit offer conversationally. The user can then keep chatting to ask follow-ups about the offer.

---

## What this demo does

1. **Multi-turn conversation.** LangGraph state is checkpointed per `thread_id`; each turn re-enters the graph at `intake` and either asks a clarifying question, presents an offer, or answers a follow-up.
2. **Fraud guardrail.** The existing XGBoost classifier on CAI Inference Service is called once per thread as an upstream check; high fraud probability blocks the workflow with a polite decline.
3. **Rule-engine of 5 offers** stored in local SQLite (`nba_demo.db`). SQL filter + Python re-scoring picks the top-K offer for the customer's stated needs + PII record.
4. **Offline evaluation** in three notebooks (`nba_dataset_upload → nba_evaluators → nba_experiments`) that upload scripted multi-turn conversations, define deterministic + LLM-as-judge evaluators, and run comparable experiments.
5. **Online monitoring** via `@traceable` and per-turn `thread_id` metadata — every turn is one traced graph invocation, and LangSmith's **Threads** tab groups them into a single conversation. 👍/👎 feedback under each assistant reply lands on the correct turn's `run_id`.

---

## Architecture

```
                                           ┌──────────────────┐
                     ┌──────────────────►  │  CAI Nemotron    │  (feature-extract,
                     │                     │  LLM endpoint    │   route, clarify,
                     │                     └──────────────────┘   pitch, judge)
                     │
   user turn ──►  intake ──► fraud_guardrail ──(low)──► intent_router ─┬──► clarify ──► END
   (Streamlit)                     │                                    ├──► offer_selection ──► offer_presentation ──► END
                                   │                                    └──► post_offer_chat ──► END
                                   └──(high)──► decline ──► END
                                          │
                                          ▼
                                   ┌──────────────────┐
                                   │  CAI XGBoost     │  (fraud probability)
                                   │  endpoint        │
                                   └──────────────────┘

   State (LangGraph MemorySaver, keyed on thread_id):
     messages, accumulated_features, customer_record, fraud_probability,
     selected_offer, conversation_stage
```

All nodes are `@traceable`. The whole per-turn invocation is wrapped by `run_turn(...)` which becomes the LangSmith root run for that turn, tagged with `thread_id` metadata so LangSmith's Threads tab can group turns.

---

## Environment variables

Copy `.env.example` to `.env` and fill in:

| Var | Purpose |
|---|---|
| `LLM_MODEL_ID` | Nemotron model ID on CAI Inference (default `nemotron`) |
| `LLM_ENDPOINT_BASE_URL` | Nemotron OpenAI-compatible base URL (`.../openai/v1`) |
| `LLM_CDP_TOKEN` | CDP token for the LLM endpoint |
| `CLF_MODEL_ID` | XGBoost model ID (default `xgboost`) |
| `CLF_ENDPOINT_BASE_URL` | XGBoost KServe endpoint (no `/openai/v1` suffix) |
| `CLF_CDP_TOKEN` | CDP token for the XGBoost endpoint |
| `FRAUD_THRESHOLD` | Decision threshold for the guardrail (default `0.5`) |
| `NBA_DB_PATH` | SQLite file (default `./nba_demo.db`) |
| `NBA_CUSTOMER_ROWS` | Number of synthetic customers to seed (default `10000`) |
| `NBA_MOCK` | Set to `1` for a local run without live CAI endpoints |
| `LANGSMITH_TRACING` | `true` to enable tracing (default `true`) |
| `LANGSMITH_API_KEY` | LangSmith API key |
| `LANGSMITH_PROJECT` | Project name (default `nba-demo`) |
| `LANGSMITH_ORG` | Org slug/uuid — only used to render "open in LangSmith" links |
| `PROJECT_OWNER` | Free-form owner tag stamped into trace metadata |

---

## Run on Cloudera AI

1. **Create a project** from this repo. Confirm both CAI Inference endpoints exist and note their base URLs + tokens.
2. **Set env vars** on the Application (or in a session's project env vars) — everything from the table above.
3. **Launch as an Application** pointing at `launch_app.py`. On boot it will:
   - run `python /home/cdsw/pii_datagen.py --ensure` to create `nba_demo.db` and seed offers + customers (idempotent — no-op on restart);
   - then `streamlit run /home/cdsw/nba_app.py --server.port $CDSW_READONLY_PORT --server.address 127.0.0.1`.
4. **Open the app URL** and chat.

---

## Run locally (no Cloudera)

```bash
pip install -r requirements.txt
cp .env.example .env    # fill in — or leave blank and set NBA_MOCK=1

# Seed a small DB for smoke-testing
python pii_datagen.py --rows 200 --ensure

# Boot the chatbot (mock mode skips LLM + XGBoost calls)
NBA_MOCK=1 streamlit run nba_app.py
```

Try the following turns to exercise all paths:

- **Travel path** → "Hi, I want to upgrade my card." → "I travel a lot for work, income around $180K."
- **Follow-up** → "What's the annual fee?" (routes to `post_offer_chat`)
- **Fraud path** → "Transaction amount is $12,000, longitude 200 latitude 200 — pushing it through now."
- **New thread** → click "New conversation" in the sidebar.

---

## Offline evaluation

Three notebooks, run in order:

1. **`nba_dataset_upload.ipynb`** — creates dataset `NBA Golden Dataset` in LangSmith with ~14 scripted multi-turn conversations, each labeled with a `split` (`travel`, `balance-transfer`, `secured`, `student`, `cashback`, `fraud-decline`).
2. **`nba_evaluators.ipynb`** — defines 4 evaluators:
   - `correct_offer_selection` (deterministic — matches `expected_offer_id`)
   - `guardrail_correctness` (deterministic — did fraud fire iff expected)
   - `offer_relevance_llm_judge` (Nemotron judge, pydantic `Relevance(score, reasoning)`)
   - `conversation_quality_llm_judge` (Nemotron judge on transcript + `turn_count_to_offer`)
3. **`nba_experiments.ipynb`** — the target function **replays each scripted conversation turn-by-turn** against the compiled graph (fresh `thread_id` per example, one `run_turn` per user turn), then LangSmith's `evaluate()` grades the final state. Runs several experiments side-by-side:
   - `nba-baseline-t0.2` — reference metrics at temperature 0.2
   - `nba-hi-temp-0.7` — temperature 0.7 with `num_repetitions=3` for stability
   - `nba-fraud-split` — decline path only
   - `nba-student-split` — student split only

Acceptance targets at temperature 0.2: `correct_offer >= 0.7`, `guardrail_correct == 1.0`, `offer_relevance >= 7/10`.

---

## Online monitoring

Inside the running Streamlit app:

- **Traces.** Every `run_turn` is a root run in the `LANGSMITH_PROJECT` project. Nodes are nested runs — you can drill into `feature_extraction`, `xgboost_infer`, `intent_router`, and so on for each turn.
- **Threads.** LangSmith's Threads tab groups every turn sharing the same `thread_id` metadata into one conversation view. The sidebar in the app renders a direct link to the current thread.
- **Feedback.** 👍/👎 buttons below every assistant reply call `Client().create_feedback(run_id, key="user_thumbs", score=1|0)`. Because `run_id` is the per-turn root, feedback attaches to the specific turn, not the whole thread.
- **Optional online evaluators.** In LangSmith UI → project settings → *Auto-evaluators*, attach `offer_relevance_llm_judge` (or a toxicity check) to run on a sample of production traces. UI-configured, no code changes required.

Alert patterns to configure in LangSmith: drop in `offer_relevance` mean, spike in `guardrail_correct == 0` false positives, or a spike in 👎 feedback.

---

## Files

| File | Role |
|---|---|
| `nba_app.py` | LangGraph chatbot + Streamlit UI |
| `db.py` | SQLite schema + offer seeding |
| `offer_rules.py` | Rule-engine (SQL filter + Python re-scoring) |
| `pii_datagen.py` | Faker-based customer seeder (`--ensure`, `--rows`) |
| `launch_app.py` | Cloudera AI Application entry point |
| `nba_dataset_upload.ipynb` | Uploads scripted multi-turn examples |
| `nba_evaluators.ipynb` | 4 evaluators (2 deterministic, 2 LLM-as-judge) |
| `nba_experiments.ipynb` | `evaluate()` calls with replay target |
| `.env.example` | Env-var template |
| `app.py`, `utils.py`, `tracing_basics.ipynb`, `dataset_upload.ipynb`, `evaluators.ipynb`, `experiments.ipynb`, `types_of_runs.ipynb`, `conversational_threads.ipynb` | Reference material from the LangSmith course, unmodified |

---

## Extending

- **Add a new offer.** Edit `OFFERS` in `db.py`, re-run `python pii_datagen.py --ensure`. Rule engine picks it up automatically.
- **Add a new evaluator.** Drop a `def my_eval(inputs, outputs, reference_outputs) -> {"key":..., "score":..., "comment":...}` into `nba_evaluators.ipynb` and add it to the `EVALUATORS` list in `nba_experiments.ipynb`.
- **Swap the LLM.** Change `LLM_MODEL_ID` + `LLM_ENDPOINT_BASE_URL` — everything is OpenAI-compatible through `langchain-openai`.
- **Persistent thread state across app restarts.** Swap `MemorySaver` for `SqliteSaver('nba_demo.db')` in `nba_app.py:_build_graph`.
