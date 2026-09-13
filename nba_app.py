#****************************************************************************
# (C) Cloudera, Inc. 2019-2026
#  All rights reserved.
#
#  Applicable Open Source License: GNU Affero General Public License v3.0
#
# #  Author(s): Paul de Fusco
#***************************************************************************/

"""Next-Best-Action multi-turn chatbot demo.

A LangGraph pipeline that guides a bank customer through a short conversation,
polls an XGBoost fraud-guardrail endpoint on Cloudera AI Inference Service,
consults a local SQLite rule engine of credit-card offers, and presents the
best-fit offer in-chat.  Fully traced through LangSmith with a per-thread
``thread_id`` so the entire conversation appears as one thread in the
LangSmith Threads tab.

Runs as a Cloudera AI Application via ``launch_app.py``.
"""

from __future__ import annotations

import json
import os
import random
import time
import uuid
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

import httpx
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langsmith import Client, traceable
from langsmith.run_helpers import get_current_run_tree
from open_inference.openapi.client import InferenceRequest, OpenInferenceClient
from pydantic import BaseModel, Field

from db import get_conn, init_schema, seed_offers
from offer_rules import select_offers


# ----------------------------------------------------------------------
# Environment / configuration
# ----------------------------------------------------------------------

# Enable LangSmith tracing by default so every turn shows up in the UI.
os.environ.setdefault("LANGSMITH_TRACING", "true")
os.environ.setdefault("LANGSMITH_PROJECT", "nba-demo")

# LLM (Nemotron on Cloudera AI Inference Service)
LLM_MODEL_ID = os.environ.get("LLM_MODEL_ID", "nemotron")
LLM_ENDPOINT_BASE_URL = os.environ.get("LLM_ENDPOINT_BASE_URL", "")
LLM_CDP_TOKEN = os.environ.get("LLM_CDP_TOKEN", "")

# Classifier (XGBoost fraud model on Cloudera AI Inference Service)
CLF_MODEL_ID = os.environ.get("CLF_MODEL_ID", "xgboost")
CLF_ENDPOINT_BASE_URL = os.environ.get("CLF_ENDPOINT_BASE_URL", "")
CLF_CDP_TOKEN = os.environ.get("CLF_CDP_TOKEN", "")

FRAUD_THRESHOLD = float(os.environ.get("FRAUD_THRESHOLD", "0.5"))
NBA_MOCK = os.environ.get("NBA_MOCK", "") == "1"

# LangSmith URLs (for UI links from the sidebar)
LANGSMITH_ORG = os.environ.get("LANGSMITH_ORG", "")
LANGSMITH_PROJECT = os.environ.get("LANGSMITH_PROJECT", "nba-demo")


# ----------------------------------------------------------------------
# LLM setup
# ----------------------------------------------------------------------

_llm: Optional[ChatOpenAI] = None


def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    """Cached ChatOpenAI client pointed at the Nemotron endpoint.

    We defer construction so importing this module doesn't require the env
    vars to be set (evaluators can point at a different LLM).
    """
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=LLM_MODEL_ID,
            base_url=LLM_ENDPOINT_BASE_URL or None,
            api_key=LLM_CDP_TOKEN or "unused",
            temperature=temperature,
        )
    return _llm


# ----------------------------------------------------------------------
# Structured-output schemas
# ----------------------------------------------------------------------


class Features(BaseModel):
    """Features extracted from a single customer message.

    Retains the 14 fields the fraud XGBoost model expects so the payload
    shape is unchanged from the original demo.  Adds NBA-specific fields
    (age, annual_income, requested_credit_line, ...) used by the rule
    engine and clarifying-question prompts.
    """

    # NBA-specific
    age: int = Field(default=0)
    annual_income: float = Field(default=0)
    requested_credit_line: float = Field(default=0)
    existing_debt: float = Field(default=0)
    employment_status: Literal[
        "EMPLOYED", "SELF_EMPLOYED", "STUDENT", "UNEMPLOYED", "RETIRED", "UNKNOWN"
    ] = "UNKNOWN"
    monthly_spend: float = Field(default=0)
    stated_goal: str = Field(default="")

    # Fraud-classifier features (payload shape must match the XGBoost endpoint).
    credit_card_balance: float = 0
    bank_account_balance: float = 0
    mortgage_balance: float = 0
    sec_bank_account_balance: float = 0
    savings_account_balance: float = 0
    sec_savings_account_balance: float = 0
    total_est_nworth: float = 0
    primary_loan_balance: float = 0
    secondary_loan_balance: float = 0
    uni_loan_balance: float = 0
    longitude: float = 0
    latitude: float = 0
    transaction_amount: float = 0


class Intent(BaseModel):
    """Router decision emitted after every user turn."""

    stage: Literal["NEED_MORE_INFO", "READY_FOR_OFFER", "POST_OFFER_CHAT"]
    reasoning: str = ""


FEATURE_ORDER = [
    "age",
    "credit_card_balance",
    "bank_account_balance",
    "mortgage_balance",
    "sec_bank_account_balance",
    "savings_account_balance",
    "sec_savings_account_balance",
    "total_est_nworth",
    "primary_loan_balance",
    "secondary_loan_balance",
    "uni_loan_balance",
    "longitude",
    "latitude",
    "transaction_amount",
]


# ----------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------

feature_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a feature-extraction engine.

Extract these fields from the user's latest message:
- age (int)
- annual_income (float)
- requested_credit_line (float)
- existing_debt (float)
- employment_status ("EMPLOYED"|"SELF_EMPLOYED"|"STUDENT"|"UNEMPLOYED"|"RETIRED"|"UNKNOWN")
- monthly_spend (float)
- stated_goal (short free-text summary of what the customer wants)
- credit_card_balance, bank_account_balance, mortgage_balance,
  sec_bank_account_balance, savings_account_balance,
  sec_savings_account_balance, total_est_nworth, primary_loan_balance,
  secondary_loan_balance, uni_loan_balance, longitude, latitude,
  transaction_amount (all floats)

Rules:
- If a value is not stated, return 0 (or "UNKNOWN" for employment_status, "" for stated_goal).
- Do NOT infer values that were not explicitly stated.
- Only return numeric values for numeric fields.
""",
        ),
        ("human", "{input}"),
    ]
)

intent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are the intent router for a credit-card recommendation chatbot.

Classify the conversation state given the full message history and the
features already gathered.

Return one of:
- NEED_MORE_INFO   — we don't yet know enough about the customer's goal,
                     income, or debt situation to make a specific recommendation.
- READY_FOR_OFFER  — the customer has stated a clear goal AND we have at
                     least an approximate income OR employment status.  The
                     next turn should present a specific offer.
- POST_OFFER_CHAT  — an offer has ALREADY been presented in a prior assistant
                     turn (look for `[OFFER PRESENTED: ...]` markers).  The
                     customer is asking follow-up questions or negotiating.

Be conservative: prefer NEED_MORE_INFO if the goal is vague.  Prefer
POST_OFFER_CHAT whenever an offer has already been presented, even for
questions that could arguably reset the flow.
""",
        ),
        (
            "human",
            "Conversation so far:\n{transcript}\n\n"
            "Accumulated features (JSON):\n{features}\n\n"
            "Offer already presented? {offer_presented}\n\n"
            "What is the current stage?",
        ),
    ]
)

clarify_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a friendly banking assistant helping a customer find
the best credit-card offer.  Ask ONE short, specific clarifying question to
close the biggest gap in what you know so far.

Guidelines:
- Never re-ask for a field that is already populated in the features JSON.
- Keep it under 40 words.
- Sound conversational, not like a form.
- If the customer mentioned a goal (e.g. travel, rebuilding credit,
  consolidating debt), acknowledge it briefly first.
""",
        ),
        (
            "human",
            "Conversation so far:\n{transcript}\n\n"
            "Accumulated features (JSON):\n{features}\n\n"
            "Ask your next clarifying question.",
        ),
    ]
)

offer_pitch_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a friendly banking assistant presenting a credit-card
offer to a customer in a chat.  Do NOT write an email.  Write a warm, concise
chat message (under 120 words) that:

1. Names the offer.
2. Highlights the two most relevant benefits given what the customer said.
3. Mentions the APR range.
4. Ends with a soft, open call-to-action question.

Prefix your message with `[OFFER PRESENTED: <offer_id>]` on its own first
line so downstream nodes can detect that an offer has already been made.
Then leave a blank line and start the customer-visible message.
""",
        ),
        (
            "human",
            "Customer profile:\n"
            "- Name: {full_name}\n"
            "- Risk tier: {risk_tier}\n"
            "- Employment: {employment_status}\n\n"
            "Conversation so far:\n{transcript}\n\n"
            "Recommended offer:\n"
            "- offer_id: {offer_id}\n"
            "- offer_name: {offer_name}\n"
            "- marketing_hook: {marketing_hook}\n"
            "- rewards_summary: {rewards_summary}\n"
            "- apr_range: {apr_range}\n"
            "- cta_text: {cta_text}\n\n"
            "Write your pitch.",
        ),
    ]
)

post_offer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a friendly banking assistant. The customer has already
been shown a specific credit-card offer (details below). Answer their next
question grounded ONLY on the offer details and general facts about credit
cards.  If they ask something you don't know, say so and offer to connect
them with a specialist.  Keep it under 120 words.
""",
        ),
        (
            "human",
            "Presented offer:\n{offer_json}\n\n"
            "Conversation so far:\n{transcript}\n\n"
            "Respond to the customer's latest message.",
        ),
    ]
)

decline_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a banking assistant.  Our fraud guardrail flagged this
session as unusual.  Reply politely, do NOT reveal the fraud score, and
suggest the customer verify recent activity in their banking app or contact
support.  Under 80 words.  Chat tone.
""",
        ),
        ("human", "Conversation so far:\n{transcript}"),
    ]
)


# ----------------------------------------------------------------------
# State
# ----------------------------------------------------------------------


class GraphState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    thread_id: str
    customer_id: Optional[int]
    accumulated_features: Dict[str, Any]
    fraud_probability: Optional[float]
    fraud_blocked: bool
    customer_record: Optional[Dict[str, Any]]
    selected_offer: Optional[Dict[str, Any]]
    candidate_offers: Optional[List[Dict[str, Any]]]
    conversation_stage: Literal["gathering", "offer_presented", "post_offer"]
    # Transient hop from intent_router -> stage_router. Must be declared on
    # the TypedDict or LangGraph drops it as an unknown key when persisting.
    _next_stage: Optional[str]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _transcript(messages: List[BaseMessage]) -> str:
    """Render message list as a compact transcript for prompt injection."""
    lines = []
    for m in messages:
        role = "CUSTOMER" if isinstance(m, HumanMessage) else "ASSISTANT"
        lines.append(f"{role}: {m.content}")
    return "\n".join(lines) if lines else "(no messages yet)"


def _merge_features(current: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """Merge freshly extracted features into the accumulator.

    Only overwrite when the new value is meaningful (non-zero, non-UNKNOWN,
    non-empty).  Preserves earlier information across turns.
    """
    merged = dict(current or {})
    for key, val in new.items():
        if val in (0, 0.0, "", None, "UNKNOWN"):
            continue
        merged[key] = val
    return merged


def _load_customer(customer_id: Optional[int]) -> Dict[str, Any]:
    """Fetch a customer row from SQLite.  Falls back to a random row."""
    conn = get_conn()
    try:
        init_schema(conn)
        if customer_id is None:
            max_id_row = conn.execute("SELECT COALESCE(MAX(customer_id), 0) FROM customers").fetchone()
            max_id = max_id_row[0] if max_id_row else 0
            if max_id == 0:
                # No customers seeded — return a synthetic default so the app
                # can still run (e.g. during ad-hoc dev).
                return {
                    "customer_id": 0,
                    "full_name": "Demo Customer",
                    "email": "demo@example.com",
                    "risk_tier": "MED",
                    "age": 35,
                    "annual_income": 60_000,
                    "employment_status": "EMPLOYED",
                    "existing_debt": 5_000,
                }
            customer_id = random.randint(1, max_id)
        row = conn.execute(
            "SELECT * FROM customers WHERE customer_id = ?", (customer_id,)
        ).fetchone()
        if row is None:
            return {"customer_id": customer_id, "full_name": "Unknown", "risk_tier": "MED"}
        return {k: row[k] for k in row.keys()}
    finally:
        conn.close()


# ----------------------------------------------------------------------
# XGBoost fraud call (unchanged payload shape from the original demo)
# ----------------------------------------------------------------------


@traceable(run_type="tool", name="xgboost_infer")
def _xgboost_infer(features: Dict[str, Any]) -> float:
    """Poll the Cloudera AI Inference Service XGBoost endpoint.

    Returns the P(fraud) probability. When NBA_MOCK=1, short-circuits so
    the demo can run without a live endpoint.
    """
    if NBA_MOCK or not CLF_ENDPOINT_BASE_URL:
        # In mock mode a large transaction_amount trips the guardrail so the
        # decline path is exercisable end-to-end without a live endpoint.
        txn = float(features.get("transaction_amount", 0) or 0)
        return 0.9 if txn > 5_000 else 0.05

    headers = {
        "Authorization": f"Bearer {CLF_CDP_TOKEN}",
        "Content-Type": "application/json",
    }
    httpx_client = httpx.Client(headers=headers)
    client = OpenInferenceClient(base_url=CLF_ENDPOINT_BASE_URL, httpx_client=httpx_client)

    client.check_server_readiness()
    _ = client.read_model_metadata(CLF_MODEL_ID)

    ordered_values = [float(features.get(f, 0) or 0) for f in FEATURE_ORDER]
    payload = {
        "parameters": {"content_type": "pd"},
        "inputs": [
            {
                "name": "input",
                "datatype": "FP32",
                "shape": [1, len(FEATURE_ORDER)],
                "data": [ordered_values],
            }
        ],
    }

    start = time.time()
    pred = client.model_infer(
        CLF_MODEL_ID,
        request=InferenceRequest(inputs=payload["inputs"]),
    )
    latency = time.time() - start
    resp_json = json.loads(pred.json())
    # Preserved from the original: fraud probability is the second element
    # of the second output row.
    fraud_proba = float(resp_json["outputs"][1]["data"][1])
    print(f"[xgboost_infer] fraud_proba={fraud_proba:.4f} latency={latency:.2f}s")
    return fraud_proba


# ----------------------------------------------------------------------
# Nodes
# ----------------------------------------------------------------------


@traceable(run_type="chain", name="intake_node")
def intake_node(state: GraphState) -> Dict[str, Any]:
    """First node every turn: extract features from the latest user message
    and merge them into the running feature dict.  Also lazy-loads the
    customer record on the first turn of the thread.
    """
    latest_user = ""
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage):
            latest_user = m.content
            break

    features_chain = feature_prompt | get_llm().with_structured_output(Features)
    if NBA_MOCK:
        # Skip a live LLM call in mock mode; return a fixed profile.  A
        # heuristic pattern-match on the user text lets the demo exercise
        # the fraud path without a live endpoint — any turn that mentions a
        # large transaction amount populates ``transaction_amount`` so the
        # mocked XGBoost stub trips the guardrail.
        import re
        txn_amount = 0.0
        for pat in (
            r"\$\s*([\d,]+(?:\.\d+)?)\s*k?",
            r"([\d,]+)\s*dollars?",
            r"transaction[^\d]*([\d,]+)",
            r"amount[^\d]*([\d,]+)",
        ):
            m = re.search(pat, latest_user, flags=re.I)
            if m:
                try:
                    txn_amount = max(txn_amount, float(m.group(1).replace(",", "")))
                except ValueError:
                    pass
        extracted = Features(
            age=42,
            annual_income=180_000,
            requested_credit_line=25_000,
            existing_debt=5_000,
            employment_status="EMPLOYED",
            monthly_spend=4_000,
            transaction_amount=txn_amount,
            stated_goal=latest_user[:80],
        )
    else:
        extracted = features_chain.invoke({"input": latest_user})

    new_features = extracted.dict() if hasattr(extracted, "dict") else dict(extracted)
    merged = _merge_features(state.get("accumulated_features", {}), new_features)

    updates: Dict[str, Any] = {"accumulated_features": merged}

    # Lazy customer lookup on first turn.
    if state.get("customer_record") is None:
        updates["customer_record"] = _load_customer(state.get("customer_id"))

    # Default conversation_stage.
    if state.get("conversation_stage") is None:
        updates["conversation_stage"] = "gathering"

    return updates


@traceable(run_type="chain", name="fraud_guardrail_node")
def fraud_guardrail_node(state: GraphState) -> Dict[str, Any]:
    """Poll the XGBoost fraud model once per thread (results cached).

    Skips the call if we haven't accumulated any financial features yet —
    an all-zero payload is not informative and burns endpoint quota.
    """
    if state.get("fraud_probability") is not None:
        return {}

    features = state.get("accumulated_features", {}) or {}
    financial_signal = sum(
        float(features.get(f, 0) or 0) for f in FEATURE_ORDER if f != "age"
    )
    if financial_signal == 0 and not NBA_MOCK:
        # Not enough info to run the classifier — assume low fraud until
        # more features come in on later turns.
        return {"fraud_probability": 0.0, "fraud_blocked": False}

    proba = _xgboost_infer(features)
    return {
        "fraud_probability": proba,
        "fraud_blocked": proba >= FRAUD_THRESHOLD,
    }


def fraud_router(state: GraphState) -> Literal["decline", "continue"]:
    return "decline" if state.get("fraud_blocked") else "continue"


@traceable(run_type="chain", name="intent_router_node")
def intent_router_node(state: GraphState) -> Dict[str, Any]:
    """LLM-classified conversation stage router.

    Writes ``conversation_stage`` in state so downstream nodes and the
    Streamlit UI can inspect it.
    """
    transcript = _transcript(state.get("messages", []))
    features = state.get("accumulated_features", {}) or {}
    offer_presented = state.get("selected_offer") is not None

    if NBA_MOCK:
        # Deterministic staging for the mock path: first turn asks a
        # clarifying question, second turn presents an offer, subsequent
        # turns are post-offer chat.
        n_human = sum(1 for m in state.get("messages", []) if isinstance(m, HumanMessage))
        if offer_presented:
            stage = "POST_OFFER_CHAT"
        elif n_human >= 2:
            stage = "READY_FOR_OFFER"
        else:
            stage = "NEED_MORE_INFO"
    else:
        chain = intent_prompt | get_llm().with_structured_output(Intent)
        decision = chain.invoke(
            {
                "transcript": transcript,
                "features": json.dumps(features, default=str),
                "offer_presented": "yes" if offer_presented else "no",
            }
        )
        stage = decision.stage

    stage_map = {
        "NEED_MORE_INFO": "gathering",
        "READY_FOR_OFFER": "offer_presented",  # will be set for real by presentation node
        "POST_OFFER_CHAT": "post_offer",
    }
    return {"_next_stage": stage, "conversation_stage": stage_map[stage]}


def stage_router(state: GraphState) -> Literal["clarify", "select_offer", "post_offer_chat"]:
    stage = state.get("_next_stage", "NEED_MORE_INFO")
    if stage == "READY_FOR_OFFER":
        return "select_offer"
    if stage == "POST_OFFER_CHAT":
        return "post_offer_chat"
    return "clarify"


@traceable(run_type="chain", name="clarify_node")
def clarify_node(state: GraphState) -> Dict[str, Any]:
    transcript = _transcript(state.get("messages", []))
    features = state.get("accumulated_features", {}) or {}

    if NBA_MOCK:
        content = (
            "Thanks for reaching out! To point you at the best card, could you tell me a bit "
            "about what you're looking for — travel rewards, everyday cashback, help with debt, "
            "or something else — and roughly what your annual income is?"
        )
    else:
        chain = clarify_prompt | get_llm()
        resp = chain.invoke(
            {"transcript": transcript, "features": json.dumps(features, default=str)}
        )
        content = resp.content

    return {"messages": [AIMessage(content=content)]}


@traceable(run_type="retriever", name="offer_selection_node")
def offer_selection_node(state: GraphState) -> Dict[str, Any]:
    conn = get_conn()
    try:
        init_schema(conn)
        seed_offers(conn)  # idempotent — cheap safety net
        candidates = select_offers(
            conn,
            customer=state.get("customer_record") or {},
            features=state.get("accumulated_features", {}),
            top_k=3,
        )
    finally:
        conn.close()

    selected = candidates[0] if candidates else None
    return {"selected_offer": selected, "candidate_offers": candidates}


@traceable(run_type="chain", name="offer_presentation_node")
def offer_presentation_node(state: GraphState) -> Dict[str, Any]:
    offer = state.get("selected_offer")
    customer = state.get("customer_record") or {}
    features = state.get("accumulated_features", {}) or {}
    transcript = _transcript(state.get("messages", []))

    if offer is None:
        # No offer available — apologise gracefully; still ends the turn.
        content = (
            "Thanks for the details — based on what you've shared I don't have a card that's a "
            "great fit right now, but I can connect you with a specialist who can look at "
            "options in more depth."
        )
        return {"messages": [AIMessage(content=content)]}

    if NBA_MOCK:
        content = (
            f"[OFFER PRESENTED: {offer['offer_id']}]\n\n"
            f"Based on what you've told me, I'd recommend our **{offer['offer_name']}**. "
            f"{offer['marketing_hook']} APR range is {offer['apr_range']}. "
            f"{offer['cta_text']}"
        )
    else:
        chain = offer_pitch_prompt | get_llm()
        resp = chain.invoke(
            {
                "full_name": customer.get("full_name", "there"),
                "risk_tier": customer.get("risk_tier", "MED"),
                "employment_status": features.get("employment_status", "UNKNOWN"),
                "transcript": transcript,
                "offer_id": offer["offer_id"],
                "offer_name": offer["offer_name"],
                "marketing_hook": offer["marketing_hook"],
                "rewards_summary": offer["rewards_summary"],
                "apr_range": offer["apr_range"],
                "cta_text": offer["cta_text"],
            }
        )
        content = resp.content

    return {
        "messages": [AIMessage(content=content)],
        "conversation_stage": "offer_presented",
    }


@traceable(run_type="chain", name="post_offer_chat_node")
def post_offer_chat_node(state: GraphState) -> Dict[str, Any]:
    offer = state.get("selected_offer") or {}
    transcript = _transcript(state.get("messages", []))

    if NBA_MOCK:
        content = (
            f"Happy to answer more about {offer.get('offer_name', 'the card')}. "
            "In a live demo this would use Nemotron to answer questions grounded on the offer's "
            "APR, rewards, and fees."
        )
    else:
        chain = post_offer_prompt | get_llm()
        resp = chain.invoke(
            {
                "offer_json": json.dumps(offer, default=str),
                "transcript": transcript,
            }
        )
        content = resp.content

    return {"messages": [AIMessage(content=content)], "conversation_stage": "post_offer"}


@traceable(run_type="chain", name="decline_node")
def decline_node(state: GraphState) -> Dict[str, Any]:
    transcript = _transcript(state.get("messages", []))

    if NBA_MOCK:
        content = (
            "Thanks for reaching out. For your protection I'm unable to continue this session "
            "in chat — please verify recent activity in your banking app or contact our support "
            "line to continue."
        )
    else:
        chain = decline_prompt | get_llm()
        content = chain.invoke({"transcript": transcript}).content

    return {"messages": [AIMessage(content=content)], "fraud_blocked": True}


# ----------------------------------------------------------------------
# Build the graph
# ----------------------------------------------------------------------


def _build_graph():
    builder = StateGraph(GraphState)
    builder.add_node("intake", intake_node)
    builder.add_node("fraud_guardrail", fraud_guardrail_node)
    builder.add_node("intent_router", intent_router_node)
    builder.add_node("clarify", clarify_node)
    builder.add_node("select_offer", offer_selection_node)
    builder.add_node("offer_presentation", offer_presentation_node)
    builder.add_node("post_offer_chat", post_offer_chat_node)
    builder.add_node("decline", decline_node)

    builder.set_entry_point("intake")
    builder.add_edge("intake", "fraud_guardrail")
    builder.add_conditional_edges(
        "fraud_guardrail",
        fraud_router,
        {"decline": "decline", "continue": "intent_router"},
    )
    builder.add_conditional_edges(
        "intent_router",
        stage_router,
        {
            "clarify": "clarify",
            "select_offer": "select_offer",
            "post_offer_chat": "post_offer_chat",
        },
    )
    builder.add_edge("select_offer", "offer_presentation")
    builder.add_edge("clarify", END)
    builder.add_edge("offer_presentation", END)
    builder.add_edge("post_offer_chat", END)
    builder.add_edge("decline", END)

    # MemorySaver keeps thread state alive within a single process; swap for
    # SqliteSaver to persist across app restarts.
    return builder.compile(checkpointer=MemorySaver())


graph = _build_graph()


# ----------------------------------------------------------------------
# Per-turn entry point (LangSmith root trace)
# ----------------------------------------------------------------------


@traceable(run_type="chain", name="nba_turn")
def run_turn(
    user_input: str,
    thread_id: str,
    customer_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Run one turn of the chatbot.

    The @traceable decorator makes this the root of the per-turn trace in
    LangSmith.  The ``thread_id`` metadata is what LangSmith Threads groups
    on; ``configurable.thread_id`` is what LangGraph's checkpointer keys on.
    Setting both is required.
    """
    config = {
        "configurable": {"thread_id": thread_id},
        "metadata": {"thread_id": thread_id, "customer_id": customer_id},
    }

    input_state = {"messages": [HumanMessage(content=user_input)]}
    # Seed customer_id only on the first invocation; the checkpointer keeps
    # it in subsequent turns.
    if customer_id is not None:
        input_state["customer_id"] = customer_id

    final_state = graph.invoke(input_state, config=config)

    rt = get_current_run_tree()
    run_id = str(rt.id) if rt else None

    return {
        "run_id": run_id,
        "assistant_message": final_state["messages"][-1].content
        if final_state.get("messages")
        else "",
        "messages": final_state.get("messages", []),
        "selected_offer": final_state.get("selected_offer"),
        "candidate_offers": final_state.get("candidate_offers"),
        "fraud_probability": final_state.get("fraud_probability"),
        "fraud_blocked": final_state.get("fraud_blocked", False),
        "conversation_stage": final_state.get("conversation_stage"),
        "accumulated_features": final_state.get("accumulated_features", {}),
        "customer_record": final_state.get("customer_record"),
    }


# ----------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------


def run_streamlit_app() -> None:
    import streamlit as st

    st.set_page_config(page_title="NBA Credit-Card Chatbot", layout="wide")
    st.title("Next-Best-Action Credit-Card Chatbot")

    # --- Session state ------------------------------------------------
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "history" not in st.session_state:
        # list of dicts: {"role": ..., "content": ..., "run_id": ...}
        st.session_state.history = []
    if "customer_id" not in st.session_state:
        st.session_state.customer_id = None
    if "last_result" not in st.session_state:
        st.session_state.last_result = None

    # --- Sidebar ------------------------------------------------------
    with st.sidebar:
        st.header("Session")
        st.text_input(
            "thread_id",
            value=st.session_state.thread_id,
            key="_thread_id_display",
            disabled=True,
        )
        cid_str = st.text_input(
            "customer_id (blank = random)",
            value="" if st.session_state.customer_id is None else str(st.session_state.customer_id),
        )
        if cid_str.strip():
            try:
                st.session_state.customer_id = int(cid_str)
            except ValueError:
                st.warning("customer_id must be an integer")
        else:
            st.session_state.customer_id = None

        if st.button("New conversation"):
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.history = []
            st.session_state.last_result = None
            st.rerun()

        st.markdown("---")
        st.caption("LangSmith")
        if LANGSMITH_ORG:
            thread_url = (
                f"https://smith.langchain.com/o/{LANGSMITH_ORG}"
                f"/projects/p/{LANGSMITH_PROJECT}/t/{st.session_state.thread_id}"
            )
            st.markdown(f"[Open thread ↗]({thread_url})")
        else:
            st.caption("Set LANGSMITH_ORG to enable one-click thread links.")

        # Debug pane — surface what the router / rule engine is doing.
        st.markdown("---")
        st.caption("Debug")
        last = st.session_state.last_result
        if last is not None:
            st.write("Stage:", last.get("conversation_stage"))
            st.write(
                "Fraud probability:",
                f"{last.get('fraud_probability'):.2%}"
                if last.get("fraud_probability") is not None
                else "n/a",
            )
            if last.get("selected_offer"):
                st.write("Selected offer:", last["selected_offer"]["offer_id"])
            with st.expander("Accumulated features"):
                st.json(last.get("accumulated_features", {}))
            with st.expander("Candidate offers"):
                st.json(
                    [
                        {"offer_id": o["offer_id"], "offer_name": o["offer_name"]}
                        for o in (last.get("candidate_offers") or [])
                    ]
                )

    # --- Chat history rendering --------------------------------------
    for i, msg in enumerate(st.session_state.history):
        with st.chat_message(msg["role"]):
            content = msg["content"]
            # Hide the [OFFER PRESENTED: ...] marker line from users; it's
            # kept in state for the router.
            if content.startswith("[OFFER PRESENTED:"):
                content = "\n".join(content.splitlines()[2:]) or content
            st.markdown(content)

            if msg["role"] == "assistant" and msg.get("run_id"):
                run_id = msg["run_id"]
                cols = st.columns([1, 1, 6])
                if cols[0].button("👍", key=f"up_{run_id}_{i}"):
                    try:
                        Client().create_feedback(run_id, key="user_thumbs", score=1)
                        st.toast("Thanks for the feedback!")
                    except Exception as exc:  # noqa: BLE001
                        st.warning(f"LangSmith feedback failed: {exc}")
                if cols[1].button("👎", key=f"dn_{run_id}_{i}"):
                    try:
                        Client().create_feedback(run_id, key="user_thumbs", score=0)
                        st.toast("Thanks for the feedback!")
                    except Exception as exc:  # noqa: BLE001
                        st.warning(f"LangSmith feedback failed: {exc}")

    # --- Input --------------------------------------------------------
    prompt = st.chat_input("Type your message...")
    if prompt:
        st.session_state.history.append({"role": "user", "content": prompt, "run_id": None})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = run_turn(
                    prompt,
                    st.session_state.thread_id,
                    st.session_state.customer_id,
                )
            content = result["assistant_message"]
            display = content
            if display.startswith("[OFFER PRESENTED:"):
                display = "\n".join(display.splitlines()[2:]) or display
            st.markdown(display)

        st.session_state.history.append(
            {"role": "assistant", "content": content, "run_id": result.get("run_id")}
        )
        st.session_state.last_result = result
        st.rerun()


if __name__ == "__main__":
    run_streamlit_app()
