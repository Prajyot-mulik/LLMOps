# main.py
import os
import json
import secrets
from datetime import datetime
from typing import Optional, Dict, Any, List

import streamlit as st
from dotenv import load_dotenv

# Monitoring
from monitoring import (
    record_request_start,
    record_request_end,
    record_llm_tokens,
    record_error,
    record_feedback,
    start_tracing_span,
    set_trace_attributes,
)

# LangChain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI  # OpenRouter via OpenAI-compatible client
from pydantic import BaseModel, Field, ValidationError

# Guardrails AI
from guardrails import Guard, Rail
from guardrails.validators import ValidJSON

# PromptLayer (manual logging)
import promptlayer

# -------------------------
# Environment and config
# -------------------------
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
PROMPTLAYER_API_KEY = os.getenv("PROMPTLAYER_API_KEY", "")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
OTLP_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "streamlit-llm-app")
PROM_PORT = int(os.getenv("PROMETHEUS_PORT", "9108"))

if not OPENROUTER_API_KEY:
    st.error("OPENROUTER_API_KEY is missing. Set it in .env or container env.")
    st.stop()

if PROMPTLAYER_API_KEY:
    promptlayer.api_key = PROMPTLAYER_API_KEY

def read_file(path: str, default: str = "") -> str:
    try:
        with open(path, "r") as f:
            return f.read().strip()
    except Exception:
        return default

SYSTEM_PROMPT = read_file("prompts/base_v1.md",
                          "You are a helpful AI assistant. Be friendly, concise, and accurate.")
STRUCTURED_INSTR = read_file("prompts/structured_v1.md", "")

# Load Guardrails RAIL
RAIL_PATH = "rails/safe_output.rail"
rail = Rail.from_string(read_file(RAIL_PATH, ""))
guard = Guard.from_rail_string(read_file(RAIL_PATH, ""))

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="LLMOps Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Chatbot with Guardrails AI, PromptLayer, Monitoring, and Developer View")
tab_chat, tab_dev = st.tabs(["Chat", "Developer view"])

# -------------------------
# Session state
# -------------------------
if "user_id" not in st.session_state:
    st.session_state.user_id = secrets.token_hex(8)
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------
# JSON logging
# -------------------------
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, f"user_{st.session_state.user_id}.json")

if not os.path.exists(log_file):
    with open(log_file, "w") as f:
        json.dump(
            {"user_id": st.session_state.user_id, "system_prompt": SYSTEM_PROMPT, "chat_history": []},
            f,
            indent=4,
        )

def log_message(role, content, feedback=None, update_last=False):
    with open(log_file, "r") as f:
        data = json.load(f)
    if update_last and role == "feedback":
        for msg in reversed(data["chat_history"]):
            if msg["role"] == "assistant" and msg.get("feedback") is None:
                msg["feedback"] = content
                break
    else:
        data["chat_history"].append(
            {"timestamp": datetime.now().isoformat(), "role": role, "content": content, "feedback": feedback}
        )
    with open(log_file, "w") as f:
        json.dump(data, f, indent=4)

if "system_logged" not in st.session_state:
    log_message("system", SYSTEM_PROMPT)
    st.session_state.system_logged = True

# -------------------------
# LLM via OpenRouter (OpenAI-compatible)
# -------------------------
llm = ChatOpenAI(
    model="meta-llama/llama-3.1-8b-instruct",
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    temperature=0.2,
    max_tokens=800,
)

# -------------------------
# Guardrails schema model (for internal parsing in UI)
# -------------------------
class SafeAnswer(BaseModel):
    summary: str = Field(..., description="Short, clear answer for the user.")
    steps: Optional[List[str]] = Field(None, description="Optional steps or bullets.")
    code: Optional[str] = Field(None, description="Optional code block if relevant.")

structured_parser = PydanticOutputParser(pydantic_object=SafeAnswer)

def needs_structured_output(user_text: str) -> bool:
    kws = ["how do i", "steps", "tutorial", "example", "code", "snippet", "script"]
    t = user_text.lower()
    return any(k in t for k in kws)

def apply_guardrails_rail(user_text: str, llm_text: str) -> str:
    """
    Validate and correct the LLM output with Guardrails AI:
    - Enforce JSON structure per rail schema
    - Apply blocked content filters
    """
    try:
        # rail expects a single string; we validate the generated text
        validated = guard.parse(llm_text)
        if isinstance(validated, dict):
            # If the rail returns structured dict, recompose a friendly output
            reply = validated.get("summary", "")
            steps = validated.get("steps", []) or []
            code = validated.get("code", "") or ""
            if steps:
                reply += "\n\nSteps:\n" + "\n".join(f"- {s}" for s in steps)
            if code:
                reply += f"\n\nCode:\n```python\n{code}\n```"
            return reply or "I can't help with that. Let's keep things safe and constructive."
        # If string, return as is
        return str(validated)
    except Exception:
        # If guardrails fails, fall back to plain text with a safety message if needed
        # Basic keyword check mirroring rail filters
        blocked_keywords = [
            "self-harm", "suicide", "harm others", "make a bomb", "weapon", "illegal", "deepfake", "spread malware"
        ]
        t_in = (user_text or "").lower()
        t_out = (llm_text or "").lower()
        if any(b in t_in or b in t_out for b in blocked_keywords):
            return "I can't help with that. Let's keep things safe and constructive."
        return llm_text

# -------------------------
# Prompts (LangChain)
# -------------------------
base_prompt = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), MessagesPlaceholder("history"), ("human", "{input}")]
)
structured_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT + "\n\n" + STRUCTURED_INSTR),
        ("system", structured_parser.get_format_instructions()),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ]
)

# -------------------------
# Chat function with traces, metrics, guardrails, and PromptLayer logging
# -------------------------
def chat_with_model(user_message: str) -> str:
    set_trace_attributes({"user_id": st.session_state.user_id})
    span = start_tracing_span("llm_request")
    try:
        record_request_start("chat")

        history = []
        for m in st.session_state.messages:
            if m["role"] == "user":
                history.append(HumanMessage(m["content"]))
            elif m["role"] == "assistant":
                history.append(AIMessage(m["content"]))

        if needs_structured_output(user_message):
            prompt = structured_prompt.format_prompt(history=history, input=user_message)
            messages = prompt.to_messages()
        else:
            prompt = base_prompt.format_prompt(history=history, input=user_message)
            messages = prompt.to_messages()

        result = llm.invoke(messages)
        raw_reply = result.content

        # Guardrails AI validation/correction
        safe_reply = apply_guardrails_rail(user_message, raw_reply)

        # Token usage (best effort)
        usage: Dict[str, Any] = getattr(result, "response_metadata", {}).get("token_usage", {})
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens))
        record_llm_tokens(prompt_tokens, completion_tokens, total_tokens)

        # PromptLayer logging (if enabled)
        try:
            if PROMPTLAYER_API_KEY:
                promptlayer.track.event(
                    name="chat_completion",
                    metadata={
                        "system_prompt_version": "base_v1.md",
                        "structured_instructions_version": "structured_v1.md",
                        "user_id": st.session_state.user_id,
                        "messages": [m.model_dump() if hasattr(m, "model_dump") else str(m) for m in messages],
                        "raw_reply": raw_reply,
                        "safe_reply": safe_reply,
                        "model": "meta-llama/llama-3.1-8b-instruct",
                        "base_url": "https://openrouter.ai/api/v1",
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                    },
                )
        except Exception:
            pass  # don't break the chat if logging fails

        record_request_end("chat")
        span.end()
        return safe_reply
    except Exception as e:
        record_error("chat", str(e))
        span.record_exception(e)
        span.end()
        return "Sorry, something went wrong. Please try again."

# -------------------------
# Chat tab UI
# -------------------------
with tab_chat:
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
        if msg["role"] == "assistant" and idx == len(st.session_state.messages) - 1:
            c1, c2, c3 = st.columns(3)
            k = f"feedback_{idx}"
            if c1.button("👍 Good", key=f"{k}_good"):
                log_message("feedback", "good", update_last=True)
                record_feedback("good")
                st.success("Thanks for your feedback!")
            if c2.button("😐 Neutral", key=f"{k}_neutral"):
                log_message("feedback", "neutral", update_last=True)
                record_feedback("neutral")
                st.info("Thanks for your feedback!")
            if c3.button("👎 Bad", key=f"{k}_bad"):
                log_message("feedback", "bad", update_last=True)
                record_feedback("bad")
                st.warning("Generating an improved response...")
                last_user_msg = next((m["content"] for m in reversed(st.session_state.messages) if m["role"] == "user"), None)
                if last_user_msg:
                    new_reply = chat_with_model(last_user_msg)
                    st.session_state.messages.append({"role": "assistant", "content": new_reply})
                    log_message("assistant", new_reply)
                    st.experimental_rerun()

    if user_input := st.chat_input("Type your message..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        log_message("user", user_input)
        with st.chat_message("user"):
            st.write(user_input)

        bot_reply = chat_with_model(user_input)
        st.session_state.messages.append({"role": "assistant", "content": bot_reply})
        log_message("assistant", bot_reply)

        with st.chat_message("assistant"):
            st.write(bot_reply)

# -------------------------
# Developer view tab (links)
# -------------------------
with tab_dev:
    st.subheader("Developer view: Monitoring and observability")
    if not PUBLIC_BASE_URL:
        st.warning("Set PUBLIC_BASE_URL in .env (e.g., http://<EC2_PUBLIC_IP>) for clickable links.")

    def make_url(port_default: str) -> str:
        return f"{PUBLIC_BASE_URL}{port_default}" if PUBLIC_BASE_URL else f"http://<EC2_PUBLIC_IP>{port_default}"

    st.markdown(f"- App: [{make_url(':8501')}]({make_url(':8501')})")
    st.markdown(f"- Prometheus: [{make_url(':9090')}]({make_url(':9090')})")
    st.markdown(f"- Grafana: [{make_url(':3000')}]({make_url(':3000')})")
    st.markdown(f"- Jaeger: [{make_url(':16686')}]({make_url(':16686')})")
    st.markdown(f"- Metrics endpoint: [{make_url(':9108')}]({make_url(':9108')})")

    st.caption("Common PromQL:")
    st.code(
        "histogram_quantile(0.95, sum(rate(llm_latency_seconds_bucket[5m])) by (le))\n"
        "sum(rate(llm_requests_total[1m]))\n"
        "sum(increase(llm_total_tokens_total[15m]))\n"
        "sum(increase(llm_errors_total[15m])) by (route)\n"
        "sum(increase(llm_feedback_total[30m])) by (value)",
        language="text",
    )
    st.caption("Jaeger service:")
    st.code(f"{SERVICE_NAME}", language="text")
    st.caption("OTLP endpoint:")
    st.code(OTLP_ENDPOINT or "Not set (example: http://<EC2_PUBLIC_IP>:4317)", language="text")
