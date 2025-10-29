# monitoring.py
import os
import time
from typing import Dict, Any
from threading import Thread

from prometheus_client import Counter, Histogram, Gauge, start_http_server

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

METRICS_PORT = int(os.getenv("PROMETHEUS_PORT", "9108"))
_started = False

def _start_metrics_server():
    global _started
    if not _started:
        start_http_server(METRICS_PORT)
        _started = True

Thread(target=_start_metrics_server, daemon=True).start()

REQUESTS = Counter("llm_requests_total", "Total LLM requests", ["route"])
LATENCY = Histogram("llm_latency_seconds", "End-to-end latency", ["route"])
TOKENS_PROMPT = Counter("llm_prompt_tokens_total", "Prompt tokens")
TOKENS_COMPLETION = Counter("llm_completion_tokens_total", "Completion tokens")
TOKENS_TOTAL = Counter("llm_total_tokens_total", "Total tokens")
ERRORS = Counter("llm_errors_total", "Errors", ["route", "type"])
FEEDBACK = Counter("llm_feedback_total", "User feedback counts", ["value"])
INFLIGHT = Gauge("llm_inflight_requests", "In-flight requests")

service_name = os.getenv("OTEL_SERVICE_NAME", "streamlit-llm-app")
otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()

provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
if otlp_endpoint:
    processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint))
    provider.add_span_processor(processor)
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)

timers: Dict[str, float] = {}

def record_request_start(route: str):
    REQUESTS.labels(route=route).inc()
    INFLIGHT.inc()
    timers[f"__timer_{route}"] = time.time()

def record_request_end(route: str):
    start_t = timers.pop(f"__timer_{route}", None)
    if start_t is not None:
        LATENCY.labels(route=route).observe(time.time() - start_t)
    INFLIGHT.dec()

def record_llm_tokens(prompt_tokens: int, completion_tokens: int, total_tokens: int):
    TOKENS_PROMPT.inc(prompt_tokens)
    TOKENS_COMPLETION.inc(completion_tokens)
    TOKENS_TOTAL.inc(total_tokens)

def record_error(route: str, err_type: str = "runtime"):
    ERRORS.labels(route=route, type=err_type).inc()

def record_feedback(value: str):
    FEEDBACK.labels(value=value).inc()

def start_tracing_span(name: str):
    return tracer.start_span(name)

def set_trace_attributes(attrs: Dict[str, Any]):
    span = trace.get_current_span()
    for k, v in attrs.items():
        span.set_attribute(k, v)
