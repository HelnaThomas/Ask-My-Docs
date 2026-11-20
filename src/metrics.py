import json
from pathlib import Path
from typing import List
import numpy as np
from pydantic import BaseModel

METRICS_LOG = "./logs/telemetry.jsonl"

class RequestTelemetry(BaseModel):
    trace_id: str
    prompt_version: str
    latency_ms: float
    cost_usd: float
    can_answer: bool
    citation_coverage: float   # 1.0 if all claims have verified citations, 0.0 otherwise
    is_error: bool

def record_telemetry(entry: RequestTelemetry) -> None:
    Path(METRICS_LOG).parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_LOG, "a", encoding="utf-8") as f:
        f.write(entry.model_dump_json() + "\n")

def summarize_sre_metrics(log_path: str = METRICS_LOG) -> dict:
    records: List[RequestTelemetry] = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(RequestTelemetry(**json.loads(line)))

    latencies = [r.latency_ms for r in records]
    costs = [r.cost_usd for r in records]
    answered = [r for r in records if r.can_answer and not r.is_error]
    failures = [r for r in records if r.is_error or not r.can_answer]

    return {
        "total_requests": len(records),
        "latency_p50_ms": round(float(np.percentile(latencies, 50)), 2),
        "latency_p95_ms": round(float(np.percentile(latencies, 95)), 2),
        "avg_cost_per_request_usd": round(float(np.mean(costs)), 6),
        "citation_coverage_pct": round(
            100.0 * sum(r.citation_coverage for r in answered) / max(len(answered), 1), 2
        ),
        "failure_or_abstention_rate_pct": round(100.0 * len(failures) / max(len(records), 1), 2),
    }
