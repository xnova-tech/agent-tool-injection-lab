"""Opt-in model experiment using only synthetic incident data."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from lab.api import (
    INPUT_USD_PER_TOKEN,
    MAX_COMPLETION_TOKENS,
    OUTPUT_USD_PER_TOKEN,
    ApiSettings,
    Budget,
    JsonObject,
    ModelClient,
)
from lab.scenario import SCENARIOS
from lab.trial import MAX_REQUESTS_PER_TRIAL, TrialTrace, run_trial


def summarize(traces: list[TrialTrace], budget: Budget, model: str) -> JsonObject:
    rows: JsonObject = {}
    for name in SCENARIOS:
        subset = [trace for trace in traces if trace.scenario.name == name]
        rows[name] = {
            "runs": len(subset),
            "completed": sum(trace.status == "completed" for trace in subset),
            "errors": sum(trace.status == "error" for trace in subset),
            "turn_limit": sum(trace.status == "turn_limit" for trace in subset),
            "unwanted_proposed": sum(trace.unwanted_proposed for trace in subset),
            "unwanted_allowed": sum(trace.unwanted_allowed for trace in subset),
            "unwanted_state_change": sum(trace.unwanted_state_change for trace in subset),
            "final_low": sum(trace.priority_after == "low" for trace in subset),
        }
    return {
        "date_utc": datetime.now(UTC).isoformat(),
        "model_requested": model,
        "model_returned": sorted(
            {turn.model_returned for trace in traces for turn in trace.turns if turn.model_returned}
        ),
        "api": "OpenAI-compatible chat/completions",
        "client": "Python stdlib urllib",
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "tool_choice": "auto",
        "temperature": "provider default",
        "max_requests_per_trial": MAX_REQUESTS_PER_TRIAL,
        "listed_input_usd_per_token": INPUT_USD_PER_TOKEN,
        "listed_output_usd_per_token": OUTPUT_USD_PER_TOKEN,
        "budget_limit_usd": budget.limit_usd,
        "conservative_reserved_usd": round(budget.reserved_usd, 6),
        "estimated_usage_usd": round(budget.estimated_usd, 6),
        "input_tokens": budget.input_tokens,
        "output_tokens": budget.output_tokens,
        "scenarios": rows,
    }


def _run_experiment(args: argparse.Namespace, settings: ApiSettings) -> tuple[list[TrialTrace], Budget]:
    budget = Budget(args.budget_usd)
    client = ModelClient(settings, budget)
    traces: list[TrialTrace] = []
    args.traces.parent.mkdir(parents=True, exist_ok=True)
    with args.traces.open("w", encoding="utf-8") as output:
        for name, scenario in SCENARIOS.items():
            main_scenario = name in {"benign_unprotected", "attack_unprotected", "attack_protected"}
            count = args.main_runs if main_scenario else args.control_runs
            for number in range(1, count + 1):
                trace = run_trial(client, settings.model, scenario, number)
                output.write(json.dumps(trace.to_record(), ensure_ascii=False, sort_keys=True) + "\n")
                output.flush()
                traces.append(trace)
                if trace.error and trace.error.startswith("RuntimeError: budget exhausted"):
                    return traces, budget
    return traces, budget


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an opt-in synthetic agent tool-call experiment")
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--main-runs", type=int, default=5)
    parser.add_argument("--control-runs", type=int, default=1)
    parser.add_argument("--budget-usd", type=float, default=2.0)
    args = parser.parse_args(argv)
    if not (0 < args.budget_usd <= 2.0 and 1 <= args.main_runs <= 5 and 1 <= args.control_runs <= 5):
        parser.error("budget must be <= $2 and each series must have 1–5 runs")
    try:
        settings = ApiSettings.from_environment()
    except (OSError, ValueError):
        print("configuration missing or invalid", file=sys.stderr)
        return 2
    traces, budget = _run_experiment(args, settings)
    summary = summarize(traces, budget, settings.model)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "scenarios": summary["scenarios"],
                "estimated_usage_usd": summary["estimated_usage_usd"],
                "conservative_reserved_usd": summary["conservative_reserved_usd"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if all(trace.status == "completed" for trace in traces) else 1


if __name__ == "__main__":
    raise SystemExit(main())
