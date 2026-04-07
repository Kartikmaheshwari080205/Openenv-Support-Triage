from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

from dotenv import load_dotenv

from openenv_support_triage.baseline import run_baseline


def _resolve_runtime_env() -> Dict[str, str | None]:
    return {
        "api_base_url": os.getenv("API_BASE_URL"),
        "api_key": os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY"),
        "model_name": os.getenv("MODEL_NAME", "gpt-4o-mini"),
    }


def infer(model: str | None = None) -> Dict[str, Any]:
    result = run_baseline(model=model)
    return result.model_dump()


def _print_structured_output(payload: Dict[str, Any]) -> None:
    model_name = str(payload.get("model", "unknown"))
    task_results = payload.get("task_results", [])
    average_score = payload.get("average_score", 0.0)

    print(f"[START] task=openenv-support-triage model={model_name}", flush=True)

    for index, task_result in enumerate(task_results, start=1):
        task_id = str(task_result.get("task_id", f"task-{index}"))
        score = task_result.get("score", 0.0)
        steps = task_result.get("steps", 0)
        done = str(task_result.get("done", False)).lower()
        print(
            f"[STEP] step={index} task={task_id} score={score} steps={steps} done={done}",
            flush=True,
        )

    print(
        f"[END] task=openenv-support-triage score={average_score} steps={len(task_results)}",
        flush=True,
    )


def main() -> None:
    load_dotenv()
    runtime_env = _resolve_runtime_env()
    default_model = str(runtime_env["model_name"] or "gpt-4o-mini")

    parser = argparse.ArgumentParser(description="Run baseline inference for OpenEnv support triage")
    parser.add_argument("--model", default=default_model, help="Model name to evaluate")
    args = parser.parse_args()

    payload = infer(model=args.model)
    _print_structured_output(payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
