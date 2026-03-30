from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

from dotenv import load_dotenv

from openenv_support_triage.baseline import run_baseline


def infer(model: str | None = None) -> Dict[str, Any]:
    result = run_baseline(model=model)
    return result.model_dump()


def main() -> None:
    load_dotenv()
    default_model = os.getenv("MODEL_NAME", "gpt-4o-mini")

    parser = argparse.ArgumentParser(description="Run baseline inference for OpenEnv support triage")
    parser.add_argument("--model", default=default_model, help="Model name to evaluate")
    args = parser.parse_args()

    payload = infer(model=args.model)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
