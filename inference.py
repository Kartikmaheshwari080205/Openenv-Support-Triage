from __future__ import annotations

import argparse
import json
from typing import Any, Dict

from openenv_support_triage.baseline import run_baseline


def infer(model: str = "gpt-4o-mini") -> Dict[str, Any]:
    result = run_baseline(model=model)
    return result.model_dump()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline inference for OpenEnv support triage")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model name to evaluate")
    args = parser.parse_args()

    payload = infer(model=args.model)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
