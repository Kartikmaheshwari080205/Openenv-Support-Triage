from __future__ import annotations

import argparse
import json

from openenv_support_triage.baseline import run_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OpenEnv support triage baseline with OpenAI API")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model name to evaluate")
    args = parser.parse_args()

    result = run_baseline(model=args.model)
    print(json.dumps(result.model_dump(), indent=2))


if __name__ == "__main__":
    main()
