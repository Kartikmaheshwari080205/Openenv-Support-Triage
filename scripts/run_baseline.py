from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openenv_support_triage.baseline import run_baseline


def main() -> None:
    load_dotenv()
    default_model = os.getenv("MODEL_NAME", "gpt-4o-mini")

    parser = argparse.ArgumentParser(description="Run OpenEnv support triage baseline with OpenAI API")
    parser.add_argument("--model", default=default_model, help="Model name to evaluate")
    args = parser.parse_args()

    result = run_baseline(model=args.model)
    print(json.dumps(result.model_dump(), indent=2))


if __name__ == "__main__":
    main()
