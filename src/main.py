from __future__ import annotations

import json
import sys
from datetime import datetime

from pipeline import run_pipeline, trace_as_json_serializable


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python src/main.py \"<question>\"", file=sys.stderr)
        sys.exit(1)
    q = sys.argv[1]
    answer, trace = run_pipeline(q)
    print(answer)

    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"logs/trace_{tag}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trace_as_json_serializable(trace), f, indent=2)
    print(f"Trace written to {path}")

if __name__ == "__main__":
    main()
