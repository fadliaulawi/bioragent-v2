"""python -m bioragent \"<question>\""""

from __future__ import annotations

import json
import sys

from pipeline import run_pipeline, trace_as_json_serializable


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python -m bioragent "<question>"', file=sys.stderr)
        sys.exit(1)
    q = sys.argv[1]
    answer, trace = run_pipeline(q)
    print(answer)
    print("\n--- trace (structured steps) ---", file=sys.stderr)
    print(json.dumps(trace_as_json_serializable(trace), indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
