# BioRAGent v2

LangChain-free biomedical RAG pipeline with:

- query medical-domain gating,
- LLM-generated multi-step planning,
- direct biomedical tool/API retrieval,
- optional OpenAI tool-loop refinement,
- final validation + structured trace logging.

## Repository Layout

| Path | Purpose |
|------|---------|
| `src/` | Main pipeline code (`pipeline.py`, `planner.py`, `tools.py`, `llm.py`, `refine_loop.py`, `trace.py`) |
| `data/Multi_hop_Task/` | Multi-hop benchmark assets |
| `data/geneturing/` | Downloaded Geneturing subset CSVs |
| `data/download.py` | Script to download all Geneturing subsets |
| `trace_*.json` | Example pipeline run traces |

## Requirements

- Python 3.10+ recommended
- Network access to external biomedical APIs and model endpoint
- Install dependencies from `requirements.txt`

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Environment Variables

This project loads environment variables using `python-dotenv` (`load_dotenv()`), so you can provide them in a local `.env` file or your shell.

Required (at least one):

- `OPENROUTER_API_KEY` or `OPENAI_API_KEY`

Optional:

- `BASE_URL` (default: `https://openrouter.ai/api/v1`)
- `MODEL_NAME` (default: `openai/gpt-4o-mini`)
- `BIOONTOLOGY_API_KEY` (for BioOntology-backed lookups)
- `NCBI_API_KEY` (for higher-rate NCBI requests)
- `HTTP_USER_AGENT` (used by some helper requests)

## Run the Pipeline

Current CLI entrypoint in this repo:

```bash
python src/main.py "What genes are associated with Marfan syndrome?"
```

Behavior:

- prints final answer to stdout,
- writes a timestamped trace file like `trace_20260505_130710.json` in the project root.

## Use from Python

```python
from pipeline import run_pipeline, trace_as_json_serializable

answer, trace = run_pipeline("Your biomedical question")
trace_rows = trace_as_json_serializable(trace)
```

## Download Geneturing Data

```bash
python data/download.py
```

This writes one CSV per subset under `data/geneturing/`.

## Notes

- There is no `pyproject.toml` in the current snapshot, so editable install and `python -m bioragent` package execution are not documented here.
- Most functionality depends on external APIs; responses and latency can vary with service availability and rate limits.
