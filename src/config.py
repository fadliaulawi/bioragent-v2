"""Environment-backed settings (OpenRouter-compatible Chat Completions API)."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

# OpenRouter uses the OpenAI-compatible endpoint; cost is returned in usage.cost (USD).
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class Settings:
    api_key: str | None
    base_url: str
    model_name: str
    planner_temperature: float = 0.0
    validation_temperature: float = 1.0


def get_settings() -> Settings:
    key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    base = (os.getenv("BASE_URL") or "").strip() or DEFAULT_OPENROUTER_BASE_URL
    model = os.getenv("MODEL_NAME") or "openai/gpt-4o-mini"
    return Settings(api_key=key, base_url=base, model_name=model)
