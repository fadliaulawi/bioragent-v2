"""Token usage and USD cost from chat completion responses (OpenRouter returns cost in usage)."""

from __future__ import annotations

from typing import Any, Mapping

# Fallback when the API does not return usage.cost (non–OpenRouter or older proxies).
_MODEL_RATES_PER_MILLION_USD: list[tuple[str, tuple[float, float]]] = [
    ("gpt-4o-mini", (0.15, 0.60)),
    ("gpt-4o", (2.50, 10.00)),
    ("gpt-4-turbo", (10.00, 30.00)),
    ("gpt-3.5-turbo", (0.50, 1.50)),
    ("gpt-4", (30.00, 60.00)),
    ("o3-mini", (1.10, 4.40)),
    ("o1-mini", (3.00, 12.00)),
    ("o1", (15.00, 60.00)),
]


def _usage_as_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, Mapping):
        return dict(usage)
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
        "cost": getattr(usage, "cost", None),
    }


def extract_usage_fields(resp: Any, model_fallback: str = "") -> dict[str, Any]:
    """Normalize usage + cost. OpenRouter sets usage.cost (USD); else estimate from tokens."""
    raw = _usage_as_dict(getattr(resp, "usage", None))
    pt = int(raw.get("prompt_tokens") or 0)
    ct = int(raw.get("completion_tokens") or 0)
    tt = raw.get("total_tokens")
    if tt is None:
        tt = pt + ct if (pt or ct) else None
    else:
        tt = int(tt)

    model = getattr(resp, "model", None) or model_fallback or ""

    # OpenRouter: usage.cost is the charged USD amount for this completion.
    direct = raw.get("cost")
    cost: float | None
    cost_source: str
    if direct is not None:
        try:
            cost = float(direct)
            cost_source = "api"
        except (TypeError, ValueError):
            cost = estimate_cost_usd(model, pt, ct)
            cost_source = "estimated" if cost is not None else "unknown"
    else:
        cost = estimate_cost_usd(model, pt, ct)
        cost_source = "estimated" if cost is not None else "unknown"

    extras = {
        k: v
        for k, v in raw.items()
        if k
        not in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "cost",
        )
    }

    cost_rounded = round(cost, 8) if cost is not None else None
    out: dict[str, Any] = {
        "cost_usd": cost_rounded,
        "cost_source": cost_source,
    }
    # Omit bulky usage when we already report cost (trace stays small).
    if cost_rounded is None:
        out["usage"] = {
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": tt,
            **extras,
        }
    return out


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    rates = _match_rates(model)
    if rates is None:
        return None
    inp_m, out_m = rates
    return (prompt_tokens / 1_000_000.0) * inp_m + (completion_tokens / 1_000_000.0) * out_m


def _match_rates(model: str) -> tuple[float, float] | None:
    ml = (model or "").lower().strip()
    if not ml:
        return None
    # Strip provider prefix for rough matching (e.g. openai/gpt-4o-mini).
    if "/" in ml:
        ml = ml.split("/", 1)[1]
    ordered = sorted(_MODEL_RATES_PER_MILLION_USD, key=lambda x: len(x[0]), reverse=True)
    for prefix, rates in ordered:
        if ml.startswith(prefix.lower()):
            return rates
    return None
