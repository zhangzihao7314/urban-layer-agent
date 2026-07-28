"""Explicit fallbacks for model, reference-data, and GIS failures."""

import json
from pathlib import Path
from typing import Callable

from src.agent.intent_router import route_intent


def safe_llm_json(raw: str, fallback_text: str) -> dict:
    try:
        start, end = raw.find("{"), raw.rfind("}") + 1
        if start < 0 or end <= start:
            raise ValueError("No JSON object found.")
        value = json.loads(raw[start:end])
        if not isinstance(value, dict):
            raise ValueError("LLM output is not a JSON object.")
        return value
    except (ValueError, TypeError, json.JSONDecodeError):
        decision = route_intent(fallback_text)
        return {
            "intent": decision.intent.value,
            "polygon_id": decision.polygon_id,
            "target_description": decision.target_description,
            "confidence": decision.confidence,
            "fallback_used": True,
        }


def require_reference_files(*paths: str | Path) -> None:
    missing = [str(Path(path)) for path in paths if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(
            "Authoritative reference data is unavailable; numerical generation is blocked: "
            + ", ".join(missing)
        )


def safe_domain_action(action: Callable, *args, **kwargs) -> dict:
    try:
        return {"success": True, "result": action(*args, **kwargs), "error": None}
    except Exception as exc:
        return {
            "success": False,
            "result": None,
            "error": str(exc),
            "message": "The action failed and was not reported as completed.",
        }
