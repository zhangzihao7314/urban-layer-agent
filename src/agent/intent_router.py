import json
import re
from typing import Optional

from src.agent.schemas import Intent, IntentDecision


def rule_based_intent(user_text: str) -> IntentDecision:
    text = user_text.lower().strip()
    if text in {"confirm", "yes", "accept", "确认", "同意"}:
        return IntentDecision(Intent.CONFIRM, confidence=1.0)
    if text in {"revise", "change", "no", "修改", "不确认"}:
        return IntentDecision(Intent.REVISE, confidence=1.0)
    if any(k in text for k in ["generate", "create files", "生成文件"]):
        return IntentDecision(Intent.GENERATE, confidence=0.95)
    if any(k in text for k in ["why", "explain", "reason", "为什么", "解释"]):
        return IntentDecision(Intent.ASK_EXPLANATION, confidence=0.9)
    if any(k in text for k in ["all polygons", "all areas", "全部区域", "所有区域"]):
        return IntentDecision(Intent.APPLY_ALL, target_description=_remove_all_phrases(user_text), confidence=0.9)

    match = re.search(r"(?:polygon|zone|区域)\s*(\d+)", text, re.IGNORECASE)
    if match:
        polygon_id = int(match.group(1))
        description = re.sub(r"(?:polygon|zone|区域)\s*\d+", "", user_text, flags=re.IGNORECASE)
        description = re.sub(
            r"\b(should become|change to|become|set to)\b|改成|变成",
            "",
            description,
            flags=re.IGNORECASE,
        ).strip(" :，。")
        return IntentDecision(Intent.SET_POLYGON, polygon_id, description or None, 0.9)

    if any(k in text for k in ["upload", "geojson", "gpkg", "shapefile", "上传"]):
        return IntentDecision(Intent.UPLOAD_HELP, confidence=0.8)
    if any(k in text for k in ["heat", "cool", "green", "vegetation", "water", "热岛", "降温", "绿化"]):
        return IntentDecision(Intent.SET_GOAL, confidence=0.75)
    return IntentDecision(Intent.CHAT, confidence=0.5)


def parse_llm_intent(raw: str) -> Optional[IntentDecision]:
    try:
        start, end = raw.find("{"), raw.rfind("}") + 1
        data = json.loads(raw[start:end])
        return IntentDecision(
            intent=Intent(data["intent"]),
            polygon_id=data.get("polygon_id"),
            target_description=data.get("target_description"),
            confidence=float(data.get("confidence", 0.0)),
        )
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def route_intent(user_text: str, llm_output: Optional[str] = None) -> IntentDecision:
    rule_decision = rule_based_intent(user_text)
    llm_decision = parse_llm_intent(llm_output) if llm_output else None
    if not llm_decision or llm_decision.confidence < 0.6:
        return rule_decision
    if llm_decision.intent == Intent.SET_POLYGON and llm_decision.polygon_id is None:
        return rule_decision
    if rule_decision.intent in {Intent.GENERATE, Intent.CONFIRM, Intent.REVISE}:
        return rule_decision
    return llm_decision


def _remove_all_phrases(text: str) -> str:
    for phrase in ["make all polygons", "change all polygons to", "all polygons", "all areas", "全部区域", "所有区域"]:
        text = re.sub(re.escape(phrase), "", text, flags=re.IGNORECASE)
    return text.strip(" :，。")

