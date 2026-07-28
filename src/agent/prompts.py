INTENT_ROUTER_SYSTEM_PROMPT = """
Classify the user message for an urban layer preparation assistant.
Return JSON only. Never invent predictor values.
Allowed intents: chat, set_goal, apply_all, set_polygon, ask_explanation,
generate, confirm, revise, upload_help.
""".strip()


def build_intent_prompt(user_text: str, polygon_ids: list[int]) -> str:
    return f"""{INTENT_ROUTER_SYSTEM_PROMPT}

Known polygon IDs: {polygon_ids}
User message: {user_text}

Return:
{{"intent":"chat","polygon_id":null,"target_description":null,"confidence":0.0}}
"""

