"""Context-aware explanations and LCZ type comparisons."""

import re
from difflib import get_close_matches

from src.agent.goal_reasoner import recommend_for_goal
from src.layer_alterator_agent.matcher import SUPPORTED_URBAN_TYPES, URBAN_TYPE_ALIASES


TYPE_DESCRIPTIONS = {
    "Compact midrise": "a dense built form with closely spaced mid-rise buildings",
    "Compact low-rise": "a dense built form dominated by closely spaced low-rise buildings",
    "Open midrise": "mid-rise buildings arranged with more open space between them",
    "Open low-rise": "low-rise buildings arranged in a relatively open layout",
    "Large low-rise": "large-footprint low-rise buildings, often associated with industrial or commercial areas",
    "Dense trees": "land dominated by closely spaced tall trees and substantial tree canopy",
    "Scattered trees": "land with trees distributed more sparsely among other surface cover",
    "Low Plants": "land dominated by grass, herbaceous vegetation, or other low-growing plants",
    "Bare rock or paved": "predominantly impervious, paved, or exposed rock surface",
    "Bare Soil or sand": "predominantly exposed soil or sand with little vegetation",
    "Water": "surface dominated by open water",
}


def asks_for_context_explanation(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text.lower()).strip(" ?.")
    phrases = (
        "what do you mean", "what does that mean", "what you mean",
        "what your mean", "what your means", "can you explain that",
        "explain what you mean",
    )
    return any(phrase in normalized for phrase in phrases)


def mentioned_urban_types(text: str) -> list[str]:
    normalized = text.lower()
    found = []
    for alias, urban_type in sorted(
        URBAN_TYPE_ALIASES.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if (
            re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", normalized)
            and urban_type not in found
        ):
            found.append(urban_type)
    # Recover simple spelling errors such as "whater".
    canonical_words = {
        urban_type.lower(): urban_type for urban_type in SUPPORTED_URBAN_TYPES
        if " " not in urban_type
    }
    for word in re.findall(r"[a-z-]+", normalized):
        match = get_close_matches(word, canonical_words, n=1, cutoff=0.78)
        if match and canonical_words[match[0]] not in found:
            found.append(canonical_words[match[0]])
    return found


def asks_for_type_comparison(text: str) -> bool:
    normalized = text.lower()
    markers = ("which", "better", "compare", " versus ", " vs ", " or ")
    return len(mentioned_urban_types(text)) >= 2 and any(
        marker in normalized for marker in markers
    )


def asks_to_change_goal(text: str) -> bool:
    normalized = text.lower().strip()
    if re.search(r"(?:polygon|zone)\s*\d+", normalized):
        return False
    phrases = (
        "change the goal", "change my goal", "new goal", "different goal",
        "instead i want", "my goal is", "i now want",
        "simulate different scenarios", "population growth",
        "economic development", "change the simulation goal",
        "i want to", "we want to", "i would like to", "our goal is",
    )
    return any(phrase in normalized for phrase in phrases)


def asks_for_type_definition(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text.lower()).strip(" ?.!")
    return (
        len(mentioned_urban_types(text)) == 1
        and any(phrase in normalized for phrase in (
            "what is", "what are", "what does", "define", "definition",
            "tell me about", "describe",
        ))
    )


def asks_for_type_values(text: str) -> bool:
    normalized = text.lower()
    return (
        len(mentioned_urban_types(text)) == 1
        and any(phrase in normalized for phrase in (
            "predictor", "reference value", "values", "indicators",
            "data for", "attributes",
        ))
    )


def asks_for_data_source(text: str) -> bool:
    normalized = text.lower()
    return (
        len(mentioned_urban_types(text)) >= 1
        and any(phrase in normalized for phrase in (
            "where does", "where do", "come from", "data source",
            "source of", "provided by", "professor",
        ))
    )


def asks_about_pending_options(text: str) -> bool:
    normalized = text.lower()
    return any(phrase in normalized for phrase in (
        "what else", "other option", "alternatives", "help me decide",
        "not sure", "i am unsure", "i'm unsure", "which should i choose",
    ))


def answer_type_question(text: str, predictor_lookup) -> str | None:
    types = mentioned_urban_types(text)
    if len(types) != 1:
        return None
    urban_type = types[0]
    values = predictor_lookup(urban_type)
    value_lines = (
        f"TCH={values['TCH']:.3f}, IMD={values['IMD']:.3f}, "
        f"BH={values['BH']:.3f}, BSF={values['BSF']:.3f}, "
        f"SVF={values['SVF']:.3f}, F_G={values['F_G']:.3f}, "
        f"F_TV={values['F_TV']:.3f}, F_W={values['F_W']:.3f}"
    )
    if asks_for_data_source(text):
        return (
            f"The reference values for **{urban_type}** come from the "
            "professor-provided `ref_predictor_values_mean.csv` table. "
            "`F_W` is calculated as the remaining surface fraction when it is "
            "not present in the original table.\n\n"
            f"Current loaded values: {value_lines}.\n\n"
            "The type-to-planning correspondence is stored separately in "
            "`LCZ_urban_types_match.xlsx`."
        )
    if asks_for_type_values(text):
        return (
            f"Professor-provided reference values for **{urban_type}**:\n\n"
            f"- {value_lines}\n\n"
            "These are reference mean values for the LCZ class, not measurements "
            "taken from the uploaded polygon."
        )
    if asks_for_type_definition(text):
        return (
            f"**{urban_type}** is an LCZ urban/land-cover type describing "
            f"{TYPE_DESCRIPTIONS[urban_type]}.\n\n"
            f"In the loaded reference table: {value_lines}.\n\n"
            "It is a reference class, not a claim that a selected polygon already "
            "has these properties."
        )
    return None


def is_informational_question(text: str) -> bool:
    """Separate questions from commands that mutate the current plan."""
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    question_starts = (
        "what ", "what's ", "how ", "how can ", "how do ", "which ",
        "can you explain", "could you explain", "tell me about",
    )
    explicit_commands = (
        "my goal is", "i want", "we want", "change the goal",
        "set the goal", "polygon ", "zone ", "make all", "generate",
    )
    return (
        (normalized.endswith("?") or normalized.startswith(question_starts))
        and not normalized.startswith(explicit_commands)
    )


def is_social_or_meta_message(text: str) -> bool:
    """Allow conversational side messages without mutating a pending plan."""
    normalized = re.sub(r"\s+", " ", text.lower()).strip(" .!?")
    phrases = (
        "hello", "hi", "hey", "thank you", "thanks", "who are you",
        "what can you do", "help me", "let's discuss", "can we discuss",
        "你好", "谢谢", "你是谁", "你能做什么", "帮帮我", "我们讨论",
    )
    return any(
        normalized == phrase
        or (
            normalized.startswith(phrase)
            and len(normalized) > len(phrase)
            and not normalized[len(phrase)].isalnum()
        )
        for phrase in phrases
    )


def answer_supported_goal_question(text: str) -> str | None:
    """Give a data-backed answer without changing the saved simulation goal."""
    recommendation = recommend_for_goal(text)
    if recommendation.goal == "Unclear":
        return None
    direction = (
        "greater heat retention"
        if recommendation.goal == "Urban heat increase"
        else recommendation.goal.lower()
    )
    return (
        f"For **{direction}**, the professor-provided predictor table ranks these "
        f"LCZ types highest: **{', '.join(recommendation.ranked_types[:3])}**.\n\n"
        f"{recommendation.explanation}\n\n"
        "This answers your question only; it does **not** replace the current "
        "simulation goal or change any polygon. The ranking indicates LCZ "
        "characteristics, not a measured temperature outcome."
    )


def explain_goal_context(goal: str) -> str:
    if not goal:
        return (
            "I do not have a confirmed simulation goal yet. Describe the outcome "
            "you want, such as reducing heat, increasing vegetation, or managing water."
        )
    recommendation = recommend_for_goal(goal)
    if not recommendation.ranked_types:
        return (
            f"Your current goal is: **{goal}**.\n\n"
            "I could not map it to one of the supported goal profiles yet. "
            "Please describe the desired environmental or spatial outcome more specifically."
        )
    return (
        f"By that, I mean I interpreted your goal — **{goal}** — as "
        f"**{recommendation.goal}**.\n\n"
        f"{recommendation.explanation}\n\n"
        "The current leading reference-table options are: "
        + ", ".join(recommendation.ranked_types[:3])
        + ". This is a suitability ranking, not proof of actual temperature reduction."
    )


def compare_urban_types(text: str, goal: str, predictor_lookup) -> str | None:
    types = mentioned_urban_types(text)
    if len(types) < 2:
        return None
    left, right = types[:2]
    objective_markers = (
        "heat", "cool", "temperature", "flood", "drainage",
        "biodiversity", "ventilation", "air flow",
    )
    question_goal = text if any(marker in text.lower() for marker in objective_markers) else ""
    recommendation = (
        recommend_for_goal(question_goal)
        if question_goal
        else recommend_for_goal(goal) if goal else None
    )
    scores = recommendation.scores if recommendation else {}
    left_values, right_values = predictor_lookup(left), predictor_lookup(right)
    left_score, right_score = scores.get(left), scores.get(right)
    if left_score is None or right_score is None:
        verdict = (
            "I need a comparison objective before ranking these options "
            "(for example cooling, drainage, or greening)."
        )
    elif left_score > right_score:
        verdict = f"**{left}** is the stronger reference-table match for **{recommendation.goal}**."
    elif right_score > left_score:
        verdict = f"**{right}** is the stronger reference-table match for **{recommendation.goal}**."
    else:
        verdict = "The two options receive the same score for the current goal."
    score_line = (
        f"- {left}: score {left_score:.3f}\n- {right}: score {right_score:.3f}"
        if left_score is not None and right_score is not None else ""
    )
    return (
        f"{verdict}\n\n"
        f"{score_line}\n\n"
        f"Reference values:\n"
        f"- {left}: F_TV={left_values['F_TV']:.3f}, "
        f"F_G={left_values['F_G']:.3f}, F_W={left_values['F_W']:.3f}, "
        f"IMD={left_values['IMD']:.3f}\n"
        f"- {right}: F_TV={right_values['F_TV']:.3f}, "
        f"F_G={right_values['F_G']:.3f}, F_W={right_values['F_W']:.3f}, "
        f"IMD={right_values['IMD']:.3f}\n\n"
        f"{recommendation.explanation if recommendation else ''}\n\n"
        "The predictor values come from the professor-provided reference table; "
        "the suitability weights are system-defined heuristics. This comparison "
        "does not replace an LST or meteorological simulation."
    )
