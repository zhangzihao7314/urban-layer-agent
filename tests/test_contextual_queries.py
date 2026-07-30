import pandas as pd

from src.agent.contextual_queries import (
    answer_supported_goal_question,
    answer_type_question,
    asks_about_pending_options,
    asks_for_data_source,
    asks_for_context_explanation,
    asks_for_type_definition,
    asks_for_type_values,
    asks_to_change_goal,
    asks_for_type_comparison,
    compare_urban_types,
    is_informational_question,
    is_social_or_meta_message,
    mentioned_urban_types,
)
from src.layer_alterator_agent.reference_loader import get_predictor_values, load_reference_table


def test_contextual_followup_and_typo_comparison_are_detected():
    assert asks_for_context_explanation("what your means")
    text = "whater or dense trees, which one is better for decrease city heat?"
    assert asks_for_type_comparison(text)
    assert mentioned_urban_types(text)[:2] == ["Dense trees", "Water"]


def test_goal_change_detection():
    assert asks_to_change_goal(
        "simulate different scenarios based on population growth"
    )
    assert asks_to_change_goal("I now want a different goal")
    assert not asks_to_change_goal("Polygon 7 should become a growth area")


def test_questions_are_not_commands_and_do_not_mutate_the_goal():
    assert is_informational_question("What is Water?")
    assert is_informational_question("How to increase the city heat")
    assert not is_informational_question("I want to increase the city heat")
    answer = answer_supported_goal_question("How to increase the city heat")
    assert "greater heat retention" in answer
    assert "does **not** replace" in answer


def test_type_definition_values_and_source_are_distinct_questions():
    frame = load_reference_table("data/reference/ref_predictor_values_mean.csv")
    lookup = lambda urban_type: get_predictor_values(frame, urban_type)

    definition_text = "What is Dense trees?"
    assert asks_for_type_definition(definition_text)
    definition = answer_type_question(definition_text, lookup)
    assert "LCZ urban/land-cover type" in definition
    assert "closely spaced tall trees" in definition

    values_text = "What are the predictor values of Dense trees?"
    assert asks_for_type_values(values_text)
    values = answer_type_question(values_text, lookup)
    assert "reference mean values" in values
    assert "not measurements taken from the uploaded polygon" in values

    source_text = "Where do the Water values come from?"
    assert asks_for_data_source(source_text)
    source = answer_type_question(source_text, lookup)
    assert "ref_predictor_values_mean.csv" in source
    assert "LCZ_urban_types_match.xlsx" in source


def test_comparison_uses_objective_in_question_before_saved_goal():
    frame = load_reference_table("data/reference/ref_predictor_values_mean.csv")
    answer = compare_urban_types(
        "Water or Dense trees, which is better for decreasing city heat?",
        "simulate population growth",
        lambda urban_type: get_predictor_values(frame, urban_type),
    )
    assert "Urban heat mitigation" in answer
    assert "Neither option" not in answer
    assert "**Dense trees** is the stronger" in answer


def test_pending_option_discussion_and_natural_goal_change_are_detected():
    assert asks_about_pending_options("I am not sure, what else can I choose?")
    assert asks_to_change_goal("I want to reduce flooding instead")


def test_type_aliases_do_not_match_inside_unrelated_words():
    assert mentioned_urban_types("Tell me about a broad planning strategy") == []
    assert mentioned_urban_types("Tell me about a road strategy") == ["Bare rock or paved"]


def test_social_messages_can_interrupt_a_pending_workflow_safely():
    assert is_social_or_meta_message("Thanks, can we discuss this first?")
    assert is_social_or_meta_message("你好，你能做什么？")
    assert not is_social_or_meta_message("Confirm")


def test_comparison_answers_without_creating_polygon_proposal():
    frame = load_reference_table("data/reference/ref_predictor_values_mean.csv")
    answer = compare_urban_types(
        "Water or Dense trees, which is better and why?",
        "I want to decrease city heat",
        lambda urban_type: get_predictor_values(frame, urban_type),
    )
    assert "Dense trees" in answer
    assert "Reference values" in answer
    assert "LST" in answer
