"""Live LLM-as-judge checks adapted from exercise 05."""

import os

import pytest
from langchain_google_genai import ChatGoogleGenerativeAI
from openevals.llm import create_llm_as_judge
from openevals.prompts import ANSWER_RELEVANCE_PROMPT, CORRECTNESS_PROMPT

from app.service import build_answer_service

QUESTION = "What does a Docker container package?"
REFERENCE = (
    "A Docker container packages an application with the runtime, libraries, "
    "and dependencies it needs to run consistently."
)


def score_value(result: dict) -> float:
    score = result.get("score")
    if not isinstance(score, (int, float)):
        raise AssertionError(f"Evaluator returned an invalid score: {result}")
    return float(score)


@pytest.mark.eval
def test_answer_is_correct_and_relevant() -> None:
    if not os.getenv("GOOGLE_API_KEY"):
        pytest.fail("GOOGLE_API_KEY is required for the LLM evaluation test.")

    service = build_answer_service()
    response = service.answer(QUESTION)
    judge = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        temperature=0,
    )
    correctness = create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT,
        judge=judge,
        feedback_key="correctness",
        continuous=True,
    )
    relevance = create_llm_as_judge(
        prompt=ANSWER_RELEVANCE_PROMPT,
        judge=judge,
        feedback_key="answer_relevance",
        continuous=True,
    )

    inputs = {"question": QUESTION}
    outputs = {"answer": response}
    correctness_result = correctness(
        inputs=inputs,
        outputs=outputs,
        reference_outputs={"answer": REFERENCE},
    )
    relevance_result = relevance(inputs=inputs, outputs=outputs)

    assert score_value(correctness_result) >= 0.7, correctness_result
    assert score_value(relevance_result) >= 0.7, relevance_result
