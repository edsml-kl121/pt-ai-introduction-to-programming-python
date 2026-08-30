"""Score GenAI response quality with OpenEvals and Gemini."""

import json

from dotenv import load_dotenv
from openevals.llm import create_llm_as_judge
from openevals.prompts import (
    ANSWER_RELEVANCE_PROMPT,
    CONCISENESS_PROMPT,
    CORRECTNESS_PROMPT,
    RAG_GROUNDEDNESS_PROMPT,
    RAG_HELPFULNESS_PROMPT,
    RAG_RETRIEVAL_RELEVANCE_PROMPT,
)

from model import get_chat_model

load_dotenv()

judge = get_chat_model()

evaluators = {
    "correctness": create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT,
        judge=judge,
        feedback_key="correctness",
        continuous=True,
    ),
    "answer_relevance": create_llm_as_judge(
        prompt=ANSWER_RELEVANCE_PROMPT,
        judge=judge,
        feedback_key="answer_relevance",
        continuous=True,
    ),
    "groundedness": create_llm_as_judge(
        prompt=RAG_GROUNDEDNESS_PROMPT,
        judge=judge,
        feedback_key="groundedness",
        continuous=True,
    ),
    "helpfulness": create_llm_as_judge(
        prompt=RAG_HELPFULNESS_PROMPT,
        judge=judge,
        feedback_key="helpfulness",
        continuous=True,
    ),
    "retrieval_relevance": create_llm_as_judge(
        prompt=RAG_RETRIEVAL_RELEVANCE_PROMPT,
        judge=judge,
        feedback_key="retrieval_relevance",
        continuous=True,
    ),
    "conciseness": create_llm_as_judge(
        prompt=CONCISENESS_PROMPT,
        judge=judge,
        feedback_key="conciseness",
        continuous=True,
    ),
}


def evaluate_quality(
    question: str,
    context: str,
    reference: str,
    response: str,
) -> dict:
    inputs = {"question": question}
    outputs = {"answer": response}
    reference_outputs = {"answer": reference}

    return {
        "correctness": evaluators["correctness"](
            inputs=inputs,
            outputs=outputs,
            reference_outputs=reference_outputs,
        ),
        "answer_relevance": evaluators["answer_relevance"](
            inputs=inputs,
            outputs=outputs,
        ),
        "groundedness": evaluators["groundedness"](
            outputs=outputs,
            context=context,
        ),
        "helpfulness": evaluators["helpfulness"](
            inputs=inputs,
            outputs=outputs,
        ),
        "retrieval_relevance": evaluators["retrieval_relevance"](
            inputs=inputs,
            context=context,
        ),
        "conciseness": evaluators["conciseness"](
            inputs=inputs,
            outputs=outputs,
        ),
    }


def main() -> None:
    result = evaluate_quality(
        question="What are the benefits of regular exercise?",
        context=(
            "Exercise supports cardiovascular health, mental well-being, and "
            "weight management. The WHO recommends 150 minutes of moderate "
            "exercise per week."
        ),
        reference=(
            "Regular exercise improves heart health, mood, and weight management."
        ),
        response=(
            "Regular exercise improves cardiovascular health, boosts mood, and "
            "helps maintain a healthy weight."
        ),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
