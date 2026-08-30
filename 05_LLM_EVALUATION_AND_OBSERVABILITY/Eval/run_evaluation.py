"""Run a LangChain application as a LangSmith batch evaluation."""

import re

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langsmith import Client
from langsmith.schemas import Example, Run

from create_dataset import DATASET_NAME
from model import get_chat_model
from traditional_metrics import token_f1_score

load_dotenv()

answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer only from the supplied context. Be concise and factually precise.",
        ),
        ("human", "Question: {question}\n\nContext: {context}"),
    ]
)
answer_chain = (
    answer_prompt
    | get_chat_model()
    | StrOutputParser()
)

judge_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You evaluate whether an assistant response directly and correctly "
                "answers a question. Reply with only PASS or FAIL."
            ),
        ),
        (
            "human",
            (
                "Question: {question}\n"
                "Reference answer: {reference_answer}\n"
                "Assistant response: {response}"
            ),
        ),
    ]
)
judge_chain = (
    judge_prompt
    | get_chat_model()
    | StrOutputParser()
)


def target(inputs: dict) -> dict:
    """Application under evaluation."""
    return {"answer": answer_chain.invoke(inputs)}


def _answer(run: Run, example: Example) -> tuple[str, str]:
    response = str((run.outputs or {}).get("answer", ""))
    reference = str((example.outputs or {}).get("answer", ""))
    return response, reference


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def exact_match(run: Run, example: Example) -> dict:
    response, reference = _answer(run, example)
    normalize = lambda value: " ".join(_tokens(value))
    return {"key": "exact_match", "score": int(normalize(response) == normalize(reference))}


def token_f1(run: Run, example: Example) -> dict:
    response, reference = _answer(run, example)
    return {"key": "token_f1", "score": token_f1_score(response, reference)}


def relevance(run: Run, example: Example) -> dict:
    response, reference = _answer(run, example)
    verdict = judge_chain.invoke(
        {
            "question": (example.inputs or {}).get("question", ""),
            "reference_answer": reference,
            "response": response,
        }
    )
    passed = verdict.strip().upper().startswith("PASS")
    return {"key": "relevance", "score": int(passed), "comment": verdict.strip()}


def main() -> None:
    client = Client()
    results = client.evaluate(
        target,
        data=DATASET_NAME,
        evaluators=[exact_match, token_f1, relevance],
        experiment_prefix="langchain-evaluation",
        max_concurrency=2,
    )
    print(results)


if __name__ == "__main__":
    main()
