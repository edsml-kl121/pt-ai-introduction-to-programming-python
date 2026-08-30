"""Build deterministic and LLM-based custom evaluators."""

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from model import get_chat_model

load_dotenv()


class BlocklistEvaluator:
    def __init__(self, blocklist: list[str]):
        self.blocklist = blocklist

    def evaluate(self, response: str) -> dict:
        matches = [
            word for word in self.blocklist if word.lower() in response.lower()
        ]
        return {
            "blocklist_pass": not matches,
            "blocked_words": matches,
        }


def response_quality(response: str) -> dict:
    words = response.split()
    return {
        "word_count": len(words),
        "has_terminal_punctuation": response.endswith((".", "!", "?")),
        "starts_capitalized": bool(response) and response[0].isupper(),
    }


class FriendlinessScore(BaseModel):
    score: int = Field(ge=1, le=5)
    reasoning: str


friendliness_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "Score the response's friendliness from 1 to 5. A score of 1 is "
                "rude or dismissive, 3 is neutral, and 5 is warm, respectful, and "
                "helpful. Judge tone, not factual correctness."
            ),
        ),
        ("human", "Question: {question}\n\nResponse: {response}"),
    ]
)
friendliness_evaluator = (
    friendliness_prompt
    | get_chat_model().with_structured_output(FriendlinessScore)
)


def evaluate_friendliness(question: str, response: str) -> FriendlinessScore:
    return friendliness_evaluator.invoke(
        {"question": question, "response": response}
    )


def main() -> None:
    blocklist = BlocklistEvaluator(["todo", "placeholder", "lorem ipsum"])
    samples = [
        "Python is a programming language. Great question!",
        "todo: replace this placeholder response",
        "Look it up yourself.",
    ]

    for response in samples:
        print(f"\nResponse: {response}")
        print(f"Rules: {response_quality(response)}")
        print(f"Blocklist: {blocklist.evaluate(response)}")
        friendliness = evaluate_friendliness("What is Python?", response)
        print(f"Friendliness: {friendliness.model_dump()}")


if __name__ == "__main__":
    main()
