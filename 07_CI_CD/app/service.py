"""Gemini-backed answer service with a small injectable interface."""

import os
from typing import Any, Protocol

from langchain_google_genai import ChatGoogleGenerativeAI


class ChatModel(Protocol):
    def invoke(self, input: str) -> Any:
        """Return a chat-model response for the supplied prompt."""


class AnswerService:
    def __init__(self, model: ChatModel, model_name: str) -> None:
        self.model = model
        self.model_name = model_name

    def answer(self, question: str) -> str:
        prompt = f"""
You are a concise study assistant for beginner software engineers.
Answer the question in no more than three sentences. Use plain language and
do not invent details.

Question: {question}
""".strip()
        response = self.model.invoke(prompt)
        content = response.content
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("The model returned an empty answer.")
        return content.strip()


def build_answer_service() -> AnswerService:
    """Build the real Gemini service used outside deterministic tests."""
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY is required to call Gemini.")

    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    model = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0,
        max_retries=2,
    )
    return AnswerService(model=model, model_name=model_name)
