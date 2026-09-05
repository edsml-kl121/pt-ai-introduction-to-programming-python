"""Deterministic unit tests for the answer service."""

from types import SimpleNamespace

import pytest

from app.service import AnswerService


class FakeModel:
    def __init__(self, content: str) -> None:
        self.content = content
        self.last_prompt = ""

    def invoke(self, input: str) -> SimpleNamespace:
        self.last_prompt = input
        return SimpleNamespace(content=self.content)


@pytest.mark.unit
def test_answer_returns_trimmed_model_content() -> None:
    model = FakeModel("  Containers package an app and its dependencies.  ")
    service = AnswerService(model=model, model_name="fake-model")

    answer = service.answer("What is a container?")

    assert answer == "Containers package an app and its dependencies."
    assert "What is a container?" in model.last_prompt


@pytest.mark.unit
def test_answer_rejects_empty_model_content() -> None:
    service = AnswerService(model=FakeModel("  "), model_name="fake-model")

    with pytest.raises(RuntimeError, match="empty answer"):
        service.answer("What is CI?")
