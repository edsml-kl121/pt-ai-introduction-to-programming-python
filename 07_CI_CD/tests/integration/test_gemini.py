"""Live provider integration test."""

import os

import pytest

from app.service import build_answer_service


@pytest.mark.integration
def test_gemini_returns_a_nonempty_answer() -> None:
    if not os.getenv("GOOGLE_API_KEY"):
        pytest.fail("GOOGLE_API_KEY is required for the integration test.")

    service = build_answer_service()
    answer = service.answer("What is continuous integration?")

    assert len(answer) >= 20
    assert len(answer.split()) <= 120
