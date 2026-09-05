"""In-process functional tests for the FastAPI contract."""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.main import app, get_answer_service
from app.service import AnswerService


class FakeModel:
    def invoke(self, input: str) -> SimpleNamespace:
        return SimpleNamespace(content="CI checks a change before it is merged.")


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("APP_API_KEY", "test-api-key")
    service = AnswerService(model=FakeModel(), model_name="fake-model")
    app.dependency_overrides[get_answer_service] = lambda: service
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.mark.functional
def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.functional
def test_answer_endpoint(client: TestClient) -> None:
    response = client.post(
        "/answer",
        headers={"X-API-Key": "test-api-key"},
        json={"question": "What is CI?"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "CI checks a change before it is merged.",
        "model": "fake-model",
    }


@pytest.mark.functional
def test_answer_validates_short_questions(client: TestClient) -> None:
    response = client.post(
        "/answer",
        headers={"X-API-Key": "test-api-key"},
        json={"question": "?"},
    )

    assert response.status_code == 422


@pytest.mark.functional
def test_answer_requires_api_key(client: TestClient) -> None:
    response = client.post("/answer", json={"question": "What is CI?"})

    assert response.status_code == 401
