"""FastAPI entry point for the learning CI/CD service."""

import os
import secrets
from functools import lru_cache
from typing import Annotated

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, status

from app.models import AnswerResponse, HealthResponse, QuestionRequest
from app.service import AnswerService, build_answer_service

load_dotenv()

app = FastAPI(
    title="Learning CI/CD API",
    version="1.0.0",
)


@lru_cache
def get_answer_service() -> AnswerService:
    return build_answer_service()


def require_api_key(
    x_api_key: Annotated[str | None, Header()] = None,
) -> None:
    expected_key = os.getenv("APP_API_KEY")
    if not expected_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="APP_API_KEY is not configured.",
        )
    if not x_api_key or not secrets.compare_digest(x_api_key, expected_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        environment=os.getenv("APP_ENV", "local"),
    )


@app.post(
    "/answer",
    response_model=AnswerResponse,
    dependencies=[Depends(require_api_key)],
)
def answer(
    request: QuestionRequest,
    service: AnswerService = Depends(get_answer_service),
) -> AnswerResponse:
    return AnswerResponse(
        answer=service.answer(request.question),
        model=service.model_name,
    )
