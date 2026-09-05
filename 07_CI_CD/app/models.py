"""API request and response models."""

from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class AnswerResponse(BaseModel):
    answer: str
    model: str


class HealthResponse(BaseModel):
    status: str
    environment: str
