import json
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

ModelName = Literal["english", "multilingual", "typed-decisions"]

MAX_QUESTIONS = 128
MAX_STATE_BYTES = 250_000  # serialized JSON size of `state`, ~250 KB


class PredictRequest(BaseModel):
    state: dict[str, Any] = Field(
        ...,
        description=(
            "The state to evaluate, e.g. {\"subject\": \"...\", \"body\": \"...\"} "
            "or any JSON document."
        ),
    )
    questions: dict[str, Any] = Field(
        ...,
        description=(
            "Typed questions to answer. Each key maps to a question spec with "
            "type (choice | score | noul), instructions and, for choice/score, criteria."
        ),
    )
    model: Optional[ModelName] = Field(
        None,
        description=(
            "Optional checkpoint override: english | multilingual | typed-decisions. "
            "When omitted the Router picks the checkpoint automatically."
        ),
    )

    @field_validator("questions")
    @classmethod
    def _cap_question_count(cls, v: dict[str, Any]) -> dict[str, Any]:
        if len(v) > MAX_QUESTIONS:
            raise ValueError(
                f"Too many questions: {len(v)} (max {MAX_QUESTIONS})"
            )
        return v

    @model_validator(mode="after")
    def _cap_state_size(self) -> "PredictRequest":
        size = len(json.dumps(self.state))
        if size > MAX_STATE_BYTES:
            raise ValueError(
                f"State too large: {size} bytes (max {MAX_STATE_BYTES})"
            )
        return self


class HealthResponse(BaseModel):
    status: str
    device: str
    gpu: Optional[str] = None
    vram: Optional[dict[str, int]] = None
    checkpoints_resident: list[str]
    max_loaded: int
    laya_version: str
