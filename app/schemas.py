from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

ModelName = Literal["english", "multilingual", "typed-decisions"]


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


class HealthResponse(BaseModel):
    status: str
    device: str
    gpu: Optional[str] = None
    vram: Optional[dict[str, int]] = None
    checkpoints_resident: list[str]
    max_loaded: int
    laya_version: str
