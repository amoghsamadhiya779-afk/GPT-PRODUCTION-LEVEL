# app/schemas.py
"""Validation schemas for the FastAPI inference service."""

from pydantic import BaseModel, Field


class GenerationRequest(BaseModel):
    """Schema for text generation request input validation."""
    prompt: str = Field(
        ..., 
        description="The starting text prompt to seed generation.",
        min_length=1,
        max_length=4000
    )
    max_new_tokens: int = Field(
        default=100, 
        ge=1, 
        le=500, 
        description="Maximum number of new tokens to generate."
    )
    temperature: float = Field(
        default=0.8, 
        ge=0.0, 
        le=2.0, 
        description="Sampling temperature. 0.0 means greedy decoding."
    )
    top_k: int = Field(
        default=50, 
        ge=1, 
        le=100, 
        description="Only sample from the top-k most probable tokens."
    )
    top_p: float | None = Field(
        default=None, 
        ge=0.0, 
        le=1.0, 
        description="Cumulative probability threshold for Top-P sampling."
    )
    repetition_penalty: float = Field(
        default=1.0, 
        ge=1.0, 
        le=2.0, 
        description="Penalty parameter applied to previously generated tokens."
    )
    use_cache: bool = Field(
        default=True,
        description="Whether to use key-value (KV) caching for fast inference."
    )
    web_search: bool = Field(
        default=False,
        description="Whether to fetch context from web search (RAG) to guide generation."
    )


class GenerationResponse(BaseModel):
    """Schema for text generation response metrics and text output."""
    prompt: str = Field(..., description="The original seed prompt.")
    generated_text: str = Field(..., description="The complete generated text (prompt + new tokens).")
    tokens_generated: int = Field(..., description="Number of new tokens generated.")
    time_taken_seconds: float = Field(..., description="Latency of the generation request in seconds.")
    tokens_per_second: float = Field(..., description="Generation speed in tokens/second.")
    sources: list[dict] | None = Field(
        default=None,
        description="List of search result sources used for RAG context."
    )


class FinetuneExample(BaseModel):
    instruction: str = Field(..., max_length=2000)
    response: str = Field(..., max_length=2000)

class FinetuneRequest(BaseModel):
    examples: list[FinetuneExample] = Field(..., min_length=1, max_length=500)
    adapter_name: str = Field(..., pattern=r"^[a-zA-Z0-9_-]+$")
    steps: int = Field(default=200, le=200, ge=1)
    lr: float = Field(default=1e-4, ge=1e-6, le=1e-2)

class FinetuneStatus(BaseModel):
    status: str
    step: int
    total_steps: int
    current_loss: float | None
    eta_seconds: float | None

class FeedbackRequest(BaseModel):
    prompt: str
    response: str
    rating: str = Field(..., pattern="^(up|down)$")
    correction: str | None = None
