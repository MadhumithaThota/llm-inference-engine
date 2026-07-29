from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt")
    max_new_tokens: int = Field(default=100, ge=1, le=512)


class GenerateResponse(BaseModel):
    response: str