from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt")
    max_new_tokens: int = Field(default=100, ge=1, le=512)
    stream: bool = True
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0


class GenerateResponse(BaseModel):
    response: str