from fastapi import FastAPI

from engine.generator import generate
from server.schemas import GenerateRequest, GenerateResponse

app = FastAPI(
    title="LLM Inference Engine",
    version="0.1.0",
)


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse)
def generate_text(request: GenerateRequest):
    response = generate(
        prompt=request.prompt,
        max_new_tokens=request.max_new_tokens,
    )

    return GenerateResponse(response=response)