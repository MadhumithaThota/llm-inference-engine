from fastapi import FastAPI

from server.schemas import GenerateRequest, GenerateResponse
from fastapi.responses import StreamingResponse
from engine.scheduler import scheduler
import engine.worker
from engine.request import GenerationRequest

app = FastAPI(
    title="LLM Inference Engine",
    version="0.1.0",
)

@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/generate")
def generate_text(request: GenerateRequest):

    generation_request = GenerationRequest(
        prompt=request.prompt,
        max_new_tokens=request.max_new_tokens,
        stream=request.stream,
    )

    future = scheduler.submit(generation_request)

    result = future.result()

    if request.stream:
        return StreamingResponse(
            result,
            media_type="text/plain",
        )

    return GenerateResponse(
        response=result,
    )
