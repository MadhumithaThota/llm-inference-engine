from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from server.schemas import GenerateRequest, GenerateResponse

from engine.scheduler import scheduler
import engine.worker
from engine.request import GenerationRequest
from engine.output_handler import (
    BufferedOutputHandler,
    StreamingOutputHandler,
)
from engine.kv_cache import KVCache

app = FastAPI(
    title="LLM Inference Engine",
    version="0.1.0",
)

@app.get("/")
def health():
    return {"status": "ok"}



@app.post("/generate")
def generate_text(request: GenerateRequest):

    if request.stream:
        handler = StreamingOutputHandler()
    else:
        handler = BufferedOutputHandler()

    generation_request = GenerationRequest(
        prompt=request.prompt,
        max_new_tokens=request.max_new_tokens,
        max_context_length=request.max_context_length,
        stream=request.stream,
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
        repetition_penalty=request.repetition_penalty,
        stop_sequences=request.stop_sequences,
    )

    generation_request.output_handler = handler
    generation_request.kv_cache = KVCache()

    scheduler.submit(generation_request)

    if request.stream:
        return StreamingResponse(
            handler.generator(),
            media_type="text/plain",
        )

    try:
        result = generation_request.future.result()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return GenerateResponse(**result)
