import json
from typing import Any
from time import time
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from server.schemas import GenerateRequest, GenerateResponse
from server.openai_compat import (
    OpenAIChatCompletionRequest,
    OpenAICompletionRequest,
    build_chat_completion_response,
    build_chat_prompt,
    build_chat_stream_chunk,
    build_completion_response,
    build_completion_stream_chunk,
    build_model_list,
    normalize_stop_sequences,
)
from engine.context_window import validate_prompt_within_context_limit, resolve_max_context_length
from engine.model_loader import MODEL_NAME, load_model, load_tokenizer

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


def _create_generation_request(
    *,
    prompt: str,
    max_new_tokens: int,
    stream: bool,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    stop_sequences: list[str],
    max_context_length: int | None = None,
):
    generation_request = GenerationRequest(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        max_context_length=max_context_length,
        stream=stream,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        stop_sequences=stop_sequences,
    )

    if stream:
        handler = StreamingOutputHandler()
    else:
        handler = BufferedOutputHandler()

    generation_request.output_handler = handler
    generation_request.kv_cache = KVCache()
    scheduler.submit(generation_request)

    return generation_request, handler


def _resolve_openai_max_tokens(
    max_tokens: int | None,
    max_completion_tokens: int | None = None,
) -> int:
    if isinstance(max_completion_tokens, int) and max_completion_tokens > 0:
        return max_completion_tokens

    if isinstance(max_tokens, int) and max_tokens > 0:
        return max_tokens

    return 100


def _validate_prompt_before_queue(prompt: str, max_context_length: int | None = None):
    tokenizer = load_tokenizer()
    model = load_model()
    resolved_limit = resolve_max_context_length(model, tokenizer, max_context_length)
    prompt_tokens = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
    validate_prompt_within_context_limit(prompt_tokens, resolved_limit)


def _to_sse(data: dict[str, Any]) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/v1/models")
def list_models():
    return build_model_list()


@app.get("/v1/models/{model_id}")
def retrieve_model(model_id: str):
    if model_id != MODEL_NAME:
        raise HTTPException(status_code=404, detail="Model not found")

    models = build_model_list()["data"]
    return models[0]



@app.post("/generate")
def generate_text(request: GenerateRequest):
    _validate_prompt_before_queue(request.prompt, request.max_context_length)

    generation_request, handler = _create_generation_request(
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


@app.post("/v1/chat/completions")
def create_chat_completion(request: OpenAIChatCompletionRequest):
    if request.n != 1:
        raise HTTPException(status_code=400, detail="Only n=1 is supported")

    stop_sequences = normalize_stop_sequences(request.stop, request.stop_sequences)
    prompt = build_chat_prompt(request.messages)
    max_new_tokens = _resolve_openai_max_tokens(
        request.max_tokens,
        request.max_completion_tokens,
    )
    _validate_prompt_before_queue(prompt, None)
    completion_id = f"chatcmpl-{uuid4().hex}"
    created = int(time())

    generation_request, handler = _create_generation_request(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        max_context_length=None,
        stream=request.stream,
        temperature=request.temperature,
        top_k=0,
        top_p=request.top_p,
        repetition_penalty=1.0,
        stop_sequences=stop_sequences,
    )

    if request.stream:
        def stream_events():
            yield _to_sse(
                build_chat_stream_chunk(
                    model=request.model,
                    completion_id=completion_id,
                    created=created,
                    content=None,
                    include_role=True,
                )
            )

            for text in handler.generator():
                yield _to_sse(
                    build_chat_stream_chunk(
                        model=request.model,
                        completion_id=completion_id,
                        created=created,
                        content=text,
                    )
                )

            try:
                result = generation_request.future.result()
            except Exception as exc:
                yield _to_sse(
                    {
                        "error": {
                            "message": str(exc),
                            "type": "invalid_request_error",
                        }
                    }
                )
                yield "data: [DONE]\n\n"
                return

                yield _to_sse(
                    build_chat_stream_chunk(
                        model=request.model,
                        completion_id=completion_id,
                        created=created,
                        content=None,
                        finish_reason=result["finish_reason"],
                    )
                )
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            stream_events(),
            media_type="text/event-stream",
        )

    try:
        result = generation_request.future.result()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return build_chat_completion_response(
        content=result["response"],
        model=request.model,
        metrics=result["metrics"],
        finish_reason=result["finish_reason"],
    )


@app.post("/v1/completions")
def create_completion(request: OpenAICompletionRequest):
    if request.n != 1:
        raise HTTPException(status_code=400, detail="Only n=1 is supported")

    stop_sequences = normalize_stop_sequences(request.stop, request.stop_sequences)
    max_new_tokens = _resolve_openai_max_tokens(request.max_tokens)
    _validate_prompt_before_queue(request.prompt, None)
    completion_id = f"cmpl-{uuid4().hex}"
    created = int(time())

    generation_request, handler = _create_generation_request(
        prompt=request.prompt,
        max_new_tokens=max_new_tokens,
        max_context_length=None,
        stream=request.stream,
        temperature=request.temperature,
        top_k=0,
        top_p=request.top_p,
        repetition_penalty=1.0,
        stop_sequences=stop_sequences,
    )

    if request.stream:
        def stream_events():
            for text in handler.generator():
                yield _to_sse(
                    build_completion_stream_chunk(
                        model=request.model,
                        completion_id=completion_id,
                        created=created,
                        text=text,
                    )
                )

            try:
                result = generation_request.future.result()
            except Exception as exc:
                yield _to_sse(
                    {
                        "error": {
                            "message": str(exc),
                            "type": "invalid_request_error",
                        }
                    }
                )
                yield "data: [DONE]\n\n"
                return

                yield _to_sse(
                    build_completion_stream_chunk(
                        model=request.model,
                        completion_id=completion_id,
                        created=created,
                        text="",
                        finish_reason=result["finish_reason"],
                    )
                )
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            stream_events(),
            media_type="text/event-stream",
        )

    try:
        result = generation_request.future.result()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return build_completion_response(
        text=result["response"],
        model=request.model,
        metrics=result["metrics"],
        finish_reason=result["finish_reason"],
    )
