from __future__ import annotations

import time
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from engine.model_loader import load_tokenizer, MODEL_NAME


class OpenAIChatMessage(BaseModel):
    role: str
    content: Any = None


class OpenAIChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    model: str
    messages: list[OpenAIChatMessage]
    stream: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 1
    stop: str | list[str] | None = None
    stop_sequences: list[str] | None = None
    max_tokens: int | None = Field(default=None, alias="max_tokens")
    max_completion_tokens: int | None = None


class OpenAICompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    model: str
    prompt: str
    stream: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 1
    stop: str | list[str] | None = None
    stop_sequences: list[str] | None = None
    max_tokens: int | None = Field(default=None, alias="max_tokens")


def _extract_text_from_content(content: Any) -> str:
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []

        for part in content:
            if not isinstance(part, dict):
                raise ValueError("Only text message content is supported")

            part_type = part.get("type")

            if part_type in {"text", "input_text"}:
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
                    continue

            raise ValueError("Only text message content is supported")

        return "".join(parts)

    raise ValueError("Only text message content is supported")


def normalize_stop_sequences(stop: str | list[str] | None, extra: list[str] | None = None) -> list[str]:
    sequences: list[str] = []

    if isinstance(stop, str) and stop:
        sequences.append(stop)
    elif isinstance(stop, list):
        sequences.extend(sequence for sequence in stop if isinstance(sequence, str) and sequence)

    if extra:
        sequences.extend(sequence for sequence in extra if sequence)

    deduped: list[str] = []
    for sequence in sequences:
        if sequence not in deduped:
            deduped.append(sequence)

    return deduped


def build_chat_prompt(messages: list[OpenAIChatMessage]) -> str:
    tokenizer = load_tokenizer()
    normalized_messages = []

    for message in messages:
        normalized_messages.append(
            {
                "role": message.role,
                "content": _extract_text_from_content(message.content),
            }
        )

    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        try:
            return tokenizer.apply_chat_template(
                normalized_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass

    prompt_parts = []
    for message in normalized_messages:
        prompt_parts.append(f"{message['role']}: {message['content']}")

    prompt_parts.append("assistant:")
    return "\n".join(prompt_parts)


def build_usage(prompt_tokens: int, completion_tokens: int) -> dict[str, int]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def build_chat_completion_response(
    *,
    content: str,
    model: str,
    metrics: dict[str, Any],
    finish_reason: str,
) -> dict[str, Any]:
    created = int(time.time())
    return {
        "id": f"chatcmpl-{uuid4().hex}",
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": build_usage(metrics["prompt_tokens"], metrics["generated_tokens"]),
    }


def build_completion_response(
    *,
    text: str,
    model: str,
    metrics: dict[str, Any],
    finish_reason: str,
) -> dict[str, Any]:
    created = int(time.time())
    return {
        "id": f"cmpl-{uuid4().hex}",
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "text": text,
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
        "usage": build_usage(metrics["prompt_tokens"], metrics["generated_tokens"]),
    }


def build_chat_stream_chunk(
    *,
    model: str,
    completion_id: str,
    created: int,
    content: str | None,
    finish_reason: str | None = None,
    include_role: bool = False,
) -> dict[str, Any]:
    chunk: dict[str, Any] = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": finish_reason,
            }
        ],
    }

    delta = chunk["choices"][0]["delta"]
    if include_role:
        delta["role"] = "assistant"

    if content:
        delta["content"] = content

    return chunk


def build_completion_stream_chunk(
    *,
    model: str,
    completion_id: str,
    created: int,
    text: str | None,
    finish_reason: str | None = None,
) -> dict[str, Any]:
    chunk: dict[str, Any] = {
        "id": completion_id,
        "object": "text_completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "text": text or "",
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
    }
    return chunk


def build_model_list() -> dict[str, Any]:
    created = int(time.time())
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": created,
                "owned_by": "local",
            }
        ],
    }
