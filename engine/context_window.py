def _pick_positive_int(value):
    if isinstance(value, int) and value > 0:
        return value

    return None


def resolve_max_context_length(model, tokenizer, requested_max_context_length=None):
    candidates = []

    requested_limit = _pick_positive_int(requested_max_context_length)
    if requested_limit is not None:
        candidates.append(requested_limit)

    model_config = getattr(model, "config", None)
    if model_config is not None:
        model_limit = _pick_positive_int(
            getattr(model_config, "max_position_embeddings", None)
        )
        if model_limit is not None:
            candidates.append(model_limit)

    tokenizer_limit = _pick_positive_int(getattr(tokenizer, "model_max_length", None))
    if tokenizer_limit is not None and tokenizer_limit < 10**9:
        candidates.append(tokenizer_limit)

    if not candidates:
        return None

    return min(candidates)


def validate_prompt_within_context_limit(prompt_tokens: int, max_context_length: int | None):
    if max_context_length is None:
        return

    if prompt_tokens > max_context_length:
        raise ValueError(
            f"Prompt has {prompt_tokens} tokens, which exceeds the maximum context length of {max_context_length}."
        )


def get_max_generation_steps(prompt_tokens: int, max_new_tokens: int, max_context_length: int | None):
    if max_context_length is None:
        return max_new_tokens

    remaining_context = max_context_length - prompt_tokens

    if remaining_context < 0:
        return 0

    return min(max_new_tokens, remaining_context)
