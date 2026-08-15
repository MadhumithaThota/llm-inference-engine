import torch

from engine.context_window import (
    get_max_generation_steps,
    resolve_max_context_length,
    validate_prompt_within_context_limit,
)
from engine.model_loader import load_model, load_tokenizer
from engine.utils.sampler import sample_next_token
from engine.metrics import Metrics


def _emit_text_buffer(buffer: str, stop_sequences: list[str], final: bool = False):
    """Return the text that can be safely emitted and the remaining buffered text."""
    if not buffer:
        return "", "", False

    normalized_stop_sequences = [sequence for sequence in stop_sequences if sequence]

    if normalized_stop_sequences:
        earliest_stop_index = None

        for sequence in normalized_stop_sequences:
            stop_index = buffer.find(sequence)

            if stop_index == -1:
                continue

            if earliest_stop_index is None or stop_index < earliest_stop_index:
                earliest_stop_index = stop_index

        if earliest_stop_index is not None:
            return buffer[:earliest_stop_index], "", True

        if not final:
            hold_back = max(len(sequence) for sequence in normalized_stop_sequences) - 1

            if hold_back > 0 and len(buffer) > hold_back:
                return buffer[:-hold_back], buffer[-hold_back:], False

            return "", buffer, False

    return buffer, "", False


@torch.inference_mode()
def generate(
    prompt: str,
    max_new_tokens: int,
    max_context_length: int | None,
    output_handler,
    kv_cache,
    temperature,
    top_k,
    top_p,
    repetition_penalty,
    stop_sequences,
):
    # Create metrics for this request
    metrics = Metrics()
    metrics.start()

    tokenizer = load_tokenizer()
    model = load_model()

    device = model.device
    effective_max_context_length = resolve_max_context_length(
        model,
        tokenizer,
        max_context_length,
    )

    # Tokenize the input prompt and move tensors to the model's device.
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(device)

    prompt_tokens = inputs["input_ids"].shape[1]
    metrics.prompt_tokens = prompt_tokens
    validate_prompt_within_context_limit(prompt_tokens, effective_max_context_length)

    max_generation_steps = get_max_generation_steps(
        prompt_tokens,
        max_new_tokens,
        effective_max_context_length,
    )

    if max_generation_steps <= 0:
        metrics.finish()
        kv_cache.clear()

        result = {
            "response": output_handler.finish(),
            "metrics": metrics.to_dict(),
        }
        print(result["metrics"])
        return result

    # Store prompt token ids for repetition penalty.
    generated_tokens = inputs["input_ids"][0].tolist()

    # First forward pass processes the entire prompt and creates the initial KV cache.
    outputs = model(
        **inputs,
        use_cache=True,
    )

    kv_cache.update(outputs.past_key_values)

    # Generate the first token.
    next_token = sample_next_token(
        outputs.logits[:, -1],
        generated_tokens,
        repetition_penalty,
        temperature,
        top_k,
        top_p,
    )

    buffered_text = ""
    stopped_by_sequence = False

    for _ in range(max_generation_steps):

        if next_token.item() == tokenizer.eos_token_id:
            break

        text = tokenizer.decode(
            next_token[0],
            skip_special_tokens=True,
        )

        generated_tokens.append(next_token.item())

        # Count the actual generated token.
        if metrics.generated_tokens == 0:
            metrics.first_token()

        metrics.generated_tokens += 1

        buffered_text += text

        emitted_text, buffered_text, stopped_by_sequence = _emit_text_buffer(
            buffered_text,
            stop_sequences,
            final=False,
        )

        if emitted_text:
            output_handler.on_text(emitted_text)

        if stopped_by_sequence:
            break

        # Generate next token using the KV cache.
        outputs = model(
            input_ids=next_token,
            past_key_values=kv_cache.get(),
            use_cache=True,
        )

        kv_cache.update(outputs.past_key_values)

        next_token = sample_next_token(
            outputs.logits[:, -1],
            generated_tokens,
            repetition_penalty,
            temperature,
            top_k,
            top_p,
        )

    if buffered_text and not stopped_by_sequence:
        emitted_text, buffered_text, _ = _emit_text_buffer(
            buffered_text,
            stop_sequences,
            final=True,
        )

        if emitted_text:
            if metrics.generated_tokens == 0:
                metrics.first_token()

            metrics.generated_tokens += 1
            output_handler.on_text(emitted_text)

    metrics.finish()

    # Clear cache after request completion.
    kv_cache.clear()

    result = {
        "response": output_handler.finish(),
        "metrics": metrics.to_dict(),
    }

    print(result["metrics"])

    return result
