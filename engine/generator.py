import torch

from engine.model_loader import load_model, load_tokenizer
from engine.utils.sampler import sample_next_token
from engine.metrics import Metrics


@torch.inference_mode()
def generate(
    prompt: str,
    max_new_tokens: int,
    output_handler,
    kv_cache,
    temperature,
    top_k,
    top_p,
    repetition_penalty,
):
    # Create metrics for this request
    metrics = Metrics()
    metrics.start()

    tokenizer = load_tokenizer()
    model = load_model()

    device = model.device

    # Tokenize the input prompt and move tensors to the model's device.
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(device)

    metrics.prompt_tokens = inputs["input_ids"].shape[1]

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

    for _ in range(max_new_tokens):

        # Stop generation if EOS token is produced.
        if next_token.item() == tokenizer.eos_token_id:
            break

        # Decode generated token.
        text = tokenizer.decode(
            next_token[0],
            skip_special_tokens=True,
        )

        generated_tokens.append(next_token.item())

        # Stream/output generated text.
        if text:

            # Record TTFT only once.
            if metrics.generated_tokens == 0:
                metrics.first_token()

            metrics.generated_tokens += 1

            output_handler.on_text(text)

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

    metrics.finish()

    # Clear cache after request completion.
    kv_cache.clear()

    print(metrics.to_dict())
    
    return {
        "response": output_handler.finish(),
        "metrics": metrics.to_dict(),
    }
