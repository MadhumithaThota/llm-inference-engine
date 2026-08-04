from threading import Thread

import torch
from transformers import TextIteratorStreamer

from engine.model_loader import load_model, load_tokenizer
import time

def sample_next_token(
    logits,
    temperature,
    top_k,
    top_p,
):
    if temperature <= 0:
        raise ValueError("temperature must be greater than 0")

    logits = logits / temperature

    # -----------------------
    # Top-k
    # -----------------------
    if top_k > 0:
        values, _ = torch.topk(logits, k=top_k)

        min_topk = values[:, -1].unsqueeze(-1)

        logits = torch.where(
            logits < min_topk,
            torch.full_like(logits, float("-inf")),
            logits,
        )

    # -----------------------
    # Top-p
    # -----------------------

    if top_p < 1.0:

        # Sort logits
        sorted_logits, sorted_indices = torch.sort(
            logits,
            descending=True,
        )

        # Convert to probabilities
        sorted_probs = torch.softmax(
            sorted_logits,
            dim=-1,
        )

        # Running cumulative probability
        cumulative_probs = torch.cumsum(
            sorted_probs,
            dim=-1,
        )

        # Remove tokens after top_p
        sorted_indices_to_remove = cumulative_probs > top_p

        # Always keep the first token above threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()

        sorted_indices_to_remove[..., 0] = False

        # Scatter back to original vocabulary order
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=1,
            index=sorted_indices,
            src=sorted_indices_to_remove,
        )

        logits = logits.masked_fill(
            indices_to_remove,
            float("-inf"),
        )

    probs = torch.softmax(
        logits,
        dim=-1,
    )

    return torch.multinomial(
        probs,
        num_samples=1,
    )

@torch.inference_mode()
def generate(
    prompt: str,
    max_new_tokens: int,
    output_handler,
    kv_cache,
    temperature,
    top_k,
    top_p,
):
    tokenizer = load_tokenizer()
    model = load_model()

    device = model.device

    # Tokenize the input prompt and move tensors to the model's device.
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(device)

    # First forward pass processes the entire prompt and creates the initial KV cache.
    outputs = model(
        **inputs,
        use_cache=True,
    )

    kv_cache.update(outputs.past_key_values)

    # Select the first generated token using greedy decoding.
    # next_token = outputs.logits[:, -1].argmax(dim=-1, keepdim=True)# Get logits for the last token
    next_token = sample_next_token(
        outputs.logits[:, -1],
        temperature,
        top_k,
        top_p,
    )


    for _ in range(max_new_tokens):

        # Stop generation if the model predicts the EOS token.
        if next_token.item() == tokenizer.eos_token_id:
            break

        # Decode and stream the generated token.
        text = tokenizer.decode(
            next_token[0],
            skip_special_tokens=True,
        )

        if text:
            output_handler.on_text(text)

        # For subsequent iterations, only the latest token is passed.
        # The KV cache contains all previous attention states.
        outputs = model(
            input_ids=next_token,
            past_key_values=kv_cache.get(),
            use_cache=True,
        )

        # Update the cache with the newly computed keys and values.
        kv_cache.update(outputs.past_key_values)

        # Greedily select the next token.
        #next_token = outputs.logits[:, -1].argmax(dim=-1, keepdim=True)

        next_token = sample_next_token(
            outputs.logits[:, -1],
            temperature,
            top_k,
            top_p,
        )

    # Clear the cache after the request completes.
    kv_cache.clear()

    return output_handler.finish()

"""
def generate(
    prompt: str,
    max_new_tokens: int,
    output_handler,
    kv_cache,
):

    print("1. Loading tokenizer")
    tokenizer = load_tokenizer()

    print("2. Loading model")
    model = load_model()

    print("3. Tokenizing")
    inputs = tokenizer(prompt, return_tensors="pt")

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        timeout=60.0,
    )
    thread = Thread(
        target=model.generate,
        kwargs={
            **inputs,
            "streamer": streamer,
            "max_new_tokens": max_new_tokens,
        },
    )
    thread.start()

    for text in streamer:
        output_handler.on_text(text)

    thread.join()
    

    return output_handler.finish()
"""