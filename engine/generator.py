
import torch

from engine.model_loader import load_model, load_tokenizer
from engine.utils.sampler import sample_next_token

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
    tokenizer = load_tokenizer()
    model = load_model()

    device = model.device

    # Tokenize the input prompt and move tensors to the model's device.
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(device)
    generated_tokens = inputs["input_ids"][0].tolist()
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
        generated_tokens,
        repetition_penalty,
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
        generated_tokens.append(next_token.item())
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
            generated_tokens,
            repetition_penalty,
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