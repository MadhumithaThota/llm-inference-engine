from threading import Thread

import torch
from transformers import TextIteratorStreamer

from engine.model_loader import load_model, load_tokenizer
import time


@torch.inference_mode()
def generate(
    prompt: str,
    max_new_tokens: int,
    output_handler,
    kv_cache,
):
    print("=" * 80)
    print("STARTING GENERATION")
    print("=" * 80)

    start_time = time.time()

    print(f"Prompt              : {prompt}")
    print(f"Max New Tokens      : {max_new_tokens}")

    print("\n[1] Loading tokenizer...")
    tokenizer = load_tokenizer()
    print("✓ Tokenizer loaded")

    print("\n[2] Loading model...")
    model = load_model()
    model.eval()
    print("✓ Model loaded")

    device = model.device
    print(f"Device              : {device}")

    print("\n[3] Tokenizing prompt...")
    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(device)

    print("Input IDs Shape     :", inputs["input_ids"].shape)
    print("Attention Mask      :", inputs["attention_mask"].shape)
    print("Prompt Tokens       :", inputs["input_ids"].shape[1])

    print("\n[4] First forward pass...")
    outputs = model(
        **inputs,
        use_cache=True,
    )

    print("✓ Forward pass complete")

    print("\n[5] Saving KV Cache...")
    kv_cache.update(outputs.past_key_values)

    cache = kv_cache.get()

    print("Cache Type          :", type(cache))
    print("Number of Layers    :", len(cache.layers))
    print("Sequence Length     :", cache.get_seq_length())

    first_layer = cache.layers[0]

    print("Layer 0 Key Shape   :", first_layer.keys.shape)
    print("Layer 0 Value Shape :", first_layer.values.shape)

    next_token = outputs.logits[:, -1].argmax(dim=-1, keepdim=True)

    print("\nFirst Predicted Token")
    print("------------------------------")
    print("Token ID            :", next_token.item())
    print(
        "Decoded Token       :",
        repr(tokenizer.decode(next_token[0], skip_special_tokens=True)),
    )

    print("\n[6] Starting decoding loop...")
    print("=" * 80)

    for step in range(max_new_tokens):

        print(f"\nIteration {step + 1}")

        if next_token.item() == tokenizer.eos_token_id:
            print("EOS token generated. Stopping generation.")
            break

        text = tokenizer.decode(
            next_token[0],
            skip_special_tokens=True,
        )

        print("Generated Token ID  :", next_token.item())
        print("Decoded Text        :", repr(text))

        if text:
            output_handler.on_text(text)

        print("Running forward pass using KV Cache...")

        outputs = model(
            input_ids=next_token,
            past_key_values=kv_cache.get(),
            use_cache=True,
        )

        kv_cache.update(outputs.past_key_values)

        cache = kv_cache.get()

        print("Cache Seq Length    :", cache.get_seq_length())

        next_token = outputs.logits[:, -1].argmax(dim=-1, keepdim=True)

        print("Next Token ID       :", next_token.item())

    print("\n[7] Clearing KV Cache...")
    kv_cache.clear()

    print("Cache Empty         :", kv_cache.is_empty())

    print("\nGeneration Finished")
    print(f"Elapsed Time        : {time.time() - start_time:.2f} sec")

    print("=" * 80)

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