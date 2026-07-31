from threading import Thread

import torch
from transformers import TextIteratorStreamer

from engine.model_loader import load_model, load_tokenizer


@torch.inference_mode()
def generate(
    prompt: str,
    max_new_tokens: int,
    output_handler,
):

    tokenizer = load_tokenizer()
    model = load_model()

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