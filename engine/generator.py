import torch

from engine.model_loader import load_model, load_tokenizer


@torch.inference_mode()
def generate(prompt: str, max_new_tokens: int = 100) -> str:
    tokenizer = load_tokenizer()
    model = load_model()

    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
    )

    generated_text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    )

    return generated_text