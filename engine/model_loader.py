from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

_tokenizer = None
_model = None


def load_tokenizer():
    global _tokenizer

    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    return _tokenizer


def load_model():
    global _model

    if _model is None:
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32, # means weights are stored as 32-bit floating-point values
        )
        _model.eval()

    return _model