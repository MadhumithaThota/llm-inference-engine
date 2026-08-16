import torch
from threading import RLock

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

_tokenizer_lock = RLock()
_model_lock = RLock()
_tokenizer = None
_model = None


class ThreadSafeTokenizer:
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    def __call__(self, *args, **kwargs):
        with _tokenizer_lock:
            return self._tokenizer(*args, **kwargs)

    def decode(self, *args, **kwargs):
        with _tokenizer_lock:
            return self._tokenizer.decode(*args, **kwargs)

    def encode(self, *args, **kwargs):
        with _tokenizer_lock:
            return self._tokenizer.encode(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        with _tokenizer_lock:
            return self._tokenizer.batch_decode(*args, **kwargs)

    def apply_chat_template(self, *args, **kwargs):
        with _tokenizer_lock:
            return self._tokenizer.apply_chat_template(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._tokenizer, name)


def load_tokenizer():
    global _tokenizer

    with _tokenizer_lock:
        if _tokenizer is None:
            _tokenizer = ThreadSafeTokenizer(
                AutoTokenizer.from_pretrained(MODEL_NAME)
            )

    return _tokenizer


def load_model():
    global _model

    with _model_lock:
        if _model is None:
            _model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                dtype=torch.float32, # means weights are stored as 32-bit floating-point values
            )
            _model.eval()

    return _model
