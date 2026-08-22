import torch
from threading import RLock
import os

from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.quantization import replace_linear_layers
from engine.tensor_parallel import apply_tensor_parallelism

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
                torch_dtype=torch.float32,  # keep the base model easy to run and inspect
            )

            quant_bits = os.getenv("MODEL_QUANT_BITS")
            if quant_bits in {"4", "8"}:
                replace_linear_layers(_model, bits=int(quant_bits))

            shard_count = int(os.getenv("TENSOR_PARALLEL_SHARDS", "1"))
            if shard_count > 1:
                apply_tensor_parallelism(
                    _model,
                    shard_count=shard_count,
                    min_out_features=int(os.getenv("TENSOR_PARALLEL_MIN_OUT_FEATURES", "1024")),
                )

            _model.eval()

    return _model
