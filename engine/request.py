from engine.kv_cache import KVCache
from concurrent.futures import Future
from dataclasses import dataclass

@dataclass
class GenerationRequest:
    prompt: str
    max_new_tokens: int
    stream: bool = False
    future: Future | None = None
    kv_cache: KVCache | None = None