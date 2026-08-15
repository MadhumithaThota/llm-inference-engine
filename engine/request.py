from engine.kv_cache import KVCache
from concurrent.futures import Future
from dataclasses import dataclass, field

@dataclass
class GenerationRequest:
    prompt: str
    max_new_tokens: int
    stream: bool = False
    future: Future | None = None
    kv_cache: KVCache | None = None
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    stop_sequences: list[str] = field(default_factory=list)
