from dataclasses import dataclass
from concurrent.futures import Future


@dataclass
class GenerationRequest:
    prompt: str
    max_new_tokens: int
    stream: bool = False
    future: Future | None = None