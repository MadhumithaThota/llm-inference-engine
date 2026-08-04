from engine.generator import generate
from engine.output_handler import (
    BufferedOutputHandler,
    StreamingOutputHandler,
)


class GenerationEngine:

    def generate(self, request):

        return generate(
            request.prompt,
            request.max_new_tokens,
            request.output_handler,
            request.kv_cache,
            request.temperature,
            request.top_k,
            request.top_p,
        )

generation_engine = GenerationEngine()