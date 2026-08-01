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
        )

generation_engine = GenerationEngine()