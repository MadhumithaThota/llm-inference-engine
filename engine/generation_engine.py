from engine.generator import generate
from engine.output_handler import (
    BufferedOutputHandler,
    StreamingOutputHandler,
)


class GenerationEngine:

    def generate(self, request):

        if request.stream:
            handler = StreamingOutputHandler()
        else:
            handler = BufferedOutputHandler()

        return generate(
            request.prompt,
            request.max_new_tokens,
            handler,
        )


generation_engine = GenerationEngine()