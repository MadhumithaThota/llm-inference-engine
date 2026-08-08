import time


class Metrics:

    def __init__(self):
        self.start_time = 0
        self.first_token_time = None
        self.end_time = 0

        self.prompt_tokens = 0
        self.generated_tokens = 0

    def start(self):
        self.start_time = time.perf_counter()

    def first_token(self):
        if self.first_token_time is None:
            self.first_token_time = time.perf_counter()

    def finish(self):
        self.end_time = time.perf_counter()

    def to_dict(self):

        latency = self.end_time - self.start_time

        ttft = (
            self.first_token_time - self.start_time
            if self.first_token_time
            else 0
        )

        generation_time = (
            self.end_time - self.first_token_time
            if self.first_token_time
            else 0
        )

        tps = (
            self.generated_tokens / generation_time
            if generation_time > 0
            else 0
        )

        return {
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "ttft_ms": round(ttft * 1000, 2),
            "latency_ms": round(latency * 1000, 2),
            "tokens_per_second": round(tps, 2),
        }