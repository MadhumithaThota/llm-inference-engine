import torch

from engine.batching.batch_utils import select_batch_rows
from engine.generator import _emit_text_buffer
from engine.model_loader import load_model, load_tokenizer
from engine.utils.sampler import sample_next_token


def select_cache_batch(cache, indices):
    """
    Compact a Hugging Face DynamicCache along the batch dimension.
    """
    if not indices:
        return

    index_tensor = torch.tensor(
        indices,
        device=cache.layers[0].keys.device,
        dtype=torch.long,
    )

    for layer in cache.layers:
        layer.keys = layer.keys.index_select(0, index_tensor)
        layer.values = layer.values.index_select(0, index_tensor)


def sample_batch_tokens(
    logits,
    active_indices,
    requests,
    generated_tokens,
):
    """
    Sample one token independently for every active request.
    """
    tokens = []

    for batch_position, request_index in enumerate(active_indices):
        request_logits = logits[batch_position].unsqueeze(0)

        token = sample_next_token(
            request_logits,
            generated_tokens[request_index],
            requests[request_index].repetition_penalty,
            requests[request_index].temperature,
            requests[request_index].top_k,
            requests[request_index].top_p,
        )

        tokens.append(token)

    return torch.cat(tokens, dim=0)


class BatchedGenerationSession:
    def __init__(self, requests):
        if not requests:
            raise ValueError("requests must not be empty")

        self.requests = requests
        self.tokenizer = load_tokenizer()
        self.model = load_model()
        self.device = self.model.device

        self.prompts = [request.prompt for request in requests]
        self.inputs = self.tokenizer(
            self.prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        ).to(self.device)

        self.batch_size = len(requests)
        self.active_indices = list(range(self.batch_size))
        self.generated_tokens = []
        self.prompt_lengths = []
        self.generated_counts = [0 for _ in range(self.batch_size)]
        self.finished = [False for _ in range(self.batch_size)]
        self.buffers = ["" for _ in range(self.batch_size)]
        self.response_parts = [[] for _ in range(self.batch_size)]
        self._done = False
        self._results = None

        print("\nBatched inference")
        print("----------------------------")
        print("Batch size       :", self.batch_size)
        print("Input IDs shape  :", self.inputs["input_ids"].shape)
        print("Attention shape  :", self.inputs["attention_mask"].shape)

        outputs = self.model(
            **self.inputs,
            use_cache=True,
        )

        print("Logits shape     :", outputs.logits.shape)

        last_token_indices = self.inputs["attention_mask"].sum(dim=1) - 1
        batch_indices = torch.arange(self.batch_size, device=self.device)

        next_token_logits = outputs.logits[batch_indices, last_token_indices]

        for index in range(self.batch_size):
            prompt_token_ids = self.inputs["input_ids"][index][
                self.inputs["attention_mask"][index].bool()
            ].tolist()
            self.generated_tokens.append(prompt_token_ids)
            self.prompt_lengths.append(len(prompt_token_ids))

        self.past_key_values = outputs.past_key_values

        print("KV cache type    :", type(self.past_key_values))

        self.next_tokens = sample_batch_tokens(
            next_token_logits,
            self.active_indices,
            self.requests,
            self.generated_tokens,
        )

    def _finalize_request(self, request_index: int):
        if self.finished[request_index]:
            return

        emitted_text, remaining_buffer, _ = _emit_text_buffer(
            self.buffers[request_index],
            getattr(self.requests[request_index], "stop_sequences", []),
            final=True,
        )

        if emitted_text:
            self.response_parts[request_index].append(emitted_text)

        self.buffers[request_index] = remaining_buffer
        self.finished[request_index] = True

    def _build_results(self):
        results = []

        for index, _request in enumerate(self.requests):
            generated_text = "".join(self.response_parts[index])

            results.append(
                {
                    "response": generated_text,
                    "metrics": {
                        "prompt_tokens": self.prompt_lengths[index],
                        "generated_tokens": self.generated_counts[index],
                        "ttft_ms": 0.0,
                        "latency_ms": 0.0,
                        "tokens_per_second": 0.0,
                        "prefix_cache_hits": 0,
                        "prefix_cache_misses": 0,
                        "prefix_cache_hit_rate": 0.0,
                    },
                }
            )

        return results

    def step(self) -> bool:
        if self._done:
            return False

        previous_active_indices = self.active_indices.copy()

        for batch_position, request_index in enumerate(previous_active_indices):
            token_id = self.next_tokens[batch_position].item()
            request = self.requests[request_index]

            if (
                self.tokenizer.eos_token_id is not None
                and token_id == self.tokenizer.eos_token_id
            ):
                self._finalize_request(request_index)
                continue

            self.generated_tokens[request_index].append(token_id)
            self.generated_counts[request_index] += 1

            text = self.tokenizer.decode(
                [token_id],
                skip_special_tokens=True,
            )
            self.buffers[request_index] += text

            emitted_text, remaining_buffer, stopped = _emit_text_buffer(
                self.buffers[request_index],
                getattr(request, "stop_sequences", []),
                final=False,
            )

            if emitted_text:
                self.response_parts[request_index].append(emitted_text)

            self.buffers[request_index] = remaining_buffer

            if stopped:
                self._finalize_request(request_index)
                continue

            if self.generated_counts[request_index] >= request.max_new_tokens:
                self._finalize_request(request_index)

        self.active_indices = [
            request_index
            for request_index in previous_active_indices
            if not self.finished[request_index]
        ]

        print(
            "Active requests:",
            [index + 1 for index in self.active_indices],
        )

        if not self.active_indices:
            self._done = True
            self._results = self._build_results()

            print("\nFinal responses")
            print("----------------------------")

            for index, result in enumerate(self._results):
                print(f"Request {index + 1}: {repr(result['response'])}")

            return False

        active_batch_positions = [
            batch_position
            for batch_position, request_index in enumerate(previous_active_indices)
            if not self.finished[request_index]
        ]

        print("Active batch positions:", active_batch_positions)

        select_cache_batch(
            self.past_key_values,
            active_batch_positions,
        )

        print("KV cache compacted to batch:", len(self.active_indices))

        self.next_tokens = select_batch_rows(
            self.next_tokens,
            active_batch_positions,
        )

        print("Next token batch shape:", self.next_tokens.shape)

        outputs = self.model(
            input_ids=self.next_tokens,
            past_key_values=self.past_key_values,
            use_cache=True,
        )

        self.past_key_values = outputs.past_key_values

        next_token_logits = outputs.logits[:, -1, :]

        self.next_tokens = sample_batch_tokens(
            next_token_logits,
            self.active_indices,
            self.requests,
            self.generated_tokens,
        )

        return True

    def run_to_completion(self):
        while self.step():
            pass

        return self.results

    @property
    def is_finished(self) -> bool:
        return self._done

    @property
    def results(self):
        if not self._done:
            return None

        return self._results


@torch.inference_mode()
def batched_prompt_inference(requests):
    session = BatchedGenerationSession(requests)
    session.run_to_completion()
    return session.results
