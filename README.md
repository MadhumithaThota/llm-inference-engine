# LLM Inference Engine

A from-scratch, production-inspired LLM inference engine built with Python, PyTorch, and FastAPI.

The project focuses on the systems behind modern serving runtimes: request scheduling, token-by-token decoding, KV caching, sampling, streaming, and batching. It is designed as a learning-oriented implementation rather than a wrapper around a hosted model API.

## What Is an Inference Engine?

An inference engine is the part of a machine learning system that takes a trained model and runs it to produce outputs for new inputs.

In this project, the inference engine is responsible for:

- Loading the model and tokenizer
- Accepting generation requests
- Scheduling and batching requests
- Decoding tokens one step at a time
- Applying sampling, repetition penalty, and stop sequences
- Returning streamed or buffered responses

## Usage

### Start the server

```bash
uvicorn server.api:app --reload
```

### Generate text

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain transformers.",
    "max_new_tokens": 128,
    "stream": false
  }'
```

### Test batching

Run the batch smoke script from Command Prompt:

```cmd
D:\Projects\llm-inference-engine\tests\test_batch_requests.cmd
```

Or run the Python batch demo:

```cmd
D:\Projects\llm\Scripts\python.exe D:\Projects\llm-inference-engine\test_batch.py
```

## Highlights

- FastAPI generation endpoint with interactive Swagger documentation
- Buffered and streaming text generation
- Background worker, request queue, and scheduler
- Manual token decoding with configurable sampling
- Per-request KV cache
- Prefix caching with LRU and TTL eviction
- Modular engine components for experimentation

## Tech Stack

- Python 3.12
- PyTorch and Hugging Face Transformers
- FastAPI and Uvicorn
- Pydantic

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/llm-inference-engine.git
cd llm-inference-engine
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the server

```bash
uvicorn server.api:app --reload
```

The API is available at `http://127.0.0.1:8000` and Swagger UI at `http://127.0.0.1:8000/docs`.

## API Usage

### Health check

```bash
curl http://127.0.0.1:8000/
```

### Generate text

Send a non-streaming request to receive the completed response as JSON:

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain transformers.",
    "max_new_tokens": 128,
    "max_context_length": 2048,
    "stream": false,
    "temperature": 1.0,
    "top_k": 0,
    "top_p": 1.0,
    "repetition_penalty": 1.0,
    "stop_sequences": ["###", "\\n\\nUser:"]
  }'
```

Example response:

```json
{
  "response": "Transformers are ...",
  "metrics": {
    "prompt_tokens": 5,
    "generated_tokens": 18,
    "ttft_ms": 412.31,
    "latency_ms": 1638.77,
    "tokens_per_second": 10.98,
    "prefix_cache_hits": 1,
    "prefix_cache_misses": 0,
    "prefix_cache_hit_rate": 100.0
  }
}
```

Set `"stream": true` to receive generated text as a streaming plain-text response. Metrics are returned with the buffered JSON response.

### OpenAI-compatible endpoints

The server also exposes OpenAI-style routes:

- `POST /v1/chat/completions`
- `POST /v1/completions`
- `GET /v1/models`

Example chat-completions request:

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain transformers in one paragraph."}
    ],
    "max_tokens": 128,
    "stream": false
  }'
```

The OpenAI-compatible responses follow the standard `choices` and `usage` shapes, and streaming uses SSE with `data: ...` chunks and a final `[DONE]`.

## Tests

```bash
pytest
```

### Batching smoke test

Run the parallel request harness from Command Prompt:

```cmd
D:\Projects\llm-inference-engine\tests\test_batch_requests.cmd
```

This script sends 3 requests at the same time, writes the responses to `response1.json`, `response2.json`, and `response3.json`, and prints scheduler logs in the server console.

If you want to compare it with the Python demo used during development:

```cmd
D:\Projects\llm\Scripts\python.exe D:\Projects\llm-inference-engine\test_batch.py
```

For a clearer batching trace, watch for these server log lines:

- `[worker] received requests`
- `[scheduler] queued request`
- `[scheduler] started session ...`
- `[scheduler] completed session ...`

## Roadmap

### Phase 1 - Completed

- [x] Model loading
- [x] Worker and request queue
- [x] Manual decoding
- [x] KV cache
- [x] Temperature, top-k, and top-p sampling
- [x] Repetition penalty
- [x] Streaming generation

### Phase 2

- [x] Performance metrics
- [x] Stop sequences
- [x] Maximum context length
- [x] OpenAI-compatible API

### Phase 3

- [x] Prefix caching
- [x] Cache eviction (LRU/TTL)

### Phase 4

- [x] Dynamic batch builder
- [x] Batched inference
- [x] Dynamic scheduler

### Phase 5

- [x] Continuous batching

### Phase 6

- [x] Paged KV cache

Learning-friendly page slicing lives in `engine/paged_kv_cache.py`.

### Phase 7

- [x] Quantization (FP16 -> INT8 -> INT4)

Weight-only quantization helpers live in `engine/quantization.py`.

### Phase 8

- [x] Tensor parallelism

Sharded linear-layer helpers live in `engine/tensor_parallel.py`.

### Phase 9

- [x] CUDA kernels and kernel fusion

This phase is represented by fused PyTorch helper ops in `engine/fused_ops.py`.

### Phase 10

- [x] Performance benchmarking

The lightweight benchmark harness lives in `engine/benchmarking.py`.

## Learning Notes

The newest phases are intentionally beginner-friendly:

- The paged KV cache keeps a paged view for learning, while the runtime still uses the normal Hugging Face cache object.
- Quantization is weight-only and opt-in through `MODEL_QUANT_BITS=8` or `MODEL_QUANT_BITS=4`.
- Tensor parallelism uses output-sharded linear layers and is also opt-in through `TENSOR_PARALLEL_SHARDS`.
- The fusion phase gives you simple fused helper functions that are easier to study before writing custom CUDA.
- The benchmarking helpers are small on purpose so you can extend them with your own experiments.

## Project Goal

This repository is an educational path toward a production-style inference engine. The emphasis is on understanding the runtime trade-offs behind scheduling, batching, decoding, and memory management.
