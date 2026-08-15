# LLM Inference Engine

A from-scratch, production-inspired LLM inference engine built with Python, PyTorch, and FastAPI.

The project focuses on the systems behind modern serving runtimes: request scheduling, token-by-token decoding, KV caching, sampling, streaming, and batching. It is designed as a learning-oriented implementation rather than a wrapper around a hosted model API.

## Highlights

- FastAPI generation endpoint with interactive Swagger documentation
- Buffered and streaming text generation
- Background worker, request queue, and scheduler
- Manual token decoding with configurable sampling
- Per-request KV cache
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
- [ ] Cache eviction (LRU/TTL)

### Phase 4

- [ ] Dynamic batch builder
- [ ] Batched inference
- [ ] Dynamic scheduler

### Phase 5

- [ ] Continuous batching

### Phase 6

- [ ] Paged KV cache

### Phase 7

- [ ] Quantization (FP16 -> INT8 -> INT4)

### Phase 8

- [ ] Tensor parallelism

### Phase 9

- [ ] CUDA kernels and kernel fusion

### Phase 10

- [ ] Performance benchmarking

## Project Goal

This repository is an educational path toward a production-style inference engine. The emphasis is on understanding the runtime trade-offs behind scheduling, batching, decoding, and memory management.
