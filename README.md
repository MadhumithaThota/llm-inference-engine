# LLM Inference Engine

A production-inspired LLM inference engine built from scratch using Python and PyTorch.

The goal of this project is to understand and implement the core components behind modern inference systems such as vLLM and TensorRT-LLM instead of simply wrapping Hugging Face models.

Current features include:

- FastAPI REST API
- Streaming text generation
- Request scheduler
- Background worker
- Request queue
- Modular generation pipeline

---

## Tech Stack

- Python 3.12
- PyTorch
- Hugging Face Transformers
- FastAPI
- Uvicorn
- Pydantic

---

## Setup

### Clone repository

```bash
git clone https://github.com/<your-username>/llm-inference-engine.git
cd llm-inference-engine
```

### Create virtual environment

```bash
python3 -m venv .venv
```

Linux/macOS

```bash
source .venv/bin/activate
```

Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run

```bash
uvicorn server.api:app --reload
```

Swagger UI

```
http://127.0.0.1:8000/docs
```

---

## Example Request

```json
{
    "prompt": "Explain transformers.",
    "max_new_tokens": 128,
    "stream": false
}
```

Streaming request

```json
{
    "prompt": "Explain transformers.",
    "max_new_tokens": 128,
    "stream": true
}
```

---

## Running Tests

```bash
pytest
```

---

## Roadmap

- [x] REST API
- [x] Streaming generation
- [x] Request scheduler
- [x] Request queue
- [x] Background worker
- [x] Unified generation pipeline
- [ ] Dynamic batch builder
- [ ] Batched inference
- [x] KV Cache
- [ ] Continuous batching
- [ ] Tensor parallelism
- [ ] Quantization
- [ ] CUDA kernels
- [ ] OpenAI-compatible API
- [ ] Performance benchmarking

---

## Motivation

This project is an educational implementation of a production-style inference engine. The focus is on understanding scheduling, batching, decoding, and systems design rather than building a chatbot application.
