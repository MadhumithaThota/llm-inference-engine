from __future__ import annotations

import torch
from torch.nn import functional as F


def fused_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    normalized = x * torch.rsqrt(variance + eps)
    return normalized * weight


def fused_linear_bias_activation(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str = "silu",
) -> torch.Tensor:
    output = F.linear(input_tensor, weight, bias)

    if activation == "silu":
        return F.silu(output)
    if activation == "gelu":
        return F.gelu(output)
    if activation == "relu":
        return F.relu(output)

    raise ValueError(f"Unsupported activation: {activation}")


def fused_softmax_sample(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be greater than 0")

    scaled_logits = logits / temperature
    probs = torch.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


