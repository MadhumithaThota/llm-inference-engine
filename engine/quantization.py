from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class QuantizedWeight:
    qweight: torch.Tensor
    scale: torch.Tensor
    bits: int


def quantize_tensor(tensor: torch.Tensor, bits: int = 8) -> QuantizedWeight:
    if bits not in {4, 8}:
        raise ValueError("bits must be 4 or 8")

    if tensor.ndim == 0:
        tensor = tensor.reshape(1)

    qmax = (2 ** (bits - 1)) - 1
    max_abs = tensor.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = max_abs / qmax
    qweight = torch.round(tensor / scale).clamp(-qmax, qmax).to(torch.int8)
    return QuantizedWeight(qweight=qweight, scale=scale, bits=bits)


def dequantize_tensor(quantized: QuantizedWeight) -> torch.Tensor:
    return quantized.qweight.float() * quantized.scale


class QuantizedLinear(nn.Module):
    def __init__(
        self,
        qweight: torch.Tensor,
        scale: torch.Tensor,
        bias: torch.Tensor | None,
        bits: int,
    ):
        super().__init__()
        self.register_buffer("qweight", qweight)
        self.register_buffer("scale", scale)
        self.register_buffer("bias", bias if bias is not None else None)
        self.bits = bits

    @classmethod
    def from_linear(cls, linear: nn.Linear, bits: int = 8):
        quantized = quantize_tensor(linear.weight.detach(), bits=bits)
        bias = linear.bias.detach().clone() if linear.bias is not None else None
        return cls(quantized.qweight, quantized.scale, bias, bits)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        weight = self.qweight.float() * self.scale
        bias = self.bias
        return F.linear(input_tensor, weight, bias)


def _should_skip(name: str, skip_names: tuple[str, ...]) -> bool:
    return any(skip_name in name for skip_name in skip_names)


def replace_linear_layers(
    module: nn.Module,
    *,
    bits: int = 8,
    skip_names: tuple[str, ...] = ("lm_head", "embed_tokens", "wte", "tok_embeddings"),
):
    for child_name, child_module in list(module.named_children()):
        if isinstance(child_module, nn.Linear) and not _should_skip(child_name, skip_names):
            setattr(module, child_name, QuantizedLinear.from_linear(child_module, bits=bits))
            continue

        replace_linear_layers(
            child_module,
            bits=bits,
            skip_names=skip_names,
        )

    return module


