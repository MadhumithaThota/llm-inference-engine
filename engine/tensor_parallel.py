from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class TensorParallelConfig:
    shard_count: int = 2
    device_ids: list[int | str] | None = None
    min_out_features: int = 1024


class ParallelLinear(nn.Module):
    """
    A small educational version of output-sharded linear layers.

    Each shard owns a slice of the output features. During forward pass, the
    layer computes the shards one by one and concatenates them back together.
    """

    def __init__(
        self,
        weight_shards: list[torch.Tensor],
        bias_shards: list[torch.Tensor | None],
        shard_devices: list[torch.device],
    ):
        super().__init__()
        self.weight_shards = nn.ParameterList(
            [nn.Parameter(weight) for weight in weight_shards]
        )
        self.bias_shards = nn.ParameterList(
            [nn.Parameter(bias) if bias is not None else nn.Parameter(torch.tensor([]), requires_grad=False)
             for bias in bias_shards]
        )
        self.shard_devices = shard_devices

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        shard_count: int = 2,
        device_ids: list[int | str] | None = None,
    ):
        if shard_count <= 1:
            raise ValueError("shard_count must be greater than 1")

        device_ids = device_ids or [linear.weight.device for _ in range(shard_count)]
        if len(device_ids) < shard_count:
            device_ids = device_ids + [device_ids[-1]] * (shard_count - len(device_ids))

        weight_chunks = torch.tensor_split(linear.weight.detach(), shard_count, dim=0)
        if linear.bias is not None:
            bias_chunks = torch.tensor_split(linear.bias.detach(), shard_count, dim=0)
        else:
            bias_chunks = [None for _ in range(shard_count)]

        shard_devices = [torch.device(device) for device in device_ids[:shard_count]]
        return cls(
            weight_shards=[
                chunk.contiguous().to(device)
                for chunk, device in zip(weight_chunks, shard_devices)
            ],
            bias_shards=[
                chunk.contiguous().to(device) if chunk is not None else None
                for chunk, device in zip(bias_chunks, shard_devices)
            ],
            shard_devices=shard_devices,
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        base_device = input_tensor.device
        shard_outputs = []

        for weight, bias, shard_device in zip(
            self.weight_shards,
            self.bias_shards,
            self.shard_devices,
        ):
            shard_input = input_tensor.to(shard_device) if input_tensor.device != shard_device else input_tensor
            shard_bias = None if bias.numel() == 0 else bias
            shard_output = F.linear(shard_input, weight, shard_bias)
            shard_outputs.append(shard_output.to(base_device))

        return torch.cat(shard_outputs, dim=-1)


def _should_skip(name: str, skip_names: tuple[str, ...]) -> bool:
    return any(skip_name in name for skip_name in skip_names)


def apply_tensor_parallelism(
    module: nn.Module,
    *,
    shard_count: int = 2,
    device_ids: list[int | str] | None = None,
    min_out_features: int = 1024,
    skip_names: tuple[str, ...] = ("lm_head", "embed_tokens", "wte", "tok_embeddings"),
):
    for child_name, child_module in list(module.named_children()):
        if isinstance(child_module, nn.Linear):
            if child_module.out_features >= min_out_features and not _should_skip(child_name, skip_names):
                setattr(
                    module,
                    child_name,
                    ParallelLinear.from_linear(
                        child_module,
                        shard_count=shard_count,
                        device_ids=device_ids,
                    ),
                )
            continue

        apply_tensor_parallelism(
            child_module,
            shard_count=shard_count,
            device_ids=device_ids,
            min_out_features=min_out_features,
            skip_names=skip_names,
        )

    return module


