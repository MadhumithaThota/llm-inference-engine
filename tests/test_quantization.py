import torch
from torch import nn

from engine.quantization import (
    QuantizedLinear,
    dequantize_tensor,
    quantize_tensor,
    replace_linear_layers,
)


def test_quantize_tensor_round_trip_keeps_shape():
    tensor = torch.tensor([[1.0, -2.0], [0.25, 0.5]])

    quantized = quantize_tensor(tensor, bits=8)
    restored = dequantize_tensor(quantized)

    assert quantized.qweight.shape == tensor.shape
    assert restored.shape == tensor.shape


def test_quantized_linear_matches_linear_shape():
    linear = nn.Linear(4, 3)
    layer = QuantizedLinear.from_linear(linear, bits=8)
    output = layer(torch.randn(2, 4))

    assert output.shape == (2, 3)


def test_replace_linear_layers_wraps_linear_modules():
    model = nn.Sequential(
        nn.Linear(4, 4),
        nn.ReLU(),
        nn.Sequential(nn.Linear(4, 2)),
    )

    replace_linear_layers(model, bits=8, skip_names=())

    assert not isinstance(model[0], nn.Linear)
    assert not isinstance(model[2][0], nn.Linear)


