import torch
from torch import nn

from engine.tensor_parallel import ParallelLinear, apply_tensor_parallelism


def test_parallel_linear_matches_output_shape():
    linear = nn.Linear(4, 6)
    parallel = ParallelLinear.from_linear(linear, shard_count=3)

    output = parallel(torch.randn(2, 4))

    assert output.shape == (2, 6)


def test_apply_tensor_parallelism_replaces_large_linear_layers():
    model = nn.Sequential(
        nn.Linear(8, 32),
        nn.ReLU(),
        nn.Sequential(nn.Linear(32, 16)),
    )

    apply_tensor_parallelism(model, shard_count=2, min_out_features=8, skip_names=())

    assert not isinstance(model[0], nn.Linear)
    assert not isinstance(model[2][0], nn.Linear)


