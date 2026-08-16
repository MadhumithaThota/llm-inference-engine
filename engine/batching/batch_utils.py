import torch


def select_batch_rows(tensor, indices):
    """
    Select active request rows from a batched tensor.
    """

    if not indices:
        return tensor[:0]

    index_tensor = torch.tensor(
        indices,
        device=tensor.device,
        dtype=torch.long,
    )

    return tensor.index_select(
        0,
        index_tensor
    )