import torch


class KVCache:

    def __init__(self):
        self.past_key_values = None

    def get(self):
        return self.past_key_values

    def update(self, past_key_values):
        self.past_key_values = past_key_values

    def select_batch(self, indices):
        """
        Keep only the specified batch rows in the KV cache.

        Example:

            Original batch:
                [R1, R2, R3]

            indices = [1, 2]

            Result:
                [R2, R3]
        """

        if self.past_key_values is None:
            return

        cache = self.past_key_values

        index_tensor = torch.tensor(
            indices,
            device=cache.layers[0].keys.device,
            dtype=torch.long,
        )

        for layer in cache.layers:

            layer.keys = layer.keys.index_select(
                0,
                index_tensor,
            )

            layer.values = layer.values.index_select(
                0,
                index_tensor,
            )

    def clear(self):
        self.past_key_values = None

    def is_empty(self):
        return self.past_key_values is None