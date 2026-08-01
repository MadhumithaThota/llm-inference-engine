class KVCache:
    def __init__(self):
        self.past_key_values = None

    def get(self):
        return self.past_key_values

    def update(self, past_key_values):
        self.past_key_values = past_key_values

    def clear(self):
        self.past_key_values = None

    def is_empty(self):
        return self.past_key_values is None