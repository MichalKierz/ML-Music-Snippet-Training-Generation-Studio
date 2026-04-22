from dataclasses import dataclass

import torch


@dataclass
class KVCache:
    keys: list[torch.Tensor | None]
    values: list[torch.Tensor | None]
    sequence_length: int = 0

    @classmethod
    def create(cls, num_layers: int):
        return cls(
            keys=[None for _ in range(int(num_layers))],
            values=[None for _ in range(int(num_layers))],
            sequence_length=0,
        )

    def get(self, layer_index: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        return self.keys[int(layer_index)], self.values[int(layer_index)]

    def set(self, layer_index: int, key: torch.Tensor, value: torch.Tensor):
        self.keys[int(layer_index)] = key
        self.values[int(layer_index)] = value
        if int(layer_index) == 0:
            self.sequence_length = int(key.shape[2])