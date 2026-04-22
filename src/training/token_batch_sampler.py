import math
import random
from collections.abc import Iterator

from torch.utils.data import Sampler


class BucketedTokenBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        lengths: list[int],
        max_tokens_per_batch: int,
        max_examples_per_batch: int,
        shuffle: bool = True,
        bucket_size_multiplier: int = 50,
        seed: int = 42,
    ):
        self.lengths = [max(1, int(value)) for value in lengths]
        self.max_tokens_per_batch = max(1, int(max_tokens_per_batch))
        self.max_examples_per_batch = max(1, int(max_examples_per_batch))
        self.shuffle = bool(shuffle)
        self.bucket_size_multiplier = max(1, int(bucket_size_multiplier))
        self.seed = int(seed)
        self.epoch = 0

        if self.lengths:
            self.max_tokens_per_batch = max(self.max_tokens_per_batch, max(self.lengths))

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _bucket_size(self) -> int:
        return max(self.max_examples_per_batch * self.bucket_size_multiplier, self.max_examples_per_batch)

    def _ordered_indices(self) -> list[int]:
        indices = list(range(len(self.lengths)))
        if not self.shuffle:
            return sorted(indices, key=lambda index: self.lengths[index])

        rng = random.Random(self.seed + self.epoch)
        rng.shuffle(indices)

        bucket_size = self._bucket_size()
        ordered = []

        for start in range(0, len(indices), bucket_size):
            bucket = indices[start : start + bucket_size]
            bucket.sort(key=lambda index: self.lengths[index])
            if rng.random() < 0.5:
                bucket.reverse()
            ordered.extend(bucket)

        return ordered

    def _build_batches_from_indices(self, ordered_indices: list[int]) -> list[list[int]]:
        batches: list[list[int]] = []
        current_batch: list[int] = []
        current_max_length = 0

        for index in ordered_indices:
            item_length = self.lengths[index]
            next_max_length = max(current_max_length, item_length)
            next_batch_size = len(current_batch) + 1
            projected_tokens = next_max_length * next_batch_size

            exceeds_example_limit = next_batch_size > self.max_examples_per_batch
            exceeds_token_limit = projected_tokens > self.max_tokens_per_batch

            if current_batch and (exceeds_example_limit or exceeds_token_limit):
                batches.append(current_batch)
                current_batch = [index]
                current_max_length = item_length
            else:
                current_batch.append(index)
                current_max_length = next_max_length

        if current_batch:
            batches.append(current_batch)

        if self.shuffle and len(batches) > 1:
            rng = random.Random(self.seed + self.epoch + 1)
            rng.shuffle(batches)

        return batches

    def __iter__(self) -> Iterator[list[int]]:
        ordered_indices = self._ordered_indices()
        batches = self._build_batches_from_indices(ordered_indices)
        for batch in batches:
            yield batch

    def __len__(self) -> int:
        if not self.lengths:
            return 0
        ordered_indices = sorted(range(len(self.lengths)), key=lambda index: self.lengths[index])
        return len(self._build_batches_from_indices(ordered_indices))


def estimate_token_budget(
    reference_batch_size: int,
    reference_sequence_length: int,
    min_budget: int = 1,
) -> int:
    return max(int(min_budget), int(reference_batch_size) * int(reference_sequence_length))


def estimate_max_examples_per_batch(
    reference_batch_size: int,
    expansion_factor: int = 4,
) -> int:
    return max(1, int(reference_batch_size) * max(1, int(expansion_factor)))


def estimate_bucket_size_multiplier(dataset_size: int) -> int:
    if int(dataset_size) >= 100000:
        return 100
    if int(dataset_size) >= 20000:
        return 75
    return 50