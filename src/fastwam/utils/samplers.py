from typing import Iterator, Sized

import torch
from torch.utils.data import Sampler


class ResumableEpochSampler(Sampler[int]):
    """Deterministic epoch order with an optional distributed resume contract.

    With ``pad_to_global_batch=True``, pad the *complete* shuffled epoch before
    removing resumed batches. Otherwise Accelerate can pad a short resumed
    suffix with that suffix's first indices instead of the original epoch's
    first indices, changing the samples seen by the last rank.

    Legacy callers retain the original unpadded order and length semantics.
    ``resume_batch_offset`` counts batches consumed by each distributed rank;
    it applies only to the first local epoch after a restore.
    """
    def __init__(self, dataset: Sized, seed: int, batch_size: int, num_processes: int,
                 pad_to_global_batch: bool = False):
        self.dataset = dataset
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.num_processes = int(num_processes)
        self.pad_to_global_batch = bool(pad_to_global_batch)
        if self.pad_to_global_batch and min(self.batch_size, self.num_processes) < 1:
            raise ValueError("global-batch padding requires positive batch_size and num_processes")
        self.epoch = 0
        self.epoch_offset = 0
        self.resume_batch_offset = 0

    @property
    def full_epoch_num_batches(self) -> int:
        """Number of batches per rank in an unconsumed, padded epoch."""
        global_batch = self.batch_size * self.num_processes
        return (len(self.dataset) + global_batch - 1) // global_batch

    def set_epoch(self, epoch: int):
        epoch = int(epoch)
        if self.pad_to_global_batch and epoch < 0:
            raise ValueError("epoch must be nonnegative")
        self.epoch = epoch

    def set_epoch_offset(self, epoch_offset: int):
        epoch_offset = int(epoch_offset)
        if self.pad_to_global_batch and epoch_offset < 0:
            raise ValueError("epoch_offset must be nonnegative")
        self.epoch_offset = epoch_offset

    def set_resume_batch_offset(self, batch_in_epoch: int):
        batch_in_epoch = int(batch_in_epoch)
        if self.pad_to_global_batch and not 0 <= batch_in_epoch <= self.full_epoch_num_batches:
            raise ValueError("resume batch offset must be within the complete padded epoch")
        self.resume_batch_offset = batch_in_epoch

    def clear_resume_batch_offset(self):
        self.resume_batch_offset = 0

    def _padded_remaining_length(self) -> int:
        full_batches = self.full_epoch_num_batches
        skipped = self.resume_batch_offset if self.epoch == 0 else 0
        # Recheck if a mutable dataset changed length after restoring progress.
        if not 0 <= skipped <= full_batches:
            raise ValueError("resume batch offset is outside the complete padded epoch")
        return (full_batches - skipped) * self.batch_size * self.num_processes

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed + self.epoch + self.epoch_offset)
        indices = torch.randperm(len(self.dataset), generator=g).tolist()
        if self.pad_to_global_batch:
            self._padded_remaining_length()  # validate before constructing a suffix
            total = self.full_epoch_num_batches * self.batch_size * self.num_processes
            padding = total - len(indices)
            if padding:
                # Repeat the original epoch prefix, including when N is smaller
                # than one global batch and several copies are necessary.
                prefix = indices[:]
                repeats = (padding + len(prefix) - 1) // len(prefix)
                indices.extend((prefix * repeats)[:padding])
        if self.epoch == 0 and self.resume_batch_offset > 0:
            sample_offset = self.resume_batch_offset * self.batch_size * self.num_processes
            indices = indices[sample_offset:]
        return iter(indices)

    def __len__(self) -> int:
        if self.pad_to_global_batch:
            return self._padded_remaining_length()
        return len(self.dataset)
