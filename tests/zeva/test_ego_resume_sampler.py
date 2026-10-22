"""Predictive Stage2 resume sees exactly the continuous distributed suffix."""
import pytest
import torch
from torch.utils.data import BatchSampler
from accelerate.data_loader import BatchSamplerShard, DataLoaderShard

from fastwam.utils.samplers import ResumableEpochSampler


def _sampler(n, *, batch_size=2, world_size=2, offset=0, epoch=0, epoch_offset=0, seed=19):
    sampler = ResumableEpochSampler(range(n), seed, batch_size, world_size, pad_to_global_batch=True)
    sampler.set_epoch_offset(epoch_offset)
    sampler.set_epoch(epoch)
    sampler.set_resume_batch_offset(offset)
    return sampler


def _shard(sampler, rank):
    batches = BatchSampler(sampler, batch_size=sampler.batch_size, drop_last=False)
    return BatchSamplerShard(batches, num_processes=sampler.num_processes, process_index=rank,
                             split_batches=False, even_batches=True)


@pytest.mark.parametrize("n,batch_size,world_size,offset", [
    (10, 2, 2, 2),  # last rank must receive the ORIGINAL epoch prefix
    (11, 2, 3, 1),
    (17, 3, 2, 2),
    (16, 2, 2, 2),
    (3, 2, 4, 0),
    (1, 2, 4, 0),
])
def test_real_accelerate_rank_suffix_matches_uninterrupted_run(n, batch_size, world_size, offset):
    for rank in range(world_size):
        continuous = _sampler(n, batch_size=batch_size, world_size=world_size, epoch=7)
        full_batches = list(_shard(continuous, rank))
        resumed = _sampler(n, batch_size=batch_size, world_size=world_size, epoch_offset=7, offset=offset)
        resumed_shard = _shard(resumed, rank)
        remaining = list(resumed_shard)
        assert remaining == full_batches[offset:]
        assert len(resumed_shard) == len(remaining)
        assert len(resumed) == len(list(resumed)) == len(remaining) * batch_size * world_size
        assert all(len(batch) == batch_size for batch in remaining)


@pytest.mark.parametrize("n,world_size", [(1, 4), (3, 4), (10, 2)])
def test_padding_cycles_the_full_original_epoch_prefix(n, world_size):
    sampler = _sampler(n, world_size=world_size)
    raw = torch.randperm(n, generator=torch.Generator().manual_seed(19)).tolist()
    total = sampler.full_epoch_num_batches * sampler.batch_size * world_size
    expected = (raw * ((total + n - 1) // n))[:total]
    ranks = [list(_shard(sampler, rank)) for rank in range(world_size)]
    reconstructed = [value for batches_at_step in zip(*ranks) for batch in batches_at_step for value in batch]
    assert reconstructed == expected
    assert list(sampler) == expected


def test_next_local_epoch_uses_saved_epoch_offset_without_reapplying_resume_skip():
    sampler = _sampler(10, epoch_offset=7, offset=2)
    first_rank = list(_shard(sampler, 1))
    assert first_rank == list(_shard(_sampler(10, epoch=7), 1))[2:]
    sampler.set_epoch(1)
    assert list(_shard(sampler, 1)) == list(_shard(_sampler(10, epoch=8), 1))
    assert len(sampler) == len(list(sampler)) == 12
    sampler.clear_resume_batch_offset()
    assert sampler.resume_batch_offset == 0


@pytest.mark.parametrize("n,world_size", [(10, 2), (1, 4)])
def test_exact_epoch_end_is_empty_and_normalizes_to_next_epoch_before_sharding(n, world_size):
    sampler = _sampler(n, world_size=world_size, epoch_offset=7)
    consumed = sampler.full_epoch_num_batches
    sampler.set_resume_batch_offset(consumed)
    assert len(sampler) == 0
    assert list(sampler) == []
    for rank in range(world_size):
        assert len(_shard(sampler, rank)) == 0
        assert list(_shard(sampler, rank)) == []

    # Resume integration should do this before constructing its first iterator:
    # an already-consumed epoch is checkpoint epoch+1, batch0. In Accelerate
    # 1.12 an empty DataLoaderShard returns before incrementing its iteration.
    # Normalizing progress also avoids version-specific empty-loader behavior.
    sampler.set_epoch_offset(8)
    sampler.clear_resume_batch_offset()
    for rank in range(world_size):
        loader = DataLoaderShard(range(n), batch_sampler=_shard(sampler, rank))
        batches = list(loader)
        assert all(batch is not None for batch in batches)
        expected = list(_shard(_sampler(n, world_size=world_size, epoch=8), rank))
        assert [batch.tolist() for batch in batches] == expected


def test_real_dataloader_shard_preserves_the_tail_after_resume():
    for rank in range(2):
        full_sampler = _sampler(10, epoch_offset=4)
        full = list(DataLoaderShard(range(10), batch_sampler=_shard(full_sampler, rank)))
        resumed_sampler = _sampler(10, epoch_offset=4, offset=2)
        tail = list(DataLoaderShard(range(10), batch_sampler=_shard(resumed_sampler, rank)))
        assert [batch.tolist() for batch in tail] == [batch.tolist() for batch in full[2:]]


@pytest.mark.parametrize("offset", [-1, 4, 100])
def test_invalid_resume_offsets_fail_before_any_iterator_is_created(offset):
    sampler = _sampler(10)
    with pytest.raises(ValueError, match="resume batch offset"):
        sampler.set_resume_batch_offset(offset)
    assert sampler.resume_batch_offset == 0


def test_empty_dataset_and_bad_global_batch_are_explicit():
    empty = _sampler(0)
    assert empty.full_epoch_num_batches == len(empty) == 0
    assert list(empty) == []
    with pytest.raises(ValueError, match="resume batch offset"):
        empty.set_resume_batch_offset(1)
    with pytest.raises(ValueError, match="positive"):
        _sampler(10, batch_size=0)


def test_legacy_default_keeps_its_existing_unpadded_length_contract():
    legacy = ResumableEpochSampler(range(10), seed=19, batch_size=2, num_processes=2)
    raw = torch.randperm(10, generator=torch.Generator().manual_seed(19)).tolist()
    assert list(legacy) == raw
    legacy.set_resume_batch_offset(2)
    assert list(legacy) == raw[8:]
    assert len(legacy) == 10  # compatibility for callers that did not opt in
