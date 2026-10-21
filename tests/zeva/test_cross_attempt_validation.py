from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from fastwam.trainer import Wan22Trainer


class _PairedValidationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.addon = nn.Linear(2, 2)
        self.seen = []

    def forward(self, sample):
        assert sample["_training_mode"] == "zeva_stage2"
        assert sample["video"].shape == (1, 3, 1, 8, 8)
        assert sample["task_context"].shape == (1, 8)
        assert sample["phase"].shape == (1, 4)
        assert sample["bit_mask"].shape == (1, 2)
        assert not self.addon.training
        self.seen.append(sample["pim_mask"].clone())
        noise = torch.rand(())
        loss = noise + sample["pim_mask"].any().float()
        return loss, {"loss_flow": float(loss), "behavior_prior_nll": 0.0}


def _sample():
    return {
        "video": torch.zeros(3, 1, 8, 8),
        "action": torch.zeros(32, 14),
        "task_context": torch.zeros(8),
        "phase": torch.zeros(4),
        "bit_effects": torch.zeros(2, 4),
        "bit_mask": torch.ones(2, dtype=torch.bool),
        "pim_phases": torch.ones(7, 4),
        "pim_effects": torch.ones(7, 4),
        "pim_mask": torch.ones(7, dtype=torch.bool),
    }


def _trainer(model, dataset, *, cfg=None, rank=0, world_size=1):
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.model = model
    trainer.zeva_training = True
    trainer.cfg = {} if cfg is None else cfg
    trainer.accelerator = SimpleNamespace(
        unwrap_model=lambda model: model,
        autocast=nullcontext,
        gather_for_metrics=lambda tensor: tensor,
        process_index=rank, num_processes=world_size, device=torch.device("cpu"),
    )
    trainer.val_dataset = dataset
    return trainer


def test_offline_validation_preserves_causal_fields_rng_and_mixed_train_modes():
    model = _PairedValidationModel().eval()
    model.addon.train()
    sample = _sample()
    trainer = _trainer(model, [sample], cfg={"eval_num_samples": 1})
    before_rng = torch.get_rng_state().clone()
    metrics = trainer.evaluate()
    assert metrics["val_memory_gain"] == pytest.approx(-1.0)
    assert len(model.seen) == 2
    assert model.seen[0].all() and not model.seen[1].any()
    assert sample["pim_mask"].all()
    assert not model.training and model.addon.training
    assert torch.equal(torch.get_rng_state(), before_rng)


class _IndexedDataset:
    def __init__(self, length):
        self.samples = [_sample() for _ in range(length)]
        for index, sample in enumerate(self.samples):
            sample["phase"].fill_(index)
        self.requested = []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        self.requested.append(index)
        torch.rand(())  # Model validation must also isolate dataset RNG use.
        return self.samples[index]


def test_default_validation_budget_spans_full_cache_deterministically_without_mutation():
    dataset = _IndexedDataset(101)
    originals = [{key: value.clone() for key, value in sample.items()} for sample in dataset.samples]
    model = _PairedValidationModel().eval()
    trainer = _trainer(model, dataset)
    before_rng = torch.get_rng_state().clone()
    first_metrics = trainer.evaluate()
    first_indices = list(dataset.requested)
    assert len(first_indices) == len(set(first_indices)) == 16
    assert first_indices[0] == 0 and first_indices[-1] == 100
    assert all(any(start <= index < start + 25 for index in first_indices) for start in (0, 25, 50, 75))
    dataset.requested.clear()
    second_metrics = trainer.evaluate()
    assert dataset.requested == first_indices
    assert second_metrics == first_metrics
    assert torch.equal(torch.get_rng_state(), before_rng)
    for sample, original in zip(dataset.samples, originals, strict=True):
        assert sample.keys() == original.keys()
        assert all(torch.equal(sample[key], value) for key, value in original.items())


class _IndexedValidationModel(_PairedValidationModel):
    def forward(self, sample):
        super().forward(sample)
        loss = sample["phase"][0, 0] + sample["pim_mask"].any().float()
        return loss, {"loss_flow": float(loss), "behavior_prior_nll": 0.0}


def test_distributed_validation_spreads_work_and_excludes_padding_from_means():
    requested = []
    for rank, expected_off_loss in enumerate((1.5, 2.5, 2.0)):
        dataset = _IndexedDataset(5)
        trainer = _trainer(
            _IndexedValidationModel().eval(), dataset,
            cfg={"eval_num_samples": 2}, rank=rank, world_size=3,
        )
        metrics = trainer.evaluate()
        assert metrics["val_loss_memory_off"] == pytest.approx(expected_off_loss)
        assert metrics["val_memory_gain"] == pytest.approx(-1.0)
        requested.append(dataset.requested)
    assert requested == [[0, 3], [1, 4], [2, 4]]
