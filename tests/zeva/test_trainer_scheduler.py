import math

import torch

from fastwam.trainer import Wan22Trainer


def _make_scheduler(
    optimizer,
    *,
    scheduler_type="cosine",
    total_train_steps=100,
    warmup_steps=0,
):
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.optimizer = optimizer
    trainer.learning_rate = 1e-4
    return trainer._build_scheduler(
        scheduler_type=scheduler_type,
        total_train_steps=total_train_steps,
        warmup_steps=warmup_steps,
    )


def _pim_optimizer():
    params = [torch.nn.Parameter(torch.zeros(1)) for _ in range(3)]
    optimizer = torch.optim.AdamW(
        [
            {"params": [params[0]], "lr": 5e-4},
            {"params": [params[1]], "lr": 5e-4},
            {"params": [params[2]], "lr": 1e-4},
        ]
    )
    return optimizer


def test_pim_lr_ratio_is_preserved_by_cosine():
    optimizer = _pim_optimizer()
    scheduler = _make_scheduler(optimizer)
    for step in range(101):
        if step in {0, 1, 10, 25, 50, 75, 99, 100}:
            lrs = [group["lr"] for group in optimizer.param_groups]
            assert math.isclose(lrs[0] / lrs[2], 5.0, rel_tol=1e-6, abs_tol=1e-8)
            assert math.isclose(lrs[1] / lrs[2], 5.0, rel_tol=1e-6, abs_tol=1e-8)
        if step < 100:
            optimizer.step()
            scheduler.step()


def test_cosine_final_lr_is_one_percent_per_group():
    optimizer = _pim_optimizer()
    scheduler = _make_scheduler(optimizer)
    for _ in range(100):
        optimizer.step()
        scheduler.step()
    lrs = [group["lr"] for group in optimizer.param_groups]
    assert math.isclose(lrs[0], 5e-6, rel_tol=1e-5)
    assert math.isclose(lrs[1], 5e-6, rel_tol=1e-5)
    assert math.isclose(lrs[2], 1e-6, rel_tol=1e-5)


def test_pim_lr_ratio_is_preserved_through_warmup():
    optimizer = _pim_optimizer()
    scheduler = _make_scheduler(optimizer, total_train_steps=100, warmup_steps=10)
    for _ in range(100):
        lrs = [group["lr"] for group in optimizer.param_groups]
        assert math.isclose(lrs[0] / lrs[2], 5.0, rel_tol=1e-6, abs_tol=1e-8)
        optimizer.step()
        scheduler.step()
