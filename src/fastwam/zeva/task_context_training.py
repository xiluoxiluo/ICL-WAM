"""Train only the static query head against frozen demonstration keys."""

from copy import deepcopy

import torch
from torch.nn import functional as F

from .static_task_context import bidirectional_supervised_contrastive_loss


def split_retrieval_episodes(semantic_ids, *, validation_fraction=0.2, seed=42):
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be in (0,1)")
    generator = torch.Generator().manual_seed(seed)
    tasks = semantic_ids.unique(sorted=True)
    if len(tasks) < 2:
        raise ValueError("contrastive retrieval training requires at least two tasks")
    train, validation = [], []
    for task in tasks:
        indices = (semantic_ids == task).nonzero().flatten()
        if len(indices) < 2:
            raise ValueError("each task needs at least two demonstrations for an episode-disjoint validation split")
        indices = indices[torch.randperm(len(indices), generator=generator)]
        count = min(len(indices) - 1, max(1, round(len(indices) * validation_fraction)))
        validation.extend(indices[:count].tolist())
        train.extend(indices[count:].tolist())
    return torch.tensor(train), torch.tensor(validation)


def train_retrieval_head(
    head, readouts, keys, semantic_ids, *, steps=2000, batch_size=32,
    learning_rate=1e-4, temperature=0.07, validation_fraction=0.2,
    seed=42, eval_every=50,
    validation_readouts=None, validation_semantic_ids=None,
):
    if steps < 1 or batch_size < 2 or eval_every < 1:
        raise ValueError("steps/eval_every must be positive and batch_size >= 2")
    if keys.ndim != 2 or readouts.ndim != 2 or keys.shape[0] != readouts.shape[0]:
        raise ValueError("readout/key training pairs must share their sample axis")
    if semantic_ids.shape != (len(keys),):
        raise ValueError("one semantic ID is required per demonstration")
    if not bool(torch.isfinite(readouts).all()) or not bool(torch.isfinite(keys).all()):
        raise ValueError("retrieval training pairs must be finite")
    external_validation = validation_readouts is not None
    if external_validation != (validation_semantic_ids is not None):
        raise ValueError("external validation requires both initial readouts and semantic IDs")
    if external_validation:
        if (validation_readouts.ndim != 2 or validation_readouts.shape[1] != readouts.shape[1]
                or validation_semantic_ids.shape != (len(validation_readouts),)
                or len(validation_readouts) < 1 or not bool(torch.isfinite(validation_readouts).all())
                or not set(validation_semantic_ids.tolist()) <= set(semantic_ids.tolist())):
            raise ValueError("invalid fixed held-out retrieval queries")
        train = torch.arange(len(readouts))
        validation = torch.arange(len(validation_readouts))
    else:
        train, validation = split_retrieval_episodes(semantic_ids, validation_fraction=validation_fraction, seed=seed)
    device = next(head.parameters()).device
    readouts, keys = readouts.detach().float().to(device), keys.detach().float().to(device)
    labels = semantic_ids.to(device)
    train_device, val_device = train.to(device), validation.to(device)
    if external_validation:
        validation_readouts = validation_readouts.detach().float().to(device)
        validation_semantic_ids = validation_semantic_ids.to(device)
    task_pools = [train[semantic_ids[train] == task] for task in semantic_ids.unique(sorted=True)]
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=0.0)
    best_score, best_loss, best_state, best_step = -1.0, float("inf"), None, 0
    history = []
    for step in range(1, steps + 1):
        head.train()
        # Every minibatch has negatives, and uses same-task positives when available.
        task_count = min(len(task_pools), max(2, batch_size // 2))
        task_order = torch.randperm(len(task_pools), generator=generator)[:task_count]
        chosen = []
        for task in task_order.tolist():
            pool = task_pools[task]
            count = min(len(pool), batch_size // task_count)
            chosen.extend(pool[torch.randperm(len(pool), generator=generator)[:count]].tolist())
        indices = torch.tensor(chosen, device=device)
        loss = bidirectional_supervised_contrastive_loss(
            head(readouts[indices]), keys[indices], labels[indices], temperature=temperature,
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("non-finite static retrieval training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % eval_every == 0 or step == steps:
            head.eval()
            with torch.no_grad():
                queries = head(validation_readouts if external_validation else readouts[val_device])
                target_labels = validation_semantic_ids if external_validation else labels[val_device]
                # Validation queries cannot retrieve their own held-out trajectory.
                scores = queries @ F.normalize(keys[train_device], dim=-1).T
                if external_validation:
                    # No held-out behavior prototype is a target/candidate.
                    # This query-to-training-bank diagnostic is not the
                    # symmetric paired loss used for actual optimizer steps.
                    positive = target_labels[:, None].eq(labels[train_device][None, :])
                    logits = scores / temperature
                    val_loss = (torch.logsumexp(logits, dim=-1)
                                - torch.logsumexp(logits.masked_fill(~positive, -torch.inf), dim=-1)).mean().item()
                else:
                    val_loss = bidirectional_supervised_contrastive_loss(
                        queries, keys[val_device], target_labels, temperature=temperature,
                    ).item()
                predicted = labels[train_device][scores.argmax(dim=-1)]
                accuracy = (predicted == target_labels).float().mean().item()
            history.append({"step": step, "loss": loss.item(), "validation_loss": val_loss, "task_top1": accuracy})
            print(f"retrieval step={step} loss={loss.item():.5f} val_loss={val_loss:.5f} task_top1={accuracy:.4f}")
            if accuracy > best_score or (accuracy == best_score and val_loss < best_loss):
                best_score, best_loss, best_step = accuracy, val_loss, step
                best_state = deepcopy(head.state_dict())
    head.load_state_dict(best_state)
    head.eval()
    return {"step": best_step, "history": history, "train_indices": train.tolist(),
            "validation_indices": validation.tolist(), "external_validation": external_validation}
