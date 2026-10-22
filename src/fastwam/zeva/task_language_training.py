"""Rebuild the frozen language classifier from CTE-train instructions only.

Zeva-Ego's published policy consumes a previously trained classifier from its
handoff. Its pi0.5 embedding weights are incompatible with FastWAM; this local
recipe trains the same normalized MLP/prototype classifier in Wan's frozen
input-embedding space, with an internal episode-disjoint selection split.
"""
from __future__ import annotations

from copy import deepcopy
import math

import torch
from torch import Tensor
from torch.nn import functional as F

from .checkpoint import checkpoint_sha256
from .task_context import TaskContextBank
from .task_context_training import split_retrieval_episodes
from .task_language import FEATURE_FORMAT, TASK_LANGUAGE_FEATURE_KIND, validate_task_language_bank


def load_task_language_training_features(path, bank_path, encoder_path) -> tuple[dict, TaskContextBank]:
    bank = TaskContextBank.load(bank_path)
    validate_task_language_bank(bank)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != FEATURE_FORMAT or payload.get("feature_kind") != TASK_LANGUAGE_FEATURE_KIND:
        raise ValueError("Classifier training requires frozen task-only input-embedding features")
    for key, expected in (("bank_sha256", checkpoint_sha256(bank_path)),
                          ("encoder_sha256", checkpoint_sha256(encoder_path))):
        if payload.get(key) != expected:
            raise ValueError(f"Task-language feature artifact identity mismatch: {key}")
    splits = payload.get("episode_splits", {})
    train, validation = set(splits.get("train", [])), set(splits.get("validation", []))
    bank_tasks = {entry["episode_id"]: str(entry["task_id"]) for entry in bank.entries}
    if not train or not validation or train & validation or train != set(bank_tasks):
        raise ValueError("Language features must preserve the CTE train/validation split and train-only bank")
    ids, tasks, instructions, features = (payload.get(key) for key in ("episode_ids", "task_ids", "instructions", "features"))
    if (not isinstance(ids, list) or len(set(ids)) != len(ids) or set(ids) != train | validation
            or not isinstance(tasks, list) or not isinstance(instructions, list)
            or len(tasks) != len(ids) or len(instructions) != len(ids)):
        raise ValueError("Task-language feature rows must uniquely cover every source query episode")
    if (not isinstance(features, Tensor) or features.dtype != torch.float32 or features.ndim != 2
            or features.shape[0] != len(ids) or not bool(torch.isfinite(features).all())):
        raise ValueError("Task-language features must be finite float32 [episode,feature]")
    if any(not isinstance(value, str) or not value.strip() for value in instructions):
        raise ValueError("Task-language feature rows require raw nonempty instructions")
    if any(identity in train and bank_tasks[identity] != str(task) for identity, task in zip(ids, tasks, strict=True)):
        raise ValueError("Task-language training labels do not match the training memory")
    for key in ("cte_checkpoint_sha256", "phase_cache_manifest_sha256", "episode_splits_sha256",
                "dataset_stats_sha256", "semantic_task_map_sha256"):
        if payload.get("metadata", {}).get(key) != bank.metadata.get(key):
            raise ValueError(f"Task-language feature/bank lineage mismatch: {key}")
    return payload, bank


def train_task_language_classifier(head, features: Tensor, keys: Tensor, labels: Tensor, *,
                                   steps: int = 2000, batch_size: int = 32, learning_rate: float = 1e-4,
                                   temperature: float = 0.07, validation_fraction: float = 0.2,
                                   seed: int = 1000, eval_every: int = 50, weight_decay: float = 1e-4) -> dict:
    """Cross-entropy to frozen task prototypes, selected by held-out top-1.

    ``features/keys/labels`` must already contain only CTE training episodes.
    The internal selection episodes never contribute to classifier prototypes.
    The 0.07 temperature is only a classifier training loss scale; retrieval
    aggregation always uses reference temperature 1.0.
    """
    if steps < 1 or batch_size < 2 or eval_every < 1:
        raise ValueError("Classifier steps/eval_every must be positive and batch_size >= 2")
    if not math.isfinite(learning_rate) or learning_rate <= 0 or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Classifier learning rate and temperature must be finite and positive")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("Classifier weight_decay must be finite and nonnegative")
    if (features.ndim != 2 or keys.ndim != 2 or len(features) != len(keys) or labels.shape != (len(keys),)
            or not bool(torch.isfinite(features).all() and torch.isfinite(keys).all())):
        raise ValueError("Classifier expects finite aligned features/keys and one task label per episode")
    task_ids = labels.unique(sorted=True)
    if not torch.equal(task_ids.cpu(), torch.arange(len(task_ids))):
        raise ValueError("Classifier labels must index a contiguous predeclared task table")
    train, validation = split_retrieval_episodes(labels, validation_fraction=validation_fraction, seed=seed)
    prototypes = F.normalize(torch.stack([keys[train[labels[train] == task]].mean(0) for task in task_ids]), dim=-1)
    if bool((prototypes.norm(dim=-1) <= 0).any()) or head.config.output_dim != keys.shape[1]:
        raise ValueError("Classifier output must match finite nonzero task prototypes")
    device = next(head.parameters()).device
    features, labels, prototypes = features.detach().float().to(device), labels.to(device), prototypes.detach().float().to(device)
    validation_device = validation.to(device)
    pools = [train[(labels.cpu()[train] == task.cpu())] for task in task_ids]
    generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_state, best_step, best_accuracy, best_loss, history = None, 0, -1.0, float("inf"), []
    for step in range(1, int(steps) + 1):
        head.train()
        selected = []
        # Balanced classes, including negatives even for a small microbatch.
        order = torch.randperm(len(pools), generator=generator)[:min(len(pools), batch_size)]
        per_task = max(1, batch_size // len(order))
        for task in order.tolist():
            pool = pools[task]
            sample = pool[torch.randperm(len(pool), generator=generator)[:min(per_task, len(pool))]]
            selected.extend(sample.tolist())
        selected = torch.tensor(selected, device=device)
        logits = head(features[selected]) @ prototypes.T / temperature
        loss = F.cross_entropy(logits, labels[selected])
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite task-language classifier loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % eval_every == 0 or step == steps:
            head.eval()
            with torch.no_grad():
                scores = head(features[validation_device]) @ prototypes.T
                accuracy = (scores.argmax(-1) == labels[validation_device]).float().mean().item()
                validation_loss = F.cross_entropy(scores / temperature, labels[validation_device]).item()
            history.append({"step": step, "loss": float(loss.detach()), "validation_loss": validation_loss,
                            "task_top1": accuracy})
            if accuracy > best_accuracy or (accuracy == best_accuracy and validation_loss < best_loss):
                best_step, best_accuracy, best_loss = step, accuracy, validation_loss
                best_state = deepcopy(head.state_dict())
    head.load_state_dict(best_state, strict=True)
    head.eval().requires_grad_(False)
    return {"step": best_step, "completed_steps": int(steps), "history": history,
            "train_indices": train.tolist(), "validation_indices": validation.tolist(),
            "task_prototypes": prototypes.cpu(), "best_task_top1": best_accuracy}
