"""Train and freeze Zeva-Ego-style task-language classification for FastWAM."""
from __future__ import annotations

import json
import os
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from fastwam.zeva.checkpoint import checkpoint_sha256
from fastwam.zeva.task_language import CLASSIFIER_FORMAT, TASK_LANGUAGE_FEATURE_KIND, TaskLanguageHead, TaskLanguageHeadConfig
from fastwam.zeva.task_language_training import load_task_language_training_features, train_task_language_classifier


def train(cfg: DictConfig) -> dict:
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Task-language classifier training uses one process; do not launch multi-rank torchrun")
    tc = cfg.model.zeva.task_context
    if tc.get("mode") != "task_language":
        raise ValueError("Task-language training requires task_context.mode=task_language")
    names = ("bank_path", "language_feature_cache_path", "language_encoder_path", "retrieval_checkpoint")
    paths = {}
    for name in names:
        if tc.get(name) in (None, "", "None", "null"):
            raise ValueError(f"Configure model.zeva.task_context.{name}")
        paths[name] = Path(str(tc[name])).expanduser().resolve()
    output = paths["retrieval_checkpoint"]
    if len(set(paths.values())) != len(paths) or output.exists():
        raise ValueError("Task-language classifier requires a distinct unused output path")
    cache, bank = load_task_language_training_features(paths["language_feature_cache_path"], paths["bank_path"],
                                                       paths["language_encoder_path"])
    train_ids = set(cache["episode_splits"]["train"])
    selected = [index for index, identity in enumerate(cache["episode_ids"]) if identity in train_ids]
    ids = [cache["episode_ids"][index] for index in selected]
    task_names = sorted({str(entry["task_id"]) for entry in bank.entries})
    entries = {entry["episode_id"]: entry for entry in bank.entries}
    features = cache["features"][selected]
    keys = torch.stack([entries[identity]["retrieval_key"] for identity in ids])
    labels = torch.tensor([task_names.index(str(entries[identity]["task_id"])) for identity in ids])
    settings = dict(tc.get("retrieval", {}))
    settings.setdefault("input_dim", features.shape[1])
    settings.setdefault("output_dim", keys.shape[1])
    config = TaskLanguageHeadConfig(**settings)
    if config.input_dim != features.shape[1] or config.output_dim != keys.shape[1]:
        raise ValueError("Task-language classifier dimensions differ from the frozen features/CTE key space")
    seed = int(cfg.get("seed", 1000))
    torch.manual_seed(seed)
    configured_device = cfg.get("device")
    device = str(configured_device) if configured_device not in (None, "", "None", "null") else "cuda" if torch.cuda.is_available() else "cpu"
    head = TaskLanguageHead(config).to(device)
    result = train_task_language_classifier(head, features, keys, labels, seed=seed, **dict(tc.training))
    payload = {
        "format": CLASSIFIER_FORMAT, "source_feature": "task_language", "feature_kind": TASK_LANGUAGE_FEATURE_KIND,
        "bank_sha256": checkpoint_sha256(paths["bank_path"]),
        "encoder_sha256": checkpoint_sha256(paths["language_encoder_path"]),
        "feature_cache_sha256": checkpoint_sha256(paths["language_feature_cache_path"]),
        "config": config.as_dict(), "model": {key: value.detach().cpu() for key, value in head.state_dict().items()},
        "task_names": task_names, "task_prototypes": result["task_prototypes"],
        "step": result["step"], "completed_steps": result["completed_steps"], "seed": seed,
        "training": dict(tc.training), "metrics": result["history"], "best_task_top1": result["best_task_top1"],
        "train_episode_ids": [ids[index] for index in result["train_indices"]],
        "validation_episode_ids": [ids[index] for index in result["validation_indices"]],
        "prototype_scope": "classifier_training_episodes_only",
        "training_objective": "task_prototype_cross_entropy", "source_sha256": checkpoint_sha256(__file__),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.building-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    report = {"checkpoint": str(output), "step": result["step"], "task_top1": result["best_task_top1"],
              "training_episodes": len(result["train_indices"]), "selection_episodes": len(result["validation_indices"]),
              "cte_validation_episodes_used_for_training": 0}
    print(json.dumps(report, sort_keys=True), flush=True)
    return report


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    train(cfg)


if __name__ == "__main__":
    main()
