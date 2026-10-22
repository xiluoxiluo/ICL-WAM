import pytest
import torch
from torch import nn
from torch.nn import functional as F

from fastwam.zeva.task_context import TaskContextBank
from fastwam.zeva.task_language import (
    CLASSIFIER_FORMAT, FEATURE_FORMAT, TASK_LANGUAGE_FEATURE_KIND,
    FrozenTaskLanguageEncoder, TaskLanguageHead, TaskLanguageHeadConfig, TaskLanguageMemory,
    export_task_language_encoder, task_language_identity,
)
from fastwam.zeva.task_language_training import load_task_language_training_features, train_task_language_classifier
from fastwam.zeva.checkpoint import checkpoint_sha256


def tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    raw = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "pick": 2, "cup": 3, "open": 4, "drawer": 5}, unk_token="[UNK]"))
    raw.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="[PAD]", unk_token="[UNK]")


class IdentityHead(nn.Module):
    def __init__(self, dim=2):
        super().__init__()
        self.config = TaskLanguageHeadConfig(input_dim=dim, hidden_dim=4, output_dim=dim, dropout=0)
        self.linear = nn.Linear(dim, dim, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(dim))

    def forward(self, value):
        return F.normalize(self.linear(value), dim=-1)


def bank():
    return TaskContextBank([
        {"retrieval_key": torch.tensor(key), "behavior_value": torch.tensor(value),
         "episode_id": f"episode{index}", "task_id": "a" if index < 2 else "b", "split": "train"}
        for index, (key, value) in enumerate([([1., 0.], [3., 0.]), ([0.6, 0.8], [1., 2.]),
                                             ([0., 1.], [0., 5.]), ([-0.8, 0.6], [-2., 1.])])
    ], key_dim=2, value_dim=2, temperature=1.,
        metadata={"split": "train", "readout_kind": TASK_LANGUAGE_FEATURE_KIND,
                  "memory_semantics": "predicted_effect_trace"})


def test_frozen_task_input_mean_masks_padding_and_is_independent_of_source_table():
    table = torch.arange(12).reshape(6, 2).float().requires_grad_()
    encoder = FrozenTaskLanguageEncoder(table, tokenizer(), max_length=9)
    state = torch.get_rng_state().clone()
    actual = encoder.encode([" pick_cup\n", "open drawer"], device="cpu")
    torch.testing.assert_close(actual, torch.stack([table[2:4].mean(0), table[4:6].mean(0)]))
    assert not actual.requires_grad and not encoder.embedding_table.requires_grad
    assert torch.equal(state, torch.get_rng_state())
    with torch.no_grad():
        table.mul_(100)
    torch.testing.assert_close(encoder.encode("pick cup"), actual[:1])
    assert encoder.embedding_table.device.type == "cpu"
    with pytest.raises(ValueError, match="nonempty"):
        encoder.encode("   ")


def test_encoder_snapshot_moves_without_external_tokenizer_or_foundation(tmp_path):
    path = tmp_path / "encoder.pt"
    table = torch.randn(6, 4).bfloat16()
    export_task_language_encoder(path, table, tokenizer(), metadata={"source_tensor": "token_embedding.weight"})
    moved = tmp_path / "moved.pt"
    path.rename(moved)
    encoder = FrozenTaskLanguageEncoder.load(moved)
    expected = table[[2, 3]].float().mean(0)
    torch.testing.assert_close(encoder.encode("pick cup")[0], expected)
    assert encoder.embedding_table.dtype == torch.bfloat16
    payload = torch.load(moved, weights_only=False)
    payload["tokenizer_files"]["tokenizer.json"] += b" "
    torch.save(payload, moved)
    with pytest.raises(ValueError, match="tokenizer snapshot identity"):
        FrozenTaskLanguageEncoder.load(moved)


def test_language_predicted_task_scopes_both_mean_key_and_top_k():
    memory = TaskLanguageMemory(bank(), IdentityHead(), ["a", "b"], torch.eye(2), top_k=5)
    result = memory.retrieve(torch.eye(2))
    assert [item["predicted_task"] for item in result.sources] == ["a", "b"]
    assert [set(row[:2].tolist()) for row in result.indices] == [{0, 1}, {2, 3}]
    assert (result.indices[:, 2:] == -1).all()
    assert torch.isneginf(result.scores[:, 2:]).all()
    keys = torch.stack([entry["retrieval_key"] for entry in bank().entries[:2]])
    scores = F.normalize(keys.mean(0), dim=0) @ keys.T
    expected = (scores.softmax(-1).unsqueeze(-1) * torch.tensor([[3., 0.], [1., 2.]])).sum(0)
    torch.testing.assert_close(result.values[0], expected)
    memory.train()
    assert not memory.training and not memory.head.training
    assert not any(parameter.requires_grad for parameter in memory.parameters())


def test_language_classifier_scope_uses_declared_bank_and_never_ground_truth_task():
    # The classifier's highest-scoring task is not declared in this bank.
    memory = TaskLanguageMemory(bank(), IdentityHead(3), ["a", "b", "unavailable"], torch.eye(3))
    result = memory.retrieve(torch.tensor([[0.2, 0.1, 9.]]))
    assert result.sources[0]["predicted_task"] == "a"
    assert {item["task_id"] for item in result.sources[0]["top_k"]} == {"a"}
    with pytest.raises(TypeError):
        memory.retrieve(torch.zeros(1, 3), task_id="b")


def test_leave_one_episode_out_excludes_source_from_task_mean_and_values():
    memory = TaskLanguageMemory(bank(), IdentityHead(), ["a", "b"], torch.eye(2), top_k=5)
    result = memory.retrieve(torch.tensor([[1., 0.]]), exclude_episode_ids=["episode0"])
    assert result.indices[0].tolist() == [1, -1, -1, -1]
    torch.testing.assert_close(result.scores[0, 0], torch.tensor(1.))
    torch.testing.assert_close(result.values, torch.tensor([[1., 2.]]))
    assert result.sources[0]["top_k"][0]["episode_id"] == "episode1"


def test_classifier_trains_on_instruction_features_and_freezes_after_selection():
    torch.manual_seed(17)
    labels = torch.arange(2).repeat_interleave(8)
    features = F.one_hot(labels, 2).float() + torch.randn(16, 2) * 0.01
    keys = F.one_hot(labels, 2).float()
    head = TaskLanguageHead(TaskLanguageHeadConfig(2, 8, 2, 0))
    result = train_task_language_classifier(head, features, keys, labels, steps=50, batch_size=8,
                                           learning_rate=.02, seed=17, eval_every=10)
    assert result["best_task_top1"] == 1.
    assert not head.training and not any(p.requires_grad for p in head.parameters())
    assert not set(result["train_indices"]) & set(result["validation_indices"])
    assert set(result["train_indices"]) | set(result["validation_indices"]) == set(range(16))
    torch.testing.assert_close(result["task_prototypes"], torch.eye(2))


def feature_bundle(tmp_path):
    bank_path, encoder_path, features_path = [tmp_path / name for name in ("bank.pt", "encoder.pt", "features.pt")]
    source = bank()
    source.save(bank_path)
    export_task_language_encoder(encoder_path, torch.randn(6, 2), tokenizer())
    payload = {"format": FEATURE_FORMAT, "feature_kind": TASK_LANGUAGE_FEATURE_KIND,
               "bank_sha256": checkpoint_sha256(bank_path), "encoder_sha256": checkpoint_sha256(encoder_path),
               "episode_ids": [f"episode{i}" for i in range(6)], "task_ids": ["a", "a", "b", "b", "a", "b"],
               "instructions": ["pick cup"] * 6, "features": torch.randn(6, 2),
               "episode_splits": {"train": [f"episode{i}" for i in range(4)], "validation": ["episode4", "episode5"]},
               "metadata": {}}
    torch.save(payload, features_path)
    return payload, bank_path, encoder_path, features_path


def test_classifier_feature_loader_rejects_validation_trajectory_as_training_source(tmp_path):
    payload, bank_path, encoder_path, features_path = feature_bundle(tmp_path)
    load_task_language_training_features(features_path, bank_path, encoder_path)
    payload["episode_splits"]["train"].append("episode4")
    payload["episode_splits"]["validation"].remove("episode4")
    torch.save(payload, features_path)
    with pytest.raises(ValueError, match="CTE train/validation"):
        load_task_language_training_features(features_path, bank_path, encoder_path)


def test_classifier_checkpoint_requires_matching_frozen_encoder_and_train_only_split(tmp_path):
    _, bank_path, encoder_path, _ = feature_bundle(tmp_path)
    head_path = tmp_path / "head.pt"
    head = TaskLanguageHead(TaskLanguageHeadConfig(2, 8, 2, 0))
    payload = {"format": CLASSIFIER_FORMAT, "source_feature": "task_language", "feature_kind": TASK_LANGUAGE_FEATURE_KIND,
               "step": 10, "bank_sha256": checkpoint_sha256(bank_path), "encoder_sha256": checkpoint_sha256(encoder_path),
               "config": head.config.as_dict(), "model": head.state_dict(), "task_names": ["a", "b"],
               "task_prototypes": torch.eye(2), "train_episode_ids": ["episode0", "episode2"],
               "validation_episode_ids": ["episode1", "episode3"]}
    torch.save(payload, head_path)
    memory = TaskLanguageMemory.load(bank_path, head_path, encoder_path=encoder_path)
    assert memory.encoder.embedding_table.device.type == "cpu"
    identity = task_language_identity(bank_path, head_path, encoder_path)
    assert identity["mode"] == "task_language" and "base_checkpoint_sha256" not in identity
    memory.retrieve(memory.encoder.encode("pick cup"))
    payload["encoder_sha256"] = "wrong"
    torch.save(payload, head_path)
    with pytest.raises(ValueError, match="classifier/encoder"):
        TaskLanguageMemory.load(bank_path, head_path, encoder_path=encoder_path)
    payload["encoder_sha256"] = checkpoint_sha256(encoder_path)
    payload["train_episode_ids"].append("episode4")
    torch.save(payload, head_path)
    with pytest.raises(ValueError, match="CTE train split"):
        TaskLanguageMemory.load(bank_path, head_path, encoder_path=encoder_path)


def test_training_entrypoint_never_uses_cte_held_out_language_features(tmp_path):
    from omegaconf import OmegaConf
    from scripts.train_zeva_task_language import train

    payload, bank_path, encoder_path, features_path = feature_bundle(tmp_path)
    cfg = OmegaConf.create({"seed": 17, "device": "cpu", "model": {"zeva": {"task_context": {
        "mode": "task_language", "bank_path": str(bank_path), "language_encoder_path": str(encoder_path),
        "language_feature_cache_path": str(features_path), "retrieval_checkpoint": str(tmp_path / "head1.pt"),
        "retrieval": {"input_dim": 2, "hidden_dim": 8, "output_dim": 2, "dropout": 0.1},
        "training": {"steps": 5, "batch_size": 4, "eval_every": 5},
    }}}})
    result = train(cfg)
    first = torch.load(tmp_path / "head1.pt", weights_only=False)
    assert result["cte_validation_episodes_used_for_training"] == 0
    payload["features"][4:] = torch.randn_like(payload["features"][4:]) * 100
    torch.save(payload, features_path)
    cfg.model.zeva.task_context.retrieval_checkpoint = str(tmp_path / "head2.pt")
    train(cfg)
    second = torch.load(tmp_path / "head2.pt", weights_only=False)
    for name, value in first["model"].items():
        torch.testing.assert_close(second["model"][name], value, rtol=0, atol=0)
    assert set(first["train_episode_ids"] + first["validation_episode_ids"]) == set(payload["episode_splits"]["train"])
