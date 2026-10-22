"""Main feature inference stays separate from policy/master-weight precision."""

from copy import deepcopy

import pytest
import torch

from fastwam.zeva.feature_precision import main_feature_autocast
from fastwam.zeva.main_data import validate_main_artifact
from fastwam.zeva.static_task_context import (
    BEHAVIOR_KEY_SPACE, BEHAVIOR_VALUE_SPACE, READOUT_KIND,
    StaticTaskContextRetrievalConfig, StaticTaskContextRetrievalHead,
    StaticTaskContextRetriever,
)
from fastwam.zeva.task_context import TaskContextBank
from fastwam.zeva_numerical_contract import checkpoint_numerical_contract
from test_main_model_contract import main_metadata


def test_cpu_main_feature_context_does_not_inherit_ambient_amp_or_round_weights():
    torch.manual_seed(123)
    layer = torch.nn.Linear(7, 13).eval()
    original = deepcopy(layer.state_dict())
    inputs = torch.randn(2, 7)
    expected = layer(inputs)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with main_feature_autocast("cpu", enabled=True):
            actual = layer(inputs)
        with main_feature_autocast("cpu", enabled=False):
            legacy = layer(inputs)
    assert actual.dtype == torch.float32 and legacy.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for name, value in layer.state_dict().items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)


def test_main_static_query_and_fp32_scores_use_the_same_path_under_outer_amp(tmp_path):
    torch.manual_seed(17)
    metadata = main_metadata(tmp_path)
    metadata.update({"key_space": BEHAVIOR_KEY_SPACE, "value_space": BEHAVIOR_VALUE_SPACE,
                     "readout_kind": READOUT_KIND, "readout_dim": 7, "split": "train"})
    entries = [{"episode_id": episode_id, "task_id": "task",
                "retrieval_key": torch.randn(5), "behavior_value": torch.randn(4)}
               for episode_id in metadata["episode_splits"]["train"]]
    bank = TaskContextBank(entries, key_dim=5, value_dim=4, metadata=metadata)
    head = StaticTaskContextRetrievalHead(StaticTaskContextRetrievalConfig(7, 16, 5, 0))
    retriever = StaticTaskContextRetriever(bank, head, top_k=2)
    inputs = torch.randn(2, 7)
    expected = retriever.retrieve(inputs)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        query = retriever.query(inputs)
        actual = retriever.retrieve(inputs)
    assert query.dtype == actual.scores.dtype == actual.values.dtype == torch.float32
    assert not query.requires_grad and not any(p.requires_grad for p in head.parameters())
    torch.testing.assert_close(actual.indices, expected.indices, rtol=0, atol=0)
    torch.testing.assert_close(actual.scores, expected.scores, rtol=0, atol=0)
    torch.testing.assert_close(actual.values, expected.values, rtol=0, atol=0)


@pytest.mark.parametrize("field,value", [
    ("artifact_lineage", "zeva_main_observed_effect_v1"),
    ("feature_precision", "float32"),
    ("reference_commit", "0" * 40),
])
def test_old_or_unverified_main_features_cannot_be_relabelled_as_v2(tmp_path, field, value):
    metadata = main_metadata(tmp_path)
    metadata[field] = value
    with pytest.raises(ValueError, match=field):
        validate_main_artifact(metadata)


def test_main_v1_addon_numerics_are_rejected_instead_of_silently_migrated():
    with pytest.raises(ValueError, match="numerical_contract"):
        checkpoint_numerical_contract({"numerical_contract": "zeva_main_v1"})
