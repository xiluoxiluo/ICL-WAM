"""Regression: splitting 27,500 episode banks must not clone N times N rows."""

import torch

from fastwam.zeva import main_data, task_context_data


def test_main_behavior_split_reads_entries_once(monkeypatch):
    class CountingBank:
        def __init__(self):
            self.calls = 0
            self.key_dim = 3
            self.value_dim = 4
            self.temperature = 0.07
            self.metadata = {
                "profile": "main_observed",
                "episode_splits": {
                    "train": ["e0", "e1", "e3"],
                    "validation": ["e2"],
                },
            }
            self._entries = [
                {
                    "episode_id": f"e{i}",
                    "task_id": str(i % 2),
                    "retrieval_key": torch.ones(3),
                    "behavior_value": torch.ones(4),
                }
                for i in range(4)
            ]

        @property
        def entries(self):
            self.calls += 1
            if self.calls > 1:
                raise AssertionError("quadratic cloning: bank.entries accessed more than once")
            return tuple(self._entries)

    class RecordingTrainBank:
        def __init__(self, entries, *, key_dim, value_dim, temperature, metadata):
            self.entries = list(entries)
            self.metadata = metadata

    monkeypatch.setattr(main_data, "validate_main_artifact", lambda _: None)
    monkeypatch.setattr(main_data, "validate_main_episode_ids", lambda *_: None)
    monkeypatch.setattr(task_context_data, "TaskContextBank", RecordingTrainBank)

    bank = CountingBank()
    readouts = torch.arange(20, dtype=torch.float32).view(4, 5)
    training_bank, training, validation = task_context_data.split_main_behavior_artifacts(bank, readouts)

    assert bank.calls == 1
    assert [e["episode_id"] for e in training_bank.entries] == ["e0", "e1", "e3"]
    assert training_bank.metadata["num_episodes"] == 3
    assert torch.equal(training, readouts[[0, 1, 3]])
    assert validation["validation_episode_ids"] == ["e2"]
    assert torch.equal(validation["validation_readouts"], readouts[[2]])
