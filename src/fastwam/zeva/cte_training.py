"""Full observed-prefix training helpers for the main effect-v3 CTE.

The original CTE/loss are public, but its complete offline optimizer recipe is
not. Stitching the existing FastWAM windows and the local AdamW schedule are
explicit local training choices; they are not Zeva-Ego's predictive recipe.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Any

import torch
from torch.utils.data import Dataset

from .stage1_sampling import CTETrainIndex
from .cte_windows import observed_prefix_transitions


class MainCTEEpisodeDataset(Dataset):
    """Join real observed portions of H32 windows without resetting context.

    One training row represents the entire available contiguous episode prefix.
    The final source window may contain storage padding. Keep every truly
    completed four-control interval, including successful terminal segments,
    and trim unobserved frames/actions before the CTE sees the trajectory.
    """

    def __init__(self, base: Dataset, rows: list[CTETrainIndex], *, preencoded: bool) -> None:
        self.base, self.preencoded = base, bool(preencoded)
        grouped: dict[str, list[CTETrainIndex]] = defaultdict(list)
        for row in rows:
            grouped[str(row.episode_id)].append(row)
        self._rows_by_index: dict[int, list[CTETrainIndex]] = {}
        self.training_index: list[CTETrainIndex] = []
        for episode_id in sorted(grouped):
            sequence = sorted(grouped[episode_id], key=lambda row: row.episode_step)
            if [row.episode_step for row in sequence] != list(range(0, 32 * len(sequence), 32)):
                raise ValueError(f"main CTE requires complete contiguous episode history: {episode_id}")
            if len({str(row.task_id) for row in sequence}) != 1:
                raise ValueError("semantic task changes within a CTE episode")
            key = int(sequence[0].dataset_index)
            if key in self._rows_by_index:
                raise ValueError("duplicate main CTE episode source index")
            self._rows_by_index[key] = sequence
            self.training_index.append(sequence[0])

    def __len__(self) -> int:
        return len(self.training_index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        rows = self._rows_by_index[int(index)]
        frames_list, actions_list, frame_valid_list, transition_valid_list = [], [], [], []
        first = None
        previous_boundary = None
        for position, row in enumerate(rows):
            sample = self.base[int(row.dataset_index)]
            episode = sample["episode"]
            if (str(episode.episode_id) != str(row.episode_id)
                    or int(episode.episode_step) != int(row.episode_step)):
                raise ValueError("main CTE source row/episode mismatch")
            if first is None:
                first = sample
            frames = sample.get("cte_frames")
            if frames is None:
                frames = torch.cat((sample["before_frames"], sample["after_frames"][-1:]), dim=0)
            if frames.shape[0] != 9 or sample["transition_actions"].shape[:2] != (8, 4):
                raise ValueError("main episode stitching requires 9 frames/32 controls per source window")
            complete = observed_prefix_transitions(sample["frame_valid"], sample["transition_valid"])
            if complete < 1:
                raise ValueError("main CTE source window has no completed observed transition")
            if complete != 8 and position != len(rows) - 1:
                raise ValueError("only the final main CTE source window may have an incomplete tail")
            frames = frames[:complete + 1]
            frame_valid = sample["frame_valid"][:complete + 1]
            transition_valid = sample["transition_valid"][:complete]
            # Frozen framewise VAE encoding must give the same shared boundary.
            # BF16 cache roundoff is already encoded in both values identically.
            if previous_boundary is not None and not torch.allclose(
                previous_boundary.float(), frames[0].float(), rtol=1e-3, atol=1e-4,
            ):
                raise ValueError("adjacent main CTE windows disagree on their observed boundary")
            frames_list.append(frames if previous_boundary is None else frames[1:])
            frame_valid_list.append(frame_valid if previous_boundary is None else frame_valid[1:])
            actions_list.append(sample["transition_actions"][:complete])
            transition_valid_list.append(transition_valid)
            previous_boundary = frames[-1]
        assert first is not None
        result = dict(first)
        result.update({
            "cte_frames": torch.cat(frames_list, dim=0),
            "cte_frames_preencoded": self.preencoded,
            "transition_actions": torch.cat(actions_list, dim=0),
            "frame_valid": torch.cat(frame_valid_list, dim=0),
            "transition_valid": torch.cat(transition_valid_list, dim=0),
            "episode": replace(first["episode"], episode_step=0),
        })
        return result


def pad_cte_trajectories(
    frames: list[torch.Tensor], actions: list[torch.Tensor],
    frame_valid: list[torch.Tensor], transition_valid: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-pad variable-length trajectories; padding cannot become an effect."""
    if not frames or len({len(frames), len(actions), len(frame_valid), len(transition_valid)}) != 1:
        raise ValueError("a non-empty aligned CTE trajectory batch is required")
    maximum = max(value.shape[0] for value in frames)
    batch_frames = frames[0].new_zeros((len(frames), maximum, *frames[0].shape[1:]))
    batch_actions = actions[0].new_zeros((len(frames), maximum - 1, *actions[0].shape[1:]))
    batch_valid = torch.zeros((len(frames), maximum), dtype=torch.bool)
    batch_transition_valid = torch.zeros((len(frames), maximum - 1, actions[0].shape[1]), dtype=torch.bool)
    for i, (observations, controls, valid, completed) in enumerate(zip(frames, actions, frame_valid, transition_valid, strict=True)):
        length = observations.shape[0]
        if (controls.shape[0] != length - 1 or valid.shape != (length,)
                or completed.shape != controls.shape[:-1]):
            raise ValueError("CTE observation/control validity is not temporally aligned")
        batch_frames[i, :length] = observations
        batch_actions[i, :length - 1] = controls
        batch_valid[i, :length] = valid.bool()
        batch_transition_valid[i, :length - 1] = completed.bool()
    return batch_frames, batch_actions, batch_valid, batch_transition_valid
