"""RoboTwin decision-boundary RGB episodes shared by CTE training and caching.

The policy still learns 32-action/9-video-frame windows.  The CTE instead runs
at the actual 24-action replan cadence, with three independent camera images.
Images are read before RobotVideoDataset's camera mosaic/normalization.  Actions
use that dataset's existing action transforms and dataset-stat normalizer.
"""
from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from typing import Any, Iterable

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset


EGO_CAMERA_KEYS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
EGO_MEMORY_SEMANTICS = "predicted_effect_trace"


def prepare_ego_rgb_views(views: Iterable[Tensor], image_size: int = 224) -> Tensor:
    """Resize original RGB [3,H,W] views to [V,3,S,S], in [-1,1].

    Float inputs must be [0,1]; uint8 inputs are divided by 255.  This exact
    function is also used online so the CTE never sees a video mosaic or a
    different image-normalization convention at deployment.
    """
    output = []
    for image in views:
        image = torch.as_tensor(image)
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError("each CTE camera must be RGB [3,H,W]")
        image = image.float().div(255) if image.dtype == torch.uint8 else image.float()
        if not bool(torch.isfinite(image).all()) or float(image.min()) < 0 or float(image.max()) > 1.00001:
            raise ValueError("CTE source RGB must be uint8 or float in [0,1]")
        output.append(F.interpolate(image.unsqueeze(0), size=(image_size, image_size),
                                    mode="bilinear", align_corners=False, antialias=True)[0].mul(2).sub(1))
    if len(output) != 3:
        raise ValueError("Zeva-Ego CTE requires head, left-wrist and right-wrist views")
    return torch.stack(output)


def _scalar(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.flatten()[0].item()
    if hasattr(value, "reshape"):
        return value.reshape(-1)[0].item()
    return value


@dataclass(frozen=True)
class EgoEpisode:
    episode_id: str
    task_id: str
    episode_index: int
    dataset_start: int
    episode_length: int


class RoboTwinEgoEpisodeDataset(Dataset):
    """Read whole causal episodes without decoding images during indexing.

    ``base_dataset`` is the normal configured RobotVideoDataset.  Its global
    stride must be one and its action window at least the executed horizon.
    Episodes shorter than one full executed chunk are omitted. A partial final
    chunk is never padded into a complete target; its length is recorded.
    """
    def __init__(self, base_dataset: Any, *, executed_horizon: int = 24,
                 image_size: int = 224, action_dim: int = 14,
                 episode_ids: Iterable[str] | None = None) -> None:
        self.base_dataset = base_dataset
        self.source = base_dataset.lerobot_dataset
        if len(self.source.multi_dataset._datasets) != 1:
            raise ValueError("Zeva-Ego currently requires exactly one dataset directory for unambiguous policy episode IDs")
        self.executed_horizon = int(executed_horizon)
        self.image_size = int(image_size)
        self.action_dim = int(action_dim)
        self.policy_horizon = int(self.source.action_size)
        if min(self.executed_horizon, self.image_size, self.action_dim) < 1:
            raise ValueError("CTE dataset dimensions must be positive")
        if int(self.source.global_sample_stride) != 1 or self.policy_horizon < self.executed_horizon:
            raise ValueError("CTE episodes require raw stride 1 and a source action window >= executed_horizon")
        if self.source.processor is None:
            raise ValueError("CTE episodes require the policy dataset's action normalization processor")
        if bool(self.source.processor.use_stepwise_action_norm):
            raise ValueError("CTE needs position-independent action normalization for previous executed actions")
        cameras = tuple(meta["key"] for meta in self.source.image_meta)
        if cameras != EGO_CAMERA_KEYS:
            raise ValueError(f"CTE camera order must be {EGO_CAMERA_KEYS}, got {cameras}")
        self.camera_keys = cameras
        # Use private shallow reader copies with current-frame-only camera
        # queries. Decoding 33 future frames just to keep image[0] would make
        # full-episode CTE training unnecessarily expensive. Tensor tables and
        # video decoder infrastructure remain shared; policy dataset sampling
        # metadata is never mutated.
        self._raw_dataset = copy(self.source.multi_dataset)
        self._raw_dataset._datasets = []
        for original in self.source.multi_dataset._datasets:
            reader = copy(original)
            if getattr(original, "delta_indices", None) is not None:
                reader.delta_indices = dict(original.delta_indices)
                for meta in self.source.image_meta:
                    reader.delta_indices[meta["lerobot_key"]] = [0]
            if getattr(original, "delta_timestamps", None) is not None:
                reader.delta_timestamps = dict(original.delta_timestamps)
                for meta in self.source.image_meta:
                    reader.delta_timestamps[meta["lerobot_key"]] = [0.0]
            self._raw_dataset._datasets.append(reader)
        self.semantic_task_identity = base_dataset.semantic_task_identity
        allowed = None if episode_ids is None else set(map(str, episode_ids))
        episodes: list[EgoEpisode] = []
        global_offset = 0
        for inner in self.source.multi_dataset._datasets:
            starts, ends = inner.episode_data_index["from"], inner.episode_data_index["to"]
            ids = list(range(inner.meta.total_episodes)) if inner.episodes is None else list(inner.episodes)
            if len(ids) != len(starts) or len(starts) != len(ends):
                raise ValueError("LeRobot episode index metadata is inconsistent")
            start_indices = [int(value) for value in starts]
            table = inner.hf_dataset.with_format(None)
            tasks = table[start_indices]["task_index"] if start_indices and "task_index" in table.column_names else [0] * len(ids)
            for position, episode_index in enumerate(ids):
                length = int(ends[position]) - int(starts[position])
                if length <= self.executed_horizon:
                    continue
                identity = f"{inner.root}::episode-{int(episode_index)}"
                if allowed is not None and identity not in allowed:
                    continue
                task = base_dataset.resolve_semantic_task_id(episode_index=int(episode_index),
                                                             raw_task_index=tasks[position], strict=True)
                episodes.append(EgoEpisode(identity, str(task), int(episode_index),
                                           global_offset + int(starts[position]), length))
            global_offset += int(inner.num_frames)
        self.episodes = tuple(episodes)
        if not self.episodes:
            raise ValueError("no episodes contain a complete CTE execution chunk")
        if allowed is not None and allowed != {episode.episode_id for episode in self.episodes}:
            raise ValueError("requested CTE episode IDs are missing or too short")

    def __len__(self) -> int:
        return len(self.episodes)

    def _boundary(self, episode: EgoEpisode, raw_step: int) -> tuple[Tensor, Tensor, Tensor]:
        index = episode.dataset_start + raw_step
        # Deliberately bypass BaseLerobotDataset's random retry. A decode failure
        # must fail this episode, never silently substitute another trajectory.
        raw = self.source._split_lerobot_sample(self._raw_dataset[index])
        if "episode_index" not in raw or "frame_index" not in raw:
            raise ValueError("CTE source must expose episode_index and frame_index")
        if int(_scalar(raw["episode_index"])) != episode.episode_index or int(_scalar(raw["frame_index"])) != raw_step:
            raise ValueError("LeRobot substituted a different episode/frame during causal decoding")
        views = []
        for meta in self.source.image_meta:
            image = torch.as_tensor(raw[meta["lerobot_key"]])
            views.append(image[0] if image.ndim == 4 else image)
            pad = torch.as_tensor(raw[f"{meta['lerobot_key']}_is_pad"]).flatten()
            if bool(pad[0]):
                raise ValueError("CTE boundary image cannot be padded")
        images = prepare_ego_rgb_views(views, self.image_size)
        data = {"action": {}, "state": {}}
        for meta in self.source.action_meta:
            data["action"][meta["key"]] = self.source._get_action(meta, raw).clone()
        for meta in self.source.state_meta:
            data["state"][meta["key"]] = self.source._get_state(meta, raw).clone()
        processor = self.source.processor
        data = processor.action_state_transform(data)
        data = processor.normalizer.forward(data)
        data = processor.action_state_merger.forward(data)
        actions = torch.as_tensor(data["action"]).float()
        if actions.ndim != 2 or actions.shape[1] != self.action_dim or not bool(torch.isfinite(actions).all()):
            raise ValueError("CTE normalized action tensor must be finite [window,action_dim]")
        valid = ~torch.as_tensor(raw[f"{self.source.action_meta[0]['lerobot_key']}_is_pad"]).bool().flatten()
        return images, actions, valid

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode = self.episodes[int(index)]
        horizon = self.executed_horizon
        transitions = (episode.episode_length - 1) // horizon
        images, actions, masks = [], [], []
        for step in range(transitions + 1):
            rgb, normalized, valid = self._boundary(episode, step * horizon)
            images.append(rgb)
            if step < transitions:
                if normalized.shape[0] < horizon or valid.shape[0] < horizon or not bool(valid[:horizon].all()):
                    raise ValueError("a supposedly complete CTE execution chunk contains padding")
                actions.append(normalized[:horizon])
                masks.append(valid[:horizon])
        raw_steps = torch.arange(transitions + 1, dtype=torch.long) * horizon
        return {
            "images": torch.stack(images), "executed_actions": torch.stack(actions),
            "valid": torch.ones(transitions, dtype=torch.bool), "action_valid": torch.stack(masks),
            "raw_steps": raw_steps, "window_indices": raw_steps + episode.dataset_start,
            "policy_valid": raw_steps + self.policy_horizon < episode.episode_length,
            "decision_valid": raw_steps < episode.episode_length - 1,
            "episode_id": episode.episode_id, "task_id": episode.task_id,
            "episode_index": episode.episode_index, "episode_length": episode.episode_length,
            "trailing_action_count": (episode.episode_length - 1) % horizon,
        }


def collate_ego_episodes(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad batch tails explicitly; no padded action is considered executed."""
    if not samples:
        raise ValueError("cannot collate an empty episode batch")
    length = max(sample["executed_actions"].shape[0] for sample in samples)
    first, batch_size = samples[0], len(samples)
    images = first["images"].new_zeros((batch_size, length + 1, *first["images"].shape[1:]))
    actions = first["executed_actions"].new_zeros((batch_size, length, *first["executed_actions"].shape[1:]))
    valid = torch.zeros(batch_size, length, dtype=torch.bool)
    boundary_valid = torch.zeros(batch_size, length + 1, dtype=torch.bool)
    action_valid = torch.zeros(*actions.shape[:-1], dtype=torch.bool)
    result: dict[str, Any] = {"images": images, "executed_actions": actions, "valid": valid,
                              "boundary_valid": boundary_valid, "action_valid": action_valid}
    for key in ("raw_steps", "window_indices", "policy_valid", "decision_valid"):
        result[key] = first[key].new_zeros((batch_size, length + 1))
    for i, sample in enumerate(samples):
        count = sample["executed_actions"].shape[0]
        images[i, :count + 1] = sample["images"]
        images[i, count + 1:] = sample["images"][-1]
        actions[i, :count] = sample["executed_actions"]
        valid[i, :count] = sample["valid"]
        boundary_valid[i, :count + 1] = True
        action_valid[i, :count] = sample["action_valid"]
        for key in ("raw_steps", "window_indices", "policy_valid", "decision_valid"):
            result[key][i, :count + 1] = sample[key]
    for key in ("episode_id", "task_id", "episode_index", "episode_length", "trailing_action_count"):
        result[key] = [sample[key] for sample in samples]
    return result
