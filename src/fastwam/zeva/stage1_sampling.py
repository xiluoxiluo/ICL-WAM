"""Deterministic, non-overlapping sampling helpers for Zeva Stage 1."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import random
from typing import Any, Iterable, Iterator, Sequence


@dataclass(frozen=True)
class CTETrainIndex:
    """Identity of one valid CTE source window in the base dataset."""

    dataset_index: int
    episode_id: str
    task_id: str | int
    episode_step: int


def _episode_value(episode: Any, name: str, default: Any = None) -> Any:
    if isinstance(episode, dict):
        return episode.get(name, default)
    return getattr(episode, name, default)


def _task_key(task_id: str | int) -> str:
    return str(task_id)


def _is_valid_window(sample: dict[str, Any]) -> bool:
    transition_valid = sample.get("transition_valid")
    if transition_valid is None:
        return False
    # A source row with padding cannot establish the next causal window. It
    # is still useful to stop the cursor, but must not be selected as a
    # training row because its temporal target is incomplete.
    frame_valid = sample.get("frame_valid")
    return bool(transition_valid.any()) and (
        frame_valid is None or bool(frame_valid.all())
    ) and bool(transition_valid.all())


def build_cte_training_index(
    dataset: Sequence[dict[str, Any]],
    source_window_actions: int = 32,
    sample_stride: int = 1,
) -> list[CTETrainIndex]:
    """Select deterministic non-overlapping windows before batching.

    The underlying RoboTwin loader exposes overlapping windows.  This pass
    owns the temporal cursor, so optimizer batching can never re-introduce a
    row at ``step + 1`` after a row at ``step`` has already been selected.
    Invalid/padded rows break the contiguous segment and do not advance the
    cursor.
    """

    source_window_actions = int(source_window_actions)
    sample_stride = int(sample_stride)
    if source_window_actions < 1 or sample_stride < 1:
        raise ValueError("source_window_actions and sample_stride must be positive")

    next_valid_step: dict[str, int] = {}
    rows: list[CTETrainIndex] = []
    for dataset_index, sample in enumerate(dataset):
        episode = sample.get("episode")
        if episode is None:
            raise ValueError("CTE samples must contain episode metadata")
        episode_id = str(_episode_value(episode, "episode_id", f"episode-{dataset_index}"))
        episode_step = int(_episode_value(episode, "episode_step", 0))
        expected = next_valid_step.get(episode_id)
        if expected is not None:
            if episode_step < expected:
                continue
            if episode_step != expected:
                # A gap means that the full causal prefix cannot be joined
                # safely; start a fresh segment at this valid row.
                next_valid_step.pop(episode_id, None)

        if not _is_valid_window(sample):
            next_valid_step.pop(episode_id, None)
            continue

        task_id = _episode_value(episode, "task_id", 0)
        if task_id in (None, 0, "0"):
            task_id = _episode_value(episode, "task_name", task_id)
        if not isinstance(task_id, (str, int)):
            task_id = str(task_id)
        rows.append(
            CTETrainIndex(
                dataset_index=int(dataset_index),
                episode_id=episode_id,
                task_id=task_id,
                episode_step=episode_step,
            )
        )
        next_valid_step[episode_id] = episode_step + source_window_actions * sample_stride
    return rows


def build_cte_query_index(
    dataset: Sequence[dict[str, Any]],
    sample_stride: int = 1,
    transition_steps: int = 4,
) -> list[CTETrainIndex]:
    """Index every complete source window eligible for a phase query.

    Unlike Stage 1 optimization, cache construction must expose deployment
    query positions such as raw steps 24/48. It therefore retains overlapping
    windows; the cache builder later deduplicates their shared boundaries and
    runs one full-episode CTE pass.
    """

    sample_stride = int(sample_stride)
    transition_steps = int(transition_steps)
    if sample_stride < 1 or transition_steps < 1:
        raise ValueError("sample_stride and transition_steps must be positive")
    rows: list[CTETrainIndex] = []
    for dataset_index, sample in enumerate(dataset):
        episode = sample.get("episode")
        if episode is None:
            raise ValueError("CTE samples must contain episode metadata")
        if not _is_valid_window(sample):
            continue
        episode_id = str(_episode_value(episode, "episode_id", f"episode-{dataset_index}"))
        episode_step = int(_episode_value(episode, "episode_step", 0))
        # CTE boundaries occur after complete transition groups. A dataset
        # window can start at every raw action step, but starts between these
        # boundaries cannot be represented by the grouped action stream.
        if episode_step % (transition_steps * sample_stride) != 0:
            continue
        task_id = _episode_value(episode, "task_id", 0)
        if task_id in (None, 0, "0"):
            task_id = _episode_value(episode, "task_name", task_id)
        if not isinstance(task_id, (str, int)):
            task_id = str(task_id)
        rows.append(
            CTETrainIndex(
                dataset_index=int(dataset_index),
                episode_id=episode_id,
                task_id=task_id,
                episode_step=episode_step,
            )
        )
    return rows


def _take_prefer_distinct_episode(
    pool: list[CTETrainIndex], count: int, rng: random.Random
) -> list[CTETrainIndex]:
    """Take rows while preferring distinct episodes for positive pairs."""

    if count > len(pool):
        raise ValueError("cannot take more rows than are available in a task pool")
    shuffled = list(pool)
    rng.shuffle(shuffled)
    selected: list[CTETrainIndex] = []
    used_episodes: set[str] = set()
    for row in shuffled:
        if row.episode_id not in used_episodes:
            selected.append(row)
            used_episodes.add(row.episode_id)
            if len(selected) == count:
                return selected
    selected.extend(row for row in shuffled if row not in selected)
    return selected[:count]


class TaskBalancedCTEBatchSampler:
    """Yield deterministic task-balanced batches from a CTE index.

    When the requested balance is impossible (for example a tiny debug
    dataset), remaining rows are emitted in ordinary shuffled batches instead
    of being silently dropped.  Production callers can inspect
    ``balanced_batches`` and fail their run if strict balance is required.
    """

    def __init__(
        self,
        rows: Iterable[CTETrainIndex],
        batch_size: int,
        *,
        tasks_per_batch: int | None = None,
        samples_per_task: int = 2,
        seed: int = 0,
    ) -> None:
        self.rows = tuple(rows)
        self.batch_size = int(batch_size)
        self.samples_per_task = int(samples_per_task)
        if self.batch_size < 1 or self.samples_per_task < 1:
            raise ValueError("batch_size and samples_per_task must be positive")
        if tasks_per_batch is None:
            tasks_per_batch = max(1, self.batch_size // self.samples_per_task)
        self.tasks_per_batch = int(tasks_per_batch)
        if self.tasks_per_batch < 1:
            raise ValueError("tasks_per_batch must be positive")
        if self.tasks_per_batch * self.samples_per_task > self.batch_size:
            raise ValueError("tasks_per_batch * samples_per_task exceeds batch_size")
        self.seed = int(seed)
        self.epoch = 0
        self.balanced_batches = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[CTETrainIndex]]:
        rng = random.Random(self.seed + self.epoch)
        pools: dict[str, list[CTETrainIndex]] = defaultdict(list)
        for row in self.rows:
            pools[_task_key(row.task_id)].append(row)
        available = {key for key, pool in pools.items() if len(pool) >= self.samples_per_task}
        batches: list[list[CTETrainIndex]] = []
        self.balanced_batches = 0

        while len(available) >= self.tasks_per_batch:
            task_keys = rng.sample(sorted(available), self.tasks_per_batch)
            batch: list[CTETrainIndex] = []
            for key in task_keys:
                pool = pools[key]
                selected = _take_prefer_distinct_episode(pool, self.samples_per_task, rng)
                batch.extend(selected)
                # Remove exactly the selected identities while preserving the
                # original row order for deterministic subsequent batches.
                pools[key] = [row for row in pool if row not in selected]
            for key in task_keys:
                if len(pools[key]) < self.samples_per_task:
                    available.discard(key)
            self.balanced_batches += 1
            batches.append(batch)

        leftovers = [row for pool in pools.values() for row in pool]
        rng.shuffle(leftovers)
        for start in range(0, len(leftovers), self.batch_size):
            batch = leftovers[start : start + self.batch_size]
            if batch:
                batches.append(batch)
        rng.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        # Balanced groups can be smaller than ``batch_size``. Reuse the same
        # deterministic plan as iteration so progress accounting reflects the
        # actual number of optimizer batches.
        return sum(1 for _ in self.__iter__())
