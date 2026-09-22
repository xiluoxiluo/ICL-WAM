import json
from pathlib import Path
from collections import defaultdict, Counter

root = Path("runs/zeva_cache/phase_effect_v4")
rows = json.loads((root / "episode_index.json").read_text())

task_episodes = defaultdict(set)

for r in rows:
    task_episodes[str(r["task_id"])].add(str(r["episode_id"]))

counts = [len(v) for v in task_episodes.values()]
hist = Counter(counts)

print("tasks:", len(task_episodes))
print("episodes:", len({e for v in task_episodes.values() for e in v}))

print("\n========== episodes per task ==========")
print("min:", min(counts))
print("max:", max(counts))
print("mean:", sum(counts) / len(counts))

for n in [1, 2, 3, 4, 5, 10]:
    print(f"tasks with exactly {n} episodes:", hist[n])

print("tasks >= 2 episodes:", sum(x >= 2 for x in counts))
print("tasks >= 4 episodes:", sum(x >= 4 for x in counts))
print("tasks >= 10 episodes:", sum(x >= 10 for x in counts))

print("\nTop 20:")
for task, eps in sorted(
    task_episodes.items(),
    key=lambda x: len(x[1]),
    reverse=True
)[:20]:
    print(task, len(eps))