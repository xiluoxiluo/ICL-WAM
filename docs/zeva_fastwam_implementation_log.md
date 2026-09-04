# Zeva -> FastWAM/RoboTwin implementation log

This log records the read-only baseline checks and the final local validation
for the V1 addon. It intentionally does not claim a real RoboTwin rollout,
because this checkout has no dataset, base checkpoint, CUDA runtime, or SAPIEN
dependencies.

## Baseline audit

- Repository commit at audit time: `7faa71108368fbb3b6885649f112af607427a2d4`
- Branch: `main`
- Worktree: contains the Zeva implementation changes listed in the handoff;
  no unrelated reset/checkout was performed.
- Protected files checked unchanged:
  `src/fastwam/models/wan22/fastwam_joint.py`,
  `src/fastwam/models/wan22/fastwam_idm.py`, and
  `src/fastwam/models/wan22/fastwam_optional_idm.py`.
- Hydra regression: `configs/sim_robotwin_zeva.yaml` resolves to
  `model.zeva.enabled=true`, `EVALUATION.zeva_mode=pim_on`,
  `skip_get_obs_within_replan=false`, 33 raw sample frames, 9 aligned RGB
  frames, 32 actions, action dimension 14, and frequency ratio 4.

## Final local validation

```text
PYTHONPATH=src pytest -q                              29 passed
python -m compileall -q ...                           passed
git diff --check                                      passed
protected-file diff check                             unchanged
Hydra train/sim overrides + all addon --help          passed
```

The tests cover transition alignment, causal leakage, invalid/padded masks,
BIT/PIM lifecycle, deterministic retrieval/merge, cache manifests/source
indices, prompt masks, frozen-parameter whitelist, gate-zero equivalence, and
fixed-seed retry/finalization hooks. The final semantic audit also checked the
CTE loss defaults and cross-attempt PIM merge/provenance against the Zeva
reference implementation.

The final command audit confirmed that `train`, `sim_robotwin_zeva`, and the
fixed-attempt wrapper resolve the same `model.zeva` namespace.  A null training
`device` now falls back to CPU/CUDA detection instead of becoming the invalid
string device `"None"`.

## Remaining empirical gates

Before a full run, execute the Phase A-D smoke gates in the task plan with real
assets: one base action batch, CTE/cache construction, held-out effect and
action-shuffle checks, a 20-step Stage 2 backward test, gate-zero numerical
comparison, and a two-attempt fixed-seed RoboTwin task. These are intentionally
not marked complete by static tests.
