# Zeva FastWAM Implementation Notes

This repository implements the V1 Zeva causal-memory addon for the RoboTwin
FastWAM action path. The original FastWAM, Joint, IDM, and Optional-IDM model
files remain separate; the addon is attached by `fastwam.runtime.create_fastwam`
only when `zeva.enabled=true`.

## Causal contract

- A 32-step normalized FastWAM action window is represented as eight
  four-action transitions over nine RGB frames.
- CTE `initialize()` consumes only the current frame. `update()` consumes the
  executed four-action group and its observed after-frame, returning the next
  state, next phase, and one observed effect. The CTE also exposes a separate
  pre-transition effect prediction used only for the auxiliary causal loss;
  the observed effect is the feature written to BIT/PIM after completion.
- BIT is cleared at an attempt boundary. PIM entries persist within an episode
  and merge phase/effect prototypes across attempts using a running
  mean/count. The merged entry is tagged with the latest attempt, so retrieval
  excludes the active attempt and exposes the accumulated prototype from the
  next attempt onward.
- PIM candidates are keyed by the post-transition phase paired with its
  observed effect; an action query uses the current pre-transition phase. This
  matches the online update order and the offline cache proxy.
- Stage 2 uses only the action flow-matching objective. Video KV construction is
  frozen and no gradient is allowed through FastWAM or CTE parameters.
- Stage 1 uses the official Zeva CTE objective defaults (next-action,
  next-vision, effect, task, and phase terms). These are auxiliary CTE heads;
  Stage 2 still introduces no future-video, joint, or FastWAM-backbone loss.
- The CTE effect target is a deterministic frozen RGB spatial projection (4x6
  pooled grid), not a Wan VAE feature. Both online and cached observed effects
  use this target; the EMA visual stem is retained only for phase/visual-key
  representation. This keeps the CTE independent of the large FastWAM
  backbone; it is an implementation choice that must be checked by held-out
  effect and action-shuffle sanity tests before full training.

## Intentional initialization detail

The design calls for both a zero output projection and `tanh(alpha)` with
`alpha=0`. Their product is an exact no-op at initialization. During training
only, a small `train_gate_epsilon` (default `1e-3`) lets the zero-initialized
output projection receive gradients; evaluation and `pim_shadow` use the exact
gate-controlled path, so gate zero is numerically identical to base FastWAM.

## Cache and checkpoints

Phase/effect features are written as safetensors shards with a JSON manifest and
episode index (schema `v2`). Cache rows retain the source dataset index and episode start
step, so a dataset retry cannot silently join a feature from another window.
Zeva dataset/cache builders reject a retry that changes the requested source
index, reject non-unit global sampling stride, and consume only ordered,
non-overlapping windows with recurrent CTE state handoff.
The manifest records CTE hash, dataset stats hash, camera order,
action normalization, dimensions, and schema version. CTE and addon checkpoints
are separate; addon loading can require matching base and CTE SHA256 values.
Stage 1 additionally records the resolved config, dataset manifest, and metrics
JSONL; `resume=<path/to/cte.pt>` restores CTE optimizer/scheduler state. Stage 2
state directories contain only addon optimizer/scheduler tensors (not a second
copy of the frozen FastWAM weights) and can resume with the same strict hashes.

## Commands

```bash
PYTHONPATH=src python scripts/build_zeva_robotwin_transitions.py --config-name train task=robotwin_zeva_fastwam_3cam_384
PYTHONPATH=src python scripts/train_zeva_cte.py --config-name train task=robotwin_zeva_fastwam_3cam_384
PYTHONPATH=src python scripts/build_zeva_robotwin_cache.py --config-name train task=robotwin_zeva_fastwam_3cam_384 model.zeva.cte.checkpoint=/path/to/cte.pt model.zeva.cache.path=/path/to/cache
PYTHONPATH=src python scripts/train_zeva_fastwam.py --config-name train task=robotwin_zeva_fastwam_3cam_384 ckpt=/path/to/base.pt model.zeva.cte.checkpoint=/path/to/cte.pt model.zeva.cache.path=/path/to/cache
PYTHONPATH=src python scripts/eval_zeva_robotwin_fixed_attempts.py --task TASK --ckpt /path/to/base.pt --cte-checkpoint /path/to/cte.pt --addon-checkpoint /path/to/addon.pt --seed 0
```

The fixed-attempt entry point passes the selected Zeva mode and lifecycle
settings to the RoboTwin policy. The policy exposes `begin_attempt(attempt_id)`
for evaluators that explicitly retry the same fixed episode.
