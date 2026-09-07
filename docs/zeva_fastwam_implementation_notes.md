# Zeva FastWAM Implementation Notes

This repository implements the V1 Zeva causal-memory addon for the RoboTwin
FastWAM action path. The original FastWAM, Joint, IDM, and Optional-IDM model
files remain separate; the addon is attached by `fastwam.runtime.create_fastwam`
only when `zeva.enabled=true`.

## Causal contract

- A 32-step normalized FastWAM action window is represented as eight
  four-action transitions over nine RGB frames.
- The CTE module is a direct Zeva port with the generic full-history interface
  `[B,T,C,H,W]`, grouped actions `[B,T-1,4,14]`, and action-level validity
  `[B,T-1,4]`. The production Zeva/FastWAM contract uses an explicit frozen
  Wan-VAE adapter (Wan2.2 VAE38 `C=48`) and records its latent channel count;
  direct RGB (`C=3`) is reserved for an explicitly separate debug checkpoint.
  The action stream is right-shifted, so state `t` cannot see the transition
  beginning at `t`.
- Phase is emitted every four raw actions. Effects are emitted only after four
  completed transitions (16 raw actions); a 32-action window therefore yields
  eight transitions and two effect windows. There is no formal
  `initialize/update` or transition-level effect head.
- BIT is cleared at an attempt boundary. PIM entries are appended as soon as
  an effect is completed, so a later query in the same attempt can retrieve
  that completed interaction. PIM entries persist within an episode and merge
  phase/effect prototypes across attempts using a running mean/count;
  incomplete/future effects are never written.
- PIM candidates pair the phase at the effect-window start with its observed
  `effect_post`; action queries use the current phase. This matches the online
  lifecycle and offline cache proxy.
- Stage 2 uses only the action flow-matching objective. Video KV construction is
  frozen and no gradient is allowed through FastWAM or CTE parameters.
- Stage 1 uses the official Zeva CTE objective defaults (next-action,
  next-vision, effect, task, and phase terms). These are auxiliary CTE heads;
  Stage 2 still introduces no future-video, joint, or FastWAM-backbone loss.
- The CTE effect target is Zeva's deterministic frozen spatial projection
  (4x6 pooled grid) over the tensor supplied to CTE. It does not call the Wan
  VAE itself. Both online and cached observed effects use this target; RGB and
  latent contracts are kept separate by checkpoint/cache metadata.

## Intentional initialization detail

The output projection uses Xavier initialization and `tanh(alpha)` starts at
`alpha=0`. Their product is an exact no-op at initialization while preserving
an expressive residual direction. Training and evaluation use the same
gate-controlled formula; after the gate moves, gradients flow through the
adapter and prompt encoder normally.

## Cache and checkpoints

Phase/effect features are written as safetensors shards with a JSON manifest and
episode index (schema `v4` for repaired full-episode-prefix caches; `v3` remains
readable for diagnostics). Cache rows retain the source dataset index, raw query
step, episode start step, and effect index, so a dataset retry cannot silently
join a feature from another window.
Zeva dataset/cache builders reject a retry that changes the requested source
index and reject non-unit global sampling stride. Stage 1 consumes ordered,
non-overlapping windows for optimization; cache construction retains all
complete query positions and joins them into contiguous full-history segments.
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
