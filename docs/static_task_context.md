# FastWAM static task-context retrieval

Select `task=robotwin_zeva_fastwam_static_3cam_384` for the learned static task
prior. This is an end-to-end implementation path: build CTE demonstration
prototypes, extract clean FastWAM initial readouts, train the retrieval head,
train the addon with retrieved context, and retrieve at deployment.

## Mechanism and correspondence

| Component | Zeva released implementation | FastWAM implementation |
| --- | --- | --- |
| Query input | Initial observation and formatted instruction | Initial RoboTwin three-camera image and the same FastWAM prompt used by the text cache |
| Frozen readout | Mean final vision hidden state at time zero, without behavior tokens | Mean final video MoT hidden state at time zero, without actions, proprioception, or addon tokens |
| Retrieval head | Linear, LayerNorm, GELU, Dropout, Linear, L2 normalization | Same layers and parameter names; input width 3072 instead of Cosmos 4096 |
| Objective | Bidirectional multi-positive supervised contrastive loss | Same loss, with fixed CTE trajectory keys and training-task labels |
| Bank | Separate retrieval keys and behavior values | One entry per training demonstration, CTE key width 128 and state value width 256 by default |
| Retrieval | Cosine top-k, softmax weights, weighted values | Same operation; default K=5, clipped to available entries |
| Online lifecycle | Static task context from the initial observation, separate from BIT/PIM | One cached prior per attempt; reset on retries and new episodes |

The local Zeva release includes `extract_batch`, the retrieval head, loss, and
serving retrieval, but does not include its original bank construction script.
The FastWAM bank recipe is therefore explicit and versioned: normalize the
masked temporal mean of CTE `retrieval` as the key, and use the masked mean of
`causal_interaction_state` as the value. This follows the CTE task-clustering
space and keeps demonstrated behavior in the value. It is an adaptation of the
published mechanism, not evidence that the original bank construction
procedure or trained results have been reproduced identically.

Relevant upstream sources:

- [Static head, loss, and retrieval](https://github.com/air-embodied-brain/Zeva/blob/main/cosmos_framework/model/zeva/static_task_context_retrieval.py)
- [Clean initial-policy readout](https://github.com/air-embodied-brain/Zeva/blob/main/cosmos_framework/scripts/action_policy_server_robocasa365_zeva.py)
- [Initial-observation task context and PIM serving](https://github.com/air-embodied-brain/Zeva/blob/main/cosmos_framework/scripts/action_policy_server_robocasa365_zeva_pim.py)

## Build the behavior bank and initial readouts

Run commands from the ICLWAM repository with its dependencies installed. The
example assumes the existing RoboTwin data config points to your training data,
text embeddings, and normalization statistics. A trained FastWAM base and a
compatible trained CTE checkpoint are required. Real readout extraction uses
CUDA; the small CPU fixtures in the tests do not replace these checkpoints.

```bash
PYTHONPATH=src python scripts/build_zeva_behavior_bank.py \
  task=robotwin_zeva_fastwam_static_3cam_384 \
  ckpt=/path/to/base.pt \
  model.zeva.cte.checkpoint=/path/to/cte.pt \
  model.zeva.task_context.bank_path=/path/to/task_behavior_bank.pt \
  model.zeva.task_context.readout_cache_path=/path/to/initial_readouts.pt
```

The builder joins valid four-action boundaries from overlapping source windows,
deduplicates frames/actions, and rejects missing initial frames or gaps. Padding
does not contribute to prototypes. It runs CTE on the demonstrated trajectory,
but the policy readout sees only its initial frame and text. Both models are
frozen. Corpus-scale construction uses the current dataset loader, including
RGB/action decoding; it is an offline job.

The bank stores `retrieval_key`, `behavior_value`, `episode_id`, `task_id`,
`instruction`, and `num_transitions` per entry. Metadata includes the feature
spaces, training split, base/CTE/stats hashes, video size, context length, and
readout dimension. The initial-readout cache records the bank hash and exact
episode order. It contains no current evaluation rollout.

## Train the retrieval head

```bash
PYTHONPATH=src python scripts/train_zeva_task_context_retrieval.py \
  task=robotwin_zeva_fastwam_static_3cam_384 \
  model.zeva.task_context.bank_path=/path/to/task_behavior_bank.pt \
  model.zeva.task_context.readout_cache_path=/path/to/initial_readouts.pt \
  model.zeva.task_context.retrieval_checkpoint=/path/to/task_retrieval.pt
```

Only the MLP retrieval head is trained. Cached readouts, CTE keys, and behavior
values stay fixed. The trainer uses task-balanced batches and an episode-disjoint
validation split inside the training corpus. At least two tasks and two
demonstrations per task are required; one-task contrastive training is rejected.
Validation queries retrieve only from the training portion of this split.
`task_top1` measures retrieval of the correct task, not robot success.

Configuration lives under `model.zeva.task_context.training`: `steps`,
`batch_size`, `learning_rate`, `temperature`, `validation_fraction`, `eval_every`.
The best validation head is saved with its bank identity, training step, metrics,
and split episode IDs. CPU training of cached readouts is supported with
`device=cpu`; feature extraction still requires the real frozen FastWAM.

## Train the addon with static context

First prepare the existing Stage 2 phase/effect cache using the same CTE, data,
and normalization statistics. Then run:

```bash
PYTHONPATH=src python scripts/train_zeva_fastwam.py \
  task=robotwin_zeva_fastwam_static_3cam_384 \
  ckpt=/path/to/base.pt \
  model.zeva.cte.checkpoint=/path/to/cte.pt \
  model.zeva.cache.path=/path/to/phase_effect_cache \
  model.zeva.task_context.bank_path=/path/to/task_behavior_bank.pt \
  model.zeva.task_context.readout_cache_path=/path/to/initial_readouts.pt \
  model.zeva.task_context.retrieval_checkpoint=/path/to/task_retrieval.pt
```

Each episode's initial readout is queried once; every Stage 2 action window in
that episode receives the same static task context. Retrieval excludes that
episode's own demonstration entry so its future target trajectory cannot be
injected directly. Other training demonstrations remain eligible. Only the
existing CausalPrompt/BehaviorPrefixAdapter/gate are optimized in Stage 2.

The addon checkpoint binds the bank, head, and requested top-k. Loading an
addon with mismatched artifacts/settings or silently switching a static addon
to pooling fails explicitly. The original trained pooling addon must be
retrained for static task-context conditioning.

## Evaluate

```bash
PYTHONPATH=src python scripts/eval_zeva_robotwin_fixed_attempts.py \
  --task TASK_NAME --seed 0 --mode pim_on --max-attempts 4 \
  --ckpt /path/to/base.pt \
  --cte-checkpoint /path/to/cte.pt \
  --addon-checkpoint /path/to/static_addon.pt \
  --task-context-bank /path/to/task_behavior_bank.pt \
  --task-context-retrieval-checkpoint /path/to/task_retrieval.pt \
  --task-context-top-k 5
```

Deployment does not load the offline readout cache or receive a target task ID
for retrieval. It extracts the initial image/text readout and queries the frozen
head/bank. Later frames update phase/BIT/PIM through the existing causal path;
they do not change the cached task prior. A retry clears the static prior and
recomputes it from the reset observation; it retains PIM under existing rules.
All static artifacts and network parameters remain fixed during evaluation.

## Baselines

`task=robotwin_zeva_fastwam_3cam_384` still defaults to `mode=pooling`.
`mode=bank` is the earlier text-prototype baseline, using
`scripts/build_zeva_task_context_bank.py`, key width 256, and top-k 1 by default.
It is not accepted by the learned `static` loader. Its top-1 self match can
equal ordinary text pooling and does not establish a behavioral prior.

`TaskContextBank.retrieve()` remains a general key/value interface for already
encoded queries. Externally supplied keys must match their query encoder.
Cosmos readouts or checkpoints cannot be substituted into the FastWAM feature
space solely because tensor dimensions happen to match.
