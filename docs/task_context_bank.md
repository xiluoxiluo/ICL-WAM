# FastWAM task-context bank

This page describes the **text-prototype baseline**. The implemented CTE behavior
bank and learned FastWAM retrieval path are documented in
[static_task_context.md](static_task_context.md).

`zeva.task_context.mode` selects the static task-context source used before the
causal prompt encoder:

```yaml
zeva:
  task_context:
    mode: bank
    bank_path: /path/to/fastwam_task_context.pt
    top_k: 1
    key_dim: 256
    value_dim: 256
    temperature: 0.07
```

The bank stores `retrieval_key` and `behavior_value` separately. The key is
compared with a masked pooled FastWAM text context; the value is passed to the
Zeva CausalPrompt as its global task feature. A bank is static and frozen.
Online BIT/PIM remains the source of causal interaction evidence.
`value_dim` must equal `zeva.prompt.global_dim` (the default config interpolates
it). `top_k` must be between one and the bank size. `temperature` is a build-time
setting persisted in the artifact; training and deployment read it from that
artifact. Use the same bank and `top_k` for Stage 2 and evaluation.

Build a deterministic bootstrap bank from the precomputed FastWAM text
contexts with:

```bash
PYTHONPATH=src python scripts/build_zeva_task_context_bank.py \
  --config-name train \
  task=robotwin_zeva_fastwam_3cam_384 \
  model.zeva.task_context.mode=bank \
  model.zeva.task_context.bank_path=/path/to/fastwam_task_context.pt
```

The selected data config must point to the training dataset, normalization
statistics, and precomputed FastWAM text embeddings. The builder groups samples
by `(task_id, instruction)` and streams their means. It uses the existing dataset
loader, including video/action loading, so a full build can take time.

This is a reproducible interface baseline, not a learned behavior bank. If each
instruction has an identical text embedding and retrieves itself at `top_k=1`,
the prototype is effectively the original pooling result. Adding a bank alone
does not establish a performance improvement or full Zeva equivalence.

To supply externally trained prototypes, construct and save the bank directly:

```python
from fastwam.zeva import TaskContextBank

bank = TaskContextBank(
    [{"retrieval_key": key_vector, "behavior_value": behavior_prototype,
      "task_id": task_id, "instruction": instruction}],
    key_dim=256, value_dim=256, temperature=0.07,
)
bank.save("/path/to/fastwam_task_context.pt", metadata={"source": "training demos"})
```

Each key/value must be a finite rank-1 vector with the declared dimensions.
Keys are normalized and retrieval uses cosine top-k followed by softmax-weighted
value aggregation. The bank owns frozen copies of these vectors. Serialized
artifacts include a versioned `format`, `config`, `entries`, and `metadata`.
`bank.retrieve(query)` also accepts already encoded queries; learned keys must
be paired with a matching query encoder. The built-in text helper only matches
keys produced in the pooled FastWAM context space. Cosmos keys are incompatible.

For learned retrieval of demonstrated behavior, use `mode=static` and follow
[the static task-context workflow](static_task_context.md). It has separate
bank construction and head training scripts, and is connected to Stage 2 and
deployment. Exclude evaluation episodes and labels from prototype construction.

Stage 2 uses the same bank retrieval path as deployment. The fixed-attempt
wrapper accepts `--task-context-bank` and enables bank mode for evaluation;
`--task-context-top-k` overrides its retrieval count (default config: 1).
Train the addon with bank mode before evaluating it with bank mode:

```bash
PYTHONPATH=src python scripts/train_zeva_fastwam.py \
  task=robotwin_zeva_fastwam_3cam_384 \
  +ckpt=/path/to/base.pt \
  model.zeva.cte.checkpoint=/path/to/cte.pt \
  model.zeva.cache.path=/path/to/cache \
  model.zeva.task_context.mode=bank \
  model.zeva.task_context.bank_path=/path/to/fastwam_task_context.pt
```

The default remains `mode=pooling` for existing checkpoints and ablations.
