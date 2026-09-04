# ICLWAM：FastWAM + Zeva 因果记忆

本目录是在 FastWAM RoboTwin action path 上增加 Zeva causal memory 的实验实现。原有 FastWAM、Joint、IDM 和 Optional-IDM 模块仍按原逻辑工作；只有选择 `model=zeva_fastwam`（或评测配置 `sim_robotwin_zeva.yaml`）并设置 `zeva.enabled=true` 时，才会挂载 CTE、BIT/PIM、CausalPromptEncoder 和 BehaviorPrefixAdapter。

FastWAM 原始的 LIBERO/RoboTwin 安装、数据下载和基线说明见 [`docs/README_newzh.md`](docs/README_newzh.md)；本文只补充 ICLWAM/Zeva 新增部分以及可复现实验的完整串联方式。

下面的命令都假设在 `ICLWAM/` 根目录执行：

```bash
cd /path/to/ICL-WAM/ICLWAM
export PYTHONPATH="$PWD/src"
export DIFFSYNTH_MODEL_BASE_PATH="$PWD/checkpoints"
```

## 1. 实验流程概览

Zeva 实验不是用一个新 checkpoint 替换 FastWAM，而是四个阶段串联：

1. 准备 RoboTwin 数据、`dataset_stats.json`、Wan/ActionDiT 预训练文件和 T5 文本 embedding cache。
2. （可选）将数据导出为带 episode 顺序的 32-action/8-transition 视图。
3. Stage 1 只训练 Causal Transition Encoder（CTE），得到 `cte.pt`。
4. 用冻结的 CTE 生成 phase/effect cache，再进行 Stage 2；Stage 2 只训练 Zeva addon，FastWAM 和 CTE 都冻结。
5. 评测时将 FastWAM 基座、CTE 和 Stage 2 addon 组合起来，并用固定 seed 比较 `base`、`pim_shadow` 和 `pim_on`。

`pim_shadow` 会运行完整的因果生命周期但不加载 addon，用于检查 memory/retrieval 是否改变；`pim_on` 才会把训练得到的 memory prefix 注入 FastWAM action path。

## 2. 环境和数据

### 2.1 Python/CUDA 环境

生产训练和 RoboTwin 仿真建议使用 Linux、CUDA GPU、Python 3.10。项目依赖（包括与 CUDA 对应的 PyTorch、DeepSpeed、torchcodec 和 torchvision）写在 `pyproject.toml` 中：

```bash
conda create -n iclwam python=3.10 -y
conda activate iclwam
python -m pip install -U pip
python -m pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
  --extra-index-url https://download.pytorch.org/whl/cu128
python -m pip install -e .
python - <<'PY'
import torch
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
PY
```

Stage 1 可以用 `device=cpu` 做小规模数据/逻辑调试；Stage 2 的冻结 Wan2.2 action path 明确要求 CUDA。CPU 只能作为 smoke test，不能作为完整训练或 RoboTwin rollout 环境。

### 2.2 Wan 和 ActionDiT 文件

Wan2.2-TI2V-5B、Wan2.1 tokenizer、T5 和 VAE 文件会由 FastWAM loader 下载或从 `DIFFSYNTH_MODEL_BASE_PATH` 读取。Stage 2/评测构造模型时还会读取 ActionDiT 初始化权重，默认路径是：

```text
checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
```

如果该文件尚未生成，先执行：

```bash
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/fastwam.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda --dtype bfloat16
```

也可以通过 Hydra 覆盖 `model.action_dit_pretrained_path=/absolute/path/to/file.pt`。不要在没有完整 base checkpoint 的情况下使用 `model.skip_dit_load_from_pretrain=true`，否则模型初始化不会得到所需的 ActionDiT 参数。

### 2.3 RoboTwin 数据和仿真 assets

当前 Zeva 配置固定使用三路相机和 14 维 RoboTwin action/state。默认路径为：

```text
data/robotwin2.0/robotwin2.0/       # LeRobot 数据集，包含 data/meta/videos
data/robotwin2.0/dataset_stats.json # 所有阶段共用的 z-score 统计
third_party/RoboTwin/               # vendored RoboTwin 代码
```

可以使用 FastWAM 发布的 RoboTwin 数据，或替换 `configs/data/robotwin.yaml` 中的 `dataset_dirs`。自定义数据必须提供 `meta/tasks.jsonl`、对应的 episode/video 文件和同一数据集计算出的 `dataset_stats.json`；Zeva 的三个构建/训练入口会拒绝不存在的 stats 文件。

RoboTwin rollout 还需要按其官方教程安装 SAPIEN、Curobo、pytorch3d 并下载 assets。vendored 安装脚本位于 `third_party/RoboTwin/script/_install.sh` 和 `third_party/RoboTwin/script/_download_assets.sh`。评测入口会自动创建 `third_party/RoboTwin/policy/fastwam_policy` 软链接；若直接调用官方 `script/eval_policy.py`，需要手动建立该链接。

### 2.4 T5 文本 embedding cache

`configs/data/robotwin.yaml` 的 `text_embedding_cache_dir` 默认是 `data/text_embeds_cache/robotwin`，`context_len=128`。训练数据默认启用 cache；每一个任务 instruction 都必须有对应的 `.t5_len128.wan22ti2v5b.pt` 文件。因此首次运行 Zeva 前先执行：

```bash
python scripts/precompute_text_embeds.py \
  --config-name train \
  task=robotwin_zeva_fastwam_3cam_384
```

多卡预计算示例：

```bash
torchrun --standalone --nproc_per_node=8 \
  scripts/precompute_text_embeds.py \
  --config-name train task=robotwin_zeva_fastwam_3cam_384
```

若改了 cache 目录，必须同时覆盖 `data.train.text_embedding_cache_dir` 和 `data.val.text_embedding_cache_dir`。只改目录而不重新生成 embedding，数据集会在第一个缺失 instruction 处报错。

## 3. Zeva 配置和关键参数

主配置文件是 [`configs/model/zeva_fastwam.yaml`](configs/model/zeva_fastwam.yaml)，任务配置是 [`configs/task/robotwin_zeva_fastwam_3cam_384.yaml`](configs/task/robotwin_zeva_fastwam_3cam_384.yaml)，仿真配置是 [`configs/sim_robotwin_zeva.yaml`](configs/sim_robotwin_zeva.yaml)。所有路径均可用 Hydra 命令行覆盖。

| 配置 | 当前 V1 值 | 作用 |
| --- | ---: | --- |
| `action_dim` | 14 | RoboTwin action/state 维度 |
| `num_frames` | 33 | 原始窗口的 RGB/action 对齐长度 |
| `action_video_freq_ratio` | 4 | 32 个 action 对应 9 个 RGB 帧 |
| `action_horizon` | 32 | 一个 FastWAM action chunk |
| `action_group_size` / CTE `transition_steps` | 4 | 每个 transition 执行 4 个 action |
| transition 数 | 8 | 一个窗口内的 `32 = 8 x 4` |
| camera 顺序 | `cam_high`, `cam_left_wrist`, `cam_right_wrist` | 顺序不能改变 |
| 最终图像 | `[3, 384, 320]` mosaic | 与 FastWAM RoboTwin processor 一致，RGB 归一化到 `[-1, 1]` |
| action normalization | `fastwam_processor_output` | 使用同一份 `dataset_stats.json` 的 z-score |
| CTE `phase_dim/effect_dim` | 128 / 128 | 必须与 prompt encoder 相同 |
| `memory.bit_size` | 4 | BIT brief 长度，必须等于 `prompt.brief_length` |
| `memory.pim_top_k` | 4 | PIM 检索条数，必须等于 `prompt.persistent_length` |
| `pim_max_entries` | 256 | 每个 episode memory bank 容量 |
| `merge_threshold` | 0.85 | V1 provisional phase/effect 合并阈值，正式实验应在 held-out 集上重新校准 |
| `beta_phase/beta_effect` | 0.5 / 0.5 | PIM 相似度两部分权重 |
| prompt hidden / adapter hidden | 256 / 1024 | CausalPromptEncoder 和 prefix adapter 宽度 |
| `adapter.gate_init` | 0 | addon 初始为精确 no-op |
| `train_gate_epsilon` | `1e-3` | 仅训练时让零初始化输出层获得梯度，评测不加 epsilon |
| action scheduler shift | 1.0 / 1.0 | `train_shift` / `infer_shift`，与 FastWAM action path 对齐 |
| Stage 2 batch/lr/steps | 16 / `2e-4` / 10000 | 任务配置默认值，可按显存覆盖 |

因果顺序必须保持：CTE `initialize()` 只看当前 frame；执行完 4-action group 后，用 observed after-frame 调用 `update()`，再把得到的 `phase_post + observed effect` 写入 memory。BIT 在每次 attempt 开始时清空，PIM 在同一 episode 的 retry 间保留并做 running mean/count 合并；当前 attempt 的条目不会被当前 attempt 自己检索。

评测时 `EVALUATION.skip_get_obs_within_replan=false` 是强制要求，否则无法为每个已执行 action 配对 after-frame。Zeva 模式下 `replan_steps` 必须是 4 的倍数，默认 8；`action_horizon` 必须是 32。

## 4. Checkpoint、cache 和路径约定

| 名称 | 示例路径 | 谁生成 | 用在哪里 |
| --- | --- | --- | --- |
| FastWAM base checkpoint | `runs/robotwin_uncond_3cam_384_1e-4/<run>/checkpoints/weights/step_XXXXXX.pt` 或 release `.pt` | 原 FastWAM 训练 | Stage 2 的 `ckpt=`、所有评测的 `--ckpt` |
| CTE checkpoint | `runs/zeva_cte/cte.pt` | Stage 1 | cache 构建、Stage 2、`pim_shadow/pim_on` 评测 |
| phase/effect cache | `data/robotwin2.0/zeva_cache/v1/` | `build_zeva_robotwin_cache.py` | Stage 2 离线 memory prefix |
| Stage 2 addon | `runs/zeva_stage2/checkpoints/weights/step_XXXXXX_addon.pt` | Stage 2 | 仅 `pim_on` 评测 |
| dataset stats | `data/robotwin2.0/dataset_stats.json` | 数据预处理/已有发布文件 | Stage 1、cache、Stage 2、评测 |
| ActionDiT backbone | `checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt` | 预处理脚本 | 构造 FastWAM 模型 |

base checkpoint 必须是兼容 14 维 RoboTwin FastWAM 的 checkpoint，通常包含 `mot`（以及可选的 `proprio_encoder`）键。CTE 和 addon 都不是完整 FastWAM，不能单独传给 `ckpt`。Stage 2 addon 只保存 `CausalPromptEncoder`、`BehaviorPrefixAdapter` 和 gate；加载时会校验 base/CTE SHA256，防止把 addon 错配到另一套模型。

cache 目录至少包含：

```text
zeva_cache/v1/
├── manifest.json
├── episode_index.json
└── phase_effect-*.safetensors
```

`manifest.json` 记录 CTE hash、stats hash、数据路径、相机顺序、action normalization、维度和 schema。默认 `model.zeva.cache.strict_manifest=true`，换数据、stats、CTE 或 action 对齐参数后必须重建 cache，不能复用旧目录。

## 5. 完整训练命令

### 5.0 先准备 FastWAM base（已有 release/base 可跳过）

Zeva 不重新训练或覆盖 FastWAM 主干。若没有兼容 RoboTwin 的 base checkpoint，先按原 FastWAM 入口训练；该任务使用同一份三相机、14 维数据配置：

```bash
RUN_ID=fastwam_base bash scripts/train_zero1.sh 8 \
  task=robotwin_uncond_3cam_384_1e-4
```

实际多卡数可将 `8` 改为可用 GPU 数。上述命令的输出目录为 `runs/robotwin_uncond_3cam_384_1e-4/fastwam_base/`；训练完成后使用其中的 `checkpoints/weights/step_XXXXXX.pt`（或 FastWAM 发布的 `robotwin_uncond_3cam_384.pt`）作为后续 Stage 2 的 `ckpt=`。base 的 action horizon、相机顺序和 `dataset_stats.json` 必须与 Zeva 配置一致。

### 5.1 （可选）导出 transition view

这一步用于检查并物化 episode-aware 的 32-action/8-transition 样本，不需要 FastWAM 或 CTE checkpoint；输出到 `model.zeva.transition_path`，默认 `data/robotwin2.0/zeva_transitions/`。Stage 1/cache builder 会直接读取同一底层数据，因此该步骤不是硬性前置条件。

```bash
python scripts/build_zeva_robotwin_transitions.py \
  --config-name train \
  task=robotwin_zeva_fastwam_3cam_384 \
  model.zeva.transition_path=./data/robotwin2.0/zeva_transitions
```

### 5.2 Stage 1：训练 CTE

Stage 1 不需要传 FastWAM base checkpoint，但需要数据、`dataset_stats.json` 和 T5 cache。CTE 训练输出 `cte.pt`、解析后的 `config.yaml`、`dataset_manifest.json` 和 `metrics.jsonl`。

```bash
python scripts/train_zeva_cte.py \
  --config-name train \
  task=robotwin_zeva_fastwam_3cam_384 \
  device=cuda \
  output_dir=./runs/zeva_cte
```

断点续训（会恢复 CTE optimizer/scheduler 状态）：

```bash
python scripts/train_zeva_cte.py \
  --config-name train task=robotwin_zeva_fastwam_3cam_384 \
  device=cuda output_dir=./runs/zeva_cte \
  resume=./runs/zeva_cte/cte.pt
```

### 5.3 构建 CTE phase/effect cache

该脚本加载冻结的 Stage 1 CTE，按 episode 顺序传递 recurrent state，只保留不重叠且完整的窗口，并写出 safetensors shards 和严格 manifest。

```bash
python scripts/build_zeva_robotwin_cache.py \
  --config-name train \
  task=robotwin_zeva_fastwam_3cam_384 \
  model.zeva.cte.checkpoint=./runs/zeva_cte/cte.pt \
  model.zeva.cache.path=./data/robotwin2.0/zeva_cache/v1
```

如果更换 `dataset_stats.json`、相机顺序、CTE 维度或数据集路径，请使用新的 cache 目录或先删除旧 cache 后重建。

### 5.4 Stage 2：只训练 Zeva addon

Stage 2 必须传三项：

* `ckpt=`：冻结的 FastWAM base checkpoint；
* `model.zeva.cte.checkpoint=`：Stage 1 的 `cte.pt`；
* `model.zeva.cache.path=`：与该 CTE、stats 和数据完全匹配的 cache。

```bash
python scripts/train_zeva_fastwam.py \
  --config-name train \
  task=robotwin_zeva_fastwam_3cam_384 \
  ckpt=./runs/robotwin_uncond_3cam_384_1e-4/fastwam_base/checkpoints/weights/step_010000.pt \
  model.zeva.cte.checkpoint=./runs/zeva_cte/cte.pt \
  model.zeva.cache.path=./data/robotwin2.0/zeva_cache/v1 \
  output_dir=./runs/zeva_stage2 \
  mixed_precision=bf16
```

输出 addon 的命名格式为 `runs/zeva_stage2/checkpoints/weights/step_XXXXXX_addon.pt`；同时会保存 `base_checkpoint.sha256`、`cte_checkpoint.sha256` 和 cache manifest 副本。Stage 2 断点续训传 state directory，而不是把 state 当成 base checkpoint：

```bash
python scripts/train_zeva_fastwam.py \
  --config-name train task=robotwin_zeva_fastwam_3cam_384 \
  ckpt=./runs/robotwin_uncond_3cam_384_1e-4/fastwam_base/checkpoints/weights/step_010000.pt \
  model.zeva.cte.checkpoint=./runs/zeva_cte/cte.pt \
  model.zeva.cache.path=./data/robotwin2.0/zeva_cache/v1 \
  output_dir=./runs/zeva_stage2 \
  resume=./runs/zeva_stage2/checkpoints/state/step_005000
```

### 5.5 文本 embedding、Wan 下载和多卡训练的关系

Zeva Stage 1/2 都沿用 FastWAM 的 dataset/processor，不会改变原有 image resize、action normalization 或 prompt 格式。需要多卡时，CTE 脚本可由外部 launcher 启动；Stage 2 推荐沿用 FastWAM 的 Accelerate/DeepSpeed 配置。不要通过随机打乱 Stage 1 的 episode 顺序来“增加数据量”，因为 CTE 的 hidden state handoff 和 cache manifest 都依赖确定的 episode 顺序。

## 6. RoboTwin 固定 seed 评测

推荐使用封装入口。它会检查 checkpoint、解析 `dataset_stats.json`、创建 policy 链接，并将固定 seed、retry 次数和 Zeva 参数传给 vendored RoboTwin：

```bash
python scripts/eval_zeva_robotwin_fixed_attempts.py \
  --task click_alarmclock \
  --ckpt ./runs/robotwin_uncond_3cam_384_1e-4/fastwam_base/checkpoints/weights/step_010000.pt \
  --cte-checkpoint ./runs/zeva_cte/cte.pt \
  --addon-checkpoint ./runs/zeva_stage2/checkpoints/weights/step_010000_addon.pt \
  --seed 0 --max-attempts 4 --mode pim_on
```

三种 mode 的依赖如下：

```text
base       只需要 --ckpt；不启动 CTE/PIM/addon
pim_shadow 需要 --ckpt + --cte-checkpoint；运行 memory lifecycle，但不加载 addon
pim_on     需要 --ckpt + --cte-checkpoint + --addon-checkpoint
```

为了进行严格的同 episode 对比，使用同一个 `--task`、`--seed` 和 `--max-attempts`，分别运行三个 mode。`--max-attempts 4` 表示同一 fixed-seed episode 最多重试四次，不等价于四个独立随机 episode。

封装入口会从 base checkpoint 的父目录向上查找 `dataset_stats.json`。如果 stats 在其他位置，直接使用 Hydra 入口显式指定：

```bash
python experiments/robotwin/eval_robotwin_single.py \
  --config-name sim_robotwin_zeva.yaml \
  ckpt=./runs/robotwin_uncond_3cam_384_1e-4/fastwam_base/checkpoints/weights/step_010000.pt \
  EVALUATION.task_name=click_alarmclock \
  EVALUATION.dataset_stats_path=./data/robotwin2.0/dataset_stats.json \
  EVALUATION.cte_checkpoint=./runs/zeva_cte/cte.pt \
  EVALUATION.addon_checkpoint=./runs/zeva_stage2/checkpoints/weights/step_010000_addon.pt \
  EVALUATION.zeva_mode=pim_on \
  EVALUATION.fixed_seed=0 \
  EVALUATION.max_attempts=4 \
  EVALUATION.action_horizon=32 \
  EVALUATION.replan_steps=8 \
  EVALUATION.skip_get_obs_within_replan=false
```

评测构造模型时仍需要 Wan/ActionDiT 文件；评测只加载 addon 的 prompt/adapter 参数，不会从 addon 恢复 FastWAM base。若 `replan_steps` 不是 4 的倍数，或 `skip_get_obs_within_replan=true`，入口会在 rollout 前直接拒绝。

## 7. 常见报错和排查顺序

1. `requires an existing ... pretrained_norm_stats`：确认 `data/robotwin2.0/dataset_stats.json` 存在，并通过 `data.train.pretrained_norm_stats=/absolute/path/to/dataset_stats.json` 覆盖；Zeva 入口不接受 `null` stats。
2. `Missing text embedding cache`：先运行 `scripts/precompute_text_embeds.py`，检查 `data.train/val.text_embedding_cache_dir` 和 `context_len=128` 是否一致。
3. `requires an existing frozen FastWAM base checkpoint`：Stage 2 的 `ckpt` 必须是 FastWAM base，不是 `cte.pt`、cache 目录或 addon。
4. `cache manifest mismatch`：不要混用不同 stats、CTE、dataset path、相机顺序或 action horizon 生成的 cache；重建 cache。
5. `pim_top_k ... persistent_length` 或 `bit_size ... brief_length`：评测/训练 override 必须与训练 addon 的配置一致，默认均为 4。
6. `CUDA is unavailable`：Stage 2 需要 CUDA；评测的 CPU fallback 只用于轻量 smoke test，不能代表 RoboTwin 性能。
7. `RoboTwin root/policy/assets not found`：确认 `EVALUATION.robotwin_root`（默认 `third_party/RoboTwin`）、policy 软链接和官方 assets 已安装。

## 8. 结果文件和可复现性

一次完整实验通常具有如下结构：

```text
runs/
├── zeva_cte/
│   ├── cte.pt
│   ├── config.yaml
│   ├── dataset_manifest.json
│   └── metrics.jsonl
└── zeva_stage2/
    ├── config.yaml
    ├── metrics.jsonl
    ├── base_checkpoint.sha256
    ├── cte_checkpoint.sha256
    └── checkpoints/
        ├── weights/step_XXXXXX_addon.pt
        └── state/step_XXXXXX/
```

不要只保存 addon 文件而丢弃对应的 base/CTE hash、cache manifest 和 stats。只有在固定 seed、固定 task、固定 retry protocol 下比较 `base → pim_shadow → pim_on`，才能判断 causal memory 是否真正提升成功率或减少重试；当前配置中的 `merge_threshold=0.85` 仍是 V1 provisional 值，不应直接解释为已经完成的最优超参。

## 9. 本地静态回归

在没有真实 RoboTwin 数据、CUDA、SAPIEN 和 checkpoint 的机器上，可以运行代码级回归：

```bash
PYTHONPATH=src pytest -q
python -m compileall -q src scripts experiments/robotwin/fastwam_policy third_party/RoboTwin/script
git diff --check
```

这些检查不能替代真实 rollout；最终验收仍应在具备 CUDA、RoboTwin assets、数据 stats、base/CTE/cache/addon 的环境中完成固定 seed 三模式对比。
