> 保留的 Zeva-Ego 工作流。默认 Zeva-main 观察式经验路线已移到 [scripts/train](../train/README.md)，两者缓存与 checkpoint 不可混用。

# ICL-WAM 严格 Zeva-Ego 版：离线训练启动器

本目录把当前训练链整理成 `01`–`11` 个独立步骤，`12_run_all.sh` 按顺序调用它们。数据、统计量、FastWAM 权重和 Wan assets 默认路径取自仓库已有 `scripts/*.sh` 与 `configs/data/robotwin.yaml`；新产物统一放到独立的 `RUN_ROOT`。

训练合同以[训练与评测指南](../../docs/zeva_ego_training_evaluation.md)、[对齐审计](../../docs/zeva_ego_alignment_audit.md)及其固定的 [Zeva-Ego 参考版本](https://github.com/air-embodied-brain/Zeva/tree/bae85ad753d3afe4f2e56762373d48ac0267e101)为准。这里调用当前 Python 入口；旧 VAE/window CTE、image+text retrieval 和旧 cache 启动脚本不参与这条流程。

## 1. 默认路径与必需输入

所有默认值集中在 [`_env.sh`](_env.sh)。`REPO_ROOT` 自动定位当前 checkout，不要求从仓库根目录启动。脚本使用当前已激活的 Python 环境，也可设置 `PYTHON_BIN=/absolute/path/to/python`；原有 `.sh` 没有提供可核实的 conda 环境名，因此不自动激活或安装环境。

| 环境变量 | 默认值 / 来源 |
| --- | --- |
| `REPO_ROOT` | 本目录向上两级；在原服务器通常是 `/data/share/1919650160032350208/zjj/ICL-WAM` |
| `DATA_ROOT` | `/data/share/1919650160032350208/foundation_model/datasets/robotwin2.0`；也兼容原有 `DATA` 变量 |
| `STATS_PATH` | `$DATA_ROOT/robotw_stats.json` |
| `SEMANTIC_TASK_MAP` | `$DATA_ROOT/meta/semantic_task_map.json` |
| `TEXT_CACHE` | `$DATA_ROOT/text_embeds_cache_new` |
| `DIFFSYNTH_MODEL_BASE_PATH` | `/data/share/1919650160032350208/zjj/fastwam/checkpoints` |
| `BASE_CKPT` | `$REPO_ROOT/checkpoints/fastwam_ckpt.pt` |
| `ACTION_DIT_CKPT` | `$REPO_ROOT/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt` |
| `RUN_ROOT` | `$REPO_ROOT/runs/zeva_ego_reference_v3` |
| `CTE_TASK_SUBSET` | 建议位置 `$DATA_ROOT/meta/cte_task_subset.json`，**需要提供真实文件** |
| `EPISODE_CONDITIONS` | 建议位置 `$DATA_ROOT/meta/episode_conditions.json`，**需要提供真实文件** |

最后两项是严格训练新增的输入，并非在旧脚本中找到的现成文件。不会推测它们的内容，也不会把旧 CTE/cache 改名作为新产物。

`CTE_TASK_SUBSET` 格式是 `{"task_names": ["真实canonical_task_name", "..."]}`，任务顺序参与 CTE sampler，必须保留实验使用的顺序。`EPISODE_CONDITIONS` 是 `{ "/真实解析后的dataset根目录::episode-0": "Clean", ... }`，value 为真实采集条件，默认完整课程需要 `Clean` 与 `Randomized`。它必须覆盖**完整源数据中的正长度 episode**，包括任务子集之外的轨迹及短轨迹；CTE 先划分 condition×task 分区，再过滤子集/短轨迹。示例中的占位符不能作为实际数据使用。CTE、boundary 和 PIM 必须使用同一来源和同一份 conditions 文件。

`01` 只读检查路径、JSON、source coverage、可导入的训练依赖和 CUDA。它不加载 checkpoint、不读 RGB/action，也不生成数据清单。完整 tensor、数据身份、CTE readiness/gain gate 由实际训练与 cache 入口检查。设置 `CHECK_ENVIRONMENT=0` 可只用标准库检查输入元数据；这不代表 CUDA 训练环境已经合格。

语义表应使用已有真实映射。如果数据集尚未生成它，可先查看 `python scripts/build_robotwin_semantic_task_map.py --help`，按真实 task 信息生成；本目录不会推断 `Clean/Randomized`，也不会自动生成 task subset。已有匹配的数据、stats、policy 文本 cache、FastWAM base 和 ActionDiT 初始化文件可以复用。预训练 ResNet/Mamba、Wan 及 tokenizer 依赖需按现有项目环境准备；启动器不下载或安装环境包。

## 2. 编号步骤

| 顺序 | 脚本 | 计算方式 | 输出 / 作用 |
| --- | --- | --- | --- |
| 01 | [`01_check_environment.sh`](01_check_environment.sh) | 单进程，只读 | 验证必需输入、真实 task/condition 元数据、激活环境及可见 CUDA 设备 |
| 02 | [`02_precompute_text_embeds.sh`](02_precompute_text_embeds.sh) | 一个启动进程，Python 内部按可见 GPU 分配 | FastWAM policy 文本 cache；已有条目不覆盖 |
| 03 | [`03_train_cte.sh`](03_train_cte.sh) | **单卡单进程** | 完整 episode CTE，默认 batch8、80 epoch；`cte/cte_epoch_080.pt` |
| 04 | [`04_build_boundary_cache.sh`](04_build_boundary_cache.sh) | 单卡单进程 | `artifacts/boundary_v2/`，包括 terminal CTE frame 和固定 splits |
| 05 | [`05_build_task_memory.sh`](05_build_task_memory.sh) | 默认 CPU | train task bank、冻结语言 encoder 与 language features |
| 06 | [`06_train_task_language.sh`](06_train_task_language.sh) | 默认 CPU | task language classifier；训练后冻结用于检索 |
| 07 | [`07_build_query_context.sh`](07_build_query_context.sh) | 默认 CPU | per-decision query context v2 |
| 08 | [`08_build_pim_train.sh`](08_build_pim_train.sh) | CPU | PIM v4 train cache，固定参考配对/采样合同 |
| 09 | [`09_build_pim_validation.sh`](09_build_pim_validation.sh) | CPU | PIM v4 validation cache；历史来源仍限制为 train |
| 10 | [`10_train_stage2a.sh`](10_train_stage2a.sh) | 默认 16 GPU | 5000 updates；joint foundation + EAP |
| 11 | [`11_train_stage2b.sh`](11_train_stage2b.sh) | 默认 16 GPU | 2000 updates；前500步 PIM warmup，之后按所选模式训练 |
| 12 | [`12_run_all.sh`](12_run_all.sh) | 顺序调度 | 运行上述完整流程或 `--from/--to` 指定范围 |

准备阶段不需要 RoboTwin 仿真器。CTE 使用单进程是为了保留完整 episode、无重复采样的参考训练合同，不按16卡复制 CTE sampler。PIM 固定使用 `reference_nearest`、`all_cte_frames_v1`、`torch_float32_first_in_source_order`、`reference_round`、4×64 memory 和 seed20260919；不自动降级到 reduced curriculum。

语言分类器的本地拟合配方仍以现有配置及主指南的公开信息边界为准；启动器不代表补齐了参考仓库未公开的实验清单或训练细节。

## 3. 最常用的启动方式

先激活已有训练环境。若真实文件在上述默认位置，下面两个 metadata 路径无需另设；否则先改为实际路径：

```bash
cd /data/share/1919650160032350208/zjj/ICL-WAM
export CTE_TASK_SUBSET=/absolute/path/to/actual_task_subset.json
export EPISODE_CONDITIONS=/absolute/path/to/actual_episode_conditions.json
export GPU_IDS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

# 先打印所有实际命令：不启动 Python，不要求训练资产存在，不创建产物目录。
bash scripts/train_ego/12_run_all.sh --dry-run

# 输入与命令核实后，执行完整离线流程。
bash scripts/train_ego/12_run_all.sh
```

可以逐个运行，或者从已完成步骤之后继续：

```bash
bash scripts/train_ego/03_train_cte.sh
bash scripts/train_ego/12_run_all.sh --from 04 --to 09
bash scripts/train_ego/12_run_all.sh --from 10 --to 11
```

启动器不自动跳过已有 CTE/cache/classifier 或覆盖旧训练目录。文本 cache 的 `overwrite=false` 支持已有匹配条目；05 的冻结 language encoder 可在身份兼容时复用，其余新输出应选空闲路径。复用已完成产物时用 `--from`，中断训练用下一节的 resume。不要提前在新的 CTE 或 Stage2 输出目录里写日志；可以重定向到另一个已有日志目录。

需要长期保存本机配置时，在自己选定的位置建立普通 shell 文件，例如 `/absolute/path/to/iclwam_train.env`：

```bash
DATA_ROOT=/data/share/1919650160032350208/foundation_model/datasets/robotwin2.0
RUN_ROOT=/data/share/1919650160032350208/zjj/ICL-WAM/runs/ego_reference_exp01
CTE_TASK_SUBSET=/absolute/path/to/actual_task_subset.json
EPISODE_CONDITIONS=/absolute/path/to/actual_episode_conditions.json
GPU_IDS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
STAGE2A_FOUNDATION_MODE=joint
STAGE2B_FOUNDATION_MODE=joint
```

然后 `export ICLWAM_TRAIN_ENV=/absolute/path/to/iclwam_train.env`。该文件在派生默认路径前读取，文件内显式赋值优先于外层同名环境变量；更改根路径会同步影响尚未显式覆盖的子路径。`RUN_ROOT` 应是固定实验目录，不要用每次调用都会变化的时间戳表达式。运行器会执行所指定 shell 配置中的内容，因此请使用自己的配置文件。

所有构建/训练步骤都支持 `--dry-run`。Hydra 入口可追加普通选项，例如 `log_every=20` 或 `num_workers=8`。工作流控制的路径、task、阶段、输出目录、resume、总步数、batch 和 foundation mode 请通过对应环境变量修改，避免一个步骤偷偷切换到另一套产物；它们的重复 CLI 覆盖会明确报错。`01`、`08`、`09` 与 `12` 不接受额外 Hydra 参数。

## 4. Foundation 开关与恢复训练

Stage2A、Stage2B 分别配置，默认均为 `joint`：

```bash
STAGE2A_FOUNDATION_MODE=joint bash scripts/train_ego/10_train_stage2a.sh

# 冻结对照：仍继承 joint Stage2A 的 addon + 配对 foundation。
STAGE2B_FOUNDATION_MODE=frozen bash scripts/train_ego/11_train_stage2b.sh
```

默认 A 输出 `stage2a_joint/`，B 输出 `stage2b_joint/` 或 `stage2b_frozen/`。**只改 B 的模式不会改变 A 的父 checkpoint 路径。** 需要选择另一个已训练父模型时，设置 `STAGE2A_CKPT=/absolute/path/to/checkpoints/weights/step_005000_addon.pt`，并保留其配对 foundation 文件。不要把基础 FastWAM checkpoint 当作 Stage2A addon。

CTE resume 指向含优化器/RNG 的训练 checkpoint 文件；Stage2 resume 必须指向完整 state **目录**：

```bash
CTE_RESUME="$PWD/runs/zeva_ego_reference_v3/cte/cte_epoch_040.pt" \
  bash scripts/train_ego/03_train_cte.sh

STAGE2A_RESUME="$PWD/runs/zeva_ego_reference_v3/stage2a_joint/checkpoints/state/step_002500" \
  bash scripts/train_ego/10_train_stage2a.sh

STAGE2B_RESUME="$PWD/runs/zeva_ego_reference_v3/stage2b_joint/checkpoints/state/step_001000" \
  bash scripts/train_ego/11_train_stage2b.sh
```

恢复 Stage2 时同时保留相应 `checkpoints/weights/`、`checkpoints/state/`、base、CTE 和离线 cache。Python trainer 会核对配方、数据身份、优化器与逐 rank RNG；切换 `joint/frozen`、有效 batch、拓扑或训练合同不属于精确续训。B resume 直接恢复自己的完整状态，不再要求原始 A checkpoint 仍在旧路径。若修改了 `RUN_ROOT` 或输出位置，恢复时应显式设置匹配的 `CTE_RUN_DIR` / `STAGE2A_RUN_DIR` / `STAGE2B_RUN_DIR`。

默认 `STAGE2A_STEPS=5000`、`STAGE2B_STEPS=2000`，A 的父 checkpoint 名字会随 `STAGE2A_STEPS` 自动派生。修改步数是新的实验配方；默认 CTE 固定正式80 epoch，不能用短 smoke checkpoint 替代。

## 5. 16卡、两机与批大小

Stage2 使用同一 `PYTHON_BIN -m torch.distributed.run` 启动普通 DDP，并使用 PyTorch ZeRO-1 分片优化器状态；不嵌套旧 Accelerate/DeepSpeed launcher。

默认 `GPU_IDS=0,...,15`，若外部已经设置 `CUDA_VISIBLE_DEVICES` 则继承它；`NPROC_PER_NODE` 默认等于所选设备数。`BATCH_SIZE=1`、`GLOBAL_BATCH_SIZE=256`，自动计算：

```text
GRAD_ACCUM_STEPS = GLOBAL_BATCH_SIZE / (NNODES × NPROC_PER_NODE × BATCH_SIZE)
```

16 ranks 对应累积16；8 ranks 对应累积32；1 rank 对应累积256且默认 `ZEVA_OPTIMIZER_SHARDING=none`。数量必须整除；显式 `GRAD_ACCUM_STEPS` 必须与声明的全局 batch 相符。减少 GPU 数量不会自动保证显存足够，尤其 joint 的 FP32 参数/梯度和优化器状态仍需实际测量。

如果16卡分布在两台各8卡的服务器，先在一台机器上执行01–09，确认产物在两台机器的共享绝对路径一致，然后在两台机器分别运行：

```bash
# 节点0；MASTER_ADDR 改为其他节点能访问的节点0地址。
NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.10 MASTER_PORT=29500 \
  GPU_IDS=0,1,2,3,4,5,6,7 \
  bash scripts/train_ego/12_run_all.sh --from 10 --to 11

# 节点1；路径、配置和端口应与节点0匹配。
NNODES=2 NODE_RANK=1 MASTER_ADDR=10.0.0.10 MASTER_PORT=29500 \
  GPU_IDS=0,1,2,3,4,5,6,7 \
  bash scripts/train_ego/12_run_all.sh --from 10 --to 11
```

单机使用 `--standalone`，多机使用明确的 rendezvous 地址；脚本不会自动 SSH 到另一台机器。多机 `12` 不接受从准备阶段开始，避免重复写入同一 cache。

CTE 与 boundary 默认使用 `GPU_IDS` 的第一张卡，可分别改 `CTE_GPU_ID` / `BOUNDARY_GPU_ID`。文本预计算可改 `TEXT_GPU_IDS`、`PRECOMPUTE_NUM_GPUS`、`PRECOMPUTE_BATCH_SIZE`；默认自动使用可见 GPU、batch8。05–07 默认 `LANGUAGE_DEVICE=cpu`，需要时可设 `cuda`（使用 `CTE_GPU_ID` 暴露的那张卡）。

## 6. 最终产物与验证边界

默认主要产物如下：

| 产物 | 相对 `RUN_ROOT` 的路径 |
| --- | --- |
| 正式 CTE | `cte/cte_epoch_080.pt` |
| Boundary | `artifacts/boundary_v2/` |
| Bank / encoder / features / classifier | `artifacts/task_memory_bank.pt`、`task_language_encoder.pt`、`task_language_features.pt`、`task_language_classifier.pt` |
| Query context | `artifacts/query_context_v2.pt` |
| PIM train / validation | `artifacts/pim_train_v4/`、`artifacts/pim_validation_v4/` |
| Stage2A 部署权重 | `stage2a_joint/checkpoints/weights/step_005000_addon.pt` 及配对 `step_005000_foundation.pt` |
| Stage2B 部署权重 | `stage2b_joint/checkpoints/weights/step_002000_addon.pt` 及配对 `step_002000_foundation.pt` |
| Stage2 精确恢复 | 对应阶段的 `checkpoints/state/step_XXXXXX/`，连同同级 `weights/` 中配对文件 |

`frozen` Stage2B 如果继承 joint Stage2A，也需要并会导出其实际使用的 foundation；以 checkpoint 的 `foundation_checkpoint_required` 与 SHA256 为准。推理不会更新 foundation。

本目录的验证范围是 shell 语法、真实入口配置组合、参数传递、拓扑/批大小、metadata preflight 和 resume 路由。没有在开发环境完成80 epoch CTE、16卡联合训练或 RoboTwin 仿真，不能据此宣称成功率或5090显存适配已验证。训练完成后的单卡 RoboTwin 评测按[训练与评测指南](../../docs/zeva_ego_training_evaluation.md)的 manifest 流程进行。
