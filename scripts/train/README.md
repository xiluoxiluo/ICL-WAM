# Zeva-main 观察式经验：离线训练入口

`scripts/train/` 是当前默认路线，参考 [Zeva main 固定版本 `17ca9fa0c68dca1fc7915dd6433de70a100e93a1`](https://github.com/air-embodied-brain/Zeva/tree/17ca9fa0c68dca1fc7915dd6433de70a100e93a1)。预测 CTE、语言 task classifier 和 cross-attempt predicted trace 的旧 Ego 工作流完整保留在 [`scripts/train_ego/`](../train_ego/README.md)。两条路线使用不同 profile、数值合同、缓存来源和 checkpoint，不能改标签后混用。

本路线的离线监督只使用有真实来源记录的成功示范。CTE 学习 phase 和观察到的 `effect_post`；Stage2A 学习 global/BIT 的动作条件，Stage2B 学习 PIM reader。部署时把实际观察到的完整交互结果写入记忆，包括失败尝试中的结果，**不反向传播、不更新 foundation/CTE/addon 参数**。当前模型采用 main 的 gated PIM 与 global 合并、BIT action prior，再适配 FastWAM 的 video/action 接口；不是 Ego 的三个独立预测条件 token。PIM 零 gate 的等价对象是已经训练好的 Stage2A parent，不是未加入 EAP 的原始 FastWAM。

原 main 发布的 benchmark 使用 RoboCasa；这里是 FastWAM + RoboTwin 的本地适配，不是论文数值复现。原 main 公开 CTE 模型及损失，但没有公开完整、可逐项复现的 CTE 优化与数据配方。下述 CTE 5000 steps、固定成功 episode holdout、离线历史构造属于本仓库可审计的训练选择，不能称为作者完整原始配方。

## 先确认成功数据来源

`SUCCESS_MANIFEST` 默认建议位置是 `$DATA_ROOT/meta/zeva_success_demonstrations.json`，脚本不会自动创建它，也不会从 task 名称、Clean/Randomized、缺少失败标签推断成功。

如果 `meta/episodes.jsonl` 自带真实 success/outcome 字段，可先运行：

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export DATA_ROOT=/data/share/1919650160032350208/foundation_model/datasets/robotwin2.0
export SUCCESS_MANIFEST="$DATA_ROOT/meta/zeva_success_demonstrations.json"
python scripts/build_zeva_success_manifest.py \
  --dataset-root "$DATA_ROOT" --output "$SUCCESS_MANIFEST"
```

外部真实结果表使用 `--outcomes-json /absolute/real_outcomes.json`，格式是 `"/absolute/dataset::episode-N": true/false`，完整覆盖所选数据来源。如果已经核实采集源只包含成功专家示范，才能由你显式添加 `--declare-success-only --evidence "实际采集流程/记录位置及核实依据"`；该声明会被保存为声明，不能解释为重新仿真验证。未确认来源时，先解决来源问题。任何编号脚本均不代替用户作这项声明。

所有阶段共享 `model.zeva.profile=main_observed`、success manifest、stats、semantic task map，以及默认 `validation_fraction=0.1`、`split_seed=42` 的按成功 episode、按 task 划分。验证 episode 不进入可检索示范 bank 或历史候选池；CTE、检索头和 Stage2 用相同 holdout。每个实际 task 需要足够的完整成功 episode 来分出训练和验证，**没有固定 50-task 上限或必须恰好 50-task 的要求**。这些划分是防止泄漏的工程设置。

## 路径和环境

默认值复用原 `scripts/*.sh` 中的路径，集中在 [`_env.sh`](_env.sh)：

| 变量 | 默认值 / 用途 |
| --- | --- |
| `DATA_ROOT` | `/data/share/1919650160032350208/foundation_model/datasets/robotwin2.0` |
| `DIFFSYNTH_MODEL_BASE_PATH` | `/data/share/1919650160032350208/zjj/fastwam/checkpoints` |
| `BASE_CKPT` | 仓库 `checkpoints/fastwam_ckpt.pt` |
| `ACTION_DIT_CKPT` | 仓库 `checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt` |
| `STATS_PATH` / `SEMANTIC_TASK_MAP` | `$DATA_ROOT/robotw_stats.json` / `$DATA_ROOT/meta/semantic_task_map.json`；可以显式换成实际文件 |
| `TEXT_CACHE` | `$DATA_ROOT/text_embeds_cache_new`；匹配的旧文本 cache 可以复用 |
| `RUN_ROOT` | 仓库 `runs/zeva_main_observed_v1`；新 CTE、cache、bank、readout、检索头和 Stage2 统一放这里 |
| `PYTHON_BIN` | `python`；使用已经激活并安装本仓库依赖的环境，不猜测或自动切换 conda 环境 |
| `GPU_IDS` | 优先继承 `CUDA_VISIBLE_DEVICES`，否则 `0,1,...,15` |

可通过 export 覆盖，或设置 `ICLWAM_TRAIN_ENV=/absolute/my_main_training.sh` 指向自己维护的变量配置文件。不要直接套用 Ego 环境文件中的旧 artifacts。`--dry-run` 只打印正确引用的命令：不执行 Python，不读取模型/GPU，不创建输出，不能作为资产兼容性或训练已完成的证明。用户配置文件本身应只设置变量。

```bash
bash scripts/train/12_run_all.sh --dry-run
bash scripts/train/01_check_environment.sh
bash scripts/train/12_run_all.sh --from 02 --to 11
```

`CHECK_ENVIRONMENT=0 bash scripts/train/01_check_environment.sh` 只做标准库来源检查，不导入 Torch/CUDA、不解码视频、不加载 checkpoint。

## 编号顺序和产物

| 编号 | 入口 | 主要输出 |
| --- | --- | --- |
| 01 | [`01_check_environment.sh`](01_check_environment.sh) | 只读检查真实 success manifest、metadata、stats、assets 和可选环境依赖 |
| 02 | [`02_precompute_text_embeds.sh`](02_precompute_text_embeds.sh) | `TEXT_CACHE`，复用兼容的已有条目 |
| 03 | [`03_precompute_cte_latents.sh`](03_precompute_cte_latents.sh) | `LATENT_CACHE`：main success 来源的 Wan VAE latents |
| 04 | [`04_train_cte.sh`](04_train_cte.sh) | `CTE_RUN_DIR/cte.pt`；`train_zeva_cte.py`，观察式、按 optimizer step 停止 |
| 05 | [`05_build_phase_effect_cache.sh`](05_build_phase_effect_cache.sh) | `PHASE_CACHE`，从完整历史导出 phase / observed effect；缺失的真实尾帧由 builder 读取 |
| 06 | [`06_build_behavior_bank.sh`](06_build_behavior_bank.sh) | `BEHAVIOR_BANK` 与 `READOUT_CACHE`；冻结初始 image+text readout，bank 仅 train，validation 只作为查询 |
| 07 | [`07_train_static_retrieval.sh`](07_train_static_retrieval.sh) | `RETRIEVAL_HEAD`；不是 Ego 纯语言 task classifier |
| 08 | [`08_build_static_context.sh`](08_build_static_context.sh) | `STATIC_CONTEXT_CACHE`；train / validation query 的静态 context |
| 09 | [`09_build_pim_cache.sh`](09_build_pim_cache.sh) | 一个 `PIM_CACHE`，包含共享划分身份；prepared trainer 创建互斥 train / validation 视图 |
| 10 | [`10_train_stage2a.sh`](10_train_stage2a.sh) | Stage2A addon，以及 joint 模式或继承场景所需的配对 foundation |
| 11 | [`11_train_stage2b.sh`](11_train_stage2b.sh) | 从选定 A parent 初始化的 PIM addon / 配对 foundation |
| 12 | [`12_run_all.sh`](12_run_all.sh) | 顺序运行 01–11，支持 `--from`、`--to`、`--dry-run` |

03、05、06、09 在单节点的 `PREP_GPU_IDS` 上使用 torchrun；04 使用 `CTE_GPU_IDS`，默认同样为全部分配 GPU。07/08 使用 `RETRIEVAL_GPU_ID` 的一个设备；07 可设 `RETRIEVAL_DEVICE=cpu`。02 的 Python 入口自行按 `TEXT_GPU_IDS` 启动文本预计算进程。不要把这些 shell 再包一层 torchrun。

## 训练设置与 foundation 开关

| 阶段 | 默认参数 |
| --- | --- |
| CTE | `CTE_STEPS=5000`、LR `2e-4`、每 rank batch 8 = 4 tasks × 2 episodes；本地 step-based recipe，不使用 Ego 80 epoch / epoch80 gate |
| Stage2A | `STAGE2A_STEPS=5000`、LR `2e-4`、有效 batch 128；16 GPU 时每卡 1 × 累积 8 |
| Stage2B | `STAGE2B_STEPS=500`、LR `5e-5`、有效 batch 64；16 GPU 时每卡 1 × 累积 4；具体参数组倍率由 main trainer/config 实现 |
| 默认 foundation | A、B 均 `frozen`；CTE、readout 来源及检索头保持其冻结身份 |
| 显式 joint 实验 | A foundation LR `5e-6`，B foundation LR `1e-6`；属于 WAM 训练实验，不声称是原 main 默认 |
| 参数与计算 | 可训练参数 FP32、BF16 AMP；不是先把训练权重转 BF16 再用小学习率更新 |
| optimizer sharding | 默认 `none`；多 rank joint 显存需要时显式设相应 `STAGE2A_OPTIMIZER_SHARDING=zero1` / `STAGE2B_OPTIMIZER_SHARDING=zero1` |

```bash
STAGE2A_FOUNDATION_MODE=joint STAGE2A_OPTIMIZER_SHARDING=zero1 \
  bash scripts/train/10_train_stage2a.sh

# B 的模式独立；仍须指定实际 A parent，而不是退回原始 base。
STAGE2A_FOUNDATION_MODE=joint STAGE2B_FOUNDATION_MODE=frozen \
  bash scripts/train/11_train_stage2b.sh
```

`STAGE2A_CKPT` 默认从 A 的目录和 `STAGE2A_STEPS` 推导为 `checkpoints/weights/step_XXXXXX_addon.pt`，可显式覆盖为实际已完成 checkpoint。改变 B 的 frozen/joint 不改变 A 的路径。joint A 后 frozen B 仍需保留部署时对应的 foundation；只复制 addon 并不能恢复该模型。

改变 GPU 数时脚本从真实 world size 自动计算累积步数，同时校验整除。有效 batch 可用 `STAGE2A_GLOBAL_BATCH_SIZE`、`STAGE2B_GLOBAL_BATCH_SIZE` 修改；手工设置累积时必须匹配。多节点先在 node 0、`NNODES=1` 运行 01–09，再在各节点设置 `NNODES`、`NODE_RANK`、`MASTER_ADDR`，运行 `12_run_all.sh --from 10 --to 11`。

已有输出默认拒绝覆盖。准备完成后从后续编号继续；latent cache 中断可显式 `LATENT_RESUME=1`，底层仍验证来源与存储。`CTE_RESUME` 指向兼容 CTE checkpoint；当前 CTE trainer 结束时才写 `cte.pt`，`save_every` 不是已经实现的周期性 CTE checkpoint。A/B 的 `STAGE2A_RESUME` / `STAGE2B_RESUME` 必须是完整 trainer state 目录，不能给 deployment addon `.pt`。单独改 `RUN_ROOT` 会得到独立实验，避免缓存/权重混配。

## RoboTwin：分别报告单次、恢复和稳定性

入口 [`../eval_zeva_main_robotwin.py`](../eval_zeva_main_robotwin.py) 使用 `sim_robotwin_zeva_main`，默认按顺序在一个 GPU 上运行同权重 memory off/on。参数见 `python scripts/eval_zeva_main_robotwin.py --help`；需要 base、CTE、A/B addon、stats、success manifest、bank 和 retrieval checkpoint。

| 参数 | 用途与解释 |
| --- | --- |
| `--protocol single_attempt` | 每个随机场景清记忆、只试一次，报告独立 `SR@1` |
| `--protocol stop_on_success --max-attempts 4` | 同场景/指令重试，首成功即停；默认关闭 replay。同权重 memory on/off 的 `CSR@k` 是失败经验收益主实验，但不能仅凭累计率上涨证明失败记忆有效 |
| `--protocol fixed_attempts --max-attempts 6` | 所有场景均试满 K 次，报告逐次成功与首次成功后的稳定率 |
| `--success-replay` | 仅 fixed 模式显式启用；只有首次确认成功后的匹配 phase/action trace 可以回放，相关稳定收益不能称为纯失败学习 |

主实验应比较相同 checkpoint 的 memory on/off。`CSR@4` 表示四次以内至少成功一次，不能写成单次 `SR=...`。重复抽样本身也会提高累计成功率；main PIM 还在当前 attempt 内积累 observed effects，因此进一步隔离“此前失败轨迹内容”的贡献需要共享第一次失败与匹配 retry noise 的对照。连续 RNG 不等于两组在轨迹长度分叉后每一步噪声仍完全相同。

部署始终是固定参数的前向计算与记忆写入。单卡顺序评测不加载两个并行 foundation，但 32 GB 显存、速度、成功率仍需在真实机器上测量；本目录的 CPU/dry-run 测试不代表已经完成离线训练或 RoboTwin rollout。
