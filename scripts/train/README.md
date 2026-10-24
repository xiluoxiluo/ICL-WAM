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
| `RUN_ROOT` | 仓库 `runs/zeva_main_observed_v2`；新 CTE、cache、bank、readout、检索头和 Stage2 统一放这里 |
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

### 本次 main-v2 复查后的重建要求

`numerical_contract=zeva_main_v2`、`artifact_lineage=zeva_main_observed_effect_v2`、`addon_schema=zeva_main_observed_policy_v2` 是本次审计后的主线。CTE 文件和缓存还记录 `cte_window_semantics=full_observed_prefix_masked_tail_v1`：磁盘保留原 H32 存储形状，但 CTE 实际历史包含末尾已经完成的四步交互，严格按真实 mask 截断；不把重复 padding 当观察。少于 32 步但已有完整四步交互的成功 episode 也会保留。

此前 `3991c47` 的 main-v1 产物需要重新运行 **03–11**，包括重训 CTE 与 Stage2A/B。匹配的 base/action checkpoint、原始数据、统计文件、semantic map、success manifest 和文本 cache 可以复用。新默认输出目录独立；不要给旧文件补版本字段来绕过检查。旧完整 trainer state 只能在原代码/原合同下恢复，不能用于 main-v2 精确续训。

CUDA 部署时，CTE 和静态检索头保持 FP32 权重、用 BF16 autocast 计算，特征/检索分数再按参考路径转 FP32；EAP/MCE 则在加载完成后随 policy 的推理精度转换，默认 BF16。CPU 开发测试的 CTE/检索特征使用 FP32。这里保留的是不同模块在原 main 中不同的精度路径；训练中的 FP32 master 参数不会为了部署测试原地舍入。完整证据见 [复查记录](../../docs/zeva_main_alignment_audit.md)。

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

## 06：两卡 FastWAM readout 首帧优化（可选）

阶段 06 的 CTE prototypes 使用已有 CTE latent 缓存，主要瓶颈通常是 FastWAM 初始 readout。每个 episode 的 readout 只需要第一张三相机观测（`video[:,0]`），但旧数据管线仍通过原始 33 帧图像查询和 9 帧重采样执行完整视频解码及预处理。**新优化在仅用于 06 的路径里把三个视频查询缩减为第 0 帧**，剩余 Action/State、FastWAMProcessor、三相机拼接、resize/crop/normalize、文本 cache、预训练模型前向和输出格式均保持不变。

默认关闭，不影响 CTE、Stage2 或普通 RoboTwin 读取。建议先检查 `df -h /dev/shm`，然后在当前 06 任务停止之后启动。**新版本支持复用已经完整保存的 rank 快照，但尚未写出完整 rank 快照之前的 batch 仍不能逐 batch 恢复**。首次任务用下述两卡/32 核配置测试：

```bash
PREP_GPU_IDS=0,1 PREP_NPROC_PER_NODE=2 \
ZEVA_BEHAVIOR_READOUT_FIRST_FRAME_ONLY=1 \
ZEVA_BEHAVIOR_READOUT_VERIFY_SAMPLES=4 \
ZEVA_BEHAVIOR_READOUT_FAST_COLLATE=1 \
ZEVA_BEHAVIOR_READOUT_BATCH_SIZE=4 \
ZEVA_BEHAVIOR_READOUT_WORKERS=8 \
ZEVA_BEHAVIOR_READOUT_PROFILE_EVERY=50 \
bash scripts/train/06_build_behavior_bank.sh
```

可单独调整批大小和工作进程数，二卡 8 workers/rank 共 16 个 CPU worker；32 核不应默认全部给 DataLoader，CPU/共享视频存储、`/dev/shm` 和 PPU 才能决定最优配置。`ZEVA_BEHAVIOR_READOUT_BATCH_SIZE` 默认 2、`ZEVA_BEHAVIOR_READOUT_WORKERS` 默认 2，`ZEVA_BEHAVIOR_READOUT_FIRST_FRAME_ONLY` / `ZEVA_BEHAVIOR_READOUT_FAST_COLLATE` 默认 0。先在 batch 4 下比较吞吐；如果 PPU 处理时间占主导且显存允许，再测试 batch 8。不要单靠减少 batch 数推算加速倍数。

启用首帧路径时，当前 rank 在 CTE prototypes 之前自动对 `ZEVA_BEHAVIOR_READOUT_VERIFY_SAMPLES` 个真实 episode 执行旧路径/新路径对照，确保 frame image、文本上下文、mask 与各类身份标识**逐元素完全一致**，否则 fail-closed。检查通过输出 `[rank N] first-frame readout equivalence PASSED`。无需自动混合有差异的两种缓存。`ZEVA_BEHAVIOR_READOUT_PROFILE_EVERY=50` 会报告 `wait_data`、`h2d_forward_d2h` 和 `throughput`（episode/s），用于确认数据等待和 FastWAM 前向谁是瓶颈。首帧优化针对 CPU 解码/预处理，不保证 FastWAM 本身的 forward 变快；真实速度须在目标 PPU 实测。

如果已有 06 作业仍在 readout 阶段：不要在同一产物路径再开一个作业。停止后，若还没有完整 rank 快照，该 rank 的 CTE prototypes/readout 仍会重新计算；已有且经验证的完整 rank 快照则能在恢复模式中跳过。过去 03/04/05 的产物不会重算。若只想调整数据读取并不想使用首帧快路径，设置 `ZEVA_BEHAVIOR_READOUT_FIRST_FRAME_ONLY=0` 即可。
## 08：Static Context 批量与在线检索索引不一致（2026-10-24）

08 原版一次对 512 条 readout 做静态查询，但 main_observed 的在线检索是 **BF16 autocast 下单次一条查询**，随后 FP32 计算 cosine、torch.topk、softmax。不同 GEMM batch shape 的舍入误差可能在多个高相似候选之间改变 Top-K 索引。该错误常见为 `static retrieval indices mismatch at episode ...`，不能简单扩大误差容忍度或跳过索引检查，因为缓存应与实际在线检索候选相同。

修复后 main_observed **默认** 使用 `ZEVA_STATIC_CONTEXT_MODE=serving_exact`：复用一次装入 GPU 的 bank keys/values、按索引直接排除当前 episode，保留线上单条查询的数值计算图与 `torch.topk` 排序。每 `ZEVA_STATIC_CONTEXT_BATCH_SIZE` 条才把结果传回 CPU，减少传输开销；脚本仍随机抽检 64 条真实 query，逐一与 `StaticTaskContextRetriever.retrieve()` 比较 Top-K index、score、value。legacy/非 main profile 默认保留 vectorized，可通过模式开关显式选择；`vectorized` 不保证主线中近似同分候选的索引一致，仅作性能调试。

08 在出错前没有发布最终 `static_context.pt`，原有 06 Bank 与 07 retrieval checkpoint **保持不变，无需重新生成或重训**：

```bash
cd /data/share/1919650160032350208/zjj/ICL-WAM
git pull --ff-only origin main
ZEVA_STATIC_CONTEXT_MODE=serving_exact ZEVA_STATIC_CONTEXT_BATCH_SIZE=256 \
  bash scripts/train/08_build_static_context.sh 2>&1 | tee /tmp/zeva_08_static_exact.log
```

看到 `STATIC EPISODE CONTEXT CACHE: PASSED` 再运行 `bash scripts/train/09_build_pim_cache.sh`。如果之前 08 已意外留下正式产物，请先核查来源和完整性，不要直接覆盖。`ZEVA_STATIC_CONTEXT_BATCH_SIZE` 只是结果分批写回 CPU 的大小，不会改变为在线单查询的精度策略。此修复未修改 07 checkpoint 或部署时 online retriever 语义。
## 06：末尾 NCCL 超时、O(N²) 修复与按 rank 恢复

2026-10-24 双卡日志：两个 FastWAM readout 都达到 100%，随后 rank 1 的 ALLREDUCE 超过默认 600 秒超时，rank 0 还没有进入同一 collective。已定位到 split_main_behavior_artifacts() 中的 O(N²) 模式：每个训练样本下标都调用一次会复制全体条目的 TaskContextBank.entries。新版本只读取一次 entries，再线性划分训练/验证，且把长时间 CPU 合并阶段的 NCCL barrier 换成定期打印等待状态的文件同步。不改 CTE、FastWAM 权重、视频帧数据或最终 bank/readout 格式。

**先确认旧快照有无保留：**

~~~bash
cd /data/share/1919650160032350208/zjj/ICL-WAM
source scripts/train/_common.sh
ls -lh "$ARTIFACT_ROOT/.behavior_bank.building"/rank_*.pt
ls -lh "$BEHAVIOR_BANK" "$READOUT_CACHE" 2>/dev/null || true
~~~

旧版在等待 NCCL 最后一轮 barrier 之前已经分别写出 rank 文件，但旧格式没有 CTE/base/data 来源哈希。如果两个快照存在，并确认它们来自**同一轮**数据及 checkpoint，显式允许对旧快照做严格结构与 episode 身份检查：

~~~bash
ZEVA_BEHAVIOR_RESUME=1 \
ZEVA_BEHAVIOR_ACCEPT_LEGACY_RANK=1 \
PREP_GPU_IDS=0,1 PREP_NPROC_PER_NODE=2 \
bash scripts/train/06_build_behavior_bank.sh
~~~

若显示 [rank 0/1] RESUME: reused ... ，说明相应 rank 直接复用了自己的 13,750 个 prototypes 和 initial readouts。只有一个合法 rank 文件时，只重算缺失的 rank（仍可开启仅首帧读取和 worker batching）。**旧快照的 source hash 无法追溯，所以必须由实验人员确认来源一致；文件结构、内容和 episode 顺序仍会自动验证。** 不要改动原 rank 文件，也不要同时启动两个 06 作业。

新生成快照的 resume_contract 绑定 checkpoint SHA、latent manifest、stats、semantic map、success lineage、episode 坐标及视频/文本参数，之后可只用 ZEVA_BEHAVIOR_RESUME=1 自动校验复用。每张卡先写临时 rank 文件，重读验证，再原子替换；CPU 合并阶段不再进行等待 600 秒的 NCCL collective。等待状态每 30 秒打印，最长时长由 ZEVA_BEHAVIOR_FILE_WAIT_SECONDS（默认 3600 秒）指定。若 rank 0 本身发生磁盘错误或被系统杀死，rank 1 的等待也不能修复该错误。

BEHAVIOR BANK: PASSED 及两个正式产物出现后，才可以运行 07。

## 03 的双卡吞吐优化与测量

主线 VAE latent 的来源、PyAV 帧选择、数值转换、缓存数组和进度协议不改变。此优化**没有**重新引入磁盘预分配，也不能解决用户/项目 quota 限制。

原来的 `identity_collate` 将包含图像的 Python 对象列表交回主进程，再执行 `torch.stack`；虽然 DataLoader 开启 `pin_memory`，自定义对象并不会自动组成 pinned tensor batch。设置 `LATENT_FAST_COLLATE=1` 后，在 worker 中按原顺序拼接纯 Tensor 字典，使 DataLoader 能直接 pin 这些张量，再用 `non_blocking=True` 送到设备。这不改变输入值、顺序或 BF16 缓存表示。该选项默认关闭，需在实际机器上对比后开启。

建议先维持 batch=8，仅在已有完整旧缓存且确认作业停下后，使用相同 `world_size=2` 续算：

```bash
LATENT_RESUME=1 PREP_GPU_IDS=0,1 PREP_NPROC_PER_NODE=2 \
LATENT_FAST_COLLATE=1 LATENT_BATCH_SIZE=8 NUM_WORKERS=8 \
LATENT_PREFETCH_FACTOR=2 LATENT_PROFILE_EVERY_BATCHES=50 \
  bash scripts/train/03_precompute_cte_latents.sh
```

会每 50 个 batch 输出 `wait_data`（CPU 解码/等待）、`h2d_vae_d2h`（传输与 VAE）、`memmap_flush`（数组写入与周期 flush）和 `throughput`（该 rank 的 windows/s）。两个 rank 速度相近时，汇总吞吐约为它们之和。与旧配置比较时，**优先比较 windows/s，不能只比较不同 batch size 的秒/batch**。

如果 `wait_data` 占主导，再测试 workers 4/6/8；如果 `h2d_vae_d2h` 占主导且显存充裕，再在相同 2 rank 下测试 `LATENT_BATCH_SIZE=16`。更高 workers/prefetch 可能增大 CPU/RAM、`/dev/shm` 和网络存储压力，导致反而变慢或 DataLoader bus error；先检查 `df -h /dev/shm`。批大小和 workers 可以改变，**生成 world size 不能从 2 改到其他值**，并且绝不能让两次作业同时写同一个缓存目录。

数据仍按成功 episode 来源及既有 `progress.json.next_offset` 恢复。增设分项计时不改变任何源数据、checkpoint 或 loss；若运行数小时后再遇 `Errno 122`，需优先处理 quota 而不是调大 worker。
### 可选的相邻窗口 PyAV 顺序解码（实验性）

如果两卡的性能日志显示 `wait_data` 主导，可以额外试验 `LATENT_STREAMING_PYAV=1`。原数据加载器在每个 H32 窗口都对三路相机重新打开视频、seek 到关键帧并解码；同一个 DataLoader worker 收到同一 episode 的连续窗口时，新路径会复用各路 PyAV VideoReader，并携带少量上一个窗口的边界帧。不同文件与超出缓冲范围的乱序请求会重新打开或 seek。此项**默认关闭**，不修改 CTE、VAE、动作、输入帧尺寸、validity mask 或缓存 schema。

为了避免解码优化悄悄改变像素，配置 `LATENT_STREAMING_VERIFY_EVERY=N` 会每 N 次读取再次调用原始 PyAV 解码器并执行逐像素 `torch.equal` 对照，发现不一致立刻报错；代价是额外的解码开销。抽检并不能证明每一帧都一致，因此首次调试应使用 `1` 检查较短的一段，然后在完成验证、停止作业后改成 `128` 做正式吞吐对比。当前并没有在目标 PPU 和真实 mp4 上完成逐帧验证或速度实测，不能直接声称加速倍数。

```bash
# 与原生成相同的两个 rank，保留已提交的部分缓存
LATENT_RESUME=1 PREP_GPU_IDS=0,1 PREP_NPROC_PER_NODE=2 \
LATENT_FAST_COLLATE=1 LATENT_BATCH_SIZE=8 NUM_WORKERS=4 \
LATENT_STREAMING_PYAV=1 LATENT_STREAMING_VERIFY_EVERY=128 \
LATENT_STREAMING_MAX_OPEN=6 LATENT_PROFILE_EVERY_BATCHES=50 \
  bash scripts/train/03_precompute_cte_latents.sh
```

`LATENT_STREAMING_PYAV=0` 可以随时退回原始逐窗口 PyAV；两种模式使用相同的进度、2-rank 分片与输出格式，不可同时运行两个写任务。该路径优化的是视频读取，不会解决 `Errno 122`（磁盘配额不足）。读完后请对比两张卡的 `throughput` 汇总及 `wait_data`；若出现数值检查失败，应改回原始模式并检查产生分歧的视频时间戳。
## RoboTwin：分别报告单次、恢复和稳定性

入口 [`../eval_zeva_main_robotwin.py`](../eval_zeva_main_robotwin.py) 使用 `sim_robotwin_zeva_main`，默认按顺序在一个 GPU 上运行同权重 memory off/on。参数见 `python scripts/eval_zeva_main_robotwin.py --help`；需要 base、CTE、A/B addon、stats、success manifest、bank 和 retrieval checkpoint。

| 参数 | 用途与解释 |
| --- | --- |
| `--protocol single_attempt` | 每个随机场景清记忆、只试一次，报告独立 `SR@1` |
| `--protocol stop_on_success --max-attempts 4` | 同场景/指令重试，首成功即停；默认关闭 replay。同权重 memory on/off 的 `CSR@k` 是失败经验收益主实验，但不能仅凭累计率上涨证明失败记忆有效 |
| `--protocol fixed_attempts --max-attempts 6` | 所有场景均试满 K 次，报告逐次成功与首次成功后的稳定率 |
| `--success-replay` | 仅 fixed 模式显式启用；只有首次确认成功后的匹配 phase/action trace 可以回放，相关稳定收益不能称为纯失败学习 |

主实验应比较相同 checkpoint 的 memory on/off。这里实际切换 PIM 条件，两组均保留 global context、当前 attempt 的 BIT 和 action prior。`CSR@4` 表示四次以内至少成功一次，不能写成单次 `SR=...`。主线使用原 main 已公开的 `base_seed + 1000 * attempt_id + replan_index` 请求种子公式，attempt_id 从 0 开始；两组相同 attempt/replan 的种子不因前次提前成功或轨迹长短而错位。每次真实请求的种子写入 attempt 日志，报表检查完整序列；旧 Ego 的连续 RNG 仅保留在独立 Ego 入口，旧主线 v1 连续 RNG 日志不能当作新 v2 结果使用。

重复抽样本身也会提高累计成功率；main PIM 还在当前 attempt 内积累 observed effects，因此进一步隔离“此前失败轨迹内容”的贡献仍需要共享第一次失败的对照。匹配请求种子不保证两组访问相同状态，也不声称与不同主干/硬件的原 Zeva 产生逐元素相同噪声。

原 main 的公开 `attempt_protocol` 要求一次丢弃 warmup。当前每个 policy 实例在首次真实请求的冻结条件准备完毕后，以相同 request seed 做一次只计算模型的 warmup，再正常推理；不执行环境动作、不写 BIT/PIM/replay，也不推进请求种子或 replan 计数。attempt/episode reset 不重复 warmup。日志记录真实 warmup 次数与 seed。其 `policy_instance` 作用域和调用时点是本地客户端的明确实现，未公开的原 RoboCasa client 不作逐调用复现声明。

部署始终是固定参数的前向计算与记忆写入。单卡顺序评测不加载两个并行 foundation，但 32 GB 显存、速度、成功率仍需在真实机器上测量；本目录的 CPU/dry-run 测试不代表已经完成离线训练或 RoboTwin rollout。
