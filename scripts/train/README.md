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

## 03 的存储预算与配额中断恢复

`OSError: [Errno 122] Disk quota exceeded` 表示写入受到配额限制。用户、项目、容器或存储平台的配额可以先于整个文件系统耗尽；`df -h` 显示大量空闲并不能排除此错误。日志中的 torchvision 视频接口弃用警告与此错误无关。

03 保存每个窗口的 BF16 latent（以 uint16 原始位存储）、FP32 action 和 uint8 validity mask。以默认 `[9, 48, 24, 20]` latent 形状为例，每窗口数组 payload 为：

```text
9 × 48 × 24 × 20 × 2 + 8 × 4 × 14 × 4 + 9 + 8 × 4 = 416,553 bytes
199,534 windows × 416,553 bytes = 83,116,486,302 bytes ≈ 77.41 GiB
```

这不包括 NPY header、完整索引、来源 metadata 和文件系统开销。实际预算以缓存 `rank_meta.json` 的形状与窗口数为准。为这个规模的新缓存安排约 **90–100 GiB 可用容量和相应配额**比较稳妥；后续其他 cache、checkpoint 和日志需要另外规划。已有部分缓存续算时，应根据已分配块数估计还需增加的空间，不必把完整容量再次全部算为新增空间。减少 `LATENT_BATCH_SIZE`、`NUM_WORKERS` 或调整 GPU 数不会减少相同数据集的完整缓存容量。

### 先只读检查，不删除已有缓存

确认原任务已退出，更新代码后运行：

```bash
source scripts/train/_common.sh
"$PYTHON_BIN" scripts/inspect_zeva_cte_latent_cache.py --cache-root "$LATENT_CACHE"
df -h "$LATENT_CACHE"
df -i "$LATENT_CACHE"
du -sh "$LATENT_CACHE"
du -sh --apparent-size "$LATENT_CACHE"
# 系统支持时检查；也可能需要在平台控制台查看项目/容器配额。
if command -v quota >/dev/null 2>&1; then quota -s; fi
```

检查工具只读 JSON、NPY header 和文件分配信息，不导入 NumPy/Torch/CUDA，不读完整 latent，也不修改文件；可加 `--json` 输出结构化报告。它显示每 rank 的 `committed_windows`、完整数组 payload、逻辑长度、已分配块、剩余分配估计，以及是否缺失 shard。它不替代来源兼容性验证，也不证明 payload 内容完整或配额充足。

文件通过 mmap 预先建立完整逻辑长度，因此 `du --apparent-size` 接近完整大小不等于计算完成。新版预分配也可能在计算开始前把整个数组范围分配好；**已提交进度只取 `progress.json` 的 `next_offset`**。默认每 8 个 batch 先 flush 数组再原子提交进度。旧版若在写 `progress.json.tmp` 时触发配额错误，最后成功提交的 `progress.json` 通常仍在；不要用进度条的 batch 数、文件大小或 `.tmp` 中的内容手改 offset。

### 路线 A：扩容或释放足够配额后，原地续算

保持原数据、来源文件、VAE 与数值配置一致，保留完整缓存目录。**未完成的缓存必须用原生成 world size 续算**；本次日志为 `rank 0/2`、`rank 1/2`，因此使用两个进程：

```bash
LATENT_RESUME=1 PREP_GPU_IDS=0,1 PREP_NPROC_PER_NODE=2 \
  bash scripts/train/03_precompute_cte_latents.sh
```

物理 GPU ID 可以换成实际可用的两个设备，rank 数仍需为 2。先等 03 输出 `VAE LATENT CACHE: PASSED`，再运行 `bash scripts/train/04_train_cte.sh`。完整缓存的读取不要求训练 GPU 数与生成 rank 数一致，04 可以按配置使用 16 GPU。

部分 rank 已完成、其余 rank 还需数小时时，完成者仍要等待统一收尾。03 的 `LATENT_PROCESS_GROUP_TIMEOUT_SECONDS` 默认 `86400`（24 小时），避免正常的长时间等待被通信超时中断；预计单次等待超过一天时可显式增加。它不改变数据分片，也不替代异常处理。

### 路线 B：迁移完整部分缓存后续算

目标必须有足够的实际容量和用户/项目配额；在同一配额域内改目录名通常无效。下面的 `NEW_CACHE` 是示例，需要换成实际已确认可用、可持久保存的挂载路径，使用与 `OLD_CACHE` 不同的新建空目录，不与其他缓存合并。停止旧任务后复制整个目录，保留数组、所有 `rank_meta.json`、`progress.json` 和已有完成标记：

```bash
source scripts/train/_common.sh
OLD_CACHE="$LATENT_CACHE"
NEW_CACHE=/your_verified_storage/iclwam/cte_latents
mkdir -p "$NEW_CACHE"
rsync -aS --info=progress2 "$OLD_CACHE/" "$NEW_CACHE/"
# 必须确认 rsync 成功退出后再继续；保留旧目录，暂不删除。
"$PYTHON_BIN" scripts/inspect_zeva_cte_latent_cache.py --cache-root "$NEW_CACHE"

export LATENT_CACHE="$NEW_CACHE"
LATENT_RESUME=1 PREP_GPU_IDS=0,1 PREP_NPROC_PER_NODE=2 \
  bash scripts/train/03_precompute_cte_latents.sh \
    "hydra.run.dir=\"$NEW_CACHE/precompute_logs\""
```

`rsync -S` 保留未写入区域的稀疏特性；复制耗时、实际占用仍取决于文件系统。不要在复制未完成或检查报错时开始续算。上述 Hydra 参数也把本次运行日志放到新存储，避免原配额已满时日志先写失败。把新 `LATENT_CACHE` 写入实际使用的 `ICLWAM_TRAIN_ENV`，让 04 及后续阶段读同一缓存；若该配置文件已有硬编码旧路径，需要同步修改。后续训练产物也要使用配额足够的 `RUN_ROOT`；仅迁移 latent 不会自动改变其他输出路径。

### 新版预检与可选预分配

03 根据各 rank 的实际数组形状计算**全体 rank**还需分配的空间，并检查文件系统可用量及元数据余量（至少 256 MiB，或完整 payload 的 1%，取较大值）。默认 `LATENT_RESERVE_SPACE=1` 会在批量编码前用 `posix_fallocate` 预留数组范围，尽早暴露 `EDQUOT` / `ENOSPC`；它不会截断已有数组或改写已提交窗口。失败时旧进度仍保留，但操作可能已分配部分块，检查实际占用后再重试。

文件系统明确不支持预分配时会警告并继续保留容量检查；意外错误不会悄悄忽略。特殊存储上可显式 `LATENT_RESERVE_SPACE=0` 关闭预分配，文件系统容量检查仍执行。**关闭预分配不会解决配额不足**，只会把可能的失败推迟到写入阶段。即使预分配成功，也不能保证后续 metadata 写入或其他进程占用后的剩余配额。

此次存储修复不改变模型、窗口内容、BF16 位表示或 `zeva_main_v2` 产物合同；符合来源与存储校验的现有 main-v2 部分缓存可以续用。前述 main-v1 → main-v2 的重建要求仍然适用，但不因这次存储修复新增重建要求。

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
