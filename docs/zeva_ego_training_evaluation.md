# ICL-WAM：Zeva-Ego 预测 CTE、语言检索、联合训练与 RoboTwin PIM 评测

本指南对应新的 `current_prediction` 路径。参考代码固定到 [Zeva `feature/zeva_ego` 的 `bae85ad753d3afe4f2e56762373d48ac0267e101`](https://github.com/air-embodied-brain/Zeva/tree/bae85ad753d3afe4f2e56762373d48ac0267e101)，不依赖分支持续变化后的实现。此前只有独立 prefix 的版本使用 `latest_completed`，其 [旧指南](zeva_ego_cross_attempt.md) 保留用于复现。

当前默认对齐参考的 **Stage2A foundation+EAP联合训练、Stage2B分阶段训练、完整train bank检索和发布版在线PIM压缩**。这不等于除了token注入就完全相同；完整差异、源码证据与验证边界见 [对齐审计](zeva_ego_alignment_audit.md)。FastWAM仍使用自己的joint动作、时间尺度、语言输入空间和policy损失。

从 `b8b6e84` / `68a6fa7` 迁移时，可复用来源匹配的CTE、boundary以及语言encoder/features/bank/classifier；重新构建static contexts（`train_query_exclusion=none`）与PIM train/validation cache（v3、round索引），然后按新配方重训Stage2A/B。CTE架构与损失不变，无需因此重训CTE或分类器。更早的image+text版 `5344a78` 还需重建语言查询产物；4-step/completed-effect版须先训练预测CTE。原FastWAM base、原始数据、归一化统计、语义任务表和匹配的policy文本embedding cache继续可用。旧完整state不能作为改变配方后的精确续训。

## 1. 实际参考了哪些内容

| 参考文档 / 源码 | 对本项目的修改 |
| --- | --- |
| [分支 README](https://github.com/air-embodied-brain/Zeva/blob/bae85ad753d3afe4f2e56762373d48ac0267e101/README.md)、[运行环境](https://github.com/air-embodied-brain/Zeva/blob/bae85ad753d3afe4f2e56762373d48ac0267e101/docs/README.md) | 分开 CTE/BIT/EAP/PIM 与独立的 `pipelines/ego_action_encoder`；后者不是这里需要训练的 CTE。正式 CTE 要求 Mamba CUDA 扩展，不再静默退回 GRU。 |
| [CTE / EAP 实现](https://github.com/air-embodied-brain/Zeva/blob/bae85ad753d3afe4f2e56762373d48ac0267e101/src/openpi/zeva/cte_eap.py)、[CTE 训练器](https://github.com/air-embodied-brain/Zeva/blob/bae85ad753d3afe4f2e56762373d48ac0267e101/scripts/robotwin/train_cte.py) | 三路 RGB、三流因果 Mamba、上一段已执行动作、EMA 视觉变化目标、epoch 训练与独立验证。EAP prior 由 phase 查询 global 产生的 8 个 waypoint。 |
| [CTE artifact 导出](https://github.com/air-embodied-brain/Zeva/blob/bae85ad753d3afe4f2e56762373d48ac0267e101/scripts/robotwin/export_cte_artifacts.py) | task memory 仅用 train episodes；仅池化有完整下一段执行目标的状态；raw task-head / phase 分别求均值后再做 L2 归一化。在线当前预测 trace 与 task prototype 的池化范围明确区分。 |
| [数据合同](https://github.com/air-embodied-brain/Zeva/blob/bae85ad753d3afe4f2e56762373d48ac0267e101/scripts/robotwin/README_DATASET.md) | episode 级 train/validation 分离，明确相机、动作尺度、真实执行长度；保留 FastWAM 的 joint14，不能混用参考 pi0.5 的 relative-EEF16 统计。 |
| [PIM policy](https://github.com/air-embodied-brain/Zeva/blob/bae85ad753d3afe4f2e56762373d48ac0267e101/src/openpi/zeva/pim_policy.py)、[PIM 训练](https://github.com/air-embodied-brain/Zeva/blob/bae85ad753d3afe4f2e56762373d48ac0267e101/scripts/robotwin/train_cross_attempt_pim.py) | 保存每个真实决策边界的预测effect；当前attempt结束后提交历史。先PIM-only，再foundation+EAP+PIM；学习率分别1e-6 / 1e-5 / 5e-5，历史模式按optimizer step固定循环。 |
| [TaskLanguageMemory](https://github.com/air-embodied-brain/Zeva/blob/bae85ad753d3afe4f2e56762373d48ac0267e101/src/openpi/zeva/cte_eap_policy.py) | 冻结纯语言输入特征和分类器；预测task后，用同任务CTE mean key查询同任务train demos。top-5 cosine softmax不除以0.07。 |
| [Stage2A训练](https://github.com/air-embodied-brain/Zeva/blob/bae85ad753d3afe4f2e56762373d48ac0267e101/scripts/robotwin/train_cte_eap.py) | foundation LR5e-6、EAP LR5e-5、固定5000步/global256，无warmup/decay；joint Stage2A成为Stage2B完整父模型。 |
| [发布版PIM加载](https://github.com/air-embodied-brain/Zeva/blob/bae85ad753d3afe4f2e56762373d48ac0267e101/src/openpi/zeva/pim_release.py) | 在线默认similarity_merge，phase/effect余弦加权相似度、阈值0.95；不能仅看训练uniform采样就推断部署也uniform。 |
| [评测指南](https://github.com/air-embodied-brain/Zeva/blob/bae85ad753d3afe4f2e56762373d48ac0267e101/scripts/robotwin/README_EVALUATION.md)、[checkpoint 包合同](https://github.com/air-embodied-brain/Zeva/blob/bae85ad753d3afe4f2e56762373d48ac0267e101/scripts/robotwin/README_CHECKPOINT.md) | 冻结 10×20 manifest、最多四次尝试、成功即停、固定分母、8 个连续 RNG slot、reset / video / artifact 身份审计。参考发布权重属于 pi0.5，不能直接当 FastWAM addon 加载。 |

### 保留的 FastWAM 适配边界

| 项目 | Zeva-Ego 参考 | 本次 ICL-WAM |
| --- | --- | --- |
| 底座 | pi0.5 | FastWAM video/action experts 与原有 MoT |
| 动作 | relative EEF16，policy horizon50，执行15 | joint14，policy horizon32，实际执行24 |
| CTE 图像 | 三路独立 RGB224 | 同样三路独立 RGB224；不使用 Wan VAE latent 或融合图像裁块 |
| policy 图像 | 参考 pi0.5 输入 | 保持 FastWAM 的三相机融合图 384×320 |
| phase / effect / global | raw interaction256 / predicted effect256 / task context256 | 对齐；检索 key128、辅助 phase head128 不与 raw phase 混淆 |
| Stage2A | foundation + EAP联合训练，固定LR/5000步/global256 | 新joint默认对齐；frozen保留为显式消融，EAP训练配方相同 |
| Stage2B 第500步后 | foundation + EAP + PIM 都更新 | `foundation_mode=joint`对齐；`frozen`保留EAP/PIM更新，仅冻结foundation，其他条件相同 |
| task retrieval query | frozen task-language classifier 先预测task，再在该task的train demos内检索 | 对齐同一推理机制；输入特征来自固定Wan文本输入embedding表，分类器须在本地重训 |
| 连续模型随机流 | 参考模型及其 CUDA 随机实现 | 显式 CPU Torch 初始动作噪声流，记录 `torch_continuous_slot_v1` |

联合训练不改变检索输入：语言encoder、分类器和train bank各自固化并绑定SHA256，均不读正在更新的video/action hidden states。部署按addon中记录的相对文件名自动加载配对foundation权重，缺失或身份不匹配时明确报错。

参考分支发布了语言分类器的consumer/MLP实现，并从既有handoff加载分类器；没有发布可直接照搬到FastWAM的独立分类器训练入口。因此这里**按相同的推理机制重建本地分类器**，采用Wan的固定输入embedding和本项目CTE key space。下文分类器的2000步、batch32等是本项目重建配方，不声称是Zeva原分类器的公开训练配方，也不使用不兼容的pi0.5权重。

## 2. 当前 context 怎样影响 action

在重规划边界 `t`，CTE 输入 `I_t` 和上一段实际执行的动作 `A_(t-1)`；首次用 SOS。训练时本段动作 `A_t` 和下一边界图像 `I_(t+1)` 只作为预测目标：

```text
z_t, e_hat_t = CTE.step(I_t, A_(t-1), attempt_local_state)
target_effect_t = stop_gradient(EMA_vision(I_(t+1)) - EMA_vision(I_t))
c_t = PIM_readout(z_t, phases_from_completed_attempts, predicted_effects_from_completed_attempts)
g_t = frozen_language_retrieved_context + U(c_t)
prefix = [G(g_t), E(e_hat_t), P(c_t)]
prior_mean, prior_std = EAP_prior(g_t, z_t)
```

`G`、`E`、`P` 是三个独立 token，action 通过 FastWAM 的 mixed attention 分别读取它们的 K/V。prefix 和干净首帧不能读取未来 video/action。另一路把 prior mean 投影到 ActionDiT hidden space；prior 使用 `global → 8 waypoints → phase attention → Gaussian`，不再把 BIT 混进 phase query。

第一次重规划已经有有效 `e_hat_t`，无需等待16步 completed effect。生成动作后将 `(z_t,e_hat_t)` 追加到当前 pending trace；当前 query 不读取自身新条目。attempt结束后才提交PIM，新episode清空历史。在线默认先做attempt内similarity merge，再以round均匀索引将每个attempt压至多64条，读取保留attempt时再做跨attempt merge；centroid按计数加权，保留原向量幅度。最多读取3个已结束attempt，在四次尝试协议下与参考可读取的历史次数一致。末尾不足24步不制造额外CTE target，终止后也不额外预测不存在的下一次决策。

`new_off` 关闭 `U(c_t)` 和独立 `P(c_t)`，仍保留当前预测effect、语言检索global、phase和EAP。它表示**同一Stage2B权重关闭PIM**；joint使用继续微调的foundation，frozen保留其Stage2A父模型的foundation。如果父模型joint训练过，frozen也必须加载配对权重，不能回退原始base。两种模式中的EAP均在后半段更新，所以off不等同于Stage2A或原始FastWAM base。

新路径没有参与输出的标量 `pim_gate`。应观察实际 action 变化、prefix 有效性、attention 读出与成功率，不能继续用旧 gate 数值判断记忆是否有效。

## 3. 环境与统一路径

先完成本项目原有 FastWAM / RoboTwin 安装及资产准备。正式新 CTE 另外需要可在当前 Torch/CUDA 环境运行的 `mamba_ssm` 和 `causal_conv1d`。参考仓库 [环境说明](https://github.com/air-embodied-brain/Zeva/blob/bae85ad753d3afe4f2e56762373d48ac0267e101/docs/README.md) 使用其 vendored 扩展；若使用该版本，固定上面的 commit，先安装 `causal-conv1d`，再安装 `mamba`，不要混用另一套 CUDA 编译产物。ResNet18 使用 torchvision 预训练权重，首次运行需已缓存或可下载。

从本仓库根目录执行。路径均替换为本机实际的绝对路径；当前新流程明确支持一个 RoboTwin dataset root。

```bash
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export DIFFSYNTH_MODEL_BASE_PATH=/absolute/path/to/fastwam_pretrained_assets

ICLWAM_DATA=/absolute/path/to/robotwin2.0
ICLWAM_STATS=/absolute/path/to/robotw_stats.json
ICLWAM_TASKS=/absolute/path/to/semantic_task_map.json
ICLWAM_TEXT=/absolute/path/to/text_embeds_cache
ICLWAM_BASE=/absolute/path/to/fastwam_base.pt
ICLWAM_RUN=/absolute/path/to/runs/iclwam_reference_alignment_v2
ICLWAM_CONDITIONS=/absolute/path/to/episode_conditions.json

DATA_ARGS=(
  "data.train.dataset_dirs=[$ICLWAM_DATA]"
  "data.train.pretrained_norm_stats=$ICLWAM_STATS"
  "data.train.semantic_task_map_path=$ICLWAM_TASKS"
  "data.train.text_embedding_cache_dir=$ICLWAM_TEXT"
)
```

`semantic_task_map.json` 必须提供规范的 RoboTwin task identity，不能把不同语言表达的 `task_index` 当成不同任务。归一化统计在 CTE、policy 训练与部署间保持相同。完整参考PIM课程还要求真实的 `episode_conditions.json`，格式为 `{canonical_episode_id: condition_string}`，例如数据实际所属的Clean/Randomized条件；不能根据success、task或随机数伪造条件。

## 4. Stage1：完整 episode 的预测 CTE

```bash
python scripts/train_zeva_ego_cte.py \
  task=robotwin_zeva_fastwam_ego_cte_3cam_384 \
  "${DATA_ARGS[@]}" \
  device=cuda output_dir="$ICLWAM_RUN/cte"
```

这条入口为**单进程、单 GPU**，不要改成多卡 `torchrun`。episode pair / 无放回 epoch 语义在该入口中明确实现；它拒绝 `WORLD_SIZE>1`，避免伪造已实现的数据并行配方。

本入口使用配置dataset中全部已索引且长度足够的episodes。参考训练入口则接收外部必填task-subset JSON；公开代码不能确定双方实际训练任务恰好相同，更不能由50任务语言分类器推断CTE也训练50任务。训练前对照实际task/condition清单；本地Stage1保存这些数据身份。本项目默认validation按semantic task分组，参考按condition×task分组，具体划分成员不保证相同；需要固定成员时使用下文显式split文件，不悄悄重新划分既有实验。

| 训练项 | 默认值 / 行为 |
| --- | --- |
| 长度 | 80 个完整 epoch，覆盖每个 train episode，非5000次随机窗口抽样 |
| batch | 8 个 episode，同任务相邻成对；无放回，保留尾部 episode |
| 数据划分 | seed1000，按任务5% episode holdout；每任务取 `ceil(0.05×N)` 并限制为1到`N−2`个 validation，保证至少2 train + 1 validation，不足时报错 |
| CTE | 三流、4层、hidden256、Mamba；三相机共享预训练 ResNet18 后融合 |
| EMA | 0.99；target encoder不更新梯度，BN统计在训练时也固定，防止跨时间视觉泄漏 |
| AdamW | online vision LR1e-5，其余1e-4，weight decay1e-4，gradient clip1；不使用 scheduler |
| 损失 | `0.1 action + 0.2 vision + 2 global + 1 phase + 0.2 effect` |
| 验证 / 保存 | 每5个epoch及最终epoch做独立验证；每个完整epoch保存权重、优化器、RNG、数据/源码身份 |
| Stage2入口检查 | 至少80完整epoch；该epoch有独立验证；action、vision、effect 相对各自基线的 gain均≥0.05 |

输入张量为 `images[B,T+1,3,3,224,224]`、`executed_actions[B,T,24,14]`、`valid[B,T]`。每个时刻的三张原始 RGB 独立缩放并映射到 `[-1,1]`；action 是按 FastWAM dataset stats 归一化后的有序24步动作。三流每层先分别进行因果时间建模，再在同一边界按 vision←previous-action、action←vision、interaction←(vision,action) 的顺序交互。action head 从 action stream 预测本段动作，vision head 从 vision stream 预测下一边界 EMA visual feature，effect head 从 interaction `z_t` 预测 EMA feature 差。

训练只保留完整24步动作及其真实下一边界观测。若一个episode有 `L` 条带观测记录，只在 `raw_step+24<L` 时构造这一段 target；最后不足24步不补零当作完整动作。批次padding只出现在轨迹尾部，并在所有损失里排除。模型可返回最后一个真实边界的 phase/effect，但该边界没有被伪造出的下一段监督目标。

三项基线分别为**归一化空间**的零动作、视觉保持不变、零视觉变化；gain 为 `1 − model_MSE / max(baseline_MSE,1e-12)`。global loss 使用同任务 episode 正样本，self 不作为正样本但保留在 softmax 分母，遵循参考实现；phase loss 使用有效时间位置的 identity 分类。action BC loss 对14个坐标求和后在24个执行时间位置上平均；视觉/effect loss对feature坐标求和。实现见 [ego_cte.py](../src/fastwam/zeva/ego_cte.py)，完整训练与readiness约束见 [ego_cte_training.py](../src/fastwam/zeva/ego_cte_training.py)。

划分在解码RGB前完成并保存在 `episode_splits.json`。已有固定划分可以用 `cte_episode_split_path=/absolute/path/to/episode_splits.json`，文件必须是 `{"train":[canonical_episode_ids],"validation":[canonical_episode_ids]}`，两部分互斥且覆盖本次索引到的全部episode。

通过验证后，通常使用 `cte_epoch_080.pt`。训练满80epoch本身不代表通过：后续 builder 会重新计算 readiness 条件；未达标应检查数据、收敛或继续训练。续训示例：

```bash
python scripts/train_zeva_ego_cte.py \
  task=robotwin_zeva_fastwam_ego_cte_3cam_384 "${DATA_ARGS[@]}" \
  device=cuda num_epochs=100 \
  resume="$ICLWAM_RUN/cte/cte_epoch_080.pt" \
  output_dir="$ICLWAM_RUN/cte"
```

续训验证配置、完整episode split、归一化和源码身份，并恢复 Torch CPU/CUDA、Python、NumPy 与 DataLoader RNG。checkpoint在完整epoch结束后保存，因此不把半个epoch当作完成。这里允许以相同配方继续到80之后；参考发布选择固定epoch80，本项目续训后的checkpoint仍必须有当前epoch的独立验证并通过同样gate。CPU测试仅可显式使用 `gru_test` / `tiny_test` 和 `cte_smoke_steps`；smoke不写模型checkpoint，测试后端即使被手动标记promotable也会被正式loader拒绝。Stage1不加载Wan VAE或文本embedding。

## 5. 重建 boundary cache、task memory 与 PIM cache

先设置新产物路径。以下命令是全新构建流程，按顺序执行，不要覆盖旧实验目录。复用已有预测版产物时，CTE、BOUNDARY可指向身份匹配的原产物；本轮新默认的PIM_TRAIN、PIM_VAL、STATIC必须使用新空路径重新构建。已有语言encoder/features/bank/classifier可跳过相应构建/训练；旧image+text版必须先完成语言产物迁移。

```bash
ICLWAM_CTE="$ICLWAM_RUN/cte/cte_epoch_080.pt"
ICLWAM_BOUNDARY="$ICLWAM_RUN/boundary_cache"
ICLWAM_BANK="$ICLWAM_RUN/task_memory.pt"
ICLWAM_LANGUAGE_ENCODER="$ICLWAM_RUN/task_language_encoder.pt"
ICLWAM_LANGUAGE_FEATURES="$ICLWAM_RUN/task_language_features.pt"
ICLWAM_HEAD="$ICLWAM_RUN/task_language_classifier.pt"
ICLWAM_STATIC="$ICLWAM_RUN/static_contexts.pt"
ICLWAM_PIM_TRAIN="$ICLWAM_RUN/pim_train"
ICLWAM_PIM_VAL="$ICLWAM_RUN/pim_validation"

ARTIFACT_ARGS=(
  "ckpt=$ICLWAM_BASE"
  "model.zeva.cte.checkpoint=$ICLWAM_CTE"
  "model.zeva.cache.path=$ICLWAM_BOUNDARY"
  "model.zeva.task_context.bank_path=$ICLWAM_BANK"
  "model.zeva.task_context.language_encoder_path=$ICLWAM_LANGUAGE_ENCODER"
  "model.zeva.task_context.language_feature_cache_path=$ICLWAM_LANGUAGE_FEATURES"
  "model.zeva.task_context.retrieval_checkpoint=$ICLWAM_HEAD"
  "model.zeva.task_context.static_episode_cache_path=$ICLWAM_STATIC"
)

python scripts/build_zeva_ego_boundary_cache.py \
  task=robotwin_zeva_ego_policy_3cam_384 "${DATA_ARGS[@]}" "${ARTIFACT_ARGS[@]}"

# policy仍需要原有文本cache；若相同模型/指令的cache存在，可复用。
# task-language分类器只使用独立导出的input embedding，不使用此policy cache。
python scripts/precompute_text_embeds.py \
  task=robotwin_zeva_ego_policy_3cam_384 "${DATA_ARGS[@]}" data.val=null +overwrite=false

python scripts/build_zeva_ego_task_memory.py \
  task=robotwin_zeva_ego_policy_3cam_384 "${DATA_ARGS[@]}" "${ARTIFACT_ARGS[@]}" \
  model.zeva.task_context.build_stage=bank

python scripts/train_zeva_task_language.py \
  task=robotwin_zeva_ego_policy_3cam_384 "${DATA_ARGS[@]}" "${ARTIFACT_ARGS[@]}"

python scripts/build_zeva_ego_task_memory.py \
  task=robotwin_zeva_ego_policy_3cam_384 "${DATA_ARGS[@]}" "${ARTIFACT_ARGS[@]}" \
  model.zeva.task_context.build_stage=static_context

python scripts/build_zeva_cross_attempt_cache.py \
  --phase-cache "$ICLWAM_BOUNDARY" --semantic-map "$ICLWAM_TASKS" \
  --episode-splits "$ICLWAM_BOUNDARY/episode_splits.json" \
  --conditions "$ICLWAM_CONDITIONS" --history-pairing reference_nearest \
  --split train --output "$ICLWAM_PIM_TRAIN" --seed 20260919

python scripts/build_zeva_cross_attempt_cache.py \
  --phase-cache "$ICLWAM_BOUNDARY" --semantic-map "$ICLWAM_TASKS" \
  --episode-splits "$ICLWAM_BOUNDARY/episode_splits.json" \
  --conditions "$ICLWAM_CONDITIONS" --history-pairing reference_nearest \
  --split validation --output "$ICLWAM_PIM_VAL" --seed 20260919
```

boundary cache 在原始action步 `0,24,48,…` 上逐次调用与部署相同的 `CTE.step`，每个episode重新初始化递归状态，manifest声明 `encoding_path=recurrent_step_v1`。它记录 current phase、predicted effect、raw interaction/global256、**未逐边界归一化的** task-head128，声明 `retrieval_space=cte_task_head_raw_v1`。末尾不足policy horizon32的查询不参与policy监督，但真实决策预测仍可属于该轨迹的历史。

task memory 的 key/value只由CTE train episodes构建。对episode `e`，令 `S_e={t: raw_step_t+24<L_e}`，严格对应 Stage1 有完整执行目标的起点。按参考导出实现：

```text
key_e   = L2Normalize(mean(task_head(z_t) for t in S_e))  # 128
value_e = L2Normalize(mean(z_t for t in S_e))            # 256
```

归一化发生在轨迹均值之后；不能先把每个 `task_head(z_t)` 归一化再求均值。partial final decision 可以保留在在线/PIM预测历史中，但不进入上述prototype池化。bank metadata明确记录该规则，旧image+text bank需按新查询合同重建；不是给旧bank改一个mode字符串。

### 语言分类器与检索

`build_stage=bank` 导出固定tokenizer快照和文本**输入embedding表**，并从原始episode metadata读取instruction，生成包含train/validation查询的独立语言特征文件；不解码RGB，也不实例化FastWAM DiT/VAE或执行完整T5。输入先去首尾空白、把下划线和换行换为空格，再补末尾换行；最多200个token，按attention mask对输入embedding求均值。这个encoder快照在离线构建、分类器训练与在线部署间共用，不随foundation更新。

`train_zeva_task_language.py` 仅从CTE train split选择样本，再按episode划分内部80%训练/20%选择集。类别prototype只由该内部训练子集的CTE keys构建；CTE held-out validation不参与分类器梯度、prototype或checkpoint选择。归一化MLP映射到128维，按与冻结task prototypes的相似度做交叉熵；默认2000步、batch32、AdamW LR1e-4/weight decay1e-4，以内部选择集top-1、再以loss选择checkpoint。训练loss的temperature0.07只用于分类器学习，不是下面检索聚合的温度。

在线只接收instruction，依次计算：

```text
language_feature = masked_mean(frozen_input_embeddings(instruction))
predicted_task = argmax(classifier(language_feature) @ task_prototypes.T)
query = L2Normalize(mean(train_bank.keys[predicted_task]))
supports = top5_cosine(query, train_bank.keys[predicted_task])
task_context = softmax(supports.cosine_scores) @ supports.values
```

分类器的候选类别限制在bank预先声明的task集合，而不是由当前评测episode的真实task标签缩小。top-5聚合temperature为1，返回的加权context不再强制单位化；不足5个合格support时只用真实条目。该路径不接收在线task标签、episode ID、图像或phase。每次attempt缓存语言检索结果；retry清空缓存后仍由相同instruction重新计算。

新默认 `task_context.train_query_exclusion=none` 对训练和部署使用同一个完整train bank：语言预测task，再从该task全部train keys形成固定mean query并读取top-5。这与reference `TaskLanguageMemory`一致，不把validation轨迹加入bank，也不按当前query的GT task选范围。旧LOO策略可显式设置 `model.zeva.task_context.train_query_exclusion=leave_one_episode_out`；它同时从mean和support移除查询episode。构建与prepared训练必须选择相同策略，metadata不匹配会拒绝旧static cache。bank、features、encoder、classifier与static context绑定CTE/stats/semantic map、boundary/split及各自SHA256；prepared训练还检查原始instruction一致性。当前static context要求同一episode的原始instruction固定，若源数据在episode内切换instruction需先规范该数据合同。

上述语言bank/features/classifier/static构建支持CPU，不需要加载FastWAM base；语言输入embedding资产必须可用。统一命令中的`ckpt`供后续policy训练使用。PIM历史源同样仅限其他train episodes；Stage2查询split与Stage1保持一致。artifact入口均为单进程，不接收多rank `torchrun`。

两个PIM builder必须使用同一份真实条件文件。新默认 `history_pairing=reference_nearest` 按初始CTE phase余弦为每个query episode固定三条历史：同任务同condition最近、同任务同condition最远、同任务异condition最近；query自身排除，候选仅来自train split。同一episode的各个决策窗口使用同一个对应partner，不再从top-3候选中轮换。内部名称 `far` 对应参考的 `same_condition_far`。

完整参考课程要求每个query均存在same/cross候选；缺条件或缺候选会报错。只有确需做缺条件数据消融时，才在两个builder中显式增加 `--allow-reduced-curriculum`，并在两阶段prepared训练命令增加 `model.zeva.prepared_stage2.allow_reduced_curriculum=true`。此时manifest明确标记可用模式及课程缩减，调度删除不可用项；例如无cross时warmup变成5:2、后续5:2:0:2，这不是完整参考5:2:1 / 5:2:1:2实验。`legacy_candidates`配对只用于旧版复现，不能把它标为reference-nearest缓存。

新版格式：CTE=`iclwam_zeva_ego_cte_v1`，boundary=`zeva_ego_boundary_cache_v1`，prepared PIM=`zeva_stage2_cross_attempt_pim_v3`，addon=`zeva_ego_prediction_v1`。新PIM缓存明确记录reference-round均匀索引，旧v2/floor仅供显式legacy配置复现。loader还校验 `current_prediction`、`prior_variant=zeva_ego`、`memory_semantics=predicted_effect_trace`及artifact SHA256。boundary builder在RGB解码前核对Stage1保存的episode metadata指纹，防同ID的长度、offset或task漂移被错当成同一源数据。

## 6. Stage2A 与 Stage2B

```bash
PREPARED_ARGS=(
  "model.zeva.prepared_stage2.pim_cache_path=$ICLWAM_PIM_TRAIN"
  "model.zeva.prepared_stage2.validation_pim_cache_path=$ICLWAM_PIM_VAL"
)

torchrun --standalone --nproc_per_node=16 scripts/train_zeva_fastwam_prepared.py \
  task=robotwin_zeva_ego_policy_16gpu_3cam_384 \
  foundation_mode=joint \
  "${DATA_ARGS[@]}" "${ARTIFACT_ARGS[@]}" "${PREPARED_ARGS[@]}" \
  output_dir="$ICLWAM_RUN/stage2a"

ICLWAM_STAGE2A="$ICLWAM_RUN/stage2a/checkpoints/weights/step_005000_addon.pt"

torchrun --standalone --nproc_per_node=16 scripts/train_zeva_fastwam_prepared.py \
  task=robotwin_zeva_ego_pim_16gpu_3cam_384 \
  foundation_mode=joint \
  "${DATA_ARGS[@]}" "${ARTIFACT_ARGS[@]}" "${PREPARED_ARGS[@]}" \
  model.zeva.policy_checkpoint="$ICLWAM_STAGE2A" \
  output_dir="$ICLWAM_RUN/stage2b"
```

Stage2A现在默认 `foundation_mode=joint` / `ego_joint_parent`，联合更新foundation（LR5e-6）与global/effect投影、EAP prior、action-prior adapter（LR5e-5）；PIM关闭。5000步、effective batch256、AdamW weight decay1e-4、clip1、无LR warmup或decay来自参考Stage2A配方，不保证在FastWAM上已是最优值。`foundation_mode=frozen`保留同配置EAP-only对照。新preset显式设置 `zeva_stage2a_schedule=zeva_ego`；旧frozen配置缺该字段按 `legacy_warmup` 解释，保留旧5% LR warmup，包括用户当时手动选择constant的情况。续训比较归一化后的schedule语义，不能通过改constant名字悄悄更换配方。两个stage均有16GPU task（batch1×accum16×world16），普通task按4GPU×accum64设计；单卡如需同batch要显式accum256。旧冻结Stage2A不能仅改recipe元数据冒充joint parent。新frozen/joint Stage2B应从同一个新Stage2A完整bundle出发比较。

新建predictive模块显式使用 `prompt.initialization=zeva_ego` 与 `adapter.initialization=zeva_ego`：global/effect投影及PIM Q/K/V保留参考 `nn.Linear` 默认初始化，EAP prior末层normal(0,0.01)、action-prior residual投影为零，PIM global残差投影为零，独立PIM prefix沿用参考的小非零初始化。旧版额外Xavier初始化保留为 `legacy_xavier`。这影响从头训练的初始状态；完整加载旧权重不会自动获得新训练的效果。

### Foundation 单开关

Stage2A和Stage2B都设置 `foundation_mode=joint`（默认）或 `foundation_mode=frozen`。YAML同步解析 `training_recipe`和`freeze_fastwam`，运行时检查三者一致。Stage2A分别为`ego_joint_parent` / `frozen_parent`，均无warmup；以下表格是Stage2B的安排。两种B模式使用同一套 `task_language`产物、完整Stage2A parent、CTE、history curriculum、batch、seed、精度与Adam设置。

| 开关 | 实际recipe | 前500次更新 | 之后的更新 |
| --- | --- | --- | --- |
| `foundation_mode=joint` | `ego_joint_warmup` | PIM；EAP/foundation LR=0 | foundation、EAP、PIM |
| `foundation_mode=frozen` | `ego_adapter_warmup` | PIM；EAP LR=0 | EAP、PIM；foundation始终冻结 |

运行上面的同一条命令，只改成 `foundation_mode=frozen` 并指定新的 `output_dir`，即可建立同架构冻结对照。单进程冻结训练可以使用普通task `robotwin_zeva_ego_pim_3cam_384`、`foundation_mode=frozen`、`zeva_optimizer_sharding=none`，并按有效batch调整梯度累积。别名task `robotwin_zeva_ego_pim_language_frozen_3cam_384`也保留新语言检索。原来的 `robotwin_zeva_ego_pim_frozen_3cam_384`仍用于旧static检索复现，不能将它当成只改变foundation的消融。

两种新Stage2B默认2000个optimizer updates、seed20260919、AdamW betas=(0.9,0.999)、weight decay1e-4、clip1。下面的LR表对应joint；frozen只有foundation一项不参与更新：

| 更新阶段 | 参数 / LR | 每轮固定循环中的历史模式次数 |
| --- | --- | --- |
| 前500次更新 | PIM reader、`U`、`P`：5e-5；EAP与foundation LR=0 | 依次 matched×5、far×2、cross×1，无empty |
| 后1500次更新 | PIM：5e-5；EAP：1e-5；foundation：1e-6 | 依次 matched×5、far×2、cross×1、empty×2 |

joint的foundation参数组包括MoT中的video/action experts及动作路径使用的proprio encoder；VAE、完整文本编码器、CTE和语言检索始终冻结。训练损失仍为action path + EAP prior；不增加video生成loss，未参与该损失的video output head等参数不会凭空得到更新。joint的foundation和EAP从开始就在optimizer中，warmup仅通过LR=0控制，避免中途改变DDP trainable集合；LR=0不会免除梯度/Adam状态内存。video输入投影在joint时保留梯度；VAE编码仍在`no_grad`中。frozen不将foundation参数加入optimizer，但仍保留context经video K/V影响action的梯度。

每个非空训练样本只取一个previous attempt，按 `linspace(0,n-1,min(n,64)).round().long()`均匀采样；这与参考训练一致，在线部署则默认similarity merge。模式沿参考实现用 `active_schedule[optimizer_step % len(active_schedule)]`选择；先删除不可用的真实历史模式，再按从0开始的全局optimizer step索引，进入后半段也不重置step。所有rank和梯度累积microbatch共享该mode；dataset worker仅预取候选，不决定warmup进度。训练seed控制数据顺序和噪声。LR=0阶段只裁剪有效更新的PIM参数，之后联合裁剪；空模式保留current predicted effect。

上面的16GPU task为batch1 × accumulation16 × world16 = 256。普通task仍保留4GPU示例的accumulation64；不要在16卡上直接沿用它，否则有效batch变成1024。单GPU冻结训练若需相同effective batch，设置 `gradient_accumulation_steps=256`。程序记录实际effective batch。EAP损失按参考采用坐标求和后的Gaussian NLL，系数0.01；prior训练dropout0.4，推理scale0.5。

**Stage2A、frozen Stage2B和joint Stage2B全部将可训练参数保留为FP32**，计算使用bf16 autocast；通常冻结模块继续使用其原始dtype，但frozen Stage2B继承的joint父foundation在训练/续训中保留FP32。FP32准备在载入Stage2A parent之前完成，避免先将父adapter量化到BF16再升回FP32。`optimizer_betas=null`在predictive Stage2统一选择(0.9,0.999)，也可显式指定；legacy路径保持原默认值。两个expert的旧 `use_gradient_checkpointing` 固定关闭，`mot_checkpoint_mixed_attn`单独控制已支持的video prefill/action cache检查点，修复此前默认构造器开关冲突；不能同时启用compiled training layers。

### 16×96GB：普通DDP + ZeRO-1

`zeva_optimizer_sharding=zero1`使用PyTorch `ZeroRedundancyOptimizer`分片Adam状态，模型参数与梯度仍由普通DDP管理；启用 `gradient_as_bucket_view`共享梯度/bucket存储。`none`使用常规AdamW。zero1要求至少两个rank，单进程不会静默退回常规optimizer；两种foundation模式都可使用同一分片设置。

当前video+action backbone约6.02B。FP32参数+梯度+两个Adam moments全复制约89.72GiB/卡；16rank均匀分摊moments后，理想状态容量约为 `N×(4+4+8/16)` bytes，即47.66GiB/卡。实际分片按参数tensor划分，会有不均衡，且这不包含冻结模块、bf16计算副本、activation、临时buffer等，所以不是已测峰值或96GB适配保证。训练日志和实机短跑应确认实际峰值。分片不会改变部署模型的参数量。

若16卡分布在两台8卡机器，在两台机器上使用相同路径与以下命令，分别设置 `ICLWAM_NODE_RANK=0` / `1`；主节点地址须为第二台可访问的地址：

```bash
ICLWAM_MASTER_ADDR=10.0.0.1
ICLWAM_NODE_RANK=0
torchrun --nnodes=2 --nproc-per-node=8 \
  --node-rank="$ICLWAM_NODE_RANK" --master-addr="$ICLWAM_MASTER_ADDR" --master-port=29500 \
  scripts/train_zeva_fastwam_prepared.py \
  task=robotwin_zeva_ego_pim_16gpu_3cam_384 foundation_mode=joint \
  "${DATA_ARGS[@]}" "${ARTIFACT_ARGS[@]}" "${PREPARED_ARGS[@]}" \
  model.zeva.policy_checkpoint="$ICLWAM_STAGE2A" \
  output_dir="$ICLWAM_RUN/stage2b"
```

多节点必须共享同一checkpoint/state输出目录；完整state包含所有rank的RNG文件。不要让两个节点独立使用 `--standalone`，否则会形成两个无关训练作业。

每250步做独立离线验证，每500步保存。Stage2B验证使用固定非空历史模式，并为on/off复用同一flow noise/time。关注 `val_memory_gain`、`memory/prior_mean_delta_rms`、各prefix有效性/幅值和实际执行动作差异。它们用于诊断，不能替代RoboTwin成功率。

Stage2续训使用 `resume=/.../checkpoints/state/step_XXXXXX`；新配方校验warmup、学习率、seed、batch、accumulation、精度和总步数，绑定train/validation cache身份，恢复数据位置和每个rank的Python/NumPy/Torch RNG，并检查world size和完整checkpoint标记。候选历史按query身份确定，DataLoader预取不改变它。DDP尾部先按完整global batch补齐再应用恢复偏移，保持连续训练与续训的样本顺序。梯度累积跨数据epoch边界继续，不在epoch尾提前产生不足effective batch的更新。

新预测训练支持单进程或普通DDP，DDP可选择上述ZeRO-1 optimizer；精度为默认bf16或`no`。这不是DeepSpeed或FSDP实现，仍明确拒绝这些未接入的后端，以及未保存scaler的fp16配置。ZeRO-1在所有rank共同归并后由rank0保存CPU optimizer state，恢复时各rank通过CPU mmap读取并仅将自己拥有的moment分片搬到GPU，完整state校验分片模式和world size。保存阶段rank0必须临时容纳完整Adam moments：仅6.02B backbone的两个FP32 moments约44.86GiB，另需模型导出、其他参数和进程内存；保存完成后释放归并快照。因此需同时预留主机RAM和磁盘空间。仅加载addon权重不能恢复优化器/RNG；`resume`要求完整state目录。不要将修改foundation模式或分片方式视为同一次训练的精确续训。

### Foundation权重与上一版对照

joint Stage2A/B，以及继承joint Stage2A的frozen Stage2B，均导出一对文件：`step_XXXXXX_addon.pt` 与 `step_XXXXXX_foundation.pt`。后者包含MoT/video/action和proprio状态，不重复保存冻结的VAE/文本/CTE；前者记录文件名、SHA256、step、原始base身份及继承来源。`foundation_checkpoint_required`独立于`foundation_mode`：前者决定是否必须加载配对权重，后者表示本阶段是否更新foundation。搬移时必须同时复制两份；`--checkpoint`仍指原始base，之后自动用sidecar覆盖。Stage2B初始化、恢复与评测均不能漏掉joint parent的foundation。

新保存产物显式记录 `foundation_contract_schema=iclwam_foundation_contract_v1`，要求完整的foundation元数据；新PIM合同也要求该版本标记，防缺失字段被误认作legacy base-only addon。已有训练后foundation身份的模型对象，仅在完整mot/proprio严格恢复原base后清除身份；legacy dit-only或partial checkpoint不能用来执行该重置。正常fresh model仍保留旧base构建兼容。

ZeRO-1只分片optimizer，`checkpoints/weights/`中的完整权重可直接用于单卡部署，无需16份optimizer分片；`checkpoints/state/`用于继续训练。只有未继承训练后foundation的frozen模型才只需要addon与原始base。载入joint父权重进行训练时，即使后续冻结，也保留其FP32数值再冻结，避免先量化造成父模型小更新丢失。部署按目标BF16 dtype加载，不要求常驻FP32，更不加载Adam状态；FP32训练中的微小差异可能在BF16转换时舍入，不承诺1e-6级每次更新都能在部署权重中分别辨识。新旧metadata矛盾、缺失sidecar、错配父模型或修改PIM语义都会被拒绝。

复现上一版“预测CTE + image/text static retrieval + 冻结foundation”的Stage2B时使用独立task `robotwin_zeva_ego_pim_frozen_3cam_384` 和它自己的旧bank/head/static/Stage2A，并设置 `model.zeva.task_context.readout_cache_path=/absolute/path/to/old_train_readouts.pt`，recipe仍为 `ego_adapter_warmup`。该readout cache只供旧prepared训练预检，部署旧模型不需要它。不要在新joint task下加载旧frozen addon；新模式会拒绝这种混用。

重评 `b8b6e84` 的语言检索版本时，使用 `robotwin_zeva_ego_pim_language_legacy_3cam_384`，并选择与原checkpoint相同的joint/frozen模式。它保留 `task_language`、top5/temperature1、uniform+legacy_floor、legacy_xavier、LOO和legacy候选历史，而不是切回image+text检索。必须使用该checkpoint原来的base、CTE、bank、classifier和language encoder。该alias用于旧Stage2B加载与评测；精确复现整个旧训练应使用原commit及其完整配置/state。

## 7. RoboTwin PIM evaluation

```bash
ICLWAM_STAGE2B="$ICLWAM_RUN/stage2b/checkpoints/weights/step_002000_addon.pt"

python scripts/eval_zeva_robotwin_manifest.py \
  --checkpoint "$ICLWAM_BASE" --dataset-stats "$ICLWAM_STATS" \
  --new-cte-checkpoint "$ICLWAM_CTE" --new-addon-checkpoint "$ICLWAM_STAGE2B" \
  --new-task-context-bank "$ICLWAM_BANK" \
  --new-task-context-retrieval-checkpoint "$ICLWAM_HEAD" \
  --new-task-language-encoder "$ICLWAM_LANGUAGE_ENCODER" \
  --conditions base new_off new_on \
  --new-task-config robotwin_zeva_ego_pim_3cam_384 \
  --new-foundation-mode joint \
  --policy-text-encoder-device cpu --policy-text-cache-size 16 \
  --new-prefix-layout zeva_ego --new-effect-source current_prediction \
  --model-rng-protocol torch_continuous_slot_v1 \
  --slots 8 --model-seed 20260907 --num-inference-steps 10 \
  --pim-diagnostic-interval 10 \
  --output-dir "$ICLWAM_RUN/eval_10x20"
```

可以先加 `--dry-run` 审查固定计划；dry-run本身也创建目录，因此正式运行要换一个全新的output-dir。dry-run不检查权重存在性，不加载模型，不验证仿真环境。使用 `--tasks ...` 得到明确标记的development子集，保留源manifest的task到slot分配，不能当完整200episode结果。自定义源manifest必须提供 `--expected-manifest-sha256`。

上例显式写出当前默认的新task、effect来源和RNG协议。`--model-rng-seed` 是 `--model-seed` 的别名，`--effect-source` 是 `--new-effect-source` 的兼容别名。回放旧采样规则才使用 `--model-rng-protocol request_seed_stream_v1`；它每个task重建request-seed流，不是参考的连续slot协议。`--pim-diagnostic-interval 10` 用于测量局部同噪声action差异，会增加memory-on推理耗时；比较部署速度时使用 `--pim-diagnostic-interval 0` 并单独保留诊断实验。

评测frozen模型时，改为它自己的addon、`--new-foundation-mode frozen`和新的output目录，语言/CTE配置保持相同。该选项表示checkpoint的**训练模式**，不会在推理时更新任何参数；无论joint还是frozen，推理都在no_grad下运行，只更新CTE时序状态与PIM历史。新旧条件可各自指定 `--old-foundation-mode` / `--new-foundation-mode`；模式与checkpoint不匹配在启动仿真前失败。

新checkpoint记录完整 `pim_memory_contract`：compression、round/floor索引、attempt/entry容量、merge阈值和phase权重。计划、实际runner和汇总审计同时记录保存的合同与实际合同。若要测量在线压缩的影响，可在同一新checkpoint上显式使用 `--new-pim-compression uniform`，创建新的output目录；这是有记录的compression-only消融，不允许顺便更换索引或容量。old条件对应 `--old-pim-compression`。省略该参数时不能静默更改checkpoint的部署记忆语义，旧uniform/floor模型也不能直接塞入新similarity/round默认。

### 单张32GB 5090部署

policy模型构造时不加载完整UMT5。独立policy文本编码器直接在 `--policy-text-encoder-device cpu`上构造，默认保持模型配置的dtype，只有文本特征搬到策略GPU；没有先加载GPU再offload的启动峰值。按完整prompt作key的LRU缓存默认16条、驻CPU，并跨replan/attempt保留；设 `--policy-text-cache-size 0`可禁用缓存做等价性检查。`cuda`设备选项用于显存充足的对照。CPU与GPU计算的浮点差异不能假定逐位相同，因此设备、dtype、文本权重/tokenizer哈希和cache容量都进入身份与评测审计。

这条policy文本路径与task retrieval的CPU输入embedding表是两个不同模块。它保留原FastWAM prompt包装、special tokens、零padding仍可见的mask语义，缓存不会替换为简单词向量均值。固定输入下cache开启/关闭应产生相同文本特征及动作。实际推理仍编码当前图像、预填video K/V并去噪action，不生成未来视频。

约6.02B backbone的BF16权重为11.21GiB，但总显存还包括VAE/CTE/addon、K/V、中间结果、CUDA工作区和RoboTwin渲染。attempt诊断记录当前与峰值allocated/reserved、device-wide free/total，供真实5090上核查；不要将11.21GiB称为完整推理峰值，也不要据此声称已经验证32GB可运行。8个slot是逻辑随机流，当前runner在一张GPU上顺序执行，不会常驻8套模型。吞吐对比需单独关闭额外PIM诊断，区分首次CPU文本编码和后续cache命中延迟。

加入旧版对照时，增加 `old` condition，并提供它**自己的** `--old-cte-checkpoint`、`--old-addon-checkpoint`、`--old-task-context-bank`、`--old-task-context-retrieval-checkpoint`。若对比上一版预测CTE/frozen foundation，另加 `--old-task-config robotwin_zeva_ego_pim_frozen_3cam_384 --old-effect-source current_prediction`；old保留static image+text retrieval，新模型为task_language。更早版本的默认old task仍为 `robotwin_zeva_fastwam_cross_attempt_3cam_384`，默认三token `zeva_ego` 布局及 `--old-effect-source latest_completed`；合并token模型需显式 `--old-prefix-layout legacy_merged`，原top-4 reader还需 `--old-task-config robotwin_zeva_fastwam_pim_3cam_384`。不能把新CTE的phase/effect256维或新bank喂给旧phase/effect128维模型。

共享 `--cte-checkpoint`、`--task-context-bank`、`--task-context-retrieval-checkpoint`、`--task-language-encoder` 仍可作为对应old/new路径未指定时的fallback。每个condition的mode从各自task读取，也可用 `--old-task-context-mode` / `--new-task-context-mode` 显式选择；joint只允许task_language。`new_off` 与 `new_on` 始终使用相同的新addon、微调foundation、CTE和语言检索组合。正式runner在启动仿真前检查foundation sidecar和recipe；frozen plan、slot RNG身份及attempt审计均记录foundation和encoder/classifier/bank的SHA256，防止实际评测了错误底座。

### 固定协议

1. 使用仓库内冻结的10任务×20scene seed及完整seen指令manifest和 `demo_randomized`。本项目源文件SHA256固定为 `588a85f45dc961ccf3b4ecba3a1651eeb126c4a8368d8ac7f7485c3aa4e145cf`；它转换了schema并记录joint14/H32/replan24，因此与Zeva原文件的 `1b9dbf74bd9d9b8871647a00d6557f84884685d4459065f004bd600a86b1679b` 不同。当前资产的expert有效性检查失败会中止，不换seed。
2. 一个episode至多4个attempt；失败后用同一scene seed、同一instruction重建环境，成功即停。每个episode开始PIM为空，retry保留此前attempt的预测trace并重置CTE状态。
3. 排序后的task按index modulo8分配slot。每个condition有独立的8个slot随机流；每次重规划只抽一次 `[1,32,14]` CPU float32初始动作噪声，同slot跨task连续。当前runner在 `--gpu-id` 指定的GPU上顺序执行slot/task并持久化状态；8个逻辑slot不表示自动启动8GPU并行。
4. task的全部rollout正常结束后才原子写出RNG状态；runner先核对完整checkpoint组合、任务顺序和日志末尾snapshot，再执行下一个task。完整run结束后才对全部attempt日志和视频统一审计。失败的部分run不能续跑或重用目录。protocol、recipe、checkpoint/foundation、CTE、bank/classifier/language encoder、slot、scene/instruction、reset scope、draw计数和视频路径参与审计；完整通过后才生成`COMPLETE`。
5. 报告第1/2/3/4次以内至少成功一次的累计 solved / total / rate，分母始终是全部200episode。没有第4次attempt的早期成功episode仍计为成功，不用“实际执行了第4次的次数”作分母。

8-slot连续协议与参考对齐，但CPU Torch噪声不能宣称与参考CUDA/pi0.5噪声逐元素相同。不同条件的早期成功/重试次数不同，也会改变之后episode在连续流中的offset；因此整轮on/off是同manifest、预算、slot初始seed和权重的协议对照。**同一重规划点的paired诊断**才严格复用同一初始动作噪声，且不额外推进随机流。

默认joint14模型满足 `action_state_transforms` 为空且 `use_stepwise_action_norm=false` 时，新预测CTE只需重规划边界观察，manifest runner为 `new_off/new_on` 设置 `skip_get_obs_within_replan=true`。旧completed-effect CTE仍为false，base保留原FastWAM的边界观测行为。若新模型切换到依赖逐步state的动作处理，runner会将它恢复为false；部署也会拒绝不兼容的观测跳过配置。不同条件的实际观测策略写入冻结计划和attempt日志并接受审计。固定单场景入口默认保守地每步取观测，不属于完整200episode benchmark。

### 重新审计报告与两个manifest哈希

runner自动保存各condition的 `.summary.json` 和 `paired_pim.summary.json`。下面的命令独立复核同一新Stage2B的on/off日志：

```bash
ICLWAM_EVAL="$ICLWAM_RUN/eval_10x20"
ICLWAM_EVAL_SHA=$(python -c 'import json, sys; print(json.load(open(sys.argv[1]))["manifest_sha256"])' "$ICLWAM_EVAL/command_plan.json")

python scripts/report_zeva_robotwin_multiattempt.py \
  --manifest "$ICLWAM_EVAL/manifest.json" \
  --expected-manifest-sha256 "$ICLWAM_EVAL_SHA" \
  --on "$ICLWAM_EVAL/new_on.jsonl" --off "$ICLWAM_EVAL/new_off.jsonl" \
  --output "$ICLWAM_EVAL/paired_review.json"
```

runner的 `--expected-manifest-sha256` 校验输入的**源文件**；reporter校验保存到run目录的**运行manifest**。后者加入了slot、模型配置、观测策略和scope，因此hash不同，应使用 `command_plan.json` 的 `manifest_sha256`，不能使用上面的源文件hash。完整源manifest的运行标记为 `frozen_full`，显式任务子集为 `development_subset`，其他源文件为 `custom`。

`cumulative_success_rate` 直接列出 `{attempt, solved, total, rate}`；同时保留micro/macro、各task的SR@1–4及配对bootstrap区间。报告拒绝缺失episode、错误reset、预算变化、运行异常、RNG不连续或重复/缺失/空视频。视频审计只检查路径唯一、文件存在且非空，不验证编码和画面内容。`--metadata-only` 显式跳过视频检查并标记 `METADATA_ONLY`。用训练/验证数据选择超参后再冻结这组200episode测试，不根据测试结果反复选择超参或有利seed。

## 8. 已验证范围与结果解释

本次在CPU环境（PyTorch 2.7.1、Accelerate 1.12.0，允许本机loopback）执行 `OMP_NUM_THREADS=1 PYTHONPATH=src python -m pytest -q tests/zeva`，结果为 **432 passed，0 skipped**，耗时29.63秒；3条warning来自原有grad-norm日志的Tensor拷贝及旧scheduler弃用提示。两个真实双进程Gloo测试均执行通过；若运行环境禁止loopback socket，这两项会明确skip，不能把该结果当作已验证分布式。新Hydra入口配置解析、全部30个变更/新增Python文件的AST语法检查、审计/指南相对文件链接检查通过。

CPU测试覆盖CTE batch/streaming一致、过去动作shift、未来图像/动作不泄漏、EMA与BN、尾部mask、损失数值、episode无放回/划分、readiness拒绝、task memory隔离、预测cache到部署生命周期，以及真实小型WanVideoDiT/ActionDiT/MoT中的梯度与动作扰动。另有连续slot RNG、fresh output、manifest身份、固定分母、reset/video审计和mock runner状态机测试。

已覆盖warmup时foundation/EAP不变、联合阶段video输入投影/action/proprio实际更新、CTE/文本/VAE保持冻结；BF16 autocast下FP32主干能保留1e-6更新；完整bundle/Adam/scheduler/RNG恢复和activation checkpoint等价性。语言检索使用真实小型tokenizer快照与分类器端到端验证，故意令held-out GT task与instruction不同仍按语言预测路由，改变CTE validation特征不影响分类器训练，LOO同时影响mean-key与support，旧artifact/encoder漂移及foundation缺失或错误均被拒绝。

真实Hydra→runtime→小型WanVideoDiT/ActionDiT/MoT构造测试覆盖Stage2A、frozen Stage2B和joint Stage2B的FP32参数/Adam moments、BF16 autocast梯度、冻结参数不变，以及先升精度再加载父权重的小更新保留。真实双rank trainer验证ZeRO-1与普通AdamW DDP的参数、microbatch loss和moments逐位相同；跨epoch累积、warmup/joint/empty PIM、step2恢复后的样本顺序、history curriculum、每rank RNG、scheduler和再次集体保存均一致。

本轮新增覆盖joint/frozen Stage2A从首步使用分组恒定LR及step2精确续训；joint A→frozen B自包含bundle与FP32继承；真实BF16部署目标不调用training prepare，加载值及同噪声动作与显式cast参考一致。缺失整组foundation元数据、partial base覆盖、错sidecar等路径在复制权重前拒绝。完整train bank静态context与在线检索一致；v3的round索引、两级计数加权merge、唯一固定partner、完整课程/显式缩减门禁、boundary source drift和旧版兼容均有回归覆盖。独立pinned-reference CPU对照还验证了CTE损失、paired sampler和72次在线memory提交；这些有限数值检查不代表CUDA逐位等价。

部署测试使用真正的小型WanTextEncoder，验证FP32/BF16文本feature与原FastWAM编码完全一致、LRU命中/淘汰/零容量、返回值不可污染缓存、base/Zeva多次重规划复用及tokenizer身份稳定性。CPU测试拦截CUDA传输边界，仅证明编码器与特征传输的接线，不代表执行过真实CUDA内核。另覆盖模式CLI到子进程、16卡有效batch、checkpoint模式/文本身份审计，以及joint-default配置下base仍记录原始frozen foundation的构造与RNG身份。

这些验证说明实现能把context接到action且遵守数据/时间/评测合同。**本次修改没有实际运行80epoch GPU训练、16卡NCCL训练、完整模型5090推理或200episode RoboTwin仿真；没有新的显存峰值、延迟或成功率结果。** 正式运行仍需CUDA/Mamba、真实数据、FastWAM权重、RoboTwin资产和视频渲染环境。

最终判断应分开三个问题：Stage2A是否保持或改善base；同一Stage2B在`new_on`相对`new_off`是否提升第2–4次累计成功率；相对旧版本是否仍有收益。如果PIM能改变动作却不能改善成功率，下一步应检查历史相关性、CTE预测质量及真实失败恢复数据分布，不能只继续放大注入幅值。
