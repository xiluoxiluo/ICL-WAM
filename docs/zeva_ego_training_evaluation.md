# ICL-WAM：Zeva-Ego 预测 CTE、语言检索、联合训练与 RoboTwin PIM 评测

本指南对应新的 `current_prediction` 路径。参考代码固定到 [Zeva `feature/zeva_ego` 的 `bae85ad753d3afe4f2e56762373d48ac0267e101`](https://github.com/air-embodied-brain/Zeva/tree/bae85ad753d3afe4f2e56762373d48ac0267e101)，不依赖分支持续变化后的实现。此前只有独立 prefix 的版本使用 `latest_completed`，其 [旧指南](zeva_ego_cross_attempt.md) 保留用于复现。

当前默认遵循 **除必要FastWAM适配外，按Zeva-Ego公开实现对齐**：包括CTE源范围与SHA/round划分、Stage2全部CTE帧、逐样本语言查询、EAP/PIM数值语义、joint训练及在线memory生命周期。完整证据与边界见 [对齐审计](zeva_ego_alignment_audit.md)。FastWAM保留自己的backbone、动作/时间尺度、输入空间和flow目标形式。

**从`4e95b26`及更早版本迁移，严格新默认需要按新task subset/conditions/split训练CTE，再重建boundary、bank、features、classifier、per-decision contexts和PIM v4，重训Stage2A/B。** CTE架构/损失未变，但数据成员、sampler顺序与来源合同已变，不能沿用上一轮“CTE无需重训”的迁移结论。原FastWAM base、原数据、统计、语义表、匹配的policy文本cache可复用；独立冻结语言encoder也可在其tokenizer/embedding资产完全匹配时复用。旧权重保留显式legacy重评入口，旧完整state只能用于原commit/配方的精确续训。

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
| 连续模型随机流 | policy设备连续slot语义 | 默认`torch_device_continuous_slot_v1`；policy设备抽噪声，记录实际设备和state；不声称跨不同模型逐元素一致 |

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

第一次重规划已经有有效 `e_hat_t`，无需等待16步 completed effect。生成动作后将 `(z_t,e_hat_t)` 追加到当前 pending trace；当前 query 不读取自身新条目。结束时保留pending，在下次attempt reset提交；新episode清空历史。在线默认attempt内similarity merge→round均匀截取最多64条→跨attempt merge，保留计数加权centroid。最多保留4个已结束attempt、总容量256，history张量保持policy device。四次尝试的最后一次action实际最多读取前三次；不能因此把模块容量改成3。线上终止后不额外读取terminal observation或预测不存在的下一次决策；离线Stage2遍历导出的全部CTE帧是独立的数据范围规则。

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
ICLWAM_RUN=/absolute/path/to/runs/iclwam_reference_alignment_v3
ICLWAM_CONDITIONS=/absolute/path/to/episode_conditions.json
ICLWAM_CTE_TASK_SUBSET=/absolute/path/to/cte_task_subset.json

DATA_ARGS=(
  "data.train.dataset_dirs=[$ICLWAM_DATA]"
  "data.train.pretrained_norm_stats=$ICLWAM_STATS"
  "data.train.semantic_task_map_path=$ICLWAM_TASKS"
  "data.train.text_embedding_cache_dir=$ICLWAM_TEXT"
)
```

`semantic_task_map.json`提供规范RoboTwin task identity，不能把不同语言表达的`task_index`当不同任务。CTE、policy和部署共享归一化统计。`cte_task_subset.json`必须为`{"task_names":["task_a","task_b",...]}`的真实有序任务列表；此顺序也用于paired sampler。`episode_conditions.json`为`{canonical_episode_id: condition_string}`，标记完整源数据各episode的实际Clean/Randomized条件。CTE与两个PIM builder使用同一文件，内容SHA参与来源检查；不能推断或伪造condition，也不能以评测10任务清单猜测原CTE训练范围。

## 4. Stage1：完整 episode 的预测 CTE

```bash
python scripts/train_zeva_ego_cte.py \
  task=robotwin_zeva_fastwam_ego_cte_3cam_384 \
  "${DATA_ARGS[@]}" \
  cte_task_subset_path="$ICLWAM_CTE_TASK_SUBSET" \
  cte_episode_conditions_path="$ICLWAM_CONDITIONS" \
  device=cuda output_dir="$ICLWAM_RUN/cte"
```

这条入口为**单进程、单 GPU**，不要改成多卡 `torchrun`。episode pair / 无放回 epoch 语义在该入口中明确实现；它拒绝 `WORLD_SIZE>1`，避免伪造已实现的数据并行配方。

新默认`cte_split_policy=zeva_ego`、`cte_conditions=[Clean,Randomized]`。先按condition×task建立有序source分区，以`SHA256("seed:partition_id:episode_position")`前8字节little-endian排序并按round选择5% holdout，然后过滤task subset和`length>H24`；短episode及未选CTE任务仍参与原分区的划分人口。`cte_split_seed=1000`独立于训练seed。保存完整partition population、原顺序、真实conditions快照、eligible IDs及输入文件/全源metadata SHA；下游在解码前验证。若源中有无法可靠恢复分区位置的空episode，严格路径明确报错。规则与参考一致，实际成员还取决于用户提供的数据与源顺序。

| 训练项 | 默认值 / 行为 |
| --- | --- |
| 长度 | 80 个完整 epoch，覆盖每个 train episode，非5000次随机窗口抽样 |
| batch | 8 个 episode，同任务相邻成对；无放回，保留尾部 episode |
| 数据划分 | 独立split seed1000，condition×task分区，SHA排序，round(0.05×N)，按参考边界处理；训练筛选在划分后 |
| CTE | 三流、4层、hidden256、Mamba；三相机共享预训练 ResNet18 后融合 |
| EMA | 0.99；target encoder不更新梯度，BN统计在训练时也固定，防止跨时间视觉泄漏 |
| AdamW | online vision LR1e-5，其余1e-4，weight decay1e-4，gradient clip1；不使用 scheduler |
| 损失 | `0.1 action + 0.2 vision + 2 global + 1 phase + 0.2 effect` |
| 验证 / 保存 | 每5个epoch及最终epoch做独立验证；每个完整epoch保存权重、优化器、RNG、数据/源码身份 |
| Stage2入口检查 | 默认精确epoch80；正式promotable架构、当前epoch验证记录的stage1_gate=true，且重算三项gain均≥0.05；显式探索仅epoch40/step26160且满足同样验证 |

输入张量为 `images[B,T+1,3,3,224,224]`、`executed_actions[B,T,24,14]`、`valid[B,T]`。每个时刻的三张原始 RGB 独立缩放并映射到 `[-1,1]`；action 是按 FastWAM dataset stats 归一化后的有序24步动作。三流每层先分别进行因果时间建模，再在同一边界按 vision←previous-action、action←vision、interaction←(vision,action) 的顺序交互。action head 从 action stream 预测本段动作，vision head 从 vision stream 预测下一边界 EMA visual feature，effect head 从 interaction `z_t` 预测 EMA feature 差。

训练只保留完整24步动作及其真实下一边界观测。若一个episode有 `L` 条带观测记录，只在 `raw_step+24<L` 时构造这一段 target；最后不足24步不补零当作完整动作。批次padding只出现在轨迹尾部，并在所有损失里排除。模型可返回最后一个真实边界的 phase/effect，但该边界没有被伪造出的下一段监督目标。

三项基线分别为**归一化空间**的零动作、视觉保持不变、零视觉变化；gain 为 `1 − model_MSE / max(baseline_MSE,1e-12)`。global loss 使用同任务 episode 正样本，self 不作为正样本但保留在 softmax 分母，遵循参考实现；phase loss 使用有效时间位置的 identity 分类。action BC loss 对14个坐标求和后在24个执行时间位置上平均；视觉/effect loss对feature坐标求和。实现见 [ego_cte.py](../src/fastwam/zeva/ego_cte.py)，完整训练与readiness约束见 [ego_cte_training.py](../src/fastwam/zeva/ego_cte_training.py)。

划分在解码RGB前完成并保存在`episode_splits.json`。可通过`cte_episode_split_path=...`提供`{"train":[...],"validation":[...]}`，但严格配方会重算参考split并校验完全一致，不允许任意手工split冒充参考。旧semantic-task/ceil划分仅在`cte_split_policy=legacy_semantic_task`显式旧路线使用。

正式新流程使用通过验证的`cte_epoch_080.pt`。训练满80epoch本身不代表通过；后续builder重新核验来源、选择条件及gain。以下为相同新配方从中间epoch继续到80：

```bash
python scripts/train_zeva_ego_cte.py \
  task=robotwin_zeva_fastwam_ego_cte_3cam_384 "${DATA_ARGS[@]}" \
  cte_task_subset_path="$ICLWAM_CTE_TASK_SUBSET" \
  cte_episode_conditions_path="$ICLWAM_CONDITIONS" \
  device=cuda num_epochs=80 \
  resume="$ICLWAM_RUN/cte/cte_epoch_040.pt" \
  output_dir="$ICLWAM_RUN/cte"
```

续训验证配置、完整split、归一化和源码身份，并恢复Torch CPU/CUDA、Python、NumPy及DataLoader RNG。checkpoint在完整epoch后保存。更长训练可作为另行研究，但其epoch不能默认为参考发布选择；`model.zeva.cte.exploratory_epoch40=true`仅解锁epoch40且step26160的显式探索，不泛化为任意早停。CPU测试仅允许显式`gru_test`/`tiny_test`和`cte_smoke_steps`；smoke不写正式checkpoint，测试backend不能通过正式loader。Stage1不加载Wan VAE或文本embedding。

探索选择必须在boundary、task-memory、Stage2各消费者的Hydra命令中显式设置，训练时设置一次不会自动传给后续进程。正式评测为argparse入口，分别使用`--new-cte-exploratory-epoch40`或`--old-cte-exploratory-epoch40`；只影响所选family，不影响base或另一组，因此可明确比较old epoch80与new epoch40。两条选择均须`manifest.promotable=true`、本epoch的`validation.stage1_gate=true`且重算gain达标，不能通过手改flag跳过验证。

## 5. 重建 boundary cache、task memory 与 PIM cache

以下为严格新配方的全新构建流程；使用新目录，不覆盖旧实验。旧CTE/split不能直接接入新默认，boundary和依赖其CTE key space/split的语言bank/classifier等必须重新生成。已存在且匹配的独立语言encoder可复用，其余输出路径必须未使用。

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

boundary v2在`0,24,48,…`逐次调用与部署相同的`CTE.step`，每episode重置状态，并按参考导出关闭matmul/cuDNN TF32。它记录current phase、predicted effect、raw global256和未逐边界归一化的task-head128。`export_scope=all_cte_frames_v1`保留全部真实CTE帧，含最后已观察边界；不再按`decision_valid`或完整H32过滤。PIM v4的`query_scope`相同。Stage2通过LeRobot已有clamp重复最后动作填满H32，EAP NLL及WAM action-flow都计入整个horizon。Stage1 CTE的H24真实执行目标仍排除不完整段，不混淆两套监督范围。

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

新默认`train_query_exclusion=none`使用完整train bank；旧LOO显式保留。`training_query_scope=per_decision`使训练按每个CTE query对应raw instruction检索，不能把episode首指令误用于全部训练帧。离线v2 context保存`episode/window/raw_step/instruction`与去重context索引；相同指令且相同排除范围才可去重。训练前重新从原始metadata核对全部query。推理按参考仅缓存attempt首条指令的global，retry重新检索，不因中途不同字符串直接报错。bank/features/classifier仍绑定CTE、stats、semantic、boundary/split和各自SHA；旧episode-static cache只在显式legacy配置使用。

上述语言bank/features/classifier/static构建支持CPU，不需要加载FastWAM base；语言输入embedding资产必须可用。统一命令中的`ckpt`供后续policy训练使用。PIM历史源同样仅限其他train episodes；Stage2查询split与Stage1保持一致。artifact入口均为单进程，不接收多rank `torchrun`。

两个PIM builder必须使用CTE训练时同一份真实条件文件。`history_pairing=reference_nearest`按初始phase为每episode固定同条件最近、同条件最远、异条件最近各一条train历史。`pairing_metric=torch_float32_first_in_source_order`用Torch FP32 normalize/matmul，argmax/argmin平局取source原序第一项；不再使用NumPy+episode ID tie。当前episode排除，各query共享partner。`far`对应参考`same_condition_far`。

完整参考课程要求每个query均存在same/cross候选；缺条件或缺候选会报错。只有确需做缺条件数据消融时，才在两个builder中显式增加 `--allow-reduced-curriculum`，并在两阶段prepared训练命令增加 `model.zeva.prepared_stage2.allow_reduced_curriculum=true`。此时manifest明确标记可用模式及课程缩减，调度删除不可用项；例如无cross时warmup变成5:2、后续5:2:0:2，这不是完整参考5:2:1 / 5:2:1:2实验。`legacy_candidates`配对只用于旧版复现，不能把它标为reference-nearest缓存。

新版格式：CTE容器仍为`iclwam_zeva_ego_cte_v1`但必须有新reference scope；boundary=`zeva_ego_boundary_cache_v2`；PIM=`zeva_stage2_cross_attempt_pim_v4`；query context=`zeva_static_query_context_v2`；addon=`zeva_ego_prediction_v1`且新增`numerical_contract=zeva_ego_v1`。保存并校验scope、全部query覆盖、pairing metric、round索引及SHA。旧boundary v1、PIM v2/v3、episode context v1和旧数值合同仅供显式legacy，不允许改schema字符串升级。

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

新默认同时指定`prompt.numerical_contract=zeva_ego_v1`和`adapter.numerical_contract=zeva_ego_v1`。训练prior残差使用参考`rand<1-p`保留mask，MHA调用`need_weights=True`，PIM Q/K在实际AMP计算dtype中normalize；EAP NLL使用转入WAM dtype之前的FP32 normalized action，对坐标求和并按完整H32平均。WAM保留自己的flow target/noise/scheduler权重公式，但严格新路径同样计入末动作重复填充的所有位置；不再被`action_valid`清空最后边界的梯度。旧`legacy_v1`保留原mask/精度行为，不可静默重标新合同。

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

ZeRO-1只分片optimizer，`checkpoints/weights/`的完整权重可用于单卡，无需optimizer分片；`checkpoints/state/`用于续训。只有未继承训练后foundation的frozen模型才仅需addon与base。训练加载joint父权重时即使随后冻结，也先保留FP32值。**部署WAM backbone用BF16，CTE及新EAP/PIM小模块保持FP32权重与推理计算；只有进入video prefix/action residual接口时转换到WAM dtype。** 不加载Adam；不承诺BF16主干可逐次辨识每个1e-6更新。`module.to(bfloat16)`也不能先舍入再转回addon；精度合同在构造、加载和转设备路径中保持。新旧数值合同、foundation或PIM身份冲突在加载前拒绝。

复现上一版“预测CTE + image/text static retrieval + 冻结foundation”的Stage2B时使用独立task `robotwin_zeva_ego_pim_frozen_3cam_384` 和它自己的旧bank/head/static/Stage2A，并设置 `model.zeva.task_context.readout_cache_path=/absolute/path/to/old_train_readouts.pt`，recipe仍为 `ego_adapter_warmup`。该readout cache只供旧prepared训练预检，部署旧模型不需要它。不要在新joint task下加载旧frozen addon；新模式会拒绝这种混用。

重评 `b8b6e84` 的语言检索版本时，使用 `robotwin_zeva_ego_pim_language_legacy_3cam_384`，并选择与原checkpoint相同的joint/frozen模式。它保留 `task_language`、top5/temperature1、uniform+legacy_floor、legacy_xavier、LOO和legacy候选历史，而不是切回image+text检索。必须使用该checkpoint原来的base、CTE、bank、classifier和language encoder。该alias用于旧Stage2B加载与评测；精确复现整个旧训练应使用原commit及其完整配置/state。

重评`4e95b26`使用`robotwin_zeva_ego_pim_4e95_3cam_384`，保留similarity_merge/round/full train bank，但明确3 attempts/192条、end_attempt提交、CPU history、episode-static查询、旧数值合同和旧CTE readiness。两种旧语言alias均须显式选择其原RNG协议`--model-rng-protocol torch_continuous_slot_v1`；不使用新task给旧权重换标签。

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
  --model-rng-protocol torch_device_continuous_slot_v1 \
  --slots 8 --model-seed 20260907 --num-inference-steps 10 \
  --pim-diagnostic-interval 10 \
  --output-dir "$ICLWAM_RUN/eval_10x20"
```

可以先加 `--dry-run` 审查固定计划；dry-run本身也创建目录，因此正式运行要换一个全新的output-dir。dry-run不检查权重存在性，不加载模型，不验证仿真环境。使用 `--tasks ...` 得到明确标记的development子集，保留源manifest的task到slot分配，不能当完整200episode结果。自定义源manifest必须提供 `--expected-manifest-sha256`。

上例显式写出新task、effect与policy-device连续slot RNG协议。`--model-rng-seed`是`--model-seed`别名，`--effect-source`是`--new-effect-source`兼容别名。`torch_continuous_slot_v1`保留旧私有CPU连续噪声，仅用于旧版重评；更早`request_seed_stream_v1`每task重建request-seed流。三种协议分别记录，handoff不能交叉加载。`--pim-diagnostic-interval 10`测量局部同噪声动作差异，会增加on条件耗时；速度实验使用0并单独保留诊断实验。

评测frozen模型时，改为它自己的addon、`--new-foundation-mode frozen`和新的output目录，语言/CTE配置保持相同。该选项表示checkpoint的**训练模式**，不会在推理时更新任何参数；无论joint还是frozen，推理都在no_grad下运行，只更新CTE时序状态与PIM历史。新旧条件可各自指定 `--old-foundation-mode` / `--new-foundation-mode`；模式与checkpoint不匹配在启动仿真前失败。

新checkpoint记录`pim_memory_contract` v2：compression、round索引、4×64/总256容量、merge参数、`commit_timing=attempt_reset`、`storage_device=policy`；还记录数值合同。计划、实际runner与审计核对保存及执行合同、CTE epoch/scope、训练query scope与首指令缓存语义。`--new-pim-compression uniform`仅允许同checkpoint显式压缩单因素消融，使用新输出目录；不能顺便更换容量、提交时点或设备。旧模型用明确legacy合同重评。

### 单张32GB 5090部署

policy模型构造时不加载完整UMT5。独立policy文本编码器直接在 `--policy-text-encoder-device cpu`上构造，默认保持模型配置的dtype，只有文本特征搬到策略GPU；没有先加载GPU再offload的启动峰值。按完整prompt作key的LRU缓存默认16条、驻CPU，并跨replan/attempt保留；设 `--policy-text-cache-size 0`可禁用缓存做等价性检查。`cuda`设备选项用于显存充足的对照。CPU与GPU计算的浮点差异不能假定逐位相同，因此设备、dtype、文本权重/tokenizer哈希和cache容量都进入身份与评测审计。

这条policy文本路径与task retrieval的CPU输入embedding表是两个不同模块。它保留原FastWAM prompt包装、special tokens、零padding仍可见的mask语义，缓存不会替换为简单词向量均值。固定输入下cache开启/关闭应产生相同文本特征及动作。实际推理仍编码当前图像、预填video K/V并去噪action，不生成未来视频。

约6.02B backbone的BF16权重为11.21GiB，但总显存还包括VAE/CTE/addon、K/V、中间结果、CUDA工作区和RoboTwin渲染。attempt诊断记录当前与峰值allocated/reserved、device-wide free/total，供真实5090上核查；不要将11.21GiB称为完整推理峰值，也不要据此声称已经验证32GB可运行。8个slot是逻辑随机流，当前runner在一张GPU上顺序执行，不会常驻8套模型。吞吐对比需单独关闭额外PIM诊断，区分首次CPU文本编码和后续cache命中延迟。

加入旧版对照时，增加 `old` condition，并提供它**自己的** `--old-cte-checkpoint`、`--old-addon-checkpoint`、`--old-task-context-bank`、`--old-task-context-retrieval-checkpoint`。若对比上一版预测CTE/frozen foundation，另加 `--old-task-config robotwin_zeva_ego_pim_frozen_3cam_384 --old-effect-source current_prediction`；old保留static image+text retrieval，新模型为task_language。更早版本的默认old task仍为 `robotwin_zeva_fastwam_cross_attempt_3cam_384`，默认三token `zeva_ego` 布局及 `--old-effect-source latest_completed`；合并token模型需显式 `--old-prefix-layout legacy_merged`，原top-4 reader还需 `--old-task-config robotwin_zeva_fastwam_pim_3cam_384`。不能把新CTE的phase/effect256维或新bank喂给旧phase/effect128维模型。

共享 `--cte-checkpoint`、`--task-context-bank`、`--task-context-retrieval-checkpoint`、`--task-language-encoder` 仍可作为对应old/new路径未指定时的fallback。每个condition的mode从各自task读取，也可用 `--old-task-context-mode` / `--new-task-context-mode` 显式选择；joint只允许task_language。`new_off` 与 `new_on` 始终使用相同的新addon、微调foundation、CTE和语言检索组合。正式runner在启动仿真前检查foundation sidecar和recipe；frozen plan、slot RNG身份及attempt审计均记录foundation和encoder/classifier/bank的SHA256，防止实际评测了错误底座。

### 固定协议

1. 使用仓库内冻结的10任务×20scene seed及完整seen指令manifest和 `demo_randomized`。本项目源文件SHA256固定为 `588a85f45dc961ccf3b4ecba3a1651eeb126c4a8368d8ac7f7485c3aa4e145cf`；它转换了schema并记录joint14/H32/replan24，因此与Zeva原文件的 `1b9dbf74bd9d9b8871647a00d6557f84884685d4459065f004bd600a86b1679b` 不同。当前资产的expert有效性检查失败会中止，不换seed。
2. 一个episode至多4个attempt；失败后用同一scene seed、同一instruction重建环境，成功即停。每个episode开始PIM为空，retry保留此前attempt的预测trace并重置CTE状态。
3. 排序后的task按index modulo8分配slot。每condition有独立8个slot流；每次重规划只抽一次`[1,32,14]` float32初始动作噪声，新默认生成器与policy在同设备，同slot跨task连续。`--gpu-id`上的runner顺序执行slot/task并持久化state；8个slot不表示常驻8套模型或自动8GPU并行。设备与protocol均接受handoff审计。
4. task的全部rollout正常结束后才原子写出RNG状态；runner先核对完整checkpoint组合、任务顺序和日志末尾snapshot，再执行下一个task。完整run结束后才对全部attempt日志和视频统一审计。失败的部分run不能续跑或重用目录。protocol、recipe、checkpoint/foundation、CTE、bank/classifier/language encoder、slot、scene/instruction、reset scope、draw计数和视频路径参与审计；完整通过后才生成`COMPLETE`。
5. 报告第1/2/3/4次以内至少成功一次的累计 solved / total / rate，分母始终是全部200episode。没有第4次attempt的早期成功episode仍计为成功，不用“实际执行了第4次的次数”作分母。

8-slot连续与policy-device生成机制对齐参考；由于噪声形状、底座和实际model-server adapter不同，不宣称CUDA/pi0.5逐元素相同。不同条件的早期成功/重试次数也会改变之后episode的stream offset；整轮on/off比较共享manifest、预算、slot初始seed和权重。**同一重规划点的paired诊断**才严格复用初始动作噪声，且不额外推进流。

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

测试环境为CPU PyTorch2.7.1、Accelerate1.12.0；整套命令`OMP_NUM_THREADS=1 PYTHONPATH=src python -m pytest -q tests/zeva`。本轮结果为 **478 passed，0 skipped，32.59秒**，3条warning来自已有grad-norm Tensor拷贝和旧scheduler弃用提示。两个真实双进程Gloo测试均执行通过；若其他环境禁止loopback而skip，不能视为分布式已验证。全部36个变更/新增Python文件AST检查、文档相对链接和`git diff --check`均通过。

CPU测试覆盖CTE batch/streaming一致、过去动作shift、未来图像/动作不泄漏、EMA与BN、尾部mask、损失数值、episode无放回/划分、readiness拒绝、task memory隔离、预测cache到部署生命周期，以及真实小型WanVideoDiT/ActionDiT/MoT中的梯度与动作扰动。另有连续slot RNG、fresh output、manifest身份、固定分母、reset/video审计和mock runner状态机测试。

已覆盖warmup时foundation/EAP不变、联合阶段video输入投影/action/proprio实际更新、CTE/文本/VAE保持冻结；BF16 autocast下FP32主干能保留1e-6更新；完整bundle/Adam/scheduler/RNG恢复和activation checkpoint等价性。语言检索使用真实小型tokenizer快照与分类器端到端验证，故意令held-out GT task与instruction不同仍按语言预测路由，改变CTE validation特征不影响分类器训练，LOO同时影响mean-key与support，旧artifact/encoder漂移及foundation缺失或错误均被拒绝。

真实Hydra→runtime→小型WanVideoDiT/ActionDiT/MoT构造测试覆盖Stage2A、frozen Stage2B和joint Stage2B的FP32参数/Adam moments、BF16 autocast梯度、冻结参数不变，以及先升精度再加载父权重的小更新保留。真实双rank trainer验证ZeRO-1与普通AdamW DDP的参数、microbatch loss和moments逐位相同；跨epoch累积、warmup/joint/empty PIM、step2恢复后的样本顺序、history curriculum、每rank RNG、scheduler和再次集体保存均一致。

本轮覆盖reference SHA split、task顺序、80/显式40选择、真实source scope与TF32导出；allframe boundary/PIM/PreparedDataset末帧、full-horizon动作梯度；per-decision不同指令与去重/LOO隔离；reset提交、4attempts与policy-device history；device RNG/handoff、首指令缓存和旧合同拒收。严格EAP/PIM保持FP32部署且训练AMP，`.to(BF16)`不吞参数；旧numerical权重不能无声明加载。独立AST提取pinned上游EAP和PIM reader，在相同权重/输入下FP32与CPU BF16 autocast的输出、NLL、参数梯度逐元素相等；另保留此前CTE loss、sampler和72次memory提交差分结果。有限CPU检查不代表GPU/kernel或整个模型逐位等价。

部署测试使用真正的小型WanTextEncoder，验证FP32/BF16文本feature与原FastWAM编码完全一致、LRU命中/淘汰/零容量、返回值不可污染缓存、base/Zeva多次重规划复用及tokenizer身份稳定性。CPU测试拦截CUDA传输边界，仅证明编码器与特征传输的接线，不代表执行过真实CUDA内核。另覆盖模式CLI到子进程、16卡有效batch、checkpoint模式/文本身份审计，以及joint-default配置下base仍记录原始frozen foundation的构造与RNG身份。

这些验证说明实现能把context接到action且遵守数据/时间/评测合同。**本次修改没有实际运行80epoch GPU训练、16卡NCCL训练、完整模型5090推理或200episode RoboTwin仿真；没有新的显存峰值、延迟或成功率结果。** 正式运行仍需CUDA/Mamba、真实数据、FastWAM权重、RoboTwin资产和视频渲染环境。

最终判断应分开三个问题：Stage2A是否保持或改善base；同一Stage2B在`new_on`相对`new_off`是否提升第2–4次累计成功率；相对旧版本是否仍有收益。如果PIM能改变动作却不能改善成功率，下一步应检查历史相关性、CTE预测质量及真实失败恢复数据分布，不能只继续放大注入幅值。
