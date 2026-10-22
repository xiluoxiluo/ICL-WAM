# Zeva-Ego 独立 prefix 与 cross-attempt PIM：实现、训练与 RoboTwin 评测

> **历史版本指南（`latest_completed`）。** 本文记录上一版独立prefix实现，保留用于复现；文中的“新模型”/`new_*`指当时的三token模型。当前 `current_prediction` CTE、Ego prior、训练流程和连续slot评测见[新指南](zeva_ego_training_evaluation.md)。下方历史评测命令显式指定旧task与旧RNG规则，避免误用当前CLI默认值。

本次更新把此前相加在一个 behavior slot 中的 global/PIM 改为独立的 global、effect、PIM 三个 prefix token，配置名为 `prefix_layout=zeva_ego`。跨尝试历史仍由 `pim_variant=cross_attempt` 读出，并同时影响 prefix 与 action prior。是否改善后续尝试，需要用重新训练的权重做配对 RoboTwin 评测；gate 大小和单元测试结果都不能替代成功率。

参考版本固定为 [Zeva `feature/zeva_ego` / `bae85ad753d3afe4f2e56762373d48ac0267e101`](https://github.com/air-embodied-brain/Zeva/tree/bae85ad753d3afe4f2e56762373d48ac0267e101)。重点参考其 [`ZevaEffectActionPrior`](https://github.com/air-embodied-brain/Zeva/blob/bae85ad753d3afe4f2e56762373d48ac0267e101/src/openpi/zeva/cte_eap.py)、[`ZevaPIMEAP`](https://github.com/air-embodied-brain/Zeva/blob/bae85ad753d3afe4f2e56762373d48ac0267e101/src/openpi/zeva/pim_policy.py)、[跨尝试训练器](https://github.com/air-embodied-brain/Zeva/blob/bae85ad753d3afe4f2e56762373d48ac0267e101/scripts/robotwin/train_cross_attempt_pim.py) 和 [RoboTwin 评测协议](https://github.com/air-embodied-brain/Zeva/blob/bae85ad753d3afe4f2e56762373d48ac0267e101/scripts/robotwin/README_EVALUATION.md)。

这里对齐的是独立 token 布局与 PIM 读取。本文历史实现全程冻结父模型；参考 Zeva-Ego 则在500步 PIM-only warmup 后更新 foundation、EAP 和 PIM，两者的训练策略不同。FastWAM 继续使用各自有独立参数的 video/action experts 及其 MoT mixed attention，保持 joint14、动作预测长度32、三相机384×320。effect 来源也不同，见下文；完整的预测版与训练适配见新指南。

## 1. 本次具体改动

下表汇总当前设计。记忆生命周期、PIM reader、cross-attempt cache 与四次尝试协议沿用前一版；本次修改集中在独立 prefix、attention 可见性、训练与 checkpoint 迁移，以及评测入口的布局选择。

| 位置 | 新行为 |
| --- | --- |
| `memory.py`、`lifecycle.py` | 当前尝试的 effect 暂存在 pending buffer；结束后一次性提交。BIT 随尝试清空，PIM 跨尝试保留，新 episode 清空全部历史。最多保留 3 次已经结束的尝试，每次均匀压缩到 64 条，保留端点。 |
| `causal_prompt.py` | 使用当前 phase 查询历史 phase，以 effect 为 value，对所有有效的有界候选做可训练读出。新模式不再先截成 top-4；空记忆严格输出零。 |
| `behavior_prefix_adapter.py` | 独立生成 `[global, effect, PIM]` 及逐样本有效性 mask。新增 `behavior_effect_projector`；PIM 仍可条件化全局 token 和 prior，但不会先加进 global prefix slot。 |
| `fastwam.py` | 在 video token 流前保留三个独立位置；action query 可分别读取其 K/V。空 effect/PIM 的 K/V 从 attention 中屏蔽；训练、prefill 与缓存 action 推理使用同一布局。 |
| `trainer.py`、`checkpoint.py` | 先训练含 effect projector 的新 Stage2A，再冻结父模块训练 Stage2B；旧 Stage2A 仅能通过显式 warmstart 迁移到新 Stage2A。checkpoint 检查布局、effect 来源、schema 和参数完整性。 |
| `cross_attempt_cache.py`、新 cache builder | 历史 episode 只存一次，查询引用同任务的其他训练 episode；排除 query 自身与验证/test 历史。记录真实来源、split 与 SHA256。 |
| RoboTwin 默认入口、HTTP 入口 | 共用同一 policy 实现。真正的 memory-off 同时关闭 prefix 与 prior 中的 PIM，保留静态 task context、当前 phase 和 BIT。 |
| manifest runner、reporter | 固定 scene seed 和指令、最多四次尝试、失败后重建同一场景、成功即停；逐次记录动作预算、模型身份、PIM 来源与视频路径，拒绝不完整结果。 |

新 Stage2A 配置为 `robotwin_zeva_fastwam_ego_policy_3cam_384`；Stage2B 配置为 `robotwin_zeva_fastwam_cross_attempt_3cam_384`，后者的顶层及 prompt/adapter 均标记 `pim_variant: cross_attempt`。两者都显式启用 `prefix_layout: zeva_ego`。基础模型配置仍默认 `legacy_merged`，原有 legacy task 不会被隐式改写。

### 注入关系

设原始全局 task token 为 `g`，当前 phase 为 `q`，从已结束尝试中读取的 PIM context 为 `c`，当前尝试 BIT 中最新一个有效 completed effect 为 `e_latest`：

```text
c = PIM_reader(q, historical_phases, historical_effects, historical_mask)
g' = g + U(c)
prefix_tokens = [G(g'), E(e_latest), P(c)]
prefix_valid  = [True, has_completed_effect, has_enabled_PIM]
(prior_mean, prior_std) = frozen_prior(g', q, current_BIT)
loss = flow_matching_loss + 0.01 * prior_NLL
```

`G`、`E`、`P` 分别对应 `behavior_global_projector`、`behavior_effect_projector`、`prefix_project`；`U` 是 `pim_to_global`。**`P(c)` 占据独立 attention 位置，不再与 `G(g')` 相加。** 保留 `g'=g+U(c)` 是另一条有意保留的路径：PIM 可以同时改变全局条件和动作 prior。两个调用入口都传原始 `g`，条件化各执行一次。实现见 [`behavior_prefix_adapter.py`](../src/fastwam/zeva/behavior_prefix_adapter.py)。

新 Stage2A 关闭 PIM，训练 prior、action-prior adapter、`G` 和 `E`。进入新 Stage2B 后这些模块冻结，只训练 PIM reader、`U` 和 `P`；`U` 零初始化，`P` 使用标准差 0.002 的小幅非零初始化。冻结 prior 的参数，但保留 prior 对输入的梯度。memory-off 同时关闭 `U(c)`、`P(c)` 和 prior 的 PIM 条件，保留同一 checkpoint 中的 global/effect、phase 与当前 BIT。

### effect 来源与 attention 边界

`effect_source=latest_completed` 表示从按时间排列的 `bit_effects/bit_mask` 中取最后一个有效项。它必须已经完成完整的 16 个 raw action step，且 `end_raw_step <= query_raw_step`。新尝试尚无 completed effect 时，effect slot 为零且无效；PIM 关闭或为空时，PIM slot 同样无效。模型不让其他 token 读取这些空 slot 的 K/V，避免把“零向量”误当成一个仍有 softmax 权重的有效条件。

这与参考 Zeva-Ego 的 effect 来源有明确区别：参考 `cte_eap.py::step` 使用当前 `effect_predictor(z_t)`；本项目保留既有 CTE/BIT 的 completed-effect 语义。v4 cache 中 `record_type=phase_query` 的 `effect` 字段是零占位，**不能拿它充当当前 effect 预测**。本次也没有添加新的当前 effect predictor，或者重解释已有 CTE 权重；对应缓存写入见 [`build_zeva_robotwin_cache.py`](../scripts/build_zeva_robotwin_cache.py)。

三个 prefix token 使用冻结 video expert 的投影与 block，继承干净首帧的时间调制和文本可见性。它们可以读取其他有效 prefix 和物理首帧；action query 可以分别读取三者及首帧。prefix 与首帧不能读取未来视频或动作 token，避免训练时泄漏未来信息。训练、video KV prefill 和 action cached attention 都保留逐样本 mask，批次中不同样本可以有不同的 effect/PIM 有效性。实现见 [`FastWAM`](../src/fastwam/models/wan22/fastwam.py)。

因此，“video/action experts 有独立参数”不意味着 action 看不到这些 context：连接发生在已有 mixed attention 的 Q/K/V 读取中。本次保留这条 FastWAM 路径，没有新增 memory expert；独立 token 使 action 能对 global、effect、PIM 分配不同的 attention 权重，但是否学会有效使用仍需训练和评测。

旧 `pim_gate` 参数只为兼容保留，在新分支冻结且不参与输出。新分支日志不再把它显示成“记忆使用率”。

## 2. 哪些旧产物可以复用

可以复用匹配身份的 FastWAM base、CTE、完整历史 v4 phase/effect cache、semantic task map、归一化 stats、static behavior bank、initial readouts、retrieval head 和 static episode context cache。**仅把 prefix 拆成三个 token 不要求重新生成 BIT 或这些缓存。**

| 现有产物 | 新流程中的用法 |
| --- | --- |
| 原 legacy prepared Stage2A cache | 新 Stage2A 继续使用。新 task 保留 legacy reader/cache 合同，PIM 在这个训练阶段关闭。 |
| 已生成且身份匹配的 cross-attempt prepared train/validation cache | 新 Stage2B 直接复用；拆 token 不改变该数据格式。只有尚未构建或源数据/身份改变时才需要执行下一节的 builder。 |
| 只有旧 top-4 prepared cache | 可用于新 Stage2A，不能代替 Stage2B 的 cross-attempt cache；按下一节构建跨尝试版本。 |
| 旧 `legacy_merged` Stage2A addon | 可用 `policy_warmstart_checkpoint` 初始化新 Stage2A 的 prior、action-prior adapter 和 global projector；新增 effect projector 必须继续训练。 |
| 旧 Stage2B addon | 用匹配 reader 和显式 `prefix_layout=legacy_merged` 复现旧结果；不能作为新 Stage2B parent 或新布局的直接部署权重。 |
| 训练完成的 `zeva_ego` Stage2A addon | 通过 `policy_checkpoint` 初始化新 Stage2B；此时 effect projector 也会完整加载并冻结。 |

**新布局需要重新训练 Stage2A 和 Stage2B。** warmstart 是可选的初始化，不会把旧 Stage2A 自动转换成训练完成的新布局。新增 slot 会改变模型条件，必须先验证新父模型的行为；不能把尚未训练的 effect projector 冻结后直接进入 Stage2B。加载器检查 parent 的训练阶段、base/CTE/task-context 身份，以及 prior、action-prior adapter、global/effect projector 的完整参数。Stage2A 的 legacy reader 可以切换为 Stage2B 的 cross-attempt reader，policy 侧布局与 effect 来源必须一致。

### checkpoint schema

| prefix layout | reader variant | addon schema | effect source |
| --- | --- | --- | --- |
| `legacy_merged` | `legacy` | `legacy`，历史文件允许缺失 | 不使用独立 effect slot |
| `legacy_merged` | `cross_attempt` | `cross_attempt_v1` | 不使用独立 effect slot |
| `zeva_ego` | `legacy` 或 `cross_attempt` | `zeva_ego_prefix_v1` | 必须为 `latest_completed` |

历史 checkpoint 缺少 `prefix_layout` 时固定按 `legacy_merged` 解释；不会依据新 task 名自动升级。新布局的 `zeva_ego_prefix_v1` 优先于 reader 的 `cross_attempt_v1` 标记。完整加载与 Stage2B parent 加载都会拒绝 layout/effect/schema 不匹配；旧 Stage2A → 新 Stage2A 的跨布局迁移只能走显式 `policy_warmstart`。这些检查在独立 checkpoint loader 和模型保存/加载入口中一致，见 [`checkpoint.py`](../src/fastwam/zeva/checkpoint.py)。

先确认 Stage2A 在固定场景上的表现。如果 Stage2A 已经明显损害 base，应该先检查其 prior、归一化与注入训练。新 Stage2B 冻结该父模型，不能保证补偿所有父模型问题。

## 3. 构建跨尝试训练/验证缓存

以下命令从仓库根目录执行；路径是需要替换的本地占位值。`DATA_DIR` 必须和 v4 source cache 中记录的数据路径一致，不能只因为文件相同就换一个未匹配的路径。

```bash
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

DATA_DIR=/absolute/path/to/robotwin2.0
BASE_CKPT=/absolute/path/to/fastwam_base.pt
CTE_CKPT=/absolute/path/to/cte.pt
STATS=/absolute/path/to/robotw_stats.json
SEMANTIC_MAP=/absolute/path/to/semantic_task_map.json
TEXT_CACHE=/absolute/path/to/text_embeds_cache
PHASE_CACHE=/absolute/path/to/zeva_phase_effect_cache_v4
BEHAVIOR_BANK=/absolute/path/to/behavior_bank.pt
READOUT_CACHE=/absolute/path/to/initial_readouts.pt
RETRIEVAL_HEAD=/absolute/path/to/static_retrieval_head.pt
STATIC_CONTEXT=/absolute/path/to/static_episode_context.pt
STAGE2A_CACHE=/absolute/path/to/original_legacy_prepared_stage2_cache
CROSS_TRAIN_CACHE=/absolute/path/to/pim_cross_train
CROSS_VAL_CACHE=/absolute/path/to/pim_cross_validation

python scripts/build_zeva_cross_attempt_cache.py \
  --phase-cache "$PHASE_CACHE" \
  --semantic-map "$SEMANTIC_MAP" \
  --output "$CROSS_TRAIN_CACHE" --split train \
  --validation-fraction 0.1 --seed 42 \
  --max-past-attempts 3 --max-entries-per-attempt 64

python scripts/build_zeva_cross_attempt_cache.py \
  --phase-cache "$PHASE_CACHE" \
  --semantic-map "$SEMANTIC_MAP" \
  --episode-splits "$CROSS_TRAIN_CACHE/episode_splits.json" \
  --output "$CROSS_VAL_CACHE" --split validation \
  --seed 42 --max-past-attempts 3 --max-entries-per-attempt 64
```

两个 cache 使用同一个 episode split 文件。每个任务至少需要两个可用训练 episode 才能做 leave-one-episode-out 历史；需要验证集时还必须有可分出的验证 episode。构建器拒绝覆盖已有输出目录。

默认每个查询抽样 1–3 次完整历史，非空历史类型权重为 matched:far:cross-condition = 5:2:1。matched/far 根据初始 phase 相似度区分。只有提供真实 `--conditions episode_conditions.json` 才会构造 cross-condition 候选；格式为 episode ID 到非空 condition 字符串的 JSON 映射，训练和验证必须传入同一份文件。没有条件标签时，按实际可用类型重新归一化权重。

这里对 Zeva 训练配比作了明确适配：新 Stage2B 的父模型全部冻结，空历史样本不能训练新模块，因此默认训练不混入空历史；空历史回退用验证时的 memory-off 和单元测试检查。也关闭了旧的 prompt/context/support dropout，避免对已经抽样的历史重复丢弃。

CTE effect 必须来自完整 16 个 raw action step；查询时的 BIT 只能看到其结束时间不晚于查询时间的 effect。新 cache 保存原始特征幅值，和新部署分支一致。manifest、split、pairing 元数据及 `.npy` payload 都参与身份或完整性校验；首次加载会有额外校验 I/O。

**离线数据的边界：** 其他专家 demonstration 是上一尝试的代理数据，不等于真实失败后恢复轨迹。这个验证 split 只隔离本次 Stage2B 的查询与 PIM 历史来源，不宣称 CTE/static bank 从未见过这些 demonstration。最终结论必须来自独立 RoboTwin rollout；如后续失败恢复仍弱，应采集真实失败尝试，并在训练/验证间保持 episode 隔离。

## 4. 先训练新 Stage2A，再训练新 Stage2B

使用新的配置和 prepared 入口，不使用旧 shell 脚本里写死的机器路径。Wan、tokenizer、VAE 和 ActionDiT 初始化资产沿用本项目安装方式；如需切换预训练资产根目录，设置 `DIFFSYNTH_MODEL_BASE_PATH`。

### 4.1 新 Stage2A：学会读取独立 global/effect

`STAGE2A_CACHE` 指向原 legacy prepared cache，不要指向 `CROSS_TRAIN_CACHE`。新 Stage2A 保留该 cache 的 phase/BIT 数据接口和 legacy reader 配置，训练时禁用 PIM；因此不需要为拆 token 改造旧的 cache builder，也不需要放开 cross-attempt 配置只能用于 Stage2B 的限制。

```bash
python scripts/train_zeva_fastwam_prepared.py \
  --config-name train \
  task=robotwin_zeva_fastwam_ego_policy_3cam_384 \
  ckpt="$BASE_CKPT" \
  "data.train.dataset_dirs=[$DATA_DIR]" \
  data.train.pretrained_norm_stats="$STATS" \
  data.train.semantic_task_map_path="$SEMANTIC_MAP" \
  data.train.text_embedding_cache_dir="$TEXT_CACHE" \
  model.zeva.cte.checkpoint="$CTE_CKPT" \
  model.zeva.cache.path="$PHASE_CACHE" \
  model.zeva.prepared_stage2.pim_cache_path="$STAGE2A_CACHE" \
  model.zeva.task_context.bank_path="$BEHAVIOR_BANK" \
  model.zeva.task_context.readout_cache_path="$READOUT_CACHE" \
  model.zeva.task_context.retrieval_checkpoint="$RETRIEVAL_HEAD" \
  model.zeva.task_context.static_episode_cache_path="$STATIC_CONTEXT" \
  learning_rate=1e-4 max_steps=5000 batch_size=1 \
  output_dir=./runs/zeva_ego_stage2a
```

如有经过验证的旧 **Stage2A** addon，可在这条首次训练命令中增加以下覆盖参数：

```bash
model.zeva.policy_warmstart_checkpoint=/absolute/path/to/legacy_stage2a_addon.pt
```

该选项只复制既有 prior、action-prior adapter 和 global projector；`behavior_effect_projector` 仍从新初始化开始学习。源 checkpoint 必须是 `training_stage=policy_injection`、`prefix_layout=legacy_merged`，且 base/CTE/task-context 身份匹配。不能传旧 Stage2B，不能和 `policy_checkpoint` 或 `resume` 一起使用。

Stage2A 续训时保留上述数据和模型路径，使用 `resume=.../checkpoints/state/step_XXXXXX`，并显式清除 `model.zeva.policy_warmstart_checkpoint=null`。warmstart 是一次性权重初始化，resume 恢复本次训练的权重及训练状态，两者用途不同。

完成训练后，先用固定场景检查新 Stage2A 是否保留了可用的 base 行为。单任务封装入口可选 `--sim-task robotwin_zeva_fastwam_ego_policy_3cam_384 --prefix-layout zeva_ego --effect-source latest_completed --mode zeva_stage2` 并加载新 Stage2A addon。确认父模型后，把下方路径设为实际训练输出；不能在未训练的情况下将旧文件重命名或改 metadata 充当新 parent。

```bash
STAGE2A_CKPT=./runs/zeva_ego_stage2a/checkpoints/weights/step_005000_addon.pt
```

### 4.2 新 Stage2B：冻结完整父模型，训练跨尝试 PIM

这一阶段使用 cross-attempt train/validation cache，`policy_checkpoint` 必须来自上一节新布局的 Stage2A。继承的 global/effect projector、prior 和 action-prior adapter 都保持冻结。

```bash
python scripts/train_zeva_fastwam_prepared.py \
  --config-name train \
  task=robotwin_zeva_fastwam_cross_attempt_3cam_384 \
  ckpt="$BASE_CKPT" \
  "data.train.dataset_dirs=[$DATA_DIR]" \
  data.train.pretrained_norm_stats="$STATS" \
  data.train.semantic_task_map_path="$SEMANTIC_MAP" \
  data.train.text_embedding_cache_dir="$TEXT_CACHE" \
  model.zeva.cte.checkpoint="$CTE_CKPT" \
  model.zeva.cache.path="$PHASE_CACHE" \
  model.zeva.policy_checkpoint="$STAGE2A_CKPT" \
  model.zeva.prepared_stage2.pim_cache_path="$CROSS_TRAIN_CACHE" \
  model.zeva.prepared_stage2.validation_pim_cache_path="$CROSS_VAL_CACHE" \
  model.zeva.task_context.bank_path="$BEHAVIOR_BANK" \
  model.zeva.task_context.readout_cache_path="$READOUT_CACHE" \
  model.zeva.task_context.retrieval_checkpoint="$RETRIEVAL_HEAD" \
  model.zeva.task_context.static_episode_cache_path="$STATIC_CONTEXT" \
  learning_rate=5e-5 max_steps=2000 batch_size=1 \
  output_dir=./runs/zeva_ego_cross_attempt
```

两个阶段的多 GPU 命令都可把开头的 `python` 替换为 `torchrun --standalone --nproc_per_node=GPU数量`。脚本在加载大模型前绑定 `LOCAL_RANK`。Stage2B 续训使用 `resume=.../checkpoints/state/step_XXXXXX`，不要将新 Stage2B addon 填进 `policy_checkpoint`，也不要设置 `policy_warmstart_checkpoint`。

默认每 250 step 做离线验证，每 rank 最多 16 个样本，索引均匀覆盖整个验证缓存；每 500 step 保存。建议先看前 250–500 step 的训练和动作诊断，再决定继续到 2000 step；这些是初始预算，不是已证实最优超参。重点看：

| 信号 | 含义 |
| --- | --- |
| `memory/history_count`、`memory/pim_count` | 训练实际读到了多少历史/entry，是否长期退化成单条或空输入。 |
| `prefix/global_valid_fraction`、`prefix/effect_valid_fraction`、`prefix/pim_valid_fraction` | 三个位置的实际有效比例。global 应恒为 1；effect 在完整 effect 可用后才有效；memory-off 的 PIM 应为 0。 |
| `prefix/global_norm`、`prefix/effect_norm`、`prefix/pim_norm` | 分开观察三个 token 的幅值，防止一个聚合数值掩盖 effect/PIM 分支未起作用。 |
| `memory/prefix_delta_ratio` | memory-on 与同 checkpoint memory-off 的 prefix 差异尺度，覆盖 global 条件化与独立 PIM token。 |
| `memory/prior_mean_delta_rms` | PIM 对动作 prior 的实际影响。 |
| attention entropy / max weight | 历史读出是否长期几乎均匀，或只关注极少候选。 |
| `val_loss_memory_off`、`val_memory_gain` | 同一验证样本、相同 flow noise/time 下，关闭 PIM 前后的损失差；正 gain 表示 memory-on 损失更低。 |
| 部署 `action_exec_delta_rms` | 相同推理噪声下，实际要执行的前 `replan_steps` 个动作是否改变。 |

离线验证使用完整 Zeva 条件，不走丢失记忆输入的通用视频验证。验证会恢复训练 RNG 和各模块 train/eval 状态。`val_memory_gain` 是含 prior 项的离线损失差，不是成功率，也不能单独证明失败恢复有效。

## 5. 固定场景的四组 RoboTwin 对照

提供的 [`robotwin_cross_attempt_10x20.json`](../configs/robotwin_cross_attempt_10x20.json) 沿用参考分支的 10 个任务 × 每任务 20 个 scene seed 和完整 seen instruction，共 200 个 episode。改为本项目 manifest schema，并明确记录 FastWAM 动作合同；因此 manifest SHA 与 Zeva 原始文件不同。

统一使用 `demo_randomized`、joint14、horizon 32、默认执行 24 步后重规划、最多 4 次尝试。每次尝试从同一 seed/指令重建场景；episode 开始清空 PIM，重试只清理当前尝试状态，成功后停止。当前 RoboTwin 资产中的 expert 检查失败会中止，不换 seed。固定 scene 在参考环境有效不代表它在另一套资产中已经通过验证。

| condition | 加载模型 | 目的 |
| --- | --- | --- |
| `base` | 原始 FastWAM | 检查任务和动作接口基线。 |
| `old` | FastWAM + 旧 Stage2B | 和原 top-4 或此前 merged cross-attempt 方案比较；reader task config 必须与旧权重一致。 |
| `new_off` | FastWAM + 新 Stage2B，关闭跨尝试 PIM | 保留同一权重、独立 global/effect、静态 context、phase、BIT 和父模型 prior 的控制组。 |
| `new_on` | 与 `new_off` 完全相同的权重，打开 PIM | 测量跨尝试记忆贡献。 |

主比较是 `new_on − new_off` 的配对 SR@4，同时报告 SR@1–4。旧 `pim_shadow` 只检查 lifecycle 与 base 输出，不可用来代替这个 memory-off 对照。`old` 与新模型必须各自满足训练时的 task-context 身份；若历史 checkpoint 的 context mode 不同，应分别运行相同 manifest 的基线，而不是绕过身份检查。

```bash
OLD_ADDON=/absolute/path/to/old_stage2b_addon.pt
NEW_ADDON=/absolute/path/to/new_cross_attempt_addon.pt

python scripts/eval_zeva_robotwin_manifest.py \
  --manifest configs/robotwin_cross_attempt_10x20.json \
  --checkpoint "$BASE_CKPT" --dataset-stats "$STATS" \
  --cte-checkpoint "$CTE_CKPT" \
  --old-addon-checkpoint "$OLD_ADDON" \
  --new-addon-checkpoint "$NEW_ADDON" \
  --old-task-config robotwin_zeva_fastwam_pim_3cam_384 \
  --new-task-config robotwin_zeva_fastwam_cross_attempt_3cam_384 \
  --old-prefix-layout legacy_merged \
  --new-prefix-layout zeva_ego \
  --old-effect-source latest_completed --new-effect-source latest_completed \
  --task-context-bank "$BEHAVIOR_BANK" \
  --task-context-retrieval-checkpoint "$RETRIEVAL_HEAD" \
  --conditions base old new_off new_on \
  --model-rng-protocol request_seed_stream_v1 --slots 1 \
  --model-seed 0 --gpu-id 0 --num-inference-steps 10 \
  --output-dir ./evaluate_results/cross_attempt_seed0
```

上例**显式**将 `old` 设为原top-4 task。当前runner默认的old已经是上一版cross-attempt三token模型，new默认是新的预测CTE模型，因此复现本节时不能省略上面的task/source/RNG参数。如果要比较此前的 `cross_attempt_v1` 合并布局，将上例old task改为 `robotwin_zeva_fastwam_cross_attempt_3cam_384`，保留 `--old-prefix-layout legacy_merged`。runner按配置解析reader容量；通过Hydra运行合并布局也必须添加 `model.zeva.adapter.prefix_layout=legacy_merged`。

先加 `--dry-run` 可以在 CPU 上验证 manifest、解析各 condition 的 reader 容量/layout/effect 合同，写出冻结的 `manifest.json` 与 `command_plan.json`。dry-run 明确跳过权重文件存在性、模型加载和仿真检查；它不是 rollout 成功。dry-run使用的目录也不能再用于正式运行，必须另取新目录。全四组产生 40 个 task/condition 子进程计划，最多执行 3200 次尝试，实际数量因成功即停而减少。

用 `--tasks task_a task_b` 可生成独立 hash 的开发子集；不能把子集结果当作完整 200 episode 报告。正式结果每个 model seed 使用新的输出目录，不在已有 JSONL 后盲目续写。

### 单任务接线检查

在正式 manifest 前，可以用新 Stage2B checkpoint 检查固定 task/seed 的 on/off：

```bash
python scripts/eval_zeva_robotwin_fixed_attempts.py \
  --task click_alarmclock --seed 0 --model-seed 0 --max-attempts 4 \
  --model-rng-protocol request_seed_stream_v1 \
  --sim-task robotwin_zeva_fastwam_cross_attempt_3cam_384 \
  --prefix-layout zeva_ego --effect-source latest_completed \
  --ckpt "$BASE_CKPT" --dataset-stats "$STATS" \
  --cte-checkpoint "$CTE_CKPT" --addon-checkpoint "$NEW_ADDON" \
  --task-context-bank "$BEHAVIOR_BANK" \
  --task-context-retrieval-checkpoint "$RETRIEVAL_HEAD" \
  --mode pim_on --cross-memory-enabled --pim-diagnostic-interval 5
```

关闭记忆时只把 `--cross-memory-enabled` 换成 `--no-cross-memory-enabled`，保持 task config、checkpoint、effect source、scene/model seed、动作预算和指令一致；如指定了 `--instruction`，两次必须使用同一内容。这个单任务入口用于接线检查，完整结果仍使用冻结 manifest。复现旧合并布局则明确设置 `--prefix-layout legacy_merged` 并选择与旧 checkpoint 匹配的 `--sim-task`。

### 噪声、终止帧与视频

本节显式选择的**旧协议 `request_seed_stream_v1`**中，每个task/condition进程只加载一次模型并创建独立request-seed流。每次重规划取一个新seed，场景重建不重置流，但下一个task重新创建流；`--slots 1`保留原来的按task排序执行。它不是目前默认的 `torch_continuous_slot_v1`。旧completed-effect CTE依然要求每步观测。

该旧协议与Zeva的八个连续Torch RNG slot不同。当前默认的新连续CPU协议及其与参考CUDA实现的数值边界见[新指南](zeva_ego_training_evaluation.md#7-robotwin-pim-evaluation)。两种协议都可能因不同条件的成功时间/规划次数而使后续随机请求错位。历史实验需要多seed时应预先确定seed集合，并分别使用全新输出目录；单次run的bootstrap区间只反映固定任务集内的episode不确定性。

开发阶段加 `--pim-diagnostic-interval 5`，每五次规划额外计算一次同噪声 memory-off，记录前 24 个实际执行动作的差异。它重算关闭 PIM 的 prior，并且不消耗主请求 seed stream。额外推理会增加耗时，正式速度比较保持相同设置或关闭该诊断。

尝试结束时读取一次新的终止观测，把已完成的 4-action transition 送入 CTE；不把缓存的动作前帧当作 after-frame，不用 padding 伪造不完整 effect。每次尝试有独立且不在写日志后重命名的视频路径。

## 6. 汇总与判断

```bash
python scripts/report_zeva_robotwin_multiattempt.py \
  --manifest ./evaluate_results/cross_attempt_seed0/manifest.json \
  --on ./evaluate_results/cross_attempt_seed0/new_on.jsonl \
  --off ./evaluate_results/cross_attempt_seed0/new_off.jsonl \
  --output ./evaluate_results/cross_attempt_seed0/paired_review.json
```

runner 自动生成每组 `.summary.json` 与 `paired_pim.summary.json`。分母始终是 manifest 的 episode 数量；reporter 拒绝丢失/重复 episode、乱序尝试、成功后继续、运行异常、身份错配或动作预算变化。输出包括 micro/macro/per-task SR@1–4、首败后恢复率、尝试数、执行步数、耗时及 `new_on − new_off` 的 task 内 episode 配对 bootstrap 区间。

当前reporter还输出 `{attempt, solved, total, rate}` 累计点。保存到run目录的manifest包含运行protocol，hash与输入的源manifest不同；若传入reporter的 `--expected-manifest-sha256`，应使用 `command_plan.json` 中的 `manifest_sha256`。对没有新protocol字段的历史日志，`protocol_audit.status` 会明确标为 `not_recorded_legacy_manifest`，不会补造当时未记录的reset/RNG证据。

正式 runner 强制保存每次尝试的视频，只有所有条件的数据、报告和视频审计通过才写入 `COMPLETE`。视频审计检查路径唯一、文件存在且非空，不验证视频内容或编码。独立 reporter 默认执行同样审计；只有显式使用 `--metadata-only` 才跳过，并把结果标记为 `METADATA_ONLY`。没有耗尽步数预算却标记为 timeout 的记录也会被拒绝。

判断次序：先确认 base 和父模型工作正常，再确认 PIM 确实改变了 prefix/prior/已执行动作，最后看独立场景中的配对成功率。若动作变了而 SR 没提升，下一步应检查历史可用性、检索与失败恢复监督；不能仅凭非零增量宣称 ICL 有效。也不要根据本次测试集反复挑超参后仍把它当未见测试集。

## 7. 验证范围

本次默认 eager 路径实际通过 **208 项 CPU 测试**。覆盖三 token 顺序和独立投影、逐样本 padding mask、空 slot 与物理移除的输出等价、真实小型 WanVideoDiT/ActionDiT/MoT forward/backward、cached prefill/action 路径衔接、Stage2A effect 学习、Stage2B 父模型冻结和新 PIM 梯度，以及旧 checkpoint 隔离、完整 parent 初始化、显式 warmstart/hash 校验。回归还覆盖在线记忆生命周期、完整 v4→prepared 数据链、因果边界、checksum、默认/HTTP 兼容和固定 manifest 报表拒绝逻辑。

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/zeva -q
python -m compileall -q src/fastwam scripts experiments/robotwin
git diff --check
```

本次运行的是 CPU 回归和小型模型集成测试，没有训练完整模型的新权重，也没有进行真实 RoboTwin rollout 或生成 SR 数字。这些测试与命令 dry-run 验证实现和协议，不代表完整 GPU/SAPIEN 集成已经验证。正式训练和 rollout 需要在具备 CUDA、模型权重和 RoboTwin 仿真资产的环境执行上面的命令。
