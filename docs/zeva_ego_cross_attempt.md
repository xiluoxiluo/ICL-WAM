# Zeva-Ego cross-attempt PIM：实现、训练与 RoboTwin 评测

这次更新在 FastWAM 上实现一个可单独选择、可归因评测的跨尝试 PIM 分支。目标是验证历史交互能否改善后续尝试，而不是把 gate 数值变大作为优化目标。

参考版本固定为 [Zeva `feature/zeva_ego` / `bae85ad753d3afe4f2e56762373d48ac0267e101`](https://github.com/air-embodied-brain/Zeva/tree/bae85ad753d3afe4f2e56762373d48ac0267e101)。重点参考其 [`pim_policy.py`](https://github.com/air-embodied-brain/Zeva/blob/bae85ad753d3afe4f2e56762373d48ac0267e101/src/openpi/zeva/pim_policy.py)、[跨尝试训练器](https://github.com/air-embodied-brain/Zeva/blob/bae85ad753d3afe4f2e56762373d48ac0267e101/scripts/robotwin/train_cross_attempt_pim.py) 和 [RoboTwin 评测协议](https://github.com/air-embodied-brain/Zeva/blob/bae85ad753d3afe4f2e56762373d48ac0267e101/scripts/robotwin/README_EVALUATION.md)。

这是对 FastWAM 的适配：保持 joint14、动作预测长度 32、三相机 384 × 320 和已有单一 behavior slot。没有把 Zeva 的 pi0.5 权重、动作空间或公布的成功率视作本项目结果。

## 1. 本次具体改动

| 位置 | 新行为 |
| --- | --- |
| `memory.py`、`lifecycle.py` | 当前尝试的 effect 暂存在 pending buffer；结束后一次性提交。BIT 随尝试清空，PIM 跨尝试保留，新 episode 清空全部历史。最多保留 3 次已经结束的尝试，每次均匀压缩到 64 条，保留端点。 |
| `causal_prompt.py` | 使用当前 phase 查询历史 phase，以 effect 为 value，对所有有效的有界候选做可训练读出。新模式不再先截成 top-4；空记忆严格输出零。 |
| `behavior_prefix_adapter.py` | 新读出同时影响全局 task token 和 behavior prefix；冻结的 prior 接收条件化后的全局 token。使用小幅非零 prefix 投影初始化；新分支不乘旧 scalar gate。 |
| `fastwam.py`、`trainer.py` | 从完整 Stage2A 父模型初始化，只训练新 PIM 模块；保留 flow-matching loss，加上 PIM 条件下的 prior NLL。冻结 prior 参数，但保留其对输入的梯度。增加成对离线验证。 |
| `cross_attempt_cache.py`、新 cache builder | 历史 episode 只存一次，查询引用同任务的其他训练 episode；排除 query 自身与验证/test 历史。记录真实来源、split 与 SHA256。 |
| RoboTwin 默认入口、HTTP 入口 | 共用同一 policy 实现。真正的 memory-off 同时关闭 prefix 与 prior 中的 PIM，保留静态 task context、当前 phase 和 BIT。 |
| manifest runner、reporter | 固定 scene seed 和指令、最多四次尝试、失败后重建同一场景、成功即停；逐次记录动作预算、模型身份、PIM 来源与视频路径，拒绝不完整结果。 |

新配置名为 `robotwin_zeva_fastwam_cross_attempt_3cam_384`，顶层及 prompt/adapter 均标记 `pim_variant: cross_attempt`。旧配置继续使用 legacy 分支。

### 注入关系

设历史读出为 `c`，原始全局 task token 为 `g`：

```text
g' = g + W_global(c)
behavior_slot = behavior_global_projector(g') + W_prefix(c)
(prior_mean, prior_std) = frozen_prior(g', current_phase, current_BIT)
loss = flow_matching_loss + 0.01 * prior_NLL
```

`W_global` 零初始化，`W_prefix` 使用标准差 0.002 的小幅非零初始化。两条调用路径都接收原始 `g`，避免重复条件化。PIM 为空或关闭时，两个 PIM 增量都严格为零；这项回退由测试验证。没有新增 transformer memory expert，也没有拼接额外 token。

旧 `pim_gate` 参数只为兼容保留，在新分支冻结且不参与输出。新分支日志不再把它显示成“记忆使用率”。

## 2. 哪些旧产物可以复用

可以复用匹配身份的 FastWAM base、CTE、完整历史 v4 phase/effect cache、semantic task map、归一化 stats、static behavior bank、initial readouts、retrieval head 和 static episode context cache。

**需要一个经过验证的 Stage2A `policy_injection` addon 作为父模型。** 加载器检查父模型的 stage、base/CTE/task-context 身份以及 prior、action-prior adapter、global projector 的完整参数。旧 Stage2B 不能冒充 Stage2A，也不能直接加载进新分支。

需要重新生成 cross-attempt prepared cache，并重新训练 Stage2B。新 addon 带有 `cross_attempt_v1` schema 标记；旧 top-4 prepared cache 和旧 Stage2B addon 都会被明确拒绝。

先确认 Stage2A 在固定场景上的表现。如果 Stage2A 已经明显损害 base，应该先检查其 prior、归一化与注入训练。新 Stage2B 冻结该父模型，不能保证补偿所有父模型问题。

## 3. 构建跨尝试训练/验证缓存

以下命令从仓库根目录执行；路径是需要替换的本地占位值。`DATA_DIR` 必须和 v4 source cache 中记录的数据路径一致，不能只因为文件相同就换一个未匹配的路径。

```bash
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

DATA_DIR=/absolute/path/to/robotwin2.0
BASE_CKPT=/absolute/path/to/fastwam_base.pt
CTE_CKPT=/absolute/path/to/cte.pt
STAGE2A_CKPT=/absolute/path/to/stage2a_addon.pt
STATS=/absolute/path/to/robotw_stats.json
SEMANTIC_MAP=/absolute/path/to/semantic_task_map.json
TEXT_CACHE=/absolute/path/to/text_embeds_cache
PHASE_CACHE=/absolute/path/to/zeva_phase_effect_cache_v4
BEHAVIOR_BANK=/absolute/path/to/behavior_bank.pt
READOUT_CACHE=/absolute/path/to/initial_readouts.pt
RETRIEVAL_HEAD=/absolute/path/to/static_retrieval_head.pt
STATIC_CONTEXT=/absolute/path/to/static_episode_context.pt
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

## 4. 训练新的 Stage2B

使用新的配置和 prepared 入口，不使用旧 shell 脚本里写死的机器路径。Wan、tokenizer、VAE 和 ActionDiT 初始化资产沿用本项目安装方式；如需切换预训练资产根目录，设置 `DIFFSYNTH_MODEL_BASE_PATH`。

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
  output_dir=./runs/zeva_cross_attempt
```

多 GPU 可把命令开头的 `python` 替换为 `torchrun --standalone --nproc_per_node=GPU数量`。脚本在加载大模型前绑定 `LOCAL_RANK`。续训使用 `resume=.../checkpoints/state/step_XXXXXX`，不要将新 Stage2B addon 填进 `policy_checkpoint`。

默认每 250 step 做离线验证，每 rank 最多 16 个样本，索引均匀覆盖整个验证缓存；每 500 step 保存。建议先看前 250–500 step 的训练和动作诊断，再决定继续到 2000 step；这些是初始预算，不是已证实最优超参。重点看：

| 信号 | 含义 |
| --- | --- |
| `memory/history_count`、`memory/pim_count` | 训练实际读到了多少历史/entry，是否长期退化成单条或空输入。 |
| `memory/prefix_delta_ratio` | 真正的 behavior prefix 增量相对父模型 prefix 的尺度。 |
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
| `old` | FastWAM + 旧 Stage2B | 和原方案比较。 |
| `new_off` | FastWAM + 新 Stage2B，关闭跨尝试 PIM | 保留同一权重、静态 context、phase、BIT 和父模型 prior 的控制组。 |
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
  --task-context-bank "$BEHAVIOR_BANK" \
  --task-context-retrieval-checkpoint "$RETRIEVAL_HEAD" \
  --conditions base old new_off new_on \
  --model-seed 0 --gpu-id 0 --num-inference-steps 10 \
  --output-dir ./evaluate_results/cross_attempt_seed0
```

先加 `--dry-run` 可以在 CPU 上验证 manifest、写出冻结的 `manifest.json` 与 `command_plan.json`。dry-run 明确跳过权重文件存在性、模型加载和仿真检查；它不是 rollout 成功。全四组产生 40 个 task/condition 子进程计划，最多执行 3200 次尝试，实际数量因成功即停而减少。

用 `--tasks task_a task_b` 可生成独立 hash 的开发子集；不能把子集结果当作完整 200 episode 报告。正式结果每个 model seed 使用新的输出目录，不在已有 JSONL 后盲目续写。

### 噪声、终止帧与视频

每个 task/condition 进程只加载一次模型，持有独立的 request seed stream。每次重新规划取一个新模型 seed，重建环境时不重置这个流。所有 condition 都使用相同规则，避免 memory-off 由于反复使用同一噪声而缺乏重试多样性。

该实现与 Zeva 的八个 slot、每 slot 连续 Torch RNG 不是完全相同的数值协议。on/off 初始 model seed 相同，但成功时间与规划次数改变后，后续随机请求可能错位。因此正式结论建议至少重复 model seed 0、1、2；单次 run 的 bootstrap 区间只反映固定任务集内的 episode 不确定性。

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

正式 runner 强制保存每次尝试的视频，只有所有条件的数据、报告和视频审计通过才写入 `COMPLETE`。视频审计检查路径唯一、文件存在且非空，不验证视频内容或编码。独立 reporter 默认执行同样审计；只有显式使用 `--metadata-only` 才跳过，并把结果标记为 `METADATA_ONLY`。没有耗尽步数预算却标记为 timeout 的记录也会被拒绝。

判断次序：先确认 base 和父模型工作正常，再确认 PIM 确实改变了 prefix/prior/已执行动作，最后看独立场景中的配对成功率。若动作变了而 SR 没提升，下一步应检查历史可用性、检索与失败恢复监督；不能仅凭非零增量宣称 ICL 有效。也不要根据本次测试集反复挑超参后仍把它当未见测试集。

## 7. 验证范围

CPU 测试覆盖新旧配置/权重隔离、完整 Stage2A 初始化、真实小型模型 forward/backward、父模型冻结、空记忆与 masked NaN、在线记忆生命周期、完整 v4→prepared 数据链、因果边界、checksum、默认/HTTP 兼容以及固定 manifest 报表拒绝逻辑。

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/zeva -q
python -m compileall -q src/fastwam scripts experiments/robotwin
git diff --check
```

本次代码交付环境没有 CUDA GPU、真实用户 checkpoint 或完整 RoboTwin 仿真资产，因此没有在这里训练新权重，也没有生成真实 SR 数字。单元测试、离线小模型与命令 dry-run 的通过只验证实现和协议，不代表完整 GPU/SAPIEN 集成已经验证。正式训练和 rollout 需要在具备这些资源的环境执行上面的命令。
