# ICLWAM：把 Zeva 因果记忆接到 FastWAM

## 当前推荐：预测 CTE、语言检索与可切换的 foundation 训练

本项目按 [Zeva `feature/zeva_ego` 的 README 与实际训练/评测代码](https://github.com/air-embodied-brain/Zeva/tree/bae85ad753d3afe4f2e56762373d48ac0267e101) 实现CTE、EAP和PIM并接到FastWAM。**新默认遵循“除必要WAM适配，其余按Zeva-Ego设计对齐”**；底座、video/action接口、joint动作/时间尺度与语言输入空间保持明确边界。逐项修复与证据见 [对齐审计](docs/zeva_ego_alignment_audit.md)；可执行命令见 [训练与RoboTwin评测指南](docs/zeva_ego_training_evaluation.md)。公开数据清单或分类器训练配方缺失的部分明确标注，不声称已经逐条复现参考实验。

**离线训练推荐从 [`scripts/train/`](scripts/train/README.md) 开始。** `01`–`11` 覆盖环境检查、文本预计算、CTE、所有新缓存/语言检索组件及 Stage2A/B；`12_run_all.sh` 支持完整流程、`--from/--to` 和 `--dry-run`。默认复用旧脚本中的真实数据/权重路径，集中在 `_env.sh`，使用单进程CTE与16卡Stage2，并提供独立 `STAGE2A_FOUNDATION_MODE` / `STAGE2B_FOUNDATION_MODE` 开关和完整state续训入口。严格CTE所需的有序 task subset 与真实 episode conditions 仍需提供。

主要变化：

- **预测式 CTE**：三路独立 RGB224 + 三流 Mamba，读取当前观察和上一段实际执行的24步 joint14动作，预测下一边界的视觉变化；不再把4-step transition、16-step completed effect当成参考分支的语义。
- **完整 episode 训练**：明确task subset与真实conditions；condition×task分区、SHA排序/round计数的5% holdout，再过滤task subset/短episode；sampler遵循外部任务顺序。80epoch、分组LR、EMA0.99；默认精确epoch80并通过验证gate/三项gain及数据来源检查，显式探索仅epoch40/step26160。正式模型缺Mamba会报错。
- **独立条件与参考 prior**：`[G(g+U(c)), E(current_prediction), P(c)]` 三个token分别可读；EAP由phase查询global生成的8个waypoint，再把prior mean投影到ActionDiT。新路径不使用旧标量gate。
- **EAP/PIM数值对齐**：`numerical_contract=zeva_ego_v1`采用参考dropout判定和MHA调用；NLL保留原FP32 action，完整horizon平均；PIM按实际计算dtype归一化。训练FP32参数+BF16 AMP，部署EAP/PIM仍为FP32，只在WAM注入接口转换。旧合同不能静默混入。
- **纯语言 task retrieval**：固定 tokenizer 与文本输入 embedding 表，对 task instruction 做 masked mean；冻结的分类器先预测 task，再用该 task 的平均 CTE key 在同任务 train bank 内做 top-5 检索。推理不接收真实 task label；检索不依赖联合训练中变化的 image/text hidden states。
- **训练/部署使用同一task memory**：`train_query_exclusion=none`使用完整train bank；训练`per_decision`按当前样本指令，部署缓存attempt首指令。相同指令离线去重，query按episode/window/raw_step绑定；validation轨迹不进入bank，旧episode-static/LOO明确保留。
- **全部CTE帧与完整动作监督**：boundary v2/PIM v4包含末观察边界；H32不足时沿LeRobot重复末动作，EAP和WAM flow都计入完整horizon。Stage1仍只监督真实完整H24段；导出关闭TF32并核对完整source scope。
- **预测历史与split隔离**：attempt结束保留pending，在下次reset提交；当前query不能读取自身trace。在线最多4个历史attempt×64/总256，保留policy device。CTE/cache/Stage2共用split，离线历史仅来自train；online只读同评测episode此前attempt。
- **固定PIM训练配对**：按初始phase固定同条件最近/最远、异条件最近各一条train历史；Torch FP32余弦，平局取source原序第一项。真实conditions与CTE来源绑定；缩减课程必须显式启用并记录。
- **PIM阶段安排**：前500次更新只训PIM，之后联合更新 FastWAM foundation（LR1e-6）、EAP（1e-5）与PIM（5e-5）；CTE、VAE、文本编码器和语言检索保持冻结。修复 video 输入投影的梯度路径，启用逐层梯度检查点；可训练参数使用 FP32，计算可用 bf16 autocast，避免小学习率更新被参数舍入吞掉。
- **Stage2A也对齐联合训练**：默认foundation LR5e-6、EAP LR5e-5，常数学习率、无warmup，5000步/effective batch256。Stage2A和Stage2B均使用 `foundation_mode=joint|frozen`；frozen是明确的消融选项。所有可训练参数保留FP32。frozen Stage2B如果继承joint Stage2A，仍保存和部署父模型的foundation，不能回退原始base。推理始终固定权重。
- **PIM压缩**：在线默认两级相似度合并，阈值0.95、phase权重0.5、计数加权centroid；训练仍用reference均匀`linspace.round`采样。v4缓存、memory contract v2及评测身份绑定容量、压缩、reset提交和设备。
- **16卡训练**：`robotwin_zeva_ego_pim_16gpu_3cam_384` 使用普通DDP加PyTorch ZeRO-1优化器状态分片，batch1×累积16×16卡=256；保存的部署权重仍可在单卡加载，完整state另用于恢复优化器与RNG。
- **32GB部署路径**：policy的完整文本编码器默认独立放在CPU，按prompt缓存至多16份文本特征；action模型不再先在GPU构造完整文本编码器。评测记录文本权重/tokenizer身份、实际设备/精度和CUDA显存统计，GPU显存与速度仍需实机测量。
- **RoboTwin评测**：冻结10×20场景、同seed/指令重试、成功即停、最多4次、固定200分母；8个逻辑slot在policy device生成连续噪声并跨task交接，审计实际设备、reset、视频和artifact身份。旧CPU连续协议明确保留，不能与新协议混用handoff。

| 阶段 | 新入口 / task |
| --- | --- |
| CTE | `scripts/train_zeva_ego_cte.py` / `robotwin_zeva_fastwam_ego_cte_3cam_384` |
| 预测缓存 | `scripts/build_zeva_ego_boundary_cache.py` |
| task memory / 语言特征 / 全query static context | `scripts/build_zeva_ego_task_memory.py`；中间用 `scripts/train_zeva_task_language.py` 重建本地分类器 |
| PIM train/validation cache | `scripts/build_zeva_cross_attempt_cache.py`，自动识别新boundary schema |
| Stage2A | `scripts/train_zeva_fastwam_prepared.py` / `robotwin_zeva_ego_policy_3cam_384`；16卡使用 `robotwin_zeva_ego_policy_16gpu_3cam_384` |
| Stage2B | 同一prepared入口 / `robotwin_zeva_ego_pim_3cam_384`，使用 `foundation_mode=frozen\|joint` |
| 16卡 Stage2B | 同一prepared入口 / `robotwin_zeva_ego_pim_16gpu_3cam_384`，默认ZeRO-1、有效batch256 |
| 正式评测 | `scripts/eval_zeva_robotwin_manifest.py`，默认新预测版；on/off使用同一Stage2B addon及其foundation权重 |

新addon schema为 `zeva_ego_prediction_v1`。joint Stage2A/B以及继承joint父模型的frozen Stage2B都会另存配对的 `step_XXXXXX_foundation.pt`；部署根据 `foundation_checkpoint_required`加载并校验SHA256。`foundation_mode`表示本阶段是否更新，不能用它推断是否只需要原始base。

新同架构冻结对照可以直接在上述Stage2B命令中设置 `foundation_mode=frozen`，或使用别名任务 `robotwin_zeva_ego_pim_language_frozen_3cam_384`。正式评测用 `--new-foundation-mode frozen` / `joint` 匹配所选checkpoint；这个选项描述训练模式，不会在推理时解冻参数。`--policy-text-encoder-device cpu --policy-text-cache-size 16` 是单卡部署默认配置。

**从`4e95b26`及更早版本进入严格新默认：按新source scope/split/sampler训练CTE，重建boundary、bank/features/classifier、per-decision context v2与PIM v4，再训练Stage2A/B。** 这轮改变了CTE数据配方，上一轮“无需重训CTE”的结论不再适用。原FastWAM base、原数据、stats、语义表、匹配的policy文本cache与独立冻结语言encoder可复用。旧4e95重评用`robotwin_zeva_ego_pim_4e95_3cam_384`，旧b8语言版用`robotwin_zeva_ego_pim_language_legacy_3cam_384`，更早static版用`robotwin_zeva_ego_pim_frozen_3cam_384`；分别匹配原foundation模式及`--model-rng-protocol torch_continuous_slot_v1`。精确续训使用原commit和完整state，不给旧artifact补标签冒充新实验。

本次已加入CPU结构、梯度、因果性、缓存、部署生命周期和评测协议测试；**没有实际完成80epoch GPU训练或200episode RoboTwin仿真，因此尚无新版成功率结果**。FastWAM仍保持video/action experts、joint14、预测32步/执行24步；本项目没有改成pi0.5，也不直接加载参考pi0.5发布权重。

## 历史 V1 说明：以下命令与合同用于旧版复现

ICLWAM 在 FastWAM 的 RoboTwin action path 上加入 Zeva 的因果记忆模块。FastWAM 仍然负责根据图像、语言和本体状态生成动作；记忆模块从历史交互中提取当前 phase、短期 BIT 和跨尝试保留的 PIM，通过 video 流中的 behavior prefix 与 action hidden space 中的 prior residual 两条路径影响动作。

一次推理可以简单理解为：

~~~text
历史图像和动作
        ↓
       CTE
        ↓
   BIT / PIM 检索
        ↓
  Causal Prompt + Gaussian Action Prior
        ↓
   冻结的 FastWAM action path
        ↓
      RoboTwin 动作
~~~

FastWAM 主干不重新训练。CTE 先单独训练，Stage 2 只训练 Zeva addon，部署时按 episode 在线更新 BIT/PIM。

本文只说明已有 FastWAM RoboTwin checkpoint 之后的训练和测试流程。FastWAM 本身的安装、预训练和数据下载请参考原项目文档。

## 1. 准备环境和数据

下面假设在仓库根目录 ICLWAM/ 执行命令：

~~~bash
cd /path/to/ICL-WAM/ICLWAM
export PYTHONPATH="$PWD/src"
export DIFFSYNTH_MODEL_BASE_PATH="$PWD/checkpoints"
~~~

至少需要：

~~~text
data/robotwin2.0/robotwin2.0/       # RoboTwin LeRobot 数据
data/robotwin2.0/dataset_stats.json # 与该数据集对应的统计量
third_party/RoboTwin/               # RoboTwin 代码和 assets
/absolute/path/to/fastwam_base.pt   # 已有的 FastWAM RoboTwin checkpoint
~~~

当前 Zeva RoboTwin V1 的数据合同固定为：

- 三路相机：cam_high、cam_left_wrist、cam_right_wrist；
- action/state 维度：14；
- 一个 action chunk：32 个 action；
- 每 4 个 action 组成一个 CTE transition；
- 每 4 个 transition 形成一个 completed effect；
- 最终图像尺寸：[3, 384, 320]；
- action 和 state 使用同一个 dataset_stats.json。

Stage 1 和评测还需要 Wan2.2、tokenizer、VAE 以及 ActionDiT 初始化文件。若 ActionDiT 文件不存在，可以先生成：

~~~bash
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/fastwam.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda --dtype bfloat16
~~~

首次使用数据集时，先生成文本 embedding cache：

~~~bash
python scripts/precompute_text_embeds.py \
  --config-name train \
  task=robotwin_zeva_fastwam_3cam_384
~~~

如果 stats 或文本 cache 不在默认位置，需要在训练和测试命令中用 Hydra 覆盖对应路径。

## 2. 训练流程

已有 FastWAM checkpoint 后，按下面顺序执行：

~~~text
FastWAM checkpoint
        ↓
Stage 1：训练 CTE
        ↓
构建 phase/effect cache
        ↓
Stage 2A：训练 policy injection addon
        ↓（可选）
Stage 2B：加载 Stage 2A，只训练 PIM adapter
        ↓
RoboTwin 固定 seed 测试
~~~

最简路径先使用 task_context.mode=pooling，只验证 Zeva 到 FastWAM 的训练和部署链路；正式实验再构建 static task-context bank。

## 3. Stage 1：训练 CTE

CTE 学习历史视觉、动作与 phase/effect 的因果表示。它不需要 FastWAM checkpoint，但需要 RoboTwin 数据、统计量和 Wan VAE。

~~~bash
python scripts/train_zeva_cte.py \
  --config-name train \
  task=robotwin_zeva_fastwam_3cam_384 \
  device=cuda \
  output_dir=./runs/zeva_cte
~~~

多卡训练使用分布式启动器。第一个参数是每台机器的 GPU 数量；`batch_size` 按每个进程计算，`NNODES`、`NODE_RANK`、`MASTER_ADDR` 和 `MASTER_PORT` 可用于多机启动：

~~~bash
bash scripts/train_zeva_cte.sh 8 \
  --config-name train \
  task=robotwin_zeva_fastwam_3cam_384 \
  device=cuda \
  output_dir=./runs/zeva_cte
~~~

训练完成后主要文件是：

~~~text
runs/zeva_cte/cte.pt
runs/zeva_cte/config.yaml
runs/zeva_cte/dataset_manifest.json
runs/zeva_cte/metrics.jsonl
~~~

断点续训：

~~~bash
python scripts/train_zeva_cte.py \
  --config-name train \
  task=robotwin_zeva_fastwam_3cam_384 \
  device=cuda \
  output_dir=./runs/zeva_cte \
  resume=./runs/zeva_cte/cte.pt
~~~

分布式断点续训将同一组覆盖参数传给启动器即可：

~~~bash
bash scripts/train_zeva_cte.sh 8 \
  --config-name train \
  task=robotwin_zeva_fastwam_3cam_384 \
  device=cuda \
  output_dir=./runs/zeva_cte \
  resume=./runs/zeva_cte/cte.pt
~~~

cte.pt 必须和后面的数据尺寸、相机顺序、action 维度保持一致。不要把 CTE checkpoint 当作 FastWAM 的 ckpt 使用。

## 4. 构建 phase/effect cache

Stage 2 使用离线 cache 对齐训练窗口中的 CTE phase、BIT 和 PIM 输入。cache 由冻结的 CTE 根据完整 episode 历史生成：

~~~bash
python scripts/build_zeva_robotwin_cache.py \
  --config-name train \
  task=robotwin_zeva_fastwam_3cam_384 \
  model.zeva.cte.checkpoint=./runs/zeva_cte/cte.pt \
  model.zeva.cache.path=./data/robotwin2.0/zeva_cache/v4
~~~

生成目录至少包含：

~~~text
data/robotwin2.0/zeva_cache/v4/
├── manifest.json
├── episode_index.json
└── phase_effect-*.safetensors
~~~

如果更换数据集、dataset_stats.json、CTE checkpoint、相机顺序或 action 对齐参数，需要重新构建 cache。manifest.json 会检查这些身份信息，防止错误复用旧 cache。

## 5. Legacy Stage 2A：训练 Zeva policy injection addon

本节的 `legacy_merged` Stage 2A 使用已有 FastWAM checkpoint 作为冻结 backbone，只训练 Gaussian action prior、action-prior adapter 和 task/global behavior projector。新独立 prefix 还需训练 effect projector，请使用上方运行指南中的新 Stage2A task。

先用 pooling 路径跑通训练，不需要额外的 static task-context 文件：

~~~bash
BASE_CKPT=/absolute/path/to/fastwam_base.pt
CTE_CKPT=./runs/zeva_cte/cte.pt
CACHE_DIR=./data/robotwin2.0/zeva_cache/v4

python scripts/train_zeva_fastwam.py \
  --config-name train \
  task=robotwin_zeva_fastwam_policy_3cam_384 \
  ckpt=$BASE_CKPT \
  model.zeva.cte.checkpoint=$CTE_CKPT \
  model.zeva.cache.path=$CACHE_DIR \
  model.zeva.task_context.mode=pooling \
  output_dir=./runs/zeva_stage2a \
  mixed_precision=bf16
~~~

输出的 addon 通常位于：

~~~text
runs/zeva_stage2a/checkpoints/weights/step_XXXXXX_addon.pt
~~~

这个文件只保存 Zeva addon，不包含 FastWAM 主干。部署时仍然必须同时提供同一份 BASE_CKPT 和 CTE_CKPT。

Stage 2A 续训使用 state directory：

~~~bash
python scripts/train_zeva_fastwam.py \
  --config-name train \
  task=robotwin_zeva_fastwam_policy_3cam_384 \
  ckpt=$BASE_CKPT \
  model.zeva.cte.checkpoint=$CTE_CKPT \
  model.zeva.cache.path=$CACHE_DIR \
  model.zeva.task_context.mode=pooling \
  output_dir=./runs/zeva_stage2a \
  resume=./runs/zeva_stage2a/checkpoints/state/step_005000
~~~

## 6. Legacy Stage 2B：可选的 PIM adapter 训练

本节的旧 Stage 2B 先加载同布局的 Stage 2A addon，然后冻结 policy-injection 分支，只训练 CausalPromptEncoder、prefix_project 和 pim_gate。新 `zeva_ego` cross-attempt Stage2B 使用独立 PIM slot 和 `pim_to_global`，旧 scalar gate 保留在 state dict 中，但不参与输出或训练。

~~~bash
POLICY_CKPT=./runs/zeva_stage2a/checkpoints/weights/step_010000_addon.pt

python scripts/train_zeva_fastwam.py \
  --config-name train \
  task=robotwin_zeva_fastwam_pim_3cam_384 \
  ckpt=$BASE_CKPT \
  model.zeva.cte.checkpoint=$CTE_CKPT \
  model.zeva.cache.path=$CACHE_DIR \
  model.zeva.policy_checkpoint=$POLICY_CKPT \
  model.zeva.task_context.mode=pooling \
  output_dir=./runs/zeva_stage2b \
  mixed_precision=bf16
~~~

如果只验证 action-prior 路径，Stage 2A 完成后可以直接测试；如果要测试完整的 PIM residual，则使用 Stage 2B addon：

~~~bash
PIM_CKPT=./runs/zeva_stage2b/checkpoints/weights/step_XXXXXX_addon.pt
~~~

## 7. Static task context（正式实验可选）

默认配置的正式模式是 static，它会从初始图像、文本和 CTE 行为原型中读取 task context。没有这些文件时，可以使用 pooling 作为 baseline；使用 static 时，先构建 behavior bank 和 FastWAM 初始 readout：

~~~bash
python scripts/build_zeva_behavior_bank.py \
  --config-name train \
  task=robotwin_zeva_fastwam_static_3cam_384 \
  ckpt=$BASE_CKPT \
  model.zeva.cte.checkpoint=$CTE_CKPT \
  model.zeva.task_context.bank_path=./runs/zeva_task_context/behavior_bank.pt \
  model.zeva.task_context.readout_cache_path=./runs/zeva_task_context/readouts.pt
~~~

再训练 retrieval head：

~~~bash
python scripts/train_zeva_task_context_retrieval.py \
  --config-name train \
  task=robotwin_zeva_fastwam_static_3cam_384 \
  model.zeva.task_context.bank_path=./runs/zeva_task_context/behavior_bank.pt \
  model.zeva.task_context.readout_cache_path=./runs/zeva_task_context/readouts.pt \
  model.zeva.task_context.retrieval_checkpoint=./runs/zeva_task_context/retrieval_head.pt
~~~

之后训练和测试时去掉 task_context.mode=pooling，并传入这三个 static 文件。static bank、retrieval head、FastWAM checkpoint、CTE checkpoint 和 stats 必须属于同一套实验，代码会检查它们的 hash 和尺寸。

例如 Stage 2A 使用 static task context：

~~~bash
STATIC_BANK=./runs/zeva_task_context/behavior_bank.pt
STATIC_READOUT=./runs/zeva_task_context/readouts.pt
STATIC_HEAD=./runs/zeva_task_context/retrieval_head.pt

python scripts/train_zeva_fastwam.py \
  --config-name train \
  task=robotwin_zeva_fastwam_policy_3cam_384 \
  ckpt=$BASE_CKPT \
  model.zeva.cte.checkpoint=$CTE_CKPT \
  model.zeva.cache.path=$CACHE_DIR \
  model.zeva.task_context.mode=static \
  model.zeva.task_context.bank_path=$STATIC_BANK \
  model.zeva.task_context.readout_cache_path=$STATIC_READOUT \
  model.zeva.task_context.retrieval_checkpoint=$STATIC_HEAD \
  output_dir=./runs/zeva_stage2a_static \
  mixed_precision=bf16
~~~

## 8. Legacy RoboTwin 测试

本节复现旧completed-effect CTE与top-4 reader。固定seed封装入口当前默认已改为预测式Ego模型，因此下例显式指定legacy task和 `request_seed_stream_v1`。新模型及完整200episode协议请使用[预测版训练与评测指南](docs/zeva_ego_training_evaluation.md#7-robotwin-pim-evaluation)，不要向默认新task传入本节旧CTE/addon。

### 8.1 固定 seed 测试

先确认 RoboTwin assets 已安装，且 third_party/RoboTwin/policy/fastwam_policy 可以由脚本创建。对同一个 task 和 seed，建议分别测试 base、zeva_stage2、pim_shadow 和 pim_on。

base 只测试原始 FastWAM：

~~~bash
python scripts/eval_zeva_robotwin_fixed_attempts.py \
  --task click_alarmclock \
  --sim-task robotwin_zeva_fastwam_pim_3cam_384 \
  --ckpt $BASE_CKPT \
  --dataset-stats ./data/robotwin2.0/dataset_stats.json \
  --model-seed 0 --model-rng-protocol request_seed_stream_v1 \
  --seed 0 \
  --max-attempts 4 \
  --mode base
~~~

pim_shadow 运行 CTE/BIT/PIM lifecycle，但不加载 addon，也不改变 FastWAM 输出：

~~~bash
python scripts/eval_zeva_robotwin_fixed_attempts.py \
  --task click_alarmclock \
  --sim-task robotwin_zeva_fastwam_pim_3cam_384 \
  --ckpt $BASE_CKPT \
  --dataset-stats ./data/robotwin2.0/dataset_stats.json \
  --model-seed 0 --model-rng-protocol request_seed_stream_v1 \
  --cte-checkpoint $CTE_CKPT \
  --seed 0 \
  --max-attempts 4 \
  --mode pim_shadow
~~~

zeva_stage2 和 pim_on 需要 addon。若沿用 Stage 2 的 pooling baseline，直接用 Hydra 入口显式覆盖 task-context mode：

~~~bash
python experiments/robotwin/eval_robotwin_single.py \
  --config-name sim_robotwin_zeva.yaml \
  task=robotwin_zeva_fastwam_policy_3cam_384 \
  model.zeva.task_context.mode=pooling \
  ckpt=$BASE_CKPT \
  EVALUATION.task_name=click_alarmclock \
  EVALUATION.dataset_stats_path=./data/robotwin2.0/dataset_stats.json \
  EVALUATION.cte_checkpoint=$CTE_CKPT \
  EVALUATION.addon_checkpoint=$PIM_CKPT \
  EVALUATION.zeva_mode=pim_on \
  EVALUATION.model_seed=0 \
  EVALUATION.model_rng_protocol=request_seed_stream_v1 \
  EVALUATION.fixed_seed=0 \
  EVALUATION.max_attempts=4 \
  EVALUATION.action_horizon=32 \
  EVALUATION.replan_steps=8 \
  EVALUATION.skip_get_obs_within_replan=false
~~~

如果要测试 Stage 2A，只需把 addon 改成 Stage 2A addon，并把 zeva_mode 改成 zeva_stage2。

如果使用 static task context，可以改用固定 seed 封装入口，并传入 bank 和 retrieval head：

~~~bash
python scripts/eval_zeva_robotwin_fixed_attempts.py \
  --task click_alarmclock \
  --sim-task robotwin_zeva_fastwam_pim_3cam_384 \
  --prefix-layout legacy_merged --effect-source latest_completed \
  --ckpt $BASE_CKPT \
  --dataset-stats ./data/robotwin2.0/dataset_stats.json \
  --model-seed 0 --model-rng-protocol request_seed_stream_v1 \
  --cte-checkpoint $CTE_CKPT \
  --addon-checkpoint $PIM_CKPT \
  --task-context-bank ./runs/zeva_task_context/behavior_bank.pt \
  --task-context-retrieval-checkpoint ./runs/zeva_task_context/retrieval_head.pt \
  --task-context-top-k 5 \
  --seed 0 \
  --max-attempts 4 \
  --mode pim_on
~~~

### 8.2 四种测试模式

| 模式 | 需要的 checkpoint | 作用 |
| --- | --- | --- |
| base | FastWAM | 原始 FastWAM baseline |
| zeva_stage2 | FastWAM + CTE + Stage 2A addon | task behavior prefix + Gaussian action prior |
| pim_shadow | FastWAM + CTE | 只运行 memory lifecycle，检查 memory 是否影响 vanilla output |
| pim_on | FastWAM + CTE + Stage 2 addon | 完整 Zeva 条件，包括 PIM residual |

固定 seed 对比时，task、seed、max_attempts 和 replan_steps 要保持一致。max_attempts=4 表示同一个 fixed-seed episode 最多重试四次，不是四个独立 episode。

本节legacy Zeva测试必须满足（新预测式CTE的观测要求见新指南）：

- action_horizon=32；
- replan_steps 是 4 的倍数；
- skip_get_obs_within_replan=false，因为 CTE 需要每个已执行 action 对应的 after-frame；
- pim_top_k 与训练时的 prompt.persistent_length 一致；本节 legacy 配置为 4，cross-attempt 配置为 192。

## 9. 结果和排错

常见输出位置：

~~~text
runs/zeva_cte/                       # CTE checkpoint 和日志
data/robotwin2.0/zeva_cache/v4/      # phase/effect cache
runs/zeva_stage2a/                   # Stage 2A addon
runs/zeva_stage2b/                   # Stage 2B addon
evaluate_results/robotwin/            # RoboTwin 测试结果和日志
~~~

常见问题：

1. requires an existing pretrained_norm_stats：检查 data/robotwin2.0/dataset_stats.json，并确认训练、cache 和评测使用同一份文件。
2. Missing text embedding cache：重新运行 scripts/precompute_text_embeds.py，并确认 context_len=128。
3. requires an existing frozen FastWAM base checkpoint：ckpt 必须是 FastWAM checkpoint，不能填 cte.pt、addon 或 cache 目录。
4. cache manifest mismatch：数据、stats、CTE、相机顺序或尺寸不一致，重新构建 cache。
5. static task context requires ...：没有 static bank 时，在训练和 pooling 测试命令中加入 model.zeva.task_context.mode=pooling。
6. addon checkpoint mismatch：addon 必须由同一个 FastWAM base、同一个 CTE 和同一套 task-context artifacts 训练得到。
7. CUDA 或 RoboTwin assets 报错：Stage 2 和正式 rollout 需要 CUDA、SAPIEN/Curobo/pytorch3d 以及 RoboTwin assets；CPU 只适合做小型代码 smoke test。
8. `prefix_layout mismatch` 或缺少 `behavior_effect_projector`：新 Stage2B 需要完整的新布局 Stage2A；用新 Stage2A task 训练后再初始化。旧权重复现请显式选择 `legacy_merged`，不要删除 metadata 来绕过检查。

## 10. 本地代码检查

没有真实 checkpoint 或 RoboTwin 环境时，可以先运行：

~~~bash
PYTHONPATH=src pytest -q
python -m compileall -q src scripts experiments/robotwin/fastwam_policy
git diff --check
~~~

这些检查只能验证代码和接口，不能替代真实 RoboTwin rollout。最终比较应在相同 task、相同 seed 和相同 retry 设置下进行 base → zeva_stage2/pim_shadow → pim_on 对照。
