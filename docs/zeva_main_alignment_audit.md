# Zeva main 对齐复查：源码、调用链与实际边界

复查日期：2026-10-22。目标是原 Zeva 的 observed-interaction 经验机制：离线使用成功示范，部署把机器人实际尝试中的交互结果作为 context，在固定模型参数下影响后续动作。

参考固定为 [Zeva main `17ca9fa0c68dca1fc7915dd6433de70a100e93a1`](https://github.com/air-embodied-brain/Zeva/tree/17ca9fa0c68dca1fc7915dd6433de70a100e93a1)。本次重新检查的本仓库基线是 `3991c47`。复查发现并修复了该基线中的真实差异，不能用上一轮测试数量代替逐项源码核对。

**结论范围：**公开的 CTE、EAP、MCE、PIM 与条件注入机制已逐项核对；WAM 的骨干、动作接口和 RoboTwin 环境继续明确适配。原 main 未完整公开的训练数据构造与优化流程，不能声称已逐项复制。本文不把 CPU 验证解释成真实 RoboTwin SR 提升。

## 1. 当前主线身份

| 项目 | 值 |
| --- | --- |
| profile | `main_observed` |
| numerical contract | `zeva_main_v2` |
| artifact lineage | `zeva_main_observed_effect_v2` |
| addon schema | `zeva_main_observed_policy_v2` |
| memory semantics | `completed_effect_trace` |
| CTE 窗口语义 | `full_observed_prefix_masked_tail_v1` |
| CTE/静态 query 精度约定 | `cuda_bf16_autocast_fp32_weights_v1`；CPU 开发回退 FP32 |
| 模型请求种子 | `zeva_main_attempt_replan_seed_v1` |
| 新训练输出根目录 | `runs/zeva_main_observed_v2` |

来源、reference commit、成功 episode、固定 train/validation 划分与精度合同会沿 CTE、cache、bank、retrieval head、static context、PIM 和 addon 传递。主线拒绝混入旧 main-v1 或 Ego 产物，不给旧文件自动补标签。

## 2. 本次发现并修复的问题

### 2.1 离线 CTE 丢失真实尾部交互

此前 Stage1 仅把完整的 32-control 源窗口拼成 episode。真实轨迹末尾不足 32 步时，其中已经完成的四步 transition 会被漏掉；离线训练因此可能看不到最后的成功交互，而在线仍会保留这些交互。

现在磁盘仍使用 WAM 的 9-frame/H32 存储形状，但索引包含至少一个真实四步 transition 的末窗口。读取时按真实 observation/action masks 截断，再拼接完整 observed prefix。padding 只用于存储和组 batch，不被当作观察。少于32步但至少存在完整四步交互的成功 episode 也保留；不足16步不会生成一个虚构的 effect。

phase/effect 导出也不再把“最后一个完整 H32 动作 query”当成观察历史的终点。完整 query 范围与真实 observed end 分开；短 episode 使用 query0，Stage2 的 action/NLL mask 只监督真实动作。

实现位置：[`cte_windows.py`](../src/fastwam/zeva/cte_windows.py)、[`stage1_sampling.py`](../src/fastwam/zeva/stage1_sampling.py)、[`cte_training.py`](../src/fastwam/zeva/cte_training.py)、两个 phase/effect cache builder 与两个 behavior-bank builder。

这项修复保证本地训练与在线时间语义一致；原 main 没有公开同一套 RoboTwin 离线 builder，因此不能称其为原离线脚本的复制。

### 2.2 Main 与 Ego 的推理精度被混用

此前 main addon 跟随 Ego 的 FP32 serving 规则。原 main 的公开模型 setup 会把整个 policy network 转为配置 precision；其发布配置为 BF16，EAP/MCE/projectors/gate 包含在该转换内。CTE 和静态 query head 是另一种路径：FP32 权重，在 CUDA BF16 autocast 内计算，然后在特征/检索边界转 FP32。

本次分开实现：

| 模块/阶段 | 当前 main 行为 |
| --- | --- |
| CTE 冻结特征提取 | FP32 参数；CUDA BF16 autocast；CPU 验证显式 FP32 |
| 静态 query head | FP32 参数；与 CTE 相同的 CUDA autocast 规则 |
| cosine/softmax 检索 | 离开 autocast，按参考 FP32 路径计算 |
| 部署 EAP/MCE | 完整加载后调用 `prepare_zeva_inference_precision()`，随 policy dtype，默认 BF16 |
| Stage2 可训练参数/optimizer | FP32 master + AMP；不把已有训练实例原地舍入后继续优化 |
| Main prior target/NLL | target 先转 policy dtype；mean/std 保留实际 compute dtype；WAM 真实动作 mask 保留 |
| Ego | 保留自身已定义的精度约定 |

另修正两处实际数值顺序：部署图像先 FP32 normalize 再转 BF16；main 初始 readout 先在 hidden dtype 中求均值，再转 FP32。离线与部署使用同一个 query/mean 路径。

参考：[`omni_mot_model.py`](https://github.com/air-embodied-brain/Zeva/blob/17ca9fa0c68dca1fc7915dd6433de70a100e93a1/cosmos_framework/model/generator/omni_mot_model.py) 的 `set_precision`、`build_net`、`_attach_stage2_behavior`；[main PIM server](https://github.com/air-embodied-brain/Zeva/blob/17ca9fa0c68dca1fc7915dd6433de70a100e93a1/cosmos_framework/scripts/action_policy_server_robocasa365_zeva_pim.py) 的 `_causal_interaction_features`、`_task_context_from_initial_observation`；[初始 readout](https://github.com/air-embodied-brain/Zeva/blob/17ca9fa0c68dca1fc7915dd6433de70a100e93a1/cosmos_framework/scripts/action_policy_server_robocasa365_zeva.py) 的 `extract_batch`。

### 2.3 PIM 满容量淘汰、同分排序与 merge 数值路径

此前即时 PIM 的淘汰 key 额外包含 `transition_index`，同分检索使用 stable argsort；这些细节可能改变保留的原型或传给 MCE 的 slot 顺序。

现在即时 main 分支跟随参考：

- merge score 使用归一化 phase/effect 的 Torch 矩阵余弦与加权和。
- 满容量按 `(merge_count, last_attempt_id, insertion_index)` 淘汰。
- retrieval 使用原 `torch.topk`。
- 合并时仍按累计 count 加权再归一化。

Ego 的 cross-attempt 分支保持自己的语义。参考：[原 PIM](https://github.com/air-embodied-brain/Zeva/blob/17ca9fa0c68dca1fc7915dd6433de70a100e93a1/cosmos_framework/model/zeva/persistent_interaction_memory.py)；本地：[`memory.py`](../src/fastwam/zeva/memory.py)。

### 2.4 Main 重试不再沿用 Ego 连续 RNG

主线使用公开 main 的请求种子公式：

```text
inference_seed = base_seed + 1000 * zero_based_attempt_id + replan_index
```

真实部署调用将这个 seed 交给 `infer_action[_zeva]`，逐次写入 `request_seeds`。前一个 attempt 多走或少走，不改变后续同一 attempt/replan 的配对 seed。报表核验 seed 序列、实际 replan 次数、噪声 device/dtype 和完整 episode；旧连续 RNG 的 main 日志不能当作新 v2 结果。

主线执行一次丢弃的模型 warmup，作用域为本地 policy 实例。它使用首个真实请求的相同冻结条件和相同 seed，不执行环境动作、不写记忆、不推进真实 request/replan 计数；随后重新用该 seed 生成首次实际动作。该作用域是本地明确实现：公开 main 声明 warmup 数量，但没有提供完整 RoboCasa client，不能推断其所有历史 run 的 warmup 调用位置。

参考：[公开 attempt protocol](https://github.com/air-embodied-brain/Zeva/blob/17ca9fa0c68dca1fc7915dd6433de70a100e93a1/cosmos_framework/model/zeva/attempt_protocol.py)；本地：[`zeva_main_protocol.py`](../src/fastwam/zeva_main_protocol.py)、[`deploy_policy.py`](../experiments/robotwin/fastwam_policy/deploy_policy.py)、[`zeva_main_evaluation.py`](../src/fastwam/zeva_main_evaluation.py)。

### 2.5 Stage2B 的 NLL 与冻结主干 activation checkpoint

补回 main PIM 阶段计算的 prior NLL 和指标。其 EAP 参数及 global/phase/BIT 输入固定，因此对 PIM 阶段而言该项是常数，补回不会给 MCE 或 foundation 引入新的 prior 梯度；它修正的是总损失/日志口径。

另修正冻结主干训练时的 activation checkpoint 条件。foundation 处于 `eval()` 仍需传递 context 到 action 的梯度，此时应允许对需要梯度的输入执行 checkpoint。现在无需把 foundation 改成 train mode，就能启用该重计算路径。相同输入下 loss 和全部参数梯度已作对照。

## 3. 为什么 WAM 的独立 video/action 分支仍能读到 context

Main 的两条条件路径保留为：

```text
global + phase + BIT + retrieved PIM
    → MCE → PIM projection → tanh(gate)
    → 与 global projection 相加成为一个 behavior slot
    → WAM video K/V → action attention

global + phase + BIT
    → Gaussian action prior mean → action projection
    → action hidden
```

PIM 不直接修改 prior 的 global 输入。训练 prior residual 的 dropout 为整样本40%，推理 scale 为0.5；使用的是 prior mean。PIM gate 初始为0与原 main 一致，空 PIM 是精确旁路。零 gate 的等价对象是训练过 EAP 的 Stage2A parent。

原 main 将 behavior slot 放在 BOS/文本之前，所在段为 causal attention；该 slot 归入 understanding 分支。因此本地 main prefix 不读取后续图像或文本，符合其因果方向。action 层实际拼接 video K/V 和 action K/V，prefix K/V 对 action 可见；训练 prefill 位于 `no_grad` 外，所以冻结主干不阻断 addon 梯度。

原始顺序依据：[sequence packing](https://github.com/air-embodied-brain/Zeva/blob/17ca9fa0c68dca1fc7915dd6433de70a100e93a1/cosmos_framework/data/generator/sequence_packing/sequence.py)；原注入依据：[`_encode_behavior_prefix` / `_encode_action`](https://github.com/air-embodied-brain/Zeva/blob/17ca9fa0c68dca1fc7915dd6433de70a100e93a1/cosmos_framework/model/generator/mot/cosmos3_vfm_network.py)。本地对应 [`fastwam.py`](../src/fastwam/models/wan22/fastwam.py) 与 [`mot.py`](../src/fastwam/models/wan22/mot.py)。

用 video expert 承载 prefix、保持 FastWAM 的 T5 输入空间和原生位置/时间调制，属于骨干适配。两个完整 policy 不能据此称为逐层数值同构；但条件影响动作的实际通路已经测试。

## 4. 逐项审计结果

| 模块 | 结论与边界 |
| --- | --- |
| CTE 核心 | 三流因果结构、历史 action 右移、phase、pre/post effect、固定 observed-delta target、EMA、损失与梯度路径对照通过；原始核心没有混入 Ego predictive CTE。 |
| BIT | 最近4个已完成 effect，右侧对齐；attempt 重置清空；不把未执行 horizon 当历史。 |
| PIM | effect 每16条实际控制完成后立即可用；当前 attempt 与本 episode 之前 attempt 均可读；不按成功与否过滤经验。 |
| 时间配对 | 4条控制对应一次真实观察边界；effect 使用本16步窗口起点 phase；执行24步重规划不会错误替换该起点。 |
| 结束处理 | terminal fresh observation；只提交已完成窗口；不足4/16步的尾部不补造 effect。 |
| Task query | 冻结初始 image + instruction readout → MLP检索头 → task-context bank top-5；不用 Ego language classifier，也不使用评测真实 task label作为query。 |
| MCE/EAP | 原 main gated merged slot + BIT prior；default gate0、dropout0.4、inference0.5；不沿用 Ego 三独立 token。 |
| PIM 阶段冻结 | 默认 foundation/EAP 冻结，只更新 MCE/projector/gate；学习率倍率5/5/1与公开 main PIM配置一致。 |
| Joint 开关 | A/B 可独立 `foundation_mode=frozen|joint`；joint 是显式 WAM 实验。joint A 后 frozen B 仍部署继承的 foundation。 |
| 部署参数 | foundation、CTE、addon、静态 head 固定；只有经验容器、缓存、计数随运行改变。 |
| Success replay | 默认关闭；显式启用时只使用首次确认成功后的实际 action trace，具有 phase 匹配；首次成功之前不可用。 |
| 指标 | 单次 `SR@1`、最多K次至少成功一次的 `CSR@K`、首次成功之后的稳定率分开报告。 |

## 5. 直接源码数值对照与回归证据

采用固定 SHA 的原文件作为独立参考，比较相同维度、输入和权重。以下等值结论只适用于实际运行过的 CPU 配置，不能外推成不同 backbone/GPU 的整策略相同。

| 对照 | 本次结果 |
| --- | --- |
| CTE | 小模型、4条不同长度/padding轨迹；初始化、18个输出、12个loss、全部参数/输入梯度、EMA均 `rtol=atol=0`；完整历史与已观察 prefix 的早期结果一致。 |
| MCE/EAP | 3个 seed、CPU BF16参数/输入、空与混合 BIT/PIM；MCE、prior mean、prior std最大绝对差0。另有共同FP32配置对照。 |
| 静态检索 | 3个 seed；head初始化/forward、对比损失与全部梯度、top-k index/score/value在CPU FP32精确一致。修正了等价表达式中计算图构建顺序导致的梯度末位舍入差。 |
| PIM | 90次写入、74次merge、容量5强制淘汰、3个attempt、90次query；向量、count、first/last attempt及返回张量精确一致。 |
| 真实 WAM 调用 | CPU小型 Wan/MoT/ActionDiT：condition进入实际K/V与action denoise，训练和推理首步对照、零gate等价、非零gate梯度、冻结checkpoint梯度对照。 |
| 真实数据入口 | raw/latent尾部导出、mmap prepared数据→模型forward/backward；padding动作不产生flow/NLL有效监督，train/validation episode隔离。 |
| 部署入口 | 真实policy方法测试RGB处理、16/24步phase配对、terminal提交、请求seed和丢弃warmup；并验证Ego旧分支未被main协议替换。 |

回归入口：

```bash
PYTHONPATH=src python -m pytest -q tests/zeva
git diff --check
```

最终完整回归：**653 passed, 2 skipped, 7 warnings**（CPU Torch 2.7.1，40.64秒）。两个跳过项分别为 joint-foundation 与 ZeRO-1 的 CPU DDP 测试，原因均为当前环境不允许本地 loopback socket；不是已经通过的多卡验证。7条 warning 来自已有 tensor-copy 和 scheduler API 提示。`git diff --check`、修改的训练环境脚本 `bash -n` 及隔离解释器下的只读 preflight 均通过。尚未运行 CUDA、多GPU训练或真实RoboTwin rollout。

## 6. 仍然保留的适配，以及不能替作者补全的细节

1. **Backbone/机器人接口：**保留 FastWAM、Wan2.2 VAE、joint14、原三相机布局、H32预测/H24执行和 WAM 采样器。原 main 的 RoboCasa policy/动作空间/图像大小不同。4/16在这里指实际 `take_action` 调用数；一次调用可能含多个仿真物理 substep。
2. **公开范围：**main 提供 CTE/model/loss、PIM机制、server和部分配置，但缺少可完整复现的 CTE optimizer/data recipe及部分基座配置依赖。CTE step-based AdamW、固定success holdout、离线示范轨迹均值prototype、支持经验采样与缓存工程属于本仓库明确选择。已核对公开的静态head/损失/检索公式，不把未公开的bank制作过程称为逐项复现。
3. **训练数值：**FP32 master + AMP、DDP/ZeRO-1和精确续训是本地训练实现；不能由模块公式对齐推导出原训练轨迹逐bit一致。主线已纠正公开可确认的输入/target转换和部署精度。
4. **评测环境：**默认 RoboTwin 场景清单沿用已冻结的10×20 scene/instruction条目，但新建独立main manifest，不继承Ego memory/protocol。原 main README 的 benchmark 为 RoboCasa；本仓库没有声称重现其表格结果。
5. **失败收益的归因：**当前 `memory on/off` 实际切换的是 PIM 条件；两组均保留 global context、当前 attempt 的 BIT 和 action prior。main PIM 在当前 attempt 内也积累经验，因此该同权重配对检验 PIM 的增益。要证明专门来自之前失败的内容，需要共享失败轨迹并控制后续对照。匹配 seed 不代表两条策略轨迹访问相同状态。

## 7. 使用与迁移

完整路径、环境和编号脚本见 [`scripts/train/README.md`](../scripts/train/README.md)。新输出目录避免与基线3991c47混用。

```bash
bash scripts/train/12_run_all.sh --dry-run
bash scripts/train/01_check_environment.sh
bash scripts/train/12_run_all.sh --from 03 --to 11
```

以上运行以已经具有匹配文本 cache 和可信 success manifest 为前提；没有文本 cache 时从02开始。旧main-v1需重建03–09并重训CTE/Stage2A/B，直接重复运行03–11即可。已有原始数据、base/action checkpoint、stats、semantic map、success manifest及匹配文本cache可复用；旧trainer state不用于新合同精确续训。

RoboTwin入口是 [`eval_zeva_main_robotwin.py`](../scripts/eval_zeva_main_robotwin.py)。先用同一新checkpoint做 `single_attempt` 和 `stop_on_success` 的memory off/on配对；`fixed_attempts`用于首成后稳定性，success replay必须单独记录。评测dry-run会创建计划目录，真正运行要使用另一个全新输出目录。

本次没有运行用户的16×96GB训练服务器、真实RoboTwin闭环或32GB RTX5090显存测试。需要在这些环境中获得SR、吞吐和峰值显存后，才能判断这套机制在用户数据与任务上的实际收益。
