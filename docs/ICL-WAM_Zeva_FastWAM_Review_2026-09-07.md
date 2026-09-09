# ICL-WAM 对照 Zeva / FastWAM 的代码核对与修复清单

核对日期：2026-09-07。目标：保留 Zeva 的因果交互编码、双时间尺度记忆、phase retrieval、Causal Prompt 和门控策略条件注入，以基础 FastWAM 作为冻结策略；不启用 Joint、IDM、Optional-IDM，不增加第三个 expert。

**结论：基本架构方向正确，记忆到 action 的梯度路径已经接通；但当前版本还不能认定为“只替换 frozen policy、其余与 Zeva 一致”，也没有足够证据证明 memory 提高了 RoboTwin 成功率。应首先修复 Stage 1 采样与离线/在线 phase 历史不一致的问题。**

| 仓库 | 本次核对的 main 提交 |
| --- | --- |
| ICL-WAM | [d6721b2a2221111011a3b9a294925f0d1eadf4b3](https://github.com/xiluoxiluo/ICL-WAM/tree/d6721b2a2221111011a3b9a294925f0d1eadf4b3) |
| Zeva | [df25844189715ff5d20f3203b880c1d526a557f7](https://github.com/air-embodied-brain/Zeva/tree/df25844189715ff5d20f3203b880c1d526a557f7) |
| FastWAM | [7faa71108368fbb3b6885649f112af607427a2d4](https://github.com/yuantianyuan01/FastWAM/tree/7faa71108368fbb3b6885649f112af607427a2d4) |

本次进行了源码对照、已有测试、合成输入上的最小复现和缩小规模的真实模型模块反向传播。没有用户的模型权重、RoboTwin 数据或可用 GPU/仿真运行环境，因此没有运行完整模型训练或真实闭环评测。

**1. 已确认正确的部分**

| 核对项 | 结果与证据 |
| --- | --- |
| 基础模型选择 | 修改接在基础 `fastwam.py`；Joint、IDM、Optional-IDM 模型文件与 FastWAM 上游的文件 SHA 一致。MoT 仍只有 video/action 两个 expert。 |
| CTE 网络计算 | 与 Zeva 的核心计算等价。将 action_dim 都设为 14，加载同一份权重，在 T=1、9、17 的合成输入上逐项比较返回张量，最大绝对差为 0。增加的主要是输入校验与 RoboTwin 维度适配。 |
| CTE 损失 | 去除文档字符串、统一 AST 表达后，`cte_losses.py` 与 Zeva 一致，包含 action、vision、task、phase 和 effect-v3 目标。 |
| 时间单位 | 32 个 action 配 9 个边界观测，4 个 action 为一个 transition，4 个 transition/16 个 action 为一个 effect window。CTE 的 action stream 使用 right shift。 |
| 冻结策略 | `attach_zeva_addon()` 冻结基座，只开放 prompt encoder 与 adapter；训练器保持基座 eval。单帧 VAE 和 video KV 构造位于 no_grad，action 去噪路径保留对输入的梯度。 |
| Stage 2 目标 | 使用 action flow-matching loss；该路径没有训练未来 video loss，也没有 Joint/IDM 目标。 |
| memory → action | 使用真实 WanVideoDiT、ActionDiT、MoT 类搭建两层小模型，配合合成 VAE/输入训练 20 步：prompt 从第 2 步开始出现非零梯度，基座梯度为空、参数逐项保持不变。 |
| 已有测试 | 在 PyTorch 2.7.1+cpu 环境运行 `pytest -q`：33 passed。 |

冻结和残差注入的位置见 [FastWAM addon / action path](https://github.com/xiluoxiluo/ICL-WAM/blob/d6721b2a2221111011a3b9a294925f0d1eadf4b3/src/fastwam/models/wan22/fastwam.py#L745-L849)，训练 loss 见 [forward_zeva_action_train](https://github.com/xiluoxiluo/ICL-WAM/blob/d6721b2a2221111011a3b9a294925f0d1eadf4b3/src/fastwam/models/wan22/fastwam.py#L860-L962)。这些结果证明机制可训练、记忆能够影响 action；不代表真实任务性能已经提高。

**2. P1：Stage 1 的 batch 组织错误，使 task clustering 经常失效**

位置：[scripts/train_zeva_cte.py，第 218–259 行](https://github.com/xiluoxiluo/ICL-WAM/blob/d6721b2a2221111011a3b9a294925f0d1eadf4b3/scripts/train_zeva_cte.py#L218-L259)。

问题有两个相互关联的原因：

1. 顺序读取 episode 时，一旦下一条样本仍来自同一个 episode，代码立即对 `pending` 调用 `train_batch()`。因此连续 episode 数据通常无法积累到配置的 batch_size。
2. `next_episode_step` 在训练后才更新，而当前样本的重叠判断在训练前已经完成；flush 后不重新判断，当前重叠窗口仍被加入下一批。

对真实 `main.__wrapped__()` 使用一个合成连续 episode，设置 batch_size=16、max_steps=6，得到：

| optimizer step | 实际 batch size | 窗口原始 action 起点 | task loss |
| --- | --- | --- | --- |
| 1 | 1 | 0 | 0 |
| 2 | 1 | 1 | 0 |
| 3 | 1 | 32 | 0 |
| 4 | 1 | 33 | 0 |
| 5 | 1 | 64 | 0 |
| 6 | 1 | 65 | 0 |

这是合成数据上的执行结果，不是用户真实训练日志。它复现了正常连续 episode 顺序下的控制流问题。`_task_identity_clustering_loss()` 在没有同任务正样本时直接返回 0；单样本 batch 必然满足这一条件。见 [cte_losses.py，第 46–59 行](https://github.com/xiluoxiluo/ICL-WAM/blob/d6721b2a2221111011a3b9a294925f0d1eadf4b3/src/fastwam/zeva/cte_losses.py#L46-L59)。即使偶尔形成多样本 batch，如果所有任务 ID 各不相同，也依然没有正样本。

给 Codex 的修复要求：

- 将“合法窗口/历史样本的选择”和“optimizer batch 的组织”拆开。先生成可复现的样本索引，再组织 batch，不在遇到同 episode 时立即训练。
- 若继续使用非重叠 32-action 目标窗口，选择样本时就更新下一个合法起点，保证 0、32、64 等起点不混入 1、33、65。CTE 的历史上下文应另外保留，见下一条。
- batch 中包含多个任务，每个任务至少有两个有效历史样本，优先来自不同轨迹，以同时提供正样本和负样本。
- 记录实际 batch_size、有效 task positive 数、不同 task 数、窗口起点和 task loss；不要只记录配置中的 batch_size。
- 验收：在已知索引的合成多 episode 数据上检查实际 batch 内容；在非退化合成特征上确认 task loss 与其梯度有效。修复后重新训练 CTE。

**3. P1：离线缓存和部署的 CTE 历史不一致，直接影响 phase 检索**

位置：

- [build_zeva_robotwin_cache.py，第 171–213 行](https://github.com/xiluoxiluo/ICL-WAM/blob/d6721b2a2221111011a3b9a294925f0d1eadf4b3/scripts/build_zeva_robotwin_cache.py#L171-L213)：每个 9-frame 窗口独立调用 CTE。
- [zeva_robotwin_dataset.py，第 98–123、166–179 行](https://github.com/xiluoxiluo/ICL-WAM/blob/d6721b2a2221111011a3b9a294925f0d1eadf4b3/src/fastwam/datasets/zeva_robotwin_dataset.py#L98-L179)：Stage 2 只使用 effect_index=0 的窗口起点，取其 `phase_pre`。
- [deploy_policy.py，第 493–511、570–593 行](https://github.com/xiluoxiluo/ICL-WAM/blob/d6721b2a2221111011a3b9a294925f0d1eadf4b3/experiments/robotwin/fastwam_policy/deploy_policy.py#L493-L593)：在线不断追加历史，读取整次 attempt 的最后一个 phase。

例如真实时间 s=32：

- 缓存的窗口起点 phase：以图像 v32 为局部首帧，CTE 从 BOS action/零循环状态启动。
- 在线的当前 phase：CTE 已读入 v0、a0:4、v4……a28:32、v32，并保留这段历史的作用。

因此，Stage 2 的 query phase 总来自局部 t=0，按因果 right shift，它看不到窗口之前执行过的 action；部署时的 query phase 却依赖已执行历史。BIT 虽然携带部分历史 effect，但不能使这两个 phase 定义自动等价。

使用同一份随机初始化的真实 CTE 权重、同一张当前图像，比较完整历史中的 phase 与独立窗口首帧 phase，复现结果为：最大绝对差 0.39143，余弦相似度 0.66723。数值只用于证明两条路径不等价，不代表训练后模型的实际误差。该测试中 effect_post 差异约为 8.9e-8，因为它主要取决于 effect 的视觉起终点；不能把问题笼统描述成所有 effect 都算错。

另一个覆盖问题是：部署默认 `replan_steps=24`，会在 0、24、48……查询记忆；缓存目前主要围绕 0、32、64……目标窗口生成训练查询。见 [sim_robotwin.yaml](https://github.com/xiluoxiluo/ICL-WAM/blob/d6721b2a2221111011a3b9a294925f0d1eadf4b3/configs/sim_robotwin.yaml#L25-L35)。

给 Codex 的修复要求：

- 为离线和在线定义一个共同的“截至当前时刻的历史”接口。优先保留 Zeva 的 full-history 因果语义，不要只为通过测试而让部署每 32 个 action 清空 CTE。
- 离线按 episode 计算连续边界观测与执行 action 对应的 phase。可以利用 CTE 因果性一次计算完整轨迹，但必须用 prefix-vs-full 测试确认未来追加不改变过去的输出。
- 将 phase 查询记录和 effect 记录分开：phase 按当前 raw step 索引；effect 记录 start_raw_step、end_raw_step、phase_at_start、observed effect_post。仅允许 end_raw_step 不晚于当前查询时间的 effect 进入历史。
- Stage 2 的当前图像、32-action 监督窗口、proprio 和 query phase 必须共享同一个 raw step；覆盖实际 replan 的采样位置。不要使用当前目标窗口结束后的 phase_post 来“补历史”，那会引入未来信息。
- 增加验收：同一轨迹、同一 raw step，离线与在线的 phase、BIT 内容及检索键在合理浮点容差内一致；改变未来 action/image 不改变当前输入记忆。
- 修复后提升 cache schema/记录历史策略标识，重新生成缓存并重新训练 addon，避免复用旧语义的 cache。

**4. 与 Zeva 的实质性差异：需要明确哪些是保留、哪些是 FastWAM 适配**

这些差异不意味着方案不能工作，但使“除了 frozen policy 外完全一致”这一表述不准确。

| 机制 | Zeva 当前公开实现 | ICL-WAM 当前实现 | 建议 |
| --- | --- | --- | --- |
| global/task context | 使用 task-context bank 的任务原型，或根据初始观测和指令的冻结策略 readout 检索原型 | 将文本 context 做 masked mean，再沿特征维 adaptive pooling 到 256 | 若严格保留 Zeva，应移植 bank/prototype 语义；若保留文本池化，应明确标注为简化设计 |
| F_mem 输出 | 一个融合后的 `[B,256]` 向量 | 融合向量之外，再输出 phase、BIT summary、PIM summary，共 `[B,4,256]` | 主干融合计算相近，输出接口已扩展；严格版本先恢复原 F_mem 输出 |
| 注入位置 | 投影 Causal Prompt，加到已有 task-context prefix，由策略 attention 读取 | 32 个 learned queries 经 cross-attention 生成 `[B,32,1024]`，加到 ActionDiT 输入 hidden | 这是合理的候选 backbone adapter，但属于新的接口设计，需单独验证，不能称为逐模块照搬 |
| 无 PIM 时的增量 | PIM 残差乘 `has_pim`，为空时严格为零 | 没有相应的 has_pim 门控；task/phase 两个 token 始终有效 | 恢复 PIM 增量的空记忆语义，并区分 task/BIT 条件与 PIM 增量 |
| PIM 写入和读取 | 对已完成交互先写入 PIM，再按当前 phase 查询，可读本次 attempt 的已完成交互 | 已完成 effect 暂存在 pending，结束 attempt 才写入；查询还排除当前 attempt | 当前是“只读此前 attempt 的 PIM”变体；严格迁移需恢复完成后写入、允许读取已完成记录 |
| 初始化 | PIM projector 使用 Xavier，gate=0，先让 gate 学习 | output=0 且 gate=0，通过仅训练时的 epsilon 启动 output 学习 | 当前可以学动；更接近 Zeva 的做法是非零 projector + 零 gate，统一 train/eval gate 公式 |

出处：

- Zeva 的 [task-context 获取](https://github.com/air-embodied-brain/Zeva/blob/df25844189715ff5d20f3203b880c1d526a557f7/cosmos_framework/scripts/action_policy_server_robocasa365_zeva_pim.py#L862-L945)，ICL-WAM 的 [文本池化](https://github.com/xiluoxiluo/ICL-WAM/blob/d6721b2a2221111011a3b9a294925f0d1eadf4b3/src/fastwam/zeva/causal_prompt.py#L12-L28)。
- Zeva 的 [F_mem 与 has_pim 注入](https://github.com/air-embodied-brain/Zeva/blob/df25844189715ff5d20f3203b880c1d526a557f7/cosmos_framework/model/zeva/persistent_interaction_memory.py#L310-L396)及 [task-context prefix 调用](https://github.com/air-embodied-brain/Zeva/blob/df25844189715ff5d20f3203b880c1d526a557f7/cosmos_framework/model/generator/mot/cosmos3_vfm_network.py#L718-L780)。
- ICL-WAM 的 [四个 memory token](https://github.com/xiluoxiluo/ICL-WAM/blob/d6721b2a2221111011a3b9a294925f0d1eadf4b3/src/fastwam/zeva/causal_prompt.py#L86-L113)及 [32-query adapter](https://github.com/xiluoxiluo/ICL-WAM/blob/d6721b2a2221111011a3b9a294925f0d1eadf4b3/src/fastwam/zeva/behavior_prefix_adapter.py#L34-L70)。
- Zeva 的 [完成交互后写入并查询](https://github.com/air-embodied-brain/Zeva/blob/df25844189715ff5d20f3203b880c1d526a557f7/cosmos_framework/scripts/action_policy_server_robocasa365_zeva_pim.py#L1212-L1247)，ICL-WAM 的 [结束 attempt 写入](https://github.com/xiluoxiluo/ICL-WAM/blob/d6721b2a2221111011a3b9a294925f0d1eadf4b3/src/fastwam/zeva/lifecycle.py#L160-L179)及 [当前 attempt 排除](https://github.com/xiluoxiluo/ICL-WAM/blob/d6721b2a2221111011a3b9a294925f0d1eadf4b3/src/fastwam/zeva/lifecycle.py#L228-L247)。
- Zeva 的 [projector/gate 初始化](https://github.com/air-embodied-brain/Zeva/blob/df25844189715ff5d20f3203b880c1d526a557f7/cosmos_framework/model/generator/mot/cosmos3_vfm_network.py#L347-L354)。

空记忆差异也做了合成验证：令 BIT/PIM mask 全假，模拟已经训练过的非零输出投影和 gate，ICL-WAM 返回的 memory mask 仍为 `[true,true,false,false]`，产生非零 action hidden 残差。因此 `pim_on` 与 `base` 的差异可能来自 task/phase 条件适配，不能全部解释为 PIM 经验的收益。

如果保留单次尝试内的 BIT 条件，需要把它与 PIM 增量的门控语义说清楚。不能为了补 has_pim 就无区别地关闭所有 task/BIT 条件，也不能把一个新加的零向量 attention token 当作严格无影响：额外 token 仍可能改变 softmax 的归一化。应对选定的 adapter 显式验证 bypass。

**5. 多卡训练的静态风险：训练调用绕过了分布式模型的 forward**

位置：[trainer.py，第 841–880 行](https://github.com/xiluoxiluo/ICL-WAM/blob/d6721b2a2221111011a3b9a294925f0d1eadf4b3/src/fastwam/trainer.py#L841-L880)。

模型经过 `accelerator.prepare()` 后，循环通过 `unwrap_model()` 获取底层模型，并直接调用其 prompt encoder 和 `forward_zeva_action_train()`；没有通过准备后的 `self.model(...)` 执行本次训练 forward。当前 `FastWAM.forward()` 仍转发普通 `training_loss()`。

这使 DDP/DeepSpeed 的 forward 生命周期无法按正常入口运行，不能用单进程的梯度测试证明多卡同步正确。上游训练器也存在类似方法调用模式；本次新增的 Zeva 训练入口继续沿用了它。此项是静态审查发现，本次尝试的双进程 Gloo 验证在通信初始化阶段被执行环境的网络限制阻止，因此没有多卡实测结论。

给 Codex 的修复要求：将 Zeva 的 prompt 构造与 action loss 统一纳入模型的 `forward(sample)` 路由，训练器通过准备后的模型调用；保留冻结 video KV 的 no_grad 边界。在可运行分布式训练的环境中，用两个 rank 输入不同样本，检查一次 optimizer step 后 addon 参数逐项一致，并核对有效 global batch 和梯度累积行为。

**6. 工程与实验上的后续检查**

- 在线 `CausalCTEHistory.forward()` 每次都重新将整个 RGB 历史送入外部 VAE；`encode_history()` 把 B×T 展平为 batch。应缓存每个已观察边界帧的冻结 VAE latent，只编码新增帧，再把 latent 历史交给 CTE。否则长 episode 会反复计算旧帧并扩大单次 VAE batch；本次未测量真实显存或延迟。见 [history encoder](https://github.com/xiluoxiluo/ICL-WAM/blob/d6721b2a2221111011a3b9a294925f0d1eadf4b3/src/fastwam/zeva/lifecycle.py#L86-L114)。
- 已有 action 集成测试使用简化的 `_ActionExpert` 和 `_MoT`，其中若干测试只检查源码中包含某段字符串。33 项通过不能替代 batch 内容、真实算子反向传播、真实 checkpoint 和 rollout 验证。见 [action integration tests](https://github.com/xiluoxiluo/ICL-WAM/blob/d6721b2a2221111011a3b9a294925f0d1eadf4b3/tests/zeva/test_fastwam_action_integration.py)。
- Stage 2 离线 bank 从同任务其他 episode 取支持样本，并非当前机器人在该环境中实际失败后积累的经验。它可作为训练代理，但失败/retry 分布上的泛化需要实验。Zeva 官方训练配置还设置了 PIM context/support dropout=0.2；本次路径没有对应的随机置空策略。见 [Zeva PIM 配置](https://github.com/air-embodied-brain/Zeva/blob/df25844189715ff5d20f3203b880c1d526a557f7/cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_robocasa365_atomic5_zeva_pim.py)。
- 当前 evaluator 在首次成功后跳出 retry 循环，最终分母按 episode 计数。因此 max_attempts=4 得到的是最多四次尝试的成功率，不能与标准单次成功率直接比较；也不会产生成功后继续尝试的完整 FSSSS 类序列。见 [eval_policy.py](https://github.com/xiluoxiluo/ICL-WAM/blob/d6721b2a2221111011a3b9a294925f0d1eadf4b3/third_party/RoboTwin/script/eval_policy.py#L415-L468)。

**7. 建议的修复和验收顺序**

1. 修复 Stage 1 的窗口选择、任务分组与实际 batch，确认 task objective 真正工作。
2. 统一 CTE 历史与 raw-step 索引，按部署查询位置重建 phase/effect cache。
3. 按本次“保留 Zeva”目标确定 task context、F_mem 输出、PIM 完成后写入与空 PIM 门控；将唯一必要的策略接口改动明确写成 FastWAM adapter。不要同时混入第三 expert、Joint 或 IDM。
4. 多卡运行前修正 prepared model 的 forward 调用，并做跨 rank 参数一致性检查。
5. 用修复后的 CTE/cache 重新训练 addon；先验证真实模型上的非零 addon 梯度、基座不变、gate-zero/bypass 等价，再进入闭环评测。
6. 固定基座、addon 权重、环境 seed、指令、去噪 seed、replan_steps、推理步数和尝试次数，至少比较以下条件。

| 条件 | 检查目的 |
| --- | --- |
| base | 基础 FastWAM 能力 |
| pim_shadow | 运行记忆更新，但策略残差为零，检查接入与评测是否改变基线 |
| pim_on + 正确 PIM | 记忆增强后的行为 |
| 相同 addon/task/phase/BIT + 空 PIM | 区分普通条件适配和 PIM 经验贡献 |
| 相同 addon/task/phase/BIT + 错误或打乱的 PIM | 检查收益是否依赖正确经验内容，保持支持条数及 mask 一致 |

应分别报告首次尝试成功率、各 attempt 的结果和 Success@K，并保存 PIM 条数、检索来源/相似度、gate、残差范数、action 差异和实际执行轨迹。使用多个固定 seed 的配对比较；不能仅凭一次失败后成功或一个 action 差值宣布提升。

**8. 本次验证记录**

```text
已有测试：33 passed in 3.99s
CTE 同权重对照：T=1/9/17，各返回张量最大绝对差 = 0
Stage 1 合成连续 episode：batch_size 配置 16，实际前 6 批均为 1
Stage 1 起点：0, 1, 32, 33, 64, 65；对应 task loss 均为 0
完整历史 vs 重置窗口：phase 最大绝对差 0.39143，cosine 0.66723
真实两层 WanVideoDiT/ActionDiT/MoT + 合成 VAE：20 步反向传播通过
  第 1 步：prompt grad norm = 0，gate grad = 0，output projection 可更新
  第 2 步：prompt grad norm ≈ 1.27e-5，gate grad ≈ -0.003884
  第 20 步：prompt grad norm ≈ 0.001603
  基座全部梯度为空，参数逐项不变
  最终 tanh(gate) ≈ 0.03685
完整 30 层 checkpoint/CUDA/RoboTwin rollout：未运行
多进程训练：通信初始化被环境限制阻止，未验证
```

当前版本适合表述为“基于 Zeva 核心思路的 FastWAM 因果记忆适配实现”。先解决采样与历史对齐，再完成明确的接口取舍和闭环消融，才有依据判断它是否忠实迁移了目标机制、是否通过 memory 改善 action。
