# ICL-WAM 第二轮核对与最小修改建议

审查日期：2026-09-08。目标保持不变：保留 Zeva 的 CTE、BIT/PIM、phase retrieval、Causal Prompt、gated residual conditioning，以冻结的基础 FastWAM 执行动作；不引入 Joint、IDM、Optional-IDM 或第三个 expert。

**结论：这次修改解决了上一轮的主要采样逻辑和 phase 历史语义问题，当前架构值得保留。剩余优先事项是两个具体工程问题：索引扫描的越界终止错误，以及离线缓存构建时保留整个数据集的解码窗口。它们都可局部修复，无需扩大网络结构。**

本结论针对下列固定版本：

| 仓库 | 提交 |
| --- | --- |
| ICL-WAM 本轮 | [7df6c8546f7c6331bee8b8e2044b93bbe77ffe7c](https://github.com/xiluoxiluo/ICL-WAM/tree/7df6c8546f7c6331bee8b8e2044b93bbe77ffe7c)，fix: align Zeva FastWAM history and training |
| ICL-WAM 上轮 | [d6721b2a2221111011a3b9a294925f0d1eadf4b3](https://github.com/xiluoxiluo/ICL-WAM/tree/d6721b2a2221111011a3b9a294925f0d1eadf4b3) |
| Zeva | [df25844189715ff5d20f3203b880c1d526a557f7](https://github.com/air-embodied-brain/Zeva/tree/df25844189715ff5d20f3203b880c1d526a557f7) |
| 基础 FastWAM | [7faa71108368fbb3b6885649f112af607427a2d4](https://github.com/yuantianyuan01/FastWAM/tree/7faa71108368fbb3b6885649f112af607427a2d4) |

本轮共比较 24 个改动文件。Zeva、FastWAM 上游提交与上轮相同。源码没有被本次审查修改。运行环境为 PyTorch 2.7.1+cpu；没有真实 Wan 权重、RoboTwin 数据、CUDA 完整训练或仿真评测结果。

## 1. 已修复且有验证支持的部分

| 上轮问题 | 本轮修改 | 核对结论 |
| --- | --- | --- |
| Stage 1 同 episode 立即 flush，出现单样本 batch 和重叠起点 | 先生成窗口索引，再用 TaskBalancedCTEBatchSampler 组批 | 在正常终止的合成 Dataset 上，真实训练入口连续三批均为 16 样本、4 个任务、8 个 episode；每批 24 对同任务正样本；起点只含 0/32/64/96；task loss 约 2.71。旧批组织问题已修复，但索引扫描尚有第 2 节的边界错误。 |
| 离线每窗口从 BOS 重启，在线使用完整历史 | v4 cache 按 episode 拼连续历史，分开保存 phase_query 和 completed effect | 实际缓存脚本生成的 phase 与逐步在线 prefix 最大差约 3.05e-7，BIT 最大差约 1.19e-7。该历史对齐修复有效。 |
| 训练查询只覆盖 0/32/64，部署重规划为 0/24/48 | query index 保留每 4 个 raw action 的合法查询 | 合成缓存覆盖 0/4/8/.../64，包含 24/48。查询仍只选择有完整 32-action 监督的起点，尾部不完整窗口被过滤。 |
| PIM 到 attempt 结束才写入，并排除当前 attempt | completed effect 立即写入，允许当前 attempt 检索已完成交互 | 时间边界合理：只写已完成 effect，BIT 在 attempt 边界清空，PIM 跨 retry 保留。 |
| output 与 gate 同时为零，依赖训练时 epsilon | output Xavier，gate=0，统一 tanh(gate) | 首步 gate 梯度非零，第二步 prompt 梯度非零；不再需要训练专用 epsilon。 |
| Stage 2 绕过 prepared model 的 forward | trainer 调用 self.model(sample)，forward 内构造 prompt 和 action loss | 正常 forward 入口已接通；真实小模型单进程验证通过，多 rank 同步尚未实测。 |
| 在线每次重编码全部 RGB 历史 | 只编码新边界，保留 CPU latent 历史 | 在线 VAE 重复编码问题已修复。CTE 本身仍重算完整 latent prefix，可先保留。 |
| 空 PIM 仍可产生 memory 增量 | 整个 adapter 残差乘 has_pim | 空 PIM 对动作路径的增量为零；准确含义见第 4 节。 |
| 可能混入其他 FastWAM 变体 | 没有新增该类修改 | Joint、IDM、Optional-IDM、MoT、ActionDiT 文件与基础 FastWAM 上游逐字节一致。 |

主要实现位置：[Stage 1 采样器](https://github.com/xiluoxiluo/ICL-WAM/blob/7df6c8546f7c6331bee8b8e2044b93bbe77ffe7c/src/fastwam/zeva/stage1_sampling.py)、[v4 cache 构建](https://github.com/xiluoxiluo/ICL-WAM/blob/7df6c8546f7c6331bee8b8e2044b93bbe77ffe7c/scripts/build_zeva_robotwin_cache.py)、[在线生命周期](https://github.com/xiluoxiluo/ICL-WAM/blob/7df6c8546f7c6331bee8b8e2044b93bbe77ffe7c/src/fastwam/zeva/lifecycle.py)、[模型 forward 路由](https://github.com/xiluoxiluo/ICL-WAM/blob/7df6c8546f7c6331bee8b8e2044b93bbe77ffe7c/src/fastwam/models/wan22/fastwam.py#L1541-L1597)。

CTE 网络和损失本轮未修改；上一轮与 Zeva 的同权重数值对照结论仍适用于其核心计算。四个 prompt token 与 ActionDiT hidden 残差属于当前 FastWAM 接口适配，本轮没有把这种接口差异单独列为实现错误。

## 2. P1：两个索引构建函数会在扫描结束时失败

位置：[stage1_sampling.py 第 65 行和第 119 行](https://github.com/xiluoxiluo/ICL-WAM/blob/7df6c8546f7c6331bee8b8e2044b93bbe77ffe7c/src/fastwam/zeva/stage1_sampling.py#L63-L121)。

两处使用：

```python
for dataset_index, sample in enumerate(dataset):
    ...
```

这里的 Dataset 使用按下标读取的 Python 迭代协议，扫描结束依赖 dataset[len(dataset)] 抛出 IndexError；它不会自动按 __len__ 停止。

实际调用链是：

1. 索引器请求下标 N，其中 N=len(dataset)。
2. BaseLerobotDataset 正确抛出 IndexError。
3. 外层 RobotVideoDataset.__getitem__ 捕获所有 Exception，把这次正常的越界终止也当作数据错误，改读随机合法下标 j。
4. ZevaRobotWinDataset 发现请求 N、返回 j，抛出 RuntimeError。
5. 因为整个索引要在训练或缓存前建完，Stage 1 与 cache 构建都无法正常结束预扫描。

对应源码：[底层越界检查](https://github.com/xiluoxiluo/ICL-WAM/blob/7df6c8546f7c6331bee8b8e2044b93bbe77ffe7c/src/fastwam/datasets/lerobot/base_lerobot_dataset.py#L192-L194)、[外层通用异常重试](https://github.com/xiluoxiluo/ICL-WAM/blob/7df6c8546f7c6331bee8b8e2044b93bbe77ffe7c/src/fastwam/datasets/lerobot/robot_video_dataset.py#L314-L323)、[Zeva 索引检查](https://github.com/xiluoxiluo/ICL-WAM/blob/7df6c8546f7c6331bee8b8e2044b93bbe77ffe7c/src/fastwam/datasets/zeva_robotwin_dataset.py#L42-L53)。

**复现证据：**提取并执行仓库中真实 RobotVideoDataset.__getitem__ 方法源码，以合成的 _get 数据源代替可选解码依赖，并把随机重试固定为 0。长度为 65 时，两个索引函数都得到：

```text
RuntimeError: Zeva requires deterministic source indexing:
requested 65, underlying dataset returned 0
```

这不是实际数据损坏；是扫描终止与重试机制相互作用。没有运行真实 MP4 解码，但异常控制流使用仓库原方法。

**最小修复：**两个索引构建函数都显式限制下标范围：

```python
for dataset_index in range(len(dataset)):
    sample = dataset[dataset_index]
    ...
```

保留 Zeva 的 requested_index == returned_index 检查。真实坏样本被重试到别处时，仍应报错，不能吞掉 RuntimeError 或放宽一致性校验。也可以在 Zeva wrapper 调用 base_dataset 之前增加下标范围检查，正确抛出 IndexError。

验收只需覆盖两件事：完整扫描 N 条样本不访问 N；合法范围内若底层替换了源下标，仍拒绝该样本。

## 3. P1：缓存脚本将全数据集的已解码窗口同时保留在内存中

位置：[build_zeva_robotwin_cache.py 第 156–158 行](https://github.com/xiluoxiluo/ICL-WAM/blob/7df6c8546f7c6331bee8b8e2044b93bbe77ffe7c/scripts/build_zeva_robotwin_cache.py#L151-L158)。

```python
grouped[index_row.episode_id].append(
    (index_row, dataset[index_row.dataset_index])
)
```

此时保存的是完整 sample，包含 video、action、context 和若干视图；后面才遍历 episode 编码，而且 grouped 始终保留所有 sample 的引用。source_samples 又保存同一批 sample 的引用。因此即使逐个 episode 计算 CTE，也没有逐个 episode 释放解码数据。

默认 float32 视频窗口 [3,9,384,320] 占 **12.65625 MiB**。一万个查询窗口，仅 video 就约 **123.6 GiB**，还没算 T5 context 和其他对象。这是按张量形状计算的占用，不是实际大数据集峰值内存测量。

合成缓存入口中的引用跟踪也确认：两个 episode 共 34 个查询窗口，在第一次 CTE forward 前，34 个解码窗口都仍然存活；第二个 episode 计算时仍全部存活。

**建议保留 full-history 语义，只调整读取方式：**

1. grouped 只保存 CTETrainIndex 或整数下标，不保存解码 sample。
2. 每次处理一个 episode；按 raw_step 去重边界观测与 action group。
3. 新 RGB 边界编码后保留小的 frozen latent，释放窗口 RGB、context 等；source_rows 只保存轻量元数据。
4. VAE 使用有上限的小批次，随后对该 episode 的 latent 历史计算 CTE；按当前 v4 的 raw_step 生成 query/effect。
5. 将记录及时转 CPU 并分 shard 写出，处理完一个 episode 后释放其临时数据。

另一个同路径问题是第 228 行把整段 RGB 历史交给 encode_history；该函数把时间展平到 batch，VAE 的 batch 会随 episode 长度增长。用固定小批次编码可同时解决这个显存风险，且不改变 phase 定义。见 [离线 VAE 调用](https://github.com/xiluoxiluo/ICL-WAM/blob/7df6c8546f7c6331bee8b8e2044b93bbe77ffe7c/scripts/build_zeva_robotwin_cache.py#L216-L236)、[encode_history](https://github.com/xiluoxiluo/ICL-WAM/blob/7df6c8546f7c6331bee8b8e2044b93bbe77ffe7c/src/fastwam/zeva/vae_adapter.py#L146-L153)。

性能优化可以先使用现有元数据计算合法窗口下标，避免在“仅建立索引”时把所有重叠视频窗口解码一遍。当前 Stage 1 的逻辑是在取到 sample 后才跳过重叠，所以预扫描仍可能昂贵。

验收：增加 episode 数量时，已解码 RGB 的驻留数量不随整个数据集线性增长；新缓存与当前小样本缓存的 phase、BIT 数值保持一致。不要为减内存把部署历史重新改成每 32 个 action 清空。

## 4. 行为约定与合理取舍

### 4.1 当前的空 PIM 语义是“整个 memory addon 关闭”

[fastwam.py 第 823–829 行](https://github.com/xiluoxiluo/ICL-WAM/blob/7df6c8546f7c6331bee8b8e2044b93bbe77ffe7c/src/fastwam/models/wan22/fastwam.py#L823-L829)给整个融合残差乘了 has_pim。因而：

- 基础 FastWAM 的文本/当前图像/proprio 条件仍正常工作。
- addon 内 task、phase、BIT、PIM 融合出的整条残差，在 PIM 为空时全部关闭。
- “Task/phase/BIT conditioning remains available when PIM is empty”这句注释不准确；这些量仍被计算，但无法通过该残差影响 action。

**建议当前版本保留这一简洁约定，修正注释和文档即可。**它保证空 PIM 回归基线；无需现在增加第二条 BIT adapter。若未来明确要研究无 PIM 时的 BIT 独立收益，再把它作为单独实验。

正常首个 attempt 的 PIM 初始为空；第一个 16-action effect 完成后才出现条目。默认 24 步重规划下，首次动作 chunk 来自基线，下一次重规划可使用已完成经验。多次 attempt 后若 PIM 非空，起始时即可使用它。不要把空 PIM 样本中的零 addon 梯度误诊为梯度断路。

### 4.2 四个 prompt token 和当前 action adapter 可以保留

当前仍使用文本池化 global context、四个融合/证据 token、32 个 learned query 生成 action hidden 残差。这些与 Zeva 的 task-context bank 和单个 F_mem/prefix 接口不同，但它们是可解释、可验证的适配选择。

本轮实际模型验证证明：该通路可训练，内容改变可影响动作预测，基座保持冻结。**目前没有充分理由为追求张量形状完全一致而重写它。**论文/方法说明中应交代 task context 的简化与 FastWAM 注入接口，避免声称所有条件语义都逐项相同。

如果之后需要简化，可把“单 F_mem + 小 projector”作为一个消融；目前优先排除数据与运行错误。仍不建议加入第三个 expert、Joint、IDM 或新的视频训练目标。

### 4.3 离线 PIM 是训练代理，在线 PIM 来自机器人已执行历史

[MemoryBank.retrieve](https://github.com/xiluoxiluo/ICL-WAM/blob/7df6c8546f7c6331bee8b8e2044b93bbe77ffe7c/src/fastwam/zeva/retrieval.py#L49-L71)仍排除当前 episode，从同任务其他 episode 检索支持；在线生命周期则可以读当前 attempt 中刚完成的交互，并对 PIM 做合并。

这不是未来信息泄漏，其他训练轨迹可以作为支持代理；但目前验证过的是 phase/BIT 的同轨迹对齐，**不能据此说训练与部署的 PIM 内容分布也已一致**。

实用建议：

- 先保留离线代理，记录每批 PIM 有效条数，确认主要样本有支持。
- 若某任务只有一个训练 episode，排除自身后 PIM 恒为空，结合 has_pim 门控，该任务不会直接为 addon 提供动作监督梯度。训练前应汇总这一比例。
- 小规模实验中覆盖少量支持、空支持和错误支持；可参考 Zeva 的 context/support dropout=0.2，作为之后的训练稳健性实验，而非新的架构要求。
- 如成功 demo 代理在真实失败/retry 中效果差，再增加实际失败轨迹，或加入当前 episode 中 end_raw_step <= query_raw_step 的完成记录进行训练模拟。始终排除当前轨迹尚未完成的 effect。

Zeva dropout 配置见 [官方 PIM 配置](https://github.com/air-embodied-brain/Zeva/blob/df25844189715ff5d20f3203b880c1d526a557f7/cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_robocasa365_atomic5_zeva_pim.py#L53-L54)。

### 4.4 任务平衡与多卡的验收边界

采样器在数据不足时会输出普通 leftover batch；因此不能只看配置中的 tasks_per_batch 就假设每批都有足够正负样本。已有实际 batch/positive/task 日志是有用的，继续保留。显存需要缩小 batch 时，同步设置 cte_samples_per_task，例如 batch_size=4、cte_samples_per_task=2 可形成两任务各两样本；仅把 batch_size 降到 2 而保持默认每任务 4 会触发配置异常。

prepared-model forward 已经修正，单进程真实模块通过。多卡发布结论仍需在用户训练环境用两个 rank、不同输入做一次更新后比较 addon 参数一致性。本轮未运行多 rank 或 DeepSpeed 训练。

## 5. 本轮验证证据

### 5.1 既有测试

```text
PyTorch 2.7.1+cpu
PYTHONPATH=src python -m pytest -q
43 passed in 7.33s
```

部分测试使用 stub 或检查源码字符串；43 项通过不覆盖真实数据加载器的越界重试，这也是第 2 节问题没有被现有测试捕获的原因。

### 5.2 实际 Stage 1 与 cache 入口，配合合成数据

调用 main.__wrapped__，替换数据 instantiate 为确定性内存 Dataset，保留本轮训练、采样、CTE、缓存写入和 Stage 2 读取代码。

| 检查 | 结果 |
| --- | --- |
| Stage 1 连续三步实际 batch | 16、16、16 |
| 每批不同 task / episode | 4 / 8 |
| 每批同任务正样本对 | 24 |
| task loss | 2.70894、2.70598、2.71119 |
| raw window 起点 | 仅 0、32、64、96 |
| cache 规模 | 两个 100-frame episode，46 条 query/effect，34 个 Stage 2 样本 |
| episode 0 查询 | 0、4、8、...、64，包含 24、48 |
| phase：缓存完整轨迹 vs 在线逐步 prefix | 最大绝对差 3.05e-7 |
| BIT：缓存 vs 在线已完成 effect 历史 | 最大绝对差 1.19e-7 |
| 两个 index builder + 原 RobotVideoDataset 重试方法 | 均复现 requested 65 / returned 0 的 RuntimeError |

这些是 RGB debug CTE 的合成输入验证，没有验证真实 Wan VAE 的 CUDA 编码误差或真实视频解码。

### 5.3 真实 WanVideoDiT、ActionDiT、MoT 的缩小模型

使用真实类组成两层 video/action 模型，配合合成 Conv3d VAE；调用本轮 model(sample) Stage 2 入口。固定合成样本、FM 随机种子，CPU float32 训练三步，仅更新 addon。

| optimizer step | loss | prompt 梯度范数 | gate 梯度 | 与基线动作 velocity 最大差 |
| --- | --- | --- | --- | --- |
| 1 | 3.020237 | 0 | 0.024668 | 0 |
| 2 | 3.020188 | 5.48e-4 | 0.024505 | 9.02e-4 |
| 3 | 3.019939 | 9.86e-4 | 0.074439 | 1.80e-3 |

进一步检查：

- 三步后 tanh(gate) 约 -0.00581；负 gate 是有符号残差的合法情况。
- 相同模型和 FM 输入，仅改变 PIM effect 内容，动作 velocity 最大差约 8.34e-4。
- 相同模型将 PIM mask 置空，与基线最大差为 0。
- 所有冻结基座参数逐项不变，梯度均为空，合成 VAE 也保持不变。

这里观察的是去噪网络的动作 velocity 输出与梯度，不是执行后的关节轨迹或任务成功率；三步合成 loss 下降也不能代替实际性能验证。

## 6. 建议的下一步

1. 修复两个索引函数的有界遍历，保持异常换样本检查。
2. 把 cache builder 改成仅索引分组、逐 episode 流式处理，固定 VAE 编码 batch 上限。
3. 保留当前网络结构与 has_pim 约定，修正注释；确认 task batch 和 PIM 支持比例满足训练需要。
4. 用修复后的采样重新训练 CTE，再建 v4 cache、训练 addon；不继续使用上轮有问题的 CTE/v3 cache。若现有 CTE 已确认来自本轮正常批次训练，且只是修复 IO/内存实现并保持特征数值等价，不必仅因缓存流式化重新训练 CTE。
5. 固定少量任务和多个 seed，先完成短训练与真实 checkpoint 的 gate-zero/empty-PIM 检查，再做完整评测。

第一轮效果验证保持简单：

| 条件 | 目的 |
| --- | --- |
| base | 原 FastWAM 能力 |
| pim_shadow | 接入生命周期但 gate=0，验证基线一致性 |
| 同一 addon + 正确 PIM | 观察记忆增强效果 |
| 同一 addon + 空 PIM | 验证空记忆回归基线 |
| 同一 addon + 错误 PIM，条数和 mask 与正确 PIM 相同 | 验证收益是否依赖记忆内容 |

比较使用相同环境 seed、指令、采样 seed、去噪步数、replan_steps 和 attempt 预算。正确/错误 PIM 消融保持 task、phase、BIT 及模型权重一致；空 PIM 当前会关闭整条 addon，这是已选定的门控语义。

现有 evaluator 首次成功就结束 retry，因此 max_attempts=4 对应 Success@4。请分开报告首次尝试成功率和 Success@K，所有条件使用相同 K，避免将多次尝试结果直接当作单次策略提升。见 [RoboTwin evaluator](https://github.com/xiluoxiluo/ICL-WAM/blob/7df6c8546f7c6331bee8b8e2044b93bbe77ffe7c/third_party/RoboTwin/script/eval_policy.py#L415-L468)。

**当前决策：架构可以稳定下来。优先完成有界遍历与流式缓存这两处小范围工程修复，再用真实任务检验收益；暂时不继续扩充记忆模块。**

