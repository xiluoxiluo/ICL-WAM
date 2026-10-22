# ICL-WAM 与 Zeva-Ego：重新对齐审计

## 结论与范围

**本轮执行“除了必要的 FastWAM 适配，其余按 Zeva-Ego 设计对齐”。** 基线为 ICL-WAM `4e95b26`，参考固定到 Zeva `feature/zeva_ego` 的 [`bae85ad753d3afe4f2e56762373d48ac0267e101`](https://github.com/air-embodied-brain/Zeva/tree/bae85ad753d3afe4f2e56762373d48ac0267e101)。上一轮列出的额外差异并非都由WAM架构要求；本轮继续修正，没有将它们仅以“适配”为由保留。

只保留FastWAM foundation、video/action注意力接口、输入特征空间、joint14/H32/执行24和原生flow目标形式；训练样本范围、padding权重、EAP/PIM数值语义不再当作必须保留的差异。完整操作见[训练与评测指南](zeva_ego_training_evaluation.md)。对齐公开源码的设计和配方，不代表与不同底座、不同数据源的pi0.5逐位等价，也不代表已经测得任务收益。

## 1. 相对4e95b26继续修正的差异

| 项目 | 4e95b26行为 | 新默认 |
| --- | --- | --- |
| CTE数据范围 | 全部可用任务，没有外部task subset约束 | 必填有序`task_names`文件和真实episode conditions；按指定conditions与task subset筛选，记录文件/源数据指纹。 |
| CTE holdout | 按semantic task，seeded permutation，ceil | 按condition×task分区，SHA256排序、round计数；先划分完整正长度源population，再过滤CTE任务和短episode。独立`cte_split_seed=1000`。 |
| Paired sampler | 按任务排序遍历 | 遵循外部task_names顺序（含空任务槽）；有样本组按此顺序分配shuffle随机流。 |
| CTE选择 | 允许≥80epoch | 默认精确epoch80；显式探索仅epoch40/step26160。均须正式可发布架构、记录的validation.stage1_gate=true，并重新核验当前epoch三项gain。旧split checkpoint不能改标签升级。 |
| CTE导出 | 未显式关闭TF32；只写decision_valid行 | 参考导出关闭matmul/cuDNN TF32；boundary v2保存全部CTE frames，包括末边界。完整source scope传到下游。 |
| Stage2 query与padding | 仅完整H32监督窗口；EAP/flow丢弃action_valid=False目标 | PIM v4使用全部CTE frames；沿原生LeRobot末动作重复填充H32，EAP NLL和action-flow按完整horizon计算。CTE Stage1仍只监督完整H24段。 |
| 训练语言查询 | 整episode固定首指令 | `training_query_scope=per_decision`，按当前样本原始指令检索；相同输入可离线去重，不跨不同指令误复用。推理仍缓存attempt首条指令。 |
| EAP dropout/attention | `rand>=0.4`；MHA不返回weights | 参考`rand<0.6`；MHA使用参考的`need_weights=True`调用。 |
| EAP NLL | 目标先转WAM dtype；mask尾部 | 保留原FP32 normalized action，坐标求和、完整horizon平均、系数0.01。 |
| PIM/部署精度 | Q/K强制FP32后归一化；整个addon跟随WAM BF16 | 训练按AMP计算dtype归一化；EAP/PIM保持FP32权重和推理运算，在注入WAM时才转换到其dtype。 |
| PIM容量/提交/设备 | 3 attempts、192条，end_attempt提交、CPU存储 | 4 attempts、每attempt64/总256；下次attempt reset提交；历史保持policy device。最后成功attempt尚未reset时留pending，避免统计假称已提交。 |
| 历史配对数值与tie | NumPy余弦，episode ID打破平局 | Torch float32 normalize/matmul；first argmax/argmin遵循train source原序。 |
| 评测模型噪声 | 私有CPU连续Torch流 | 默认policy device连续slot流，设备、protocol及state参与handoff和审计；CPU旧协议只用于显式复现。 |

新合同为`numerical_contract=zeva_ego_v1`、boundary v2、PIM cache v4、per-query context v2、PIM memory contract v2。旧产物在对应legacy配置下仍可重评，不允许缺失元数据后静默进入新默认。

### 已完成并保留的上一轮修正

| 项目 | 此前 ICL-WAM | Zeva-Ego参考与本轮处理 |
| --- | --- | --- |
| Stage2A参数范围 | 冻结foundation，仅训练EAP | 参考共同训练foundation/EAP；新默认`ego_joint_parent`，仍提供同配方frozen控制组。 |
| Stage2A优化配方 | LR1e-4、5% warmup、cosine；默认有效batch未按256配置 | 改为foundation5e-6、EAP5e-5、5000 updates、global batch256、恒定LR。新16GPU task为batch1×accum16×world16。 |
| joint A→frozen B继承 | A原来不训练foundation，保存/加载判断主要依赖B的joint标志 | 区分“本阶段是否更新”和“是否必须携带已训练权重”；frozen B继承joint A后仍导出并加载配对foundation，不能退回base。 |
| 训练静态检索 | 从task mean query及support中额外排除当前train episode（LOO） | 参考始终使用固定完整train bank；新默认`train_query_exclusion=none`，validation仍不能进入bank。LOO成为显式旧版选项。 |
| PIM训练partner | 每类保留多个候选，再按query window选择；同episode不同边界可能使用不同partner | 新默认按初始phase余弦固定同条件最近、同条件最远、异条件最近各一条；当前query episode排除，只从train histories中选。 |
| PIM课程可用性 | 缺conditions/cross候选时自动删除该模式 | 新完整参考配方要求真实conditions且每query的same/cross均可用；缩减课程须在builder与训练配置中显式启用并记录。 |
| 在线PIM压缩 | uniform | 参考**发布版加载器**默认similarity_merge；新默认对齐attempt内merge→每attempt最多64条→跨attempt merge，阈值0.95、phase/effect各0.5。 |
| uniform索引 | `linspace(...).long()`截断 | 参考先`.round()`；新在线cap与离线训练均使用reference_round。训练历史仍是uniform，不误换成部署的similarity merge。 |
| 新模块初始化 | global/effect投影及PIM Q/K/V额外Xavier | 新predictive配置保留参考`nn.Linear`默认初始化；旧`legacy_xavier`显式保留。 |
| boundary来源验证 | 验证split/ID，但未比较Stage1保存的episode metadata指纹 | 新导出在RGB解码前核对长度、offset、semantic task等metadata，防同ID内容漂移。 |
| 部署语义审计 | 历史压缩/索引未形成完整checkpoint合同 | 新checkpoint与评测计划记录PIM合同；只有明确compression-only消融可覆盖压缩，不能静默更换round/floor或容量。 |

主要参考证据：

- [Stage2A训练器](https://github.com/air-embodied-brain/Zeva/blob/bae85ad753d3afe4f2e56762373d48ac0267e101/scripts/robotwin/train_cte_eap.py)与[父policy构建](https://github.com/air-embodied-brain/Zeva/blob/bae85ad753d3afe4f2e56762373d48ac0267e101/src/openpi/zeva/cte_eap_policy.py)：foundation/EAP optimizer groups、全train bank的TaskLanguageMemory。
- [PIM训练配对构建](https://github.com/air-embodied-brain/Zeva/blob/bae85ad753d3afe4f2e56762373d48ac0267e101/scripts/robotwin/build_cross_attempt_pim.py)与[Stage2B训练器](https://github.com/air-embodied-brain/Zeva/blob/bae85ad753d3afe4f2e56762373d48ac0267e101/scripts/robotwin/train_cross_attempt_pim.py)：每episode固定三类partner、训练uniform采样、500步warmup及课程。
- [发布版PIM配置](https://github.com/air-embodied-brain/Zeva/blob/bae85ad753d3afe4f2e56762373d48ac0267e101/src/openpi/zeva/pim_release.py)与[PIM实现](https://github.com/air-embodied-brain/Zeva/blob/bae85ad753d3afe4f2e56762373d48ac0267e101/src/openpi/zeva/pim_policy.py)：在线similarity_merge、两级压缩、计数加权centroid、reader初始化。
- [CTE/EAP实现](https://github.com/air-embodied-brain/Zeva/blob/bae85ad753d3afe4f2e56762373d48ac0267e101/src/openpi/zeva/cte_eap.py)：预测effect、EAP prior、prefix和residual初始化。

这些修正解决的是“实验实际训练/读取了什么”的问题。它们不能直接证明此前无收益一定由某一项引起，也不能在未重新训练和仿真时宣称成功率已经提高。

## 2. Context怎样影响action：算法结构已对齐，执行主干仍不同

令当前CTE phase为`z_t`，当前预测effect为`e_hat_t`；此前已结束attempt提供`(z_i,e_hat_i)`。两者采用相同的PIM读取结构：

```text
c_t = sum_i softmax_i(cos(Wq z_t, Wk z_i)) * Wv e_hat_i
g'_t = language_retrieved_global + U(c_t)
prefix = [G(g'_t), E(e_hat_t), P(c_t)]
(mean_t, std_t) = EAP(global=g'_t, phase=z_t)
action_embedding += scale * Project(mean_t)
```

这保留三个独立prefix。`U(c_t)`改变global，并继续影响EAP；`P(c_t)`提供独立PIM读取入口。训练prior residual采用0.6保留概率，推理scale=0.5。当前predictive路径不通过旧标量`pim_gate`缩放，因此旧gate的大小不再是有效性指标。

FastWAM的video与action是不同expert，但通过mixed attention和video K/V cache相连，并非完全互不通信。独立prefix先进入FastWAM可处理的video侧token空间，再由action mixed attention读取；另一路EAP residual直接进入action hidden space。prefix/干净首帧不读取未来video或action。该适配保持了可学习的context→action路径，没有把三个token重新相加成一个token。

对应实现：[causal_prompt.py](../src/fastwam/zeva/causal_prompt.py)、[behavior_prefix_adapter.py](../src/fastwam/zeva/behavior_prefix_adapter.py)、[fastwam.py](../src/fastwam/models/wan22/fastwam.py)。它与pi0.5的prefix/suffix接口、位置编码和注意力执行不同；“路径存在且有梯度”与“真实任务收益”是两个需要分别验证的结论。

## 3. 已核对一致的核心设计

### CTE与task memory

三路独立RGB224，共享ResNet18后融合；三流4层因果Mamba；输入当前观测与上一段真实已执行动作，首次使用SOS。当前段动作与下一边界图像仅作为监督，不作为当前query输入。EMA视觉目标、EMA0.99及固定BN统计、phase/global/effect的职责保持参考结构。

CTE五项损失及默认权重为`0.1 action + 0.2 vision + 2 global + 1 phase + 0.2 effect`；80个完整epoch、batch8、同任务成对无放回、vision LR1e-5/其他1e-4、weight decay1e-4、clip1。global对比项不把self当正例，但保留self于softmax分母。bank仅由train episodes的**完整执行段起点**形成：先对raw task-head/phase求轨迹均值，再L2 normalize，不先逐时刻归一化。

独立CPU差分检查在全有效及不均匀padding样例上比较了reference与本地五项loss及总loss，最大绝对差为0；3个seed×4个epoch的paired sampler顺序一致。这是有限样例的公式/采样验证，不是CUDA Mamba或完整训练逐位等价证明。参考见[CTE训练器](https://github.com/air-embodied-brain/Zeva/blob/bae85ad753d3afe4f2e56762373d48ac0267e101/scripts/robotwin/train_cte.py)与[artifact导出](https://github.com/air-embodied-brain/Zeva/blob/bae85ad753d3afe4f2e56762373d48ac0267e101/scripts/robotwin/export_cte_artifacts.py)。

### EAP、PIM与Stage2B

EAP使用global→8个waypoints，当前raw phase作为attention query，预测完整policy horizon的Gaussian；辅助NLL系数0.01。PIM使用phase查询历史phase、加权读取预测effect；当前attempt只积累pending trace，在下次attempt reset才提交已结束trace，retry重置CTE并读取该历史，新episode清空PIM。当前query不读取自身pending。第一次重规划已有当前预测effect，不等待completed-effect窗口。

Stage2B默认2000个optimizer updates，前500步只更新PIM；后1500步joint的foundation/EAP/PIM学习率分别为1e-6/1e-5/5e-5。课程按全局optimizer step固定循环，warmup为matched×5/far×2/cross×1，之后加empty×2。新完整配方会检查数据确实能够提供这些模式。frozen对照保留相同输入、EAP/PIM和课程，仅不更新foundation。

### RoboTwin评测协议

冻结10任务×20场景manifest；最多4次attempt，成功即停；retry保持同scene seed与instruction，分母固定200；采用8个连续逻辑RNG slot，初始动作噪声在policy device生成，并审计reset、artifact身份、数值合同、RNG设备/延续与视频。`new_off/new_on`始终复用同一Stage2B bundle，off只关闭PIM路径，仍保留CTE/EAP与该bundle的foundation。

完整run的连续slot协议相同，但不同条件成功时间不同，会改变后续噪声offset。严格复用同一初始动作噪声的是**单一重规划点的paired诊断**；不能把整轮on/off描述为每个动作都用了相同噪声。参考[评测指南](https://github.com/air-embodied-brain/Zeva/blob/bae85ad753d3afe4f2e56762373d48ac0267e101/scripts/robotwin/README_EVALUATION.md)，本项目入口为[manifest runner](../scripts/eval_zeva_robotwin_manifest.py)。

## 4. 仍不能宣称相同的部分

| 范围 | 实际边界及含义 |
| --- | --- |
| Foundation与注入 | Zeva使用pi0.5，ICL-WAM使用video/action experts、FastWAM mixed attention与自己的flow目标；参数量、预训练、位置编码、条件输入不同。 |
| 动作与时间尺度 | reference relative EEF16、H50/执行15；本项目joint14、H32/执行24。不能只修改维度或沿用EEF归一化统计来追求表面一致。 |
| Policy视觉输入 | reference pi0.5输入；本项目三相机融合384×320再经Wan VAE。两边CTE仍使用三张独立RGB224，CTE与policy视觉通道须分清。 |
| 检索语言特征 | 相同的冻结语言分类器→task-scoped检索机制；reference Gemma2048与本地Wan4096输入embedding空间不同，classifier权重不可互换。 |
| 分类器训练 | reference公开consumer/MLP并从handoff读取既有分类器，没有完整独立训练配方；本项目2000步CE与内部episode划分是本地重建方案，不能称为复刻原训练。 |
| 实际数据成员 | CTE task-subset、conditions和SHA/round划分算法已对齐；实际任务名、episode顺序与源数据必须由真实外部manifest决定。没有原训练的完整数据清单，不能声称训练成员逐条相同或猜测为10/50任务。空source episode无法可靠重建分区位置时，严格路径明确拒绝。 |
| 硬件与随机流 | policy-device连续slot机制已对齐；FastWAM噪声为[1,32,14]，底座、张量形状、硬件/kernel及模型server adapter不同，不能宣称与pi0.5的CUDA逐元素相同。 |
| 运行与硬件 | ZeRO-1、FP32训练参数、CPU完整文本编码器/LRU缓存、严格checkpoint及评测审计是本地工程扩展；不代表已有16GPU/5090实测。 |

reference数据范围与split依据：[train_cte.py](https://github.com/air-embodied-brain/Zeva/blob/bae85ad753d3afe4f2e56762373d48ac0267e101/scripts/robotwin/train_cte.py)、[robotwin_dataset.py](https://github.com/air-embodied-brain/Zeva/blob/bae85ad753d3afe4f2e56762373d48ac0267e101/src/openpi/zeva/robotwin_dataset.py)与[数据文档](https://github.com/air-embodied-brain/Zeva/blob/bae85ad753d3afe4f2e56762373d48ac0267e101/scripts/robotwin/README_DATASET.md)。源码省略conditions时默认Clean，公开recipe明确写Clean+Randomized；本项目要求提供真实conditions，不推断或伪造。

EAP/PIM的上述数值差异本轮已经修正，不能再作为未处理的“WAM适配”列出。新数值合同显式区分training AMP与inference FP32，包括训练中暂时冻结/处于eval模式的EAP，不因子模块模式误判而关闭训练AMP。不同模型的模块构造数量、顺序和张量尺寸仍会消耗不同随机数；这不意味着初始权重或整次训练轨迹可逐位对应。

## 5. 迁移与实验次序

| 已有产物 | 本轮如何处理 |
| --- | --- |
| FastWAM原base、原数据、joint统计 | 继续使用并核验身份。 |
| 4e95及更早CTE | 新默认改变数据范围、split/sampler与readiness来源合同；按新配方重训。旧CTE保留legacy复现，不补标签冒充新来源。 |
| 旧boundary、bank、features、classifier | 因新CTE权重/split身份改变，重新导出/训练；独立冻结语言encoder若资产与tokenizer完全相同可以复用。 |
| 旧static contexts | 重建per-decision v2；不只是把LOO改none或修改metadata。 |
| 旧PIM v2/v3 cache | 重建v4，全部CTE帧、Torch配对/source-order tie、完整真实conditions课程。 |
| 旧Stage2A/B | 保留用于旧版对照；新Stage2A/B使用新目录重新训练，不能将旧权重改名为新recipe。 |
| 完整optimizer/resume state | 仅用于相同配方精确续训；foundation模式、schedule、缓存合同变化属于新实验。 |

建议先完成一个新joint Stage2A，再让joint与frozen Stage2B从**同一个完整父bundle**出发；分别报告同一B checkpoint的PIM off/on，而不是将不同foundation的差异全部归于PIM。在线压缩可另用同一checkpoint的显式`--new-pim-compression uniform`做单因素消融。

重评`4e95b26`使用`robotwin_zeva_ego_pim_4e95_3cam_384`；`b8b6e84`语言版使用`robotwin_zeva_ego_pim_language_legacy_3cam_384`；更早的image+text版仍使用独立static任务。旧版评测显式选择`--model-rng-protocol torch_continuous_slot_v1`，保留CPU噪声协议。旧产物必须保持其base/CTE/bank/classifier/encoder身份。旧alias用于重评；精确训练续跑应使用原commit。

## 6. 验证与效果边界

回归覆盖真实小型WanVideoDiT/ActionDiT/MoT的context/action梯度、joint/frozen训练范围、恒定LR、精确resume、父foundation继承与sidecar身份、FP32训练及BF16部署转换；还覆盖检索全train/LOO隔离、PIM两级压缩、配对、数据来源拒绝及评测审计。独立CPU对照在72次history commit中比较了reference和本地在线memory，覆盖两种压缩、不同容量、空attempt与淘汰，结果逐元素一致。

新bundle使用显式`foundation_contract_schema=iclwam_foundation_contract_v1`。独立复查还覆盖了两项容易漏掉的异常路径：冻结继承bundle的foundation字段被整体删除，不能再被识别为旧base-only addon；已加载训练后foundation的model若尝试用dit-only或partial base覆盖，必须在复制任何参数前拒绝，不能清掉身份却遗留旧action/proprio。旧格式兼容限制在明确的历史合同，完整base严格恢复后才可清除继承身份。

测试命令与整套计数见[指南第8节](zeva_ego_training_evaluation.md#8-已验证范围与结果解释)。这些检查用于证明实现和合同，不替代真实GPU训练与仿真。**本次未执行80epoch训练、16GPU NCCL、完整模型5090推理或200episode RoboTwin，未产生新的成功率、显存峰值或延迟结论。**

最终应分别判断Stage2A的基础任务能力、同一Stage2B中PIM对第2–4次累计成功率的贡献，以及相对旧版本的收益。若context已改变动作但成功率不升，应继续检查CTE预测质量、训练history与真实失败轨迹的差异、检索和配对相关性；仅增加注入幅度不足以定位原因。
