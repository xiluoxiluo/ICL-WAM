# ICL-WAM 与 Zeva-Ego：重新对齐审计

## 结论与范围

**不能将此前的实现概括为“只有 token 注入不同，其余完全一致”。** 本次以 ICL-WAM `b8b6e84` 为基线，逐项核对 Zeva `feature/zeva_ego` 的固定提交 [`bae85ad753d3afe4f2e56762373d48ac0267e101`](https://github.com/air-embodied-brain/Zeva/tree/bae85ad753d3afe4f2e56762373d48ac0267e101)。不仅检查README，也检查实际训练、artifact构建、发布版加载和评测调用链。

本次修正不必要的配方与算法偏差；保留FastWAM的原生foundation、动作与数值执行合同。新默认更接近参考设计，但不等于重新实现一个与pi0.5逐位等价的模型。全部可执行命令、产物迁移和评测说明见[训练与评测指南](zeva_ego_training_evaluation.md)。

## 1. 本次查出并修正的差异

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

EAP使用global→8个waypoints，当前raw phase作为attention query，预测完整policy horizon的Gaussian；辅助NLL系数0.01。PIM使用phase查询历史phase、加权读取预测effect；当前attempt只积累pending trace，在结束后才提交历史，retry重置CTE但保留已结束历史，新episode清空PIM。第一次重规划已有当前预测effect，不等待completed-effect窗口。

Stage2B默认2000个optimizer updates，前500步只更新PIM；后1500步joint的foundation/EAP/PIM学习率分别为1e-6/1e-5/5e-5。课程按全局optimizer step固定循环，warmup为matched×5/far×2/cross×1，之后加empty×2。新完整配方会检查数据确实能够提供这些模式。frozen对照保留相同输入、EAP/PIM和课程，仅不更新foundation。

### RoboTwin评测协议

冻结10任务×20场景manifest；最多4次attempt，成功即停；retry保持同scene seed与instruction，分母固定200；采用8个连续逻辑RNG slot，并审计reset、artifact身份、RNG延续与视频。`new_off/new_on`始终复用同一Stage2B bundle，off只关闭PIM路径，仍保留CTE/EAP与该bundle的foundation。

完整run的连续slot协议相同，但不同条件成功时间不同，会改变后续噪声offset。严格复用同一初始动作噪声的是**单一重规划点的paired诊断**；不能把整轮on/off描述为每个动作都用了相同噪声。参考[评测指南](https://github.com/air-embodied-brain/Zeva/blob/bae85ad753d3afe4f2e56762373d48ac0267e101/scripts/robotwin/README_EVALUATION.md)，本项目入口为[manifest runner](../scripts/eval_zeva_robotwin_manifest.py)。

## 4. 仍不能宣称相同的部分

| 范围 | 实际边界及含义 |
| --- | --- |
| Foundation与注入 | Zeva使用pi0.5，ICL-WAM使用video/action experts、FastWAM mixed attention与自己的flow目标；参数量、预训练、位置编码、条件输入不同。 |
| 动作与时间尺度 | reference relative EEF16、H50/执行15；本项目joint14、H32/执行24。不能只修改维度或沿用EEF归一化统计来追求表面一致。 |
| Policy视觉输入 | reference pi0.5输入；本项目三相机融合384×320再经Wan VAE。两边CTE仍使用三张独立RGB224，CTE与policy视觉通道须分清。 |
| 检索语言特征 | 相同的冻结语言分类器→task-scoped检索机制；reference Gemma2048与本地Wan4096输入embedding空间不同，classifier权重不可互换。 |
| 分类器训练 | reference公开consumer/MLP并从handoff读取既有分类器，没有完整独立训练配方；本项目2000步CE与内部episode划分是本地重建方案，不能称为复刻原训练。 |
| CTE训练任务范围 | reference使用外部必填task-subset JSON和adapter conditions；本项目使用配置dataset中全部可用episodes。没有双方实际manifests，不能断言均为10或50任务，亦不能把50任务classifier注释当CTE训练范围。 |
| Validation成员 | reference按condition×task分组、SHA排序并round；本项目按semantic task、seeded permutation并ceil，保证至少2 train+1 val。可导入明确episode split，但默认成员不保证相同。 |
| Policy监督窗口 | reference遍历CTE cache中仍位于episode内的query frames；本地prepared训练要求完整H32动作监督，尾部不足32步不作为policy训练query。历史可保留真实决策的预测trace，但无后续动作的纯末观测不作为decision；reference cache可保留导出的末frame。训练样本集合不能宣称逐条相同。 |
| Readiness | reference发布选择epoch80，另有显式探索例外；本项目允许≥80，要求该epoch验证、重新核验gain和来源，是更严格的工程合同。 |
| 数值细节 | AMP、native flow/action target dtype、kernel、TF32配置、噪声维度与设备算法不同。本项目EAP NLL target沿用native action dtype路径；不声称逐位GPU复现。 |
| 数据指令 | 当前static context要求同episode原始instruction固定；reference动态consumer更一般。切换instruction的数据必须先明确其训练合同。 |
| 历史容量与提交时点 | 本项目最多3个已结束attempt，在4次尝试评测中可读取的历史数与参考等价；不能外推到任意5次以上协议。本地在end_attempt提交，reference通常在下次reset提交，下一次action可见历史一致，最终快照/计数未必相同。 |
| 运行与硬件 | ZeRO-1、FP32训练参数、CPU完整文本编码器/LRU缓存、严格checkpoint及评测审计是本地工程扩展；不代表已有16GPU/5090实测。 |

reference数据范围与split依据：[train_cte.py](https://github.com/air-embodied-brain/Zeva/blob/bae85ad753d3afe4f2e56762373d48ac0267e101/scripts/robotwin/train_cte.py)、[robotwin_dataset.py](https://github.com/air-embodied-brain/Zeva/blob/bae85ad753d3afe4f2e56762373d48ac0267e101/src/openpi/zeva/robotwin_dataset.py)与[数据文档](https://github.com/air-embodied-brain/Zeva/blob/bae85ad753d3afe4f2e56762373d48ac0267e101/scripts/robotwin/README_DATASET.md)。源码省略conditions时默认Clean，公开recipe明确写Clean+Randomized；本项目要求提供真实conditions，不推断或伪造。

“初始化/随机正则对齐”指分布和尺度：模块构造顺序不同，权重样本不相同；参考`rand<0.6`与本地`rand>=0.4`具有相同保留分布，但同seed不产生同一mask。prior attention是否返回weights可能选择不同kernel；本地BF16部署还会量化EAP/PIM，而参考模块默认FP32。本地NLL使用`action_valid`排除padding，reference对完整H50平均；完整有效目标上的公式对应，尾部padding下的权重不相同。

## 5. 迁移与实验次序

| 已有产物 | 本轮如何处理 |
| --- | --- |
| FastWAM原base、原数据、joint统计 | 继续使用并核验身份。 |
| 来源与readiness匹配的预测CTE、boundary | 可复用；本轮不改变CTE架构/损失或Stage1源码合同。新的boundary导出会额外核验episode metadata。 |
| 同一CTE/数据来源的language encoder、features、train bank、classifier | 可复用；无需仅因改LOO就重训classifier。 |
| 旧static contexts | 按`train_query_exclusion=none`重新构建；不要只修改旧metadata。 |
| 旧PIM v2/cache | 用真实conditions重新构建v3，reference-nearest配对、reference-round索引及完整课程身份。 |
| 旧Stage2A/B | 保留用于旧版对照；新Stage2A/B使用新目录重新训练，不能将旧权重改名为新recipe。 |
| 完整optimizer/resume state | 仅用于相同配方精确续训；foundation模式、schedule、缓存合同变化属于新实验。 |

建议先完成一个新joint Stage2A，再让joint与frozen Stage2B从**同一个完整父bundle**出发；分别报告同一B checkpoint的PIM off/on，而不是将不同foundation的差异全部归于PIM。在线压缩可另用同一checkpoint的显式`--new-pim-compression uniform`做单因素消融。

重评`b8b6e84`语言版使用`robotwin_zeva_ego_pim_language_legacy_3cam_384`；更早的image+text版仍使用独立static任务。旧产物必须保持其base/CTE/bank/classifier/encoder身份。两条路径不能相互冒充。

## 6. 验证与效果边界

回归覆盖真实小型WanVideoDiT/ActionDiT/MoT的context/action梯度、joint/frozen训练范围、恒定LR、精确resume、父foundation继承与sidecar身份、FP32训练及BF16部署转换；还覆盖检索全train/LOO隔离、PIM两级压缩、配对、数据来源拒绝及评测审计。独立CPU对照在72次history commit中比较了reference和本地在线memory，覆盖两种压缩、不同容量、空attempt与淘汰，结果逐元素一致。

新bundle使用显式`foundation_contract_schema=iclwam_foundation_contract_v1`。独立复查还覆盖了两项容易漏掉的异常路径：冻结继承bundle的foundation字段被整体删除，不能再被识别为旧base-only addon；已加载训练后foundation的model若尝试用dit-only或partial base覆盖，必须在复制任何参数前拒绝，不能清掉身份却遗留旧action/proprio。旧格式兼容限制在明确的历史合同，完整base严格恢复后才可清除继承身份。

测试命令与整套计数见[指南第8节](zeva_ego_training_evaluation.md#8-已验证范围与结果解释)。这些检查用于证明实现和合同，不替代真实GPU训练与仿真。**本次未执行80epoch训练、16GPU NCCL、完整模型5090推理或200episode RoboTwin，未产生新的成功率、显存峰值或延迟结论。**

最终应分别判断Stage2A的基础任务能力、同一Stage2B中PIM对第2–4次累计成功率的贡献，以及相对旧版本的收益。若context已改变动作但成功率不升，应继续检查CTE预测质量、训练history与真实失败轨迹的差异、检索和配对相关性；仅增加注入幅度不足以定位原因。
