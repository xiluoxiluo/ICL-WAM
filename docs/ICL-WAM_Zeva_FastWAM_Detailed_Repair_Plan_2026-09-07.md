# ICL-WAM：Zeva → Frozen FastWAM 详细修复实施方案

> 基线核对日期：2026-09-07  
> ICL-WAM：`d6721b2a2221111011a3b9a294925f0d1eadf4b3`  
> Zeva：`df25844189715ff5d20f3203b880c1d526a557f7`  
> FastWAM：`7faa71108368fbb3b6885649f112af607427a2d4`

## 0. 修复目标

目标不是机械复制 Cosmos 的 tensor 形状，而是：

1. 保留 Zeva 的 CTE、BIT/PIM、phase-aware retrieval、Causal Prompt 和 gated residual conditioning；
2. 仅把 Zeva 的 frozen Cosmos Policy 替换为 frozen Base FastWAM；
3. 允许最后一层策略接口变成 **FastWAM-specific action-hidden adapter**；
4. 不使用 FastWAM-Joint、IDM、Optional-IDM；
5. 不新增第三个 memory expert；
6. 不改变 CTE architecture；
7. 不加入未经验证的复杂 auxiliary loss。

最终结构：

```text
RGB / executed action history
        ↓
       CTE
        ↓
 current causal phase
        ↓
   ┌────┴─────┐
   ↓          ↓
  BIT        PIM
   └────┬─────┘
        ↓
 phase retrieval
        ↓
 Causal Prompt / F_mem
        ↓
FastWAM-specific Adapter
        ↓
tanh(g) × memory residual
        ↓
Frozen FastWAM Action Hidden
        ↓
Frozen Action MoT
        ↓
Action
```

---

# 1. 修复优先级

必须按顺序执行：

```text
P0-1  Stage 1 CTE 采样与 batch
P0-2  Offline / Online CTE history 统一
P1-1  PIM 生命周期恢复 Zeva 语义
P1-2  Empty-PIM exact bypass
P1-3  Adapter 初始化恢复 zero-gate / nonzero-projector
P1-4  Stage 2 通过 prepared model.forward 训练
P2    Online VAE latent cache / 日志 / ablation
```

重要：

```text
Stage 1 修复后 → 重新训练 CTE
history/cache 修复后 → 重新生成 cache
PIM/adapter 修复后 → 重新训练 Stage 2 addon
```

旧 CTE、旧 cache、旧 addon 不应混用。

---

# 2. P0-1：修复 Stage 1 CTE 采样与 batch

## 2.1 问题

文件：

```text
scripts/train_zeva_cte.py
```

当前逻辑在同 episode 的下一条 sample 到来时会立刻 flush `pending`，而 `next_episode_step` 又是在 `train_batch()` 后更新。

结果可能变成：

```text
0, 1, 32, 33, 64, 65 ...
```

而正确的 non-overlap 32-action window 应是：

```text
0, 32, 64, 96 ...
```

同时，小 batch 甚至 batch=1 会让：

```text
_task_identity_clustering_loss()
```

找不到同 task positive pair，从而返回 0。

## 2.2 修复原则

必须拆成两个阶段：

```text
合法 window 选择
        ↓
optimizer batch 组织
```

不要边遍历 dataset、边处理 overlap、边 flush optimizer batch。

## 2.3 新增合法索引构建

建议新增：

```python
@dataclass
class CTETrainIndex:
    dataset_index: int
    episode_id: str
    task_id: str | int
    episode_step: int
```

函数：

```python
def build_cte_training_index(
    dataset,
    source_window_actions=32,
    sample_stride=1,
):
    ...
```

每个 episode：

```python
next_valid_step = {}

for dataset_index, sample in enumerate(dataset):
    eid = str(sample["episode"].episode_id)
    step = int(sample["episode"].episode_step)

    expected = next_valid_step.get(eid)
    if expected is not None and step < expected:
        continue

    if not valid_window(sample):
        next_valid_step.pop(eid, None)
        continue

    rows.append(
        CTETrainIndex(
            dataset_index=dataset_index,
            episode_id=eid,
            task_id=sample["episode"].task_id,
            episode_step=step,
        )
    )

    next_valid_step[eid] = step + 32 * sample_stride
```

验收必须得到：

```text
0, 32, 64, ...
```

而不是：

```text
0, 1, 32, 33, ...
```

---

# 3. Stage 1 task-balanced batch

## 3.1 目标

每个 minibatch 至少：

```text
>= 2 个 task
每个 task >= 2 个 sample
```

推荐：

```yaml
batch_size: 16
tasks_per_batch: 4
samples_per_task: 4
```

同 task positive 优先来自不同 episode。

## 3.2 建议 sampler

新增：

```text
src/fastwam/datasets/cte_balanced_sampler.py
```

伪代码：

```python
class TaskBalancedCTEBatchSampler:
    def __iter__(self):
        task_to_rows = group_by_task(self.rows)

        while True:
            tasks = sample_tasks(self.tasks_per_batch)

            batch = []
            for task in tasks:
                selected = sample_prefer_distinct_episode(
                    task_to_rows[task],
                    self.samples_per_task,
                )
                batch.extend(selected)

            yield batch
```

## 3.3 Stage 1 日志

每个 optimizer step 记录：

```text
cte/loss
cte/loss_action
cte/loss_vision
cte/loss_task
cte/loss_phase
cte/loss_effect
cte/actual_batch_size
cte/distinct_task_count
cte/task_positive_anchor_count
cte/task_positive_pair_count
cte/distinct_episode_count
```

debug 模式记录：

```text
dataset_index
episode_id
task_id
episode_step
```

---

# 4. Stage 1 必须新增测试

文件：

```text
tests/zeva/test_cte_stage1_sampling.py
```

## Test A：non-overlap

synthetic episode：

```text
step 0..100
```

断言：

```python
starts == [0, 32, 64, 96]
```

## Test B：task positive

构造：

```text
task A: ep1, ep2
task B: ep3, ep4
```

一个 batch 中：

```text
A >= 2
B >= 2
```

并在非退化 synthetic feature 上：

```python
loss_task > 0
loss_task.backward()
grad_norm > 0
```

## Test C：reproducibility

相同 seed：

```text
training index 相同
batch sequence 相同
```

---

# 5. Stage 1 修复后必须重新训练 CTE

旧 checkpoint 不再作为正式 Stage 2 输入。

验收：

```text
loss_task 不再长期为 0
loss_phase 正常
loss_effect 正常
grad_norm finite
```

---

# 6. P0-2：统一 offline / online CTE history

这是最重要的 semantic bug。

## 6.1 当前 offline

当前 cache writer 对每个 32-action window 独立调用 CTE：

```text
[v_s, v_s+4, ..., v_s+32]
[a_s:s+4, ..., a_s+28:s+32]
        ↓
       CTE
```

因此 s=32 时：

```text
phase_offline(32)
=
CTE(v32, BOS, reset history)
```

## 6.2 当前 online

部署持续追加：

```text
v0
a0:4
v4
...
a28:32
v32
```

所以：

```text
phase_online(32)
=
CTE(full prefix 0..32)
```

两者不是同一 query 定义。

## 6.3 正确 contract

统一：

```text
At raw step s:

frames:
[v0, v4, ..., vs]

executed transitions:
[a0:4, a4:8, ..., a(s-4):s]
```

定义：

```python
phase_at_raw_step(s)
effect_completed_at_raw_step(s)
```

---

# 7. 新增 canonical history API

建议新增：

```text
src/fastwam/zeva/history.py
```

核心：

```python
class CausalCTESequence:
    def append_boundary(self, frame, executed_action_group):
        ...

    def encode(self):
        ...

    def phase_at_raw_step(self, raw_step):
        ...
```

离线和在线必须调用同一语义层。

---

# 8. 显式统一时间索引

不要再混淆：

```text
dataset_index
episode_step
transition_index
effect_index
raw_step
```

统一定义：

```text
transition_steps = 4
effect_window_transitions = 4
```

因此：

```text
boundary raw step:
0,4,8,12,...

effect completion:
16,32,48,64,...
```

所有 cache record 必须显式带 `raw_step`。

---

# 9. Cache schema 升级到 v4

建议：

```text
zeva_fastwam_robotwin_cache_v4
```

manifest：

```json
{
  "schema_version": "zeva_fastwam_robotwin_cache_v4",
  "history_semantics": "full_episode_prefix",
  "transition_steps": 4,
  "effect_window_transitions": 4,
  "query_step_unit": "raw_action_step"
}
```

## 9.1 phase query record

```python
{
    "record_type": "phase_query",
    "episode_id": ...,
    "task_id": ...,
    "raw_step": ...,
    "phase": ...,
    "dataset_index": ...,
    "valid": True,
}
```

## 9.2 effect record

```python
{
    "record_type": "effect",
    "episode_id": ...,
    "task_id": ...,
    "attempt_id": 0,
    "start_raw_step": ...,
    "end_raw_step": ...,
    "phase_at_start": ...,
    "effect_post": ...,
    "valid": True,
}
```

---

# 10. phase record 与 effect record 必须拆开

原因：

部署 replan 可能：

```text
0,24,48,72,...
```

而 effect cadence：

```text
16,32,48,64,...
```

所以 query phase 和 completed effect 不是同一个时钟。

正确规则：

```text
当前 query raw_step = s

只有：
effect.end_raw_step <= s

的 effect 才能进入 BIT/PIM
```

不能使用未来 `phase_post` 或未来 completed effect 给当前 query “补历史”。

---

# 11. Offline cache 构造

推荐每个 episode 一次做 full-history CTE：

```python
output = cte(
    episode_boundary_frames,
    episode_action_groups,
)
```

然后建立：

```text
phase_by_raw_step
effect_by_end_raw_step
```

这样不再每个 32-action window 重置 CTE。

---

# 12. CTE prefix causality test

新增：

```text
tests/zeva/test_cte_prefix_consistency.py
```

同一 trajectory：

```python
full = cte(full_frames, full_actions)
prefix = cte(prefix_frames, prefix_actions)
```

要求：

```python
torch.testing.assert_close(
    full["phase"][:, :prefix_len],
    prefix["phase"],
)
```

同时：

```text
改变未来 image
改变未来 action
```

不能改变当前 `phase`。

---

# 13. Offline / Online parity test

新增：

```text
tests/zeva/test_offline_online_history_parity.py
```

对同一 synthetic trajectory：

```text
offline full-history encoder
vs
online CausalCTEHistory step-by-step
```

在相同 raw step：

```python
phase_offline ~= phase_online
```

BIT 内容和检索 query key 也应一致。

---

# 14. Stage 2 sample 对齐

`ZevaRobotWinDataset` 返回的：

```text
current image
clean 32-action target
proprio
query phase
BIT prefix
PIM retrieval
```

必须共享同一个：

```text
raw_step
```

建议 sample 增加：

```python
sample["raw_step"]
```

并做强校验：

```python
assert image_raw_step == raw_step
assert action_target_start == raw_step
assert proprio_raw_step == raw_step
assert phase_raw_step == raw_step
```

---

# 15. 覆盖真实 replan 位置

不要只训练：

```text
0,32,64,...
```

部署默认 `replan_steps=24` 时至少要覆盖：

```text
0,24,48,72,...
```

更推荐：

```text
cache phase at every valid boundary
Stage 2 sampler 按 replan_steps 选 query
```

---

# 16. P1-1：恢复 Zeva-style PIM 生命周期

## 16.1 当前实现

`observe_completed_effect()`：

```text
BIT.append
_pending.append
```

直到：

```text
end_attempt()
```

才：

```text
_pending → PIM
```

而 query 还：

```python
exclude_attempt_id=self._attempt_id
```

所以当前 attempt 已完成经验无法被当前 attempt 的 PIM retrieval 使用。

## 16.2 目标

completed effect 一旦形成：

```text
1. append BIT
2. immediately append PIM
3. subsequent query can retrieve it
```

即：

```text
PIM = all completed interactions up to current time
```

## 16.3 修改 lifecycle.py

建议：

```python
def observe_completed_effect(...):
    self.bit.append(effect_post, effect_index)

    self.pim.append_completed(
        task_cluster=self.pim.task_cluster,
        phase=phase_at_window_start,
        effect=effect_post,
        attempt_id=self._attempt_id,
        transition_index=...,
        metadata=...,
    )

    self._effect_index += 1
```

`_pending`：

```text
删除
```

或只作为 debug trace，不再负责 commit。

## 16.4 end_attempt

```python
def end_attempt(self):
    self.bit.reset()
```

不要清空 PIM。

## 16.5 reset_attempt

```text
BIT reset
PIM begin_attempt
PIM history preserved
```

## 16.6 query

如果遵循 Zeva：

```python
exclude_attempt_id=None
```

允许检索当前 attempt **已经完成** 的 interaction。

---

# 17. PIM causal visibility test

新增：

```text
tests/zeva/test_pim_causal_visibility.py
```

scenario：

```text
t=0
PIM empty

effect0 在 t=16 完成
→ append PIM

query t>16
→ 可以看到 effect0

effect1 尚未完成
→ 不可见
```

必须严格保证：

```text
future / incomplete effect 不进入 PIM
```

---

# 18. P1-2：Empty-PIM exact bypass

这是判断 “PIM experience 是否真的改善 action” 的关键。

## 18.1 目标语义

```text
PIM empty
→ memory residual exact 0
→ policy exactly returns base FastWAM path
```

## 18.2 当前问题

当前 memory tokens 即使 PIM empty，仍有：

```text
fused task/phase
current phase
```

因此训练后可能：

```text
PIM empty
但 Δh != 0
```

这样：

```text
pim_on > base
```

不能证明 PIM experience 有用。

## 18.3 推荐最小修复

保留 `[B,4,256]` FastWAM adapter 接口，不机械恢复 `[B,256]`。

新增：

```python
has_pim = pim_mask.any(dim=-1)
```

在最终 action residual 上：

```python
gated_residual = (
    torch.tanh(self.pim_gate)
    * residual
    * has_pim[:, None, None].to(residual.dtype)
)
```

即：

```text
PIM empty
→ exact residual 0
```

这比往 attention 里塞一个 zero token 更可靠，因为 zero token 仍可能改变 softmax normalization。

---

# 19. Empty-PIM bypass test

新增：

```text
tests/zeva/test_empty_pim_bypass.py
```

先人为制造：

```text
nonzero trained-like adapter
nonzero gate
```

再令：

```text
pim_mask = all false
```

固定：

```text
obs
context
proprio
noise
timestep
```

要求：

```python
conditioned_action_hidden == base_action_hidden
action_pred_empty_pim == action_pred_base
```

在浮点容差内成立。

---

# 20. P1-3：Adapter 初始化恢复 Zeva 风格

当前：

```text
output projection = 0
gate = 0
train_gate_epsilon = 1e-3
```

建议改成：

```text
output projection = Xavier/nonzero
gate = 0
train_gate_epsilon = 删除
```

文件：

```text
src/fastwam/zeva/behavior_prefix_adapter.py
```

改：

```python
nn.init.xavier_uniform_(self.output.weight)
nn.init.zeros_(self.output.bias)

self.pim_gate = nn.Parameter(torch.tensor(0.0))
```

统一：

```python
def gated(...):
    residual = self.forward(...)
    return torch.tanh(self.pim_gate) * residual
```

不要 train/eval 使用两套 gate 公式。

## 20.1 预期梯度

初始：

```text
residual != 0
gate = 0
final delta = 0
```

所以：

```text
exact base bypass
```

第一步：

```text
gate grad != 0
```

随后 gate 离开 0 后：

```text
prompt / adapter upstream grad 开始增加
```

---

# 21. FastWAM-specific adapter 是否保留

**保留。**

不要为了“完全照搬 Zeva”机械恢复 Cosmos 的 prefix tensor 结构。

需要保持的是：

```text
CTE semantics
BIT/PIM semantics
phase retrieval
Causal Prompt semantics
has_pim gated residual
```

允许改变的是：

```text
Cosmos prefix injection
        ↓
FastWAM ActionDiT hidden residual
```

推荐论文表述：

> We preserve Zeva's causal-memory semantics while replacing its Cosmos-specific policy-prefix injection with a FastWAM-specific action-hidden adapter.

---

# 22. Task context

当前：

```text
text context
→ masked mean
→ adaptive pooling
→ 256
```

Zeva 更接近：

```text
task-context prototype bank
```

或：

```text
initial observation + instruction
→ frozen policy readout
→ task prototype retrieval
```

这一项优先级低于：

```text
Stage1
history
PIM lifecycle
has_pim
DDP
```

建议先保留当前 text-pool 版本作为：

```text
task_context_mode=text_pool_v1
```

后续增加：

```text
task_context_mode=zeva_prototype_v1
```

做单独 ablation。

---

# 23. P1-4：Stage 2 通过 prepared model.forward

当前 trainer 会调用底层特殊方法：

```text
unwrap_model(...)
→ forward_zeva_action_train(...)
```

这会绕过 DDP/DeepSpeed/FSDP prepared model 的正常 forward 生命周期。

## 23.1 FastWAM.forward 路由

修改：

```text
src/fastwam/models/wan22/fastwam.py
```

建议：

```python
def forward(self, sample):
    mode = sample.get("_training_mode", "base")

    if mode == "zeva_stage2":
        return self._forward_zeva_stage2(sample)

    return self.training_loss(sample)
```

## 23.2 Trainer

改：

```python
sample["_training_mode"] = "zeva_stage2"
loss, metrics = self.model(sample)
```

不要再直接调用 unwrapped module 的特殊 forward。

## 23.3 PromptEncoder 也放进 model.forward

完整 chain 应为：

```text
prepared self.model
 ↓
task_tokens_from_context
 ↓
CausalPromptEncoder
 ↓
BehaviorPrefixAdapter
 ↓
Frozen action computation
 ↓
action FM loss
```

---

# 24. 正确冻结与 autograd 边界

继续保留：

```python
self.requires_grad_(False)
self.zeva_prompt_encoder.requires_grad_(True)
self.zeva_behavior_prefix_adapter.requires_grad_(True)
```

但不要：

```python
with torch.no_grad():
    entire action forward
```

正确关系：

```text
FastWAM parameter frozen
≠
FastWAM action computation graph removed
```

必须：

```text
FastWAM grad = None
Prompt/Adapter grad != None
```

Stage 2 不要求：

```text
Action Loss → CTE
```

因为 CTE 是 Stage 1 单独训练并通过 cache/lifecycle 提供特征。

---

# 25. Stage 2 允许 no_grad 的部分

可以：

```text
first-frame VAE
frozen video expert prefill
frozen video KV construction
```

使用 `torch.no_grad()`。

但：

```text
action token
+
memory residual
+
frozen ActionDiT/MoT computation
+
prediction
+
action loss
```

必须保留 autograd graph。

---

# 26. Gradient-flow test

新增：

```text
tests/zeva/test_real_gradient_flow.py
```

使用：

```text
真实 WanVideoDiT
真实 ActionDiT
真实 MoT
缩小层数
synthetic VAE/input
```

检查：

```python
loss.backward()
```

要求：

```text
FastWAM backbone grads are None
BehaviorPrefixAdapter grad_norm > 0
CausalPromptEncoder grad_norm > 0
```

允许 zero-gate 初始化第一步 prompt grad 很小或为 0，但 gate 打开后必须非零。

---

# 27. 多卡验收

真实环境：

```text
2 GPU / 2 rank
```

不同 rank 输入不同 sample。

一个 optimizer step 后：

```text
PromptEncoder params rank0 == rank1
BehaviorPrefixAdapter params rank0 == rank1
```

同时核对：

```text
global batch
gradient accumulation
optimizer step
scheduler step
```

---

# 28. P2：Online CTE VAE latent cache

当前在线 history 每次可能重复编码旧 RGB。

改成：

```text
new boundary RGB
      ↓
encode once
      ↓
store frozen VAE latent
```

`CausalCTEHistory` 内部保存 latent history：

```python
class CausalCTEHistory:
    def reset(self, first_image):
        self.frame_latents = [
            self.frame_encoder(first_image)
        ]

    def append_transition(self, actions, next_image):
        self.actions.append(actions)
        self.frame_latents.append(
            self.frame_encoder(next_image)
        )

    def forward(self):
        return self.cte(
            torch.stack(self.frame_latents),
            torch.stack(self.actions),
        )
```

---

# 29. Schema / checkpoint version

修复后建议：

```text
CTE checkpoint:
zeva_cte_v2

cache:
zeva_fastwam_robotwin_cache_v4

addon:
zeva_fastwam_addon_v2
```

保存：

```json
{
  "history_semantics": "full_episode_prefix",
  "pim_semantics": "append_on_completed_effect",
  "empty_pim_bypass": true,
  "adapter_injection": "fastwam_action_hidden",
  "task_context_mode": "text_pool_v1"
}
```

加载时 strict check，避免旧语义 checkpoint/cache 被误用。

---

# 30. Debug metrics

保留并记录：

```text
train/loss_action
memory/gate

memory/base_action_hidden_norm
memory/memory_delta_hidden_norm
memory/conditioned_action_hidden_norm
memory/memory_residual_ratio

memory/bit_count
memory/pim_count

memory/retrieval_top1_similarity
memory/retrieval_mean_similarity
```

定义：

```text
memory_residual_ratio
=
||delta_hidden||
/
(||base_hidden|| + eps)
```

只做诊断，不硬编码为训练阈值。

---

# 31. 必须新增的 8 个验收测试

1. `test_cte_non_overlap_sampling`
   - 只允许 `0,32,64,...`

2. `test_cte_task_positive_batch`
   - batch 内有 same-task positives

3. `test_cte_prefix_causality`
   - 改未来不影响当前 phase

4. `test_offline_online_phase_parity`
   - 同 raw step offline ≈ online

5. `test_pim_causal_visibility`
   - completed 可见，future/incomplete 不可见

6. `test_empty_pim_bypass`
   - empty PIM → exact base FastWAM

7. `test_memory_sensitivity`
   - 固定 obs/noise/timestep，仅换 PIM，action 应变化

8. `test_gradient_and_frozen_backbone`
   - backbone grad None，addon grad nonzero

---

# 32. RoboTwin 正式训练前 smoke test

只跑：

```text
100~500 optimizer steps
```

观察：

```text
loss_action finite
gate begins moving
memory_delta_hidden_norm nonzero after gate opens
memory_residual_ratio finite
prompt grad valid
adapter grad valid
base params unchanged
```

全部满足后再大规模训练。

---

# 33. 闭环消融

至少：

| 条件 | 目的 |
|---|---|
| `base` | 原始 FastWAM |
| `pim_shadow` | lifecycle 正常但 residual=0 |
| `addon_empty_pim` | 控制普通条件适配影响 |
| `pim_on_correct` | 正确 PIM |
| `pim_on_shuffle` | 检查收益是否依赖正确 memory |
| `pim_on_wrong_task` | diagnostic，可选 |

关键判据：

```text
pim_on_correct
>
addon_empty_pim
```

才支持：

```text
PIM experience 本身改善 action
```

如果只是：

```text
addon_empty_pim > base
```

说明可能只是 task/phase adapter 有作用。

---

# 34. 固定 paired comparison 条件

必须固定：

```text
FastWAM checkpoint
CTE checkpoint
addon checkpoint
environment seed
instruction
policy denoise seed
noise seed
num inference steps
replan_steps
action horizon
max attempts
```

---

# 35. 评测指标

分别报告：

```text
First Attempt SR
Attempt-2 SR
Attempt-3 SR
Attempt-4 SR

Success@1
Success@2
Success@3
Success@4
```

不要只报告：

```text
max_attempts=4 最终成功率
```

---

# 36. Rollout 日志

至少保存：

```json
{
  "task": "...",
  "seed": 0,
  "attempt": 1,
  "raw_step": 48,

  "bit_count": 2,
  "pim_count": 3,

  "retrieval_sources": [],
  "retrieval_scores": [],

  "gate": 0.0,
  "memory_residual_ratio": 0.0,

  "action_base": [],
  "action_conditioned": [],

  "success": false
}
```

---

# 37. 建议 commit 顺序

## Commit 1

```text
fix: rebuild CTE Stage-1 sampling and balanced batching
```

## Commit 2

```text
fix: unify offline and online CTE full-history semantics
```

## Commit 3

```text
fix: restore Zeva-style completed-effect PIM lifecycle
```

## Commit 4

```text
fix: enforce empty-PIM exact policy bypass
```

## Commit 5

```text
refactor: restore zero-gate nonzero-projector initialization
```

## Commit 6

```text
fix: route Zeva Stage-2 through prepared FastWAM forward
```

## Commit 7

```text
perf: cache online CTE frame latents
```

## Commit 8

```text
test: add gradient, memory-sensitivity and rollout diagnostics
```

---

# 38. 当前阶段不要做

不要：

```text
新增 memory expert
把 PIM 变成第三个 MoT expert
使用 FastWAM-Joint
使用 IDM
使用 Optional-IDM
加入 future-video supervision
端到端更新 FastWAM
修改 CTE architecture
加入复杂 retry loss
加入未经消融的 auxiliary loss
```

---

# 39. 最终验收 checklist

## Architecture

- [ ] Base `fastwam.py`
- [ ] no Joint
- [ ] no IDM
- [ ] no Optional-IDM
- [ ] no third expert

## Stage 1

- [ ] source windows = `0,32,64,...`
- [ ] actual batch 正常
- [ ] task positive pairs 存在
- [ ] `loss_task` 不长期为 0
- [ ] CTE 已重新训练

## History

- [ ] offline full-prefix
- [ ] online full-prefix
- [ ] raw_step 显式
- [ ] future 不影响 current phase
- [ ] offline / online parity 通过

## PIM

- [ ] completed effect 立即写入
- [ ] 当前 attempt 已完成 PIM 可读
- [ ] future/incomplete PIM 不可读
- [ ] BIT 每 attempt 清空
- [ ] PIM 跨 attempt 保留

## Conditioning

- [ ] FastWAM action-hidden adapter 保留
- [ ] gate init = 0
- [ ] adapter output nonzero init
- [ ] empty PIM exact bypass
- [ ] correct PIM changes action

## Gradient

- [ ] FastWAM params frozen
- [ ] FastWAM `.grad is None`
- [ ] PromptEncoder grad 有效
- [ ] Adapter grad 有效
- [ ] action path 没被整体 `no_grad`

## Distributed

- [ ] Stage 2 通过 prepared `self.model(...)`
- [ ] 2-rank addon 参数同步
- [ ] global batch 正确

## Evaluation

- [ ] base
- [ ] shadow
- [ ] empty PIM
- [ ] correct PIM
- [ ] shuffled PIM
- [ ] paired seed
- [ ] First Attempt SR
- [ ] Success@K
- [ ] retrieval/action trace 保存

---

# 40. 最终项目定义

完成修复后，建议表述为：

> **ICL-WAM is a Zeva-style causal-memory adaptation of a frozen Base FastWAM. It preserves the causal transition encoder, brief and persistent interaction memory, phase-conditioned retrieval, causal prompt construction, and gated residual conditioning, while replacing Cosmos-specific policy-prefix injection with a FastWAM-specific action-hidden adapter.**

中文：

> **ICL-WAM 保留 Zeva 的因果交互编码、BIT/PIM 双时间尺度记忆、phase-aware retrieval、Causal Prompt 与门控残差条件注入，仅将 Cosmos-specific policy prefix 接口替换为适配 Base FastWAM ActionDiT hidden space 的条件注入接口。**

最终必须同时满足：

```text
offline phase == online phase
future information cannot affect current memory
completed effect becomes retrievable at the correct time
empty PIM → exact base FastWAM
addon receives action-loss gradients
FastWAM parameters never update
correct PIM > empty/shuffled PIM
```

只有这些成立后，才可以有充分依据判断：

```text
memory 真正改善了 frozen FastWAM 的 action policy
```

而不是仅仅：

```text
给 FastWAM 增加了一个额外条件网络
```
