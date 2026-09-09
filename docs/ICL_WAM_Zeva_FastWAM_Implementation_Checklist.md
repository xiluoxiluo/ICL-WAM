# ICL-WAM（Zeva → Frozen FastWAM）实现核对与修改清单

> 目标：保持 **Zeva 的 causal memory / causal prompt / gated residual conditioning 思想不变**，仅将 Zeva 中的 **Frozen Cosmos Policy** 替换为 **Frozen FastWAM**。  
> 不使用 FastWAM-Joint、IDM、Optional IDM，也不把 causal memory 做成第三个 expert。

---

## 1. 最终目标架构

整体应该保持为两条路径：

### 1.1 Frozen FastWAM 主干

```text
Observation / Language / Proprio / Noise / Timestep
                │
                ▼
        Frozen FastWAM Backbone
                │
                ▼
        Action Hidden / Action Expert
                │
                ▼
          Action Prediction
```

### 1.2 Zeva causal-memory 路径

```text
History / Transition
        │
        ▼
Causal Transition Encoder (CTE)
        │
        ▼
     BIT / PIM
        │
        ▼
Phase-aware Retrieval
        │
        ▼
   Causal Prompt
        │
        ▼
Behavior Prefix / Policy Adapter
        │
        ▼
Memory Residual / Conditioning
        │
        └──────────────► 注入 FastWAM Action Hidden
```

最终应满足：

```text
h_action' = h_fastwam + g * Δh_memory
```

其中：

- `h_fastwam`：FastWAM 原始 action hidden；
- `Δh_memory`：由 Zeva causal memory 路径产生的修正；
- `g`：可学习 gate；
- FastWAM 本体参数冻结；
- memory 相关模块可训练。

---

# 2. 核心设计原则

## 2.1 不要把 memory 做成第三个 expert

### 正确

```text
FastWAM Video / Action Experts
          +
Zeva-style Causal Memory Conditioning
```

memory 只负责：

```text
history
→ CTE
→ BIT/PIM
→ retrieval
→ causal prompt
→ adapter
→ action hidden residual
```

### 不要改成

```text
Video Expert
Action Expert
Memory Expert
```

也不要让 memory 独立做一个 action prediction head。

---

## 2.2 保留 Zeva 的 causal-memory 语义

至少需要保留：

- Causal Transition Encoder；
- BIT；
- PIM；
- phase-aware retrieval；
- causal prompt；
- gated residual policy conditioning。

建议源码结构继续保持类似：

```text
src/fastwam/zeva/
├── causal_transition_encoder.py
├── memory.py
├── retrieval.py
├── causal_prompt.py
├── behavior_prefix_adapter.py
└── ...
```

---

# 3. 最重要：Frozen FastWAM ≠ `torch.no_grad()`

这是目前最需要检查的地方。

## 3.1 正确冻结方式

FastWAM 参数：

```python
for p in fastwam.parameters():
    p.requires_grad_(False)
```

或者：

```python
fastwam.requires_grad_(False)
```

这样：

```text
FastWAM parameter gradient = None
```

但整个 forward 的 autograd graph 仍然存在。

---

## 3.2 不要把 conditioned FastWAM forward 整体包进 `torch.no_grad()`

### 错误示例

```python
with torch.no_grad():
    action_pred = fastwam(
        obs,
        memory_condition=memory_condition,
    )
```

如果 memory residual / prefix / conditioning 是在 FastWAM forward 内部参与计算，那么这样会直接截断：

```text
Action Loss
   X
Memory Adapter
   X
Causal Prompt
   X
CTE
```

导致 memory pathway 学不到东西。

---

## 3.3 正确目标

需要满足：

```text
∂L / ∂θ_fastwam = 0
```

但是：

```text
∂L / ∂θ_memory ≠ 0
```

即：

```text
Action Loss
    │
    ▼
FastWAM computation graph
    │
    ▼
memory residual
    │
    ▼
BehaviorPrefixAdapter
    │
    ▼
Causal Prompt
    │
    ▼
CTE
```

这条计算图必须存在。

---

# 4. BehaviorPrefixAdapter 是本次迁移最关键的位置

Cosmos Policy 和 FastWAM 的 action architecture 不完全相同，所以不能机械照搬 Zeva 原始 adapter。

需要确认当前 `behavior_prefix_adapter.py` 的输出，最终真的进入：

```text
FastWAM 的 action hidden representation
```

而不是只生成一个没有被 action expert 使用的 token。

---

## 4.1 推荐注入位置

优先选择：

```text
FastWAM action token embedding
        │
        ▼
Action Transformer / Action Expert
```

中的 hidden state。

例如：

```python
h_action = action_hidden

delta_h = behavior_prefix_adapter(
    causal_prompt,
    target_len=h_action.shape[1],
)

h_action = h_action + gate * delta_h
```

然后：

```python
action_pred = action_expert(
    h_action,
    ...
)
```

---

## 4.2 判断注入是否真实有效

必须能满足：

```text
相同 observation
相同 language
相同 proprio
相同 timestep
相同 noise

只替换 memory A → memory B

action prediction 必须发生变化
```

即：

```text
a(memory_A) != a(memory_B)
```

如果完全一样，说明 memory 没真正进入 action decision。

---

# 5. 建议保持 gated residual conditioning

推荐：

```text
h' = h + g * Δh_memory
```

而不是直接：

```text
concat([h, memory])
```

原因：

1. 更接近 Zeva；
2. 对 pretrained FastWAM 扰动更小；
3. 可以从接近 baseline 的状态开始训练；
4. 更容易做 zero-memory parity test。

---

# 6. Gate / Adapter 初始化需要检查

建议初始化满足：

```text
训练开始时：
g ≈ 0
```

因此：

```text
h' ≈ h_fastwam
```

避免 causal memory 在训练刚开始就大幅破坏 pretrained FastWAM。

---

## 6.1 注意不要双重完全 zero-init

如果：

```text
最后 projection = 全 0
且
gate = 0
```

则训练初始时可能导致：

```text
CTE grad ≈ 0
CausalPrompt grad ≈ 0
```

甚至前几步完全没有有效梯度。

更稳妥的选择：

### 方案 A

```text
adapter output projection 小随机初始化
gate = 0
```

### 方案 B

```text
adapter output projection zero-init
gate = 一个很小的非零值，例如 1e-3
```

### 方案 C

使用：

```text
gate = sigmoid(gate_logit)
```

并让初始 gate 较小但非严格 0。

---

# 7. BIT / PIM 生命周期必须和 Zeva 一致

## 7.1 BIT

BIT 表示当前 attempt / 当前 rollout 中的短期 causal trajectory。

应满足：

```text
BIT_t = {e_1, e_2, ..., e_t}
```

当前 rollout 持续更新。

当当前 attempt / episode 结束：

```text
BIT → clear
```

---

## 7.2 PIM

PIM 表示跨 attempt 持久 memory。

应该是：

```text
PIM_(k+1) = update(PIM_k, useful evidence from attempt k)
```

不能随着 episode reset 一起清零。

---

## 7.3 推荐显式区分 API

例如：

```python
memory.reset_bit()
memory.update_bit(...)
memory.commit_to_pim(...)
memory.retrieve_from_pim(...)
```

避免把 BIT / PIM 的生命周期混在一个 `reset()` 里。

---

# 8. Phase-aware Retrieval 必须真正参与 memory selection

不能只是：

```text
把所有 memory 全部平均
```

或者：

```text
取最近 K 条
```

应该保留 Zeva 的 phase-aware 逻辑：

```text
current observation / task state
          │
          ▼
      phase estimate
          │
          ▼
retrieve causal evidence
from BIT / PIM for this phase
```

最终：

```text
retrieved_memory = retrieval(
    current_phase,
    bit,
    pim,
)
```

---

# 9. Causal Prompt 应该是 memory 的语义瓶颈

推荐保持：

```text
Retrieved Memory
      │
      ▼
Causal Prompt Module
      │
      ▼
compact causal representation
      │
      ▼
BehaviorPrefixAdapter
```

不要直接让整个 memory bank：

```text
[B, huge_memory_len, dim]
```

全部 concat 到 FastWAM action tokens。

这样既不符合 Zeva 的主要思想，也会大幅增加 FastWAM attention 开销。

---

# 10. 严格检查 causal leakage

预测当前动作：

```text
a_t
```

时，causal memory 只能使用：

```text
o_<=t
a_<t
past attempt memory
current task / phase
```

不能使用：

```text
o_(t+1:T)
future state
future RGB
future proprio
ground-truth a_t:T
```

---

## 10.1 训练阶段特别容易出现的错误

数据 loader 一次拿出完整 trajectory：

```python
obs = trajectory["obs"]
actions = trajectory["actions"]
```

然后直接把整个 trajectory 输入 CTE。

这会导致：

```text
current action prediction
读取到 future transition
```

训练效果可能非常好，但 inference 无法复现。

---

## 10.2 推荐明确 mask

例如：

```python
causal_history = trajectory[:t]
```

或使用 causal mask：

```text
memory_i 只能读取 <= i 的 transition
```

---

# 11. FastWAM 原始 baseline 路径应尽量保持不动

建议原则：

> 不修改 FastWAM 原始 action prediction 逻辑，只增加一个 memory residual hook。

理想修改方式：

```python
h_action = original_fastwam_action_embedding(...)

if causal_memory is not None:
    delta = memory_adapter(causal_memory, h_action)
    h_action = h_action + gate * delta

action_pred = original_fastwam_action_layers(h_action, ...)
```

不要大面积重写 FastWAM action expert。

---

# 12. FastWAM Joint / IDM 不应参与

当前目标只使用：

```text
Base FastWAM
```

不要引入：

```text
FastWAM-Joint
IDM
Optional IDM
```

避免将 Zeva memory 与 IDM 混在一起。

---

# 13. Loss 设计

最简单、最符合当前目标的是：

```text
仍然使用 FastWAM 原始 action objective
```

即：

```text
L = L_action_fastwam
```

然后让：

```text
L_action
→ memory-conditioned FastWAM
→ BehaviorPrefixAdapter
→ Causal Prompt
→ CTE
```

训练 memory pathway。

---

## 13.1 如果还保留 FastWAM video objective

可以：

```text
L = λ_a L_action + λ_v L_video
```

但需要明确：

- FastWAM backbone frozen；
- video loss 不能只是白算；
- 如果 video branch 完全冻结且没有 trainable memory 注入，`L_video` 对当前 memory 训练可能没有意义。

当前项目重点是：

```text
memory → improve action
```

所以优先保证 `L_action` 的梯度链正确。

---

# 14. 必须加入的 5 个 sanity checks

---

## Test 1：Zero-memory parity

### 目的

确认加入 Zeva 模块后，没有破坏 FastWAM baseline。

### 测试

同一个：

```text
observation
language
proprio
noise
timestep
```

比较：

```text
Original FastWAM
```

和：

```text
ICL-WAM with memory OFF
```

### 通过标准

```text
prediction_original ≈ prediction_memory_off
```

最好数值误差只来自 dtype / implementation tolerance。

推荐：

```python
torch.testing.assert_close(
    pred_original,
    pred_memory_off,
    rtol=1e-4,
    atol=1e-5,
)
```

BF16 可以适当放宽。

---

# 15. Test 2：Memory sensitivity

### 目的

确认 memory 真的会改变 action。

固定：

```text
obs
language
proprio
noise
timestep
```

只替换：

```text
memory_A
memory_B
```

运行：

```text
pred_A = model(..., memory_A)
pred_B = model(..., memory_B)
```

计算：

```text
mean(|pred_A - pred_B|)
```

必须：

```text
> 0
```

训练后应该显著大于数值噪声。

---

# 16. Test 3：Gradient flow

一次训练 step：

```python
loss.backward()
```

然后检查。

### FastWAM

```python
assert all(
    p.grad is None
    for p in fastwam.parameters()
)
```

### Memory modules

至少：

```text
BehaviorPrefixAdapter
CausalPrompt
CTE
```

应存在非零梯度。

例如：

```python
def grad_norm(module):
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total += p.grad.detach().float().norm().item()
    return total
```

期待：

```text
grad_norm(fastwam) = 0
grad_norm(adapter) > 0
grad_norm(causal_prompt) > 0
grad_norm(cte) > 0
```

---

# 17. Test 4：BIT / PIM 生命周期

写独立 unit test。

### Attempt 1

```text
BIT = []
PIM = []
```

不断执行 transition：

```text
BIT size 增加
```

attempt 结束：

```text
有价值信息 → PIM
BIT → clear
```

### Attempt 2

需要看到：

```text
BIT = []
PIM != []
```

如果：

```text
PIM == []
```

说明持久 memory 被错误 reset。

---

# 18. Test 5：Causal leakage

随机选择 timestep：

```text
t
```

分别构造：

```text
trajectory_A
trajectory_B
```

要求：

```text
trajectory_A[:t] == trajectory_B[:t]
```

但是：

```text
trajectory_A[t+1:] != trajectory_B[t+1:]
```

如果当前 `a_t` prediction 因未来部分不同而变化，则发生 leakage。

正确情况下：

```text
pred_A_t ≈ pred_B_t
```

---

# 19. 建议增加 runtime debug metrics

训练日志建议增加：

```text
train/loss_action

memory/gate_mean
memory/gate_max

memory/prefix_norm
memory/delta_hidden_norm
memory/retrieved_memory_norm

grad/adapter
grad/causal_prompt
grad/cte

memory/bit_size
memory/pim_size
memory/retrieval_count
```

---

## 19.1 特别建议记录 residual ratio

定义：

```text
r = ||g * Δh_memory|| / (||h_fastwam|| + eps)
```

记录：

```text
memory/residual_ratio
```

训练初期应较小，例如：

```text
0 ~ 1%
```

后续逐渐增加。

如果长期：

```text
≈ 0
```

说明 memory 没学起来。

如果迅速变成：

```text
> 50%
```

则可能 memory 在破坏 pretrained policy。

---

# 20. 推荐增加 forward debug 输出

开发阶段可以让 forward 额外返回：

```python
debug = {
    "gate": gate.detach(),
    "base_action_hidden_norm": h_action.detach().norm(),
    "memory_delta_norm": delta_h.detach().norm(),
    "conditioned_action_hidden_norm": h_action_cond.detach().norm(),
    "retrieved_memory_count": ...,
}
```

便于快速判断 memory 是否真正进入 FastWAM。

---

# 21. 推荐增加配置开关

建议 config 明确增加：

```yaml
zeva_memory:
  enabled: true

  use_cte: true
  use_bit: true
  use_pim: true
  use_phase_retrieval: true
  use_causal_prompt: true

  conditioning:
    type: gated_residual
    target: action_hidden

  gate:
    init: small

  freeze_fastwam: true
```

并支持：

```yaml
zeva_memory:
  enabled: false
```

这样可以直接跑 FastWAM baseline。

---

# 22. 建议增加 ablation 开关

至少支持：

```text
FastWAM
FastWAM + CTE only
FastWAM + BIT
FastWAM + BIT + PIM
FastWAM + full Zeva memory
```

以及：

```text
full memory
w/o phase retrieval
w/o PIM
w/o gate
```

方便后续论文实验。

---

# 23. 优先检查的代码路径

建议按照下面顺序检查，不要从 peripheral module 开始。

---

## Priority 1：FastWAM action forward

找到真正：

```text
action embedding
→ action transformer
→ action output
```

的位置。

确认：

```text
memory residual
```

是插入这里。

---

## Priority 2：BehaviorPrefixAdapter

确认：

```text
input = causal prompt
output shape = action hidden compatible
```

例如：

```text
[B, L_memory, D_memory]
→
[B, L_action, D_fastwam]
```

或者：

```text
[B, D_memory]
→
[B, L_action, D_fastwam]
```

必须明确 shape。

---

## Priority 3：Training loop

确认：

```text
FastWAM parameters requires_grad=False
```

但：

```text
没有把整个 conditioned FastWAM forward 包进 no_grad
```

---

## Priority 4：Optimizer

optimizer 只包含：

```text
CTE
Causal Prompt
BehaviorPrefixAdapter
可学习 retrieval / phase 模块
```

不要包含：

```text
FastWAM backbone parameters
```

推荐显式：

```python
trainable_params = [
    p for p in model.parameters()
    if p.requires_grad
]
```

同时打印 trainable parameter names。

---

# 24. 启动训练时必须打印参数统计

建议：

```text
Total parameters:
Frozen FastWAM parameters:
Trainable Zeva-memory parameters:
Trainable ratio:
```

以及 trainable module names：

```text
zeva.cte.*
zeva.causal_prompt.*
zeva.behavior_prefix_adapter.*
...
```

如果看到：

```text
fastwam.blocks.*
```

出现在 optimizer 中，需要立即检查。

---

# 25. 推荐训练阶段

## Stage 0：FastWAM baseline verification

先确保原始：

```text
FastWAM
```

在 RoboTwin 配置上能正常训练 / inference。

---

## Stage 1：Frozen FastWAM + Adapter only

先关闭复杂 memory：

```text
固定一个简单 learned prompt
→ Adapter
→ FastWAM
```

验证：

```text
gradient flow
memory sensitivity
zero-memory parity
```

---

## Stage 2：CTE + BIT

加入：

```text
history
→ CTE
→ BIT
→ Causal Prompt
```

先不要加 PIM。

---

## Stage 3：PIM + Phase Retrieval

最后加入：

```text
cross-attempt memory
phase-aware retrieval
```

这样出问题时更容易定位。

---

# 26. 推荐最小训练目标

当前阶段先不要追求复杂辅助 loss。

先保证：

```text
L_action
```

能够训练：

```text
CTE
CausalPrompt
Adapter
```

即：

```text
L_action
→ conditioned FastWAM
→ memory residual
→ memory network
```

这条链闭环后，再考虑：

```text
phase loss
retrieval contrastive loss
memory consistency loss
```

---

# 27. 正确实现应满足的最终数学关系

Frozen FastWAM：

```text
θ_F = constant
```

Memory network：

```text
M_t = f_memory(H_<=t ; θ_M)
```

Adapter：

```text
ΔH_t = f_adapter(M_t ; θ_A)
```

Conditioned FastWAM hidden：

```text
H_t' = H_t^F + g · ΔH_t
```

Action：

```text
â_t = FastWAM_action(H_t'; θ_F)
```

Loss：

```text
L_action = L(â_t, a_t)
```

训练时：

```text
∂L / ∂θ_F = 0
```

但：

```text
∂L / ∂θ_A != 0
∂L / ∂θ_M != 0
```

这是整个项目是否实现正确的最核心判据。

---

# 28. 最终验收 Checklist

在开始大规模训练前，下面每条必须确认。

## Architecture

- [ ] Frozen Cosmos Policy 已完全替换为 Base FastWAM。
- [ ] 未使用 FastWAM-Joint。
- [ ] 未使用 IDM / Optional IDM。
- [ ] memory 没有被实现成第三个 expert。
- [ ] CTE 存在。
- [ ] BIT 存在。
- [ ] PIM 存在。
- [ ] Phase Retrieval 存在。
- [ ] Causal Prompt 存在。
- [ ] Gated Residual Conditioning 存在。

## Conditioning

- [ ] Causal Prompt 最终进入 FastWAM action hidden。
- [ ] Adapter 输出 shape 与 FastWAM action hidden 完全一致。
- [ ] Memory 改变时 action prediction 会改变。
- [ ] Memory 关闭时恢复 Original FastWAM。

## Gradient

- [ ] FastWAM parameters `requires_grad=False`。
- [ ] FastWAM conditioned forward 未被整体包进 `torch.no_grad()`。
- [ ] FastWAM parameter grad 全部为 `None`。
- [ ] Adapter grad 非零。
- [ ] Causal Prompt grad 非零。
- [ ] CTE grad 非零。

## Memory

- [ ] BIT 在当前 attempt 内更新。
- [ ] BIT 在 attempt 结束后 reset。
- [ ] PIM 跨 attempt 保留。
- [ ] PIM 不会被 episode reset 清空。
- [ ] Retrieval 根据 current phase 工作。

## Causality

- [ ] 当前 action 不读取未来 observation。
- [ ] 当前 action 不读取未来 proprio。
- [ ] 当前 action 不读取未来 ground-truth action。
- [ ] CTE 输入严格 causal。
- [ ] Retrieval 输入严格 causal。

## Tests

- [ ] Zero-memory parity test 通过。
- [ ] Memory sensitivity test 通过。
- [ ] Gradient-flow test 通过。
- [ ] BIT/PIM lifecycle test 通过。
- [ ] Causal leakage test 通过。

---

# 29. 推荐优先修改顺序

```text
1. 检查 FastWAM action hidden 的真实位置
        ↓
2. 确认 BehaviorPrefixAdapter 注入到该 hidden
        ↓
3. 检查是否错误使用 torch.no_grad()
        ↓
4. 检查 optimizer / requires_grad
        ↓
5. 做 gradient-flow test
        ↓
6. 做 zero-memory parity
        ↓
7. 做 memory sensitivity
        ↓
8. 检查 BIT/PIM lifecycle
        ↓
9. 检查 phase retrieval
        ↓
10. 检查 causal leakage
        ↓
11. 再启动 RoboTwin 正式训练
```

---

# 30. 最终项目定义

如果以上全部通过，可以将当前方法准确描述为：

> **Zeva-style causal memory augmented FastWAM, where the original frozen Cosmos Policy is replaced by a frozen Base FastWAM policy backbone, while retaining the causal transition encoder, BIT/PIM memory, phase-aware retrieval, causal prompt generation, and gated residual policy conditioning.**

简称：

```text
Frozen FastWAM
+
Zeva Causal Memory
```

而不是：

```text
FastWAM + Memory Expert
```

也不是：

```text
FastWAM-Joint + IDM + Zeva
```

---

# 31. 最关键的一句话

实现是否正确，最终只看：

```text
Action Loss
   │
   ▼
Frozen FastWAM computation graph
   │
   ▼
Memory residual
   │
   ▼
BehaviorPrefixAdapter
   │
   ▼
Causal Prompt
   │
   ▼
CTE
```

是否真实可反向传播。

同时要求：

```text
FastWAM 参数不更新
Memory pathway 参数可以更新
```

满足这两点，才真正实现了：

```text
用 Zeva-style causal memory 提升 Frozen FastWAM 的 action prediction
```
