# ICL-WAM：将 CTE 收敛为 Zeva 原始实现的迁移修改方案

## 0. 文档目的

本文用于记录并审视当前 `xiluoxiluo/ICL-WAM` 的 CTE 收敛修改。除第 0.1 节
标出的未完成 gate 外，不把已经落地的意见重复列为新的改码任务。

目标不是继续扩展一个 “Zeva-inspired CTE”，而是把当前 CTE 收敛到 **Zeva 官方当前实现的语义**，仅做 FastWAM / RoboTwin 必需的接口适配，从而使第一版方法可以清晰描述为：

本轮评估结论：计划中“把 Wan VAE 写进 CTE”不合理；但“在 CTE 外用冻结的 Wan VAE 将 RGB 转成 latent”与 Zeva 当前服务路径一致，属于合理的上游适配。最终边界是：`causal_transition_encoder.py` 与 `cte_losses.py` 按 Zeva 原实现语义迁入，仅保留 RoboTwin 必需适配；CTE 只接收通用 `[B,T,C,H,W]`；RGB→Wan latent 由显式 adapter 完成，并把 `cte_input_type`、latent 通道数和 VAE 身份写入 checkpoint/cache，禁止混用 RGB 与 latent 合同。

## 0.1 本轮合理性核对结论（2026-09-06）

当前工作树已经完成了本方案的大部分合理修改；本文后续的“当前 ICL-WAM”在第 2 节中特指**修改前的审阅基线**，不是当前代码状态。与当前代码和本地 Zeva 实现核对后，以下意见继续保留：

- Wan VAE 只能作为 CTE 外部的冻结 adapter；CTE 仍保持通用 `[B,T,C,H,W]` 接口。
- phase 每 4 个 raw actions 更新，effect 每 4 个 transition（16 个 raw actions）完成后才可用。
- 保留 action-level `transition_valid=[B,T-1,4]`、right-shift action stream 和 full-history CTE；移除 transition-level effect head/loss。
- BIT 只接收 completed `effect_post`；PIM 使用 effect-window 起点的 phase 与对应的 observed `effect_post`，并在 attempt 边界提交。
- RGB/latent、通道数、VAE 身份必须由 checkpoint/cache metadata 严格隔离；`image_channels=48` 只适用于当前 Wan2.2 VAE38 latent 合同，不能成为 CTE 的无条件内置假设。

以下边界需要明确，避免把合理意见执行得过宽：

1. “迁入官方实现”指语义和张量合同与 Zeva 对齐，不要求逐字复制；RoboTwin 的 `action_dim=14`、输入/mask 校验和 adapter 边界属于允许的最小适配。
2. 删除的是神经网络的 transition-level effect 分支及其 loss；cache/lifecycle 中用于来源定位的 `transition_index`、`effect_index`，以及旧调用方的兼容 alias 可以保留，但不得重新解释为每 4 action 产生一个 effect。
3. 完整 32-action window 才应有 2 个 `effect_complete=True` 条目；CTE 对部分或
   带 padding 的 window 可以保留固定形状的无效槽位，但 cache/BIT/PIM 只能写入
   实际 complete 的条目，不能为了满足固定条数而补造 effect。
4. 本地静态验证已通过（`PYTHONPATH=src pytest -q`：31 passed、compileall、`git diff --check`）。由于当前环境没有真实 RoboTwin 数据、CUDA 和 SAPIEN，端到端训练/rollout 仍是后续 empirical gate，不能在文档中提前宣称完成。

另有一项需要保留在风险清单中的合同细节：adapter 层的
`validate_vae_metadata` 已按语义身份允许不同 `vae_path`，但当前
`CacheManifest.validate` 对 `vae_metadata` 仍做整体字典精确比较。若 cache
需要跨机器搬运，这可能仅因绝对路径变化而误拒；应在实现阶段将 manifest
比较也规范为语义身份字段，或先统一路径规范。这不改变 CTE/Zeva 语义，属于
小的 metadata portability gate，不能与模型效果 gate 混为一谈。

另有一项需要保留在风险清单中的语义差异：当前 offline cache/Stage 1 以不重叠
的 32-action window 独立运行 CTE，而 online runtime 会在一次 attempt 内累积
更长的 full history。两者都遵守 Zeva 的 `[B,T,C,H,W]` 接口和 causal mask，
但不自动保证长历史下的 phase 数值完全一致；若实验要求严格 offline/online
等价，应改为 episode-prefix cache 并新增对应的等价性测试，不能仅凭现有
window-level tests 宣称已经证明。

本轮意见与代码证据对应如下：

| 意见 | 合理性结论 | 当前证据 |
|---|---|---|
| VAE 放在 CTE 外部 | 保留 | `src/fastwam/zeva/vae_adapter.py`、部署侧 `FastWAMCTELatentEncoder` |
| 4-action phase / 16-action effect | 保留 | `CausalTransitionEncoder.effect_window_transitions=4`、`test_causal_state.py` |
| action-level validity 与 right shift | 保留 | `schemas.py` 的 `[B,T-1,4]`、`test_transition_alignment.py` / `test_causal_state.py` |
| 删除 transition-level effect | 保留 | CTE/loss 中无对应 head/weight，源码静态审计通过 |
| full-history runtime | 保留 | `CausalCTEHistory.forward()`；无正式 `initialize/update` handoff |
| pending phase + observed effect | 保留 | `CausalMemoryLifecycle`、`test_memory_lifecycle.py` |
| offline cache 与 online 长历史严格等价 | 暂不宣称 | 当前 cache 是 window-local proxy，属于未完成 empirical gate |
| VAE metadata 跨主机路径可移植 | 需补合同修订 | adapter 已忽略路径；manifest validator 仍整体精确比较 |

\[
\boxed{
\text{Original Zeva ICCL Mechanism}
+
\text{Frozen FastWAM Backbone}
}
\]

保留：

- Zeva 原 CTE
- BIT
- PIM
- phase-conditioned retrieval
- Causal Prompt
- gated residual policy conditioning
- deployment 时 neural parameters 全冻结，仅 memory 更新

仅替换：

\[
\text{Zeva frozen policy backbone}
\rightarrow
\text{Frozen FastWAM}
\]

明确不使用：

- `FastWAM-Joint`
- `IDM`
- optional IDM
- third causal expert
- 新增 world-model joint loss
- 对 FastWAM backbone 做联合训练

---

# 1. Canonical Source：以后以 Zeva 官方实现为准

CTE 的主定义不应以历史 ICL-WAM 实现为准，而应以 Zeva 官方代码作为 canonical source：

```text
air-embodied-brain/Zeva

cosmos_framework/model/zeva/
├── causal_transition_encoder.py
└── cte_losses.py
```

ICL-WAM 中对应：

```text
src/fastwam/zeva/
├── causal_transition_encoder.py
└── cte_losses.py
```

当前实现已按该版本迁入，并只保留 RoboTwin action width、padding/mask、checkpoint
metadata 等最小适配；后续若 Zeva 上游更新，应先做语义 diff，再决定是否同步。

文件头应保留 Zeva 原始 SPDX / license attribution，并额外注明：

```python
# Adapted from air-embodied-brain/Zeva
#
# ICL-WAM adaptations:
# 1. RoboTwin action_dim = 14
# 2. RoboTwin data / padding / cache interface compatibility
#
# CTE architecture and objectives remain otherwise unchanged.
```

---

# 2. 需要纠正的核心差异

修改前 ICL-WAM 与目标版本最重要的区别如下。

| 项目 | 当前 ICL-WAM | Zeva 官方语义 | 修改要求 |
|---|---|---|---|
| CTE 输入 | RGB mosaic | Zeva 服务端传入 Wan VAE latent（模块仍是 `[B,T,C,H,W]`） | CTE 模块不改；上游显式 adapter 负责 RGB→latent |
| phase cadence | 每 4 action | 每 4 action | 保留 |
| effect cadence | 每 4 action | 每 4 transitions = 16 actions | 必须修改 |
| effect target | RGB random projection | Zeva `_FrozenVAEDeltaTarget` 的固定空间投影 | 保持官方实现；类名不等于增加 VAE 调用 |
| effect branch | transition-level effect + window effect | 官方 effect-window branch | 删除额外 transition effect |
| CTE loss | 增加 transition effect loss | Zeva effect-v3 objective | 恢复官方版本 |
| online CTE | custom `initialize/update` recurrent handoff | full causal history forward | 正式 V1 改为 full-history |
| `transition_valid` | `[B,T-1]` | `[B,T-1,4]` | 必须恢复 |
| PIM phase/effect pair | 常用 `phase_post + effect` | pending phase + observed effect_post | 必须修改 |

注：上表左列是本轮修改前的差异，用来说明为何提出修改；当前工作树已经按“修改要求”一列完成了大部分收敛。未在代码中重复实现的旧字段，不应再作为新的改动任务提出。

---

# 3. CTE 输入契约：保持 Zeva 的帧接口

## 3.1 目标

当前/目标：

```text
RoboTwin RGB mosaic
        ↓
Frozen Wan VAE adapter（仅当 checkpoint 声明 wan_vae_latent）
        ↓
Original Zeva CTE (`[B,T,C,H,W]`)
```

即（`wan_vae_latent` 合同时先执行 adapter）：

\[
o_{0:t}^{\mathrm{RGB}}
\xrightarrow{\text{optional frozen Wan-VAE adapter}}
z_{0:t}^{\mathrm{CTE}}
\xrightarrow{\text{Original Zeva CTE}}
p_t,e_t
\]

这样既保持 CTE 源码和接口一致，又与 Zeva 当前 serving 的实际输入一致。

## 3.2 VAE 边界

VAE 不能放在 CTE 内部，也不能根据输入 shape 静默猜测。正式实现使用独立
`vae_adapter.py`，由配置中的 `input_type` 选择，并在 checkpoint/cache 中规范
记录为 `cte_input_type`；RGB frame
和 Wan latent 是两个不可混用的 checkpoint/cache 合同。

原计划中的新增文件建议保留，但仅作为 CTE 外部的显式适配层：

```text
src/fastwam/zeva/vae_adapter.py
```

外部 adapter 的接口：

```python
class FastWAMCTELatentEncoder:
    """
    Convert RoboTwin RGB observation to the latent expected by Zeva CTE.

    RGB [-1, 1]
        -> frozen FastWAM Wan VAE
        -> one-frame latent [B, C_latent, H_latent, W_latent]
    """

    def __init__(self, fastwam):
        self.fastwam = fastwam

    @torch.no_grad()
    def encode(self, rgb: torch.Tensor) -> torch.Tensor:
        ...
```

输入：

```text
rgb: [B, 3, H, W]
```

输出：

```text
latent: [B, C_latent, H_latent, W_latent]
```

不要把 `C_latent` 硬编码在 CTE 中，运行时从真实 VAE 输出读取并写进 CTE config / checkpoint metadata。

其中 `model_id`、`z_dim`、时间下采样因子和上采样因子是跨机器必须一致的
语义身份；`vae_path` 只作来源记录，可以因机器/挂载点不同而变化。cache
兼容性检查不得仅因绝对路径变化拒绝同一 VAE 合同。

---

# 4. 保持 Zeva 官方 `_FrozenVAEDeltaTarget`

当前 ICL-WAM 中若存在类似：

```python
class _FrozenVisualDeltaTarget(nn.Module):
```

其语义是：

```text
CTE frame tensor（RGB 或 Wan latent）
↓
AdaptiveAvgPool
↓
LayerNorm
↓
固定随机投影
```

不要另行设计一个新的 RGB target；目标版本就是 Zeva 官方的固定空间投影。

应保持 Zeva 官方：

```python
class _FrozenVAEDeltaTarget(nn.Module):
```

其输入就是 CTE 的 frame tensor（与 Zeva 相同）：

```text
CTE frame tensor
↓
4×6 spatial pool
↓
fixed projection
↓
effect delta target
```

即：

\[
\phi(z_t)
=
P\left(
Pool_{4\times6}(z_t)
\right)
\]

effect target：

\[
\Delta\phi_j
=
\phi(z_{j+4})
-
\phi(z_j)
\]

注意：

这里的 `j` 是 transition/frame-boundary 索引，因此 `j→j+4` 等价于 raw action
偏移 `+16`；若用 raw-action 索引表示，才可写成 `t→t+16`。不要把 CTE 的
frame 索引和 raw-action 索引混写。

- projector 可以继续保持 fixed random projection；
- 不要把类名 `_FrozenVAEDeltaTarget` 解读为再次调用 FastWAM VAE；它是 Zeva
  原始实现中的固定空间投影；
- 不要再额外设计一个新 learned effect teacher。

---

# 5. `image_channels` 必须与上游合同一致

CTE 本身不固定通道数；`image_channels` 必须等于实际传入 CTE 的 tensor
通道数。直接 frame 合同使用 `3`，Zeva/FastWAM Wan-VAE 合同通常使用
VAE 的 `z_dim`（Wan2.2 VAE38 为 `48`）。不能把 `48` 写成 CTE 的无条件
默认值，也不能让 CTE 在运行时自动转换。

启动 latent 合同时，配置应先声明预期的 `image_channels`；由真实
adapter 输出做一次严格校验：

```python
latent = cte_latent_encoder.encode(sample_rgb)
if latent.shape[1] != cte_cfg.image_channels:
    raise ValueError("VAE latent channels do not match the CTE checkpoint/config")
```

并将实际通道数与 VAE 身份写入：

```text
cte.pt metadata
cache manifest
Stage2 compatibility check
evaluation compatibility check
```

---

# 6. Zeva 的 phase cadence 和 effect cadence 必须分开

## 6.1 Phase

保留：

```text
transition_steps = 4
```

即每 4 个 raw actions 形成一个 transition。

所以：

\[
1\ transition = 4\ actions
\]

phase 每个 transition boundary 都可以更新：

\[
p_0,p_1,\dots,p_8
\]

## 6.2 Effect

保留 Zeva：

```text
effect_window_transitions = 4
```

即：

\[
1\ effect
=
4\ transitions
=
16\ actions
\]

因此 RoboTwin 一个 32-action FastWAM chunk：

\[
32\ actions
\rightarrow
8\ transitions
\rightarrow
2\ effects
\]

而不是 8 个 effects。

---

# 7. RoboTwin 32-action 时间轴必须改成下面这样

```text
o0
│
├─ a0 a1 a2 a3
│
o1                  phase p1
│
├─ a4 a5 a6 a7
│
o2                  phase p2
│
├─ a8 a9 a10 a11
│
o3                  phase p3
│
├─ a12 a13 a14 a15
│
o4                  phase p4
│
└──────────────→ effect e0
                  window: o0 → o4
                  actions: a0:a15

├─ a16 a17 a18 a19
│
o5                  phase p5
│
├─ a20 a21 a22 a23
│
o6                  phase p6
│
├─ a24 a25 a26 a27
│
o7                  phase p7
│
├─ a28 a29 a30 a31
│
o8                  phase p8
│
└──────────────→ effect e1
                  window: o4 → o8
                  actions: a16:a31
```

总结：

\[
\boxed{
32\ actions
\rightarrow
8\ phase\ transitions
\rightarrow
2\ effect\ windows
}
\]

---

# 8. 删除额外 transition-level effect 分支

当前 ICL-WAM 中如果存在下面这些分支：

```python
transition_effect_predictor
transition_effect_head
transition_effect_outcome
```

以及：

```python
transition_effect
transition_effect_prediction
transition_effect_observed_outcome
transition_effect_weight
transition_observed_nce
```

全部从正式 V1 CTE 中删除。

目标版本只保留 Zeva 官方 effect-window 相关：

```text
effect_pre
effect_post
effect_outcome_pre
effect_outcome_post
effect_actions
effect_complete
effect_delta_target
```

不要再维护：

```text
4 actions -> one memory effect
```

这里的“删除”仅针对 CTE 的 effect 神经分支和对应 loss。为保证 cache 行与
episode/source 对齐，`transition_index`、`effect_index` 等 provenance 字段，
以及 `observe_completed_transition` 这类兼容入口可以继续存在；它们不得改变
effect 仍按 16 raw actions 完成一次的语义。

---

# 9. `cte_losses.py` 恢复 Zeva 官方 effect-v3 objective

目标 CTE loss：

\[
L_{CTE}
=
L_{action}
+
L_{vision}
+
0.2L_{task}
+
0.1L_{phase}
+
0.25L_{effect}
\]

其中：

\[
L_{effect}
=
L_{NCE}
+
0.05L_{effect-action}
+
0.1L_{align}
+
L_{variance}
+
0.04L_{covariance}
\]

建议保留官方默认：

```python
action_weight = 1.0
vision_weight = 1.0
task_weight = 0.2
phase_weight = 0.1

effect_weight = 0.25
effect_contrastive_weight = 1.0
effect_action_weight = 0.05
effect_align_weight = 0.1
effect_variance_weight = 1.0
effect_covariance_weight = 0.04

effect_temperature = 0.07
temperature = 0.1
```

删除 ICL-WAM 自增的：

```python
transition_effect_weight
transition_observed_nce
```

以及相关 loss 计算。

---

# 10. 恢复 `transition_valid` 的 action-level 粒度

当前目标：

```text
transition_valid: [B, T-1, 4]
```

RoboTwin 一个 32-action window：

```text
[B, 8, 4]
```

含义：

```text
8 transitions
×
4 raw actions / transition
```

不要提前压成：

```text
[B, 8]
```

Zeva CTE 内部用：

```python
transition_valid.all(dim=-1)
```

判断 transition 是否完整。

对于 effect window：

```python
effect_action_valid.all(dim=(-1, -2))
```

判断完整 16-action effect window 是否有效。

---

# 11. 保留 right-shift causal action stream

这部分当前方向是正确的，不要改。

核心语义：

\[
v_b
\]

只能看到已经完成的：

\[
u_{b-1}
\]

不能看到当前未执行 transition：

\[
u_b
\]

对应实现：

```python
action[:, 0] = BOS
action[:, 1:] = transition_embed
```

即：

```text
state o0:
    sees BOS

state o1:
    sees transition T0

state o2:
    sees transition T1
```

这是 CTE 防止 future leakage 的关键要求。

---

# 12. 正式 V1 不再使用 custom incremental `initialize/update`

修改前的 ICL-WAM 若使用：

```python
cte.initialize(...)
cte.update(...)
initial_state
initial_state_mask
```

作为主 evaluation path，正式 V1 应取消这种 handoff。

原因：这属于 ICL-WAM 自己增加的 incremental optimization，不是当前 Zeva 官方主路径。
当前实现已经改为 full-history；若未来保留兼容 alias，也不得将其作为正式
evaluation/training path。

正式 V1 使用：

```text
all executed CTE-tensor boundaries
+
all completed transition actions
↓
full causal CTE forward
```

Runtime 可缓存原始 RGB boundary，但在进入 CTE 前必须按
`cte_input_type` 经过外部 adapter；这不改变 CTE 的公开输入合同。

---

# 13. Runtime 改成缓存 causal history

评测时维护（可以保存 RGB 原始边界，但送入 CTE 前必须按合同转换）：

```python
self._cte_boundary_frames: list[Tensor]
self._cte_transition_actions: list[Tensor]
self._cte_transition_valid: list[Tensor]
```

episode reset：

```python
self._cte_boundary_frames = [initial_rgb]
self._cte_transition_actions = []
self._cte_transition_valid = []
```

每完成 4 个实际执行动作：

```python
self._cte_transition_actions.append(
    executed_action_group
)

self._cte_transition_valid.append(
    action_valid_group
)

self._cte_boundary_frames.append(
    after_rgb
)
```

需要当前 phase / effect 时（latent 合同先在 CTE 外执行 adapter）：

```python
frames = torch.stack(
    self._cte_boundary_frames,
    dim=0
).unsqueeze(0)

actions = torch.stack(
    self._cte_transition_actions,
    dim=0
).unsqueeze(0)

transition_valid = torch.stack(
    self._cte_transition_valid,
    dim=0
).unsqueeze(0)

valid_mask = torch.ones(
    (1, frames.shape[1]),
    dtype=torch.bool,
    device=frames.device,
)

encoded = self.cte(
    frames,
    actions,
    valid_mask=valid_mask,
    transition_valid=transition_valid,
)
```

当前 phase：

```python
current_phase = encoded["phase"][:, -1]
```

completed effect：

```python
completed_effects = encoded["effect_post"][
    encoded["effect_complete"]
]
```

当前代码由 `CausalCTEHistory` 封装上述缓存、外部 frame encoder 和 full-history
forward；私有字段名称可以不同，但不得退回到 recurrent `initial_state` handoff。

---

# 14. VAE 只能位于 CTE 外部适配层

CTE 不得自行调用 VAE。根据 checkpoint 的 `cte_input_type`，正式路径要么
直接传入已经准备好的 frame tensor，要么先由冻结 Wan VAE adapter 编码，
然后再传入完全相同的 CTE forward。两条路径必须使用不同的 metadata/cache
合同，不能把 RGB cache 当作 latent cache。

运行时可以缓存 boundary RGB frame，并在 adapter 层得到 CTE 输入：

```text
RGB
↓
optional frozen Wan VAE adapter
↓
CTE frame tensor
```

后续 CTE 只重跑：

```text
VisionStem
GRU/Mamba
same-step cross attention
CTE heads
```

因此：

```python
self._cte_boundary_frames
```

不需要在 Zeva CTE 内维护 latent；CTE 只维护传入的 frame/latent tensor，
而 RGB 历史由外部 adapter 按合同转换。

---

# 15. 如果以后优化 incremental CTE，必须放 experimental

如果后面 full-history CTE 成为 latency bottleneck，可以新建：

```text
src/fastwam/zeva/experimental/cte_incremental.py
```

但不能直接替代正式实现。

必须做数值等价测试：

```python
out_full = cte(full_history)

out_incremental = cte_incremental(...)
```

要求至少：

```python
torch.testing.assert_close(
    phase_full,
    phase_incremental,
    rtol=...,
    atol=...,
)
```

以及：

```python
torch.testing.assert_close(
    effect_full,
    effect_incremental,
    rtol=...,
    atol=...,
)
```

验证通过后才允许用于正式 evaluation。

---

# 16. BIT 也必须改成 16-action effect cadence

当前不要再：

```text
every 4 actions
→ effect
→ BIT
```

改成：

```text
every 4 actions
→ phase update

every 16 actions
→ completed effect_post
→ BIT append
```

BIT 中：

```text
brief_length = 4
```

所以保留最近 4 个 completed effect windows：

\[
BIT_t
=
[e_{k-3},e_{k-2},e_{k-1},e_k]
\]

一个 effect 对应 16 raw controls。

---

# 17. PIM 必须存 pending phase + observed effect_post

不要继续默认：

```text
phase_post + effect
```

建议明确实现 pending causal pair；当前实现既可以在窗口开始时缓存该 phase，
也可以在 effect 完成时从同一次 full-history CTE 输出读取窗口起点的 phase，
二者必须数值上指向同一个 boundary。

在一次 effect window 开始时保存：

```python
pending_phase = phase_at_window_start
```

16 个 actions 完成、outcome 可观察以后：

```python
effect_post = observed_effect
```

commit：

```python
pim.append_completed(
    phase=pending_phase,
    effect=effect_post,
    ...
)
```

其语义是：

\[
\boxed{
\text{At phase }p,
\text{ after actual execution, I observed effect }e
}
\]

不要用 effect 完成后的 `phase_post` 替代窗口起点 phase。`phase_post` 可作为
诊断或 cache 字段保留，但不是该 PIM pair 的 query key。

---

# 18. 与 attempt-end PIM commit 结合

建议采用：

```text
current attempt:
    effect → BIT
    effect → pending PIM buffer

attempt end:
    pending PIM buffer
    → commit to PIM

next attempt:
    PIM visible
```

因此 lifecycle 推荐：

```python
observe_completed_effect(...)
    -> BIT append
    -> pending append

end_attempt(...)
    -> pending entries commit to PIM
    -> pending clear
```

这样：

\[
BIT = current\ attempt
\]

\[
PIM = completed\ previous\ attempts
\]

---

# 19. Cache schema 必须拆分 phase 和 effect

不要再一条 transition row 对应一个 effect。

V1 继续使用现有 safetensors + JSON index 的扁平存储；逻辑上每个
effect-window 一行（而不是每个 transition 一行）：

```python
{
    "window_index": int,
    "episode_id": str,
    "task_id": ...,

    "effect_index": int,              # 0 or 1
    "transition_index": int,          # 0 or 4, compatibility/source offset
    "phase_pre": Tensor[128],         # phase at effect-window start
    "phase_post": Tensor[128],
    "effect": Tensor[128],            # completed effect_post
    "valid": bool,
}
```

不要再假设：

```python
transition_index == effect_index
```

因为：

\[
8\ transitions
\neq
2\ effects
\]

当前实现将上述格式落为 `zeva_fastwam_robotwin_cache_v3`。`transition_index`
保留为来源 offset（0、4），真正区分两条记录的是 `effect_index`；Stage 2
必须拒绝旧的 transition-level v2 cache，避免把旧语义静默混入训练。

---

# 20. `build_zeva_robotwin_cache.py` 修改要求

当前 cache builder 已按以下合同实现；若重建 cache，应继续保持：

1. 从 FastWAM dataset 获取 RoboTwin RGB frame tensor；
2. 原样构造 9 个 boundary frames；若 CTE checkpoint 声明
   `wan_vae_latent`，先通过冻结的外部 adapter 编码，不在 CTE 内调用 Wan VAE；
3. 构造：
   ```text
   transition_actions [8,4,14]
   transition_valid [8,4]
   ```
4. 调用 original Zeva CTE：
   ```python
   encoded = cte(...)
   ```
5. 每个完整 32-action window 保存两条 effect-window 记录；部分 window 只保存
   实际 complete 的记录，不补造无效 effect：
   ```text
   phase_pre/phase_post/effect [128] × 2
   ```
6. cache manifest 记录：
   ```text
   CTE checkpoint hash
   dataset stats hash
   cte_input_type（`rgb_frame` 或 `wan_vae_latent`）
   image_channels / latent_channels
   vae_metadata（仅 `wan_vae_latent` 合同）
   action_dim=14
   action_group_size=4（等价于 CTE `transition_steps=4`）
   transition_count=8
   effect_window_transitions=4
   phase_dim=128
   effect_dim=128
   ```

cache builder 只消费按 episode 排序的不重叠 source windows；每个 32-action
window 内使用完整的 9-frame history，正式 CTE 不跨 window 携带 recurrent
`initial_state`。这是当前 Stage 2 的 window-local offline proxy，可避免滑动
窗口重复计入 PIM，但它与 online attempt-prefix history 不是自动数值等价。
若未来需要严格的跨 window 长历史，应先定义新的 cache schema、来源索引规则
和等价性测试。

---

# 21. Stage 2 Dataset 必须只把 completed `effect_post` 当 memory

`ZevaStage2Dataset` 已改为只把 completed `effect_post` 作为 memory；后续修改
不得退回：

```text
each 4-action transition
→ add one memory effect
```

应改成：

```python
for effect_idx in range(num_effect_windows):
    if effect_complete[effect_idx]:
        memory_bank.add(
            phase=phase_at_effect_start,
            effect=effect_post[effect_idx],
        )
```

例如 32-action window：

```text
effect 0:
phase = p0
effect = e0

effect 1:
phase = p4
effect = e1
```

---

# 22. Stage 1 数据输入建议

CTE Stage 1 数据逻辑（严格复用 Zeva frame tensor 接口；上游输入由合同决定）：

这里的“9 boundary RGB frames”指数据集的原始/中间表示；送入 CTE 前，
`rgb_frame` 合同直接使用 `C=3`，`wan_vae_latent` 合同必须先经冻结外部
adapter 变成真实的 `C_latent`。不要把下方的 RGB 数据形状误读成 latent
checkpoint 的输入形状。

```text
FastWAM RoboTwin Dataset
        ↓
9 boundary RGB frames

32 actions
        ↓
reshape
        ↓
8 × 4 × 14
```

最终：

```text
frames:
[B, 9, C, H', W']

其中 `C=3` 仅对应 `rgb_frame`；`wan_vae_latent` 必须使用冻结 adapter
输出的实际 `C_latent` 和空间尺寸。

transition_actions:
[B, 8, 4, 14]

valid_mask:
[B, 9]

transition_valid:
[B, 8, 4]
```

---

# 23. Stage 1 训练模块

只训练：

```text
CausalTransitionEncoder
```

Stage 1 只训练 CTE；如果 checkpoint 采用 `wan_vae_latent`，Wan VAE 仅作为
冻结的外部数据编码器，不参与反向传播。然后直接：

```python
outputs = cte(
    frames,
    transition_actions,
    valid_mask,
    transition_valid,
)

losses = causal_transition_encoder_loss(...)
```

---

# 24. Stage 2 仍维持 Frozen FastWAM

CTE 改回 Zeva 官方后，Stage 2 总原则不变；正式 cache/运行时必须传入与
Stage 1 相同的 CTE tensor 合同（RGB frame 或 Wan-VAE latent）。

冻结：

```text
FastWAM
CTE
```

训练：

```text
CausalPromptEncoder
BehaviorPrefixAdapter
gate
```

loss：

\[
L_{Stage2}=L_{FastWAM-action}
\]

不要额外加入：

```text
video loss
joint loss
IDM loss
```

---

# 25. 配置修改建议

`configs/model/zeva_fastwam.yaml`：

```yaml
zeva:
  enabled: true

  cte:
    action_dim: 14
    hidden_dim: 256
    retrieval_dim: 128
    phase_dim: 128
    effect_dim: 128

    transition_steps: 4
    effect_window_transitions: 4

    effect_target_grid: [4, 6]

    num_layers: 4
    num_heads: 8

    ema_decay: 0.996
    use_mamba: false

    # Production Zeva/FastWAM contract: frozen Wan-VAE38 output.
    image_channels: 48
    input_type: wan_vae_latent
    # For an isolated CPU/debug run, override both to image_channels=3 and
    # input_type=rgb_frame; never mix the resulting checkpoints or caches.

  memory:
    bit_size: 4
    pim_top_k: 4
```

删除：

```yaml
transition_effect_weight:
transition_effect_*:
rgb_random_effect:
```

如果有。

---

# 26. 必须增加的 tests

## 26.1 Zeva CTE tensor contract test

```text
RGB frame（仅 rgb_frame 合同）或 Wan latent tensor（wan_vae_latent 合同由外部 adapter 产生）
→ Original Zeva CTE
```

检查：

```python
assert frames.ndim == 5
assert frames.shape[2] == cte.cfg.image_channels
outputs = cte(frames, transition_actions)
```

## 26.2 32 actions → 8 transitions → 2 effects

```python
assert transition_actions.shape == (B, 8, 4, 14)
assert outputs["phase"].shape[1] == 9
assert outputs["effect_post"].shape[1] == 2
assert outputs["effect_complete"].all()
```

带 padding 的输入仍可能保持 `effect_post.shape[1] == 2`，此时必须检查
`effect_complete`，不能仅凭张量长度把无效槽位写入 memory。

## 26.3 right shift test

修改 transition `T_k`：

- 不应该影响 `phase[k]`
- 可以影响之后状态

## 26.4 effect cadence test

执行不足 4 transitions 时：

```python
assert not bool(effect_complete.any())  # 也可能是 shape [B,0]
```

完成第 4 transition 后：

```python
effect_complete == True
```

## 26.5 PIM pair test

确保 commit：

```text
pending phase at effect-window start
+
observed effect_post
```

不是：

```text
phase_post + effect
```

## 26.6 full-history deterministic test

相同：

```text
CTE tensor history（RGB 合同直接使用；latent 合同先经外部冻结 Wan-VAE adapter）
executed actions
mask
```

CTE 输出必须 deterministic。

## 26.7 official fixed-target test

确保正式 CTE 只存在 Zeva 官方固定 target：

```text
_FrozenVAEDeltaTarget
（不额外调用 VAE）
```

## 26.8 no-transition-effect test

确保正式代码中不存在：

```text
transition_effect_predictor
transition_effect_head
transition_effect_weight
transition_observed_nce
```

本轮已有 `tests/zeva/` 覆盖 tensor 对齐、effect cadence、right shift、mask、
VAE adapter、BIT/PIM lifecycle、cache schema 和 gate 等行为；26.7/26.8 更适合
作为源码静态审计（当前 `rg` 未发现这些正式字段），不必为了形式再复制一套
测试实现。当前正式 CTE 只有 `_FrozenVAEDeltaTarget`，VAE encode 调用只存在于
`vae_adapter.py`，与该边界一致。真实 VAE 输出尺寸、held-out effect/action
shuffle 与 rollout 仍属于第 0.1 节的 empirical gate。

---

# 27. Codex 修改顺序（历史记录；后续只处理未完成 gate）

下面顺序记录本轮合理修改的依赖关系。当前工作树已完成 Step 1–8 的代码落地；
后续不应机械重复执行，只有发现回归或上游 Zeva 语义变化时才按依赖关系复核。

## Step 1：迁入官方 CTE

用 Zeva 官方：

```text
causal_transition_encoder.py
cte_losses.py
```

覆盖当前 ICL-WAM 主实现。

只做：

```text
action_dim
frame input shape
padding/data-interface
```

必要适配。

## Step 2：确认 frame 输入与 checkpoint 元数据

确认外部 VAE adapter 与 CTE checkpoint/cache manifest 写入：

```text
image_channels=实际 CTE 输入通道数
cte_input_type=rgb_frame 或 wan_vae_latent
```

## Step 3：修改 Stage 1

训练输入保持 Zeva 的 `[B,T,C,H,W]` CTE tensor 接口；latent 合同由冻结
外部 Wan VAE adapter 生成。

## Step 4：修改 cache builder

改成：

```text
9 frame states
8 transitions
2 effect windows
```

## Step 5：修改 runtime CTE

取消正式路径：

```text
initialize/update
```

改成：

```text
cached frame history
+
executed action history
→ full CTE forward
```

## Step 6：修改 BIT/PIM cadence

```text
phase:
every 4 actions

effect:
every 16 actions

BIT/PIM:
only completed effect_post
```

## Step 7：修改 PIM phase pairing

改为：

```text
pending phase + observed effect_post
```

## Step 8：修改 Stage 2 dataset

只从 completed effect windows 构造 memory。

## Step 9：运行完整 tests

本地单元/静态 tests 已通过；真实 Stage 1 正式训练前仍须完成第 0.1 节列出的
empirical gates。

---

# 28. 当前暂时不要做的增强

在完成严格 Zeva→FastWAM V1 前，不要同时加入：

```text
10-token memory prompt
action-token queried adapter
layer-wise memory cross-attention
memory rank loss
Wan expert feature teacher
new CTE auxiliary loss
third memory expert
```

这些都可以作为后续 ICL-WAM V2。

第一版先保持：

\[
\boxed{
\text{Zeva mechanism unchanged}
+
\text{FastWAM backbone substitution}
}
\]

---

# 29. 最终目标代码结构

```text
src/fastwam/zeva/
├── causal_transition_encoder.py
├── cte_losses.py
├── memory.py                  # BIT/PIM 实现（合并模块）
├── lifecycle.py
├── causal_prompt.py
├── behavior_prefix_adapter.py
├── retrieval.py
├── vae_adapter.py             # CTE 外部 RGB→Wan latent adapter
└── checkpoint.py
```

当前仓库将 BIT/PIM 放在合并的 `memory.py` 中，而不是拆成两个文件；这是文件
组织差异，不是机制差异。`vae_adapter.py` 是本轮边界核对后保留的显式上游
适配层。

FastWAM 本体尽量只保留：

```text
attach_zeva_addon(...)
memory-conditioned action denoise entry
```

不要把 CTE 算法逻辑重新揉进：

```text
fastwam.py
mot.py
action_dit.py
```

---

# 30. 最终方法定义

完成上述修改后，方法可以统一表述为：

\[
o_{0:t}^{\mathrm{RGB}}
\xrightarrow{\text{optional frozen Wan-VAE adapter}}
z_{0:t}^{\mathrm{CTE}}
\]

\[
(z_{0:t}^{\mathrm{CTE}},a_{0:t-1})
\xrightarrow{\text{Original Zeva CTE}}
(p_t,e_{1:k})
\]

\[
e_{1:k}
\rightarrow
BIT/PIM
\]

\[
p_t
\rightarrow
Phase\ Retrieval
\]

\[
(g,p_t,BIT,PIM)
\rightarrow
CausalPrompt
\]

\[
h'
=
h
+
\tanh(\alpha)P(M_t)
\]

\[
a_t
=
FrozenFastWAM(o_t,h')
\]

最终应达到：

\[
\boxed{
\text{Original Zeva CTE}
+
\text{Original Zeva Memory Mechanism}
+
\text{Frozen FastWAM}
}
\]

而不是：

\[
\text{Zeva-inspired memory + custom CTE + FastWAM}
\]

---

# 31. 验收标准

本节保留原验收标准，但在本轮核对中补充状态。`[x]` 表示已有代码/单元
测试证据，`[ ]` 表示尚待完成的合同修订或需要真实模型、数据、仿真环境的
empirical gate：

- [x] CTE 源码与 Zeva 原实现语义一致，公开接口保持 `[B,T,C,H,W]`
- [x] RGB/latent 由显式外部 adapter 选择并写入合同 metadata
- [ ] cache manifest 对 VAE metadata 仅比较语义身份，允许 host-local `vae_path`
- [x] 正式 CTE 不新增 Zeva 之外的 effect target（保留 `_FrozenVAEDeltaTarget`）
- [x] `transition_steps=4`
- [x] `effect_window_transitions=4`
- [x] 32 actions 形成 8 transitions
- [x] 32 actions 在完整窗口中形成 2 个 completed effects
- [x] `transition_valid=[B,8,4]`
- [x] right-shift action stream 保留
- [x] transition-level effect 分支删除（provenance/兼容字段除外）
- [x] CTE loss 恢复 Zeva effect-v3
- [x] BIT 只写 completed `effect_post`
- [x] PIM 使用 pending phase + `effect_post`
- [x] formal runtime 使用 full-history CTE
- [x] Stage 1 只训练 CTE；若使用 latent，Wan VAE 仅冻结编码、不反传
- [x] FastWAM / CTE Stage 2 均冻结（只训练 addon whitelist）
- [x] Joint/IDM/optional-IDM 文件未改
- [ ] base FastWAM 原路径可独立运行（本环境无真实 checkpoint/CUDA）
- [x] Zeva memory path 可以通过配置完全关闭
- [x] 所有新增本地 tests 通过（31 passed）

未勾选项中，VAE metadata 路径可移植性是一处尚待处理的小合同修订；其余项
受限于当前环境缺少真实 FastWAM checkpoint、RoboTwin 数据和 CUDA/SAPIEN。
详见第 0.1 节与实现日志的 Remaining empirical gates。
