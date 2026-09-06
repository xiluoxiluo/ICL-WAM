# ICL-WAM 当前版本剩余修改项（合理性审查版）

> 适用仓库：`https://github.com/xiluoxiluo/ICL-WAM`  
> 基于当前检查版本：`fd7c6345cec93f77e99fa7a73db23f1e3b2eca60`  
> 目标：在当前已完成的 Zeva CTE 对齐基础上，修复剩余 correctness 问题，量化训练/推理 memory 分布差异，并在证据支持时缩小该差异，使得：
>
> \[
> \boxed{
> \text{Zeva CTE + BIT/PIM + Phase Retrieval + Causal Conditioning}
> +
> \text{Frozen FastWAM}
> }
> \]
>
> 能够稳定用于 RoboTwin repeated-attempt 实验。

> **本版说明（2026-09-06）**：以下条目已经按“是否影响 correctness、是否有可证伪的验收标准、实现成本及论文必要性”重新分级。`P0 / 必须` 仅用于不修复就会改变实验含义或造成状态丢失的问题；`P1 / 应做` 用于降低分布偏移、提高可复现性或提供关键证据；`P2 / 研究建议` 不作为当前版本的发布阻塞项，须以消融实验结果决定。任何“建议”都不能在没有基线、容错和回归测试的情况下升级为强制改动。

## 0.1 合理性判定原则

1. **先区分事实性错误与性能假设。** 失败 attempt 丢失已完成 effect 属于 correctness；“训练和部署使用同一分辨率一定更好”以及“加入 retry rollout 一定提升 CSR”属于待验证假设，不能仅凭直觉作为硬性验收条件。
2. **要求状态转换可追踪且幂等。** 每个 attempt 的 terminal finalize 逻辑效果必须只生效一次；正常终止、超时和异常终止要区分处理；不能为了凑齐 4-action transition 伪造 after-frame。
3. **要求实验可证伪。** 关键主张必须有固定随机性、正/负对照、容差和失败诊断；单次 `action diff > 0` 不足以证明 memory 内容真正被使用。
4. **要求先保持可运行基线。** V1 先保证 Vanilla FastWAM、Zeva shadow 和当前 addon 三条路径可独立运行，再引入昂贵的 retry fine-tuning；新路径必须能关闭并回退到已有路径。
5. **论文表述必须与实现边界一致。** 只有 CTE、BIT/PIM、phase retrieval 和 causal prompt 语义保持不变时，才可称为 Zeva unchanged；当前 4-token prompt + `BehaviorPrefixAdapter` 应明确标为 FastWAM-specific adaptation。

## 0.2 审查结论摘要

| 条目 | 合理性结论 | 修订后的要求 |
| --- | --- | --- |
| 失败 attempt terminal finalize | **P0，必须修复** | 对有可靠 terminal observation 的正常终止提交完整 transition；保证幂等；无可靠 after-frame 时只清理/提交已有 pending effect，不伪造 transition。 |
| CTE 与 FastWAM 输入尺寸统一 | **P1，应做但不是绝对定理** | 建立显式 `(H,W)` 合同并写入 metadata；默认与 RoboTwin mosaic 一致，同时允许经实验验证的例外，不得静默混用。 |
| `Memory_A != Memory_B → Action_A != Action_B` | **P1 核心证据，测试形式需修订** | 同时验证 memory/prompt 确实不同、gate 非零、骨干冻结，并在多组受控扰动下报告 action delta；不把数学上的“必然不同”写成无条件不变量。它不是运行时 correctness bug，但在发布机制性结论前必须提供。 |
| Stage 2b retry rollout | **P2，研究建议** | 先完成 Stage 2a 与基线；仅在确定性、数据来源和计算预算满足时实施，并与 Stage 2a 做 held-out 消融。 |
| 严格 Zeva policy injection | **当前不阻塞** | V1 采用路线 B，准确披露 adapter；路线 A 作为后续研究，不作为当前结果前置条件。 |
| attempt-wise 日志 | **P1，应做** | 记录定义明确、可聚合、可复现的 attempt 与 memory 字段；不要只记录成功样本或用未定义的均值替代空值。 |

## 0.3 本次要求调整

- 将“失败 attempt finalize”保留为唯一明确的 P0 correctness 修复，并增加 terminal observation 可信性与幂等要求；
- 将 VAE 尺寸统一从“必然提升性能”改为“默认合同 + 显式兼容检查 + 统计验证”；
- 将 memory→action 从逻辑蕴含式要求改为受控、多 seed、带负对照的机制证据；
- 将 Stage 2b 从默认必做改为有触发条件的 P2 分支，避免在没有收益证据时扩大 rollout 和训练成本；
- 将 CSR 提升、improvement 随 attempt 增加等内容明确为研究假设，不作为代码正确性的验收门槛。

---

# 0. 当前已经确认正确的部分

以下部分暂时不要再大改：

- CTE 输入已经改为 FastWAM Wan VAE latent；
- RoboTwin `action_dim=14`；
- `transition_steps=4`；
- `effect_window_transitions=4`；
- 32 actions → 8 transitions；
- 32 actions → 2 completed effects；
- `transition_valid=[B,8,4]`；
- CTE right-shift causal action stream；
- transition-level effect branch 已删除；
- CTE loss 已恢复为 Zeva effect-v3；
- 正式 runtime 已使用 full-history CTE；
- BIT 只接收 completed effect；
- PIM 使用 effect-window 起点 phase 与 `effect_post`；
- PIM 已采用 attempt-end commit；
- memory 已经真实进入 FastWAM action hidden path；
- FastWAM backbone 在 Stage 2 冻结；
- Joint / IDM / optional IDM 均未参与。

本轮修改不要重新设计这些部分。

---

# 1. P0：失败 Attempt 也必须 Finalize Terminal Transition（必须修复）

## 1.1 当前问题

当前 RoboTwin evaluator 仍然只在成功时调用：

```python
if succ and hasattr(model, "finalize_attempt"):
    model.finalize_attempt(TASK_ENV)
```

这会造成失败 attempt 的最后一组已执行动作可能没有 after-frame 被消费。该问题会直接改变 CTE history 和后续 PIM，属于 correctness，而不是普通性能优化。

例如：

```text
Attempt 0

...
a28
a29
a30
a31
↓
environment 已经到达 terminal / step limit
↓
attempt failure
↓
while loop 直接退出
↓
没有下一次 model.step()
↓
最后 4-action transition 没有 append 到 CTE history
```

如果这一组动作恰好完成一个 16-action effect window，则最重要的失败 effect 可能被丢弃。

---

## 1.2 涉及文件（按实际调用链核对）

```text
third_party/RoboTwin/script/eval_policy.py
experiments/robotwin/fastwam_policy/deploy_policy.py
```

---

## 1.3 evaluator 修改

当前：

```python
if succ and hasattr(model, "finalize_attempt"):
    model.finalize_attempt(TASK_ENV)
```

修改为（对 success、failure、timeout 和 step limit 的**正常结束**均调用）：

```python
if hasattr(model, "finalize_attempt"):
    model.finalize_attempt(
        TASK_ENV,
        success=succ,
    )
```

`finalize_attempt` 必须可安全重复调用（幂等）。若进程级异常或环境尚未提供可信 observation，不应强行伪造 terminal frame；但正常结束时仍需清理 transient state，并提交已经观察到的 pending effect。

---

## 1.4 Policy 接口修改

当前：

```python
def finalize_attempt(self, task_env) -> None:
```

修改为：

```python
def finalize_attempt(
    self,
    task_env,
    success: bool = False,
) -> None:
```

内部流程（terminal observation 可信时）：

```text
terminal observation
↓
检查是否残留完整 4-action group
↓
append_transition()
↓
full-history CTE forward
↓
读取新的 completed effect_post
↓
BIT / pending PIM buffer
↓
end_attempt()（幂等）
```

---

## 1.5 推荐实现（伪代码，需按生命周期实现幂等）

```python
def finalize_attempt(self, task_env, success: bool = False) -> None:
    if self._attempt_finalized:
        return

    if self.zeva_mode == "base":
        self._transition_actions.clear()
        self._attempt_finalized = True
        return

    if self.cte is None or self.lifecycle is None:
        self._transition_actions.clear()
        self._attempt_finalized = True
        return

    if self._cte_history is None:
        self._transition_actions.clear()
        self.lifecycle.end_attempt()
        self._attempt_finalized = True
        return

    terminal_obs = getattr(task_env, "now_obs", None)

    if (
        isinstance(terminal_obs, dict)
        and "observation" in terminal_obs
        and len(self._transition_actions) == self.cte.cfg.transition_steps
    ):
        next_image = self._build_robotwin_image_tensor(
            terminal_obs
        )[0].float()

        action_group = torch.stack(
            self._transition_actions
        ).to(self.model.device)

        self._cte_history.append_transition(
            action_group,
            next_image,
        )

        encoded = self._cte_history.forward()

        complete = encoded["effect_complete"][0]

        while self._observed_effect_count < complete.shape[0]:
            effect_index = self._observed_effect_count

            if bool(complete[effect_index]):
                start = (
                    effect_index
                    * self.cte.cfg.effect_window_transitions
                )

                self.lifecycle.observe_completed_effect(
                    encoded["phase"][0, start],
                    encoded["effect_post"][0, effect_index],
                    metadata={
                        "source_step":
                            start
                            * self.cte.cfg.transition_steps,
                        "terminal_attempt": True,
                        "attempt_success": bool(success),
                    },
                )

            self._observed_effect_count += 1

    self._transition_actions.clear()
    self.lifecycle.end_attempt()
    self._attempt_finalized = True
```

实现时不要把“`terminal_obs` 存在”直接等同于“after-frame 一定有效”；应沿用环境已有的 observation 时间语义，并记录 `terminal_observation_used` 供审计。

`_attempt_finalized` 应在 `begin_attempt()` 时重置；若 evaluator 可能在 finalize 后继续复用同一 policy 对象，必须把该标志纳入生命周期状态，而不是依赖调用方自觉避免重复调用。生产实现不宜只靠一个“先置位再执行”的布尔值：应使用 attempt id/transition id 或事务式提交，避免 finalize 中途异常后既不能重试又可能留下半提交状态。

---

## 1.6 必须增加测试

新增：

```text
tests/zeva/test_terminal_attempt_commit.py
```

至少测试：

### Case A：success terminal transition

```text
最后 4 actions
→ finalize_attempt(success=True)
→ effect 被 commit
```

### Case B：failed terminal transition

```text
最后 4 actions
→ finalize_attempt(success=False)
→ effect 同样被 commit
```

### Case C：不足 4 actions

```text
len(_transition_actions) < 4
→ 不伪造 transition
→ 仅结束 attempt
```

### Case D：重复 finalize

```text
连续调用两次 finalize_attempt()
→ transition、effect 和 PIM 均不重复
```

验收：

```python
assert failed_attempt_effect_is_not_lost
assert finalize_attempt_is_idempotent
```

---

# 2. P1：CTE Wan-VAE 输入尺寸与 FastWAM RoboTwin 输入保持一致

## 2.1 当前问题

FastWAM RoboTwin 当前最终 video input 是：

```yaml
video_size: [384, 320]
```

但 `FastWAMCTELatentEncoder` 默认：

```python
resize=(480, 832)
```

当前实际上是：

```text
FastWAM policy path:
RoboTwin mosaic
→ 384×320
→ Wan VAE

CTE path:
RoboTwin mosaic
→ 480×832
→ Wan VAE
→ CTE
```

虽然仍是同一个 Wan VAE，但两边 latent distribution 不完全一致。统一尺寸可以减少一个明确的分布偏移，具有合理性；但它不保证下游指标必然提升，仍需用 held-out latent 统计和回归实验确认。

---

## 2.2 推荐目标（P1，而非无条件性能结论）

优先采用并验证：

\[
\boxed{
CTE\ VAE\ input\ size
=
FastWAM\ RoboTwin\ input\ size
=
384\times320
}
\]

即：

```text
same RGB mosaic
↓
same resize
↓
same Wan VAE
↓
FastWAM / CTE
```

这里的“一致”指送入 Wan VAE 前的实际 tensor（尺寸、数值范围、通道排列和插值方式）一致，不只是配置字段相同。实现时需确认 FastWAM helper 不会再次隐式 resize 或采用不同的 antialias/归一化；否则应把完整预处理签名写入 metadata，而不是只记录 `(H,W)`。

---

## 2.3 需要修改

主要文件：

```text
src/fastwam/zeva/vae_adapter.py
scripts/train_zeva_cte.py
scripts/build_zeva_robotwin_cache.py
experiments/robotwin/fastwam_policy/deploy_policy.py
src/fastwam/zeva/checkpoint.py
src/fastwam/zeva/schemas.py
```

---

## 2.4 不要硬编码，并明确尺寸语义

不要：

```python
resize=(384, 320)
```

直接散落在多个文件。

建议从单一配置源读取：

```yaml
data.train.video_size
```

获取。

注意 `video_size` 的语义与 torchvision resize 的 `(H,W)` / `(W,H)` 约定必须统一。

推荐新增：

```yaml
zeva:
  cte:
    vae_input_size: ${data.train.video_size}
```

或者在构造 adapter 时显式传：

```python
resize=tuple(cfg.data.train.video_size)
```

---

## 2.5 Checkpoint / Cache Manifest 必须记录输入尺寸

CTE checkpoint metadata 增加：

```python
"cte_vae_input_size": [384, 320]
```

Cache manifest 增加：

```python
cte_vae_input_size: tuple[int, int]
```

Stage 2 和 eval 都进行 strict compatibility check。检查失败应显式报错；只有在 checkpoint 明确声明兼容且经过离线验证时，才允许使用不同尺寸，不能静默放行。

禁止：

```text
Stage1: 480×832
Cache: 384×320
Eval: 480×832
```

这种静默混用。

---

## 2.6 必须增加测试

```text
tests/zeva/test_vae_input_contract.py
```

验证（同时覆盖默认值缺失和 `(H,W)` / `(W,H)` 误置）：

```python
assert checkpoint_size == cache_size
assert cache_size == eval_size
assert cte_size == robotwin_video_size
```

验收重点是“不会静默混用”，不是把固定的 `384×320` 数字写死在测试里。

---

# 3. P1：证明 Memory 内容能够影响 Action（核心证据，非数学必然性）

## 3.1 当前测试不足

当前已有测试可以证明：

```text
gate = 0
→ FastWAM == Zeva shadow

gate != 0
→ addon 可以改变 action
```

但这还不能证明：

\[
Memory_A \neq Memory_B
\Longrightarrow\text{在受控条件下，Action 分布/预测会发生可重复变化}
\]

因为当前测试可以通过人为设置：

```python
output.bias.fill_(1.0)
```

让 residual 恒为非零。

这种测试证明的是：

> addon branch 可以改变 action。

但不是：

> memory content 可以改变 action。

---

## 3.2 必须新增测试

建议：

```text
tests/zeva/test_memory_content_action_effect.py
```

测试固定：

```text
same observation
same action noise
same timestep
same task context
same current phase
same gate
same FastWAM weights
same video KV cache
```

只改变：

```text
PIM_A / PIM_B
```

测试前必须断言 `memory_a != memory_b`（以及 mask 的语义正确），并确认 gate 非零、adapter 输出非零且 FastWAM backbone 无梯度。不要把“任意两个不同 memory 都必然产生不同 action”作为模型不变量；饱和、对称或被 mask 的 memory 可能合法地产生相同输出。

---

## 3.3 推荐测试形式

```python
def test_different_pim_content_changes_action_prediction():
    torch.manual_seed(0)

    model = make_test_model()

    # 保证 adapter 已经不是 zero-output 状态
    with torch.no_grad():
        model.zeva_behavior_prefix_adapter.output.weight.normal_(
            mean=0.0,
            std=0.01,
        )
        model.zeva_behavior_prefix_adapter.pim_gate.fill_(0.5)

    task = ...
    phase = ...
    bit = ...
    bit_mask = ...

    pim_phase_a = ...
    pim_effect_a = ...

    pim_phase_b = pim_phase_a.clone()
    pim_effect_b = -pim_effect_a

    memory_a, mask_a = model.zeva_prompt_encoder(
        task,
        phase,
        bit,
        bit_mask,
        pim_phase_a,
        pim_effect_a,
        pim_mask,
    )

    memory_b, mask_b = model.zeva_prompt_encoder(
        task,
        phase,
        bit,
        bit_mask,
        pim_phase_b,
        pim_effect_b,
        pim_mask,
    )

    action_a = model._denoise_action_with_video_cache_zeva(
        ...,
        behavior_memory=memory_a,
        behavior_memory_mask=mask_a,
    )

    action_b = model._denoise_action_with_video_cache_zeva(
        ...,
        behavior_memory=memory_b,
        behavior_memory_mask=mask_b,
    )

    assert torch.norm(memory_a - memory_b) > 1e-6
    diff = torch.norm(action_a - action_b)
    assert torch.isfinite(diff)
    assert diff > 1e-5
```

更稳妥的验收是：对多组固定 seed 和多种 PIM 扰动重复测试，并报告 `action_delta` 的中位数/最小值；加入 `gate=0` 的负对照（此时 action 应回到 Vanilla FastWAM）。若个别扰动不改变 action，应检查 mask、gate 和 adapter 梯度，而不是简单放宽阈值。

---

## 3.4 再增加一个 Retrieval Test

最好再验证：

```text
phase query A
→ retrieve PIM A

phase query B
→ retrieve PIM B
```

之后：

```text
retrieval result different
→ causal prompt different
→ action different
```

形成完整链（作为证据链，而非无条件保证）：

\[
\boxed{
Phase
\rightarrow
PIM\ Retrieval
\rightarrow
CausalPrompt
\rightarrow
Action
}
\]

---

# 4. P1：Stage 2 PIM 训练分布差异需要量化，不宜直接宣称“必须对齐”

## 4.1 当前状态

当前 Stage 2：

```text
same task
+
other episode
→ offline PIM proxy
```

也就是：

\[
PIM_{train}
=
\text{other episodes of same task}
\]

在线部署：

```text
same episode
+
previous attempts
→ PIM
```

即：

\[
PIM_{test}
=
\text{same episode previous attempts}
\]

所以当前存在潜在的 train-test gap：

\[
\boxed{
PIM_{train}
\neq
PIM_{test}
}
\]

这不是代码 correctness bug，也不能仅由数据来源不同推出性能一定下降。应先比较两种 memory 的长度、失败/成功比例、phase 覆盖和 effect 统计；只有 gap 与 CSR 下降存在可重复关联时，才需要增加专门的 retry 数据。

最低要求：在 Stage 2a 报告 memory proxy 与在线 memory 的统计差异，并在 held-out task/seed 上保留无 memory、Stage 2a memory 两个基线。

---

# 5. P2：可选 Stage 2b——Retry Rollout Memory Fine-Tuning

## 5.1 保留当前 Stage 2a

当前 demo-based offline proxy 不要删。Stage 2b 是在证实分布差异影响性能、且具备可接受 rollout 成本后再启用的研究分支，不是 V1 发布阻塞项。

定义：

```text
Stage 2a
success demonstration
+
same-task other-episode memory proxy
→ train CausalPrompt / Adapter / Gate
```

作用：

- 提供大量稳定训练样本；
- 先让 addon 学会读取 phase/effect memory；
- 不需要额外 rollout。

---

## 5.2 新增 Stage 2b（触发条件与安全边界）

仅在以下条件同时满足时，使用 Frozen FastWAM 收集 fixed-seed retry rollout：

- Stage 2a 与 no-memory 基线的差异已被日志和 held-out 结果确认；
- 训练/评估 seed、任务划分和 rollout 预算已预先固定；
- 失败 action 只作为 memory context，不进入 BC target；
- Stage 2b 可通过配置关闭，并保留 Stage 2a checkpoint 作为回退。

新增：

```text
scripts/collect_zeva_robotwin_retry_data.py
scripts/train_zeva_fastwam_retry.py
```

数据协议：

```text
Task + Seed S

Attempt 0
↓
Frozen FastWAM rollout
↓
fail / success
↓
CTE extracts completed effects
↓
commit to PIM

reset same task + same seed

Attempt 1
↓
PIM = Attempt 0 memory

Attempt 2
↓
PIM = Attempt 0 + Attempt 1
```

---

## 5.3 Retry Dataset 推荐字段

每个 training sample：

```python
{
    "task_id": ...,
    "task_name": ...,
    "instruction": ...,

    "seed": ...,
    "attempt_id": ...,

    "observation": ...,
    "proprio": ...,

    "phase": [128],

    "bit_effects": [4,128],
    "bit_mask": [4],

    "pim_phases": [4,128],
    "pim_effects": [4,128],
    "pim_mask": [4],

    "target_action": [32,14],

    "source_attempt_ids": [...],

    "success_before": ...,
    "success_after": ...,
}
```

---

## 5.4 最重要的监督规则

失败 rollout 的 action **不能直接作为 behavior cloning target**。

也就是说不能：

```text
failed action
→ target action
```

正确做法（并在数据校验脚本中强制检查）：

```text
failed previous attempt
→ memory context

current target
→ expert / successful demonstration action
```

或者：

```text
failed transitions
→ 只用于 CTE/PIM
```

不要训练 FastWAM imitate bad actions；若没有 expert/successful target，则该样本只能用于 CTE/PIM 统计，不能用于 policy loss。

---

## 5.5 推荐两阶段训练（需消融，不保证 Stage 2b 一定提升）

```text
Stage 2a
demo proxy PIM
↓
addon checkpoint

Stage 2b
same-seed previous-attempt PIM
+
successful/expert target
↓
addon fine-tuning
```

FastWAM 与 CTE 全程继续冻结。

---

# 6. P1：明确论文定位——是否要求 Causal Prompt / Policy Injection 与 Zeva 完全一致

## 6.1 当前实现

当前 ICL-WAM 的：

```text
CausalPromptEncoder
```

输出：

```text
4 memory tokens
```

大致为：

```text
[fused task/phase]
[current phase]
[BIT summary]
[PIM summary]
```

之后：

```text
BehaviorPrefixAdapter
```

用：

```text
32 learnable action queries
```

cross-attend memory token，并得到：

```text
[B,32,1024]
```

residual 加入 FastWAM action hidden。

---

## 6.2 与 Zeva 官方差异

Zeva 官方 Causal Prompt 更接近：

\[
M_t\in\mathbb{R}^{B\times256}
\]

单个 causal prompt vector。

然后：

\[
h_{base}
+
\tanh(\alpha)
P(M_t)
\]

并通过 `persistent_valid` 判断是否真正有 PIM。

因此当前 ICL-WAM 的：

```text
4-token prompt
+
32-query BehaviorPrefixAdapter
```

仍然属于：

\[
\boxed{
\text{FastWAM-specific adaptation}
}
\]

---

# 7. 两种论文路线二选一

## 路线 A：严格 Zeva Backbone Replacement

如果论文要写：

> We preserve Zeva unchanged and only replace its frozen policy backbone with FastWAM.

那么需要继续把：

```text
CausalPromptEncoder
Policy Injection
```

收敛到 Zeva 官方语义。

目标：

```text
task + phase + BIT + PIM
↓
single causal prompt [B,256]
↓
project to FastWAM policy-prefix hidden
↓
gated residual
```

问题是 FastWAM 原本没有 Zeva 的 dedicated task/behavior prefix。

因此必须找到一个明确且已有的 FastWAM policy conditioning representation 来承接：

```text
base_prefix
```

不能凭空新增一个 always-active base prefix，否则 gate=0 时不再严格等于 Vanilla FastWAM。

---

## 路线 B：Zeva Memory Mechanism Adapted to FastWAM

如果论文定位允许：

> We preserve Zeva's causal memory mechanism while adapting its policy-conditioning interface to FastWAM's ActionDiT.

那么当前：

```text
4-token causal prompt
+
BehaviorPrefixAdapter
```

可以保留。

这种情况下建议论文明确写：

```text
Zeva-aligned CTE
Zeva-aligned BIT/PIM
Zeva-aligned phase retrieval
FastWAM-specific causal action adapter
```

不要声称整个 policy injection 完全 unchanged。

---

# 8. 当前 V1 采用路线 B（论文与实现边界）

为了尽快获得 RoboTwin 结果，当前 V1 暂时：

```text
保留 BehaviorPrefixAdapter
```

不要现在再大改 policy injection。

原因：

1. memory → action 链已经接通；
2. gate=0 可以严格回退 Vanilla FastWAM；
3. 不需要修改 `mot.py` / ActionDiT 30 层结构；
4. 容易做 baseline / ablation；
5. 可以先验证 repeated-attempt 是否真的有收益。

论文表述改成：

> We retain Zeva's causal transition representation, dual-timescale interaction memory, phase-conditioned retrieval, and causal prompt construction, while introducing a lightweight gated action-conditioning adapter to interface these memories with a frozen FastWAM backbone.

等第一版有收益，再考虑做严格 Zeva prefix mapping。这里的“gate=0 回退”必须由数值容差测试定义（例如 `allclose` 的 `rtol/atol`），不能写成绝对的逐 bit 相等；若未来改为路线 A，应单独记录接口映射、参数量和新的基线。

---

# 9. P1：增加 Attempt-wise Evaluation Log

当前最终实验必须输出（并给出定义、分母和置信区间）：

```text
SR@1
CSR@2
CSR@3
CSR@4
```

建议固定定义为：`SR@1` = Attempt 0 首次成功率；`CSR@k` = 同一 task/seed 在前 `k` 次 attempt 内至少成功一次的比例。`CSR@k` 应使用同一批 seed 的累计指标，不能把各 attempt 的独立成功率相加，也不能因提前成功而继续产生虚拟 attempt。

并记录 memory 状态。

建议每个 seed 写：

```json
{
  "task": "stack_blocks",
  "seed": 1001,
  "attempts": [
    {
      "attempt_id": 0,
      "success": false,
      "bit_count_before": 0,
      "pim_size_before": 0,
      "retrieval_count": 0
    },
    {
      "attempt_id": 1,
      "success": false,
      "bit_count_before": 0,
      "pim_size_before": 2,
      "mean_pim_score": 0.77
    },
    {
      "attempt_id": 2,
      "success": true,
      "pim_size_before": 4,
      "mean_pim_score": 0.82
    }
  ]
}
```

最终汇总：

```text
FastWAM:
SR@1
CSR@2
CSR@3
CSR@4

ICL-WAM:
SR@1
CSR@2
CSR@3
CSR@4
```

重点验证：

\[
CSR@k_{\text{ICL-WAM}}
>
CSR@k_{\text{FastWAM}}
\]

这是待验证的研究假设，不是代码验收条件。应使用相同 task/seed 的 paired comparison，报告点估计及置信区间，并允许结果无提升或退化；同时观察 improvement 是否随 attempt 增加，而不是只看单次成功率。

---

# 10. 经合理性审查后的修改顺序

不要同时大改所有模块。

## Step 1 — P0：先修 correctness

修：

```text
failed attempt terminal finalize
```

必须保证失败经验不会丢，并补充幂等、无可信 terminal observation、异常退出边界测试。

---

## Step 2 — P1：建立输入合同

统一：

```text
CTE Wan VAE input size
=
FastWAM RoboTwin video size
```

并加入 metadata strict check；先验证训练、cache、eval 的 `(H,W)` 语义一致，再比较统一尺寸前后的 latent 统计。

---

## Step 3 — P1：建立 memory→action 证据（发布机制性结论前必须）

增加：

```text
Memory A/B 受控扰动
→ prompt/memory 可观测差异
→ action delta（含 gate=0 负对照）
```

integration test。

---

## Step 4

跑完整：

```bash
pytest tests/zeva -q
```

并至少做一个 GPU smoke test：

```text
Stage1 forward/backward
cache build
Stage2 one-step backward
base inference
pim_shadow inference
pim_on inference
```

---

## Step 5 — P1：完成 Stage 2a 基线

先跑当前 Stage2a offline proxy。

检查：

```text
gate 是否离开 0
adapter output norm
prompt gradient
memory-conditioned action delta
```

---

## Step 6 — P2：按触发条件决定是否实施 Stage 2b

若第 4 节的 train-test gap 与性能下降得到重复证据，再采集 fixed-seed retry rollout；否则保留 Stage 2a，不为“可能有收益”承担不必要的 rollout 成本。

新增：

```text
Stage2b previous-attempt memory fine-tuning（可关闭、需消融）
```

---

## Step 7

正式 RoboTwin repeated-attempt evaluation。

输出：

```text
SR@1
CSR@2
CSR@3
CSR@4
```

---

# 11. 验收 Checklist（按优先级）

## Correctness

- [ ] P0：success attempt 在有可信 terminal observation 时 finalize terminal transition
- [ ] P0：failed/timeout/step-limit attempt 也执行 finalize
- [ ] incomplete <4-action tail 不伪造 transition
- [ ] attempt 结束后已有 pending effect 按生命周期规则 commit 到 PIM
- [ ] finalize 重复调用不重复写入 transition/effect/PIM
- [ ] BIT 在新 attempt 清空
- [ ] PIM 在 same episode retry 保留
- [ ] PIM 在 new episode 清空

## VAE / CTE

- [ ] CTE 使用 frozen FastWAM Wan VAE
- [ ] P1：CTE VAE resize 默认与 RoboTwin FastWAM 输入一致，或有显式兼容声明
- [ ] resize 写入 CTE checkpoint metadata
- [ ] resize 写入 cache manifest
- [ ] Stage2 / eval 对 resize 做 strict check

## Memory → Action

- [ ] gate=0 时在声明的 `rtol/atol` 容差内等于 Vanilla FastWAM
- [ ] gate!=0 时 addon 可影响 action
- [ ] P1：受控 PIM 扰动可在多组 seed 下产生可重复 action delta；不要求任意 memory 对都不同
- [ ] different phase retrieval 在检索结果确实不同的样本上可以导致 different memory
- [ ] FastWAM backbone 无梯度
- [ ] CausalPrompt / Adapter / Gate 有梯度

## Stage2

- [ ] 当前 offline proxy 明确标为 Stage2a
- [ ] 不把 failed action 当 BC target
- [ ] P2：若启用 Stage2b，使用 same-seed previous-attempt memory
- [ ] 若启用 Stage2b，FastWAM / CTE 继续冻结

## Evaluation

- [ ] fixed seed retries 使用同一 task initialization，并记录环境版本
- [ ] Attempt 1 PIM 为空
- [ ] Attempt 2 能看到 Attempt 1 memory
- [ ] Attempt 3 能看到 Attempt 1/2 memory
- [ ] 输出 SR@1 / CSR@2 / CSR@3 / CSR@4
- [ ] 输出 PIM size / retrieval score / memory source；空 memory 用显式 null/0 表示
- [ ] paired seed 统计提供样本数、分母及置信区间

---

# 12. 当前最终建议

当前版本已经不需要继续大改 CTE。

本轮真正需要关注的是：

\[
\boxed{
\textbf{1. Failure terminal memory correctness}
}
\]

\[
\boxed{
\textbf{2. VAE input contract and distribution measurement}
}
\]

\[
\boxed{
\textbf{3. Controlled evidence that memory content can change action}
}
\]

\[
\boxed{
\textbf{4. Quantify, then decide whether to reduce the Stage2 train-test PIM gap}
}
\]

其中：

- 第 1 项属于必须修复的 correctness；
- 第 2 项属于表示一致性与可审计性，统一尺寸是默认方案而非性能保证；
- 第 3 项属于 memory-action 链路的证据；
- 第 4 项是研究假设，需由 paired held-out 实验决定是否投入 Stage 2b。

第一版不建议继续修改 CTE architecture，也暂时不建议加入新的 expert、layer-wise memory attention、memory ranking loss 或 Joint/IDM。完成 P0 后即可运行可复现的 V1；P1/P2 项应以日志、消融和统计不确定性为依据逐项推进。

先把当前版本跑通成：

\[
\boxed{
\text{Zeva-aligned causal memory}
+
\text{FastWAM-specific gated action adapter}
+
\text{Frozen FastWAM}
}
\]

再用 RoboTwin repeated-attempt 结果决定后续是否继续增强。
