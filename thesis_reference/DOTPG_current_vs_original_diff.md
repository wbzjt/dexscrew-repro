# 当前版本 DOTPG 与原始论文版本 DOTPG 的差别记录

Date: 2026-05-22  
Status: **用于记录代码迭代后当前 DOTPG 理论版本与原始 draft 的差异**  
Scope: 对比当前 `dual_bc5` code-faithful DOTPG 与 `DOTPG-draft.md/pdf` 中的原始理论版本。

---

## 0. Source Documents

本差异记录以如下文件为 source：

1. `dc24bacd-7d89-48c0-8ae0-bbba4174e913.md`  
   当前代码忠实版 DOTPG 框架说明。
2. `f922a724-36f2-41e8-a46a-6d29e90eea05.md`  
   CoDrive DOTPG 优化迭代记录。
3. `DOTPG-draft.md` / `DOTPG-draft.pdf`  
   原始论文版本 DOTPG 理论草稿。

---

## 1. 一句话总结

原始 DOTPG draft 的核心构想是：

\[
\text{Wasserstein occupancy matching}
+ \text{OT dual critic}
+ \text{Q-based actor-critic policy improvement}.
\]

当前代码成功版本 `dual_bc5` 的真实算法是：

\[
\boxed{
\text{teacher-initialized deterministic actor}
+ \text{empirical OT dual critic}
+ \text{direct dual actor loss}
+ \text{BC anchoring}
+ \text{TD3-style Q auxiliary}
}
\]

最重要的变化是：**当前主 actor objective 已经不是原始 draft 中的 Q-only policy improvement，而是 direct dual actor objective 加 BC anchoring。**

---

## 2. 总体差异表

| 维度 | 原始 draft 版本 | 当前 `dual_bc5` 代码版本 | 理论处理建议 |
|---|---|---|---|
| 算法定位 | OT dual + Q-network policy gradient / RLfD 框架 | deterministic off-policy teacher-student imitation / distillation | 改成 code-faithful DOTPG-Dual-BC |
| Policy 类型 | 文中写 Gaussian stochastic actor，带 maximum entropy 表述 | deterministic actor，动作 clamp 到 `[-1,1]` | 删除 SAC / MaxEnt 主线 |
| Actor architecture | generic policy network / MLP | `teacher_actor`，复用 PPO teacher `actor_mlp + mu` | 写成 policy-class prior |
| Policy initialization | 未强调 teacher initialization | `policy_init_from_teacher=True` | 写成近 teacher manifold 的初始化 |
| Dual objective | \(\max_f E_E[f]-E_\pi[f]\) | 同方向，代码中 minimize `-W_hat + lambda_GP GP` | 保留，但强调 empirical surrogate |
| Actor sign | Stage 3 写成 \(\min E[Q(s,\pi(s))]\)，伪代码又写成 gradient ascent on Q | 当前 actor minimize `-E[f]`，Q variant minimize `-E[Q]` | 修正：actor 必须 maximize student-side \(f\) 或 Q |
| Main actor objective | Q-based policy improvement | `policy_loss_mode=dual`：\(-E[f_\phi(s,\pi(s))]\) | 主算法写 direct dual |
| BC | 原始理论中不是核心组件 | BC pretrain + BC anchoring 是关键稳定项 | 提升为主方法的一部分 |
| Q-network | 原始版本中是核心 actor guidance | 当前是 TD3-style auxiliary / diagnostic；Q-only ablation 弱 | 不再把 Q-only 当主结果 |
| Q architecture | single Q + target Q | twin Q、target smoothing、policy delay、target action noise | 用 TD3-style auxiliary 描述 |
| Reward | \(f_\phi\) 作为 intrinsic reward | Q target 使用 normalized / clipped / scaled \(+f_\phi\) | 写清 \(+f\)，不是 \(-f\) |
| Expert data | static expert demonstrations | teacher PPO rollouts + online expert refresh | 写成 DAgger-like data management |
| Student state | 未与当前代码完全对齐 | `obs_norm + z_eta(proprio_history)`，adapter warmup 可 freeze | 加入 teacher-student representation section |
| Sinkhorn | draft 写了 Sinkhorn initialization | 当前代码无 Sinkhorn 初始化 | 删除或放 background |
| MaxEnt / KL theorem | draft 有 maximum entropy theorem 和 KL 推导 | 当前无 entropy alpha、无 log-prob、无 SAC actor | 删除相关 theorem |
| Theoretical guarantee | draft 声称 strong convergence、monotonic improvement、sim-to-real invariance | 当前代码无法支持这些无条件 claim | 改成 cautious propositions + limitations |
| Model selection | draft 泛称 performance improvement | 当前发现 train reward 不可靠，fixed-step deploy eval 更重要 | 实验章节应说明 selection protocol |

---

## 3. 符号方向差异

### 3.1 原始 draft 的正确起点

原始 draft 的 Wasserstein dual 起点是合理的：

\[
W_1(\rho_E,\rho_\pi)
=
\sup_{\|f\|_{Lip}\le 1}
\mathbb E_{\rho_E}[f(s,a)]
-
\mathbb E_{\rho_\pi}[f(s,a)].
\]

在这个 sign convention 下，dual critic 会把 expert samples 的 score 推高，把 student samples 的 score 推低。

### 3.2 当前代码的方向

当前代码中的 empirical dual estimate 是：

\[
\hat W_\phi
=\operatorname{expert\_score}-\operatorname{policy\_score}.
\]

Dual loss 是：

\[
\mathcal L_{dual}= -\hat W_\phi+\lambda_{GP}\mathcal L_{GP}.
\]

由于 actor 要 minimize Wasserstein discrepancy，固定 \(f_\phi\) 时必须提高 student-side score：

\[
\max_\theta \mathbb E[f_\phi(s,\pi_\theta(s))].
\]

因此 actor loss 是：

\[
\mathcal L_\pi^{dual}= -\mathbb E[f_\phi(s,\pi_\theta(s))].
\]

### 3.3 原始 draft 的错误点

原始 draft 的 Stage 3 写成：

\[
\min_\theta \mathbb E[Q_\psi(s,\pi_\theta(s))].
\]

这与 dual sign convention 和伪代码中的 gradient ascent on Q 矛盾。正确方向应是：

\[
\max_\theta \mathbb E[Q_\psi(s,\pi_\theta(s))]
\quad\Longleftrightarrow\quad
\min_\theta -\mathbb E[Q_\psi(s,\pi_\theta(s))].
\]

当前主版本甚至进一步改成：

\[
\min_\theta
-
\mathbb E[f_\phi(s,\pi_\theta(s))]
+
\alpha_{BC}\mathbb E\|\pi_\theta(s_E)-a_E\|^2.
\]

---

## 4. Actor Objective 的核心变化

### 4.1 原始版本

原始 DOTPG draft 中的 actor update 设计为：

\[
Q(s,a)\leftarrow f_\phi(s,a)+\gamma Q_{target}(s',\pi(s')),
\]

然后通过 Q 来更新 actor：

\[
\theta\leftarrow \theta+\alpha_\theta\nabla_\theta E[Q(s,\pi_\theta(s))].
\]

理论动机是：\(f_\phi\) 是 instant OT signal，Q 是 cumulative long-horizon distribution matching value。

### 4.2 当前版本

当前 selected run 使用：

```text
policy_loss_mode = dual
policy_arch = teacher_actor
policy_init_from_teacher = True
bc_coef = 5.0
bc_alpha_max = 20.0
```

主 actor loss 是：

\[
\mathcal L_\pi
= -E_{s\sim B}[f_\phi(s,\pi_\theta(s))]
+\alpha_{BC}E_{(s_E,a_E)\sim D_E}\|\pi_\theta(s_E)-a_E\|^2.
\]

该变化的理论含义是：

1. 直接使用 OT dual potential 的 local action gradient；
2. 避免 Q approximation error 在 dexterous contact task 中放大；
3. 用 BC anchor 防止 actor 离开 teacher action manifold。

### 4.3 经验原因

迭代记录显示，早期 Q-only actor objective 在 contact manipulation 中明显弱于 direct dual：

| Candidate | max train best | 解释 |
|---|---:|---|
| `teacher_actor_q` | 869.14 | Q-only actor objective 弱 |
| `teacher_actor_dual` | 2622.18 | direct dual 明显更强 |
| `teacher_actor_qdual_metric` | 1327.30 | Q/metric mixture 未解决问题 |
| `dual_bc5` | 2904.14 | selected deploy baseline |

因此，论文中应把 Q path 从“主算法核心”降级为“long-horizon auxiliary / ablation variant”。

---

## 5. Policy Architecture 和 Initialization 的变化

### 5.1 原始版本

原始 draft 将 policy network 写成通用 actor，且描述中出现了 Gaussian distribution、entropy regularization、maximum entropy consistency 等表述。

### 5.2 当前版本

当前代码版本使用 deterministic teacher actor architecture：

- 复用 PPO teacher 的 `actor_mlp + mu`；
- student actor copy 后参数可训练；
- selected output mode 为 `clamp`；
- action space 连续且 clipped to `[-1,1]`。

这不是一个小的实现细节，而是理论上很重要的 **policy-class prior**：

\[
\Pi_{student}\approx \Pi_{teacher}\text{ around expert behavior manifold}.
\]

它减少了 function approximation error，也让 dual critic 在 expert / student sample 支持附近提供更可信的 gradient。

---

## 6. BC Anchoring 从辅助项变成主稳定机制

### 6.1 原始版本

原始 draft 没有将 BC anchoring 作为主要理论组件。

### 6.2 当前版本

当前 selected run 使用 BC pretraining 和 actor update 中的 BC regularization：

\[
\mathcal L_{BC}=E_{D_E}\|\pi_\theta(s_E)-a_E\|^2.
\]

完整 actor objective：

\[
\mathcal L_\pi=\mathcal L_\pi^{dual}+\alpha_{BC}\mathcal L_{BC}.
\]

BC anchoring 的角色不是替代 DOTPG，而是让 direct dual gradient 在可靠区域发挥作用。论文中应表述为：

> BC term acts as a proximal action-manifold constraint that keeps off-policy actor updates inside teacher-supported regions where the OT dual critic is meaningful.

---

## 7. Q-Network 的变化

### 7.1 原始版本

原始 draft 写的是 single-Q target：

\[
y=f_\phi(s,a)+\gamma Q_{target}(s',\pi(s')).
\]

并把 actor update 主要建立在 Q 上。

### 7.2 当前版本

当前代码的 Q path 是 TD3-style：

\[
a'=\operatorname{clip}(\bar\pi(s')+\epsilon,-1,1),
\]

\[
y=r_{OT}(s,a)+\gamma(1-d)\min_i\bar Q_i(s',a'),
\]

\[
\mathcal L_Q=\sum_{i=1}^2 \operatorname{Huber}(Q_i(s,a),y).
\]

同时：

\[
r_{OT}=+\operatorname{NormalizeClipScale}(f_\phi(s,a)).
\]

当前论文应把 Q path 写成：

- optional long-horizon value learning；
- TD3-style auxiliary critic；
- Q actor variants are implemented but not selected as main result。

---

## 8. Expert Data 和 Replay Distribution 的变化

### 8.1 原始版本

原始 draft 更像是标准 demonstration dataset setting：

\[
D_E=\{(s_i,a_i)\}.
\]

### 8.2 当前版本

当前实现包括：

- frozen PPO teacher checkpoint；
- teacher rollouts 写入 expert buffer；
- online expert refresh；
- replay buffer 中的 student transitions；
- dual update 使用 replay states + current actor actions。

这意味着当前理论应避免把 \(\rho_\pi\) 写成完全 on-policy 精确 occupancy。更准确的说法是：

\[
\hat\rho_\theta
=\frac1B\sum_{s_i\sim B}\delta_{(s_i,\pi_\theta(s_i))}.
\]

也就是 **off-policy empirical state distribution with current-policy action relabeling**。

---

## 9. Student State Representation 的变化

### 9.1 原始版本

原始 draft 没有把当前 teacher-student CoDrive 的 state representation 写清楚。

### 9.2 当前版本

当前 student state 是：

\[
s_\eta(o,h)=\operatorname{concat}(N_o(o),z_\eta(N_h(h))).
\]

其中 adapter \(z_\eta\) 通过 teacher privileged latent distillation warmup 学习，之后可 freeze。Teacher 则使用 privileged info 和 point cloud latent。

这应放到方法章节开头，因为它决定了当前 DOTPG 是一个 **student distillation / sim-to-real deployment** 算法，而不是普通 RL policy optimization。

---

## 10. Sinkhorn / Entropic OT 的变化

### 10.1 原始版本

原始 draft 有 entropic regularization 和 Sinkhorn algorithm section，并声称将 Sinkhorn 用作 critic initialization。

### 10.2 当前版本

当前代码没有 Sinkhorn initialization。

### 10.3 修改建议

论文中应二选一：

1. 删除 Sinkhorn 作为 method component，只保留为 background；或
2. 真的实现 Sinkhorn initialization 后再写入算法。

在当前版本中，建议删除或降级为 background，否则理论与代码不一致。

---

## 11. Maximum Entropy / SAC 相关内容的变化

### 11.1 原始版本

原始 draft 包含 maximum entropy theorem、soft optimal policy、KL-to-soft-optimal-policy 推导等内容。

### 11.2 当前版本

当前代码没有：

- stochastic Gaussian actor；
- action log probability；
- entropy temperature \(\alpha\)；
- SAC-style soft Bellman backup；
- KL-to-soft-optimal-policy update。

因此，这些 theorem 不应出现在当前方法主线中。若保留，只能作为 future extension 或 unrelated background，不应作为当前 DOTPG 的理论保证。

---

## 12. Theorem 层面的变化建议

| 原始 theorem / claim | 当前处理建议 | 原因 |
|---|---|---|
| KR duality | 保留，修正 Lipschitz typo | 是 DOTPG 基础 |
| General Kantorovich duality | 简化保留或放 background | 有用但不是当前实现核心 |
| Entropic / Sinkhorn | 删除 method claim | 当前未实现 |
| OT deterministic policy gradient | 改成 proposition：fixed-replay semi-gradient + full occupancy Q-gradient distinction | 原证明忽略 state distribution derivative |
| Dual convergence to exact \(f^*\) | 弱化为 empirical neural dual optimization | 非凸 NN + GP 不保证全局最优 |
| Wasserstein estimation error \(O(\lambda^{-1})\) / exponential | 删除或重建 | 当前推导无充分条件 |
| Maximum entropy policy theorem | 删除 | 当前不是 SAC |
| KL-to-soft-optimal-policy theorem | 删除 | 当前未实现且符号方向有问题 |
| Alternating stability | 改成 fixed surrogate local descent | 不能保证 true W monotonic |
| Two-time-scale convergence | 降级为 design rationale | 当前 Adam + fixed finite updates 不满足 Robbins-Monro theorem |
| Target network \(Q^*\) bound | 删除 | 原 bound 可疑，尤其 \(O(\tau^{-1})\) |
| Smooth target transition | 保留 | 是精确 identity |
| Sim-to-real transfer bound | 删除或 discussion | 当前无充分假设与实验支撑 |
| Monotonic policy improvement | 改成 sufficient-condition discussion | 当前 nonconvex off-policy 不保证 |
| End-to-end convergence | 删除或 limitations | 过强 |
| Sample complexity | 删除或未来工作 | 当前非 code-faithful |

---

## 13. 论文主算法应替换成什么

当前版本建议在论文中使用下面的核心公式。

### 13.1 Dual critic

\[
\mathcal L_{dual}(\phi)
= -\left(
\mathbb E_{D_E}[f_\phi(s_E,a_E)]
-
\mathbb E_{B}[f_\phi(s,\pi_\theta(s))]
\right)
+\lambda_{GP}\mathcal L_{GP}(\phi).
\]

### 13.2 Main actor

\[
\mathcal L_{actor}(\theta)
= -\mathbb E_{B}[f_\phi(s,\pi_\theta(s))]
+\alpha_{BC}\mathbb E_{D_E}\|\pi_\theta(s_E)-a_E\|^2.
\]

### 13.3 Optional auxiliary Q

\[
y=r_{OT}(s,a)+\gamma(1-d)
\min_i \bar Q_i(s',\operatorname{clip}(\bar\pi(s')+\epsilon,-1,1)).
\]

\[
\mathcal L_Q=\sum_i \operatorname{Huber}(Q_i(s,a),y).
\]

### 13.4 Correct actor sign statement

\[
\boxed{
\text{Because } W=E_E[f]-E_\pi[f],\text{ minimizing }W\text{ requires maximizing }E_\pi[f].
}
\]

这句话应作为修正原始 draft 符号问题的中心说明。

---

## 14. 实验与叙述层面的变化

### 14.1 当前实验证据支持的结论

当前优化记录支持如下较稳妥结论：

1. Q-only actor objective 在该 dexterous contact task 中表现弱；
2. direct dual actor objective 明显更强；
3. BC anchoring 对 deploy performance 很重要；
4. `dual_bc5` 是当前 selected deploy baseline；
5. train reward 不能单独作为 DOTPG model selection 标准，fixed-step deploy eval 更可靠。

### 14.2 不应过度扩大的结论

不建议当前论文声称：

- DOTPG 已经全面超过所有 SOTA baseline；
- DOTPG 全局收敛；
- DOTPG 保证 monotonic improvement；
- DOTPG policy 对 sim-to-real dynamics mismatch invariant；
- Q path 是当前最成功的 actor objective。

更合适的论文表述是：

> In high-DOF contact-rich teacher-student distillation, the direct OT dual actor update is empirically more stable than a Q-only long-horizon dual-reward actor update. BC anchoring further constrains the student to remain near the teacher action manifold, making the empirical OT dual gradient useful rather than extrapolative.

---

## 15. 最终差异结论

当前 DOTPG 与原始 DOTPG draft 的差异不是简单超参数变化，而是一次 **算法实现路线修正**：

1. 从 generic / Q-only actor-critic 转向 teacher-initialized direct-dual actor；
2. 从原始 Q-based 主 actor loss 转向 \(-E[f_\phi]\) 主 loss；
3. 从无约束 actor update 转向 BC-anchored actor update；
4. 从强理论保证转向 code-faithful empirical surrogate theory；
5. 从 maximum-entropy / Sinkhorn 叙述转向 deterministic OT dual imitation 叙述；
6. 从 train reward selection 转向 deploy-eval-aware selection。

因此，当前论文理论部分最应该重写成：

\[
\boxed{
\text{Empirical Wasserstein dual imitation}
\rightarrow
\text{direct dual actor semi-gradient}
\rightarrow
\text{BC-anchored teacher-student policy improvement}
\rightarrow
\text{optional TD3-style Q auxiliary}.
}
\]

这才是当前 `dual_bc5` DOTPG 代码版本的准确理论记录。
