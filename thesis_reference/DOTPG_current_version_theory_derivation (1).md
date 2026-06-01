# 当前版本 DOTPG 的代码忠实版理论推导

Date: 2026-05-22  
Status: **用于替换 / 重写论文中 DOTPG 理论部分的工作稿**  
Scope: 基于当前 `dual_bc5` 代码边界、CoDrive 优化记录和原始 DOTPG draft，重新整理一个符号一致、推导无反号错误、不过度宣称的理论版本。

---

## 0. Source Documents

本稿以如下文件为 source：

1. `dc24bacd-7d89-48c0-8ae0-bbba4174e913.md`  
   当前代码忠实版 DOTPG 框架说明，记录了 `dual_bc5` 的实现边界、符号到代码对象的映射、实际 loss、TD3-style Q、BC anchoring 和 theorem revision plan。
2. `f922a724-36f2-41e8-a46a-6d29e90eea05.md`  
   DOTPG CoDrive optimization note，记录了从早期 Q-only / generic actor 到当前 `teacher_actor + dual + BC` 路线的迭代原因和实验现象。
3. `DOTPG-draft.md` / `DOTPG-draft.pdf`  
   原始 DOTPG 论文草稿，作为理论重写的参照版本。

本文档的核心目标不是证明深度非凸 off-policy imitation learning 的全局收敛，而是给出一个**论文规范、符号自洽、代码忠实、可作为方法章节基础**的 DOTPG 理论推导版本。

---

## 1. 当前版本 DOTPG 的理论定位

当前代码版本应表述为：

> **DOTPG-Dual-BC** 是一个 deterministic、off-policy、teacher-student imitation / distillation 算法。它在 teacher 与 student 的 state-action 样本之间学习一个 Wasserstein-1 optimal-transport dual critic，并主要用该 dual potential 的直接梯度更新 student actor，同时用 BC anchoring 将 actor 约束在 teacher action manifold 附近；TD3-style twin-Q 模块保留为 long-horizon auxiliary / diagnostic component，而不是当前成功版本的主要 actor objective。

因此，当前版本不应表述为：

- stochastic maximum-entropy RL；
- SAC-style policy；
- second-stage PPO；
- Sinkhorn-initialized OT critic；
- 具有无条件 monotonic improvement / global convergence / sim-to-real invariance guarantee 的算法。

当前版本的主线是：

\[
\boxed{
\text{teacher-student imitation}
+ \text{empirical Wasserstein dual critic}
+ \text{direct dual actor semi-gradient}
+ \text{BC anchoring}
+ \text{optional TD3-style Q auxiliary}
}
\]

---

## 2. Teacher-Student Problem Formulation

### 2.1 MDP and teacher-student setting

考虑一个 discounted MDP：

\[
\mathcal M=(\mathcal S,\mathcal A,P,r_{env},\gamma,\rho_0),
\quad \gamma\in[0,1).
\]

当前 DOTPG 训练中，环境 reward \(r_{env}\) 主要用于 logging、checkpoint selection 和 evaluation，不进入 DOTPG 的 Bellman target。DOTPG 的 imitation signal 来自 teacher demonstrations 与 student behavior 的 state-action distribution matching。

令 PPO teacher policy 为：

\[
\pi_E(a\mid o,p,c),
\]

其中 \(o\) 表示非特权 observation，\(p\) 表示 privileged information，\(c\) 表示 point-cloud / extrinsic features。Student deployment 时只能使用非特权 observation 与 proprioceptive history。当前 student state representation 可抽象为：

\[
s_\eta(o,h)=\operatorname{concat}\big(N_o(o), z_\eta(N_h(h))\big),
\]

其中：

- \(N_o,N_h\) 是 observation / history normalization；
- \(h\) 是 proprioceptive history；
- \(z_\eta\) 是 student adapter / latent encoder；
- \(\eta\) 可在 warmup 阶段通过 teacher privileged latent distillation 训练，之后可 freeze。

Student actor 是 deterministic policy：

\[
a=\pi_\theta(s_\eta)\in[-1,1]^{d_a}.
\]

当前 selected run 使用 teacher actor architecture 与 teacher initialization：

\[
\pi_\theta \leftarrow \operatorname{copy}(\pi_E\text{ actor\_mlp + mu}),
\]

但拷贝后的参数是 trainable，而 teacher 本身保持 frozen。

### 2.2 Expert buffer and replay buffer

Expert dataset / buffer 定义为：

\[
\mathcal D_E=\{(s_i^E,a_i^E)\}_{i=1}^{N_E},
\quad a_i^E=\pi_E(o_i,p_i,c_i).
\]

Student replay buffer 定义为：

\[
\mathcal B=\{(s_i,a_i,s_i',d_i)\}_{i=1}^{N_B},
\]

其中 \(d_i\in\{0,1\}\) 是 done indicator。当前 dual update 默认使用 replay states 加 current policy actions：

\[
a_i^\theta=\pi_\theta(s_i),
\]

而不是完全使用 replay action \(a_i\)。因此，dual critic 比较的是 teacher state-action samples 与 student current-action samples。

### 2.3 Occupancy measure and empirical approximation

理论上的 discounted occupancy measure 为：

\[
\rho_\pi(s,a)
=(1-\gamma)\sum_{t=0}^{\infty}\gamma^t
\Pr(s_t=s,a_t=a\mid \pi,P,\rho_0).
\]

对于 deterministic student：

\[
\rho_{\pi_\theta}(s,a)=d_{\pi_\theta}(s)\,\delta(a-\pi_\theta(s)),
\]

其中 \(d_{\pi_\theta}\) 是 normalized discounted state visitation distribution。

当前代码不是精确 on-policy occupancy optimization，而是使用 replay-buffer empirical distribution 近似：

\[
\hat\rho_E = \frac1{B}\sum_{i=1}^B\delta_{(s_i^E,a_i^E)},
\quad
\hat\rho_\theta = \frac1{B}\sum_{i=1}^B\delta_{(s_i,\pi_\theta(s_i))},
\]

其中 \(s_i\sim\mathcal B\)，\((s_i^E,a_i^E)\sim\mathcal D_E\)。因此，论文中应明确：当前 DOTPG 使用的是 **empirical off-policy occupancy surrogate**，而不是对真实 \(\rho_{\pi_\theta}\) 的精确全量优化。

---

## 3. Wasserstein-1 Distribution Matching Objective

### 3.1 State-action metric

令：

\[
x=(s,a)\in\mathcal X=\mathcal S\times\mathcal A.
\]

当前实现允许 dual input scaling：

\[
m_f(s,a)=(c_s s, c_a a),
\]

其中 \(c_s=\texttt{dual\_state\_scale}\)，\(c_a=\texttt{dual\_action\_scale}\)。若使用 Euclidean metric，则实际 OT geometry 是经过 \(m_f\) 缩放后的 state-action geometry：

\[
d_f(x,x')=\|m_f(x)-m_f(x')\|_2.
\]

这点很重要：Wasserstein distance 的含义依赖 metric。若论文声称 Wasserstein matching，就应说明 state/action normalization 与 scaling 是 metric design 的一部分。

### 3.2 Primal objective

DOTPG 的 imitation objective 是令 student state-action distribution 接近 teacher / expert distribution：

\[
\min_\theta W_1(\rho_E,\rho_{\pi_\theta}),
\]

其中：

\[
W_1(\rho_E,\rho_{\pi_\theta})
=\\inf_{\Gamma\in\Pi(\rho_E,\rho_{\pi_\theta})}
\mathbb E_{(x_E,x_\theta)\sim\Gamma}\big[d_f(x_E,x_\theta)\big].
\]

注意 \(W_1\) 对两个分布对称，但 dual 写法的正负号必须固定。本文采用当前代码一致的 sign convention：

\[
\boxed{
W_1(\rho_E,\rho_{\pi_\theta})
=
\sup_{\|f\|_{\mathrm{Lip}}\le 1}
\mathbb E_{x\sim\rho_E}[f(x)]
-
\mathbb E_{x\sim\rho_{\pi_\theta}}[f(x)]
}
\]

其中 Lipschitz norm 是相对于 \(d_f\) 的。

### 3.3 Empirical neural dual objective

用 neural dual critic \(f_\phi\) 近似 Kantorovich potential。给定 expert batch 与 student batch，定义：

\[
\hat W_\phi(\theta)
=\frac1B\sum_{i=1}^B f_\phi(m_f(s_i^E,a_i^E))
-
\frac1B\sum_{i=1}^B f_\phi(m_f(s_i,\pi_\theta(s_i))).
\]

当前 dual critic 最大化：

\[
\max_\phi\;\hat W_\phi(\theta)-\lambda_{GP}\mathcal L_{GP}(\phi).
\]

由于 PyTorch optimizer 默认最小化 loss，代码中的 dual loss 应写为：

\[
\boxed{
\mathcal L_{dual}(\phi)
= -\hat W_\phi(\theta)+\lambda_{GP}\mathcal L_{GP}(\phi)
}
\]

即：

\[
\mathcal L_{dual}(\phi)
= -\Big(
\mathbb E_{\mathcal D_E}[f_\phi(s_E,a_E)]
-
\mathbb E_{\mathcal B}[f_\phi(s,\pi_\theta(s))]
\Big)
+
\lambda_{GP}\mathcal L_{GP}(\phi).
\]

### 3.4 Gradient penalty

对 paired expert-policy samples 构造插值：

\[
\hat x_i=\epsilon_i m_f(s_i^E,a_i^E)+(1-\epsilon_i)m_f(s_i,\pi_\theta(s_i)),
\quad \epsilon_i\sim U[0,1].
\]

WGAN-GP style penalty 为：

\[
\mathcal L_{GP}(\phi)
=
\mathbb E_{\hat x}
\left[
\left(\|\nabla_{\hat x}f_\phi(\hat x)\|_2-1\right)^2
\right].
\]

论文中必须谨慎表述：

- \(\mathcal L_{GP}\) 是 empirical soft Lipschitz regularizer；
- 它不严格证明 \(f_\phi\) 在整个 \(\mathcal X\) 上全局 1-Lipschitz；
- 因此 \(\hat W_\phi\) 是 empirical Wasserstein dual estimate / surrogate，而不是无误差的 exact Wasserstein distance。

---

## 4. Correct Sign Convention and Actor Direction

当前 sign convention 是：

\[
\hat W_\phi(\theta)
=
\underbrace{\mathbb E_{\mathcal D_E}[f_\phi(s_E,a_E)]}_{\text{expert score}}
-
\underbrace{\mathbb E_{\mathcal B}[f_\phi(s,\pi_\theta(s))]}_{\text{student current-action score}}.
\]

固定 \(f_\phi\) 时，expert score 与 \(\theta\) 无关。因此：

\[
\min_\theta \hat W_\phi(\theta)
\quad\Longleftrightarrow\quad
\max_\theta
\mathbb E_{s\sim\mathcal B}[f_\phi(s,\pi_\theta(s))].
\]

所以 actor 的 direct-dual loss 必须是：

\[
\boxed{
\mathcal L_\pi^{dual}(\theta)
= -\mathbb E_{s\sim\mathcal B}[f_\phi(m_f(s,\pi_\theta(s)))]
}
\]

这与代码中的 `policy_loss_mode=dual` 一致。

若 actor 使用 Q，则方向也应是：

\[
\mathcal L_\pi^Q(\theta)
= -\mathbb E_{s\sim\mathcal B}
\left[\min_i Q_{\psi_i}(s,\pi_\theta(s))\right].
\]

因此，原始 draft 中直接写：

\[
\min_\theta \mathbb E[Q_\psi(s,\pi_\theta(s))]
\]

在当前 sign convention 下是错误方向；正确写法是 minimize negative Q 或 maximize Q。

---

## 5. Deterministic OT Policy Semi-Gradient

### 5.1 Fixed-batch empirical surrogate

当前成功版本的 actor update 不是直接优化完整 occupancy derivative，而是在 replay states 固定的情况下优化 empirical actor surrogate：

\[
J_{dual}(\theta;\phi,\mathcal B)
=
\mathbb E_{s\sim\mathcal B}
\left[f_\phi(m_f(s,\pi_\theta(s)))\right].
\]

于是：

\[
\hat W_\phi(\theta)
= C_E - J_{dual}(\theta;\phi,\mathcal B),
\quad
C_E=\mathbb E_{\mathcal D_E}[f_\phi(s_E,a_E)].
\]

固定 \(\phi\) 与 batch states \(s\)，对 \(\theta\) 求导：

\[
\nabla_\theta \hat W_\phi(\theta)
= -\nabla_\theta J_{dual}(\theta;\phi,\mathcal B).
\]

由 chain rule：

\[
\boxed{
\nabla_\theta \hat W_\phi(\theta)
= -\mathbb E_{s\sim\mathcal B}
\left[
\nabla_\theta\pi_\theta(s)^\top
\nabla_a f_\phi(m_f(s,a))\big|_{a=\pi_\theta(s)}
\right]
}
\]

如果 \(m_f(s,a)=(c_s s,c_a a)\)，则上式中的 \(\nabla_a f_\phi(m_f(s,a))\) 已包含 action scaling 的 chain-rule factor \(c_a\)。更显式地：

\[
\nabla_a f_\phi(m_f(s,a))
= c_a\,\nabla_{\tilde a} f_\phi(c_s s,c_a a).
\]

因此，对 \(\hat W_\phi\) 做 gradient descent 等价于对 \(J_{dual}\) 做 gradient ascent。PyTorch 中通过最小化：

\[
\mathcal L_\pi^{dual}=-J_{dual}
\]

实现。

### 5.2 与原始 Theorem 3 的关系

原始 draft 中写的形式：

\[
\nabla_\theta W(\rho_{\pi_\theta},\rho_E)
= -\mathbb E_{s\sim d_{\pi_\theta}}
\left[
\nabla_\theta\pi_\theta(s)^\top
\nabla_a f^*(s,a)\big|_{a=\pi_\theta(s)}
\right]
\]

作为 **fixed-state empirical surrogate** 或 **contextual-bandit / one-step approximation** 是合理方向；但若把 \(\rho_{\pi_\theta}\) 理解为完整 MDP occupancy measure，则这个式子一般不完整，因为 state visitation \(d_{\pi_\theta}(s)\) 也依赖 \(\theta\)。

完整 MDP 下，若固定 dual reward \(f\)，定义：

\[
J_f(\theta)
=\mathbb E_{(s,a)\sim\rho_{\pi_\theta}}[f(s,a)]
=(1-\gamma)\mathbb E_{\tau\sim\pi_\theta}
\left[\sum_{t=0}^{\infty}\gamma^t f(s_t,a_t)\right].
\]

令：

\[
Q_f^{\pi_\theta}(s,a)
=\mathbb E_{\pi_\theta}
\left[\sum_{t=0}^{\infty}\gamma^t f(s_t,a_t)
\mid s_0=s,a_0=a\right].
\]

根据 deterministic policy gradient theorem，有：

\[
\boxed{
\nabla_\theta J_f(\theta)
=\mathbb E_{s\sim d_{\pi_\theta}}
\left[
\nabla_\theta\pi_\theta(s)^\top
\nabla_a Q_f^{\pi_\theta}(s,a)\big|_{a=\pi_\theta(s)}
\right]
}
\]

因此，在完整 occupancy 目标下，固定最优 dual potential \(f^*\) 并使用 envelope theorem 时：

\[
\nabla_\theta W_1(\rho_E,\rho_{\pi_\theta})
= -\nabla_\theta J_{f^*}(\theta)
= -\mathbb E_{s\sim d_{\pi_\theta}}
\left[
\nabla_\theta\pi_\theta(s)^\top
\nabla_a Q_{f^*}^{\pi_\theta}(s,a)\big|_{a=\pi_\theta(s)}
\right].
\]

这个式子说明：

- \(f_\phi(s,a)\) 是 OT dual potential / learned imitation reward；
- \(Q_f^\pi(s,a)\) 是该 dual reward 的 long-horizon value；
- 直接使用 \(\nabla_a f_\phi\) 是 fixed replay-state / one-step semi-gradient；
- 使用 \(\nabla_a Q_\psi\) 是 long-horizon extension，但会引入 Q approximation error。

当前 CoDrive contact task 中，实验证据显示 Q-only actor objective 噪声过大，因此 selected version 采用 direct dual actor semi-gradient 作为主 actor objective，并用 BC anchoring 保持 actor 在 teacher manifold 附近。

---

## 6. BC-Anchored Direct-Dual Actor Objective

### 6.1 BC anchor

定义 expert BC loss：

\[
\mathcal L_{BC}(\theta)
=\mathbb E_{(s_E,a_E)\sim\mathcal D_E}
\left[\|\pi_\theta(s_E)-a_E\|_2^2\right].
\]

当前主 actor objective 为：

\[
\boxed{
\mathcal L_\pi^{DOTPG-Dual-BC}(\theta)
= -\mathbb E_{s\sim\mathcal B}
\left[f_\phi(m_f(s,\pi_\theta(s)))\right]
+\alpha_{BC}\mathcal L_{BC}(\theta)
}
\]

其中 \(\alpha_{BC}\) 是 adaptive / clipped BC weight。实现中 selected run 使用：

\[
\texttt{bc\_coef}=5.0,
\quad
\texttt{bc\_alpha\_min}=0.01,
\quad
\texttt{bc\_alpha\_max}=20.0.
\]

理论上，\(\alpha_{BC}\) 可以视为 proximal coefficient。它并不把 DOTPG 退化成 pure BC，因为 actor update 仍包含 direct dual improvement term；它的作用是防止 off-policy actor 被 dual critic extrapolation 引导到 teacher demonstrations 未覆盖的 state-action 区域。

### 6.2 Actor gradient

对 actor loss 求梯度：

\[
\nabla_\theta\mathcal L_\pi^{DOTPG-Dual-BC}
= -\mathbb E_{s\sim\mathcal B}
\left[
\nabla_\theta\pi_\theta(s)^\top
\nabla_a f_\phi(m_f(s,a))\big|_{a=\pi_\theta(s)}
\right]
+2\alpha_{BC}\mathbb E_{\mathcal D_E}
\left[
\nabla_\theta\pi_\theta(s_E)^\top(\pi_\theta(s_E)-a_E)
\right].
\]

因此，gradient descent 的方向包含两部分：

1. **dual ascent direction**：提高 student current action 在 OT dual critic 下的 score；
2. **BC anchoring direction**：将 student action 拉回 teacher action manifold。

### 6.3 Local surrogate descent statement

令 \(\mathcal L_\pi(\theta)\) 表示固定 \(\phi\)、固定 batch distribution 后的 actor surrogate，并假设 \(\mathcal L_\pi\) 是 \(L\)-smooth。若使用普通 gradient step：

\[
\theta^+ = \theta-\eta\nabla_\theta\mathcal L_\pi(\theta),
\quad 0<\eta\le \frac1L,
\]

则由 smoothness descent lemma：

\[
\mathcal L_\pi(\theta^+)
\le
\mathcal L_\pi(\theta)
-
\eta\left(1-\frac{L\eta}{2}\right)
\|\nabla_\theta\mathcal L_\pi(\theta)\|_2^2.
\]

这可以作为论文中安全的 stability statement：**actor step 对固定 critic 与固定 replay batch 下的 surrogate objective 有局部下降性质**。它不能被扩大为真实 \(W_1(\rho_E,\rho_{\pi_\theta})\) 的无条件 monotonic improvement。

---

## 7. Optional TD3-Style Long-Horizon Q Auxiliary

虽然当前 selected actor objective 是 direct dual + BC，代码仍保留 TD3-style twin-Q 模块。它应作为 auxiliary long-horizon extension / diagnostic component 介绍。

### 7.1 Dual reward used by Q

定义 raw dual reward：

\[
r_{raw}(s,a)=f_\phi(m_f(s,a)).
\]

实现中 Q backup 使用经过 normalization / scale / clipping 的 reward：

\[
r_{OT}(s,a)
=\operatorname{clip}\Big(c_r\,\operatorname{RMSNorm}(r_{raw}(s,a)),
-r_{max},r_{max}\Big),
\]

其中默认方向是：

\[
\boxed{r_{OT}=+f_\phi(s,a)}
\]

不是 \(-f_\phi\)。这与 actor 最大化 student-side dual score 的方向一致。

### 7.2 Twin-Q Bellman target

令 target actor 与 target Q 为 \(\bar\pi_{\bar\theta}\)、\(\bar Q_{\bar\psi_1},\bar Q_{\bar\psi_2}\)。TD3-style target action：

\[
\epsilon\sim\operatorname{clip}(\mathcal N(0,\sigma^2),-c,c),
\quad
a'=\operatorname{clip}(\bar\pi_{\bar\theta}(s')+\epsilon,-1,1).
\]

Target value：

\[
\bar Q_{min}(s',a')=
\min_{i=1,2}\bar Q_{\bar\psi_i}(m_Q(s',a')),
\]

其中 \(m_Q\) 可包含 critic state/action scaling。Bellman target：

\[
\boxed{
 y=r_{OT}(s,a)+\gamma(1-d)\bar Q_{min}(s',a')
}
\]

Twin-Q loss：

\[
\mathcal L_Q(\psi_1,\psi_2)
=
\sum_{i=1}^2
\mathbb E_{\mathcal B}
\left[\ell\big(Q_{\psi_i}(m_Q(s,a)),y\big)\right],
\]

其中 \(\ell\) 是 Huber loss 或 MSE。当前默认使用 Huber loss。

### 7.3 Actor variants

代码支持多个 actor modes：

\[
\mathcal L_\pi^Q
= -\mathbb E_{s\sim\mathcal B}
\left[\min_i Q_{\psi_i}(m_Q(s,\pi_\theta(s)))\right],
\]

\[
\mathcal L_\pi^{Q+dual}
= -\mathbb E_{s\sim\mathcal B}
\left[\min_i Q_{\psi_i}(m_Q(s,\pi_\theta(s)))\right]
-\beta\mathbb E_{s\sim\mathcal B}
\left[f_\phi(m_f(s,\pi_\theta(s)))\right].
\]

但对于当前论文主版本，建议写作：

\[
\boxed{
\mathcal L_\pi^{main}
=\mathcal L_\pi^{dual}+\alpha_{BC}\mathcal L_{BC}
}
\]

并将 Q-only / Q+dual 表述为 ablation variants，而不是主算法的理论核心。

---

## 8. Target Network Update

当前实现对 twin-Q targets 与 policy target 使用 soft update：

\[
\bar\psi_i\leftarrow \tau\psi_i+(1-\tau)\bar\psi_i,
\quad i=1,2,
\]

\[
\bar\theta\leftarrow \tau\theta+(1-\tau)\bar\theta.
\]

可保留一个简单且正确的 lemma：

**Lemma.** 对任意参数向量 \(u\) 与 target \(\bar u\)，若：

\[
\bar u^+ = \tau u+(1-\tau)\bar u,
\quad 0<\tau<1,
\]

则：

\[
\|\bar u^+-\bar u\|=\tau\|u-\bar u\|.
\]

这说明 target update 的移动幅度被 \(\tau\) 线性控制。不要写成过强的 \(Q^*\) error bound，也不要引入与稳定性叙述冲突的 \(O(\tau^{-1})\) bound。

---

## 9. Code-Faithful DOTPG-Dual-BC Algorithm

```text
Algorithm: Code-Faithful DOTPG-Dual-BC

Input:
  teacher checkpoint omega
  environment M
  student actor pi_theta
  OT dual critic f_phi
  optional twin Q critics Q_psi1, Q_psi2 with target copies
  expert buffer D_E and replay buffer B
  dual updates J, policy delay d
  discount gamma, target update tau
  gradient penalty weight lambda_GP
  BC weight alpha_BC

1. Load frozen PPO teacher pi_E and normalizers from omega.

2. Initialize student actor pi_theta.
   If policy_arch = teacher_actor and policy_init_from_teacher = True:
      deep-copy teacher actor_mlp + mu into pi_theta;
      set copied student parameters requires_grad = True.

3. Initialize f_phi, Q_psi1, Q_psi2 and target networks.

4. Optional adapter warmup:
      collect teacher rollouts;
      minimize ||z_eta(proprio_history) - z_E(privileged_info, point_cloud)||^2;
      optionally freeze eta and proprio normalizer.

5. Expert collection:
      roll out teacher pi_E;
      store (s_E, a_E) in D_E, where a_E = pi_E(o, priv, point_cloud).

6. Optional BC pretraining:
      minimize E_{D_E}[||pi_theta(s_E)-a_E||^2].

7. For each environment step:
      build student state s_t = concat(normalized_obs, z_eta(history));
      execute a_t = clip(pi_theta(s_t) + exploration_noise, -1, 1);
      step environment and store (s_t, a_t, s_{t+1}, done_t) in B;

      if online_expert is enabled:
          query frozen teacher on current state;
          store teacher-labeled sample in D_E.

      Repeat updates_per_env_step times:

          Sample replay batch (s,a,s',done) from B.
          Sample expert batch (s_E,a_E) from D_E.

          # Dual critic update
          For j = 1,...,J:
              a_pi = pi_theta(s)       # current policy action
              x_E  = m_f(s_E, a_E)
              x_pi = m_f(s, a_pi)
              x_hat = eps*x_E + (1-eps)*x_pi
              L_GP = E[(||grad_x f_phi(x_hat)||_2 - 1)^2]
              W_hat = E[f_phi(x_E)] - E[f_phi(x_pi)]
              phi <- Adam step minimizing -W_hat + lambda_GP * L_GP

          # Optional Q update
          a_next = clip(pi_bar(s') + clipped_noise, -1, 1)
          r_OT = normalize_clip_scale(f_phi(m_f(s,a)))
          y = r_OT + gamma*(1-done)*min_i Q_bar_i(m_Q(s',a_next))
          psi_1, psi_2 <- Adam step on Huber(Q_psi_i(m_Q(s,a)), y)

          # Actor update, selected main path
          if update_index mod policy_delay == 0:
              L_actor = -E[f_phi(m_f(s, pi_theta(s)))]
              if BC anchoring enabled:
                  L_actor += alpha_BC * E[||pi_theta(s_E)-a_E||^2]
              theta <- Adam step minimizing L_actor
              soft-update target actor and target critics
```

---

## 10. What Can Be Claimed Theoretically

### 10.1 Safe claims

当前论文可以安全陈述：

1. **Distribution matching objective.** DOTPG formulates teacher-student imitation as empirical Wasserstein-1 state-action distribution matching.
2. **Dual critic.** The dual critic approximates a Kantorovich-Rubinstein potential under soft Lipschitz regularization.
3. **Correct actor direction.** Under the adopted sign convention, minimizing the empirical Wasserstein dual surrogate requires increasing student-side dual score; hence the actor minimizes \(-E[f_\phi(s,\pi_\theta(s))]\).
4. **Semi-gradient interpretation.** The direct dual actor gradient is the exact gradient of the fixed-replay empirical dual surrogate and can be viewed as a one-step OT policy semi-gradient.
5. **Long-horizon interpretation.** If one wants the full MDP occupancy gradient for fixed dual reward, the correct deterministic policy gradient uses \(Q_f^\pi\), not \(f\) alone. The implemented twin-Q module is an auxiliary approximation to this long-horizon signal.
6. **BC anchoring.** BC regularization is a proximal / manifold constraint that reduces extrapolation of the actor and keeps policy updates near teacher-supported actions.
7. **Local surrogate stability.** For fixed critic and fixed replay batch, sufficiently small gradient steps reduce the smooth actor surrogate.

### 10.2 Claims to avoid or weaken

当前理论不应无条件声称：

1. neural dual critic converges globally to exact Kantorovich potential;
2. gradient penalty guarantees global 1-Lipschitzness;
3. every actor update monotonically decreases true \(W_1(\rho_E,\rho_{\pi_\theta})\);
4. DOTPG is maximum-entropy RL / SAC;
5. Sinkhorn initialization is part of current method;
6. target networks provide a closed-form \(Q^*\) error bound;
7. DOTPG is provably invariant to sim-to-real dynamics mismatch;
8. sample complexity bound holds for current nonconvex neural implementation without much stronger assumptions.

---

## 11. Suggested Replacement for the Original Theory Section

若要重写论文 Section 4，建议采用如下结构：

1. **Teacher-student imitation problem**  
   定义 teacher、student state representation、expert buffer、replay buffer。

2. **Empirical Wasserstein dual objective**  
   给出 \(W_1\) primal / KR dual；固定 sign convention；写清 \(L_{dual}=-\hat W+\lambda GP\)。

3. **Direct dual policy semi-gradient**  
   先推导 fixed-replay empirical surrogate 的 exact gradient：
   \[
   \nabla_\theta \hat W=-E[J_\pi^T\nabla_a f].
   \]
   再说明完整 occupancy gradient 应使用 \(Q_f^\pi\)。这样不会犯原始 Theorem 3 的过强推导错误。

4. **BC-anchored actor objective**  
   写当前主版本：
   \[
   \min_\theta -E_B[f_\phi(s,\pi_\theta(s))]+\alpha_{BC}E_{D_E}\|\pi_\theta(s_E)-a_E\|^2.
   \]

5. **TD3-style auxiliary value learning**  
   写 twin-Q、target smoothing、done mask、normalized/clipped \(+f\) reward。明确 Q 是 auxiliary，不是 selected main actor objective。

6. **Code-faithful algorithm**  
   使用上面的 Algorithm 作为主伪代码。

7. **Theoretical remarks and limitations**  
   将原始强 theorem 改写成 propositions / remarks：KR duality、semi-gradient direction、target soft update identity、local surrogate descent、limitations。

---

## 12. Final Current-Version Objective Summary

当前 `dual_bc5` DOTPG 可用下面四个公式概括。

**Dual critic:**

\[
\mathcal L_{dual}(\phi)
= -\left(
\mathbb E_{\mathcal D_E}[f_\phi(s_E,a_E)]
-
\mathbb E_{\mathcal B}[f_\phi(s,\pi_\theta(s))]
\right)
+
\lambda_{GP}\mathcal L_{GP}(\phi).
\]

**Main actor objective:**

\[
\mathcal L_\pi(\theta)
= -\mathbb E_{s\sim\mathcal B}[f_\phi(s,\pi_\theta(s))]
+
\alpha_{BC}\mathbb E_{(s_E,a_E)\sim\mathcal D_E}
\|\pi_\theta(s_E)-a_E\|_2^2.
\]

**Optional Q target:**

\[
y=r_{OT}(s,a)+\gamma(1-d)
\min_{i=1,2}\bar Q_i(s',\operatorname{clip}(\bar\pi(s')+\epsilon,-1,1)),
\quad r_{OT}=+\operatorname{NormalizeClipScale}(f_\phi(s,a)).
\]

**Soft target update:**

\[
\bar\psi_i\leftarrow \tau\psi_i+(1-\tau)\bar\psi_i,
\quad
\bar\theta\leftarrow \tau\theta+(1-\tau)\bar\theta.
\]

这四个公式是当前代码版本理论部分最应保留的核心。其最关键的符号结论是：

\[
\boxed{
\text{dual critic makes expert score high and student score low; actor must increase student-side } f_\phi.
}
\]

因此，actor loss 应为 \(-E[f_\phi]\) 或 \(-E[Q]\)，而不能写成直接 minimize \(E[Q]\)。
