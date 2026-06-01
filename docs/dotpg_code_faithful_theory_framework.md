# DOTPG Code-Faithful Theory Framework

Date: 2026-05-22

Scope: reverse-engineering the current `dexscrew/dotpg` implementation against
`thesis_reference/DOTPG-draft.md`, for revising the DOTPG theory, notation,
derivations, pseudocode, and theorem claims.

## Executive Verdict

The current implemented DOTPG should be described as:

> a deterministic, off-policy, teacher-student imitation/distillation algorithm
> that learns an OT dual critic over teacher and student state-action samples,
> updates the student actor mainly by the direct dual potential gradient, and
> uses TD3-style twin-Q machinery plus BC/DAgger-like anchoring as stabilizers.

It is not SAC, not stochastic maximum-entropy RL, and not a second-stage PPO/RL
policy. The PPO teacher remains the upstream policy. DOTPG replaces the PAdapt
student trainer with an OT-guided student adaptation procedure.

The central sign convention is:

```text
W_hat_phi(theta) = E_{rho_E}[f_phi(s,a)] - E_{rho_theta}[f_phi(s,a)]
```

The dual critic maximizes this expression. Therefore, minimizing Wasserstein
distance with respect to the student policy means increasing `f_phi` on current
student actions. The code-faithful actor loss is:

```text
L_pi = - E_{s ~ B_pi}[ f_phi(s, pi_theta(s)) ]
       + alpha_BC E_{(s_E,a_E) ~ D_E}[ ||pi_theta(s_E) - a_E||^2 ]
```

For the selected `dual_bc5` run, this direct-dual actor path is the main
algorithmic result. The Q path is implemented and should be documented, but the
paper should not present Q-only actor optimization as the primary successful
DOTPG objective.

## Implementation Summary

### Algorithm Version

- Current code version name: DOTPG `dual_bc5` / code-faithful DOTPG-dual actor.
- Main change from the original draft: the successful path uses
  `policy_loss_mode=dual`, `policy_arch=teacher_actor`, teacher initialization,
  BC anchoring, online expert refresh, and TD3-style twin-Q stabilization.
- Base style: deterministic off-policy actor-critic with TD3-like components.
- Not SAC: no stochastic actor log-prob, no entropy temperature, no soft value.
- Policy: deterministic.
- Action space: continuous, clipped to `[-1, 1]`.

Evidence:

- `dexscrew/dotpg/networks.py`: deterministic `PolicyNetwork` returns
  `tanh(MLP(s))`.
- `dexscrew/dotpg/dotpg.py`: `TeacherActorPolicy` returns `raw`, `tanh`, or
  `clamp` outputs; selected runs use `teacher_actor` and `clamp`.
- `dexscrew/dotpg/dotpg.py`: twin Q networks, target policy smoothing, delayed
  actor update.

### Environment And Task

- Main task: `Dexh13HoraLightbulbSim2RealTwoFingerCoDrive`.
- Environment class: `Dexh13HoraLightbulb` mapped to `Dexh13Hora`.
- Episode length: `800`.
- Raw observation dimension: `32 * 3 = 96`.
- Action dimension: `16`.
- Active actions: index finger `0:3` and thumb `12:15`.
- Masked actions: middle/ring indices `[4,5,6,7,8,9,10,11]`.
- Proprio history: `30 x 32`.
- Point cloud: `100 x 3`.
- Teacher/student extrinsic latent: `8 + 32 = 40`.
- DOTPG state dimension in the point-cloud path: approximately `96 + 40 = 136`.
- Controller: torque control, `controlFrequencyInv=10`, `action_scale=0.05`,
  `torque_limit=300`.
- Sim-to-real: yes, via randomization of mass, COM, friction, restitution,
  object scale, PD gains, observation/action noise, pose noise, and force
  disturbance.

### Expert Data

- Expert source: frozen PPO teacher checkpoint, e.g.
  `sim2real/codrive/best_reward_4159.37.pth`.
- Expert action: PPO teacher `actor_mlp + mu` applied to normalized observation
  plus teacher extrinsic latent from privileged information and point cloud.
- Expert buffer format:
  - legacy state mode: `(state, teacher_action)`;
  - dynamic state mode: `(obs, proprio_hist, teacher_action)`.
- Replay buffer format:
  - legacy state mode: `(state, action, next_state, done)`;
  - dynamic state mode: `(obs, proprio_hist, action, next_obs,
    next_proprio_hist, done)`.
- Environment reward is not stored in replay and does not enter DOTPG Bellman
  targets; it is used for logging and checkpoint selection.

## Symbol-To-Code Mapping

| Theory symbol | Code object | Notes |
|---|---|---|
| `pi_theta` | `self.policy` | `PolicyNetwork` or `TeacherActorPolicy`; selected path uses `teacher_actor`. |
| `bar pi_theta` | `self.policy_target` | Soft-updated target actor for TD target. |
| `f_phi` | `self.dual` / `_dual_value` | Spectral-normalized dual critic. |
| `Q_{psi_1}, Q_{psi_2}` | `self.q1`, `self.q2` | Twin Q networks. |
| `bar Q_{psi_i}` | `self.q1_target`, `self.q2_target` | Soft-updated target Q networks. |
| `rho_E` | `self.expert_buffer` | Teacher state-action distribution, sampled from PPO teacher rollouts and optional online expert refresh. |
| `rho_theta` / `rho_pi` | `self.replay_buffer` plus current actor actions | Replay states are sampled off-policy; dual update may replace replay actions with current `pi_theta(s)`. |
| `r_OT` | `_dual_value(states, actions)` after optional RMS/scale/clip | Used only for Q target. |
| `W_hat` | `wasserstein_dist` | `expert_value.mean() - policy_value.mean()`. |
| `L_GP` | `compute_gradient_penalty` | WGAN-GP style penalty on interpolated state-action pairs. |
| `z_eta` | `self.adapt_tconv` | Student extrinsic latent from proprio history. |
| `z_E` | `teacher_model.env_mlp(priv_info)` plus point features | Teacher extrinsic latent. |

## Actual Implemented Formulas

### Dual Critic

Let `x=(s,a)`. Let `m_f(x)` denote the optional dual input metric scaling:

```text
m_f(s,a) = (dual_state_scale * s, dual_action_scale * a).
```

The dual scores are:

```text
expert_score = E_{(s_E,a_E) ~ D_E}[ f_phi(m_f(s_E,a_E)) ]
policy_score = E_{s ~ B_pi, a_pi ~ policy side}[ f_phi(m_f(s,a_pi)) ]
W_hat_phi = expert_score - policy_score
```

In the default code path for the dual update:

```text
a_pi = pi_theta(s)
```

rather than the replay action, because `dual_use_current_policy_actions=True`.
Optional noise may be added by `dual_policy_noise`.

The gradient penalty is:

```text
epsilon ~ Uniform(0,1)
x_hat = epsilon x_E + (1 - epsilon) x_pi
L_GP = E[(||grad_{x_hat} f_phi(m_f(x_hat))||_2 - 1)^2]
```

The optimizer minimizes:

```text
L_dual(phi) = - W_hat_phi + lambda_GP L_GP
```

which is equivalent to maximizing:

```text
W_hat_phi - lambda_GP L_GP.
```

Default `lambda_GP = 10.0`.

### OT Reward

Raw dual reward:

```text
r_raw(s,a) = f_phi(m_f(s,a)).
```

The actual reward used in Q backup is:

```text
r_OT = clip(
  dual_reward_scale * RMS(r_raw),
  -dual_reward_clip,
  dual_reward_clip
)
```

with default:

```text
normalize_dual_reward = True
dual_reward_scale = 1.0
dual_reward_clip = 5.0
```

The Q target computation is under `torch.no_grad()`, so `r_OT` is detached for
Q learning. The actor direct-dual objective uses the non-normalized
`f_phi(s, pi_theta(s))` path, not the RMS-normalized Q reward.

Direction:

```text
r_OT = + f_phi(s,a)
```

not `-f_phi`.

### Q Update

Let critic input scaling be:

```text
m_Q(s,a) = (critic_state_scale * s, critic_action_scale * a).
```

Target action:

```text
epsilon ~ clip(N(0, target_policy_noise^2), -target_noise_clip, target_noise_clip)
a_next = clip(bar pi_theta(s_next) + epsilon, -1, 1)
```

Twin target:

```text
Q_targ = min_i bar Q_{psi_i}(m_Q(s_next, a_next))
```

Bellman target:

```text
y = r_OT(s,a) + gamma (1 - done) Q_targ
```

Q loss:

```text
L_Q = Huber(Q_{psi_1}(m_Q(s,a)), y)
    + Huber(Q_{psi_2}(m_Q(s,a)), y)
```

or MSE if `use_huber_q_loss=False`.

Defaults:

```text
gamma = 0.99
target_policy_noise = 0.1
target_noise_clip = 0.2
use_huber_q_loss = True
```

No entropy term is present.

### Actor Update

Code computes:

```text
a_theta = pi_theta(s)
Q_pi = min_i Q_{psi_i}(m_Q(s, a_theta))
F_pi = f_phi(m_f(s, a_theta))
```

Supported modes:

```text
policy_loss_mode = q:
  L_pi = - E[Q_pi]

policy_loss_mode = dual:
  L_pi = - E[F_pi]

policy_loss_mode = q_dual:
  L_pi = - E[Q_pi] - policy_dual_coef E[F_pi]
```

If BC regularization is enabled:

```text
L_BC = E_{(s_E,a_E) ~ D_E}[ ||pi_theta(s_E) - a_E||^2 ]
alpha_BC = bc_coef / mean(|Q_pi|)
alpha_BC = clamp(alpha_BC, bc_alpha_min, bc_alpha_max)
L_pi <- L_pi + alpha_BC L_BC
```

Selected `dual_bc5` run:

```text
policy_loss_mode = dual
bc_coef = 5.0
bc_alpha_min = 0.01
bc_alpha_max = 20.0
```

Thus the main thesis-faithful implementation objective is:

```text
min_theta - E_{s ~ B_pi}[f_phi(s, pi_theta(s))]
          + alpha_BC E_{(s_E,a_E) ~ D_E}[||pi_theta(s_E)-a_E||^2].
```

This is equivalent to maximizing the policy-side OT dual potential while keeping
the actor close to the teacher action manifold.

### Target Updates

Soft updates are applied to both twin Q targets and policy target:

```text
bar psi_i <- tau psi_i + (1 - tau) bar psi_i
bar theta <- tau theta + (1 - tau) bar theta
```

Default:

```text
tau = 0.005
policy_delay = 2
```

## Training Procedure

### Actual Iteration Order

1. Build IsaacGym environment through `train.py`.
2. Instantiate `DOTPGStudent`.
3. Load PPO teacher checkpoint and normalizers in `restore_train()`.
4. If `policy_arch=teacher_actor` and `policy_init_from_teacher=True`, deep-copy
   teacher `actor_mlp + mu` into the student actor and make copied parameters
   trainable.
5. Reset environment.
6. If `state_mode=student`, optionally warm up `adapt_tconv` with latent
   distillation from teacher privileged latent; optionally freeze `adapt_tconv`
   and proprio normalizer.
7. Collect expert samples by rolling out the PPO teacher and writing teacher
   actions to the expert buffer.
8. Optionally BC-pretrain the policy on expert `(s_E,a_E)`.
9. For each environment step:
   - build student state `s=[obs_norm, z_eta(proprio_hist)]`;
   - optionally continue training `adapt_tconv`;
   - execute `clip(pi_theta(s) + exploration_noise, -1, 1)`;
   - step the environment;
   - store transition in replay buffer;
   - optionally add online teacher labels to expert buffer;
   - run `updates_per_env_step` DOTPG updates.
10. Each DOTPG update:
   - sample replay and expert batches;
   - reconstruct dynamic states if needed;
   - build current actor actions for dual update;
   - update dual critic `J=dual_updates_per_iter` times;
   - update twin Q using `r_OT=+f_phi`;
   - every `policy_delay` updates, update actor and target networks.
11. Log environment reward and DOTPG losses; save best checkpoints by training
    reward unless an eval-selection wrapper is active.

### Algorithm 1: Code-Faithful DOTPG-Dual-BC

```text
Input:
  teacher checkpoint omega
  environment M
  actor pi_theta, dual critic f_phi
  twin Q networks Q_psi1, Q_psi2 and targets
  expert buffer D_E, replay buffer B
  update counts J, d, discount gamma, soft-update tau
  BC weight lambda_BC

Load teacher pi_E and teacher normalizers from omega
Initialize pi_theta from teacher actor if policy_arch = teacher_actor
Initialize f_phi, Q_psi1, Q_psi2 and target copies

Optional adapter warmup:
  Roll out pi_E
  Minimize ||z_eta(proprio_hist) - z_E(priv, point_cloud)||^2
  Optionally freeze eta and proprio normalizer

Expert collection:
  For t = 1 ... warmup_steps:
    a_E = pi_E(obs, priv, point_cloud)
    Store (obs, proprio_hist, a_E) or (s_E, a_E) in D_E
    Step environment with a_E

Optional BC pretrain:
  Minimize E_{D_E} ||pi_theta(s_E) - a_E||^2

For each environment step:
  s_t = build_student_state(obs_t, proprio_hist_t)
  a_t = clip(pi_theta(s_t) + exploration_noise, -1, 1)
  obs_{t+1}, done_t = M.step(a_t)
  Store (obs_t, h_t, a_t, obs_{t+1}, h_{t+1}, done_t) in B

  If online_expert:
    a_E,t = pi_E(obs_t, priv_t, point_cloud_t)
    Store teacher-labeled samples in D_E

  Repeat updates_per_env_step times:
    Sample (s,a,s',done) from B
    Sample (s_E,a_E) from D_E

    For j = 1 ... J:
      a_pi = pi_theta(s)              # current-policy action for dual
      x_hat = eps (s_E,a_E) + (1-eps) (s,a_pi)
      L_GP = E[(||grad f_phi(x_hat)||_2 - 1)^2]
      W_hat = E[f_phi(s_E,a_E)] - E[f_phi(s,a_pi)]
      phi <- Adam step on -W_hat + lambda_GP L_GP

    a_next = clip(bar pi_theta(s') + clipped_noise, -1, 1)
    r_OT = normalize_clip_scale(f_phi(s,a))
    y = r_OT + gamma (1-done) min_i bar Q_i(s',a_next)
    psi_1, psi_2 <- Adam step on Huber(Q_1(s,a),y)+Huber(Q_2(s,a),y)

    If update_index mod d == 0:
      L_actor = -E[f_phi(s,pi_theta(s))]
      If BC enabled:
        L_actor += alpha_BC E[||pi_theta(s_E)-a_E||^2]
      theta <- Adam step on L_actor
      Soft-update target Q networks and target actor
```

## Hyperparameters

### DOTPG Defaults

| Parameter | Default |
|---|---:|
| `gamma` | `0.99` |
| `lambda_gp` | `10.0` |
| `tau` | `0.005` |
| `lr_dual` | `3e-4` |
| `lr_q` | `3e-4` |
| `lr_policy` | `1e-4` |
| `batch_size` | `256` |
| `dual_updates_per_iter` | `5` |
| `policy_delay` | `2` |
| `target_policy_noise` | `0.1` |
| `target_noise_clip` | `0.2` |
| `exploration_noise` | `0.1` |
| `buffer_size` | `1,000,000` |
| `expert_buffer_size` | `500,000` |
| `warmup_steps` | `10,000` |
| `bc_coef` | `0.0` |
| entropy alpha | not implemented |

### Selected `dual_bc5` CoDrive Run

| Parameter | Value |
|---|---:|
| `task.env.numEnvs` | `1024` |
| `train.ppo.minibatch_size` | `12288` |
| `state_mode` | `student` |
| `dynamic_state` | `True` |
| `buffer_size` | `500000` |
| `expert_buffer_size` | `300000` |
| `expert_add_num_envs` | `64` |
| `warmup_steps` | `2000` |
| `batch_size` | `2048` |
| `updates_per_env_step` | `1` |
| `adapt_warmup_steps` | `500` |
| `freeze_adapt_after_warmup` | `True` |
| `freeze_sa_mean_std_after_warmup` | `True` |
| `bc_coef` | `5.0` |
| `bc_pretrain_steps` | `3000` |
| `bc_pretrain_lr` | `0.0003` |
| `bc_batch_size` | `4096` |
| `bc_alpha_min` | `0.01` |
| `bc_alpha_max` | `20.0` |
| `online_expert` | `True` |
| `reuse_expert_buffer` | `False` |
| `policy_arch` | `teacher_actor` |
| `policy_init_from_teacher` | `True` |
| `policy_output_mode` | `clamp` |
| `policy_loss_mode` | `dual` |
| `dual_state_scale` | `1.0` |
| `dual_action_scale` | `1.0` |
| `lr_policy` | `0.0001` |

## Code vs Draft: Main Differences

1. The draft presents a Gaussian/maximum-entropy policy; the code uses a
   deterministic actor.
2. The draft's stage-3 formula says `min E[Q(s,pi(s))]`; the code minimizes
   `-E[Q]`, `-E[f]`, or their mixture. With the chosen sign convention, actor
   optimization must maximize policy-side `f` or Q.
3. The implemented Q update is TD3-style twin Q with target policy smoothing,
   not the single-Q formula in the draft.
4. The successful selected path uses direct dual actor loss. Q-only actor was
   empirically weak in this task.
5. BC anchoring and BC pretraining are essential in current runs. They are not
   merely optional cosmetic additions; they keep the actor near the teacher
   action manifold.
6. Online expert refresh makes the method DAgger-like in data management.
7. Sinkhorn/entropic initialization is described in the draft but not present in
   the implementation.
8. Strong convergence, monotonic improvement, sim-to-real invariance, and sample
   complexity claims in the draft are not justified by the current nonconvex
   neural implementation or by the current experimental evidence.
9. `sim2real/codrive/dotpg_bc5/*.train.yaml` is a frozen teacher/train YAML and
   says `algo: PPO`; the real DOTPG run config is in the output run directory.

## Theorem Revision Plan

| Draft item | Recommendation | Reason |
|---|---|---|
| Theorem 1, Kantorovich-Rubinstein duality | Keep, fix typo | Foundational and consistent; Lipschitz typo should be corrected. |
| Theorem 2, general Kantorovich duality | Keep concise or move to background | Useful context, not central to implemented algorithm. |
| Entropic regularization / Sinkhorn section | Delete from method or mark as background only | No Sinkhorn critic initialization in code. |
| Theorem 3, OT deterministic policy gradient | Keep but rewrite as a proposition / semi-gradient result | Sign direction matches code, but proof ignores full occupancy derivative and relies on envelope/local assumptions. |
| Stage 1 dual critic objective | Keep with softened claims | Code matches, but GP is a soft empirical Lipschitz regularizer. |
| Stage 2 Q backup | Rewrite | Code uses twin Q, target actor, noise, done mask, normalized/clipped dual reward. |
| Stage 3 actor update | Rewrite | Primary code-faithful actor objective is direct dual + BC; Q actor is a variant. |
| Theorem 4, dual convergence to optimal Kantorovich potential | Weaken | Nonconvex NN + finite samples + GP do not guarantee convergence to `f*`. |
| Theorem 5, Wasserstein estimation error `O(lambda^-1)` / exponential | Delete or heavily qualify | General bound is not supported by implementation or proof. |
| Theorem 6, maximum entropy policy | Delete from method | No entropy term or stochastic actor in code. |
| Theorem 7, KL to soft optimal policy | Delete | Not implemented and sign/proof are not aligned. |
| Theorem 8, alternating stability | Weaken to local descent under strong assumptions | Can motivate update order, not guarantee actual monotonic training. |
| Theorem 9, two-time-scale convergence | Weaken to design rationale | Code uses fixed Adam and finite updates, not Robbins-Monro schedules. |
| Theorem 10, target network Q* bound | Delete | Bound is questionable; `O(tau^-1)` conflicts with the stability story. |
| Theorem 11, smooth target transition | Keep as lemma | Exact soft-update identity. |
| Theorem 12, transfer performance bound | Delete or move to discussion | Not supported by current experiments; avoid invariance claims. |
| Theorem 13, monotonic policy improvement | Rewrite as sufficient-condition discussion | Current off-policy nonconvex implementation does not guarantee monotonic W decrease. |
| Theorem 14, end-to-end convergence | Delete/replace with limitations | Too strong for current code and evidence. |
| Theorem 15, sample complexity | Delete unless rebuilt rigorously | Current derivation is not code-faithful or experimentally established. |

## Rewritten Theory Skeleton

### Problem

Let the teacher policy `pi_E` be a PPO policy trained in simulation. The student
has access at deployment only to non-privileged observations and proprioceptive
history. Its state representation is:

```text
s_eta(o,h) = concat(N_o(o), z_eta(N_h(h)))
```

where `z_eta` is the adapter latent. Teacher demonstrations define an empirical
state-action distribution:

```text
D_E = {(s_E, a_E)},   a_E = pi_E(o, priv, point_cloud).
```

The student induces an empirical off-policy distribution through replay:

```text
B_pi = {(s, a, s', d)}.
```

The objective is not to optimize the task reward directly. Instead, DOTPG learns
a dual potential that measures mismatch between `D_E` and current student
state-actions, then uses that potential as the actor improvement signal.

### Empirical OT Objective

With the sign convention:

```text
W_1(rho_E, rho_theta)
  = sup_{||f||_Lip <= 1} E_{rho_E}[f(x)] - E_{rho_theta}[f(x)],
  x=(s,a),
```

the neural empirical objective is:

```text
max_phi  E_{D_E}[f_phi(s_E,a_E)]
       - E_{B_pi}[f_phi(s, pi_theta(s))]
       - lambda_GP L_GP(phi).
```

The gradient penalty is an empirical soft constraint, not a proof of global
Lipschitzness:

```text
L_GP = E_{x_hat}[(||grad_x f_phi(x_hat)||_2 - 1)^2].
```

### Deterministic OT Policy Semi-Gradient

Assume the dual critic is locally optimized for the current policy and consider
the direct dependence of the policy action on `theta`. Then:

```text
grad_theta W_hat(theta)
  approximately - E_{s ~ B_pi}[
      grad_theta pi_theta(s) grad_a f_phi(s,a)|_{a=pi_theta(s)}
  ].
```

Thus descending `W_hat` corresponds to ascending:

```text
E_{s ~ B_pi}[f_phi(s, pi_theta(s))].
```

This directly justifies the implemented actor loss:

```text
L_pi^dual(theta) = - E_{s ~ B_pi}[f_phi(s, pi_theta(s))].
```

The approximation should be stated honestly: it ignores or treats as fixed the
full derivative of the state occupancy under the policy, and it relies on a
finite-sample neural critic rather than an exact Kantorovich potential.

### BC-Anchored Actor Objective

The practical actor update is:

```text
L_pi(theta)
  = - E_{s ~ B_pi}[f_phi(s, pi_theta(s))]
    + alpha_BC E_{(s_E,a_E) ~ D_E}[||pi_theta(s_E)-a_E||^2].
```

This should be described as a proximal/manifold anchor, not as pure BC. The dual
term supplies OT improvement pressure; the BC term prevents extrapolating the
actor into state-action regions where the dual critic is unreliable.

### Optional Long-Horizon Q Auxiliary

The code also learns a TD3-style value approximation to cumulative dual reward:

```text
Q_phi^pi(s,a) = E[sum_t gamma^t r_OT(s_t,a_t)],
  r_OT = normalize/clip/scale(f_phi(s,a)).
```

In code:

```text
y = r_OT(s,a)
  + gamma (1-d) min_i bar Q_i(s', clip(bar pi(s') + epsilon)).
```

Actor variants may use:

```text
L_pi^Q = -E[min_i Q_i(s, pi_theta(s))]
L_pi^{Q+dual} = -E[min_i Q_i(s, pi_theta(s))]
                - beta E[f_phi(s, pi_theta(s))]
```

However, the current evidence supports `L_pi^dual + BC` as the main thesis
version. Q should be framed as an auxiliary long-horizon extension and diagnostic
component, not as the proven core actor objective.

## Experimental Evidence To Cite

### CoDrive DOTPG Optimization

Teacher:

```text
PPO teacher best_reward = 4159.37
```

Selected DOTPG:

```text
run: theory_iter2_20260504_092416 / dual_bc5
train max_best = 2904.14
checkpoint: sim2real/codrive/dotpg_bc5/model_best_deploy.ckpt
```

Fixed 256-step deploy eval:

| Condition | avg_reward | avg_done_rate | score |
|---|---:|---:|---:|
| train_like | `4.420655` | `0.000092` | `4.236655` |
| clean | `4.496476` | `0.000122` | `4.252476` |
| light | `3.824169` | `0.000198` | `3.428169` |
| hard | `2.960864` | `0.000687` | `1.586864` |

Objective ablation:

| Candidate | train max_best | Interpretation |
|---|---:|---|
| `teacher_actor_q` | `869.14` | Q-only actor is weak. |
| `teacher_actor_dual` | `2622.18` | Direct dual is much stronger. |
| `teacher_actor_qdual_metric` | `1327.30` | Q mixture/metric scaling did not fix Q noise. |
| `dual_metric_action2` | `1386.54` | Action metric scaling hurt. |
| `dual_bc5` | `2904.14` | Selected baseline. |

BC scan:

| Candidate | train max_best | Deploy-eval interpretation |
|---|---:|---|
| `dual_bc4` | `2916.54` | Train looked good but deploy eval collapsed. |
| `dual_bc6` | `2809.14` | Worse than BC5 in deploy eval. |
| `dual_bc8` | `3004.40` | Highest train reward but worse deploy eval. |

Conclusion: DOTPG model selection should use fixed-step deploy eval, not training
`Current Best` alone.

### Broader Paper-Eval Snapshot

In `outputs/paper_codrive_thesis_full_20260506_210723/aggregate_csv/main_aggregate.csv`:

| Method | fixed_step_reward_mean | episode_return_mean_mean |
|---|---:|---:|
| `teacher_ppo` | `5.478732` | `3764.16` |
| `padapt` | `4.880223` | `3275.30` |
| `dotpg` | `3.446266` | `2330.40` |

This table is a formal eval aggregate, not a DOTPG optimization ablation. Use it
for baseline positioning, while using the DOTPG ablations above to justify the
theory/code iteration.

## Paper Wording Recommendations

Use:

- "OT-guided teacher-student imitation/distillation"
- "deterministic DOTPG actor"
- "direct dual actor objective"
- "BC-anchored actor update"
- "TD3-style auxiliary Q critic"
- "empirical Wasserstein dual estimate"

Avoid:

- "second-stage PPO/RL policy"
- "SAC / maximum entropy policy"
- "removes reward design entirely" without qualification
- "provably invariant under sim-to-real mismatch"
- "monotonic improvement guaranteed" for the implemented deep version
- "Sinkhorn-initialized critic" unless that code is actually added

## Recommended Next Theory Edit

Rewrite Section 4 of `thesis_reference/DOTPG-draft.md` around this sequence:

1. Teacher-student problem and student state representation.
2. Empirical Wasserstein dual objective.
3. Deterministic OT policy semi-gradient and sign convention.
4. BC-anchored direct-dual actor objective.
5. Optional TD3-style Q auxiliary.
6. Code-faithful Algorithm 1.
7. Replace strong theorems with cautious propositions and limitations.

The most important single correction is to replace:

```text
min_theta E[Q_psi(s, pi_theta(s))]
```

with:

```text
min_theta -E[f_phi(s, pi_theta(s))] + alpha_BC L_BC
```

for the main selected DOTPG version, and to document Q-based actor losses only
as implemented variants.
