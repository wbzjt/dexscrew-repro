# DOTPG Project Overview For Paper And Agent Handoff

Date: 2026-05-26

Purpose: provide a compact, non-code project overview for agents or reviewers
who need to understand the original project context, the current
teacher-student pipeline, and the role of DOTPG before revising theory,
experiments, or manuscript text.

## One-Paragraph Summary

This project studies sim-to-real dexterous manipulation with a teacher-student
architecture. A privileged PPO teacher is first trained in IsaacGym on the
DexH13 lightbulb manipulation task. The teacher can use richer simulation
signals, including privileged object information and point-cloud-derived
features, to learn a strong manipulation policy. A student policy is then
distilled for deployment from proprioceptive history and adapter-based latent
state construction. The current DOTPG branch replaces the original PAdapt-style
supervised student trainer with an OT-guided deterministic student
distillation algorithm: it learns a Wasserstein dual critic between teacher and
student state-action samples, then updates the student actor mainly through a
direct dual-potential policy gradient with behavior-cloning anchoring.

## Project Lineage

The codebase originates from a HORA-style dexterous-hand reinforcement learning
pipeline. The canonical structure is:

```text
task environment -> PPO teacher -> student distillation -> evaluation/export
```

The original student baseline is PAdapt: the PPO teacher is frozen, and the
student learns an adaptation module from proprioceptive history using action
behavior cloning and latent distillation. In the current DOTPG-focused line,
the PPO teacher stage is preserved, but the student-training stage is changed
from mostly supervised PAdapt-style distillation to OT-guided actor adaptation.

Important framing:

- DOTPG is the current student distillation method, not a second-stage PPO
  policy.
- The upstream PPO teacher remains the source of expert behavior.
- DOTPG is deterministic and off-policy; it is not SAC and does not implement
  a stochastic maximum-entropy actor.
- The current selected DOTPG version is `dual_bc5`: teacher-actor
  initialization, direct dual actor loss, and BC anchoring.

## Main Task Context

The main DOTPG development task is:

```text
Dexh13HoraLightbulbSim2RealTwoFingerCoDrive
```

The task uses a DexH13 hand to manipulate a lightbulb-like object in simulation
with sim-to-real randomization. The current CoDrive setup emphasizes a
two-finger manipulation pattern: index finger and thumb are active, while other
finger actions are masked or treated as non-primary for the task. The policy
acts in a continuous action space and is evaluated under fixed-step rollout
protocols with clean and randomized perturbation settings.

Typical task characteristics:

- continuous 16-dimensional action interface;
- two-finger active manipulation emphasis;
- torque-control style simulation;
- episode length around 800 simulation steps;
- privileged teacher observations include richer object/context features;
- student-side state relies on observable/proprioceptive inputs and an adapter
  latent;
- sim-to-real robustness is encouraged through randomization of object and
  sensor/contact-related properties.

The exact numeric state dimensions and environment details are documented in
the code-faithful DOTPG framework. For paper writing, the key conceptual point
is that the teacher has access to stronger simulation information, while the
student is trained for a more deployable observation regime.

## Teacher-Student Architecture

### Stage 1: PPO Teacher

The first stage trains a PPO teacher in simulation. The teacher is the
performance reference and the expert data source for student training.

The PPO teacher:

- learns directly from the environment reward;
- can use privileged simulation information;
- contains the actor backbone later reused by DOTPG;
- provides action labels for expert buffers;
- produces the checkpoint used as the fixed expert for distillation.

Representative CoDrive teacher artifact:

```text
sim2real/codrive/best_reward_4159.37.pth
```

The teacher should be described as the upstream RL policy. Student algorithms
should be compared against it as a high-capability reference, not as a directly
deployable final policy in all observation regimes.

### Stage 2: DOTPG Student

The second stage trains a student policy from the frozen PPO teacher. In the
current DOTPG path, the student does not optimize the original task reward as
its main learning signal. Instead, DOTPG builds an imitation/adaptation signal
from optimal transport between teacher and student state-action distributions.

The implemented DOTPG student uses:

- a deterministic actor initialized from the PPO teacher actor;
- a student-side adapter latent from proprioceptive history;
- an expert buffer populated by frozen teacher rollouts;
- a replay buffer populated by student interaction;
- a Wasserstein dual critic over state-action samples;
- direct actor updates that increase the dual potential on student actions;
- BC anchoring to keep the actor near the teacher action manifold;
- TD3-style twin-Q machinery as an implemented auxiliary/variant, not the main
  selected actor objective.

The selected actor objective should be written conceptually as:

```text
min_theta  - E[f_phi(s, pi_theta(s))]
           + alpha_BC E[||pi_theta(s_E) - a_E||^2]
```

where `f_phi` is the learned OT dual potential and `(s_E, a_E)` are teacher
expert samples.

## DOTPG Algorithm Positioning

DOTPG treats imitation as state-action occupancy matching. The empirical
Wasserstein estimate is:

```text
W_hat = E_{rho_E}[f_phi(s,a)] - E_{rho_pi}[f_phi(s,a)]
```

The dual critic is trained to maximize this difference with a Lipschitz-style
gradient penalty. Since the policy side appears with a negative sign in
`W_hat`, reducing the Wasserstein distance with respect to the student means
increasing `f_phi(s, pi_theta(s))` on current policy actions.

That sign convention is the central link between code and theory:

```text
actor minimizes -E[f_phi(s, pi_theta(s))]
```

The early DOTPG implementation included a Q-based actor route:

```text
Q(s,a) <- f_phi(s,a) + gamma Q_target(s', pi_target(s'))
actor minimizes -E[Q(s, pi_theta(s))]
```

In the CoDrive lightbulb task, this Q-only route was much less reliable than
the direct dual route. Therefore the current paper-facing DOTPG should present
direct dual actor learning with BC anchoring as the main method, and present
the Q route only as an implemented auxiliary or ablation variant.

## Experimental Design

The current project experiments are organized around the following sequence.

1. Train or select a PPO teacher.
   The teacher is trained with environment reward and privileged information.
   The best teacher checkpoint is frozen.

2. Collect expert data.
   The frozen teacher generates state-action samples. These samples populate
   the DOTPG expert buffer and also support BC warmup/anchoring.

3. Initialize the student.
   The selected DOTPG variant reuses the PPO teacher actor architecture and
   initializes the student actor from the teacher. This keeps the student near
   a known valid manipulation manifold.

4. Warm up student state construction.
   Adapter/state-normalization and BC pretraining stabilize the initial
   student distribution before off-policy DOTPG updates dominate.

5. Train DOTPG.
   Each iteration alternates between student environment interaction, dual
   critic updates, Q updates, and delayed actor/target updates. The selected
   actor update is direct dual plus BC anchoring.

6. Evaluate with fixed-step rollouts.
   Training reward alone is not reliable for DOTPG model selection. Candidate
   checkpoints are compared with fixed-step evaluation under `train_like`,
   `clean`, `light`, and `hard` perturbation settings.

7. Package deploy candidates.
   Selected checkpoints are preserved with matching teacher checkpoints and
   task/train YAMLs for visualization, export, or further comparison.

## Current DOTPG Baseline

The current CoDrive DOTPG baseline is:

```text
sim2real/codrive/dotpg_bc5/
```

Selected run:

```text
theory_iter2_20260504_092416 / dual_bc5
```

Core settings:

```text
policy_arch = teacher_actor
policy_init_from_teacher = True
policy_output_mode = clamp
policy_loss_mode = dual
bc_coef = 5.0
bc_alpha_max = 20.0
dual_state_scale = 1.0
dual_action_scale = 1.0
lr_policy = 0.0001
```

Why this version was selected:

- Q-only actor learning was weak on the contact-rich lightbulb task.
- Direct dual actor learning aligned better with the deterministic OT policy
  gradient and improved training reward substantially.
- BC anchoring prevented the student from drifting away from the teacher action
  manifold.
- A BC scan showed that training reward alone could be misleading; `bc4` and
  `bc8` looked competitive by training reward but degraded in deploy-style
  fixed-step evaluation.

Key CoDrive DOTPG optimization evidence:

| Candidate | Max Train Best | Interpretation |
|---|---:|---|
| `teacher_actor_q` | `869.14` | Q-only actor objective was weak. |
| `teacher_actor_dual` | `2622.18` | Direct dual actor update was much stronger. |
| `teacher_actor_qdual_metric` | `1327.30` | Q/metric mixture did not fix the issue. |
| `dual_metric_action2` | `1386.54` | Action metric scaling hurt. |
| `dual_bc5` | `2904.14` | Selected deploy-evaluated baseline. |

Selected `dual_bc5` fixed-step 256 evaluation:

| Condition | Avg Reward | Avg Done Rate |
|---|---:|---:|
| `train_like` | `4.420655` | `0.000092` |
| `clean` | `4.496476` | `0.000122` |
| `light` | `3.824169` | `0.000198` |
| `hard` | `2.960864` | `0.000687` |

## Baseline Context

For the current DOTPG-centered manuscript work, the relevant baselines are:

- PPO teacher: upper reference and expert-data source.
- PAdapt: original student distillation baseline from the source project.
- DOTPG `dual_bc5`: current OT-guided student method under analysis.

Other student families exist in the repository, including pure BC, DAgger, and
diffusion/flow/consistency variants. They can be mentioned as broader project
comparison methods if needed, but they are not central to the current DOTPG
theory revision. Avoid spending manuscript-method space on them unless the
experiment table requires context.

Important comparison caveat:

- Formal paper-grade ranking should use fixed-step, preferably multi-seed
  evaluation.
- Single-run `Current Best` training reward is useful for screening but not a
  final deploy-quality metric.
- Existing DOTPG evidence strongly supports the direct-dual-plus-BC design
  over Q-only DOTPG, but stronger paper claims still need multi-seed DOTPG
  evaluation if DOTPG is promoted from baseline to primary contribution.

## What The Project Has Implemented

At a high level, the project already implements:

- PPO teacher training and checkpoint selection;
- teacher-student distillation entrypoints;
- DOTPG student training with expert and replay buffers;
- Wasserstein dual critic with gradient penalty;
- direct dual actor objective;
- TD3-style twin-Q backup machinery;
- BC pretraining and actor anchoring;
- teacher-actor initialization for the DOTPG student;
- fixed-step DOTPG evaluation output compatible with checkpoint selection;
- deploy packages containing teacher checkpoint, student checkpoint, and YAMLs.

For a paper-writing agent, the most important implementation fact is not the
file layout. It is the algorithmic story:

```text
PPO learns the manipulation skill with privileged simulation information.
DOTPG distills that skill into a student by matching teacher and student
state-action occupancy through an OT dual potential, while BC anchoring keeps
the actor close to the expert manifold.
```

## Current Writing Guidance

Use this framing:

- "two-stage teacher-student framework";
- "privileged PPO teacher";
- "OT-guided deterministic student distillation";
- "Wasserstein dual critic over state-action samples";
- "direct dual actor update with BC anchoring";
- "TD3-style Q backup as auxiliary/variant";
- "fixed-step deploy-oriented evaluation".

Avoid this framing:

- "DOTPG is a second-stage PPO policy";
- "DOTPG is SAC or maximum-entropy RL";
- "Q-only actor optimization is the main successful method";
- "training reward alone proves deployment quality";
- "DOTPG guarantees sim-to-real transfer";
- "all theorem claims from the original draft are already code-faithful".

## Recommended Upload Bundle

For an agent working on DOTPG paper writing or algorithm iteration, upload:

```text
docs/dotpg_project_overview.md
docs/dotpg_codrive_optimization.md
docs/dotpg_code_faithful_theory_framework.md
thesis_reference/DOTPG-draft.md
```

If the agent must verify implementation details, also include:

```text
dexscrew/dotpg/dotpg.py
dexscrew/dotpg/networks.py
dexscrew/dotpg/buffer.py
```

## Recommended Next Paper Step

Use this overview to write the paper's system/method context before rewriting
the DOTPG math. A clean paper structure is:

1. Task and teacher-student problem setup.
2. PPO teacher and expert-data generation.
3. Student observation/adaptation setting.
4. DOTPG Wasserstein dual objective.
5. Direct dual actor update with BC anchoring.
6. Training and evaluation protocol.
7. Baseline comparison: PPO teacher, PAdapt, DOTPG.

