---
applyTo: '**'
---

# DexScrew Quick Context For Code Agents

## 1. Current branch identity
- Active task: MuJoCo sim2sim validation.
- Goal: deploy and inspect the frozen CoDrive PPO + PAdapt policy in MuJoCo.
- Non-goal: new IsaacGym training, new distillation, diffusion/flow sweeps, or
  cloud training.

## 2. Current workflow files
- Stable agent rules: `AGENTS.md`
- Current plan: `PLANS_sim2sim.md`
- Running handoff: `docs/session_handoff_v2.md`
- Historical plans: `PLANS_v*.md` and old stage summaries are archive context
  only.

## 3. Frozen policy package
Primary package:

- `sim2real/codrive/`

Primary artifacts:

- PPO teacher: `sim2real/codrive/best_reward_4159.37.pth`
- PAdapt student: `sim2real/codrive/model_best_codrive.ckpt`
- task YAML:
  `sim2real/codrive/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.task.yaml`
- train YAML:
  `sim2real/codrive/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.train.yaml`

Reference-only package:

- `sim2real/codrive/dotpg_bc5/`

DOTPG BC5 can be used later for comparison. The default target is PPO +
PAdapt.

## 4. Main code references
- IsaacGym training/eval entry, preserved but not default for this branch:
  `train.py`
- PAdapt implementation:
  `dexscrew/algo/ppo/padapt.py`
- PPO/teacher implementation:
  `dexscrew/algo/ppo/ppo.py`
- Shared Hora task contract:
  `dexscrew/tasks/xhand_hora.py`
- Pasini/Paxini-related task code:
  `dexscrew/tasks/xhand_pasini.py`
- External deployment runtime reference:
  `xhand-deploy/xhand_deploy.py`

## 5. Deployment semantics to preserve
The MuJoCo runner should reproduce the policy runtime contract:

- 20 Hz policy loop.
- 200 Hz physics loop.
- policy action clamped to `[-1, 1]`.
- CoDrive action mask from the frozen task YAML.
- target integration:
  `target = prev_target + action_scale * action`.
- explicit PD torque:
  `tau = pgain * (target - q) - dgain * qvel`.
- target clipping to joint limits.
- PAdapt-compatible observation/history buffers.

Use `xhand-deploy/xhand_deploy.py` for reference on normalization, history
buffers, action masking, target integration, and joint-order mapping, but adapt
it to the selected Pasini/Paxini MuJoCo body and CoDrive policy dimensions.

## 6. First sim2sim implementation path
1. Inventory the frozen CoDrive artifacts.
2. Confirm selected Pasini/Paxini hand asset and joint order.
3. Build a minimal MuJoCo scene and log joint/actuator names.
4. Run zero-action stability smoke.
5. Add PAdapt policy loading.
6. Run action/target/controller trace logging.
7. Add behavior metrics and video/plot outputs.

## 7. Environment notes
- Existing root `requirements.txt` is IsaacGym-oriented and does not include
  MuJoCo.
- Add MuJoCo-specific dependencies in a sim2sim-local requirements file if a
  runner is created.
- Keep IsaacGym scripts usable, but do not treat them as the active workflow
  unless the user explicitly asks.
