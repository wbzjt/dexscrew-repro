# AGENTS.md

## Scope
This file tells codeagent how to work in this repo.
Keep it short, stable, and execution-oriented.

The active branch task is now **MuJoCo sim2sim validation**, not new IsaacGym
training or algorithm development.

## Canonical path
Default active path:

`sim2real/codrive PPO teacher + PAdapt student -> Pasini/Paxini hand MuJoCo runner -> sim2sim validation`

The frozen reference policy package is:

- `sim2real/codrive/`
  - teacher PPO: `best_reward_4159.37.pth`
  - primary PAdapt student: `model_best_codrive.ckpt`
  - task YAML: `Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.task.yaml`
  - train YAML: `Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.train.yaml`

`sim2real/codrive/dotpg_bc5/` is a useful deploy-eval reference, but the
default branch objective is PPO + PAdapt unless the user explicitly switches
policy.

## Current objective
The goal is to validate the frozen CoDrive policy in MuJoCo.

Do not restart PPO, retrain PAdapt, run new student distillation, or continue
old diffusion/robustness sweeps unless the user explicitly asks for training.

## Current plan file
Use `PLANS_sim2sim.md` for current priorities, milestones, and acceptance
criteria.

Historical files such as `PLANS_v*.md`, `docs/stage_acceptance_summary.md`,
and old cloud handoffs are archive context only. Read them only when needed to
explain provenance or recover a specific artifact.

## What codeagent should do
Focus on local sim2sim execution work:

- read only the files needed for the current task
- protect frozen artifacts under `sim2real/codrive/`
- prefer small, reversible changes
- build MuJoCo runner code in a separate sim2sim area rather than changing
  IsaacGym training code
- preserve existing train/eval/export entrypoints
- reference `xhand-deploy/xhand_deploy.py` for deployment semantics:
  normalization, proprio history, action masking, target integration, and
  joint-order handling
- treat the Pasini/Paxini hand naming and joint order as a must-confirm
  integration contract before control code is finalized

## MuJoCo sim2sim contract
The MuJoCo runner should reproduce the IsaacGym policy interface before
tuning physics:

- policy rate: 20 Hz
- physics step: 200 Hz (`dt=0.005`, 10 physics steps per policy step)
- policy action is normalized and clamped to `[-1, 1]`
- action updates target position, not direct torque:
  `target = prev_target + action_scale * action`
- low-level control should initially be explicit PD torque:
  `tau = pgain * (target - q) - dgain * qvel`
- clamp targets to the selected hand joint limits
- mask inactive CoDrive fingers as defined by the frozen task YAML
- maintain the same observation/history convention used by the PAdapt policy

If using a TorchScript export, remember that external runtimes may need to
perform observation normalization explicitly. Confirm this from the model and
reference runtime before assuming the traced module normalizes inputs.

## Validation order
For MuJoCo work, prefer this order:

1. Artifact inventory: confirm selected teacher/student/config files exist.
2. Scene load: Pasini/Paxini hand + CoDrive object load in MuJoCo.
3. Zero-action smoke: stable reset and no exploding physics.
4. Policy I/O parity: same synthetic or recorded inputs produce expected
   policy action shapes and ranges.
5. Closed-loop smoke: policy runs for a short rollout without NaN or actuator
   saturation.
6. Behavior metrics: record nut angle/velocity, contact summaries, action and
   target traces, reset reasons, and video when available.
7. Only after the above, tune MuJoCo contact/friction/actuator parameters.

## What codeagent can handle directly
Codeagent may directly do:

- MuJoCo scene and runner scaffolding
- policy loader and observation/history builder wiring
- action scaling, masking, target integration, and PD controller code
- asset path fixes and joint-order mapping
- smoke tests, logging, plotting, and small validation scripts
- documentation updates tied to real sim2sim changes

## What codeagent should avoid
Avoid by default:

- new IsaacGym training
- cloud training orchestration
- new PPO/PAdapt/DOTPG/diffusion sweeps
- changing frozen `sim2real/codrive/` artifacts
- using old Plan v2 diffusion gates as current decision rules
- judging MuJoCo success from reward alone before policy I/O and control
  parity are confirmed

## When codeagent must escalate
Stop and write `codeagent_issue.md` if any of the following happens:

- the selected Pasini/Paxini hand asset cannot be matched to the frozen
  CoDrive policy action/joint order
- the frozen PAdapt policy cannot be loaded or its input/output contract is
  ambiguous after direct inspection
- MuJoCo cannot represent the required hand/object/screw joint setup without a
  major asset rewrite
- the task would require changing the project goal from sim2sim validation back
  to training or algorithm research
- external proprietary assets/repos become necessary and are not present

## Required issue format
If escalation is needed, `codeagent_issue.md` must contain:

- task background
- current blocker
- evidence
- what has been tried
- local conclusion
- recommended next action

## Session handoff
After each meaningful execution session, update:

- `docs/session_handoff_v2.md`

The handoff update must include:

- target milestone/subgoal
- what changed, including files and behavior impact
- what was verified, including commands and key outcomes
- what remains blocked/risky
- the single recommended next step

## Session bootstrap
At the beginning of a new execution session, read:

- `docs/session_handoff_v2.md`
- `PLANS_sim2sim.md`

Before code changes or new experiments:

- confirm the latest single recommended next step
- confirm the selected frozen policy artifact
- avoid rerunning already-failed MuJoCo settings unless testing a clear fix
  hypothesis
- record the current sim2sim milestone/subgoal in the first execution update

## Continuous execution preference
Unless the user explicitly pauses or redirects, continue routine local sim2sim
validation work without asking for per-probe confirmation.

Stop and report when:

- a MuJoCo runner milestone is clearly achieved
- a significant behavior/parity result appears
- an escalation boundary is triggered
- the user asks to stop or reprioritize
