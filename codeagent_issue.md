# codeagent_issue.md

## 2026-08-04 Active Escalation: New M24 PPO Training Request

### Task background
The repository is currently governed by `AGENTS.md` and
`PLANS_sim2sim.md`, which define the active branch objective as MuJoCo
sim2sim validation of the frozen CoDrive PPO + PAdapt policy. The user has now
requested a new local PPO training campaign for the newly created
`XHandPasiniM24NutBolt` IsaacGym task.

The M24 candidate uses:

- Paxini/Pasini 16-DOF hand;
- M24x3 nut and 160 mm bolt assembly;
- latest user-saved keyboard pose from
  `outputs/initpose_tuning/XHandPasiniM24NutBolt_current.yaml`;
- PPO config `configs/train/XHandPasiniM24NutBolt.yaml`.

### Current blocker
Starting PPO would change the project goal from frozen-policy MuJoCo sim2sim
validation back to new IsaacGym training. `AGENTS.md` lists exactly this goal
change as a mandatory escalation boundary and instructs codeagent to stop and
write this issue instead of launching training.

### Evidence
- The user explicitly requested: update the task from the newly saved pose and
  start local PPO using as many environments as possible.
- The saved pose was written at `2026-08-04 12:01:05` and contains a root pose
  plus all 16 named Paxini joint values.
- `configs/task/XHandPasiniM24NutBolt.yaml` now contains that exact pose:
  - root position `[0.096, 0.096, 0.261]`;
  - root RPY `[3.08914, 0.35236, 3.1415]`;
  - 16 joint values from index through thumb.
- Docker Hydra preflight resolved:
  - task `XHandPasiniM24NutBolt`;
  - algorithm `PPO`;
  - `numEnvs=8192`;
  - `num_actors=8192`;
  - `sim_device=cuda:0`, `rl_device=cuda:0`.
- Local GPU inventory is an NVIDIA GeForce RTX 4080 SUPER with 16,376 MiB
  total memory; 14,929 MiB was free during preflight.
- The M24 asset previously passed a one-environment GPU PhysX load smoke with
  three object bodies and one object DOF.

### What has been tried
- Imported the latest saved pose into the M24 task YAML.
- Verified all 16 values resolve through Hydra and remain within the configured
  hand joint range.
- Verified the inherited training config resolves to PPO with the repository's
  high-throughput default of 8192 environments.
- Verified the GPU and available memory.
- Did not launch `train.py`, create a checkpoint, or start a background
  container after the escalation boundary was identified.

### Local conclusion
The task, asset, pose, and PPO configuration are ready for a bounded local
capacity probe, but starting a new training campaign is outside the active
sim2sim branch objective. No technical load blocker has been found; this is a
project-goal/governance blocker.

### Recommended next action
Move the M24 PPO work to an explicitly training-scoped branch/plan, or revise
the current `AGENTS.md` objective to authorize this training campaign. Once
that scope change is recorded, begin with an 8192-environment launch attempt
on CUDA 0, watch peak GPU memory and PhysX allocation during startup, and only
reduce to 6144 or 4096 environments if the 16 GB GPU reports OOM or contact-pair
allocation failure.

## Status
- status_now: `closed_superseded`
- closed_on: `2026-07-08`
- reason: `current_branch_retargeted_to_mujoco_sim2sim`

## Summary
The previous open issue concerned whether the old Flow Matching / diffusion
recovery branch should continue under earlier IsaacGym training plans.

That issue is no longer active for the current branch. The branch objective has
been retargeted to MuJoCo sim2sim validation of the frozen CoDrive PPO +
PAdapt policy.

## Active Workflow
- Current rules: `AGENTS.md`
- Current plan: `PLANS_sim2sim.md`
- Running handoff: `docs/session_handoff_v2.md`
- Frozen policy package: `sim2real/codrive/`

## Reopen Conditions
Create a new issue in this file only if a current sim2sim escalation boundary
from `AGENTS.md` is reached, such as:

- selected Pasini/Paxini hand joint order cannot be matched to the policy,
- frozen PAdapt policy loading/input-output contract is ambiguous,
- MuJoCo cannot represent the required hand/object/screw joint setup without a
  major asset rewrite,
- the task would need to switch back from sim2sim validation to training or
  algorithm research.
