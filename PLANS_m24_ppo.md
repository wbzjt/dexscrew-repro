# M24 Paxini PPO Plan

## 1. Current objective

Train a new IsaacGym PPO teacher for:

`XHandPasiniM24NutBolt -> PPO teacher -> checkpoint/evaluation`

The task uses:

- Paxini/Pasini 16-DOF hand;
- M24x3 black-oxide hex nut;
- 160 mm M24 bolt;
- 36 mm nut across flats;
- 21.5 mm nut thickness;
- user-saved keyboard init pose;
- 20 Hz policy control and 200 Hz physics.

MuJoCo sim2sim and previous Hora diffusion/student work are paused. They remain
historical context and must not be mixed into the active training run.

## 2. Source and branch contract

- Active branch: `diffusion`.
- M24 source snapshot: `sim2sim@159d994`.
- Import only M24 assets/configs, Pasini pose support, task registration, and
  init-pose utilities.
- Do not import `sim2sim/`, MuJoCo assets, or sim2sim-specific documentation.
- Preserve existing diffusion/Hora entrypoints.

## 3. Training configuration

- Task: `XHandPasiniM24NutBolt`.
- Algorithm: PPO.
- GPU: local NVIDIA RTX 4080 SUPER, CUDA 0, 16 GB.
- First capacity attempt:
  - `task.env.numEnvs=8192`
  - `train.ppo.num_actors=8192`
  - `train.ppo.minibatch_size=16384`
- Backoff order only on observed allocation/OOM failure:
  1. 6144 envs / 12288 minibatch
  2. 4096 envs / 8192 minibatch
- Headless training with a persistent log and resumable checkpoint directory.
- No wall-clock timeout unless the user specifies one; training remains active
  until completion, failure, or explicit stop.

## 4. Milestones

### M0: Governance and selective migration

Completion conditions:

- `AGENTS.md` points to this plan and M24 PPO path.
- Required M24 files are present on `diffusion`.
- No MuJoCo sim2sim directory is imported.

### M1: Configuration and scene smoke

Completion conditions:

- Python and shell syntax checks pass.
- Hydra resolves task/train configs and all 16 saved joint values.
- One-environment GPU PhysX reset loads:
  - 16 hand DOFs;
  - three object rigid bodies;
  - one nut DOF;
  - no NaN/non-finite reset state.

### M2: Capacity probe

Completion conditions:

- Attempt 8192 environments first.
- Confirm startup reaches PPO iteration output with finite metrics.
- Record GPU memory, selected environment count, minibatch size, and initial
  FPS.
- Reduce capacity only with explicit OOM/PhysX/contact allocation evidence.

### M3: Persistent PPO run

Completion conditions:

- Run in a persistent local session with exact command and log path.
- Record start timestamp, PID/session, git commit, output directory, and active
  capacity.
- Verify checkpoints begin appearing under `stage1_nn/`.

### M4: Training monitoring and first verdict

Completion conditions:

- Track reward, episode length, done/reset rate, FPS, NaN/errors, and GPU use.
- Preserve the best PPO checkpoint.
- Run headed evaluation after a meaningful checkpoint exists.
- Decide whether pose/reward/contact behavior is acceptable before any student
  or diffusion follow-up.

## 5. Acceptance criteria

- M24 asset and saved pose are exactly the configured training inputs.
- Training runs without NaN, repeated simulation reset explosions, or silent
  fallback to a different object/task.
- The chosen environment count is the largest locally demonstrated stable
  setting from the defined probe order.
- At least one valid PPO checkpoint and reproducible log are preserved.
- Success is judged by both reward trends and headed nut-rotation behavior.

## 6. Current next step

Commit and push the completed M0/M1 migration baseline, then run the monitored
8192-environment capacity probe. Back off only on observed allocation failure.
