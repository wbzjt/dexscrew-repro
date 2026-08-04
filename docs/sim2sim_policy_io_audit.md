# MuJoCo Sim2Sim Policy I/O Audit

Date: 2026-07-08

## Scope
This audit fixes the runtime contract for the current branch task:

- frozen package: `sim2real/codrive/`
- primary student: `sim2real/codrive/model_best_codrive.ckpt`
- teacher reference: `sim2real/codrive/best_reward_4159.37.pth`
- task config: `sim2real/codrive/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.task.yaml`
- train config: `sim2real/codrive/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.train.yaml`

The objective is MuJoCo sim2sim validation, not PPO/PAdapt retraining.

## External References Checked
The useful public pattern is to make deployment contracts explicit:

- NVIDIA Isaac Lab sim-to-sim notes identify joint/link ordering differences as
  a primary transfer issue and solve it with observation/action remapping:
  https://isaac-sim.github.io/IsaacLab/main/source/experimental-features/newton-physics-integration/sim-to-sim.html
- Unitree RL Gym uses a `Train -> Play -> Sim2Sim -> Sim2Real` workflow and a
  MuJoCo deploy script with explicit `simulation_dt`, `control_decimation`,
  `action_scale`, default joint pose, and PD torque:
  https://github.com/unitreerobotics/unitree_rl_gym
  https://raw.githubusercontent.com/unitreerobotics/unitree_rl_gym/main/deploy/deploy_mujoco/deploy_mujoco.py
- MuJoCo actuators are SISO controls whose scalar control is mapped to force
  through the selected transmission; this matters when choosing between
  `d.ctrl`, position actuators, motors, and direct `qfrc_applied`:
  https://mujoco.readthedocs.io/en/stable/computation/index.html#actuation-model
- MuJoCo `dcmotor` can interpret controls as voltage, position, or velocity
  targets when configured that way; do not assume `ctrl` means an IsaacGym-style
  joint torque:
  https://mujoco.readthedocs.io/en/stable/XMLreference.html#actuator-dcmotor
- MuJoCo contact softness is controlled via `solref`/`solimp`, and friction
  coefficients differ between geom-level 3-tuples and contact-pair 5-tuples:
  https://mujoco.readthedocs.io/en/stable/modeling.html#solver-parameters
  https://mujoco.readthedocs.io/en/stable/modeling.html#contact-parameters

## Local Deploy Reference
The prior local project `/data/Codefield/py/dex-bulb` contains a useful DexH13
deployment path:

- `/data/Codefield/py/dex-bulb/sim2real/dexh13_deploy.py`
  - 16-DOF DexH13/Pasini policy loop
  - `ACTION_SCALE = 0.05`
  - locked middle/ring indices `4:12`
  - reset-filled 30-frame proprio history
  - 20 Hz policy timing decoupled from hardware read/write speed
- `/data/Codefield/py/dex-bulb/sim2real/export_padapt_single_pt.py`
  - lightweight PAdapt checkpoint wrapper
  - direct use of `actor_mlp`, `mu`, and `adapt_tconv`
  - TorchScript export without needing IsaacGym at deployment time

The MuJoCo MVP in `sim2sim/mujoco/` follows this DexH13 deploy structure while
using direct `.ckpt` loading first.

## Checkpoint Contract
Torch checkpoint inspection:

- top-level keys:
  `model`, `running_mean_std`, `sa_mean_std`, `priv_mean_std`,
  `point_cloud_mean_std`
- `running_mean_std.running_mean`: `(96,)`
- `running_mean_std.running_var`: `(96,)`
- `sa_mean_std.running_mean`: `(30, 32)`
- `sa_mean_std.running_var`: `(30, 32)`
- `priv_mean_std.running_mean`: `(71,)`
- `point_cloud_mean_std.running_mean`: `(3,)`
- `model.sigma`: `(16,)`
- `model.mu.weight`: `(16, 128)`
- `model.actor_mlp.mlp.0.weight`: `(512, 136)`
- `model.env_mlp.mlp.0.weight`: `(256, 71)`
- `model.adapt_tconv.channel_transform.0.weight`: `(32, 32)`

Implication:

- action dimension is 16
- policy observation vector is 96
- proprioceptive history is 30 frames x 32 features
- privileged info is checkpointed but not required for PAdapt inference
- actor input is 136 = 96 observation + 40 adapted latent
- adapted latent is 40 because PAdapt uses 8 env-latent channels plus 32
  point-cloud latent channels

Loader warning:

- The bundled train YAML still says `algo: PPO` and `proprio_adapt: False`.
  That YAML is not sufficient by itself to instantiate the frozen student.
- Student PAdapt scripts enable the correct runtime through Hydra overrides:
  `train.algo=ProprioAdapt` and `train.ppo.proprio_adapt=True`.
- A direct MuJoCo loader must therefore construct the actor as PAdapt stage 2,
  with `proprio_adapt=True`, before loading `model_best_codrive.ckpt`.
  Loading it as PPO/teacher will build the wrong model contract.

## Inference Inputs
The direct `.ckpt` runtime should reproduce `dexscrew/algo/ppo/padapt.py`:

```python
input_dict = {
    "obs": running_mean_std(obs),
    "proprio_hist": sa_mean_std(proprio_hist),
    "point_cloud_info": point_cloud_mean_std(point_cloud_info.reshape(-1, 3)).reshape(1, -1, 3),
}
mu, extrin, extrin_gt = model.act_inference(input_dict)
action = torch.clamp(mu, -1.0, 1.0)
```

Required input shapes for one MuJoCo environment:

- `obs`: `(1, 96)`
- `proprio_hist`: `(1, 30, 32)`
- `point_cloud_info`: `(1, 100, 3)`

The PAdapt action path depends on normalized `obs` and normalized
`proprio_hist`. `point_cloud_info` is still required by the model call, but in
stage-2 PAdapt the action uses the temporal adapter output. Use zeros for the
first MuJoCo MVP unless an exact point-cloud source is added later.

Normalization must use the checkpoint buffers with `eval()` semantics:

- raw `obs` should first be clipped to `[-5, 5]`, matching `VecTask.step`
- subtract running mean
- divide by `sqrt(running_var + 1e-5)`
- clamp normalized values to `[-5, 5]`

## Observation And History
The IsaacGym task builds observations from a lag buffer, not from a single
state sample:

- one frame = `[joint_pos_16, current_target_16]`, so frame dim is 32
- `proprio_hist` = last 30 frames, shape `(30, 32)`
- `obs` = last 3 frames, flattened to 96
- at reset, the lag buffer is filled with the initial hand pose for both joint
  position and target position

For MuJoCo:

1. Set `prev_target = init_qpos_16`.
2. Fill the 30-frame history with `[init_qpos_16, init_qpos_16]`.
3. At each policy tick, form `obs` from the last 3 history frames.
4. After physics/control updates, append `[measured_qpos_16, current_target_16]`.

The old `xhand-deploy/xhand_deploy.py` is a runtime pattern reference, but it is
12-DOF xHand-specific. Do not reuse its joint mapping or `ACTION_SCALE`
unchanged for this 16-DOF CoDrive policy.

## Action And Control
The frozen task config says:

- `numActions: 16`
- `apply_action_mask: True`
- `action_mask_indices: [4, 5, 6, 7, 8, 9, 10, 11]`
- active fingers are index actions `0:4` and thumb actions `12:16`
- middle and ring actions are ignored
- `sim.dt: 0.005`
- `controlFrequencyInv: 10`
- physics/control loop: 200 Hz
- policy loop: 20 Hz
- `action_scale: 0.05`
- `pgain: 3`
- `dgain: 0.01`
- `torque_limit: 300.0`

MuJoCo control should therefore do:

```python
action = clamp(policy_mu, -1, 1)
action[4:12] = 0
target = prev_target + 0.05 * action
target = clip(target, lower_limits, upper_limits)
tau = 3.0 * (target - qpos) - 0.01 * qvel
tau = clip(tau, -300.0, 300.0)
prev_target = target
```

Run the low-level torque at every MuJoCo physics step and refresh policy action
every 10 physics steps.

MuJoCo stability note:

- The frozen IsaacGym contract is 20 Hz policy over 200 Hz physics
  (`dt=0.005`, decimation `10`).
- The current `dexh13_right_fixed.xml` MuJoCo asset is numerically more stable
  for smoke tests at smaller internal timesteps such as `dt=0.001`,
  decimation `50`, which preserves the same 20 Hz policy interval.
- Treat `dt=0.001/50` as a diagnostic substep setting, not proof that the
  final physics contract is solved. The final verdict should explicitly report
  which MuJoCo timestep/decimation was used.

Important actuator choice:

- The safest first MVP is direct joint torque semantics, either through a
  16-actuator MJCF or by writing joint-space torques through MuJoCo state APIs.
- Do not evaluate the frozen policy first through the current
  `assets/dexh13_right_description2/urdf/dexh13_right_fixed.xml` actuator set
  without documenting the change: that XML has 16 joints but only 12 motors, and
  couples distal index/middle/ring joints through tendons. That is a different
  actuator/action space from the IsaacGym policy.

## Joint Order
Policy order for the 16-DOF hand is:

1. `right_index_joint_0`
2. `right_index_joint_1`
3. `right_index_joint_2`
4. `right_index_joint_3`
5. `right_middle_joint_0`
6. `right_middle_joint_1`
7. `right_middle_joint_2`
8. `right_middle_joint_3`
9. `right_ring_joint_0`
10. `right_ring_joint_1`
11. `right_ring_joint_2`
12. `right_ring_joint_3`
13. `right_thumb_joint_0`
14. `right_thumb_joint_1`
15. `right_thumb_joint_2`
16. `right_thumb_joint_3`

This order appears in:

- `dexscrew/tasks/xhand_pasini.py`
- `assets/dexh13_right_description2/config/joint_names_dexh13_right.yaml`
- `assets/dexh13_hand/config/joint_names_dexh13_right.yaml`
- `assets/dexh13_right_description2/urdf/dexh13_right*.xml`
- `assets/dexh13_hand/urdf/dexh13_hand_right_sim.urdf`

Still log and assert MuJoCo `model.joint(i).name`, `qposadr`, `dofadr`, and
actuator transmission names at runner startup. A name-based mapping should be
mandatory even if the initial local order appears aligned.

## Reset Pose And Object Contract
Use the frozen task config values before adding MuJoCo tuning:

- hand asset used by IsaacGym training:
  `assets/dexh13_hand/urdf/dexh13_hand_right_sim.urdf`
- hand root position: `[0.11, 0.020, 0.217]`
- hand root RPY: `[3.1415, 0.3, 3.1415]`
- object type: `screw_contactviz`
- object initial position: `[0.012, -0.018, 0.0]`
- rotation axis: `+z`
- base object scale: `1.20`
- training scale support roughly covers `1.15` to `1.25`
- mass support: `0.04` to `0.06`
- friction support: `1.0` to `5.0`
- restitution support: `0.0` to `0.05`

For the first deterministic MuJoCo smoke, use nominal values inside these
ranges, for example scale `1.20`, mass `0.05`, friction around the middle of the
training range, and no extra pose/action/observation noise. Then add a small
grid over the training support.

## False-Failure Checklist
Treat these as environment/config confounders before judging policy ability:

- wrong 12-DOF xHand mapping reused for the 16-DOF Pasini/Paxini policy
- using `xhand-deploy` `ACTION_SCALE = 0.04167` instead of frozen CoDrive
  `action_scale = 0.05`
- forgetting the CoDrive action mask `4:12`
- initializing `obs`/`proprio_hist` with zeros instead of reset pose history
- using position actuators or tendon motors with semantics different from
  IsaacGym direct PD torque
- running a different policy/control frequency than 20 Hz over 200 Hz physics
- MuJoCo model has different joint axes, sign, limits, or order
- object/lightbulb/screw joint axis sign differs from IsaacGym `rotation_axis: +z`
- contact geoms are too sharp/soft/sticky/slippery because `friction`,
  `solref`, `solimp`, `condim`, or collision meshes differ
- using the 12-motor `dexh13_right_fixed.xml` actuator coupling as the first
  policy verdict
- changing hand root pose, object scale, object mass, or object COM outside the
  frozen config support
- adding MuJoCo solver/contact tuning before a zero-action and one-joint poke
  smoke has passed

## Recommended Runner Sequence
1. Load MuJoCo scene and log joint/actuator order.
2. Assert a name-based policy-to-MuJoCo joint map.
3. Run zero-action hold with PD torque only; target must remain finite and
   stable.
4. Run one-joint poke tests for each active index/thumb joint to verify sign,
   limit, and qpos address.
5. Load checkpoint directly and run one synthetic reset-pose inference.
6. Compare the direct checkpoint output against any future TorchScript export
   before using the export in the runner.
7. Run closed-loop policy with deterministic nominal contact values.
8. Add logging for action, masked action, target, qpos, qvel, torque,
   contact forces, object/nut angle, and reset reason.
9. Only after stable nominal rollout, sweep friction/contact/object values
   inside the original training randomization support.
