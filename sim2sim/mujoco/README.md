# MuJoCo Sim2Sim MVP

This directory contains the first MuJoCo scaffold for the frozen CoDrive
PPO + PAdapt policy. It is intentionally small and diagnostic-first.

The runner borrows the deployment contract from:

- `docs/sim2sim_policy_io_audit.md`
- `/data/Codefield/py/dex-bulb/sim2real/dexh13_deploy.py`
- `xhand-deploy/xhand_deploy.py`

## Commands

Print MuJoCo joint/actuator mapping:

```bash
python sim2sim/mujoco/run_codrive_sim2sim.py --mode audit
```

Run a short zero-action PD hold:

```bash
python sim2sim/mujoco/run_codrive_sim2sim.py \
  --mode zero \
  --dt 0.001 \
  --control-decimation 50 \
  --object-pos 0 0 1.0 \
  --policy-steps 20
```

Run a one-joint poke smoke:

```bash
python sim2sim/mujoco/run_codrive_sim2sim.py \
  --mode poke \
  --dt 0.001 \
  --control-decimation 50 \
  --object-pos 0 0 1.0 \
  --poke-joint 0 \
  --poke-action 0.5 \
  --policy-steps 20
```

Run the frozen PAdapt policy closed-loop for a short smoke:

```bash
python sim2sim/mujoco/run_codrive_sim2sim.py \
  --mode policy \
  --dt 0.001 \
  --policy-hz 20 \
  --object-pos 0.012 -0.018 0.05 \
  --joint-limit-mode task \
  --policy-steps 20
```

Run the current frozen-YAML hinge contact candidate:

```bash
python sim2sim/mujoco/run_codrive_sim2sim.py \
  --mode policy \
  --scene sim2sim/mujoco/scene_dexh13_lightbulb_hinge_fingertips.xml \
  --task-config sim2real/codrive/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.task.yaml \
  --dt 0.001 \
  --policy-hz 20 \
  --object-pos 0.020 -0.020 0.045 \
  --joint-limit-mode task \
  --policy-steps 200 \
  --output-dir outputs/sim2sim_mujoco_validation/codrive_frozen_yaml_aligned_thumb_dt001_manual
```

Run the current hinge validation pack:

```bash
python sim2sim/mujoco/validate_codrive_sim2sim.py \
  --dt 0.001 \
  --policy-hz 20 \
  --object-pos 0.020 -0.020 0.045 \
  --output-dir outputs/sim2sim_mujoco_validation/codrive_frozen_yaml_aligned_thumb_dt001
```

Run the anti-shortcut gate on a known-bad full-mesh pose that visually behaves
like index rubbing:

```bash
python sim2sim/mujoco/validate_codrive_sim2sim.py \
  --scene sim2sim/mujoco/scene_dexh13_lightbulb_hinge_fingertips.xml \
  --task-config sim2real/codrive/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.task.yaml \
  --dt 0.001 \
  --policy-hz 20 \
  --object-pos 0.014 -0.023 0.022 \
  --hand-root-pos 0.106 0.028 0.245 \
  --hand-root-rpy 3.1415 0.370 3.1415 \
  --joint-limit-mode task \
  --policy-steps 120 \
  --max-contract-violation 0.02 \
  --max-zero-active-contact-fraction 1.0 \
  --anti-shortcut-gate \
  --output-dir outputs/sim2sim_mujoco_validation/anti_shortcut_gate_fullmesh_known_bad_120
```

Expected verdict: `validation_failed_single_finger_shortcut`.

Export an IsaacGym reference reset/action snapshot from the frozen CoDrive
student. This is the current source-of-truth path for diagnosing MuJoCo
single-finger rubbing. The command explicitly restores the frozen reset fields
from `sim2real/codrive/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.task.yaml`
because the same-named working config under `configs/task/` has different
reset pose values:

```bash
DEXSCREW_SIM2SIM_REFERENCE_OUT=outputs/sim2sim_isaacgym_reference/codrive_ref \
DEXSCREW_SIM2SIM_REFERENCE_STEPS=0 \
DEXSCREW_SIM2SIM_REFERENCE_POST_STEP=0 \
CUDA_VISIBLE_DEVICES=0 python train.py \
  task=Dexh13HoraLightbulbSim2RealTwoFingerCoDrive \
  train=Dexh13HoraLightbulbSim2RealTwoFingerCoDrive \
  headless=False \
  seed=0 \
  sim_device=cuda:0 \
  rl_device=cuda:0 \
  graphics_device_id=0 \
  task.env.numEnvs=1 \
  test=True \
  train.algo=ProprioAdapt \
  train.ppo.proprio_adapt=True \
  train.ppo.minibatch_size=12 \
  task.env.randomization.randomizePDGains=False \
  task.env.randomization.action_noise_e_scale=0.0 \
  task.env.randomization.action_noise_t_scale=0.0 \
  task.env.randomization.obs_noise_e_scale=0.0 \
  task.env.randomization.obs_noise_t_scale=0.0 \
  task.env.randomization.noisy_rpy_scale=0.0 \
  task.env.randomization.noisy_pos_scale=0.0 \
  task.env.forceScale=0.0 \
  task.env.randomForceProbScalar=0.0 \
  'task.env.asset.handRootPos=[0.11,0.020,0.217]' \
  'task.env.asset.handRootRPY=[3.1415,0.3,3.1415]' \
  task.env.asset.handInitPose.right_index_joint_0=0.3426670736 \
  task.env.asset.handInitPose.right_index_joint_1=0.925279522 \
  task.env.asset.handInitPose.right_index_joint_2=0.2828971177 \
  task.env.asset.handInitPose.right_index_joint_3=1.1292990017 \
  task.env.asset.handInitPose.right_middle_joint_0=0.0 \
  task.env.asset.handInitPose.right_middle_joint_1=0.0 \
  task.env.asset.handInitPose.right_middle_joint_2=0.0 \
  task.env.asset.handInitPose.right_middle_joint_3=0.0 \
  task.env.asset.handInitPose.right_ring_joint_0=0.0 \
  task.env.asset.handInitPose.right_ring_joint_1=0.0 \
  task.env.asset.handInitPose.right_ring_joint_2=0.0 \
  task.env.asset.handInitPose.right_ring_joint_3=0.0 \
  task.env.asset.handInitPose.right_thumb_joint_0=-0.3599407768 \
  task.env.asset.handInitPose.right_thumb_joint_1=1.5656383038 \
  task.env.asset.handInitPose.right_thumb_joint_2=0.220286703 \
  task.env.asset.handInitPose.right_thumb_joint_3=0.411966312 \
  wandb_activate=False \
  ++test_num_steps=1 \
  checkpoint=sim2real/codrive/model_best_codrive.ckpt
```

Replay the first exported IsaacGym action from the same state in MuJoCo:

```bash
python sim2sim/mujoco/run_codrive_sim2sim.py \
  --mode reference_action \
  --scene sim2sim/mujoco/scene_dexh13_lightbulb_hinge_isaacgym_parity.xml \
  --task-config sim2real/codrive/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.task.yaml \
  --reference-state-json outputs/sim2sim_isaacgym_reference/codrive_ref/isaacgym_ref_env0_step0000_pre_step.json \
  --joint-limit-mode task \
  --no-clamp-init-q \
  --policy-steps 1 \
  --output-dir outputs/sim2sim_mujoco_reference_replay/step0000
```

Run the same exported state with zero action as a reset/contact control:

```bash
python sim2sim/mujoco/run_codrive_sim2sim.py \
  --mode zero \
  --scene sim2sim/mujoco/scene_dexh13_lightbulb_hinge_isaacgym_parity.xml \
  --task-config sim2real/codrive/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.task.yaml \
  --reference-state-json outputs/sim2sim_isaacgym_reference/codrive_ref/isaacgym_ref_env0_step0000_pre_step.json \
  --joint-limit-mode task \
  --no-clamp-init-q \
  --policy-steps 1 \
  --output-dir outputs/sim2sim_mujoco_reference_replay/step0000_zero
```

Use only `*_pre_step.json` for replay. The runner rejects post-step snapshots
because replaying a post-step target with the same action would integrate that
action twice. `reference_action` is a first-step probe, not a complete
long-horizon state replay. If the exported IsaacGym action immediately becomes
single-index rubbing while the zero-action control is calm, the remaining gap
is likely contact/asset/reset-frame parity rather than policy loading.

Summarize a multi-step IsaacGym reference pack:

```bash
python sim2sim/mujoco/analyze_isaacgym_reference.py \
  outputs/sim2sim_isaacgym_reference/codrive_ref_cpu_20_bodycontacts \
  --output-json outputs/sim2sim_isaacgym_reference/codrive_ref_cpu_20_bodycontacts/isaacgym_reference_summary.json \
  --output-tsv outputs/sim2sim_isaacgym_reference/codrive_ref_cpu_20_bodycontacts/isaacgym_reference_steps.tsv
```

The current CPU reference baseline starts with index tactile contact, then
switches to thumb tactile contact. Therefore an initial index contact is not a
failure by itself; sustained index-pair dominance and excessive axis gain are
the MuJoCo mismatch to reduce.

For reversible MuJoCo contact diagnostics, keep the same exported state and add
contact overrides instead of editing XML by hand:

```bash
python sim2sim/mujoco/run_codrive_sim2sim.py \
  --mode policy \
  --scene sim2sim/mujoco/scene_dexh13_lightbulb_hinge_isaacgym_parity.xml \
  --task-config sim2real/codrive/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.task.yaml \
  --reference-state-json outputs/sim2sim_isaacgym_reference/codrive_ref_cpu_20_bodycontacts/isaacgym_ref_env0_step0000_pre_step.json \
  --joint-limit-mode task \
  --no-clamp-init-q \
  --policy-steps 20 \
  --object-friction 1.0 0.005 0.0001 \
  --output-dir outputs/sim2sim_mujoco_contact_profiles/refstep0_full_fric1p0_20
```

Available object/contact overrides:

- `--object-friction SLIDE TORSION ROLL`
- `--object-solref TIMECONST DAMPING`
- `--object-solimp MIN MAX WIDTH`
- `--object-condim 1|3|4|6`
- `--object-contact-pos-offset X Y Z`
- `--object-contact-z-offset VALUE`
- `--object-contact0-mesh PATH`
- `--object-contact1-mesh PATH`
- `--active-tactile-friction SLIDE TORSION ROLL`
- `--active-tactile-solref TIMECONST DAMPING`
- `--active-tactile-solimp MIN MAX WIDTH`
- `--hinge-frictionloss VALUE`

Run a fixed-reference object-contact sweep:

```bash
python sim2sim/mujoco/sweep_object_contact.py \
  --policy-steps 20 \
  --frictions '0.8,0.005,0.0001;1.0,0.005,0.0001;1.2,0.005,0.0001;1.4,0.005,0.0001' \
  --solrefs '0.008,1;0.01,1;0.015,1;0.02,1' \
  --solimps 'none;0.6,0.95,0.001;0.8,0.98,0.001' \
  --output-dir outputs/sim2sim_mujoco_contact_sweeps/object_contact_refstep0_fric_solref_solimp_48
```

Compare whole contact0 mesh variants:

```bash
python sim2sim/mujoco/sweep_object_contact.py \
  --policy-steps 20 \
  --contact0-meshes 'none;assets/lightbulb/smooth_head_collision.stl;assets/lightbulb/rounded_contact_head.stl' \
  --frictions '0.8,0.005,0.0001;1.2,0.005,0.0001;1.6,0.005,0.0001' \
  --solrefs '0.008,1;0.015,1' \
  --solimps '0.8,0.98,0.001' \
  --output-dir outputs/sim2sim_mujoco_contact_sweeps/object_contact_refstep0_mesh_fric_solref_18
```

The current best diagnostic candidate is not a final pass, but is useful for
viewer inspection:

```bash
python sim2sim/mujoco/run_codrive_sim2sim.py \
  --mode policy \
  --scene sim2sim/mujoco/scene_dexh13_lightbulb_hinge_isaacgym_parity.xml \
  --task-config sim2real/codrive/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.task.yaml \
  --reference-state-json outputs/sim2sim_isaacgym_reference/codrive_ref_cpu_20_bodycontacts/isaacgym_ref_env0_step0000_pre_step.json \
  --joint-limit-mode task \
  --no-clamp-init-q \
  --policy-steps 0 \
  --viewer \
  --finger-contact-mode full \
  --object-friction 0.8 0.005 0.0001 \
  --object-solref 0.015 1 \
  --object-solimp 0.8 0.98 0.001 \
  --output-dir outputs/sim2sim_mujoco_manual/refstep0_best_contact_viewer
```

Open a live MuJoCo viewer for the current policy candidate:

```bash
python sim2sim/mujoco/run_codrive_sim2sim.py \
  --mode policy \
  --scene sim2sim/mujoco/scene_dexh13_lightbulb_hinge_fingertips.xml \
  --task-config sim2real/codrive/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.task.yaml \
  --dt 0.001 \
  --policy-hz 20 \
  --object-pos 0.020 -0.020 0.045 \
  --joint-limit-mode task \
  --policy-steps 0 \
  --viewer \
  --output-dir outputs/sim2sim_mujoco_manual/policy_frozen_yaml_aligned_thumb_viewer
```

`--policy-steps 0` keeps the viewer rollout running until the MuJoCo window is
closed or the terminal receives Ctrl+C.

Render policy/zero videos for the current hinge candidate:

```bash
python sim2sim/mujoco/validate_codrive_sim2sim.py \
  --dt 0.001 \
  --policy-hz 20 \
  --object-pos 0.020 -0.020 0.045 \
  --render-video \
  --render-every 2 \
  --output-dir outputs/sim2sim_mujoco_validation/codrive_frozen_yaml_aligned_thumb_dt001_video
```

Outputs are written under `outputs/sim2sim_mujoco_mvp/` by default.

Run environment-alignment sweeps:

```bash
python sim2sim/mujoco/sweep_env_alignment.py \
  --policy-steps 10 \
  --modes zero \
  --dt-decimations 0.001:50,0.002:25,0.005:10 \
  --object-positions '0,0,1.0;0.012,-0.018,0.05;0.012,-0.018,0.08;0.012,-0.018,0.12' \
  --joint-limit-modes task
```

See `docs/sim2sim_env_alignment.md` for the current aligned baseline and
latest sweep results.

## Important Limitations

- The scene is an MVP scene, not a final physics match.
- The frozen IsaacGym contract is 20 Hz policy over 200 Hz physics. With the
  current MuJoCo MJCF, `dt=0.005, decimation=10` can destabilize hand contacts.
  The stable diagnostic smoke uses `dt=0.001, decimation=50`, preserving the
  same 20 Hz policy interval while separating MuJoCo numerical stability from
  policy behavior.
- It includes the existing `dexh13_right_fixed.xml`, but the runner bypasses
  the default 13 MuJoCo actuators and writes 16 joint torques through
  `qfrc_applied` to preserve the IsaacGym-style PD torque contract.
- The runner overwrites MuJoCo joint limits with the frozen task YAML
  `dofLowerLimits` / `dofUpperLimits` so middle/ring locked fingers match the
  CoDrive action mask. MuJoCo limits are still soft constraints, so small
  locked-joint violation should be inspected in the CSV logs.
- The runner applies the frozen YAML `handRootPosZScaleComp` against
  `baseObjScale`. For the CoDrive package this raises the hand root z from
  `0.217` to about `0.229`, matching IsaacGym reset semantics for the 1.2x
  lightbulb scale.
- The MuJoCo DexH13 palm mesh is visual-only for object contact parity because
  the IsaacGym training URDF intentionally omits palm collision.
- The lightbulb object uses the repository STL contact meshes and nominal
  mass/friction values; object pose/contact tuning still needs validation.
- `--object-pos 0 0 1.0` moves the bulb away for mechanics smoke tests. Remove
  that override when tuning the actual contact start pose.
- The current aligned contact-start candidate is
  `--object-pos 0.012 -0.018 0.05`.
- The current behavior candidate is `--object-pos 0.02 -0.02 0.1`; it gives
  sustained index/thumb contact in the MVP scene and is documented in
  `docs/sim2sim_env_alignment.md`.
- Newer anti-shortcut checks supersede the loose MVP contact candidate. Use
  reset-contact sweeps before trusting a viewer pose:

```bash
python sim2sim/mujoco/sweep_reset_contact.py \
  --scene sim2sim/mujoco/scene_dexh13_lightbulb_hinge_fingertips.xml \
  --task-config sim2real/codrive/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.task.yaml \
  --dt 0.001 \
  --finger-contact-mode tip_proxy \
  --object-x-values 0.010,0.012,0.014,0.016,0.018 \
  --object-y-values=-0.032,-0.028,-0.024,-0.020,-0.016 \
  --object-z-values 0.018,0.020,0.022,0.024,0.026 \
  --hand-z-values 0.245,0.249,0.253,0.257 \
  --hand-pitch-values 0.330,0.350,0.370,0.390,0.410 \
  --output-dir outputs/sim2sim_mujoco_alignment/reset_contact_tipproxy_grid_dt001_v2
```

The corrected `tip_proxy` disables all active index/thumb non-tip meshes, not
just distal link 2/3 meshes. If this sweep reports index-only or thumb-only
reset contact, inspect asset/contact parity before judging policy behavior.
- The more faithful validation scene is
  `scene_dexh13_lightbulb_hinge_fingertips.xml`; it uses a 1-DOF hinge
  matching the lightbulb URDF screw/nut structure and adds the 5 mm fingertip
  sphere geoms from the training hand URDF for contact diagnostics. Current
  verdict is documented in `docs/sim2sim_validation_status.md`.
- Current hinge validation passes for both thumb-dominant and index-dominant
  poses. Stable two-finger index+thumb contact remains a contact-parity gap,
  not a policy I/O failure.
- Hinge rotation alone is not success. Use `--anti-shortcut-gate` to reject
  poses dominated by one index geom/tip contact pair.
- The first verdict should come only after audit, zero-action, poke, and short
  policy traces are inspected.
