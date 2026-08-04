# MuJoCo Sim2Sim Validation Status

Date: 2026-07-08

## Verdict
Current verdict:

- `policy_runs_and_drives_hinge`

Meaning:

- The frozen CoDrive PAdapt checkpoint loads and runs in MuJoCo.
- The action mask, target integration, PD torque, 20 Hz policy loop, and task
  joint limits are functioning.
- In the more faithful 1-DOF hinge scene, the policy drives the lightbulb/nut
  hinge in the positive screw DOF direction compared with zero-action.
- The current standard validation uses the frozen original task YAML from
  `sim2real/codrive/`, not the `Exper` task YAML.
- The current passing frozen-YAML MuJoCo pose is thumb-dominant. It verifies
  policy runtime and hinge actuation, but it does not yet reproduce the
  intended index+thumb two-finger contact pattern.

This is therefore a policy-runtime pass, but not a final two-finger contact
parity pass.

## Standard Validation Command

```bash
python sim2sim/mujoco/validate_codrive_sim2sim.py \
  --dt 0.001 \
  --policy-hz 20 \
  --object-pos 0.020 -0.020 0.045 \
  --output-dir outputs/sim2sim_mujoco_validation/codrive_frozen_yaml_aligned_thumb_dt001
```

Current default validation uses:

- scene: `sim2sim/mujoco/scene_dexh13_lightbulb_hinge_fingertips.xml`
- checkpoint: `sim2real/codrive/model_best_codrive.ckpt`
- task/initpose: `sim2real/codrive/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.task.yaml`
- object root pose override: `0.020 -0.020 0.045`
- policy rate: `20 Hz`
- MuJoCo internal step: `dt=0.001`, decimation `50`
- joint limit mode: `task`
- hand root z after YAML scale compensation: `0.229`

Report generated:

- `outputs/sim2sim_mujoco_validation/codrive_frozen_yaml_aligned_thumb_dt001/validation_report.md`
- `outputs/sim2sim_mujoco_validation/codrive_frozen_yaml_aligned_thumb_dt001/validation_report.json`

## Main Numeric Result

From the standard validation report:

| Metric | Policy | Zero | Delta |
|---|---:|---:|---:|
| hinge `object_axis_delta` | `0.562009` | `0.000000` | `0.562009` |
| world yaw delta | `-0.562009` | `0.000000` | `-0.562009` |
| max contract violation | `0.000502` | `0.000502` | - |
| active contact fraction | `0.990` | `0.000` | - |
| index contact fraction | `0.000` | `0.000` | - |
| thumb contact fraction | `0.990` | `0.000` | - |
| active tip contact fraction | `0.0` | `0.0` | - |

The hinge scene uses `axis="0 0 -1"` to match
`assets/screw/contactviz/0000_lightbulb.urdf`. Positive hinge qpos therefore
appears as negative world yaw. Use `object_axis_delta`, not raw yaw, as the
primary screw-progress metric for hinge validation.

## Key Environment Fixes

Two MuJoCo-side parity fixes were required after confirming that the original
YAML and policy look normal in IsaacGym:

- Apply `handRootPosZScaleComp` from the frozen YAML. With `baseObjScale=1.2`
  and `handRootPosZScaleComp=0.06`, the effective hand root z is
  `0.217 + 0.06 * (1.2 - 1.0) ~= 0.229`. Without this, MuJoCo places the hand
  about 12 mm too low relative to the scaled bulb.
- Disable MuJoCo palm-object collision on the converted DexH13 MJCF. The
  IsaacGym training URDF intentionally omits palm collision, while the MuJoCo
  converted hand had a colliding `right_palm` mesh. Before this fix, zero-action
  rollouts could push the hinge through palm/index preload.

Older `Exper`/pre-fix artifacts remain useful only as diagnostics, not as the
standard frozen-YAML sim2sim verdict.

## Contact-Pose Findings

The tip-aware hinge scene adds the 5 mm `right_*_tip` sphere geoms present in
the IsaacGym training URDF. This improves asset parity diagnostics, but current
MuJoCo object contacts still occur on the distal/tactile mesh geoms rather than
the added tip spheres.

Three useful 200-step validation points are now preserved:

| Pose | Verdict | Axis delta gain | Active | Index | Thumb | Note |
|---|---|---:|---:|---:|---:|---|
| `0.020 -0.020 0.045` | `policy_runs_and_drives_hinge` | `0.562009` | `0.990` | `0.000` | `0.990` | frozen YAML, thumb-dominant |
| `0.012 -0.020 0.035` | `validation_failed` by weak contact | `0.030948` | `0.020` | `0.010` | `0.010` | frozen YAML, tiny two-finger overlap |
| `0.020 -0.020 0.040` | historical diagnostic | `0.398957` | `0.985` | `0.000` | `0.985` | pre-fix/old pose |
| `0.008 -0.020 0.040` | historical diagnostic | `0.904055` | `0.645` | `0.640` | `0.005` | `Exper`-era index-dominant |

The partial overlap pose is important diagnostically: it shows the frozen
policy can produce hinge progress while both index and thumb participate in
MuJoCo contacts, but the overlap is not stable enough for the current
two-finger pass threshold.

## Initpose Findings

Tested initpose/config variants:

- Frozen CoDrive task config:
  `sim2real/codrive/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.task.yaml`
- Tuned CoDriveExper config:
  `configs/task/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveExper.yaml`
- Standalone tuned initpose:
  `outputs/initpose_tuning/codrive4159_initpose.yaml`

Findings:

- Original frozen pose can run but hinge contact is not two-finger.
- The original frozen YAML now remains the standard task source. `Exper` is a
  diagnostic branch, not the default sim2sim validation config.
- The frozen-YAML aligned MuJoCo pose `0.020 -0.020 0.045` gives the current
  cleanest stable policy-vs-zero separation after environment parity fixes.
- `codrive4159_initpose.yaml` also drives the hinge positively
  (`object_axis_delta ~= 0.3946` over 80 steps), with zero static, but it is
  still thumb-only and has higher soft-limit violation (`~= 0.0156`).
- Hand-root z/pitch sweeps did not recover stable index+thumb contact in the
  hinge scene.

## Pose And Rate Findings

Hinge scene:

- Root object pose `0.02 -0.02 0.04` places the contact mesh center near
  `z ~= 0.099`.
- `dt=0.001`, decimation `50` is the current standard.
- `dt=0.002`, decimation `25` also runs and preserves positive hinge motion,
  but produces smaller 200-step axis progress (`~= 0.2477` in the checked run).

Freejoint MVP scene:

- `--object-pos 0.02 -0.02 0.1` can produce sustained index/thumb contact and
  policy-vs-zero yaw separation.
- It is less faithful to the IsaacGym screw asset because the bulb is a
  freejoint body rather than a 1-DOF nut hinge.

## Remaining Gap

The remaining gap is contact/geometry parity, not policy I/O:

- hinge scene: faithful screw-like DOF, policy drives positive screw motion
  from multiple contact regimes, but stable index+thumb contact is not yet
  recovered.
- freejoint scene: easier two-finger contact, but less faithful object joint.

The next engineering target is to recover stable index+thumb contact in the
hinge scene without changing the frozen policy. Candidate levers:

- refine hand-object mesh pose and orientation,
- tune hand root x/y/pitch together with object root pose,
- compare MuJoCo hand geometry against the IsaacGym `dexh13_hand_right_sim.urdf`
  training asset,
- inspect contact meshes visually frame-by-frame.
