# MuJoCo Sim2Sim Environment Alignment

Date: 2026-07-08

## Purpose
This note records the current environment-configuration alignment for the
frozen CoDrive PPO + PAdapt MuJoCo runner.

The goal is to reduce false failures caused by setup mismatch before judging
policy ability.

## Aligned Baseline
Current mechanics/rate diagnostic baseline:

```bash
python sim2sim/mujoco/run_codrive_sim2sim.py \
  --mode policy \
  --scene sim2sim/mujoco/scene_dexh13_lightbulb_hinge_fingertips.xml \
  --task-config sim2real/codrive/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.task.yaml \
  --dt 0.001 \
  --policy-hz 20 \
  --object-pos 0.020 -0.020 0.045 \
  --joint-limit-mode task \
  --policy-steps 200
```

Latest frozen-YAML aligned result:

- output: `outputs/sim2sim_mujoco_validation/codrive_frozen_yaml_aligned_thumb_dt001/`
- no NaN
- `policy_runs_and_drives_hinge`
- `policy_rate_hz = 20.0`
- `max_abs_action = 1.0`
- `max_abs_tau ~= 0.904`
- `max_contract_violation ~= 5.0e-4`
- policy/zero `object_axis_delta`: `0.562009 / 0.000000`
- active contact fraction: `0.990 / 0.000`
- index/thumb contact fraction: `0.000 / 0.990`

Equivalent low-level rate:

- MuJoCo internal timestep: `0.001`
- control decimation derived from `--policy-hz 20`: `50`
- policy interval: `0.05 s`
- policy inference rate: `20 Hz`

This preserves the frozen policy timing while using smaller MuJoCo substeps for
numerical stability.

Important frozen-YAML parity details now implemented:

- The task source is `sim2real/codrive/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.task.yaml`.
- `configs/task/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveExper.yaml` is a
  diagnostic/tuned branch only.
- MuJoCo applies the original YAML scale compensation:
  `handRootPos.z = 0.217 + 0.06 * (1.2 - 1.0) ~= 0.229`.
- The converted MuJoCo DexH13 palm mesh is visual-only for object contact,
  matching the IsaacGym training URDF where palm collision is intentionally
  omitted.

Current hinge validation candidate:

```bash
python sim2sim/mujoco/validate_codrive_sim2sim.py \
  --dt 0.001 \
  --policy-hz 20 \
  --object-pos 0.020 -0.020 0.045 \
  --output-dir outputs/sim2sim_mujoco_validation/codrive_frozen_yaml_aligned_thumb_dt001
```

This uses `sim2sim/mujoco/scene_dexh13_lightbulb_hinge_fingertips.xml`, where
the bulb nut is a 1-DOF hinge matching the repository lightbulb URDF screw/nut
structure and the hand includes 5 mm fingertip sphere geoms matching the
training URDF. Current frozen-YAML verdict:

- `policy_runs_and_drives_hinge`
- policy hinge `object_axis_delta`: `0.562009`
- zero hinge `object_axis_delta`: `0.000000`
- max contract violation: `0.000502`
- active contact fraction: `0.990`
- index contact fraction: `0.000`
- thumb contact fraction: `0.990`

This is a policy-runtime and hinge-actuation pass, but still a contact-parity
gap because contact is thumb-dominant rather than index+thumb.

See `docs/sim2sim_validation_status.md` for the current validation report and
remaining contact-parity gap.

## Pose Alignment
Frozen task values:

- hand root position: `[0.11, 0.020, 0.217]`
- effective MuJoCo/IsaacGym reset hand root after scale compensation:
  `[0.11, 0.020, 0.229]`
- hand root RPY: `[3.1415, 0.3, 3.1415]`
- object init position from task config: `[0.012, -0.018, 0.0]`

MuJoCo freejoint position uses a world-space z value. The current MVP scene uses
`z=0.05` as the first mechanics/contact smoke candidate:

- `--object-pos 0.012 -0.018 0.05`

Longer behavior diagnostics showed that this low-z pose is thumb dominated:

- 200-step policy trace:
  `outputs/sim2sim_mujoco_behavior/policy_aligned_200/policy_trace.csv`
- active contact fraction: `0.46`
- index contact fraction: `0.0`
- thumb contact fraction: `0.46`
- object yaw delta: `-0.950 rad`
- object final z: `0.031`

The current contact-behavior candidate is higher and slightly shifted:

- `--object-pos 0.02 -0.02 0.1`

Coarse and fine pose sweeps used:

- `outputs/sim2sim_mujoco_contact_pose/coarse_xyz_20/alignment_summary.tsv`
- `outputs/sim2sim_mujoco_contact_pose/fine_xyz_around_002_m002_50/alignment_summary.tsv`

Useful rejected candidates:

- `0.035 -0.005 0.1`: strong short-horizon two-finger contact, but the object
  falls to `z ~= 0.031` over 200 policy steps and zero-action also produces
  large passive motion.
- `0.015 -0.025 0.11`: good 50-step contact/violation tradeoff, but the
  200-step policy run loses the stable pose and drifts to `z ~= 0.031`.

No-contact mechanics smoke remains available with:

- `--object-pos 0 0 1.0`

## Rate Alignment
All tested rate settings preserve a 20 Hz policy interval, but not all are
numerically stable with the current MJCF:

| dt | decimation | policy Hz | zero-action result |
|---:|---:|---:|---|
| 0.001 | 50 | 20 | stable |
| 0.002 | 25 | 20 | stable |
| 0.005 | 10 | 20 | unstable for current MJCF |

Use `dt=0.001, decimation=50` as the first contact-evaluation setting. Keep
`dt=0.005, decimation=10` as a later parity target after contact/asset tuning,
not as the first verdict setting.

At the current contact candidate `0.02 -0.02 0.1`, a 50-step rate sweep showed:

- `0.001:50` and `0.002:25` preserve the same contact/yaw behavior.
- `0.005:10` is invalid for the current MJCF contact scene:
  `max_contract_violation ~= 4.21`, object drift `~= 4.32`.

Detailed table:

- `outputs/sim2sim_mujoco_contact_pose/candidate_002_m002_010_rate_sweep_50/alignment_summary.tsv`

## Joint-Limit Alignment
Default mode:

- `--joint-limit-mode task`

This overwrites MuJoCo joint ranges using the frozen task YAML
`dofLowerLimits` / `dofUpperLimits`, so middle and ring fingers are constrained
near zero consistently with the CoDrive action mask.

Important details:

- The frozen hand init pose has one value slightly outside the task lower
  limit; the runner clips init q by default and records `init_q_clipped_count`.
- MuJoCo joint limits remain soft constraints. The runner logs
  `max_contract_violation` in JSON and CSV outputs.
- `--hard-clamp-joints` exists only as a diagnostic. Do not use it for the main
  behavior verdict unless explicitly reporting that hard clamp was enabled.
- `--joint-limit-mode model` is useful as a comparison against raw MJCF ranges,
  but it is not the default CoDrive-aligned setting.

For the current contact candidate, the 200-step policy trace has
`max_contract_violation ~= 0.0109`. The largest per-joint violations are:

- `right_middle_joint_0`: `0.01093`
- `right_middle_joint_1`: `0.00738`
- `right_thumb_joint_1`: `0.00537`

The middle and ring fingers have zero object contact fraction in this trace.
The locked-finger violation is therefore treated as a MuJoCo soft-limit/model
coupling diagnostic, not as unintended middle/ring object manipulation.

Hard-clamp diagnostic at the same pose:

- policy hard-clamp yaw: `-0.726 rad`
- zero hard-clamp yaw: `0.359 rad`
- clamp events are very high (`41610` policy, `47324` zero)

Because hard clamp changes passive dynamics substantially, it is not part of
the main sim2sim verdict.

## Sweep Results
Commands run:

```bash
python sim2sim/mujoco/sweep_env_alignment.py \
  --policy-steps 10 \
  --modes zero \
  --dt-decimations 0.001:50,0.002:25,0.005:10 \
  --object-positions '0,0,1.0;0.012,-0.018,0.05;0.012,-0.018,0.08;0.012,-0.018,0.12' \
  --joint-limit-modes task \
  --output-dir outputs/sim2sim_mujoco_alignment/zero_pose_rate_task
```

```bash
python sim2sim/mujoco/sweep_env_alignment.py \
  --policy-steps 5 \
  --modes policy \
  --dt-decimations 0.001:50,0.002:25 \
  --object-positions '0.012,-0.018,0.05;0.012,-0.018,0.08' \
  --joint-limit-modes task \
  --output-dir outputs/sim2sim_mujoco_alignment/policy_pose_rate_task
```

Key zero-action findings:

- `dt=0.001/50`, object `(0.012,-0.018,0.05)`:
  `max_contract_violation ~= 5.0e-4`
- `dt=0.002/25`, object `(0.012,-0.018,0.05)`:
  `max_contract_violation ~= 5.0e-4`
- `dt=0.005/10`, object `(0.012,-0.018,0.05)`:
  `max_contract_violation ~= 2.10`

Key short policy findings:

- `dt=0.001/50`, object `(0.012,-0.018,0.05)`:
  `max_contract_violation ~= 4.9e-4`, `max_abs_tau ~= 0.425`
- `dt=0.001/50`, object `(0.012,-0.018,0.08)`:
  `max_contract_violation ~= 4.9e-4`, `max_abs_tau ~= 0.413`
- `dt=0.002/25`, object `(0.012,-0.018,0.05)`:
  `max_contract_violation ~= 4.9e-4`, `max_abs_tau ~= 0.426`
- 20-step aligned baseline at `dt=0.001/50`, object `(0.012,-0.018,0.05)`:
  `max_contract_violation ~= 5.0e-4`, `max_abs_tau ~= 0.889`

Detailed tables:

- `outputs/sim2sim_mujoco_alignment/zero_pose_rate_task/alignment_summary.tsv`
- `outputs/sim2sim_mujoco_alignment/policy_pose_rate_task/alignment_summary.tsv`
- `outputs/sim2sim_mujoco_alignment/zero_limit_modes/alignment_summary.tsv`
- `outputs/sim2sim_mujoco_contact_pose/coarse_xyz_20/alignment_summary.tsv`
- `outputs/sim2sim_mujoco_contact_pose/fine_xyz_around_002_m002_50/alignment_summary.tsv`
- `outputs/sim2sim_mujoco_contact_pose/candidate_002_m002_010_rate_sweep_50/alignment_summary.tsv`

## Current Passive-Parity Candidate
After aligning the MuJoCo hand's passive joint parameters to the IsaacGym
torque-control contract, the current best frozen-YAML hinge candidate is:

```bash
python sim2sim/mujoco/run_codrive_sim2sim.py \
  --mode policy \
  --viewer \
  --policy-steps 0 \
  --scene sim2sim/mujoco/scene_dexh13_lightbulb_hinge_fingertips.xml \
  --task-config sim2real/codrive/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.task.yaml \
  --dt 0.001 \
  --policy-hz 20 \
  --object-pos 0.014 -0.020 0.020 \
  --hand-root-pos 0.106 0.028 0.253 \
  --hand-root-rpy 3.1415 0.390 3.1415 \
  --joint-limit-mode task \
  --output-dir outputs/sim2sim_mujoco_manual/passive_parity_viewer
```

The corresponding non-viewer validation is:

```bash
python sim2sim/mujoco/validate_codrive_sim2sim.py \
  --scene sim2sim/mujoco/scene_dexh13_lightbulb_hinge_fingertips.xml \
  --task-config sim2real/codrive/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.task.yaml \
  --dt 0.001 \
  --policy-hz 20 \
  --object-pos 0.014 -0.020 0.020 \
  --hand-root-pos 0.106 0.028 0.253 \
  --hand-root-rpy 3.1415 0.390 3.1415 \
  --joint-limit-mode task \
  --policy-steps 200 \
  --max-contract-violation 0.02 \
  --max-zero-active-contact-fraction 1.0 \
  --min-axis-gain 0.2 \
  --min-two-finger-contact-fraction 0.1 \
  --min-index-thumb-overlap-fraction 0.1 \
  --output-dir outputs/sim2sim_mujoco_validation/passive_parity_aligned_validation_dt001_200
```

Result:

- verdict: `policy_runs_with_two_finger_contact`
- policy/zero `object_axis_delta`: `0.873338 / -0.020660`
- policy-minus-zero axis gain: `0.893997`
- policy index/thumb contact fraction: `0.990 / 0.610`
- policy index-thumb simultaneous overlap: `0.605`
- zero index-thumb simultaneous overlap: `0.000`
- fingertip-sphere contact is still `0.000`, so remaining asset parity work is
  specifically about tip/tactile rigid-body/contact semantics.

See `docs/sim2sim_asset_parity.md` for the URDF-vs-MJCF audit.

## Visual Inspection Note
The high-overlap passive-parity candidate above can look twitchy in the live
viewer because it is a strong contact pose. Its 200-step trace has saturated
actions and nontrivial contact impulses, so it is useful as a contact/axis
validation point but not necessarily as the calmest visual baseline.

For a lower-impulse visual check, use this softer pose:

```bash
python sim2sim/mujoco/run_codrive_sim2sim.py \
  --mode policy \
  --viewer \
  --viewer-realtime-factor 0.5 \
  --policy-steps 0 \
  --scene sim2sim/mujoco/scene_dexh13_lightbulb_hinge_fingertips.xml \
  --task-config sim2real/codrive/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.task.yaml \
  --dt 0.001 \
  --policy-hz 20 \
  --object-pos 0.014 -0.023 0.022 \
  --hand-root-pos 0.106 0.028 0.245 \
  --hand-root-rpy 3.1415 0.370 3.1415 \
  --joint-limit-mode task \
  --output-dir outputs/sim2sim_mujoco_manual/visual_softer_overlap_viewer
```

Short sweep metrics for this pose over 120 policy steps:

- `object_axis_delta ~= 0.967`
- `index/thumb contact fraction ~= 0.97 / 0.48`
- `index_thumb_overlap_fraction ~= 0.45`
- `mean_active_contact_force ~= 8.93`
- `max_active_contact_force ~= 36.77`
- `max_contract_violation ~= 2.3e-4`

This is still a contact-rich MuJoCo simulation, but it should be less violent
than the strongest validation candidates.

Follow-up visual inspection showed that this softer full-mesh pose can still
look like the index finger is rubbing the bulb and rotating the hinge in place.
Trace evidence supports that observation:

- full mesh contact pairs are dominated by
  `right_index_link_3/geom_8 <-> codrive_lightbulb_contact0`
- `tip_proxy` removes the active index/thumb side meshes, but the policy still
  mostly drives the hinge from `right_index_tip`
- this means the current hinge scene is useful for checking policy loading and
  axis actuation, but it is not yet a visually faithful two-finger CoDrive
  behavior baseline

Best current `tip_proxy` diagnostic command:

```bash
python sim2sim/mujoco/run_codrive_sim2sim.py \
  --mode policy \
  --viewer \
  --viewer-realtime-factor 0.5 \
  --policy-steps 0 \
  --finger-contact-mode tip_proxy \
  --scene sim2sim/mujoco/scene_dexh13_lightbulb_hinge_fingertips.xml \
  --task-config sim2real/codrive/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.task.yaml \
  --dt 0.001 \
  --policy-hz 20 \
  --object-pos 0.016 -0.026 0.020 \
  --hand-root-pos 0.108 0.026 0.249 \
  --hand-root-rpy 3.1415 0.370 3.1415 \
  --joint-limit-mode task \
  --output-dir outputs/sim2sim_mujoco_manual/tipproxy_thumb_participation_viewer
```

Short sweep metrics for this tip-only diagnostic:

- `object_axis_delta ~= 1.65`
- index/thumb contact fraction `~= 0.80 / 0.42`
- simultaneous index-thumb overlap `~= 0.275`
- mean/max active contact force `~= 4.15 / 15.86`
- `max_contract_violation ~= 2.3e-4`

This is a diagnostic, not a final parity baseline. It shows that thumb can be
made to participate under a stricter contact proxy, but the policy still finds
a single-finger/tip-friction route too easily.

## Anti-Shortcut Validation Gate
The validator now has an optional anti-shortcut gate:

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
  --min-axis-gain 0.2 \
  --min-two-finger-contact-fraction 0.1 \
  --min-index-thumb-overlap-fraction 0.1 \
  --anti-shortcut-gate \
  --output-dir outputs/sim2sim_mujoco_validation/anti_shortcut_gate_fullmesh_known_bad_120
```

This known-bad full-mesh pose rotates the hinge and passes the loose two-finger
overlap check, but the anti-shortcut gate correctly rejects it:

- verdict: `validation_failed_single_finger_shortcut`
- dominant contact pair:
  `right_index_link_3/geom_8@right_index_link_3<->codrive_lightbulb_nut/codrive_lightbulb_contact0`
- dominant contact-pair step fraction: `0.966667`
- index-only contact fraction: `0.516667`
- index/thumb contact fraction: `0.966667 / 0.483333`
- index-thumb overlap fraction: `0.450000`
- policy-minus-zero axis gain: `0.991287`

This is the correct interpretation: hinge rotation plus loose overlap is not
enough. A valid MuJoCo visual baseline must also avoid dominant single-index
contact.

## Next Step
Use the passive-parity and tip-proxy candidates above only as diagnostics, then
inspect:

- object pose drift and contact count
- action saturation
- target versus qpos tracking
- locked-finger violation
- torque traces

Recommended trace analysis command:

```bash
python sim2sim/mujoco/analyze_trace.py \
  outputs/sim2sim_mujoco_validation/passive_parity_aligned_validation_dt001_200/policy/policy_trace.csv
```
