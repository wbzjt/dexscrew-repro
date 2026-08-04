# DexH13 Sim2Sim Asset Parity

Date: 2026-07-08

## Scope
This note tracks IsaacGym-to-MuJoCo asset parity for the frozen CoDrive
PPO+PAdapt policy.

Frozen IsaacGym task asset:

- `sim2real/codrive/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.task.yaml`
- `assets/dexh13_hand/urdf/dexh13_hand_right_sim.urdf`

Current MuJoCo hand asset:

- `assets/dexh13_right_description2/urdf/dexh13_right_fixed_fingertips.xml`
- diagnostic IsaacGym-style fixed-body include:
  `assets/dexh13_right_description2/urdf/dexh13_right_isaacgym_parity_fingertips.xml`

## Confirmed Alignment
- The policy uses the frozen original YAML under `sim2real/codrive`, not the
  historical `Exper` YAML.
- The MuJoCo runner applies the same 20 Hz policy interval used by the frozen
  policy. The stable diagnostic setting is `dt=0.001`, decimation `50`.
- The runner applies the frozen YAML hand-root z scale compensation:
  `0.217 + 0.06 * (1.2 - 1.0) ~= 0.229`.
- The MuJoCo palm mesh is visual-only for contact with
  `contype="0" conaffinity="0"`, matching the IsaacGym URDF where palm
  collision was intentionally removed to avoid concave-shell self-collision.
- The MuJoCo fingertip scene contains 5 mm sphere geoms named
  `right_index_tip`, `right_middle_tip`, `right_ring_tip`, and
  `right_thumb_tip`. Their distal-frame positions match the URDF fixed
  tactile-link plus tip offsets closely enough for contact diagnostics.
- MuJoCo compiled DOF passive parameters now match the IsaacGym torque-control
  contract:
  - `dof_damping = 0`
  - `joint_stiffness = 0`
  - `dof_frictionloss = 0.01`
  - `dof_armature = 0.001`

The passive parameter fix is important: before this, the MuJoCo hand carried
extra passive spring/damping/friction from the MJCF defaults, so the frozen
policy was not seeing the same effort-mode joint dynamics as IsaacGym.

## Remaining Differences
- IsaacGym loads `dexh13_hand_right_sim.urdf` with
  `collapse_fixed_joints=False`, so tactile links and fingertip links are
  rigid bodies.
- The MuJoCo hand collapses tactile links into mesh geoms on parent bodies.
  Fingertips are sphere geoms on distal bodies rather than separate fixed
  bodies.
- IsaacGym uses PhysX convex decomposition from submeshes and
  `thickness=0.001`; MuJoCo uses mesh collision with MJCF `solimp/solref` and
  friction settings. These are not physically identical contact models.
- MuJoCo XML contains 13 actuators and tendon coupling for some distal joints,
  but the CoDrive runner intentionally bypasses them and applies 16-DOF torque
  through `data.qfrc_applied`. The 13-actuator path should not be used as the
  default policy-control interface.
- Self-collision filtering is not proven to be 1:1. The current MuJoCo XML has
  explicit palm-vs-link0/link1 excludes; IsaacGym relies on the URDF importer
  and asset collision semantics.

## Current Verification
Asset audit:

```bash
python sim2sim/mujoco/run_codrive_sim2sim.py \
  --mode audit \
  --scene sim2sim/mujoco/scene_dexh13_lightbulb_hinge_fingertips.xml \
  --task-config sim2real/codrive/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.task.yaml \
  --dt 0.001 \
  --policy-hz 20 \
  --joint-limit-mode task \
  --output-dir outputs/sim2sim_mujoco_debug/passive_parity_audit
```

Compiled result:

- all 16 policy joints report `dof_damping=0`
- all 16 policy joints report `joint_stiffness=0`
- all 16 policy joints report `dof_frictionloss=0.01`
- all 16 policy joints report `dof_armature=0.001`

Policy-vs-zero validation after passive parity:

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
- fingertip-sphere contact is still `0.000`, so contact is currently carried
  by distal/tactile mesh geoms rather than the added tip spheres.

Follow-up visual inspection and contact-pair logging showed that this
high-overlap pass should not be treated as final behavior parity:

- the full-mesh scene can rotate the hinge mostly from index distal/tactile
  mesh rubbing
- `tip_proxy` removes active side meshes and lowers impulses, but still allows
  mostly index-tip-driven hinge rotation
- therefore, current MuJoCo validation proves policy I/O, timing, passive DOF
  parameters, and hinge actuation, but not yet IsaacGym-like two-finger
  manipulation behavior

The runner/analyzer/validator now expose anti-shortcut metrics:

- `dominant_object_contact_pair`
- `dominant_object_contact_pair_step_fraction`
- `dominant_object_contact_pair_event_fraction`
- `index_only_contact_fraction`
- `thumb_only_contact_fraction`
- `index_thumb_tip_overlap_fraction`

Known-bad validation with `--anti-shortcut-gate` rejects the full-mesh pose
that visually looked like index rubbing:

- output:
  `outputs/sim2sim_mujoco_validation/anti_shortcut_gate_fullmesh_known_bad_120/`
- verdict: `validation_failed_single_finger_shortcut`
- dominant pair step fraction: `0.966667`
- index-only contact fraction: `0.516667`

## Interpretation
The frozen policy and YAML are not the primary problem. The MuJoCo asset had a
real dynamics mismatch in passive joint parameters, and fixing it recovers
stable policy-driven hinge motion with simultaneous index+thumb contact in the
current hinge scene.

This is still not a perfect IsaacGym asset replica. The next fidelity question
is no longer just whether mesh-contact parity is sufficient. The current hinge
scene is too permissive to single-finger frictional rotation. For visual
CoDrive parity, the next engineering step should either rebuild the MuJoCo hand
with URDF-like fixed fingertip/tactile bodies and better contact filtering, or
revise the object/contact constraint so a single index contact cannot look like
a successful two-finger screw motion.

## Reset-Contact And Proxy Profile Update

The stricter contact profiles now distinguish three diagnostic modes:

- `full`: all compiled hand mesh contacts are available.
- `active_pad_proxy`: active index/thumb structural meshes are disabled for
  object contact; tactile pad meshes and 5 mm tip spheres remain.
- `tip_proxy`: all active index/thumb non-tip meshes are disabled; only
  `right_index_tip` and `right_thumb_tip` spheres can drive object contact.

Important new evidence:

- The earlier `tip_proxy` implementation only disabled distal link 2/3 mesh
  contacts. It has been fixed to disable all active index/thumb non-tip meshes.
- Local reset sweeps near the original MuJoCo pose still produce 2500/2500
  index-only tip precontacts under the corrected `tip_proxy`, so the previous
  viewer behavior was not a policy-only issue.
- A broader reset sweep under corrected `tip_proxy` found no true tip-only
  index+thumb reset candidates: 10477 no-active-contact, 1477 thumb-only, and
  2446 index-only reset states.
- A no-reset-contact policy candidate
  (`object=(0.020,-0.030,0.040)`, `hand=(0.100,0.020,0.229)`,
  `pitch=0.330`) runs the frozen policy cleanly but becomes thumb-only:
  thumb contact fraction about `0.59`, index contact fraction `0.0`, and
  dominant pair
  `right_thumb_link_3/right_thumb_tip<->codrive_lightbulb_nut/codrive_lightbulb_contact1`.
- The same candidate under `active_pad_proxy` becomes thumb tactile-pad-only
  with high contact force and near-zero useful rotation.
- The old index-rubbing pose under fixed `active_pad_proxy` is still dominated
  by `right_index_tactile_link_2`, with reset precontact already present.

Conclusion: current MuJoCo behavior can be made index-only or thumb-only by
pose changes, but the existing substitute MJCF/contact model has not reproduced
the frozen IsaacGym two-finger contact semantics. The next parity asset should
be derived from the frozen training hand URDF
`assets/dexh13_hand/urdf/dexh13_hand_right_sim.urdf`; the current scene uses
`assets/dexh13_right_description2/urdf/dexh13_right_fixed_fingertips.xml`,
which is useful for diagnostics but is no longer sufficient as the final
sim2sim hand asset.

## 2026-07-09 Reset-Source Audit

The frozen YAML is not the complete reset source of truth for the IsaacGym
viewer run. In `dexscrew/tasks/xhand_pasini.py`, `screwdriver_inclined` has
hardcoded reset constants that differ from the YAML:

- active init q differs from `env.asset.handInitPose`
  - `right_index_joint_1`: code `1.2325279`, YAML `0.9252795`
  - `right_thumb_joint_2`: code `0.5520287`, YAML `0.2202867`
- hand root reset is hardcoded near `[0.14, 0.072, 0.177]`, not the YAML
  `[0.11, 0.020, 0.217]` plus scale compensation.
- the code-generated midpoint orientation is close to
  `rpy=(3.0543, 0.0, pi)`, not exactly YAML `rpy=(pi, 0.3, pi)`.

Runner support added:

- `run_codrive_sim2sim.py --reset-source yaml|isaacgym_screwdriver`
- `sweep_env_alignment.py --reset-source ...`
- `sweep_reset_contact.py --reset-source ...`

Key checks:

- Raw IsaacGym root in MuJoCo:
  `outputs/sim2sim_mujoco_debug/isaacgym_reset_source_full_policy_120/`
  - no object contact for 120 policy steps
  - confirms raw IsaacGym root coordinates cannot be pasted into MuJoCo as-is.
- IsaacGym init q with MuJoCo/YAML root:
  `outputs/sim2sim_mujoco_debug/isaacgym_initq_yaml_root_full_policy_120/`
  - reset already has 10 active index contacts
  - max active contact force reaches about `17132 N`
  - this explains visible twitching and is not a valid closed-loop baseline.
- Best no-reset-contact candidate from the new sweep:
  `outputs/sim2sim_mujoco_debug/isaacinitq_noresetcontact_candidate_full_policy_160/`
  - reset contact count is zero
  - policy still becomes thumb-only:
    `thumb_only_contact_fraction ~= 0.994`, `index_contact_fraction = 0.0`
  - dominant pair:
    `right_thumb_tactile_link_0<->codrive_lightbulb_contact0`

Conclusion: the policy loop is running, but current MuJoCo behavior is not a
normal sim2sim reproduction. The remaining problem is reset/asset/contact
parity, especially mapping the IsaacGym reset frame and contact semantics into
MuJoCo without precontact or single-finger shortcuts.
