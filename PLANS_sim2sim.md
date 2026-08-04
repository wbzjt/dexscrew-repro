# PLANS_sim2sim.md

## 1. Current Stage
The active stage is **MuJoCo sim2sim behavior parity**.

The goal is not to train a new policy. The goal is to deploy and inspect the
frozen CoDrive PPO + PAdapt policy in MuJoCo and determine whether its behavior
transfers from IsaacGym-style deployment semantics to a MuJoCo runtime.

Current status:

- Policy loading, observation/history wiring, action integration, 20 Hz policy
  timing, task-YAML joint limits, hand-root scale compensation, palm collision,
  and passive joint parameters have been aligned enough for closed-loop MuJoCo
  execution.
- The policy can rotate the MuJoCo hinge, but visual inspection and contact-pair
  logs show a major remaining failure mode: the MuJoCo scene is still too
  index-dominant and rotates faster than the IsaacGym reference under the same
  exported reset.
- A reset-source audit found that the frozen YAML is not the whole IsaacGym
  reset contract. `xhand_pasini.py` hardcodes a different
  `screwdriver_inclined` active-finger init pose and a different hand root
  pose. Directly applying the raw IsaacGym root in MuJoCo produces no contact,
  while applying only the IsaacGym init q at the MuJoCo/YAML root creates severe
  precontact and very high contact forces. Reset pose parity is therefore still
  open.
- A CPU IsaacGym reference pack has now been exported from the frozen CoDrive
  checkpoint and frozen task YAML. It shows that initial index tactile contact
  is normal, but the short rollout transitions to thumb tactile contact. The
  current MuJoCo replay has thumb participation, but the dominant object-contact
  pair remains an index tactile link and the nut axis gain is too high.
- Explicit MuJoCo active tactile/contact0 pair profiles can change the failure
  from index-dominant rubbing to thumb-dominant contact, confirming that the
  policy is running and the remaining gap is contact-model dominated. The best
  current watch/debug candidate is `--active-pair-profile low_pair`, but it is
  still diagnostic rather than successful parity.
- Further proxy/contact-offset probes show that disabling
  `right_index_tactile_link_2` removes the reset precontact and almost removes
  zero-action axis drift, but then the policy transitions to thumb too early.
  MuJoCo `margin` probes recover more index-tip participation without large
  zero-action drift, but still do not match the IsaacGym release sequence.
  The runner can now generate local active distal proxy spheres. The first
  `active_distal_proxy + distal_spheres` probe removes raw-mesh rubbing and
  keeps zero-action drift low, but makes thumb tip/contact0 dominate too early.
  The proxy-only geometry sweep confirms the policy is running through the
  intended loop: zero-action drift stays low and raw fingertip/mesh shortcuts
  can be removed. However, shared proxy geometry is thumb-dominant, while a
  strong index-biased proxy improves the contact balance but still undershoots
  the IsaacGym axis delta and keeps one proxy pair dominant too long.
  The next repair should therefore replace the ad hoc sphere proxy family with
  a closer IsaacGym-style collision approximation for the active distal links.
- Therefore the current MuJoCo result is **diagnostic only**, not a successful
  sim2sim reproduction.
- First-pass anti-shortcut metrics/gate are implemented in the MuJoCo
  runner/analyzer/validator. A known-bad full-mesh pose that rotates the hinge
  but is dominated by index rubbing now returns
  `validation_failed_single_finger_shortcut`.
- The current diagnostic path is to export exact IsaacGym reset/action
  snapshots from `Dexh13Hora -> XHandHora`, replay the same state/action in
  MuJoCo, and then tune contact/hinge physics against the observed reference
  contact sequence.

## 2. Frozen Policy Package
Primary package:

- `sim2real/codrive/`

Primary artifacts:

- teacher PPO checkpoint:
  `sim2real/codrive/best_reward_4159.37.pth`
- PAdapt student checkpoint:
  `sim2real/codrive/model_best_codrive.ckpt`
- task config:
  `sim2real/codrive/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.task.yaml`
- train config:
  `sim2real/codrive/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.train.yaml`

Reference-only package:

- `sim2real/codrive/dotpg_bc5/`

DOTPG BC5 is useful for later comparison, but the default implementation target
is PPO + PAdapt.

## 3. Target Embodiment
The intended MuJoCo body is the user-described Paxini hand. In this repository,
the matching code/assets appear under the spelling `Pasini` / `XHandPasini`.

Before controller implementation is finalized, verify the exact repository
asset/class naming, joint names, joint order, actuator order, limits, and root
pose. Do not assume the policy order matches MuJoCo model order.

Relevant existing code and assets to inspect first:

- `dexscrew/tasks/xhand_pasini.py`
- `configs/task/XHandPasini*.yaml`
- `assets/dexh13_right_description2/urdf/*.xml`
- `assets/dexh13_hand/urdf/*.urdf`
- `xhand-deploy/xhand_deploy.py`
- `docs/sim2sim_policy_io_audit.md`

## 4. Deployment Semantics To Preserve
The sim2sim runner should mimic the existing deployment/control contract:

- fixed policy artifact, no retraining
- normalized policy action clamped to `[-1, 1]`
- CoDrive action mask from the frozen task YAML
- target-position integration:
  `target = prev_target + action_scale * action`
- explicit PD torque:
  `tau = pgain * (target - q) - dgain * qvel`
- target clipping to joint limits
- 20 Hz policy loop
- 200 Hz MuJoCo physics loop
- observation/history buffers compatible with PAdapt

Use `xhand-deploy/xhand_deploy.py` as the closest reference for an external
runtime loop, while adapting it to the selected Pasini/Paxini MuJoCo model and
the 16-DOF CoDrive policy.

## 5. Milestones

### M0: Workflow And Artifact Freeze
Completion conditions:

- `AGENTS.md` points to this file.
- `docs/session_handoff_v2.md` records sim2sim as the active branch task.
- The selected teacher/student/config artifacts under `sim2real/codrive/` are
  confirmed to exist.
- No frozen artifact is modified.

### M1: Policy Interface Audit
Completion conditions:

- The PAdapt student loading path is chosen:
  - direct checkpoint loader, or
  - TorchScript export/runtime path.
- Input shapes, normalization buffers, output action shape, and action range are
  documented.
- A small action parity test exists for synthetic or recorded observation
  inputs.

### M2: MuJoCo Scene Smoke
Completion conditions:

- A MuJoCo scene loads the selected hand and CoDrive object.
- Joint names/order and actuator names/order are logged.
- Reset/zero-action rollout is stable for at least one episode-length smoke or
  an explicitly bounded shorter smoke.

### M3: Closed-Loop Policy Smoke
Completion conditions:

- The frozen PAdapt policy runs closed-loop in MuJoCo.
- The runner maintains observation/history buffers, target integration, action
  mask, and PD torque without NaN or shape errors.
- Logs include action, target, qpos, qvel, torque, and object/nut angle traces.

### M4: Behavior And Contact Diagnostics
Completion conditions:

- Record MuJoCo videos or equivalent visual inspection artifacts.
- Report object/nut rotation direction and velocity.
- Report contact summaries for active fingers.
- Identify whether failure is policy-interface, controller, asset pose,
  actuator, or contact/friction dominated.

### M5: Sim2Sim Verdict
Completion conditions:

- Produce a concise verdict:
  - transfers sufficiently for follow-up,
  - partially transfers but needs MuJoCo physics/contact tuning,
  - or does not transfer because of a specific interface/asset mismatch.
- Preserve commands, configs, logs, and videos/plots needed for reproduction.
- Update `docs/session_handoff_v2.md` with the next single action.

### M6: IsaacGym Reference Behavior Pack
Purpose:

- Establish the behavior that MuJoCo must reproduce, using the frozen original
  YAML and frozen CoDrive PAdapt checkpoint as the source of truth.

Required work:

- Record a short headed IsaacGym viewer/video or frame sequence using the frozen
  original package:
  - `sim2real/codrive/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.task.yaml`
  - `sim2real/codrive/model_best_codrive.ckpt`
- Export or log per-step reference evidence where available:
  - object/lightbulb screw-axis motion
  - index and thumb contact/body involvement
  - action saturation or torque magnitude if already available
  - reset hand/object pose and object scale
- Write the reference summary to `docs/sim2sim_isaacgym_reference.md`.

Current implementation status:

- `XHandHora.dump_sim2sim_reference_state()` can export compact JSON snapshots
  containing hand/object root state, hand q/qd, target buffers, nut DOF state,
  fingertip/body contact forces, policy action, action mask, and selected
  observation inputs.
- `ProprioAdapt.test()` can call the exporter when
  `++sim2sim_reference_out=...` is set.
- `run_codrive_sim2sim.py --reference-state-json ... --mode reference_action`
  can replay an exported IsaacGym pre-step policy action from the exported
  reset state. It rejects post-step snapshots and validates exported DOF order
  before applying the state.
- `sim2sim/mujoco/analyze_isaacgym_reference.py` summarizes exported reference
  packs and writes:
  - `isaacgym_reference_summary.json`
  - `isaacgym_reference_steps.tsv`
- `docs/sim2sim_isaacgym_reference.md` records the current reference baseline,
  MuJoCo baseline replay, best current contact-sweep candidate, and remaining
  risk.
- CPU/headless reference pack:
  `outputs/sim2sim_isaacgym_reference/codrive_ref_cpu_20_bodycontacts/`
  - 20 pre-step snapshots
  - `dt ~= 0.005`, `control_freq_inv = 10`, policy rate `20 Hz`
  - `object_scale = 1.1938252449035645`
  - nut axis delta over 20 policy steps: `1.161540`
  - nut contact partner sequence:
    - steps 0-4: `right_index_tactile_link_2`
    - steps 7-13: `right_thumb_tactile_link_1`
    - steps 5-6 and 14-18: no matched single active partner
    - step 19: `right_index_tactile_link_2`
- MuJoCo replay from the same exported step-0 pre-state:
  `outputs/sim2sim_mujoco_reference_replay/latest_cpu_bodycontacts_step0000_policy_20/`
  - nut/object axis delta over 20 policy steps: `1.482288`
  - dominant object-contact pair:
    `right_index_tactile_link_2/right_index_tactile_link_2<->codrive_lightbulb_nut/codrive_lightbulb_contact0`
  - dominant pair step fraction: `0.722222`
  - index contact fraction: `0.65`
  - thumb contact fraction: `0.60`
  - index+thumb overlap fraction: `0.35`
- Interpretation: initial index contact is not automatically wrong; the current
  MuJoCo mismatch is sustained index-pair dominance, excessive axis gain, and
  an imperfect index-to-thumb contact transition.
- Contact-profile probes from the same exported state are under:
  `outputs/sim2sim_mujoco_contact_profiles/`
  - default/full mesh: axis delta `1.482`, dominant pair fraction `0.722`,
    dominant pair is index tactile.
  - full mesh with object friction `1.0 0.005 0.0001`: axis delta `1.224`,
    closer to the IsaacGym `1.162`, but dominant pair fraction rises to
    `0.842` and dominant pair becomes thumb tactile.
  - full mesh with object friction `2.0 0.005 0.0001`: axis delta `1.470`,
    dominant pair fraction `0.722`, dominant pair thumb tactile.
  - full mesh with object friction `2.5 0.005 0.0001`: axis delta `1.500`,
    dominant pair fraction `0.722`, dominant pair returns to index tactile.
  - `tip_proxy`: axis delta `0.998`, but becomes thumb-tip dominant and should
    not be treated as successful parity.
  - object `condim=3` did not solve the issue and can increase contract
    violation or axis gain.
- Interpretation from the probes: friction strongly controls axis gain and
  which finger dominates, but no tested fixed profile yet reproduces the
  IsaacGym contact rhythm. The next repair must target contact persistence and
  solver/contact geometry balance, not simply disable the index or accept a
  thumb-only replacement shortcut.
- Targeted object-contact sweep script:
  `sim2sim/mujoco/sweep_object_contact.py`
  - 16-run coarse sweep:
    `outputs/sim2sim_mujoco_contact_sweeps/object_contact_refstep0_fric_solref_16/`
  - 48-run focused sweep:
    `outputs/sim2sim_mujoco_contact_sweeps/object_contact_refstep0_fric_solref_solimp_48/`
  - best current diagnostic candidate:
    `--object-friction 0.8 0.005 0.0001 --object-solref 0.015 1 --object-solimp 0.8 0.98 0.001`
  - candidate policy axis delta: `1.230929`
  - candidate zero-action axis delta: `0.070188`
  - candidate policy-minus-zero axis gain: about `1.160742`
  - candidate dominant pair fraction: `0.684211`, still above the target gate
  - candidate sequence match fraction: `0.50`, still not parity
  - candidate min active contact distance during policy rollout:
    `-0.001852`
  - first zero-action index precontact detail:
    `dist ~= -7.97e-05 m`, with comparable normal/tangent force
- `run_codrive_sim2sim.py` now logs contact details and minimum object/active
  contact distance for reset/contact geometry repair.
- object contact z-offset probe:
  `outputs/sim2sim_mujoco_contact_sweeps/object_contact_refstep0_best_zoffset_7/`
  - small `--object-contact-z-offset` changes alter sequence but did not
    improve the best parity score or remove sustained single-pair dominance.
- object contact xy-offset probe:
  `outputs/sim2sim_mujoco_contact_sweeps/object_contact_refstep0_best_xyoffset_9/`
  - small `--object-contact-pos-offset` changes did not improve the best
    parity score, suggesting the mismatch is not a simple rigid translation of
    the object contact meshes.
- contact0 mesh probe:
  `outputs/sim2sim_mujoco_contact_sweeps/object_contact_refstep0_mesh_fric_solref_18/`
  - `--object-contact0-mesh assets/lightbulb/smooth_head_collision.stl` can
    remove reset contact, but becomes thumb-dominant.
  - `--object-contact0-mesh assets/lightbulb/rounded_contact_head.stl` can
    make axis delta closer and reduce contract violation, but sequence match
    does not improve.
  - default `contact0.stl` remains the best-ranked diagnostic profile in the
    tested grid, even though it is still not a parity result.
- active tactile override probes:
  - `outputs/sim2sim_mujoco_contact_sweeps/object_contact_refstep0_active_tactile_stiff_20/`
  - `outputs/sim2sim_mujoco_contact_sweeps/object_contact_refstep0_active_tactile_lowfric_stiff_20/`
  - shared active tactile solver/friction changes can reduce penetration but
    increased thumb dominance or axis gain in the tested settings.
- Interpretation from the sweeps: solver/friction tuning can make the net axis
  gain close to IsaacGym, but it does not remove the reset index precontact or
  reproduce the late release phase. The next fix should address contact
  geometry/reset precontact rather than only friction/solver values. Whole-mesh
  translation or replacement is not sufficient; the likely repair is more local
  and selective active tactile/contact0 filtering or proxy contact geometry.

Acceptance conditions:

- The reference command is reproducible.
- The user-visible IsaacGym behavior is described concretely enough to compare
  against MuJoCo.
- MuJoCo gates below must be compared to this reference, not to a convenient
  MuJoCo-only behavior.
- The CPU reference is useful for repair work, but a headed/GPU IsaacGym
  reference should still be exported when VRAM is available because CPU PhysX is
  not guaranteed to be bitwise equivalent to the viewer run.

### M7: Anti-Single-Finger Contact Gate
Purpose:

- Prevent the current failure mode where the hinge rotates because one index
  geom or one index tip rubs the object.

Status:

- First-pass implementation is complete.
- Implemented metrics:
  - `dominant_object_contact_pair`
  - `dominant_object_contact_pair_step_fraction`
  - `dominant_object_contact_pair_event_fraction`
  - `index_only_contact_fraction`
  - `thumb_only_contact_fraction`
  - `index_thumb_tip_overlap_fraction`
- The validator supports `--anti-shortcut-gate`.
- Known-bad evidence:
  - output:
    `outputs/sim2sim_mujoco_validation/anti_shortcut_gate_fullmesh_known_bad_120/`
  - verdict: `validation_failed_single_finger_shortcut`
  - dominant pair:
    `right_index_link_3/geom_8@right_index_link_3<->codrive_lightbulb_nut/codrive_lightbulb_contact0`
  - dominant pair step fraction: `0.966667`
  - index-only contact fraction: `0.516667`

Required work:

- Tune the thresholds against an IsaacGym reference behavior pack rather than
  treating the first-pass values as final.
- Add any additional metrics needed by M8/M9, such as normal/tangential contact
  decomposition if MuJoCo exposes enough signal.

Minimum pass gate for a candidate visual baseline:

- policy runs without NaN
- policy-minus-zero axis gain is positive and meaningful
- index contact fraction and thumb contact fraction are both nontrivial
- simultaneous index+thumb overlap is sustained
- dominant object-contact pair fraction is below a configured threshold
- index-only rotation is not the main driver
- zero-action does not produce comparable axis motion

Initial numeric gate to tune against reference evidence:

- `policy_minus_zero.object_axis_delta >= 0.2`
- `index_contact_fraction >= 0.25`
- `thumb_contact_fraction >= 0.25`
- `index_thumb_overlap_fraction >= 0.20`
- `index_thumb_tip_overlap_fraction >= 0.10` when using tip/proxy contact
- `dominant_object_contact_pair_fraction <= 0.60`
- `index_only_contact_fraction <= 0.50`
- `max_contract_violation <= 0.02`

These are starting gates. The reference pack shows that a brief initial index
phase is expected, so the gate should penalize sustained dominant-pair behavior
rather than any early index contact.

### M8: MuJoCo Hand Contact Asset Repair
Purpose:

- Make the MuJoCo hand contact representation closer to the IsaacGym special
  URDF instead of continuing to search poses around a known geometry shortcut.

Preferred implementation path:

1. Build a new diagnostic MJCF include, not by overwriting the current asset:
   - `assets/dexh13_right_description2/urdf/dexh13_right_isaacgym_parity.xml`
2. Preserve the already aligned passive joint parameters:
   - `damping=0`
   - `stiffness=0`
   - `frictionloss=0.01`
   - `armature=0.001`
3. Preserve palm visual-only collision behavior.
4. Reduce or disable active index/thumb distal side-mesh object contact that
   creates the rubbing shortcut.
5. Represent active fingertip/tactile contact with explicit proxy geoms or
   fixed child bodies that are easier to audit:
   - index active pad/tip
   - thumb active pad/tip
6. Keep middle/ring locked and non-active for the CoDrive task unless they are
   needed for self-collision stability.

Alternative implementation path if full MJCF rebuild is too risky:

- Add a runner-level contact-profile option that applies geom filters/friction
  overrides at load time:
  - `full_mesh`: current diagnostic mode
  - `tip_proxy`: current active-tip-only mode
  - `active_pad_proxy`: disable distal side mesh contact and add/use simplified
    pad/tip contact proxies for index and thumb
  - `low_tangent_friction_debug`: lower object tangential friction to expose
    pure rubbing shortcuts

Acceptance conditions:

- New asset/profile loads without NaN.
- Audit summary records the active contact profile and disabled/enabled geoms.
- Zero-action remains stable.
- Single-index rubbing no longer produces a successful visual verdict.

### M9: Object/Hinge Contact Constraint Repair
Purpose:

- Stop the object from being too easy to rotate by one tangential contact.

Required work:

- Audit the MuJoCo object contact geoms and hinge settings against the
  IsaacGym/lightbulb URDF:
  - mesh scale
  - contact geom position/orientation
  - hinge axis and frictionloss
  - object mass/inertia
  - contact friction and `condim`
  - solver/contact parameters
- Add at least one stricter object/contact diagnostic scene or profile:
  - lower tangential friction
  - contact geometry that requires opposing contacts
  - hinge friction/inertia setting that prevents trivial single-contact spin
- Use the new runner overrides for small, reversible sweeps:
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
- Use `sim2sim/mujoco/sweep_object_contact.py` for fixed-reference sweeps and
  rank by axis closeness, dominant-pair concentration, active finger balance,
  sequence match, and contract violation.

Acceptance conditions:

- A single index contact no longer rotates the hinge at a rate that would pass
  behavior validation.
- A legitimate index+thumb contact candidate can still produce positive axis
  gain.

### M10: Behavior-Parity Sweep And Final Verdict
Purpose:

- Search for a valid MuJoCo pose/contact configuration only after M6-M9 remove
  known false positives.

Required work:

- Run paired policy/zero sweeps over:
  - object pose
  - hand root pose/RPY
  - contact profile
  - object/contact profile
- Rank candidates using the anti-shortcut gate, not only axis delta.
- For the best candidate, produce:
  - live viewer command
  - policy/zero traces
  - contact-pair summary
  - optional video
  - comparison against the IsaacGym reference pack

Final success conditions:

- Frozen policy and original YAML are used.
- The same command can be rerun locally.
- Visual behavior does not look like single-index rubbing.
- Metrics pass the anti-shortcut gate.
- User confirms the viewer resembles the IsaacGym reference closely enough for
  the intended sim2sim validation.

## 6. Acceptance Rules
Do not call a MuJoCo result successful unless:

- the frozen artifact path is recorded,
- policy input/output shapes are verified,
- the action-to-target-to-PD loop matches the intended contract,
- the selected hand joint order is explicitly mapped,
- the run has a reproducible command,
- logs include enough traces to diagnose behavior.
- visual behavior is not dominated by single-index rubbing,
- contact-pair logs do not show one index geom/tip as the dominant driver,
- index and thumb both participate under the chosen contact profile,
- the verdict is compared against the IsaacGym reference behavior pack.

The following are not sufficient for success by themselves:

- hinge/object axis rotates,
- `policy_runs_with_two_finger_contact` under loose full-mesh contact,
- brief index+thumb overlap while one contact pair dominates,
- a pose that looks acceptable only because object friction is too permissive.

## 7. Non-Goals
The current stage does not include:

- PPO retraining
- PAdapt retraining
- DOTPG training
- diffusion or flow matching experiments
- cloud training pipelines
- paper-grade multi-seed IsaacGym robustness sweeps

These can be reopened only by explicit user instruction.

## 8. Recommended Next Step
The MVP scaffold exists, M7 anti-shortcut gating is implemented, and M6 CPU
reference export/replay has produced a concrete comparison target. Active pair
profile probes show that local pair tuning alone is not enough. Proceed to M8/M9
contact repair:

1. keep the exported IsaacGym reference pack as the comparison baseline,
2. use the current best contact-sweep, `low_pair`, and margin/proxy candidates
   only as diagnostic/watch baselines,
   not as a success state,
3. repair the local active tactile/contact0 geometry or contact filtering that
   creates sustained index precontact without replacing it with thumb-only
   dominance,
4. sweep the generated PhysX-style local distal proxy positions, sizes, and
   margin settings for active fingers,
5. then rerun `sweep_object_contact.py` or a dedicated proxy sweep and compare
   policy/zero axis gain,
   dominant-pair fraction, and sequence match,
6. rerun a headed/GPU IsaacGym reference export once VRAM is available.
