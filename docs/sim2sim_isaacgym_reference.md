# IsaacGym Reference For MuJoCo Sim2Sim

This document records the current source-of-truth behavior pack for CoDrive
MuJoCo sim2sim validation.

## Frozen Reference

- task YAML:
  `sim2real/codrive/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.task.yaml`
- policy checkpoint:
  `sim2real/codrive/model_best_codrive.ckpt`
- exported CPU/headless reference pack:
  `outputs/sim2sim_isaacgym_reference/codrive_ref_cpu_20_bodycontacts/`
- analyzed summary:
  `outputs/sim2sim_isaacgym_reference/codrive_ref_cpu_20_bodycontacts/isaacgym_reference_summary.json`

The pack contains 20 pre-step snapshots from `Dexh13Hora -> XHandHora`.

Key reference values:

- `dt ~= 0.005`
- `control_freq_inv = 10`
- policy rate: `20 Hz`
- object scale: `1.1938252449035645`
- nut axis delta over 20 policy steps: `1.1615401157177985`
- nut partner counts:
  - `right_index_tactile_link_2`: 6
  - `right_thumb_tactile_link_1`: 7
  - `none`: 7

Reference contact sequence:

```text
0-4:   right_index_tactile_link_2
5-6:   none
7-13:  right_thumb_tactile_link_1
14-18: none
19:    right_index_tactile_link_2
```

Interpretation: initial index tactile contact is expected. A MuJoCo result is
not wrong just because it starts with index contact. The mismatch to fix is
sustained single-pair dominance, excessive or insufficient axis gain, and a
failure to reproduce the index-to-thumb-to-release rhythm.

## Current MuJoCo Baseline

Baseline replay from exported `step0000_pre_step`:

- output:
  `outputs/sim2sim_mujoco_reference_replay/latest_cpu_bodycontacts_step0000_policy_20/`
- object/nut axis delta: `1.482287888772467`
- dominant pair:
  `right_index_tactile_link_2/right_index_tactile_link_2<->codrive_lightbulb_nut/codrive_lightbulb_contact0`
- dominant pair step fraction: `0.7222222222222222`
- index contact fraction: `0.65`
- thumb contact fraction: `0.60`
- index+thumb overlap fraction: `0.35`

Zero-action control from the same reset:

- output:
  `outputs/sim2sim_mujoco_reference_replay/latest_cpu_bodycontacts_step0000_zero_20/`
- object/nut axis delta: `0.0400234524849244`
- dominant pair is index tactile.

Interpretation: the policy is contributing the main rotation, but the reset has
an index-side precontact bias and the closed-loop contact sequence remains too
single-pair dominated.

## Active Proxy Geometry Probes

The runner can now generate temporary active distal proxy geoms and can disable
raw active finger mesh/tip contacts with `--finger-contact-mode active_proxy_only`.
This is a diagnostic approximation of IsaacGym collision behavior, not a final
asset conversion.

Useful runs:

- shared proxy sweep:
  `outputs/sim2sim_mujoco_contact_sweeps/active_proxy_only_index_bias_24/`
- strong index-biased proxy sweep:
  `outputs/sim2sim_mujoco_contact_sweeps/active_proxy_only_strong_index_36/`

Best current diagnostic proxy-only candidate:

- output:
  `outputs/sim2sim_mujoco_contact_sweeps/active_proxy_only_strong_index_36/0015_idx0x0p01x0p0035_th0x0p001x0p0035_size0p0085_pm0p002_is0p0085_ts0p0035_im0p002_tmarg0p0005_om0p002_tm0p001/`
- proxy settings:
  `index_proxy_pos = 0 0.010 0.0035`,
  `thumb_proxy_pos = 0 0.001 0.0035`,
  `index_proxy_size = 0.0085`,
  `thumb_proxy_size = 0.0035`,
  `index_proxy_margin = 0.002`,
  `thumb_proxy_margin = 0.0005`
- policy axis delta: `0.9770558700461041`
- zero-action axis delta: `0.012909235851534093`
- dominant pair:
  `right_index_tactile_link_2/right_index_distal_proxy<->codrive_lightbulb_nut/codrive_lightbulb_contact0`
- dominant pair step fraction: `0.7058823529411765`
- index contact fraction: `0.60`
- thumb contact fraction: `0.55`
- sequence match to the 20-step IsaacGym reference: `0.60`

Interpretation: proxy-only contact removes raw fingertip/mesh shortcuts and
keeps passive zero-action rotation low. The strong index-biased proxy improves
contact balance compared with thumb-dominant candidates, but it still is not
IsaacGym parity: the axis delta undershoots the reference `1.161540`, and the
dominant single-pair fraction remains too high versus the reference dominant
partner fraction `0.5384615`.

## Best Current Contact Sweep Candidate

Object-contact sweep output:

- `outputs/sim2sim_mujoco_contact_sweeps/object_contact_refstep0_fric_solref_solimp_48/`
- script:
  `sim2sim/mujoco/sweep_object_contact.py`

Best-ranked diagnostic candidate:

```bash
python sim2sim/mujoco/run_codrive_sim2sim.py \
  --mode policy \
  --scene sim2sim/mujoco/scene_dexh13_lightbulb_hinge_isaacgym_parity.xml \
  --task-config sim2real/codrive/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.task.yaml \
  --reference-state-json outputs/sim2sim_isaacgym_reference/codrive_ref_cpu_20_bodycontacts/isaacgym_ref_env0_step0000_pre_step.json \
  --joint-limit-mode task \
  --no-clamp-init-q \
  --policy-steps 20 \
  --finger-contact-mode full \
  --object-friction 0.8 0.005 0.0001 \
  --object-solref 0.015 1 \
  --object-solimp 0.8 0.98 0.001
```

Policy result:

- axis delta: `1.2309294121449692`
- dominant pair: thumb tactile
- dominant pair step fraction: `0.6842105263157895`
- index contact fraction: `0.60`
- thumb contact fraction: `0.65`
- index+thumb overlap fraction: `0.30`
- max contract violation: `0.01727771759033203`
- min active contact distance during policy rollout:
  `-0.0018520093307695979`

Zero-action control for this candidate:

- output:
  `outputs/sim2sim_mujoco_contact_sweeps/object_contact_refstep0_best_zero_20/`
- axis delta: `0.07018757978104528`
- policy-minus-zero axis gain: about `1.160741832363924`
- first zero-action contacts show light index tactile penetration:
  `right_index_tactile_link_2 <-> codrive_lightbulb_contact0`,
  `dist ~= -7.97e-05 m`, with comparable normal and tangential force.

Interpretation: this candidate is a useful diagnostic improvement because the
policy-minus-zero axis gain is very close to the IsaacGym 20-step reference.
It is not yet a successful parity result because zero-action still has
sustained index precontact and the policy contact sequence does not reproduce
the late release phase.

An additional z-offset probe around this candidate is available at:

- `outputs/sim2sim_mujoco_contact_sweeps/object_contact_refstep0_best_zoffset_7/`

Small `--object-contact-z-offset` changes altered the contact sequence but did
not improve the parity score or remove sustained single-pair dominance. Keep it
as a diagnostic parameter, not as the current main fix.

An additional xy contact-mesh offset probe is available at:

- `outputs/sim2sim_mujoco_contact_sweeps/object_contact_refstep0_best_xyoffset_9/`

Small `--object-contact-pos-offset` changes also did not improve the best
parity score. This suggests that the remaining mismatch is not a simple rigid
translation of the object contact meshes.

## Contact0 Mesh Probe

The runner can override the generated scene's head contact mesh without editing
the source XML:

- `--object-contact0-mesh assets/lightbulb/smooth_head_collision.stl`
- `--object-contact0-mesh assets/lightbulb/rounded_contact_head.stl`

Mesh sweep output:

- `outputs/sim2sim_mujoco_contact_sweeps/object_contact_refstep0_mesh_fric_solref_18/`

Key observations:

- default `contact0.stl` remains the best-ranked diagnostic profile in the
  tested grid:
  - axis delta: `1.230929`
  - dominant pair fraction: `0.684211`
  - sequence match fraction: `0.50`
- `rounded_contact_head.stl` makes axis delta closer in several runs and lowers
  contract violation, but does not improve sequence match:
  - representative axis delta: `1.191872`
  - dominant pair fraction: `0.736842`
  - sequence match fraction: `0.50`
- `smooth_head_collision.stl` can remove reset contact, but the rollout becomes
  too thumb-dominant:
  - representative axis delta: `1.216915`
  - dominant pair fraction: `0.833333`
  - sequence match fraction: `0.45`

Interpretation: mesh choice matters, but neither whole-mesh replacement nor
whole-mesh translation solves the current parity gap. The next repair should
focus on local active tactile/contact0 interaction, contact filtering, or a
purpose-built proxy that preserves the IsaacGym index-to-thumb-to-release
sequence instead of accepting a smooth/rounded replacement shortcut.

## Active Tactile Probe

The runner can also override active index/thumb tactile and tip geoms:

- `--active-tactile-friction SLIDE TORSION ROLL`
- `--active-tactile-solref TIMECONST DAMPING`
- `--active-tactile-solimp MIN MAX WIDTH`

Initial probes:

- `outputs/sim2sim_mujoco_contact_sweeps/object_contact_refstep0_active_tactile_stiff_20/`
  - axis delta: `1.228109`
  - dominant pair fraction: `0.944444`
  - min active contact distance: `-0.000637`
  - verdict: worse single-pair concentration
- `outputs/sim2sim_mujoco_contact_sweeps/object_contact_refstep0_active_tactile_lowfric_stiff_20/`
  - axis delta: `1.269034`
  - dominant pair fraction: `0.789474`
  - min active contact distance: `-0.000981`
  - verdict: still worse than the current best diagnostic candidate

Interpretation: changing active tactile solver/friction can reduce penetration,
but the tested settings increase thumb dominance or axis gain. The next local
repair should be more selective than a shared active tactile override.

## Active Pair Profile Probe

The runner can generate temporary MuJoCo `<contact><pair ...>` profiles for the
active tactile/contact0 pairs:

- `--active-pair-profile balanced_soft`
- `--active-pair-profile index_release`
- `--active-pair-profile index_only_release`
- `--active-pair-profile index_only_soft`
- `--active-pair-profile thumb_guard`
- `--active-pair-profile thumb_only_guard`
- `--active-pair-profile low_pair`

These profiles were tested from the same IsaacGym step-0 pre-state with the
current diagnostic object profile:

```bash
--object-friction 0.8 0.005 0.0001 \
--object-solref 0.015 1 \
--object-solimp 0.8 0.98 0.001
```

Key results:

- `index_only_release`: best sequence match so far (`0.70`) but over-rotates:
  axis delta `1.468651`.
- `low_pair`: best current watch/debug candidate:
  - policy axis delta: `1.052727`
  - zero-action axis delta: `0.052390`
  - policy-minus-zero gain: `1.000338`
  - sequence match: `0.65`
  - dominant pair fraction: `0.80`
- `thumb_guard`: similar diagnostic value but lower policy-minus-zero gain:
  `0.984774`.
- lowering `--hinge-frictionloss` with `low_pair` is not smooth:
  `0.15` stays near axis `1.05`, `0.10` jumps to `1.33`, and `0.05` jumps to
  `1.97`.

Interpretation: the pair profiles confirm the visible issue is contact-model
dominated, not a frozen-policy loading failure. They can move the failure away
from index rubbing, but currently tend to replace it with thumb dominance or
miss the IsaacGym late release phase. Treat `low_pair` as a viewer/debug
candidate, not as a final parity result.

## Contact Offset And Proxy Probe

A subagent audit of the training-side asset path found that the frozen CoDrive
task uses:

- hand asset: `assets/dexh13_hand/urdf/dexh13_hand_right_sim.urdf`
- PhysX-like asset settings in task code:
  `convex_decomposition_from_submeshes=True`, `thickness=0.001`,
  `collapse_fixed_joints=False`
- simulation contact settings:
  `contact_offset: 0.002`, `rest_offset: 0.0`

The current parity MuJoCo scene uses
`assets/dexh13_right_description2/urdf/dexh13_right_isaacgym_parity_fingertips.xml`.
Checked index/thumb distal tactile and tip origins match the URDF values, so
the remaining mismatch is more likely collision-shape generation/contact-offset
behavior than a gross frame-origin error.

New runner diagnostics:

- `--finger-contact-mode no_index_tactile2`
- `--finger-contact-mode index_tip_thumb_pad_proxy`
- `--finger-contact-mode active_distal_proxy`
- `--active-proxy-profile distal_spheres`
- `--object-margin VALUE`
- `--object-gap VALUE`
- `--active-tactile-margin VALUE`
- `--active-tip-margin VALUE`

Key outputs:

- `outputs/sim2sim_mujoco_contact_sweeps/object_contact_refstep0_no_index_tactile2_policy_20/`
  - reset object contacts: `0`
  - policy axis delta: `1.260622`
  - index contact fraction: `0.25`
  - thumb contact fraction: `0.85`
  - verdict: removes the strongest index precontact channel, but switches to
    thumb too early.
- `outputs/sim2sim_mujoco_contact_sweeps/object_contact_refstep0_no_index_tactile2_zero_20/`
  - zero axis delta: `0.001607`
  - verdict: contact remains possible but no meaningful zero-action rotation.
- `outputs/sim2sim_mujoco_contact_sweeps/object_contact_refstep0_proxy_margin_object2_active1_policy_20/`
  - command additions:
    `--object-margin 0.002 --active-tip-margin 0.001 --active-tactile-margin 0.001`
  - policy axis delta: `1.016932`
  - zero axis delta: `0.011418`
  - index contact fraction: `0.45`
  - thumb contact fraction: `0.85`
  - verdict: contact offset/margin can recover more index-tip participation
    with low zero-action drift, but thumb/contact0 is still dominant too long.
- `outputs/sim2sim_mujoco_contact_sweeps/object_contact_refstep0_active_distal_proxy_policy_20/`
  - command additions:
    `--finger-contact-mode active_distal_proxy --active-proxy-profile distal_spheres`
  - generated contact geoms:
    `right_index_distal_proxy`, `right_thumb_distal_proxy`
  - reset object contacts: `0`
  - policy axis delta: `1.276058`
  - zero-action axis delta: `0.012049`
  - policy-minus-zero gain: `1.264009`
  - index contact fraction: `0.35`
  - thumb contact fraction: `0.75`
  - sequence match: `0.40`
  - verdict: raw active mesh rubbing is removed, but this first proxy placement
    makes `right_thumb_tip <-> codrive_lightbulb_contact0` dominate too early.

Interpretation: these probes narrow the likely repair to a local PhysX-style
collision proxy for active distal contacts. The first proxy implementation is
functional but not tuned. Whole mesh translation, whole mesh replacement, pure
friction sweeps, and pair-only overrides have not reproduced the IsaacGym
index-to-thumb-to-release sequence.

## Remaining Risk

- The reference pack is CPU/headless. It is useful for local repair, but a
  headed/GPU IsaacGym reference should be exported when VRAM is available.
- Contact sweep results show that friction and solver softness can tune axis
  gain, but they do not fully fix the contact rhythm by themselves.
- The next repair should target object/contact geometry or reset precontact so
  the policy can move from early index contact to mid-rollout thumb contact and
  late release without replacing the original failure with a thumb-only
  shortcut.
