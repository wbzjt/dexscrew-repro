# codeagent_issue.md

## Status
- status_now: `resolved`
- resolved_in: `v2-071`
- resolution_note:
  - Initial blocker was triggered by early-stopped low-signal replay.
  - Full-budget replay (`controlreplay_full`) recovered high training signal (`Current Best=1774.05`), so “reference irreproducible” is not sustained.

## Original Blocker (Archived)
- Suspected reproducibility drift for kept reference under current repo state.
- Trigger reason: short-horizon replay and nearby probes repeatedly plateaued early and were judged as collapse.

## Resolution Evidence
- Full replay run:
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_controlreplay_full_seed42_15min/`
  - training outcome: `Current Best max=1774.05` (high-signal regime restored).
- Config consistency:
  - saved `config_*.yaml` diff vs historical kept reference shows only `output_name` change; core training overrides and checkpoint path are consistent.

## Post-Resolution Conclusion
- The active issue is no longer a governance-level reproducibility blocker.
- Remaining problem is performance selection:
  - replayed/new full-budget candidates still do not beat kept reference on unified robust eval.

## Recommended Next Action
- Continue Plan v2 execution (no escalation hold).
- Adjust local execution policy:
  - avoid overly aggressive early-stop for near-reference probes,
  - prefer full-budget/late-window validation before rejection,
  - keep strict `nominal/light_v2/hard` keep/drop gate unchanged.
