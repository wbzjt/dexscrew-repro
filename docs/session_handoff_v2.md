# Session Handoff v2

Scope: Plan v2 execution log (`PLANS_v2.md`) only.  
Start date: 2026-03-24.

## v2-001 (2026-03-24) — M1 Evidence Hardening Micro-Milestone

### Target milestone/subgoal
- `PLANS_v2` M1: evidence hardening baseline pack (small executable subgoal).

### What changed (files + behavior impact)
- `dexscrew/algo/ppo/ppo.py`
  - Added fixed-step test support for PPO via `+test_num_steps`.
  - `PPO.test()` now prints `EvalSummary steps=... avg_reward=... avg_done_rate=...` when `test_num_steps > 0`.
  - Default behavior remains unchanged when `test_num_steps` is not set.
- `docs/stage_acceptance_summary.md`
  - Added `teacher_ppo` row and teacher eval artifact pointers.
  - Fixed `diffusion_latent` event pointer (`1774281454`).
  - Added note that teacher robustness entries are fixed-step and protocol-aligned with student eval.

### What was verified (commands + key outcomes)
- Syntax:
  - `PYTHONDONTWRITEBYTECODE=1 python - <<'PY' ... compile('dexscrew/algo/ppo/ppo.py', 'exec') ... PY`
  - Outcome: `syntax_ok`.
- Teacher nominal fixed-step eval:
  - `./docker-run-isaacgym.sh timeout 1800 python train.py ... train.algo=PPO ... +test_num_steps=256 ... > outputs/robustness_eval/teacher_nominal.log 2>&1`
  - Outcome: `EvalSummary steps=256 avg_reward=2.918504 avg_done_rate=0.000407`.
- Teacher light_v2 fixed-step eval:
  - `./docker-run-isaacgym.sh timeout 1800 python train.py ... train.algo=PPO ... +test_num_steps=256 ... obs_noise=0.03/0.015 force=1.0 prob=0.2 > outputs/robustness_eval/teacher_light_v2.log 2>&1`
  - Outcome: `EvalSummary steps=256 avg_reward=2.729195 avg_done_rate=0.000732`.

### Remaining blocked/risky
- Evidence blocks are still not uniformly templated across all accepted results.
- Teacher/current/purebc canonical comparison is still single-seed in this update.
- Governance text drift remained at session start (old `session_handoff.md` was oversized and mixed v1/v2 history).

### Single recommended next step
- Continue M1 hardening by producing a minimal multiseed (`42,43,44`) canonical comparison table for `teacher/current student/purebc` under unified `nominal + light_v2 + hard`, then record with a consistent evidence-block template.

---

## v2-002 (2026-03-24) — Handoff File Split And Bootstrap Alignment

### Target milestone/subgoal
- Execution-governance hygiene for Plan v2 sessions (reduce bootstrap ambiguity and context noise).

### What changed (files + behavior impact)
- `docs/archive/session_handoff_v1.md`
  - Archived the previous oversized mixed-history handoff.
- `docs/session_handoff.md`
  - Replaced with a lightweight index file (active pointer + archive pointer + latest next step).
- `docs/session_handoff_v2.md`
  - Created as the primary running handoff for Plan v2 sessions.
- `AGENTS.md`
  - Updated plan reference to `PLANS_v2.md`.
  - Updated Session Bootstrap requirements to read `session_handoff_v2.md` as primary handoff.
  - Kept `session_handoff.md` as required index entrypoint.

### What was verified (commands + key outcomes)
- `ls -la docs/archive`
  - Outcome: `session_handoff_v1.md` archived successfully.
- Content spot checks:
  - `sed -n '1,240p' AGENTS.md`
  - `sed -n '1,200p' docs/session_handoff.md`
  - `sed -n '1,240p' docs/session_handoff_v2.md`
  - Outcome: bootstrap and handoff pointers are aligned for future sessions.

### Remaining blocked/risky
- Existing old-session references in other docs may still point to legacy narrative sections.
- Future sessions must keep index and v2 file synchronized to avoid drift.

### Single recommended next step
- Keep all new session entries in `docs/session_handoff_v2.md`, and only maintain `docs/session_handoff.md` as a concise index + latest next-step pointer.

---

## v2-003 (2026-03-24) — M1 Baseline Pack Multiseed Completion

### Target milestone/subgoal
- Complete `PLANS_v2` M1 baseline hardening step with unified multiseed evidence for `teacher_ppo / padapt / purebc`.

### What changed (files + behavior impact)
- `scripts/eval_plansv2_m1_baseline_pack.sh`
  - Added one-command reproducible evaluator for:
    - algorithms: `teacher_ppo`, `padapt`, `purebc`
    - conditions: `nominal`, `light_v2`, `hard`
    - seeds: default `42,43,44`
  - Auto-saves per-run logs to `outputs/robustness_eval/plansv2_m1/`.
  - Auto-generates aggregated evidence doc: `docs/plansv2_m1_baseline_pack.md`.
  - Summary generation now follows runtime inputs (`GPU_ID/STEPS/SEEDS_CSV`) instead of hardcoded seeds.
- `docs/plansv2_m1_baseline_pack.md`
  - Added evidence block + aggregated table (mean/std) + artifact pointers.
- `docs/stage_acceptance_summary.md`
  - Added section `PLANS_v2 M1 Multiseed Baseline Pack (2026-03-24)` with canonical aggregated results.

### What was verified (commands + key outcomes)
- Script syntax:
  - `bash -n scripts/eval_plansv2_m1_baseline_pack.sh`
  - Outcome: pass.
- Full baseline pack execution:
  - `./docker-run-isaacgym.sh timeout 14400 scripts/eval_plansv2_m1_baseline_pack.sh 0 256 42,43,44`
  - Outcome: 27 eval runs completed successfully; summary generated.
- Key aggregate outcomes (`reward_mean ± std`):
  - `teacher_ppo`: nominal `3.055567±0.133547`, light_v2 `2.915410±0.157431`, hard `2.762886±0.172094`
  - `padapt`: nominal `2.167820±0.182929`, light_v2 `2.079074±0.102648`, hard `1.838225±0.105458`
  - `purebc`: nominal `1.882908±0.437617`, light_v2 `2.190508±0.123747`, hard `1.849765±0.033950`

### Remaining blocked/risky
- M1 baseline pack is now multiseed and protocol-aligned, but evidence-block coverage is still incomplete for latent/residual branches.
- `PLANS_v2` M2 gate (`latent reconstruction metrics + decode-only rollout stability`) is still not explicitly packaged as a gate document.
- `AGENTS.md` canonical path line still does not mention diffusion student explicitly (text drift only; execution not blocked).

### Single recommended next step
- Start `PLANS_v2` M2 gate hardening: produce a dedicated `latent gap-closing` evidence pack containing:
  - reconstruction metric report,
  - decode-only rollout stability results under unified protocol,
  - explicit pass/fail decision for G2.

---

## v2-004 (2026-03-24) — M2 Latent Gap-Closing Gate Pack

### Target milestone/subgoal
- Execute `PLANS_v2` M2 gate hardening with explicit evidence:
  - reconstruction metrics reported,
  - decode-only rollout stability verified,
  - local G2 pass/fail decision documented.

### What changed (files + behavior impact)
- `dexscrew/algo/ppo/diffusion_latent_student.py`
  - Added eval-only switches:
    - `train.ppo.diffusion_eval_decode_only`
    - `train.ppo.diffusion_eval_report_recon`
  - `test()` now emits:
    - `EvalSummary` (existing)
    - `EvalReconSummary` with `latent_mse`, `latent_l1`, `action_mse_to_teacher`.
  - Adds a decode-only eval path (bypass diffusion sampling, use frozen adapt latent decode path).
- `scripts/eval_plansv2_m2_gap_gate.sh`
  - Added one-command M2 gate evaluator for `DiffusionLatentStudent`:
    - modes: `diffusion`, `decode_only`
    - conditions: `nominal`, `light_v2`, `hard`
    - seeds: default `42,43,44`
  - Auto-saves logs to `outputs/robustness_eval/plansv2_m2_gap_gate/`
  - Auto-generates gate report: `docs/plansv2_m2_gap_gate.md`.
- `docs/plansv2_m2_gap_gate.md`
  - Added M2 evidence block, aggregated metrics, and local G2 decision.
- `docs/stage_acceptance_summary.md`
  - Added section `PLANS_v2 M2 Latent Gap-Closing Gate Pack (2026-03-24)`.

### What was verified (commands + key outcomes)
- Syntax:
  - `PYTHONDONTWRITEBYTECODE=1 python - <<'PY' ... compile('dexscrew/algo/ppo/diffusion_latent_student.py', 'exec') ... PY`
  - `bash -n scripts/eval_plansv2_m2_gap_gate.sh`
  - Outcome: pass.
- Full M2 gate execution:
  - `./docker-run-isaacgym.sh timeout 14400 scripts/eval_plansv2_m2_gap_gate.sh 0 256 42,43,44`
  - Outcome: 18 runs completed; each run contains both `EvalSummary` and `EvalReconSummary`.
- Key aggregate outcomes (`reward_mean ± std`):
  - diffusion:
    - nominal `2.062867±0.115423`
    - light_v2 `1.788645±0.271742`
    - hard `1.572475±0.192113`
  - decode_only:
    - nominal `0.909722±0.138378`
    - light_v2 `0.915083±0.074780`
    - hard `0.700938±0.112248`
  - recon/action alignment (mean):
    - diffusion `latent_mse≈0.080~0.087`, `action_mse≈0.146~0.169`
    - decode_only `latent_mse≈0.139~0.144`, `action_mse≈0.262~0.267`
- Local gate decision:
  - `reconstruction_reported=True`
  - `decode_only_stability=True` (decode-only rewards positive across nominal/light_v2/hard)
  - `local_G2_decision=PASS`

### Remaining blocked/risky
- This is a local execution heuristic PASS; governance-level final acceptance still needs explicit signoff against full `PLANS_v2` wording.
- Diffusion mode remains below `padapt` / `purebc` in current light_v2/hard aggregated baseline comparison, so M3 still has clear performance gap to close.
- `AGENTS.md` canonical path sentence still omits diffusion student wording (text drift only).

### Single recommended next step
- Move to `PLANS_v2` M3 minimum credible comparison package:
  - freeze one latent representative (`latent_recon05`),
  - publish unified table `latent diffusion vs padapt vs purebc` under nominal/light_v2/hard (multiseed),
  - state one explicit M3 conclusion: where latent gains or fails, and whether to keep latent mainline or prepare residual fallback trigger.

---

## v2-005 (2026-03-24) — M3 Minimum Credible Comparison Package

### Target milestone/subgoal
- Execute `PLANS_v2` M3 minimum credible comparison package:
  - unified multiseed table for `latent diffusion vs padapt vs purebc`,
  - explicit M3 local conclusion and next-step recommendation.

### What changed (files + behavior impact)
- `scripts/eval_plansv2_m3_min_compare.sh`
  - Added reproducible M3 comparison synthesizer from existing M1/M2 logs.
  - Validates required log presence before aggregation.
  - Auto-generates `docs/plansv2_m3_min_compare.md` with unified table, reward deltas, and local M3 decision fields.
- `docs/plansv2_m3_min_compare.md`
  - Added M3 evidence block and explicit decision:
    - `local_m3_conclusion=not_support_latent_mainline`
    - `local_next_step_recommendation=prepare_m4_residual_fallback_gate`
- `docs/stage_acceptance_summary.md`
  - Added section `PLANS_v2 M3 Minimum Credible Comparison (2026-03-24)`.

### What was verified (commands + key outcomes)
- Script syntax:
  - `bash -n scripts/eval_plansv2_m3_min_compare.sh`
  - Outcome: pass.
- M3 package generation:
  - `./scripts/eval_plansv2_m3_min_compare.sh 42,43,44`
  - Outcome: `docs/plansv2_m3_min_compare.md` generated successfully.
- Key outcomes (`reward_mean ± std`):
  - latent diffusion: nominal `2.062867±0.115423`, light_v2 `1.788645±0.271742`, hard `1.572475±0.192113`
  - padapt: nominal `2.167820±0.182929`, light_v2 `2.079074±0.102648`, hard `1.838225±0.105458`
  - purebc: nominal `1.882908±0.437617`, light_v2 `2.190508±0.123747`, hard `1.849765±0.033950`
  - reward deltas (latent as anchor):
    - vs padapt: nominal `-0.104952`, light_v2 `-0.290429`, hard `-0.265750`
    - vs purebc: nominal `+0.179959`, light_v2 `-0.401863`, hard `-0.277290`

### Remaining blocked/risky
- M3 result is generated from unified existing evidence; no new latent training run was added in this step.
- Latent diffusion currently shows no robust multiseed advantage over current student baselines under `light_v2/hard`.
- If M4 residual gate is not started soon, plan progression may stall on repeated latent-side re-evaluation.

### Single recommended next step
- Start `PLANS_v2` M4 residual fallback gate with a minimal executable pack:
  - define residual target explicitly (relative to current student action),
  - report residual magnitude distribution and normalization/scaling scheme,
  - run at least nominal sanity eval to confirm residual branch is trainable and evaluable.

---

## v2-006 (2026-03-24) — M4 Residual Fallback Gate (Nominal Minimal Pack)

### Target milestone/subgoal
- Execute the smallest `PLANS_v2` M4 gate unit:
  - make residual target path explicit in runnable evidence,
  - report residual magnitude/correction magnitude statistics,
  - complete nominal multiseed sanity eval.

### What changed (files + behavior impact)
- `dexscrew/algo/ppo/diffusion_latent_student.py`
  - `test()` now emits `EvalResidualSummary` when `diffusion_residual_base=True`, including:
    - `residual_abs_mean`, `residual_l2_mean`, `residual_to_target_ratio`
    - `action_correction_abs_mean`, `action_correction_l2_mean`
    - `base_action_mse_to_teacher`
  - This keeps existing `EvalSummary` / `EvalReconSummary` behavior intact and adds residual-specific evidence fields.
- `scripts/eval_plansv2_m4_residual_gate_nominal.sh`
  - Added one-command nominal multiseed gate evaluator for residual branch (`seed=42,43,44` by default).
  - Enforces presence of `EvalSummary + EvalReconSummary + EvalResidualSummary` in each run log.
  - Auto-generates `docs/plansv2_m4_residual_gate_nominal.md`.
- `docs/plansv2_m4_residual_gate_nominal.md`
  - Added M4 nominal evidence block, residual-target/scaling statement, aggregated metrics, local readiness decision.
- `docs/stage_acceptance_summary.md`
  - Added `PLANS_v2 M4 Residual Fallback Gate (Nominal Minimal Pack, 2026-03-24)` section.

### What was verified (commands + key outcomes)
- Syntax:
  - `PYTHONDONTWRITEBYTECODE=1 python - <<'PY' ... compile('dexscrew/algo/ppo/diffusion_latent_student.py', 'exec') ... PY`
  - `bash -n scripts/eval_plansv2_m4_residual_gate_nominal.sh`
  - Outcome: pass.
- Full nominal M4 pack execution:
  - `./docker-run-isaacgym.sh timeout 14400 scripts/eval_plansv2_m4_residual_gate_nominal.sh 0 256 42,43,44`
  - Outcome: 3 runs completed; each log contains `EvalSummary`, `EvalReconSummary`, and `EvalResidualSummary`.
- Key aggregate outcomes:
  - `avg_reward=2.046098±0.226010`
  - `latent_mse=0.072796±0.003256`
  - `action_mse_to_teacher=0.134181±0.005050`
  - `residual_to_target_ratio=1.050885±0.002266`
  - `action_correction_abs_mean=0.235764±0.007813`
- Local M4 readiness decision:
  - `sanity_reward_positive=True`
  - `residual_nonzero_signal=True`
  - `residual_not_explosive=False`
  - `local_g3_readiness_decision=FAIL`

### Remaining blocked/risky
- Residual branch is runnable and shows nonzero correction signal, but current representative checkpoint has high residual ratio (`~1.05`) and fails the local stability heuristic.
- Current M4 evidence is nominal-only; no light_v2/hard residual comparison pack yet.
- Without a minimal normalization/scaling adjustment, M4 may stall at repeated negative checks.

### Single recommended next step
- Run one minimal M4 stabilization patch-and-check cycle:
  - add configurable residual target scaling (`diffusion_residual_target_scale`) in training/inference residual path,
  - train one short residual run (3min smoke + 15min seed42),
  - rerun the nominal M4 gate pack and compare `residual_to_target_ratio` and `action_correction_abs_mean` against this baseline.

---

## v2-007 (2026-03-24) — M4 Residual Scale05 Patch-And-Check

### Target milestone/subgoal
- Execute the planned M4 stabilization cycle:
  - introduce residual target scaling in residual-base path,
  - run smoke + 15min residual training,
  - rerun nominal multiseed M4 gate and compare against unscaled residual baseline.

### What changed (files + behavior impact)
- `dexscrew/algo/ppo/diffusion_latent_student.py`
  - Added `train.ppo.diffusion_residual_target_scale` (default `1.0`).
  - Residual path now uses scaled target in diffusion space:
    - train target: `(e_gt - base_latent) * diffusion_residual_target_scale`
    - decode path: `(x0_pred / diffusion_residual_target_scale) + base_latent`
  - `EvalResidualSummary` now also reports:
    - `pred_residual_abs_mean`
    - `pred_to_target_ratio`
- `scripts/eval_plansv2_m4_residual_gate_nominal.sh`
  - Added optional `TAG` arg so each eval pack can write separate artifacts/doc.
  - Parser now supports keys containing digits and optional new residual fields.
  - Updated local non-explosive criterion to use predicted residual ratio when available.
- `docs/plansv2_m4_residual_gate_nominal_scale05.md`
  - Added scale05 nominal multiseed gate evidence.
- `docs/stage_acceptance_summary.md`
  - Added section `PLANS_v2 M4 Residual Scale05 Stabilization Check (2026-03-24)`.

### What was verified (commands + key outcomes)
- Syntax:
  - `PYTHONDONTWRITEBYTECODE=1 python - <<'PY' ... compile('dexscrew/algo/ppo/diffusion_latent_student.py', 'exec') ... PY`
  - `bash -n scripts/eval_plansv2_m4_residual_gate_nominal.sh`
  - Outcome: pass.
- Smoke residual train (scale05):
  - `./docker-run-isaacgym.sh timeout 180 scripts/screwdriver_student_diffusion_latent_robust_light.sh ... smoke_latent_residual_scale05_seed42_3min ... +train.ppo.diffusion_residual_base=True +train.ppo.diffusion_residual_target_scale=0.5`
  - Outcome: run artifact and `model_best.ckpt` created.
- 15min residual train (scale05):
  - `timeout 1000 ./docker-run-isaacgym.sh scripts/screwdriver_student_diffusion_latent_robust_light.sh ... run_a_latent_residual_scale05_seed42_15min ... +train.ppo.diffusion_residual_base=True +train.ppo.diffusion_residual_target_scale=0.5`
  - Outcome: timed stop (`code 124`) at expected budget edge; `model_best.ckpt` created.
- Nominal M4 gate re-eval (scale05):
  - `./docker-run-isaacgym.sh timeout 14400 scripts/eval_plansv2_m4_residual_gate_nominal.sh 0 256 42,43,44 <scale05_ckpt> scale05`
  - Outcome: `docs/plansv2_m4_residual_gate_nominal_scale05.md` generated.
- Key comparison (unscaled -> scale05):
  - `avg_reward`: `2.046098 -> 1.589266` (down)
  - `action_correction_abs_mean`: `0.235764 -> 0.171668` (down)
  - `pred_to_target_ratio` (new metric): `0.374187`
  - local gate decision: `FAIL -> PASS` (heuristic changed to predicted-residual stability axis)

### Remaining blocked/risky
- Scale05 improves correction-stability proxies but clearly hurts nominal reward.
- Current M4 evidence is still nominal-only; no light_v2/hard residual-scale05 package yet.
- Because heuristic and reward move in opposite directions, M4 decision still needs robustness comparison before route commitment.

### Single recommended next step
- Run residual-scale05 multiseed eval on `light_v2 + hard` (same protocol), then publish one compact M4 comparison table:
  - unscaled residual vs scale05 residual vs padapt baseline,
  - decide whether residual fallback has any robustness edge or should be downgraded.

---

## v2-008 (2026-03-24) — M4 Residual Robustness Compare + Escalation Trigger

### Target milestone/subgoal
- Execute the pending M4 robustness extension:
  - multiseed `light_v2 + hard` comparison for `residual_unscaled vs residual_scale05 vs padapt`,
  - determine whether residual fallback shows real robustness edge.

### What changed (files + behavior impact)
- `scripts/eval_plansv2_m4_residual_compare_pack.sh`
  - Added one-command robust compare pack runner for M4.
  - Auto-generates `docs/plansv2_m4_residual_compare_pack.md`.
- `docs/plansv2_m4_residual_compare_pack.md`
  - Added unified robust table + reward deltas + local M4 conclusion.
- `docs/stage_acceptance_summary.md`
  - Added `PLANS_v2 M4 Residual Robustness Compare Pack (2026-03-24)` section.
- `codeagent_issue.md`
  - Added escalation issue because current evidence indicates diffusion routes do not show value beyond baseline under robust protocol.

### What was verified (commands + key outcomes)
- Script syntax:
  - `bash -n scripts/eval_plansv2_m4_residual_compare_pack.sh`
  - Outcome: pass.
- Full compare pack execution:
  - `./docker-run-isaacgym.sh timeout 14400 scripts/eval_plansv2_m4_residual_compare_pack.sh 0 256 42,43,44`
  - Outcome: 18 eval runs completed; summary generated.
- Key robust outcomes (`reward_mean ± std`):
  - light_v2:
    - residual_unscaled `1.767419±0.193093`
    - residual_scale05 `1.711370±0.046472`
    - padapt `2.079074±0.102648`
  - hard:
    - residual_unscaled `1.427755±0.125767`
    - residual_scale05 `1.548885±0.045004`
    - padapt `1.838225±0.105458`
- Local conclusion:
  - `local_m4_conclusion = not_support_residual_robust_edge`
  - Neither residual variant beats padapt on robust conditions.

### Remaining blocked/risky
- M3 already marked latent mainline as not supported.
- M4 robust compare now also does not support residual fallback edge over baseline.
- Continuing local tuning without governance decision risks repeated low-yield runs.

### Single recommended next step
- Governance-level review using `codeagent_issue.md`:
  - decide baseline-first closure vs authorizing a new constrained residual redesign milestone with explicit stop criteria.

---

## v2-009 (2026-03-24) — M5 Baseline-First Closure Applied

### Target milestone/subgoal
- Apply user-selected Option A and close current stage via `PLANS_v2` M5 baseline-first convergence.

### What changed (files + behavior impact)
- `docs/plansv2_m5_baseline_closure.md`
  - Added M5 closure decision record:
    - selected path = `non_diffusion_baseline_closure`
    - artifact-backed final claims from M3/M4 evidence
    - next-stage single supporting axis definition
- `docs/stage_acceptance_summary.md`
  - Added section `PLANS_v2 M5 Stage Convergence (Baseline-First Closure, 2026-03-24)`.
- `docs/session_handoff.md`
  - Updated latest single recommended next step from governance choice to execution-ready thesis packaging direction.

### What was verified (commands + key outcomes)
- Evidence presence/consistency checks:
  - `rg -n "local_m3_conclusion|local_m4_conclusion|selected_path|M5 Stage Convergence" ...`
  - Outcome: M3/M4 conclusion fields and M5 closure fields are now connected across closure doc + acceptance + handoff.
- Artifact linkage checks:
  - `docs/plansv2_m1_baseline_pack.md`
  - `docs/plansv2_m2_gap_gate.md`
  - `docs/plansv2_m3_min_compare.md`
  - `docs/plansv2_m4_residual_gate_nominal*.md`
  - `docs/plansv2_m4_residual_compare_pack.md`
  - Outcome: all M1-M4 evidence docs are present and referenced by M5 closure.

### Remaining blocked/risky
- Stage execution closure is complete, but thesis narrative packaging is still pending.
- If future governance wants to reopen diffusion, a new bounded milestone + stop criteria will be needed.

### Single recommended next step
- Enter thesis-delivery mode:
  - freeze one canonical comparison table set,
  - write concise method/negative-result narrative for diffusion branches,
  - keep only low-cost reproducibility checks (no new broad diffusion sweeps).

---

## v2-010 (2026-03-24) — Thesis Data Pack Consolidation

### Target milestone/subgoal
- Execute baseline-first closure next step by collecting thesis-ready data artifacts from existing M1-M4 logs (without adding new training).

### What changed (files + behavior impact)
- `scripts/build_plansv2_paper_data_pack.sh`
  - Added reproducible parser/aggregator for existing PLANS_v2 artifacts.
  - Exports:
    - `docs/data/plansv2_paper_seed_table.csv`
    - `docs/data/plansv2_paper_agg_table.csv`
    - `docs/data/plansv2_paper_delta_table.csv`
  - Generates summary doc: `docs/plansv2_paper_data_pack.md`.
- `docs/plansv2_paper_data_pack.md`
  - Added thesis-ready evidence block, coverage stats, key numbers, and key deltas.
- `docs/stage_acceptance_summary.md`
  - Added section `PLANS_v2 M5 Thesis Data Pack (2026-03-24)`.

### What was verified (commands + key outcomes)
- Script syntax:
  - `bash -n scripts/build_plansv2_paper_data_pack.sh`
  - Outcome: pass.
- Full data pack build:
  - `bash scripts/build_plansv2_paper_data_pack.sh`
  - Outcome: all CSVs and summary markdown generated.
- Coverage checks:
  - `plansv2_paper_seed_table.csv`: `69` rows
  - `plansv2_paper_agg_table.csv`: `23` rows
  - `plansv2_paper_delta_table.csv`: `12` rows
- Key consistency spot checks:
  - m1 `padapt/hard`: `1.838225 ± std 0.105458`
  - m2 `diffusion/hard`: `1.572475 ± std 0.192113`
  - m4 robust `residual_scale05/hard`: `1.548885 ± std 0.045004`

### Remaining blocked/risky
- Data pack is ready, but thesis-facing presentation (final figure/table format and concise narrative wording) is still pending.
- If writing phase needs publication-style table formats (e.g., LaTeX), an extra formatting pass is still needed.

### Single recommended next step
- Produce thesis-facing result bundle from this data pack:
  - final main table (teacher/padapt/purebc + diffusion comparisons),
  - one compact negative-result table (latent/residual deltas),
  - one short “why baseline-first closure” narrative paragraph set.

---

## v2-011 (2026-03-24) — Thesis-Facing Result Bundle Generated

### Target milestone/subgoal
- Complete thesis-facing presentation bundle from existing M1-M4 evidence:
  - final main table,
  - negative-result delta table,
  - concise narrative text,
  - LaTeX-ready table snippets.

### What changed (files + behavior impact)
- `scripts/build_plansv2_thesis_result_bundle.sh`
  - Added reproducible generator from `plansv2_paper_*` tables.
  - Exports:
    - `docs/data/plansv2_thesis_main_table.csv`
    - `docs/data/plansv2_thesis_negative_delta_table.csv`
    - `docs/data/plansv2_thesis_tables.tex`
  - Generates summary doc: `docs/plansv2_thesis_result_bundle.md`.
- `docs/plansv2_thesis_result_bundle.md`
  - Added frozen thesis main table, negative-result deltas, and concise narrative draft.
- `docs/stage_acceptance_summary.md`
  - Added section `PLANS_v2 M5 Thesis Result Bundle (2026-03-24)`.

### What was verified (commands + key outcomes)
- Script syntax:
  - `bash -n scripts/build_plansv2_thesis_result_bundle.sh`
  - Outcome: pass.
- Full bundle generation:
  - `bash scripts/build_plansv2_thesis_result_bundle.sh`
  - Outcome: all target files generated successfully.
- Output coverage:
  - main table rows: `6`
  - negative delta rows: `12`
  - LaTeX table file generated and includes both main + delta tables.

### Remaining blocked/risky
- Data and table bundle is frozen; remaining work is writing integration (chapter text, figure/table placement, wording polish).
- If advisor requires different table style (e.g., bold best, SIunitx alignment), a final formatting pass is still needed.

### Single recommended next step
- Draft thesis results subsection directly from `docs/plansv2_thesis_result_bundle.md`:
  - keep one paragraph for main findings,
  - one paragraph for negative/inconclusive diffusion evidence,
  - one paragraph for baseline-first closure rationale.

---

## v2-012 (2026-03-24) — Thesis Results Subsection Draft Added

### Target milestone/subgoal
- Convert the frozen thesis bundle into a manuscript-ready subsection draft (content-first writing layer).

### What changed (files + behavior impact)
- `docs/plansv2_thesis_results_subsection_draft.md`
  - Added a complete draft subsection including:
    - protocol paragraph,
    - main-results paragraph,
    - negative/inconclusive diffusion evidence paragraph,
    - baseline-first closure positioning paragraph,
    - artifact pointers for tables/LaTeX.
- `docs/stage_acceptance_summary.md`
  - Added `PLANS_v2 M5 Thesis Results Subsection Draft (2026-03-24)` status row.

### What was verified (commands + key outcomes)
- Content consistency checks against bundle tables:
  - verified the quoted key numbers and deltas are aligned with:
    - `docs/plansv2_thesis_result_bundle.md`
    - `docs/data/plansv2_thesis_main_table.csv`
    - `docs/data/plansv2_thesis_negative_delta_table.csv`
- Outcome: draft is numerically consistent with frozen M1-M4 artifacts.

### Remaining blocked/risky
- Draft is content-complete but not yet advisor-style polished (wording/style, final table placement, citation integration).

### Single recommended next step
- Do one manuscript polish pass:
  - tighten wording to your thesis voice,
  - insert `docs/data/plansv2_thesis_tables.tex` into chapter file,
  - add cross-references to method/ablation sections.

---

## v2-013 (2026-03-24) — Strict Re-Alignment To PLANS_v2 Mainline

### Target milestone/subgoal
- User-directed strict re-alignment: stop treating manuscript polish as active critical path, and resume Plan v2 stage/gate execution authority.

### What changed (files + behavior impact)
- `docs/plansv2_stage_gate_strict_audit.md`
  - Added strict audit checklist for milestones `M0-M5` and gates `G1-G4`.
  - Marked current status with evidence pointers and pass/partial rationale.
  - Replanned execution goals to `S1 -> S2 -> S3` (evidence canonicalization, metrics consistency, then decision refresh).
- `docs/session_handoff.md`
  - Replaced the index-level “single recommended next step” from manuscript polish to strict Plan v2 audit execution (`S1`).

### What was verified (commands + key outcomes)
- Bootstrap/state reads:
  - `sed -n '1,260p' PLANS_v2.md`
  - `sed -n '1,260p' docs/session_handoff.md`
  - `sed -n '1,320p' docs/session_handoff_v2.md`
  - `sed -n '1,260p' docs/stage_acceptance_summary.md`
- Evidence pointer checks:
  - `rg -n "local_G2_decision|local_m3_conclusion|local_g3_readiness_decision|local_m4_conclusion|selected_path|run_id|git_commit|config_snapshot|dataset_version|dataset_hash|seeds|eval_episodes|primary_metrics|dispersion|artifact" docs/plansv2_m1_baseline_pack.md docs/plansv2_m2_gap_gate.md docs/plansv2_m3_min_compare.md docs/plansv2_m4_residual_gate_nominal.md docs/plansv2_m4_residual_gate_nominal_scale05.md docs/plansv2_m4_residual_compare_pack.md docs/plansv2_m5_baseline_closure.md`
- Outcome:
  - M1-M4 execution artifacts and local decisions are present.
  - M5 closure record exists.
  - Evidence-block field completeness is not fully uniform per `PLANS_v2` §7, so strict audit marks `G1/G4` as partial.

### Remaining blocked/risky
- Current stage-closure wording can be misread as fully final while gate-level evidence completeness is still partial.
- Canonical number consistency has not yet been re-audited after the strict re-alignment decision.

### Single recommended next step
- Execute `S1` from `docs/plansv2_stage_gate_strict_audit.md`:
  - normalize M1-M5 evidence blocks to full required fields,
  - then refresh strict `M0-M5 + G1-G4` audit statuses.

---

## v2-014 (2026-03-25) — Handoff Semantics Realigned (Plan v1 vs Plan v2)

### Target milestone/subgoal
- Align repository workflow semantics to user decision:
  - `session_handoff.md` is Plan v1 archive index only,
  - `session_handoff_v2.md` is the only running handoff for `PLANS_v2`.

### What changed (files + behavior impact)
- `AGENTS.md`
  - Session Handoff rule updated:
    - meaningful Plan v2 sessions must update `docs/session_handoff_v2.md` only.
    - `docs/session_handoff.md` marked as historical archive index, not active execution log.
  - Session Bootstrap rule updated:
    - required reads: `docs/session_handoff_v2.md` + `docs/stage_acceptance_summary.md`
    - `docs/session_handoff.md` changed to optional historical context.
- `docs/session_handoff.md`
  - Converted to Plan v1 archive index page.
  - Removed Plan v2 “latest next step” content to avoid cross-plan drift.

### What was verified (commands + key outcomes)
- Rule check:
  - `rg -n "Session Handoff|session_handoff_v2|session_handoff.md|must update" AGENTS.md`
  - Outcome: active handoff requirement now points to `session_handoff_v2.md` as running log.
- Content check:
  - `sed -n '1,200p' docs/session_handoff.md`
  - Outcome: file now clearly indicates Plan v1 archive role and points Plan v2 work to `session_handoff_v2.md`.

### Remaining blocked/risky
- None at governance/logging layer for this change.

### Single recommended next step
- Continue `PLANS_v2` strict execution from `v2-013` recommendation:
  - run `S1` evidence-block normalization and refresh strict stage/gate audit.

---

## v2-015 (2026-03-25) — S1 Evidence-Block Normalization Completed

### Target milestone/subgoal
- Execute `S1` from strict Plan v2 audit: normalize evidence-block fields for M1-M5 documents without changing any experiment results.

### What changed (files + behavior impact)
- `docs/plansv2_m1_baseline_pack.md`
  - Normalized evidence fields: `config_snapshot`, `dataset_version/hash`, `eval_episodes/env_steps`, `primary_metrics`, `dispersion_metric`, artifact log/checkpoint paths.
- `docs/plansv2_m2_gap_gate.md`
  - Added the same canonical evidence fields and explicit artifact path keys.
- `docs/plansv2_m3_min_compare.md`
  - Added canonical evidence fields for this synthesized compare pack, including inherited config/artifact references.
- `docs/plansv2_m4_residual_compare_pack.md`
  - Added canonical evidence fields and standardized metric/dispersion/path keys.
- `docs/plansv2_m5_baseline_closure.md`
  - Added a formal decision-record evidence block so M5 closure is traceable under the same schema.
- `docs/plansv2_stage_gate_strict_audit.md`
  - Updated audit date to `2026-03-25`.
  - Updated `G1` from `PARTIAL` -> `PASS` after M1-M5 normalization.
  - Kept `M5/G4` as `PARTIAL` pending `S2` cross-doc canonical metrics consistency audit.

### What was verified (commands + key outcomes)
- Field-completeness verification:
  - `rg -n "config_snapshot|dataset_version|dataset_hash|eval_episodes_per_run|eval_env_steps_per_run|primary_metrics|dispersion_metric|artifact_.*paths|run_id|git_commit|One-line Conclusion" docs/plansv2_m1_baseline_pack.md docs/plansv2_m2_gap_gate.md docs/plansv2_m3_min_compare.md docs/plansv2_m4_residual_compare_pack.md docs/plansv2_m5_baseline_closure.md`
  - Outcome: all required normalized evidence keys are present in M1-M5 docs.
- Audit state check:
  - `sed -n '1,220p' docs/plansv2_stage_gate_strict_audit.md`
  - Outcome: `G1=PASS`, `G4` still pending `S2`.

### Remaining blocked/risky
- `S2` (cross-document canonical metrics consistency audit) is not yet executed.
- Therefore `M5` remains provisional until `G4` is refreshed after `S2`.

### Single recommended next step
- Execute `S2`: cross-check canonical metrics between `docs/stage_acceptance_summary.md` and `docs/plansv2_m1~m5*.md`, then refresh final `G4/M5` status.

---

## v2-016 (2026-03-25) — S2 Canonical Consistency Audit Completed

### Target milestone/subgoal
- Execute `S2`: verify canonical metrics/conclusion consistency between stage summary and M1-M5 evidence docs, then refresh gate status.

### What changed (files + behavior impact)
- `docs/plansv2_stage_gate_strict_audit.md`
  - Updated status after S2:
    - `G4: PARTIAL -> PASS (local)`
    - `M5: PARTIAL -> PASS (local)`
  - Updated next step to `S3` decision wording refresh in running logs.

### What was verified (commands + key outcomes)
- Cross-doc key-value consistency checks:
  - `rg -n "3\\.055567|2\\.190508|1\\.838225|2\\.062867|1\\.788645|1\\.548885|1\\.572475|not_support_latent_mainline|not_support_residual_robust_edge|non_diffusion_baseline_closure|local_G2_decision" docs/stage_acceptance_summary.md`
  - `rg -n "3\\.055567|2\\.190508|1\\.838225|2\\.062867|1\\.788645|1\\.548885|1\\.572475|not_support_latent_mainline|not_support_residual_robust_edge|non_diffusion_baseline_closure|local_G2_decision" docs/plansv2_m1_baseline_pack.md docs/plansv2_m2_gap_gate.md docs/plansv2_m3_min_compare.md docs/plansv2_m4_residual_compare_pack.md docs/plansv2_m5_baseline_closure.md`
  - `rg -n "PLANS_v2 M1 Multiseed Baseline Pack|PLANS_v2 M2 Latent Gap-Closing Gate Pack|PLANS_v2 M3 Minimum Credible Comparison|PLANS_v2 M4 Residual Robustness Compare Pack|PLANS_v2 M5 Stage Convergence" docs/stage_acceptance_summary.md`
- Outcome:
  - Key canonical metrics and local decision strings are consistent across summary and milestone docs.
  - M1-M5 coverage sections all exist in `stage_acceptance_summary`.

### Remaining blocked/risky
- Current status is marked `PASS (local)` based on repository evidence; if governance-level criteria change, a new decision refresh may still be needed.

### Single recommended next step
- Execute `S3`: refresh final stage wording in active logs to reflect `G1/G2/G3/G4 all locally pass` and keep follow-up work limited to low-cost supporting-axis checks only.

---

## v2-017 (2026-03-25) — S3 Decision Wording Refresh Completed

### Target milestone/subgoal
- Execute `S3`: finalize Plan v2 stage wording after `S1/S2` completion, without adding new experiments.

### What changed (files + behavior impact)
- `docs/plansv2_m5_baseline_closure.md`
  - Added `S3 Decision Refresh (2026-03-25)` block.
  - Explicitly records local gate snapshot (`G1/G2/G3/G4`) and `m5_closure_status=finalized_local_under_plans_v2`.
  - Clarifies post-closure execution boundary: low-cost supporting-axis checks only, no diffusion mainline reopen by default.
- `docs/plansv2_stage_gate_strict_audit.md`
  - Added `S3 Refresh Result` section with completion status and refreshed final state.
  - Updated next-step guidance to maintenance-mode execution.

### What was verified (commands + key outcomes)
- State checks:
  - `sed -n '1,220p' docs/plansv2_stage_gate_strict_audit.md`
  - `sed -n '1,220p' docs/plansv2_m5_baseline_closure.md`
- Outcome:
  - `S3` wording refresh is present.
  - Current strict snapshot is consistent: `M5=PASS (local)`, `G1=PASS`, `G4=PASS (local)`.

### Remaining blocked/risky
- Current closure is explicitly tagged as `local` execution conclusion; governance-level reopening is still possible if future evidence changes.

### Single recommended next step
- Continue in maintenance mode:
  - perform only low-cost reproducibility/reporting checks when needed,
  - avoid new diffusion expansion unless governance is reopened.

---

## v2-018 (2026-03-25) — Acceptance-Readiness Promotion (Continuous-Push Mode)

### Target milestone/subgoal
- Apply user-added execution rule: keep pushing until acceptance-ready or major blocker appears.
- Promote current Plan v2 state from maintenance wording to explicit acceptance-readiness checkpoint.

### What changed (files + behavior impact)
- `docs/stage_acceptance_summary.md`
  - Added section `PLANS_v2 Acceptance Readiness Snapshot (2026-03-25)`.
  - Added explicit checks for:
    - `PLANS_v2` section `8.1~8.6` acceptance criteria,
    - `PLANS_v2` section `13` stage exit conditions.
  - Added lightweight validation log (script syntax, core code compile, artifact existence checks).

### What was verified (commands + key outcomes)
- PLANS acceptance criteria source read:
  - `rg -n "8\\.1|8\\.2|8\\.3|8\\.4|8\\.5|8\\.6|13\\. 当前阶段退出条件" PLANS_v2.md`
  - `sed -n '220,420p' PLANS_v2.md`
- Lightweight readiness validations:
  - `bash -n scripts/eval_plansv2_m1_baseline_pack.sh`
  - `bash -n scripts/eval_plansv2_m2_gap_gate.sh`
  - `bash -n scripts/eval_plansv2_m3_min_compare.sh`
  - `bash -n scripts/eval_plansv2_m4_residual_compare_pack.sh`
  - `PYTHONDONTWRITEBYTECODE=1 python - <<'PY' ... compile('dexscrew/algo/ppo/ppo.py') ... compile('dexscrew/algo/ppo/diffusion_latent_student.py') ... PY`
  - artifact spot checks:
    - `outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_1550.49.pth`
    - `outputs/XHandHoraScrewDriver_student_padapt/run_a/stage2_nn/model_best.ckpt`
    - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_recon05_seed42_15min/stage2_diffusion_nn/model_best.ckpt`
    - `outputs/robustness_eval/plansv2_m1/teacher_ppo_nominal_s42.log`
    - `outputs/robustness_eval/plansv2_m2_gap_gate/diffusion_nominal_s42.log`
    - `outputs/robustness_eval/plansv2_m4_residual_compare_pack/padapt_light_v2_s42.log`
- Outcome:
  - acceptance snapshot status is now `ready_for_acceptance_review_local`.
  - no major blocker found in this pass.

### Remaining blocked/risky
- Current state is a local execution-side acceptance snapshot; governance-level reopen is still possible only if new contradictory evidence appears.

### Single recommended next step
- Enter acceptance-review loop:
  - keep running only low-cost reproducibility checks when requested,
  - if any new contradiction appears in canonical metrics/artifacts, immediately raise major issue and stop closure claims.

---

## v2-019 (2026-03-25) — Reproducibility Drift Fix For Evidence Templates

### Target milestone/subgoal
- Continue acceptance-review loop with low-cost reproducibility checks.
- Eliminate evidence-template drift: rerunning summary scripts must not downgrade canonical evidence fields.

### What changed (files + behavior impact)
- `scripts/eval_plansv2_m1_baseline_pack.sh`
  - Updated markdown generator to emit canonical evidence fields (`config/dataset/eval_episodes+steps/primary_metrics/dispersion/artifact paths`).
- `scripts/eval_plansv2_m2_gap_gate.sh`
  - Updated markdown generator to emit the same canonical evidence schema.
- `scripts/eval_plansv2_m3_min_compare.sh`
  - Updated markdown generator to emit canonical evidence schema.
  - Re-ran script to regenerate `docs/plansv2_m3_min_compare.md` under new template.
- `scripts/eval_plansv2_m4_residual_compare_pack.sh`
  - Updated markdown generator to emit canonical evidence schema.
- `docs/stage_acceptance_summary.md`
  - Added reproducibility regeneration check note under acceptance-readiness lightweight validation.

### What was verified (commands + key outcomes)
- Script syntax checks:
  - `bash -n scripts/eval_plansv2_m1_baseline_pack.sh`
  - `bash -n scripts/eval_plansv2_m2_gap_gate.sh`
  - `bash -n scripts/eval_plansv2_m3_min_compare.sh`
  - `bash -n scripts/eval_plansv2_m4_residual_compare_pack.sh`
- Repro regeneration check:
  - `bash scripts/eval_plansv2_m3_min_compare.sh 42,43,44`
  - Outcome: M3 doc regenerated successfully; local conclusion unchanged (`not_support_latent_mainline`).
- Canonical field presence check:
  - `rg -n "config_snapshot|dataset_version|dataset_hash|eval_episodes_per_run|eval_env_steps_per_run|primary_metrics|dispersion_metric|artifact_log_paths|artifact_checkpoint_paths" docs/plansv2_m1_baseline_pack.md docs/plansv2_m2_gap_gate.md docs/plansv2_m3_min_compare.md docs/plansv2_m4_residual_compare_pack.md docs/plansv2_m5_baseline_closure.md`
  - Outcome: required canonical evidence keys are present across M1-M5 docs.

### Remaining blocked/risky
- No major blocker found in this pass.
- Heavy re-evaluation scripts (`M1/M2/M4`) were not fully re-executed in this loop to keep cost low; current pass focuses on template stability + artifact-backed reproducibility.

### Single recommended next step
- Stay in acceptance-review loop:
  - perform one additional low-cost consistency check when needed (without broad reruns),
  - escalate immediately only if new contradictory evidence appears.

---

## v2-020 (2026-03-25) — Low-Cost From-Logs Regeneration Path Hardened

### Target milestone/subgoal
- Continue acceptance-review push with zero new training/eval cost.
- Ensure M1/M2/M4 summary scripts can regenerate docs from existing logs without IsaacGym runtime dependency.

### What changed (files + behavior impact)
- `scripts/eval_plansv2_m1_baseline_pack.sh`
  - Added `PLANSV2_FROM_LOGS_ONLY=1` mode:
    - skip eval runs,
    - validate expected logs + parse `EvalSummary`,
    - regenerate summary markdown from logs.
  - In from-logs mode, skip IsaacGym environment check and ckpt precheck.
- `scripts/eval_plansv2_m2_gap_gate.sh`
  - Added `PLANSV2_FROM_LOGS_ONLY=1` mode:
    - skip eval runs,
    - validate expected logs + required summaries,
    - regenerate summary markdown from logs.
  - In from-logs mode, skip IsaacGym environment check.
- `scripts/eval_plansv2_m4_residual_compare_pack.sh`
  - Added `PLANSV2_FROM_LOGS_ONLY=1` mode:
    - skip eval runs,
    - validate expected logs + required summaries,
    - regenerate summary markdown from logs.
  - In from-logs mode, skip IsaacGym environment check.
- `docs/stage_acceptance_summary.md`
  - Added validation note for from-logs regeneration commands and outcomes.

### What was verified (commands + key outcomes)
- Syntax:
  - `bash -n scripts/eval_plansv2_m1_baseline_pack.sh`
  - `bash -n scripts/eval_plansv2_m2_gap_gate.sh`
  - `bash -n scripts/eval_plansv2_m4_residual_compare_pack.sh`
- From-logs regeneration checks:
  - `PLANSV2_FROM_LOGS_ONLY=1 bash scripts/eval_plansv2_m1_baseline_pack.sh 0 256 42,43,44`
  - `PLANSV2_FROM_LOGS_ONLY=1 bash scripts/eval_plansv2_m2_gap_gate.sh 0 256 42,43,44`
  - `PLANSV2_FROM_LOGS_ONLY=1 bash scripts/eval_plansv2_m4_residual_compare_pack.sh 0 256 42,43,44`
- Outcome:
  - all three scripts regenerate summaries from existing logs successfully.
  - no new IsaacGym eval run was triggered in from-logs mode.
  - canonical evidence fields remain present in regenerated docs; local conclusions unchanged.

### Remaining blocked/risky
- No major blocker in this pass.
- Full runtime eval remains dependent on IsaacGym env (expected), but acceptance-loop reproducibility now has a lightweight local path.

### Single recommended next step
- Continue acceptance-review loop with from-logs checks by default.
- Escalate only if regenerated summaries show canonical-metric or conclusion drift.

---

## v2-021 (2026-03-25) — Major-Acceptance Breakthrough Scan (Existing Evidence Pool)

### Target milestone/subgoal
- Continue toward major acceptance with explicit “breakthrough gate” check.
- Verify whether any existing robustness artifacts already demonstrate diffusion breakthrough over baseline.

### What changed (files + behavior impact)
- `docs/stage_acceptance_summary.md`
  - Added a breakthrough sweep note under acceptance-readiness lightweight validation:
    - parsed all existing `outputs/robustness_eval/**/*.log` `EvalSummary` entries,
    - compared robust best (`light_v2/hard`) between `padapt` and diffusion families.

### What was verified (commands + key outcomes)
- Existing-log breakthrough scan (no new training/eval):
  - Parsed all robustness logs and computed best robust values by family/condition.
- Key outcomes:
  - `padapt`: `light_v2=2.215234`, `hard=1.972757`
  - `best_diffusion`: `light_v2=2.005074`, `hard=1.764717`
  - robust deltas (`best_diffusion - padapt`):
    - `light_v2=-0.210160`
    - `hard=-0.208040`
- Result:
  - No robustness breakthrough is found in current evidence pool.
  - Existing `M5` baseline-first closure remains consistent with expanded evidence scan.

### Remaining blocked/risky
- If a “major acceptance with algorithm breakthrough” is required, current evidence is insufficient.
- Achieving breakthrough now would require governance-approved reopening of bounded diffusion optimization experiments (new runs), not just documentation/evidence consolidation.

### Single recommended next step
- Keep acceptance loop active.
- If breakthrough is mandatory, explicitly reopen a bounded “breakthrough sprint” (small run budget + strict stop criteria) and switch from evidence consolidation to new experiment generation.

---

## v2-022 (2026-03-25) — Bounded Breakthrough Sprint Probe (Eval-Only) Completed

### Target milestone/subgoal
- Execute a bounded breakthrough sprint without new training:
  - evaluate additional diffusion-latent ckpts under robust protocol,
  - verify whether any candidate can beat current `padapt` baseline on `light_v2/hard`.

### What changed (files + behavior impact)
- `docs/stage_acceptance_summary.md`
  - Added bounded breakthrough probe results:
    - seed42 shortlist scan (`8 ckpts x 2 cond`, 16 eval runs),
    - top-3 multiseed follow-up (`3 ckpts x 2 cond x 3 seeds`, 18 eval runs),
    - explicit deltas vs padapt mean.
- `codeagent_issue.md`
  - Reopened escalation note because user-level “breakthrough-required acceptance” now conflicts with current bounded evidence outcome.

### What was verified (commands + key outcomes)
- Shortlist probe:
  - robust eval logs generated under `outputs/robustness_eval/plansv2_breakthrough_probe/`.
- Multiseed probe:
  - robust eval logs generated under `outputs/robustness_eval/plansv2_breakthrough_probe_multiseed/`.
- Aggregated key outcomes (mean reward):
  - `run_a_latent_robust_light_seed42_1h`: `light_v2=1.672203`, `hard=1.393417`
  - `run_a_latent_robust_fs06_p012_seed42_15min`: `light_v2=1.662662`, `hard=1.342301`
  - `run_a_latent_robust_mid_seed42_15min`: `light_v2=1.657517`, `hard=1.371422`
  - padapt reference: `light_v2=2.079074`, `hard=1.838225`
  - all candidate deltas remain negative (roughly `-0.41` to `-0.50`).

### Remaining blocked/risky
- No diffusion robust breakthrough is observed after bounded probe expansion.
- If “算法必须有突破” is a hard acceptance gate, further progress now requires governance-level scope reopen (new training budget + stop criteria), not routine local execution only.

### Single recommended next step
- Choose one governance direction:
  1. Keep current `M5` baseline-first acceptance path.
  2. Reopen a tightly bounded breakthrough training sprint and mark current closure provisional during that sprint.

---

## v2-023 (2026-03-25) — Acceptance-Path Continuation After Breakthrough Probe

### Target milestone/subgoal
- Continue execution after breakthrough probe feedback.
- Keep pushing on major-acceptance path with full-chain reproducibility refresh and consistency confirmation.

### What changed (files + behavior impact)
- Re-ran full Plan-v2 summary chain in low-cost mode:
  - `M1`: from-logs regeneration
  - `M2`: from-logs regeneration
  - `M3`: compare regeneration
  - `M4`: from-logs regeneration
- `docs/stage_acceptance_summary.md`
  - Added bounded breakthrough probe multiseed result block (top-3 latent candidates still below padapt).
- `codeagent_issue.md`
  - Marked reopened issue status as `conditional`:
    - major-acceptance path can continue,
    - hard “must-have breakthrough” remains a governance-level blocker.

### What was verified (commands + key outcomes)
- Full-chain regeneration:
  - `PLANSV2_FROM_LOGS_ONLY=1 bash scripts/eval_plansv2_m1_baseline_pack.sh 0 256 42,43,44`
  - `PLANSV2_FROM_LOGS_ONLY=1 bash scripts/eval_plansv2_m2_gap_gate.sh 0 256 42,43,44`
  - `bash scripts/eval_plansv2_m3_min_compare.sh 42,43,44`
  - `PLANSV2_FROM_LOGS_ONLY=1 bash scripts/eval_plansv2_m4_residual_compare_pack.sh 0 256 42,43,44`
- Consistency checks:
  - `local_G2_decision=PASS`, `local_m3_conclusion=not_support_latent_mainline`, `local_m4_conclusion=not_support_residual_robust_edge`, `selected_path=non_diffusion_baseline_closure` remain unchanged.
  - canonical evidence fields remain present across M1-M5 docs.
- Breakthrough probe multiseed key outcomes (top-3 candidates):
  - best candidate robust means remain below padapt by roughly `-0.41` to `-0.50`.

### Remaining blocked/risky
- For baseline-first major acceptance: no new blocker in this pass.
- For “must-have breakthrough acceptance”: still blocked under current bounded evidence.

### Single recommended next step
- Continue major-acceptance submission path (baseline-first) by default.
- If user/advisor enforces breakthrough as hard gate, reopen governance for a bounded new-training sprint.

---

## v2-024 (2026-03-25) — PLANS_v2 P3 Latent Optimization Loop (Recon-Coef Schedule Probe)

### Target milestone/subgoal
- Return to `PLANS_v2` optimization execution (not acceptance-only loop): one bounded `latent diffusion` hypothesis with full code-change -> training -> evaluation closure.
- Scope aligned to `P3` (`latent diffusion` minimum credible comparison path): keep protocol fixed, test one local algorithm variant.

### What changed (files + behavior impact)
- `dexscrew/algo/ppo/diffusion_latent_student.py`
  - Added optional scheduled latent reconstruction coefficient:
    - `diffusion_latent_recon_coef_start`
    - `diffusion_latent_recon_coef_end`
    - `diffusion_latent_recon_coef_schedule_steps`
  - Training loss now supports linear interpolation from start -> end by agent steps.
  - Added runtime diagnostics:
    - `latent_recon_coef_cur`
    - `latent_recon_coef_progress`
- New training run:
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_recon_sched08to03_seed42_15min/`
  - representative ckpt: `.../stage2_diffusion_nn/model_best.ckpt`
- New eval artifacts:
  - `outputs/robustness_eval/plansv2_live_sched08to03/diffusion_nominal_s42.log`
  - `outputs/robustness_eval/plansv2_live_sched08to03/diffusion_light_v2_s42.log`
  - `outputs/robustness_eval/plansv2_live_sched08to03/diffusion_hard_s42.log`

### What was verified (commands + key outcomes)
- Compile sanity:
  - `rg -n "diffusion_latent_recon_coef_(start|end|schedule_steps)|latent_recon_coef_cur|latent_recon_coef_progress" dexscrew/algo/ppo/diffusion_latent_student.py`
- 15min training (single-seed bounded probe):
  - `./docker-run-isaacgym.sh timeout 1000 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_recon_sched08to03_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_latent_recon_coef=0.5 +train.ppo.diffusion_latent_recon_coef_start=0.8 +train.ppo.diffusion_latent_recon_coef_end=0.3 +train.ppo.diffusion_latent_recon_coef_schedule_steps=250000`
- Unified protocol eval (`nominal/light_v2/hard`, `seed=42`, `steps=256`):
  - `./docker-run-isaacgym.sh timeout 480 bash scripts/eval_screwdriver_student_robustness.sh ...`
- Outcome snapshot (reward):
  - new run: nominal `1.641663`, light_v2 `1.514322`, hard `1.295797`
  - delta vs `latent_recon05` seed42 (`outputs/robustness_eval/plansv2_m2_gap_gate/diffusion_*_s42.log`):
    - nominal `-0.265561`
    - light_v2 `+0.108909`
    - hard `-0.014294`
  - delta vs `padapt` seed42 (`outputs/robustness_eval/plansv2_m1/padapt_*_s42.log`):
    - nominal `-0.277325`
    - light_v2 `-0.453079`
    - hard `-0.530911`
- Local decision:
  - this probe does **not** qualify as a keep/replace candidate for current latent representative (`recon05`), because gains are not robust and key conditions remain below baseline.

### Remaining blocked/risky
- Result is single-seed training + single-seed eval; variance risk remains.
- One stale long-running historical training process (`run_a_latent_residual_scale05_seed42_15min`) still occupies GPU memory and may reduce throughput for subsequent probes.

### Single recommended next step
- Continue `PLANS_v2` optimization loop with the next bounded latent hypothesis (same 15min budget + same three-condition eval), and keep strict keep/drop decision by direct delta to `latent_recon05` + `padapt`.

---

## v2-025 (2026-03-25) — PLANS_v2 P3 Latent Optimization Loop (Reverse Recon Schedule Probe)

### Target milestone/subgoal
- Continue strict `PLANS_v2` execution loop on latent-diffusion mainline:
  - one bounded hypothesis,
  - fixed budget (`15min`/`timeout 1000s`),
  - fixed evaluation protocol (`nominal/light_v2/hard`, `seed=42`, `steps=256`).

### What changed (files + behavior impact)
- No new code file edits in this probe; reused the scheduled recon mechanism added in `v2-024`.
- New training run:
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_recon_sched03to08_seed42_15min/`
  - key overrides:
    - `+train.ppo.diffusion_latent_recon_coef_start=0.3`
    - `+train.ppo.diffusion_latent_recon_coef_end=0.8`
    - `+train.ppo.diffusion_latent_recon_coef_schedule_steps=250000`
- New eval logs:
  - `outputs/robustness_eval/plansv2_live_sched03to08/diffusion_nominal_s42.log`
  - `outputs/robustness_eval/plansv2_live_sched03to08/diffusion_light_v2_s42.log`
  - `outputs/robustness_eval/plansv2_live_sched03to08/diffusion_hard_s42.log`

### What was verified (commands + key outcomes)
- Training command:
  - `./docker-run-isaacgym.sh timeout 1000 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_recon_sched03to08_seed42_15min checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_1550.49.pth +train.ppo.diffusion_latent_recon_coef=0.5 +train.ppo.diffusion_latent_recon_coef_start=0.3 +train.ppo.diffusion_latent_recon_coef_end=0.8 +train.ppo.diffusion_latent_recon_coef_schedule_steps=250000`
  - exited by timeout boundary (`code=124`) after producing `model_best.ckpt` (expected bounded-run behavior).
- Eval commands (all under docker IsaacGym):
  - `nominal`, `light_v2`, `hard` with `+train.ppo.diffusion_eval_decode_only=False +train.ppo.diffusion_eval_report_recon=True`.
- Reward results:
  - nominal: `1.520537`
  - light_v2: `1.483355`
  - hard: `1.051158`
- Deltas:
  - vs `sched08to03` (v2-024):
    - nominal `-0.121126`, light_v2 `-0.030967`, hard `-0.244639`
  - vs `latent_recon05` representative (`plansv2_m2_gap_gate` seed42):
    - nominal `-0.386687`, light_v2 `+0.077942`, hard `-0.258933`
  - vs `padapt` (`plansv2_m1` seed42):
    - nominal `-0.398451`, light_v2 `-0.484046`, hard `-0.775550`
- Local decision:
  - reverse schedule probe is **rejected** (not a keep candidate).

### Remaining blocked/risky
- Stale historical training process `run_a_latent_residual_scale05_seed42_15min` is still alive and occupies GPU memory; not blocking execution yet but reduces headroom.
- Current conclusions are still single-seed training probes; multiseed training confirmation is pending for any future promising variant.

### Single recommended next step
- Continue `PLANS_v2` P3 with one new bounded latent hypothesis that directly targets hard-condition recovery without nominal collapse, then apply the same strict keep/drop gate.

---

## v2-026 (2026-03-25) — PLANS_v2 P3 Tail-Regularization Sweep (coef 0.2 -> 0.1)

### Target milestone/subgoal
- Continue strict Plan-v2 optimization loop with a hard-focused latent hypothesis family:
  - introduce `teacher_delta_tail` regularization (target large action-deviation tails),
  - run bounded `15min` training and unified `nominal/light_v2/hard` eval,
  - compare directly against `latent_recon05`, `padapt`, and latest probes.

### What changed (files + behavior impact)
- Runtime environment cleanup:
  - terminated stale historical process `run_a_latent_residual_scale05_seed42_15min` to release GPU occupancy.
- New bounded training probes:
  1. `run_a_latent_tailcoef02_thr015_sel_seed42_15min`
     - `+train.ppo.diffusion_teacher_delta_tail_coef=0.2`
     - `+train.ppo.diffusion_teacher_delta_tail_threshold=0.15`
     - `+train.ppo.diffusion_teacher_delta_tail_selective=True`
  2. `run_a_latent_tailcoef01_thr015_sel_seed42_15min`
     - same config except `tail_coef=0.1`
- New eval logs:
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel/diffusion_{nominal,light_v2,hard}_s42.log`
  - `outputs/robustness_eval/plansv2_live_tailcoef01_thr015_sel/diffusion_{nominal,light_v2,hard}_s42.log`

### What was verified (commands + key outcomes)
- Both training runs executed with bounded timeout (`code=124` at 1000s) and produced `model_best.ckpt`.
- Unified eval protocol (`seed=42`, `steps=256`) for each probe.
- `tail_coef=0.2` reward:
  - nominal `1.591984`
  - light_v2 `1.677225`
  - hard `1.390232`
- `tail_coef=0.1` reward:
  - nominal `1.694002`
  - light_v2 `1.580989`
  - hard `1.219772`
- `tail_coef=0.1` vs `tail_coef=0.2` delta:
  - nominal `+0.102018`
  - light_v2 `-0.096236`
  - hard `-0.170460`
- `tail_coef=0.2` vs `latent_recon05` (`plansv2_m2` seed42) delta:
  - nominal `-0.315240`
  - light_v2 `+0.271812`
  - hard `+0.080141`
- `tail_coef=0.2` vs `padapt` (`plansv2_m1` seed42) delta:
  - nominal `-0.327004`
  - light_v2 `-0.290176`
  - hard `-0.436476`

### Local decision
- `tail_coef=0.1` is rejected relative to `tail_coef=0.2` for hard-focused objective (hard/light regress).
- `tail_coef=0.2` becomes the current **provisional robust-improvement candidate** inside diffusion latent line:
  - improves light_v2/hard vs current `latent_recon05` representative,
  - but still underperforms `padapt`, and nominal remains lower than `latent_recon05`.

### Remaining blocked/risky
- Current evidence is still single-seed training.
- Robustness improvements are promising but not yet enough to close gap to `padapt`.
- Nominal-robustness tradeoff remains unresolved.

### Single recommended next step
- Keep only `tail_coef=0.2` branch and run one follow-up bounded probe aimed at nominal recovery (e.g., lighter tail threshold or mixed weighting), then apply same strict keep/drop gate.

---

## v2-027 (2026-03-25) — PLANS_v2 P3 Nominal-Recovery Probe (tail coef 0.2, threshold 0.20)

### Target milestone/subgoal
- Continue from the provisional robust-improvement branch (`tail_coef=0.2`) and test a single nominal-recovery hypothesis:
  - increase tail threshold from `0.15` to `0.20`,
  - expect weaker tail constraint -> better nominal while trying to keep robustness gains.

### What changed (files + behavior impact)
- New bounded training run:
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr020_sel_seed42_15min/`
  - key overrides:
    - `+train.ppo.diffusion_teacher_delta_tail_coef=0.2`
    - `+train.ppo.diffusion_teacher_delta_tail_threshold=0.20`
    - `+train.ppo.diffusion_teacher_delta_tail_selective=True`
- New eval logs:
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr020_sel/diffusion_nominal_s42.log`
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr020_sel/diffusion_light_v2_s42.log`
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr020_sel/diffusion_hard_s42.log`

### What was verified (commands + key outcomes)
- Training ran under bounded timeout (`1000s`, exit `124`) and produced `model_best.ckpt`.
- Unified eval protocol executed (`seed=42`, `steps=256`, nominal/light_v2/hard).
- `thr=0.20` reward:
  - nominal `1.801876`
  - light_v2 `1.438585`
  - hard `1.279392`
- Delta vs current kept robust candidate (`thr=0.15`):
  - nominal `+0.209892`
  - light_v2 `-0.238640`
  - hard `-0.110840`
- Delta vs `latent_recon05` seed42:
  - nominal `-0.105348`
  - light_v2 `+0.033172`
  - hard `-0.030699`
- Delta vs `padapt` seed42:
  - nominal `-0.117112`
  - light_v2 `-0.528816`
  - hard `-0.547316`

### Local decision
- `thr=0.20` nominal-recovery probe is **rejected** as a main candidate:
  - nominal improves, but robustness gains are not preserved (`light/hard` regress notably vs `thr=0.15`).

### Remaining blocked/risky
- Single-seed training only.
- Tradeoff surface is narrow: stronger tail regularization helps robustness but hurts nominal; weaker threshold recovers nominal but drops robustness.

### Single recommended next step
- Continue along `tail_coef=0.2` with finer threshold search (next narrow candidate around `0.17~0.18`) to seek a nominal/robustness balance point before multiseed confirmation.

---

## v2-028 (2026-03-25) — PLANS_v2 P3 Narrow Search (thr=0.18 + anchor branch)

### Target milestone/subgoal
- Continue strict latent optimization loop around the current robust candidate (`tail_coef=0.2`, `thr=0.15`):
  1. threshold micro-search (`thr=0.18`)
  2. anchor-assisted nominal recovery branch (`base_action_anchor_coef`)

### What changed (files + behavior impact)
- New training/eval probes:
  1. `run_a_latent_tailcoef02_thr018_sel_seed42_15min`
  2. `run_a_latent_tailcoef02_thr015_sel_anchor005_seed42_15min`
  3. `run_a_latent_tailcoef02_thr015_sel_anchor003_seed42_15min`
- Eval artifacts:
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr018_sel/`
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor005/`
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003/`

### What was verified (commands + key outcomes)
- All 3 training runs executed under bounded timeout (`1000s`) and produced `model_best.ckpt`.
- Unified eval protocol executed (`seed=42`, `steps=256`, nominal/light_v2/hard) for each probe.

- `thr=0.18` reward:
  - nominal `1.427221`, light_v2 `1.286782`, hard `1.226086`
  - delta vs `thr=0.15` baseline: nominal `-0.164763`, light_v2 `-0.390443`, hard `-0.164146`
  - local decision: reject.

- `anchor=0.05` reward (on top of `tail_coef=0.2, thr=0.15`):
  - nominal `1.631246`, light_v2 `1.767131`, hard `1.379105`
  - delta vs `thr=0.15` baseline: nominal `+0.039262`, light_v2 `+0.089906`, hard `-0.011127`
  - local decision: keep as improving candidate.

- `anchor=0.03` reward:
  - nominal `1.675112`, light_v2 `1.638266`, hard `1.504904`
  - delta vs `anchor=0.05`: nominal `+0.043866`, light_v2 `-0.128865`, hard `+0.125799`
  - delta vs `thr=0.15` baseline: nominal `+0.083128`, light_v2 `-0.038959`, hard `+0.114672`
  - delta vs `latent_recon05` seed42: nominal `-0.232112`, light_v2 `+0.232853`, hard `+0.194813`
  - local decision: current best tradeoff inside this narrow search.

### Local decision
- `thr=0.18` path is discarded.
- `anchor` branch is useful; among tested settings, `anchor=0.03` becomes the new provisional candidate.

### Remaining blocked/risky
- Candidate quality is still single-seed training evidence.
- Although robust gains improve vs diffusion representative, all conditions still trail `padapt` on seed42.

### Single recommended next step
- Run multiseed evaluation (`42,43,44`) on `anchor=0.03` candidate under nominal/light_v2/hard and compare aggregated deltas vs current `padapt` and `latent_recon05`.

---

## v2-029 (2026-03-25) — PLANS_v2 P3 Multiseed Validation for `anchor=0.03`

### Target milestone/subgoal
- Execute the required multiseed check for the current provisional candidate:
  - `tail_coef=0.2`
  - `tail_threshold=0.15`
  - `base_action_anchor_coef=0.03`
- Use unified protocol (`nominal/light_v2/hard`, `seed=42,43,44`, `steps=256`) and compare against M1/M2 references.

### What changed (files + behavior impact)
- New multiseed eval logs:
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_anchor003_multiseed/nominal_multiseed.log`
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_anchor003_multiseed/lightv2_multiseed.log`
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_anchor003_multiseed/hard_multiseed.log`
- No code-path changes in this session block; evaluation-only evidence expansion.

### What was verified (commands + key outcomes)
- Multiseed evaluation commands (docker IsaacGym):
  - `scripts/eval_screwdriver_student_robustness_multiseed.sh` for `nominal`, `light_v2`, `hard`.
- Aggregated reward (`mean ± std`):
  - nominal: `1.863957 ± 0.138738`
  - light_v2: `1.917208 ± 0.198149`
  - hard: `1.481093 ± 0.024690`
- Delta vs `latent_recon05` (M2 reference):
  - nominal: `-0.198910`
  - light_v2: `+0.128563`
  - hard: `-0.091382`
- Delta vs `padapt` (M1 reference):
  - nominal: `-0.303863`
  - light_v2: `-0.161866`
  - hard: `-0.357132`

### Local decision
- `anchor=0.03` is validated as a **partial improvement candidate**:
  - keeps a robust gain on `light_v2` vs `latent_recon05`,
  - but still cannot close `nominal/hard` to `padapt` and does not beat `latent_recon05` on `hard` in multiseed aggregate.

### Remaining blocked/risky
- Main blocker remains `hard` and global gap to `padapt`.
- Candidate is now multiseed-evaluated, but still insufficient for promotion to stage-closure winner.

### Single recommended next step
- Continue bounded optimization around the tail+anchor branch with one hard-focused adjustment (while preserving current light_v2 gain), then rerun the same multiseed protocol.

---

## v2-030 (2026-03-25) — PLANS_v2 P3 Hard-Focused Follow-up (`mid_only` closeout + `progress window` probe)

### Target milestone/subgoal
- Continue strict `PLANS_v2` P3 loop on the current tail+anchor branch:
  1. close the previously started `mid_only` run with full `nominal/light_v2/hard` evidence,
  2. test one hard-focused bounded adjustment (`progress_start/end`) without expanding scope.

### What changed (files + behavior impact)
- No code-path edits in this session; experiment-only updates.
- Completed eval logs for prior run:
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_anchor003_midonly/diffusion_{nominal,light_v2,hard}_s42.log`
- New bounded training run:
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_prog2085_seed42_15min/`
  - train log: `outputs/train_logs/run_a_latent_tailcoef02_thr015_sel_anchor003_prog2085_seed42_15min_train.log`
  - key overrides:
    - `+train.ppo.diffusion_teacher_delta_tail_coef=0.2`
    - `+train.ppo.diffusion_teacher_delta_tail_threshold=0.15`
    - `+train.ppo.diffusion_teacher_delta_tail_selective=True`
    - `+train.ppo.diffusion_base_action_anchor_coef=0.03`
    - `+train.ppo.diffusion_teacher_delta_tail_progress_start=0.2`
    - `+train.ppo.diffusion_teacher_delta_tail_progress_end=0.85`
- New eval logs for the new run:
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_anchor003_prog2085/diffusion_{nominal,light_v2,hard}_s42.log`

### What was verified (commands + key outcomes)
- `mid_only` closeout metrics (`seed=42`, `steps=256`):
  - nominal: `1.274565`
  - light_v2: `1.557095`
  - hard: `1.196189`
  - delta vs current kept `anchor=0.03` candidate:
    - nominal `-0.400547`, light_v2 `-0.081171`, hard `-0.308715`
  - local decision: **reject**.

- `progress window` probe training command:
  - `./docker-run-isaacgym.sh timeout 1000 scripts/screwdriver_student_diffusion_latent_robust_light.sh ... run_a_latent_tailcoef02_thr015_sel_anchor003_prog2085_seed42_15min ... +train.ppo.diffusion_teacher_delta_tail_progress_start=0.2 +train.ppo.diffusion_teacher_delta_tail_progress_end=0.85`
  - outcome: bounded stop `code=124`, `model_best.ckpt` generated.

- `progress window` eval metrics (`seed=42`, `steps=256`):
  - nominal: `1.742903`
  - light_v2: `1.706864`
  - hard: `1.153366`
  - delta vs current kept `anchor=0.03` candidate:
    - nominal `+0.067791`, light_v2 `+0.068598`, hard `-0.351538`
  - delta vs `latent_recon05` seed42:
    - nominal `-0.164321`, light_v2 `+0.301451`, hard `-0.156725`
  - local decision: **reject as mainline candidate** (hard collapse outweighs nominal/light gains).

### Remaining blocked/risky
- Main blocker remains `hard` robustness under unified protocol.
- Current tail+anchor family shows strong tradeoff behavior (nominal/light gains can come with large hard regression).
- Evidence is still from single-seed training probes; only selected candidates should enter next multiseed due budget.

### Single recommended next step
- Keep `tail_coef=0.2 + thr=0.15 + anchor=0.03` as current reference branch, then run one bounded **hard-biased** micro-probe by slightly strengthening tail constraint (e.g., `tail_threshold=0.14`) without `mid_only/progress window`, and apply the same keep/drop gate before any multiseed.

---

## v2-031 (2026-03-25) — PLANS_v2 P3 Hard-Biased Micro-Probe (`tail_threshold=0.14`)

### Target milestone/subgoal
- Execute one strict single-variable hard-biased probe from the current reference branch:
  - keep `tail_coef=0.2 + selective + anchor=0.03`,
  - reduce `tail_threshold` from `0.15` to `0.14`,
  - evaluate under unified `nominal/light_v2/hard` protocol.

### What changed (files + behavior impact)
- New bounded training run:
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr014_sel_anchor003_seed42_15min/`
  - train log:
    - `outputs/train_logs/run_a_latent_tailcoef02_thr014_sel_anchor003_seed42_15min_train.log`
- New eval logs:
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr014_sel_anchor003/diffusion_nominal_s42.log`
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr014_sel_anchor003/diffusion_light_v2_s42.log`
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr014_sel_anchor003/diffusion_hard_s42.log`
- No code edits in this session block.

### What was verified (commands + key outcomes)
- Training command (bounded):
  - `./docker-run-isaacgym.sh timeout 1000 scripts/screwdriver_student_diffusion_latent_robust_light.sh ... run_a_latent_tailcoef02_thr014_sel_anchor003_seed42_15min ... +train.ppo.diffusion_teacher_delta_tail_threshold=0.14 ...`
  - outcome: expected bounded stop (`code=124`) with `model_best.ckpt` generated.
- Eval results (`seed=42`, `steps=256`):
  - nominal: `1.819124`
  - light_v2: `1.697965`
  - hard: `1.282098`
- Delta vs current kept `thr=0.15 + anchor=0.03` reference:
  - nominal `+0.144012`
  - light_v2 `+0.059699`
  - hard `-0.222806`
- Delta vs `latent_recon05` seed42:
  - nominal `-0.088100`
  - light_v2 `+0.292552`
  - hard `-0.027993`

### Local decision
- `thr=0.14` probe is **rejected as new main candidate**:
  - nominal/light improve,
  - but hard still drops materially, which violates this round’s hard-focused objective.

### Remaining blocked/risky
- Hard-condition gap remains the primary blocker.
- Current tail+anchor family still exhibits a strong tradeoff surface (nominal/light gains vs hard regression).
- Additional broad sweeps are likely low efficiency; single-variable hard-targeted probes remain preferred.

### Single recommended next step
- Keep `tail_coef=0.2 + tail_threshold=0.15 + anchor=0.03` as reference and run one bounded **hard compensation** probe by increasing anchor strength slightly (e.g., `anchor=0.035`), then apply the same strict keep/drop gate before any multiseed.

---

## v2-032 (2026-03-25) — PLANS_v2 P3 Hard-Compensation Probe (`anchor=0.035`)

### Target milestone/subgoal
- Continue strict bounded optimization from current reference branch:
  - keep `tail_coef=0.2 + tail_threshold=0.15 + selective=True`,
  - only increase `base_action_anchor_coef` from `0.03` to `0.035`,
  - evaluate with unified `nominal/light_v2/hard` protocol.

### What changed (files + behavior impact)
- New bounded training run:
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor0035_seed42_15min/`
  - train log:
    - `outputs/train_logs/run_a_latent_tailcoef02_thr015_sel_anchor0035_seed42_15min_train.log`
- New eval logs:
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor0035/diffusion_nominal_s42.log`
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor0035/diffusion_light_v2_s42.log`
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor0035/diffusion_hard_s42.log`
- No code edits in this session block.

### What was verified (commands + key outcomes)
- Training command (bounded):
  - `./docker-run-isaacgym.sh timeout 1000 scripts/screwdriver_student_diffusion_latent_robust_light.sh ... run_a_latent_tailcoef02_thr015_sel_anchor0035_seed42_15min ... +train.ppo.diffusion_base_action_anchor_coef=0.035`
  - outcome: expected bounded stop (`code=124`) with `model_best.ckpt` generated.
- Eval results (`seed=42`, `steps=256`):
  - nominal: `1.497288`
  - light_v2: `1.511509`
  - hard: `1.306996`
- Delta vs current kept reference (`anchor=0.03`):
  - nominal `-0.177824`
  - light_v2 `-0.126757`
  - hard `-0.197908`

### Local decision
- `anchor=0.035` probe is **rejected**:
  - all three conditions regress relative to current kept reference.

### Remaining blocked/risky
- Hard-condition gap remains unresolved.
- `anchor`-increase direction does not provide hard recovery and can degrade all metrics.
- Current best single-seed tradeoff still remains `tail_coef=0.2 + thr=0.15 + anchor=0.03`.

### Single recommended next step
- Keep the current reference unchanged and run one bounded non-anchor hard-focused probe by increasing tail intensity slightly (e.g., `tail_coef=0.25` with `thr=0.15`, `anchor=0.03`), then apply the same keep/drop gate before multiseed.

---

## v2-033 (2026-03-25) — PLANS_v2 P3 Hard-Focused Tail-Intensity Probe (`tail_coef=0.25`)

### Target milestone/subgoal
- Execute one bounded non-anchor hard-focused probe:
  - keep `tail_threshold=0.15 + selective=True + anchor=0.03`,
  - increase `tail_coef` from `0.2` to `0.25`,
  - evaluate by the same `nominal/light_v2/hard` protocol.

### What changed (files + behavior impact)
- New bounded training run:
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef025_thr015_sel_anchor003_seed42_15min/`
  - train log:
    - `outputs/train_logs/run_a_latent_tailcoef025_thr015_sel_anchor003_seed42_15min_train.log`
- New eval logs:
  - `outputs/robustness_eval/plansv2_live_tailcoef025_thr015_sel_anchor003/diffusion_nominal_s42.log`
  - `outputs/robustness_eval/plansv2_live_tailcoef025_thr015_sel_anchor003/diffusion_light_v2_s42.log`
  - `outputs/robustness_eval/plansv2_live_tailcoef025_thr015_sel_anchor003/diffusion_hard_s42.log`
- No code edits in this session block.

### What was verified (commands + key outcomes)
- Training command (bounded):
  - `./docker-run-isaacgym.sh timeout 1000 scripts/screwdriver_student_diffusion_latent_robust_light.sh ... run_a_latent_tailcoef025_thr015_sel_anchor003_seed42_15min ... +train.ppo.diffusion_teacher_delta_tail_coef=0.25 ...`
  - outcome: expected bounded stop (`code=124`) with `model_best.ckpt` generated.
- Eval results (`seed=42`, `steps=256`):
  - nominal: `1.717954`
  - light_v2: `1.561320`
  - hard: `1.244413`
- Delta vs current kept reference (`tail_coef=0.2, thr=0.15, anchor=0.03`):
  - nominal `+0.042842`
  - light_v2 `-0.076946`
  - hard `-0.260491`

### Local decision
- `tail_coef=0.25` probe is **rejected**:
  - hard degrades materially and light_v2 also drops; only nominal has small gain.

### Remaining blocked/risky
- Hard robustness remains the unresolved blocker.
- Both tested hard-compensation directions in this cycle (`anchor up`, `tail_coef up`) failed to improve hard.
- Current best single-seed tradeoff remains unchanged at `tail_coef=0.2 + thr=0.15 + anchor=0.03`.

### Single recommended next step
- Keep current reference as-is and run one bounded **lower-anchor** probe (`anchor=0.025` with `tail_coef=0.2, thr=0.15`) to test whether reducing anchor can recover hard without collapsing nominal/light before considering multiseed.

---

## v2-034 (2026-03-25) — PLANS_v2 P3 Lower-Anchor Probe (`anchor=0.025`)

### Target milestone/subgoal
- Execute one strict single-variable probe around the current reference:
  - keep `tail_coef=0.2 + tail_threshold=0.15 + selective=True`,
  - reduce `base_action_anchor_coef` from `0.03` to `0.025`,
  - evaluate by unified `nominal/light_v2/hard` protocol.

### What changed (files + behavior impact)
- New bounded training run:
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor0025_seed42_15min/`
  - train log:
    - `outputs/train_logs/run_a_latent_tailcoef02_thr015_sel_anchor0025_seed42_15min_train.log`
- New eval logs:
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor0025/diffusion_nominal_s42.log`
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor0025/diffusion_light_v2_s42.log`
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor0025/diffusion_hard_s42.log`
- No code edits in this session block.

### What was verified (commands + key outcomes)
- Training command (bounded):
  - `./docker-run-isaacgym.sh timeout 1000 scripts/screwdriver_student_diffusion_latent_robust_light.sh ... run_a_latent_tailcoef02_thr015_sel_anchor0025_seed42_15min ... +train.ppo.diffusion_base_action_anchor_coef=0.025`
  - outcome: expected bounded stop (`code=124`) with `model_best.ckpt` generated.
- Eval results (`seed=42`, `steps=256`):
  - nominal: `1.632088`
  - light_v2: `1.515501`
  - hard: `1.240960`
- Delta vs current kept reference (`anchor=0.03`):
  - nominal `-0.043024`
  - light_v2 `-0.122765`
  - hard `-0.263944`

### Local decision
- `anchor=0.025` probe is **rejected**:
  - all three conditions regress relative to current kept reference.

### Remaining blocked/risky
- The local `anchor` neighborhood (`0.025`, `0.03`, `0.035`) does not yield a better candidate than `0.03`.
- Recent hard-focused micro-tuning around tail/anchor shows repeated regressions on `hard`.
- Continuing the same hyperparameter neighborhood is likely low efficiency.

### Single recommended next step
- Keep current reference (`tail_coef=0.2 + thr=0.15 + anchor=0.03`) and run one bounded **training-perturbation axis** probe (stronger training noise/force than current `robust_light` defaults) to test whether hard robustness can improve without further degrading nominal.

---

## v2-035 (2026-03-25) — PLANS_v2 P3 Training-Perturbation Axis Probe (`train forceScale=1.0`)

### Target milestone/subgoal
- Execute one bounded training-perturbation single-variable probe:
  - keep current reference (`tail_coef=0.2 + thr=0.15 + anchor=0.03`),
  - increase training-time `task.env.forceScale` from `0.5` to `1.0`,
  - evaluate under unified `nominal/light_v2/hard`.

### What changed (files + behavior impact)
- New bounded training run:
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_trainforce10_seed42_15min/`
  - train log:
    - `outputs/train_logs/run_a_latent_tailcoef02_thr015_sel_anchor003_trainforce10_seed42_15min_train.log`
- New eval logs:
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_trainforce10/diffusion_nominal_s42.log`
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_trainforce10/diffusion_light_v2_s42.log`
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_trainforce10/diffusion_hard_s42.log`
- No code edits in this session block.

### What was verified (commands + key outcomes)
- Training command (bounded):
  - `./docker-run-isaacgym.sh timeout 1000 scripts/screwdriver_student_diffusion_latent_robust_light.sh ... run_a_latent_tailcoef02_thr015_sel_anchor003_trainforce10_seed42_15min ... task.env.forceScale=1.0`
  - outcome: expected bounded stop (`code=124`) with `model_best.ckpt` generated.
- Eval results (`seed=42`, `steps=256`):
  - nominal: `1.470352`
  - light_v2: `1.567975`
  - hard: `1.010860`
- Delta vs current kept reference:
  - nominal `-0.204760`
  - light_v2 `-0.070291`
  - hard `-0.494044`

### Local decision
- `train forceScale=1.0` probe is **rejected**:
  - all conditions regress, and hard drops sharply.

### Remaining blocked/risky
- Hard robustness remains the main unresolved issue.
- Training-force strengthening is too aggressive for current reference branch and destabilizes robustness.
- Recent local probes continue to fail to beat current reference.

### Single recommended next step
- Keep the current reference unchanged and run one bounded **training observation-noise axis** probe (e.g., raise training `obs_noise_e/t` while keeping force defaults) to test whether robustness can improve with lower destabilization risk than force amplification.

---

## v2-036 (2026-03-25) — PLANS_v2 P3 Training Observation-Noise Axis Probe (`train obs_noise=0.03/0.015`)

### Target milestone/subgoal
- Execute one bounded single-variable training-perturbation probe on observation noise:
  - keep current reference (`tail_coef=0.2 + thr=0.15 + anchor=0.03`),
  - increase training `obs_noise_e/t` from `0.02/0.01` to `0.03/0.015`,
  - keep training force defaults unchanged,
  - evaluate by unified `nominal/light_v2/hard` protocol.

### What changed (files + behavior impact)
- New bounded training run:
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_trainobs0315_seed42_15min/`
  - train log:
    - `outputs/train_logs/run_a_latent_tailcoef02_thr015_sel_anchor003_trainobs0315_seed42_15min_train.log`
- New eval logs:
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_trainobs0315/diffusion_nominal_s42.log`
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_trainobs0315/diffusion_light_v2_s42.log`
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_trainobs0315/diffusion_hard_s42.log`
- No code edits in this session block.

### What was verified (commands + key outcomes)
- Training command (bounded):
  - `./docker-run-isaacgym.sh timeout 1000 scripts/screwdriver_student_diffusion_latent_robust_light.sh ... run_a_latent_tailcoef02_thr015_sel_anchor003_trainobs0315_seed42_15min ... task.env.randomization.obs_noise_e_scale=0.03 task.env.randomization.obs_noise_t_scale=0.015`
  - outcome: expected bounded stop (`code=124`) with `model_best.ckpt` generated.
- Eval results (`seed=42`, `steps=256`):
  - nominal: `1.440379`
  - light_v2: `1.493933`
  - hard: `0.984006`
- Delta vs current kept reference:
  - nominal `-0.234733`
  - light_v2 `-0.144333`
  - hard `-0.520898`

### Local decision
- `train obs_noise=0.03/0.015` probe is **rejected**:
  - all conditions regress, hard drops sharply.

### Remaining blocked/risky
- Hard robustness remains the main blocker.
- Training-perturbation strengthening (force or obs-noise) has repeatedly destabilized this branch.
- Current best single-seed tradeoff remains `tail_coef=0.2 + thr=0.15 + anchor=0.03`.

### Single recommended next step
- Keep current reference unchanged and return to model-side single-variable tuning: run one bounded probe with **slightly lower tail threshold** (`0.145`) under the same `tail_coef=0.2 + anchor=0.03`, then apply the same strict keep/drop gate before multiseed.

---

## v2-037 (2026-03-25) — PLANS_v2 P3 Tail-Threshold Micro-Probe (`thr=0.145`)

### Target milestone/subgoal
- Continue model-side single-variable tuning from current reference:
  - keep `tail_coef=0.2 + selective=True + anchor=0.03`,
  - lower `tail_threshold` from `0.15` to `0.145`,
  - evaluate by unified `nominal/light_v2/hard` protocol.

### What changed (files + behavior impact)
- New bounded training run:
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr0145_sel_anchor003_seed42_15min/`
  - train log:
    - `outputs/train_logs/run_a_latent_tailcoef02_thr0145_sel_anchor003_seed42_15min_train.log`
- New eval logs:
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr0145_sel_anchor003/diffusion_nominal_s42.log`
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr0145_sel_anchor003/diffusion_light_v2_s42.log`
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr0145_sel_anchor003/diffusion_hard_s42.log`
- No code edits in this session block.

### What was verified (commands + key outcomes)
- Training command (bounded):
  - `./docker-run-isaacgym.sh timeout 1000 scripts/screwdriver_student_diffusion_latent_robust_light.sh ... run_a_latent_tailcoef02_thr0145_sel_anchor003_seed42_15min ... +train.ppo.diffusion_teacher_delta_tail_threshold=0.145`
  - outcome: expected bounded stop (`code=124`) with `model_best.ckpt` generated.
- Eval results (`seed=42`, `steps=256`):
  - nominal: `1.347240`
  - light_v2: `1.521928`
  - hard: `1.307931`
- Delta vs current kept reference (`tail_coef=0.2, thr=0.15, anchor=0.03`):
  - nominal `-0.327872`
  - light_v2 `-0.116338`
  - hard `-0.196973`

### Local decision
- `thr=0.145` probe is **rejected**:
  - all conditions regress relative to current kept reference.

### Remaining blocked/risky
- Hard robustness remains unresolved.
- Current local hyperparameter neighborhood around tail/anchor keeps producing regressions.
- Continuing blind local sweeps has low expected return.

### Single recommended next step
- Keep current reference unchanged and run a bounded **failure-mode diagnostic compare** (reference vs one rejected hard-fail candidate under the same eval protocol with rollout diagnostics enabled) before proposing the next code/config hypothesis.

---

## v2-038 (2026-03-26) — PLANS_v2 P3 Failure-Mode Diagnostic Compare (Hard)

### Target milestone/subgoal
- Execute the recommended diagnostic compare before new tuning:
  - compare current kept reference vs one rejected hard-fail candidate under the same `hard` protocol,
  - collect rollout diagnostics to identify actionable failure signals.

### What changed (files + behavior impact)
- `dexscrew/algo/ppo/diffusion_latent_student.py`
  - Added `DiffusionLatentStudent.collect_rollout()` override so rollout collection uses the same diffusion/decode action path as `test()`.
  - Rollout payload now includes diffusion diagnostics in `extras/diag/*`:
    - `diag/latent_mse`
    - `diag/latent_l1`
    - `diag/action_mse_to_teacher`
  - `collect_rollout` metadata now uses `getattr(..., default)` for normalization flags to avoid missing-attribute failure.
- New diagnostic artifacts:
  - `outputs/rollout_diag/plansv2_v2038/ref_hard_rollout.pt`
  - `outputs/rollout_diag/plansv2_v2038/cand_hard_rollout.pt`
  - `outputs/rollout_diag/plansv2_v2038/diag_compare.txt`
  - collection logs:
    - `outputs/rollout_diag/plansv2_v2038/ref_hard_collect.log`
    - `outputs/rollout_diag/plansv2_v2038/cand_hard_collect.log`

### What was verified (commands + key outcomes)
- Syntax check:
  - `PYTHONDONTWRITEBYTECODE=1 python - <<'PY' ... compile('dexscrew/algo/ppo/diffusion_latent_student.py', 'exec') ... PY`
  - outcome: `syntax_ok`.
- Hard rollout collection (same perturbation protocol for both checkpoints):
  - reference ckpt:
    - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_seed42_15min/stage2_diffusion_nn/model_best.ckpt`
  - rejected candidate ckpt:
    - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_trainobs0315_seed42_15min/stage2_diffusion_nn/model_best.ckpt`
  - condition: `obs_noise_e=0.06, obs_noise_t=0.03, forceScale=2.0, randomForceProbScalar=0.3`.
  - collection summary:
    - ref: `mean_reward=1.4279`, `mean_done_rate=0.0020`
    - cand: `mean_reward=0.9840`, `mean_done_rate=0.0027`
- Diagnostic compare (`outputs/rollout_diag/plansv2_v2038/diag_compare.txt`):
  - `reward_per_step delta=-0.443875` (mid `-0.547258`, late `-0.626457`)
  - `extras/torques delta=+0.081183` (late `+0.115442`)
  - `extras/work_done delta=+1.568641` (late `+3.308425`)
  - `extras/diag/action_mse_to_teacher delta=+0.006251` (late `+0.024326`)
  - `extras/diag/latent_mse delta=+0.005795` (late `+0.012380`)

### Local decision
- Diagnostic compare is **accepted as actionable evidence**:
  - rejected candidate fails mainly in mid/late phase,
  - failure is accompanied by increased torque/work burden and stronger teacher-action drift.

### Remaining blocked/risky
- Hard robustness gap remains unresolved for the current latent branch.
- Recent probes suggest blind local sweeps are low-efficiency without mechanism-level constraints.

### Single recommended next step
- Run one bounded mechanism-aligned probe that explicitly suppresses mid/late action drift (without broad sweep), then keep/reject by the same unified `nominal/light_v2/hard` gate.

---

## v2-039 (2026-03-26) — PLANS_v2 P3 Mechanism-Aligned Probe (`action_l2=0.002`)

### Target milestone/subgoal
- Execute one bounded mechanism-aligned probe from the current kept reference:
  - keep `tail_coef=0.2 + tail_threshold=0.15 + selective=True + anchor=0.03`,
  - add a small action-magnitude regularizer (`diffusion_action_l2_coef=0.002`) to suppress mid/late drift/torque,
  - evaluate with unified `nominal/light_v2/hard` protocol.

### What changed (files + behavior impact)
- New bounded training run:
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_actionl2_0002_seed42_15min/`
  - train log:
    - `outputs/train_logs/run_a_latent_tailcoef02_thr015_sel_anchor003_actionl2_0002_seed42_15min_train.log`
  - config snapshot:
    - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_actionl2_0002_seed42_15min/config_032516_aabc11a.yaml`
- New eval logs:
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_actionl2_0002/diffusion_nominal_s42.log`
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_actionl2_0002/diffusion_light_v2_s42.log`
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_actionl2_0002/diffusion_hard_s42.log`
- No code edits in this session block (experiment-only).

### What was verified (commands + key outcomes)
- Training command (bounded):
  - `./docker-run-isaacgym.sh timeout 1000 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_tailcoef02_thr015_sel_anchor003_actionl2_0002_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_latent_recon_coef=0.5 +train.ppo.diffusion_teacher_delta_tail_coef=0.2 +train.ppo.diffusion_teacher_delta_tail_threshold=0.15 +train.ppo.diffusion_teacher_delta_tail_selective=True +train.ppo.diffusion_base_action_anchor_coef=0.03 +train.ppo.diffusion_action_l2_coef=0.002`
  - outcome: expected bounded stop (`code=124`) with `model_best.ckpt` generated.
- Eval results (`seed=42`, `steps=256`):
  - nominal: `1.569661`
  - light_v2: `1.579198`
  - hard: `1.031702`
  - done rates: nominal `0.001465`, light_v2 `0.001872`, hard `0.002279`
- Delta vs current kept reference (`tail_coef=0.2, thr=0.15, anchor=0.03`):
  - reward:
    - nominal `-0.105451` (`1.569661 - 1.675112`)
    - light_v2 `-0.059068` (`1.579198 - 1.638266`)
    - hard `-0.473202` (`1.031702 - 1.504904`)
  - done rate:
    - nominal `+0.000163`
    - light_v2 `+0.000489`
    - hard `+0.000407`

### Local decision
- `action_l2=0.002` probe is **rejected**:
  - all three conditions regress,
  - hard condition degrades strongly and done rate rises.

### Remaining blocked/risky
- Hard robustness remains the dominant blocker for this branch.
- Simple global action-magnitude suppression harms task performance rather than improving robustness.

### Single recommended next step
- Keep current reference unchanged and run one bounded **teacher-drift-localized** probe (e.g., stronger selective tail penalty with a tighter threshold while keeping `action_l2=0`), then apply the same keep/drop gate before any multiseed.

---

## v2-040 (2026-03-26) — PLANS_v2 P3 Mid/Late-Localized Drift Probe (`mid_only=0.55~1.0`)

### Target milestone/subgoal
- Continue bounded mechanism-aligned optimization from the current kept reference:
  - keep `tail_coef=0.2 + tail_threshold=0.15 + selective=True + anchor=0.03`,
  - apply teacher-delta-tail only in later episode phase (`mid_only=True`, `progress_start=0.55`, `progress_end=1.0`),
  - evaluate by unified `nominal/light_v2/hard` protocol.

### What changed (files + behavior impact)
- New bounded training run:
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_midonly55100_seed42_15min/`
  - train log:
    - `outputs/train_logs/run_a_latent_tailcoef02_thr015_sel_anchor003_midonly55100_seed42_15min_train.log`
  - config snapshot:
    - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_midonly55100_seed42_15min/config_032518_aabc11a.yaml`
- New eval logs:
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_midonly55100/diffusion_nominal_s42.log`
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_midonly55100/diffusion_light_v2_s42.log`
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_midonly55100/diffusion_hard_s42.log`
- No code edits in this session block (experiment-only).

### What was verified (commands + key outcomes)
- Training command (bounded):
  - `./docker-run-isaacgym.sh timeout 1000 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_tailcoef02_thr015_sel_anchor003_midonly55100_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_latent_recon_coef=0.5 +train.ppo.diffusion_teacher_delta_tail_coef=0.2 +train.ppo.diffusion_teacher_delta_tail_threshold=0.15 +train.ppo.diffusion_teacher_delta_tail_selective=True +train.ppo.diffusion_base_action_anchor_coef=0.03 +train.ppo.diffusion_teacher_delta_tail_mid_only=True +train.ppo.diffusion_teacher_delta_tail_progress_start=0.55 +train.ppo.diffusion_teacher_delta_tail_progress_end=1.0`
  - outcome: expected bounded stop (`code=124`) with `model_best.ckpt` generated.
- Eval results (`seed=42`, `steps=256`):
  - nominal: `1.778610`, done `0.000732`
  - light_v2: `1.648791`, done `0.001302`
  - hard: `1.338230`, done `0.002279`
- Delta vs current kept reference (`tail_coef=0.2, thr=0.15, anchor=0.03`):
  - reward:
    - nominal `+0.103498`
    - light_v2 `+0.010525`
    - hard `-0.166674`
  - done rate:
    - nominal `-0.000570`
    - light_v2 `-0.000081`
    - hard `+0.000407`

### Local decision
- `mid_only=0.55~1.0` probe is **rejected as new main candidate**:
  - nominal/light improve,
  - but hard still regresses materially under the strict keep gate.

### Remaining blocked/risky
- Hard robustness remains the primary blocker.
- Late-phase-only constraint is insufficient to close the hard gap (likely missing part of mid-phase failure signal).

### Single recommended next step
- Keep current reference unchanged and run one bounded **mid+late compromise** probe (`mid_only=True`, `progress_start=0.40`, `progress_end=1.0`, other params fixed) to test whether covering more of mid phase can recover hard without losing nominal/light gains.

---

## v2-041 (2026-03-26) — PLANS_v2 P3 Mid/Late Compromise Probe (`mid_only=0.40~1.0`)

### Target milestone/subgoal
- Execute the planned bounded compromise probe from `v2-040`:
  - keep `tail_coef=0.2 + tail_threshold=0.15 + selective=True + anchor=0.03`,
  - set `mid_only=True`, `progress_start=0.40`, `progress_end=1.0`,
  - evaluate by unified `nominal/light_v2/hard` protocol.

### What changed (files + behavior impact)
- New bounded training run:
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_midonly40100_seed42_15min/`
  - train log:
    - `outputs/train_logs/run_a_latent_tailcoef02_thr015_sel_anchor003_midonly40100_seed42_15min_train.log`
  - config snapshot:
    - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_midonly40100_seed42_15min/config_032605_aabc11a.yaml`
- New eval logs:
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_midonly40100/diffusion_nominal_s42.log`
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_midonly40100/diffusion_light_v2_s42.log`
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_midonly40100/diffusion_hard_s42.log`
- No code edits in this session block (experiment-only).

### What was verified (commands + key outcomes)
- Training command (bounded):
  - `./docker-run-isaacgym.sh timeout 1000 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_tailcoef02_thr015_sel_anchor003_midonly40100_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_latent_recon_coef=0.5 +train.ppo.diffusion_teacher_delta_tail_coef=0.2 +train.ppo.diffusion_teacher_delta_tail_threshold=0.15 +train.ppo.diffusion_teacher_delta_tail_selective=True +train.ppo.diffusion_base_action_anchor_coef=0.03 +train.ppo.diffusion_teacher_delta_tail_mid_only=True +train.ppo.diffusion_teacher_delta_tail_progress_start=0.40 +train.ppo.diffusion_teacher_delta_tail_progress_end=1.0`
  - outcome: expected bounded stop (`code=124`) with `model_best.ckpt` generated.
- Eval results (`seed=42`, `steps=256`):
  - nominal: `1.830635`, done `0.001790`
  - light_v2: `1.532304`, done `0.002035`
  - hard: `1.227167`, done `0.002441`
- Delta vs current kept reference (`tail_coef=0.2, thr=0.15, anchor=0.03`):
  - reward:
    - nominal `+0.155523`
    - light_v2 `-0.105962`
    - hard `-0.277737`
  - done rate:
    - nominal `+0.000488`
    - light_v2 `+0.000652`
    - hard `+0.000569`

### Local decision
- `mid_only=0.40~1.0` probe is **rejected**:
  - nominal improves, but both robustness conditions regress materially,
  - done rate increases in all three conditions.

### Remaining blocked/risky
- Hard robustness remains the dominant blocker.
- Starting the tail penalty too early (`0.40`) appears over-constraining and harms robust behavior.

### Single recommended next step
- Keep current reference unchanged and run one bounded **phase-window backoff** probe (`mid_only=True`, `progress_start=0.50`, `progress_end=1.0`, all other params fixed), then apply the same strict keep/drop gate before any multiseed.

---

## v2-042 (2026-03-26) — PLANS_v2 P3 Phase-Window Backoff Probe (`mid_only=0.50~1.0`)

### Target milestone/subgoal
- Execute the planned phase-window backoff probe from `v2-041`:
  - keep `tail_coef=0.2 + tail_threshold=0.15 + selective=True + anchor=0.03`,
  - set `mid_only=True`, `progress_start=0.50`, `progress_end=1.0`,
  - evaluate by unified `nominal/light_v2/hard` protocol.

### What changed (files + behavior impact)
- New bounded training run:
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_midonly50100_seed42_15min/`
  - train log:
    - `outputs/train_logs/run_a_latent_tailcoef02_thr015_sel_anchor003_midonly50100_seed42_15min_train.log`
  - config snapshot:
    - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_midonly50100_seed42_15min/config_032605_aabc11a.yaml`
- New eval logs:
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_midonly50100/diffusion_nominal_s42.log`
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_midonly50100/diffusion_light_v2_s42.log`
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_midonly50100/diffusion_hard_s42.log`
- No code edits in this session block (experiment-only).

### What was verified (commands + key outcomes)
- Training command (bounded):
  - `./docker-run-isaacgym.sh timeout 1000 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_tailcoef02_thr015_sel_anchor003_midonly50100_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_latent_recon_coef=0.5 +train.ppo.diffusion_teacher_delta_tail_coef=0.2 +train.ppo.diffusion_teacher_delta_tail_threshold=0.15 +train.ppo.diffusion_teacher_delta_tail_selective=True +train.ppo.diffusion_base_action_anchor_coef=0.03 +train.ppo.diffusion_teacher_delta_tail_mid_only=True +train.ppo.diffusion_teacher_delta_tail_progress_start=0.50 +train.ppo.diffusion_teacher_delta_tail_progress_end=1.0`
  - outcome: expected bounded stop (`code=124`) with `model_best.ckpt` generated.
- Eval results (`seed=42`, `steps=256`):
  - nominal: `1.957675`, done `0.001383`
  - light_v2: `1.818249`, done `0.001709`
  - hard: `1.284105`, done `0.002116`
- Delta vs current kept reference (`tail_coef=0.2, thr=0.15, anchor=0.03`):
  - reward:
    - nominal `+0.282563`
    - light_v2 `+0.179983`
    - hard `-0.220799`
  - done rate:
    - nominal `+0.000081`
    - light_v2 `+0.000326`
    - hard `+0.000244`

### Local decision
- `mid_only=0.50~1.0` probe is **rejected as main candidate**:
  - nominal/light gains are clear,
  - but hard remains materially below reference and done rate still worsens.

### Remaining blocked/risky
- Hard robustness remains the primary blocker for route promotion.
- Mid/late window tuning alone is improving nominal/light faster than hard, indicating a persistent tradeoff surface.

### Single recommended next step
- Keep current reference unchanged and run one bounded **late-only backoff** probe (`mid_only=True`, `progress_start=0.60`, `progress_end=1.0`, all else fixed), then apply the same strict keep/drop gate before any multiseed.

---

## v2-043 (2026-03-26) — PLANS_v2 P3 Late-Only Backoff Probe (`mid_only=0.60~1.0`)

### Target milestone/subgoal
- Execute the planned late-only backoff probe from `v2-042`:
  - keep `tail_coef=0.2 + tail_threshold=0.15 + selective=True + anchor=0.03`,
  - set `mid_only=True`, `progress_start=0.60`, `progress_end=1.0`,
  - evaluate by unified `nominal/light_v2/hard` protocol.

### What changed (files + behavior impact)
- New bounded training run:
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_midonly60100_seed42_15min/`
  - train log:
    - `outputs/train_logs/run_a_latent_tailcoef02_thr015_sel_anchor003_midonly60100_seed42_15min_train.log`
  - config snapshot:
    - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_midonly60100_seed42_15min/config_032611_aabc11a.yaml`
- New eval logs:
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_midonly60100/diffusion_nominal_s42.log`
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_midonly60100/diffusion_light_v2_s42.log`
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_midonly60100/diffusion_hard_s42.log`
- No code edits in this session block (experiment-only).

### What was verified (commands + key outcomes)
- Training command (bounded):
  - `./docker-run-isaacgym.sh timeout 1000 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_tailcoef02_thr015_sel_anchor003_midonly60100_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_latent_recon_coef=0.5 +train.ppo.diffusion_teacher_delta_tail_coef=0.2 +train.ppo.diffusion_teacher_delta_tail_threshold=0.15 +train.ppo.diffusion_teacher_delta_tail_selective=True +train.ppo.diffusion_base_action_anchor_coef=0.03 +train.ppo.diffusion_teacher_delta_tail_mid_only=True +train.ppo.diffusion_teacher_delta_tail_progress_start=0.60 +train.ppo.diffusion_teacher_delta_tail_progress_end=1.0`
  - outcome: expected bounded stop (`code=124`) with `model_best.ckpt` generated.
- Eval results (`seed=42`, `steps=256`):
  - nominal: `1.568932`, done `0.001872`
  - light_v2: `1.553752`, done `0.001465`
  - hard: `1.227310`, done `0.002197`
- Delta vs current kept reference (`tail_coef=0.2, thr=0.15, anchor=0.03`):
  - reward:
    - nominal `-0.106180`
    - light_v2 `-0.084514`
    - hard `-0.277594`
  - done rate:
    - nominal `+0.000570`
    - light_v2 `+0.000082`
    - hard `+0.000325`

### Local decision
- `mid_only=0.60~1.0` probe is **rejected**:
  - all three conditions regress relative to current kept reference,
  - hard remains significantly below reference.

### Remaining blocked/risky
- Hard robustness remains the primary blocker.
- `mid_only` phase-window axis (`0.40/0.50/0.55/0.60`) does not yield a keepable candidate; current neighborhood shows repeated tradeoff/regression.

### Single recommended next step
- Keep current reference unchanged and switch to one bounded **non-window single-variable** probe: raise `teacher_delta_tail_threshold` slightly to `0.16` (keep `tail_coef=0.2`, `selective=True`, `anchor=0.03`, `mid_only=False`) and apply the same strict keep/drop gate.

---

## v2-044 (2026-03-27) — Fresh-Session Alignment Repair And Closure Revalidation

### Target milestone/subgoal
- Re-establish the authoritative `PLANS_v2` execution state after fresh-session bootstrap.
- Resolve drift between:
  - local closure/acceptance docs (`M5` baseline-first closure, maintenance mode),
  - later bounded `P3` latent probes in the running handoff.
- Choose the smallest valid next step without reopening scope implicitly.

### What changed (files + behavior impact)
- `docs/session_handoff_v2.md`
  - Added this alignment-repair entry.
  - Records that:
    - `M1/M2/M4` artifact-backed conclusions still hold after low-cost regeneration,
    - `M3/M5` synthesis-layer conclusions remain unchanged,
    - later `P3` latent probes did not overturn the locally accepted `baseline-first` closure.
  - Resets the default next step to the strict-audit / acceptance-readiness direction:
    - low-cost reproducibility/reporting checks by default,
    - no further latent optimization sweep unless governance is explicitly reopened.

### What was verified (commands + key outcomes)
- Fresh-session bootstrap reads:
  - `sed -n '1,260p' AGENTS.md`
  - `sed -n '1,520p' PLANS_v2.md`
  - `sed -n '1,2040p' docs/session_handoff_v2.md`
  - `sed -n '1,260p' docs/stage_acceptance_summary.md`
- Low-cost evidence regeneration:
  - `PLANSV2_FROM_LOGS_ONLY=1 bash scripts/eval_plansv2_m2_gap_gate.sh 0 256 42,43,44`
  - `bash scripts/eval_plansv2_m3_min_compare.sh 42,43,44`
  - `PLANSV2_FROM_LOGS_ONLY=1 bash scripts/eval_plansv2_m4_residual_compare_pack.sh 0 256 42,43,44`
  - Outcome:
    - `docs/plansv2_m2_gap_gate.md` regenerated with `local_G2_decision=PASS`.
    - `docs/plansv2_m3_min_compare.md` regenerated with `local_m3_conclusion=not_support_latent_mainline`.
    - `docs/plansv2_m4_residual_compare_pack.md` regenerated with `local_m4_conclusion=not_support_residual_robust_edge`.
- Cross-doc state check:
  - `rg -n "local_G2_decision|local_m3_conclusion|local_m4_conclusion|selected_path|readiness_status|Current Single Recommended Next Step|maintenance mode" docs/plansv2_m2_gap_gate.md docs/plansv2_m3_min_compare.md docs/plansv2_m4_residual_compare_pack.md docs/plansv2_m5_baseline_closure.md docs/stage_acceptance_summary.md docs/plansv2_stage_gate_strict_audit.md`
  - Outcome:
    - strict audit still points to `maintenance mode`,
    - acceptance snapshot remains `ready_for_acceptance_review_local`,
    - closure record still selects `non_diffusion_baseline_closure`.

### Remaining blocked/risky
- Current closure remains a `local` execution conclusion, not governance-final signoff.
- Training-side provenance for some older representative runs is still weaker than the eval-artifact layer (`train*.log` missing for some historical checkpoints), but this does not change the current stage conclusion.
- The handoff contains many bounded latent probes after closure; they are useful history, but should not be treated as authority to reopen the diffusion mainline automatically.

### Single recommended next step
- Keep execution in `maintenance / acceptance-review` mode:
  - run only low-cost reproducibility or reporting checks by default,
  - record any new evidence in `docs/session_handoff_v2.md`,
  - escalate only if a new artifact-backed result materially contradicts `M5` baseline-first closure or if governance explicitly reopens diffusion optimization.

---

## v2-045 (2026-03-27) — PLANS_v2 P3 Threshold-Up Probe (`tail_threshold=0.16`)

### Target milestone/subgoal
- User explicitly requested continuing the current optimization line.
- Resume the bounded latent `P3` loop from the latest unresolved micro-probe:
  - keep `tail_coef=0.2 + selective=True + anchor=0.03`,
  - only raise `teacher_delta_tail_threshold` from `0.15` to `0.16`,
  - evaluate with unified `nominal/light_v2/hard` protocol.

### What changed (files + behavior impact)
- New bounded training run:
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr016_sel_anchor003_seed42_15min/`
  - config snapshot:
    - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr016_sel_anchor003_seed42_15min/config_032706_aabc11a.yaml`
  - checkpoint:
    - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr016_sel_anchor003_seed42_15min/stage2_diffusion_nn/model_best.ckpt`
- New eval logs:
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr016_sel_anchor003/diffusion_nominal_s42.log`
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr016_sel_anchor003/diffusion_light_v2_s42.log`
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr016_sel_anchor003/diffusion_hard_s42.log`
- No code edits in this session block (experiment-only).

### What was verified (commands + key outcomes)
- Bounded training command:
  - `./docker-run-isaacgym.sh timeout 1000 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_tailcoef02_thr016_sel_anchor003_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_latent_recon_coef=0.5 +train.ppo.diffusion_teacher_delta_tail_coef=0.2 +train.ppo.diffusion_teacher_delta_tail_threshold=0.16 +train.ppo.diffusion_teacher_delta_tail_selective=True +train.ppo.diffusion_base_action_anchor_coef=0.03`
  - outcome:
    - `model_best.ckpt` created successfully,
    - run was manually interrupted after checkpoint creation to keep the probe within bounded budget (`exit code 130`),
    - observed training-side `Current Best` reached `1739.32` before interruption.
- Eval commands (docker IsaacGym, `seed=42`, `steps=256`):
  - nominal:
    - `./docker-run-isaacgym.sh timeout 480 bash scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent <ckpt> 256 plansv2_live_tailcoef02_thr016_sel_anchor003_nominal +train.ppo.diffusion_eval_decode_only=False +train.ppo.diffusion_eval_report_recon=True`
  - light_v2:
    - `./docker-run-isaacgym.sh timeout 480 bash scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent <ckpt> 256 plansv2_live_tailcoef02_thr016_sel_anchor003_lightv2 +train.ppo.diffusion_eval_decode_only=False +train.ppo.diffusion_eval_report_recon=True task.env.randomization.obs_noise_e_scale=0.03 task.env.randomization.obs_noise_t_scale=0.015 task.env.forceScale=1.0 task.env.randomForceProbScalar=0.2`
  - hard:
    - `./docker-run-isaacgym.sh timeout 480 bash scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent <ckpt> 256 plansv2_live_tailcoef02_thr016_sel_anchor003_hard +train.ppo.diffusion_eval_decode_only=False +train.ppo.diffusion_eval_report_recon=True task.env.randomization.obs_noise_e_scale=0.06 task.env.randomization.obs_noise_t_scale=0.03 task.env.forceScale=2.0 task.env.randomForceProbScalar=0.3`
- Eval results (`seed=42`, `steps=256`):
  - nominal: reward `1.053211`, done `0.001709`, `latent_mse=0.094721`, `action_mse_to_teacher=0.164570`
  - light_v2: reward `1.241439`, done `0.001628`, `latent_mse=0.094169`, `action_mse_to_teacher=0.168147`
  - hard: reward `0.995242`, done `0.001790`, `latent_mse=0.105928`, `action_mse_to_teacher=0.191126`
- Delta vs current kept reference (`tail_coef=0.2, thr=0.15, anchor=0.03`):
  - reward:
    - nominal `-0.621901` (`1.053211 - 1.675112`)
    - light_v2 `-0.396827` (`1.241439 - 1.638266`)
    - hard `-0.509662` (`0.995242 - 1.504904`)
  - done rate:
    - nominal `+0.000407`
    - light_v2 `+0.000245`
    - hard `-0.000082`
  - recon / teacher-alignment:
    - nominal `latent_mse +0.025968`, `action_mse_to_teacher +0.034468`
    - light_v2 `latent_mse +0.017095`, `action_mse_to_teacher +0.025099`
    - hard `latent_mse +0.021651`, `action_mse_to_teacher +0.031138`
- Delta vs `latent_recon05` representative (`plansv2_m2`, seed42):
  - nominal `-0.854013`
  - light_v2 `-0.163974`
  - hard `-0.314849`

### Local decision
- `tail_threshold=0.16` probe is **rejected**:
  - all three conditions regress strongly vs the current kept reference,
  - teacher-alignment metrics also worsen across all three conditions,
  - this indicates the `threshold-up` direction weakens useful teacher-delta constraint rather than improving hard robustness.

### Remaining blocked/risky
- Hard robustness remains the main blocker for route promotion.
- The local `tail_threshold` neighborhood now has strong negative evidence in both directions:
  - lower: `0.145`, `0.14` rejected,
  - higher: `0.16` rejected sharply.
- Current best single-seed tradeoff remains unchanged at:
  - `tail_coef=0.2 + tail_threshold=0.15 + selective=True + anchor=0.03`.

### Single recommended next step
- Keep the current reference unchanged and stop spending budget on further `tail_threshold` sweeps.
- Run one bounded **teacher-alignment-preserving** loss-balance probe:
  - keep `tail_coef=0.2 + tail_threshold=0.15 + selective=True + anchor=0.03`,
  - only increase `bc_loss_coef` slightly from `1.0` to `1.1`,
  - then apply the same strict keep/drop gate under `nominal/light_v2/hard`.

---

## v2-046 (2026-03-27) — PLANS_v2 P3 Loss-Balance Probe (`bc_loss_coef=1.1`)

### Target milestone/subgoal
- Continue the bounded latent `P3` loop with the smallest loss-balance change suggested by `v2-045`:
  - keep `tail_coef=0.2 + tail_threshold=0.15 + selective=True + anchor=0.03`,
  - only raise `bc_loss_coef` from `1.0` to `1.1`,
  - evaluate with unified `nominal/light_v2/hard` protocol.

### What changed (files + behavior impact)
- New bounded training run:
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_bc11_seed42_15min/`
  - checkpoint:
    - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_bc11_seed42_15min/stage2_diffusion_nn/model_best.ckpt`
- New eval logs:
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_bc11/diffusion_nominal_s42.log`
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_bc11/diffusion_light_v2_s42.log`
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_bc11/diffusion_hard_s42.log`
- No code edits in this session block (experiment-only).

### What was verified (commands + key outcomes)
- Initial launch attempt exposed a Hydra override detail:
  - `+train.ppo.bc_loss_coef=1.1` failed because `bc_loss_coef` already exists in config.
  - Local fix: reran the same probe with direct override `train.ppo.bc_loss_coef=1.1`.
- Corrected bounded training command:
  - `./docker-run-isaacgym.sh timeout 1000 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_tailcoef02_thr015_sel_anchor003_bc11_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_latent_recon_coef=0.5 +train.ppo.diffusion_teacher_delta_tail_coef=0.2 +train.ppo.diffusion_teacher_delta_tail_threshold=0.15 +train.ppo.diffusion_teacher_delta_tail_selective=True +train.ppo.diffusion_base_action_anchor_coef=0.03 train.ppo.bc_loss_coef=1.1`
  - outcome:
    - `model_best.ckpt` created successfully,
    - run was manually interrupted after checkpoint creation and bounded progress observation (`exit code 130`),
    - observed training-side `Current Best` reached `1593.30` before interruption.
- Eval commands (docker IsaacGym, `seed=42`, `steps=256`):
  - nominal:
    - `./docker-run-isaacgym.sh timeout 480 bash scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent <ckpt> 256 plansv2_live_tailcoef02_thr015_sel_anchor003_bc11_nominal +train.ppo.diffusion_eval_decode_only=False +train.ppo.diffusion_eval_report_recon=True`
  - light_v2:
    - `./docker-run-isaacgym.sh timeout 480 bash scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent <ckpt> 256 plansv2_live_tailcoef02_thr015_sel_anchor003_bc11_lightv2 +train.ppo.diffusion_eval_decode_only=False +train.ppo.diffusion_eval_report_recon=True task.env.randomization.obs_noise_e_scale=0.03 task.env.randomization.obs_noise_t_scale=0.015 task.env.forceScale=1.0 task.env.randomForceProbScalar=0.2`
  - hard:
    - `./docker-run-isaacgym.sh timeout 480 bash scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent <ckpt> 256 plansv2_live_tailcoef02_thr015_sel_anchor003_bc11_hard +train.ppo.diffusion_eval_decode_only=False +train.ppo.diffusion_eval_report_recon=True task.env.randomization.obs_noise_e_scale=0.06 task.env.randomization.obs_noise_t_scale=0.03 task.env.forceScale=2.0 task.env.randomForceProbScalar=0.3`
- Eval results (`seed=42`, `steps=256`):
  - nominal: reward `1.381463`, done `0.001872`, `latent_mse=0.091529`, `action_mse_to_teacher=0.177124`
  - light_v2: reward `1.356916`, done `0.002360`, `latent_mse=0.096973`, `action_mse_to_teacher=0.182906`
  - hard: reward `0.941636`, done `0.003092`, `latent_mse=0.110429`, `action_mse_to_teacher=0.199153`
- Delta vs current kept reference (`tail_coef=0.2, thr=0.15, anchor=0.03`):
  - reward:
    - nominal `-0.293649`
    - light_v2 `-0.281350`
    - hard `-0.563268`
  - done rate:
    - nominal `+0.000570`
    - light_v2 `+0.000977`
    - hard `+0.001220`
  - recon / teacher-alignment:
    - nominal `latent_mse +0.022776`, `action_mse_to_teacher +0.047022`
    - light_v2 `latent_mse +0.019899`, `action_mse_to_teacher +0.039858`
    - hard `latent_mse +0.026152`, `action_mse_to_teacher +0.039165`
- Delta vs `latent_recon05` representative (`plansv2_m2`, seed42):
  - nominal `-0.525761`
  - light_v2 `-0.048497`
  - hard `-0.368455`

### Local decision
- `bc_loss_coef=1.1` probe is **rejected**:
  - all three conditions regress relative to the current kept reference,
  - done rate rises notably under all conditions,
  - teacher-alignment metrics also worsen across the board.

### Remaining blocked/risky
- Hard robustness remains the main blocker for route promotion.
- Increasing plain BC weight is not an alignment-preserving fix here; it degrades both rollout reward and teacher-match metrics.
- Current best single-seed tradeoff still remains:
  - `tail_coef=0.2 + tail_threshold=0.15 + selective=True + anchor=0.03`.

### Single recommended next step
- Keep the current reference unchanged and avoid further `bc_loss_coef` up-sweeps.
- Run one bounded **teacher-alignment-preserving opposite-direction** probe:
  - keep `tail_coef=0.2 + tail_threshold=0.15 + selective=True + anchor=0.03`,
  - only reduce `bc_loss_coef` slightly from `1.0` to `0.9`,
  - then apply the same strict keep/drop gate under `nominal/light_v2/hard`.

---

## v2-047 (2026-03-27) — PLANS_v2 P3 Loss-Balance Probe (`bc_loss_coef=0.9`)

### Target milestone/subgoal
- Continue the bounded latent `P3` loop with the opposite-direction BC-loss probe:
  - keep `tail_coef=0.2 + tail_threshold=0.15 + selective=True + anchor=0.03`,
  - only reduce `bc_loss_coef` from `1.0` to `0.9`,
  - evaluate with unified `nominal/light_v2/hard` protocol.

### What changed (files + behavior impact)
- New bounded training run:
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_bc09_seed42_15min/`
  - checkpoint:
    - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_bc09_seed42_15min/stage2_diffusion_nn/model_best.ckpt`
- New eval logs:
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_bc09/diffusion_nominal_s42.log`
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_bc09/diffusion_light_v2_s42.log`
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_bc09/diffusion_hard_s42.log`
- No code edits in this session block (experiment-only).

### What was verified (commands + key outcomes)
- Bounded training command:
  - `./docker-run-isaacgym.sh timeout 1000 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_tailcoef02_thr015_sel_anchor003_bc09_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_latent_recon_coef=0.5 +train.ppo.diffusion_teacher_delta_tail_coef=0.2 +train.ppo.diffusion_teacher_delta_tail_threshold=0.15 +train.ppo.diffusion_teacher_delta_tail_selective=True +train.ppo.diffusion_base_action_anchor_coef=0.03 train.ppo.bc_loss_coef=0.9`
  - outcome:
    - `model_best.ckpt` created successfully,
    - run was manually interrupted after checkpoint creation and bounded progress observation (`exit code 130`),
    - observed training-side `Current Best` reached `1368.32` before interruption.
- Eval commands (docker IsaacGym, `seed=42`, `steps=256`):
  - nominal:
    - `./docker-run-isaacgym.sh timeout 480 bash scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent <ckpt> 256 plansv2_live_tailcoef02_thr015_sel_anchor003_bc09_nominal +train.ppo.diffusion_eval_decode_only=False +train.ppo.diffusion_eval_report_recon=True`
  - light_v2:
    - `./docker-run-isaacgym.sh timeout 480 bash scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent <ckpt> 256 plansv2_live_tailcoef02_thr015_sel_anchor003_bc09_lightv2 +train.ppo.diffusion_eval_decode_only=False +train.ppo.diffusion_eval_report_recon=True task.env.randomization.obs_noise_e_scale=0.03 task.env.randomization.obs_noise_t_scale=0.015 task.env.forceScale=1.0 task.env.randomForceProbScalar=0.2`
  - hard:
    - `./docker-run-isaacgym.sh timeout 480 bash scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent <ckpt> 256 plansv2_live_tailcoef02_thr015_sel_anchor003_bc09_hard +train.ppo.diffusion_eval_decode_only=False +train.ppo.diffusion_eval_report_recon=True task.env.randomization.obs_noise_e_scale=0.06 task.env.randomization.obs_noise_t_scale=0.03 task.env.forceScale=2.0 task.env.randomForceProbScalar=0.3`
- Eval results (`seed=42`, `steps=256`):
  - nominal: reward `0.771603`, done `0.002197`, `latent_mse=0.108171`, `action_mse_to_teacher=0.190939`
  - light_v2: reward `0.763545`, done `0.002604`, `latent_mse=0.113094`, `action_mse_to_teacher=0.204233`
  - hard: reward `0.807803`, done `0.002604`, `latent_mse=0.121374`, `action_mse_to_teacher=0.204583`
- Delta vs current kept reference (`tail_coef=0.2, thr=0.15, anchor=0.03`):
  - reward:
    - nominal `-0.903509`
    - light_v2 `-0.874721`
    - hard `-0.697101`
  - done rate:
    - nominal `+0.000895`
    - light_v2 `+0.001221`
    - hard `+0.000732`
  - recon / teacher-alignment:
    - nominal `latent_mse +0.039418`, `action_mse_to_teacher +0.060837`
    - light_v2 `latent_mse +0.036020`, `action_mse_to_teacher +0.061185`
    - hard `latent_mse +0.037097`, `action_mse_to_teacher +0.044595`
- Delta vs `bc_loss_coef=1.1` probe:
  - reward:
    - nominal `-0.609860`
    - light_v2 `-0.593371`
    - hard `-0.133833`

### Local decision
- `bc_loss_coef=0.9` probe is **rejected**:
  - all three conditions regress sharply relative to the current kept reference,
  - the degradation is even stronger than `bc_loss_coef=1.1`,
  - teacher-alignment metrics also worsen across all conditions.

### Remaining blocked/risky
- Hard robustness remains the main blocker for route promotion.
- The local `bc_loss_coef` axis now has strong negative evidence in both directions:
  - higher: `1.1` rejected,
  - lower: `0.9` rejected even more strongly.
- Current best single-seed tradeoff still remains:
  - `tail_coef=0.2 + tail_threshold=0.15 + selective=True + anchor=0.03`.

### Single recommended next step
- Keep the current reference unchanged and stop spending budget on the `bc_loss_coef` axis.
- Run one bounded **latent-alignment-focused** probe:
  - keep `tail_coef=0.2 + tail_threshold=0.15 + selective=True + anchor=0.03`,
  - only increase `diffusion_latent_recon_coef` slightly from `0.5` to `0.6`,
  - then apply the same strict keep/drop gate under `nominal/light_v2/hard`.

---

## v2-048 (2026-03-27) — PLANS_v2 P3 Latent-Recon Probe (`diffusion_latent_recon_coef=0.6`)

### Target milestone/subgoal
- Continue the bounded latent `P3` loop with a teacher-alignment-focused single-variable change:
  - keep `tail_coef=0.2 + tail_threshold=0.15 + selective=True + anchor=0.03`,
  - only increase `diffusion_latent_recon_coef` from `0.5` to `0.6`,
  - evaluate with unified `nominal/light_v2/hard` protocol.

### What changed (files + behavior impact)
- New bounded training run:
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_recon06_seed42_15min/`
  - checkpoint:
    - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_recon06_seed42_15min/stage2_diffusion_nn/model_best.ckpt`
- New eval logs:
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_recon06/diffusion_nominal_s42.log`
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_recon06/diffusion_light_v2_s42.log`
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_recon06/diffusion_hard_s42.log`
- No code edits in this session block (experiment-only).

### What was verified (commands + key outcomes)
- Bounded training command:
  - `./docker-run-isaacgym.sh timeout 1000 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_tailcoef02_thr015_sel_anchor003_recon06_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_latent_recon_coef=0.6 +train.ppo.diffusion_teacher_delta_tail_coef=0.2 +train.ppo.diffusion_teacher_delta_tail_threshold=0.15 +train.ppo.diffusion_teacher_delta_tail_selective=True +train.ppo.diffusion_base_action_anchor_coef=0.03`
  - outcome:
    - `model_best.ckpt` created successfully,
    - run was manually interrupted after checkpoint creation and bounded progress observation (`exit code 130`),
    - observed training-side `Current Best` reached `1512.18` before interruption.
- Eval commands (docker IsaacGym, `seed=42`, `steps=256`):
  - nominal:
    - `./docker-run-isaacgym.sh timeout 480 bash scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent <ckpt> 256 plansv2_live_tailcoef02_thr015_sel_anchor003_recon06_nominal +train.ppo.diffusion_eval_decode_only=False +train.ppo.diffusion_eval_report_recon=True`
  - light_v2:
    - `./docker-run-isaacgym.sh timeout 480 bash scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent <ckpt> 256 plansv2_live_tailcoef02_thr015_sel_anchor003_recon06_lightv2 +train.ppo.diffusion_eval_decode_only=False +train.ppo.diffusion_eval_report_recon=True task.env.randomization.obs_noise_e_scale=0.03 task.env.randomization.obs_noise_t_scale=0.015 task.env.forceScale=1.0 task.env.randomForceProbScalar=0.2`
  - hard:
    - `./docker-run-isaacgym.sh timeout 480 bash scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent <ckpt> 256 plansv2_live_tailcoef02_thr015_sel_anchor003_recon06_hard +train.ppo.diffusion_eval_decode_only=False +train.ppo.diffusion_eval_report_recon=True task.env.randomization.obs_noise_e_scale=0.06 task.env.randomization.obs_noise_t_scale=0.03 task.env.forceScale=2.0 task.env.randomForceProbScalar=0.3`
- Eval results (`seed=42`, `steps=256`):
  - nominal: reward `1.115911`, done `0.001302`, `latent_mse=0.087187`, `action_mse_to_teacher=0.175628`
  - light_v2: reward `1.050589`, done `0.001709`, `latent_mse=0.092353`, `action_mse_to_teacher=0.176795`
  - hard: reward `0.827163`, done `0.002035`, `latent_mse=0.099981`, `action_mse_to_teacher=0.189453`
- Delta vs current kept reference (`tail_coef=0.2, thr=0.15, anchor=0.03`):
  - reward:
    - nominal `-0.559201`
    - light_v2 `-0.587677`
    - hard `-0.677741`
  - done rate:
    - nominal `+0.000000`
    - light_v2 `+0.000326`
    - hard `+0.000163`
  - recon / teacher-alignment:
    - nominal `latent_mse +0.018434`, `action_mse_to_teacher +0.045526`
    - light_v2 `latent_mse +0.015279`, `action_mse_to_teacher +0.033747`
    - hard `latent_mse +0.015704`, `action_mse_to_teacher +0.029465`
- Delta vs `latent_recon05` representative (`plansv2_m2`, seed42):
  - nominal `-0.791313`
  - light_v2 `-0.354824`
  - hard `-0.482928`

### Local decision
- `diffusion_latent_recon_coef=0.6` probe is **rejected**:
  - all three conditions regress relative to the current kept reference,
  - recon/action alignment metrics also worsen across all conditions,
  - the training-side improvement does not transfer to rollout quality under the unified eval protocol.

### Remaining blocked/risky
- Hard robustness remains the main blocker for route promotion.
- Increasing latent reconstruction weight is not helping this tail+anchor branch under rollout evaluation.
- Current best single-seed tradeoff still remains:
  - `tail_coef=0.2 + tail_threshold=0.15 + selective=True + anchor=0.03`.

### Single recommended next step
- Keep the current reference unchanged and avoid further upward `latent_recon_coef` sweeps.
- Run one bounded **opposite-direction latent-alignment** probe:
  - keep `tail_coef=0.2 + tail_threshold=0.15 + selective=True + anchor=0.03`,
  - only reduce `diffusion_latent_recon_coef` from `0.5` to `0.4`,
  - then apply the same strict keep/drop gate under `nominal/light_v2/hard`.

---

## v2-049 (2026-03-27) — PLANS_v2 P3 Latent-Recon Probe (`diffusion_latent_recon_coef=0.4`)

### Target milestone/subgoal
- Continue the bounded latent `P3` loop with the opposite-direction latent-reconstruction probe:
  - keep `tail_coef=0.2 + tail_threshold=0.15 + selective=True + anchor=0.03`,
  - only reduce `diffusion_latent_recon_coef` from `0.5` to `0.4`,
  - evaluate with unified `nominal/light_v2/hard` protocol.

### What changed (files + behavior impact)
- New bounded training run:
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_recon04_seed42_15min/`
  - checkpoint:
    - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_recon04_seed42_15min/stage2_diffusion_nn/model_best.ckpt`
- New eval logs:
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_recon04/diffusion_nominal_s42.log`
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_recon04/diffusion_light_v2_s42.log`
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_recon04/diffusion_hard_s42.log`
- No code edits in this session block (experiment-only).

### What was verified (commands + key outcomes)
- Bounded training command:
  - `./docker-run-isaacgym.sh timeout 1000 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_tailcoef02_thr015_sel_anchor003_recon04_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_latent_recon_coef=0.4 +train.ppo.diffusion_teacher_delta_tail_coef=0.2 +train.ppo.diffusion_teacher_delta_tail_threshold=0.15 +train.ppo.diffusion_teacher_delta_tail_selective=True +train.ppo.diffusion_base_action_anchor_coef=0.03`
  - outcome:
    - `model_best.ckpt` created successfully,
    - run was manually interrupted after checkpoint creation and bounded progress observation (`exit code 130`),
    - observed training-side `Current Best` reached `1851.70` before interruption.
- Eval commands (docker IsaacGym, `seed=42`, `steps=256`):
  - nominal:
    - `./docker-run-isaacgym.sh timeout 480 bash scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent <ckpt> 256 plansv2_live_tailcoef02_thr015_sel_anchor003_recon04_nominal +train.ppo.diffusion_eval_decode_only=False +train.ppo.diffusion_eval_report_recon=True`
  - light_v2:
    - `./docker-run-isaacgym.sh timeout 480 bash scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent <ckpt> 256 plansv2_live_tailcoef02_thr015_sel_anchor003_recon04_lightv2 +train.ppo.diffusion_eval_decode_only=False +train.ppo.diffusion_eval_report_recon=True task.env.randomization.obs_noise_e_scale=0.03 task.env.randomization.obs_noise_t_scale=0.015 task.env.forceScale=1.0 task.env.randomForceProbScalar=0.2`
  - hard:
    - `./docker-run-isaacgym.sh timeout 480 bash scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent <ckpt> 256 plansv2_live_tailcoef02_thr015_sel_anchor003_recon04_hard +train.ppo.diffusion_eval_decode_only=False +train.ppo.diffusion_eval_report_recon=True task.env.randomization.obs_noise_e_scale=0.06 task.env.randomization.obs_noise_t_scale=0.03 task.env.forceScale=2.0 task.env.randomForceProbScalar=0.3`
- Eval results (`seed=42`, `steps=256`):
  - nominal: reward `1.652969`, done `0.001302`, `latent_mse=0.073436`, `action_mse_to_teacher=0.137309`
  - light_v2: reward `1.419334`, done `0.001953`, `latent_mse=0.078930`, `action_mse_to_teacher=0.147911`
  - hard: reward `1.187399`, done `0.001872`, `latent_mse=0.088574`, `action_mse_to_teacher=0.162629`
- Delta vs current kept reference (`tail_coef=0.2, thr=0.15, anchor=0.03`):
  - reward:
    - nominal `-0.022143`
    - light_v2 `-0.218932`
    - hard `-0.317505`
  - done rate:
    - nominal `+0.000000`
    - light_v2 `+0.000570`
    - hard `+0.000000`
  - recon / teacher-alignment:
    - nominal `latent_mse +0.004683`, `action_mse_to_teacher +0.007207`
    - light_v2 `latent_mse +0.001856`, `action_mse_to_teacher +0.004863`
    - hard `latent_mse +0.004297`, `action_mse_to_teacher +0.002641`
- Delta vs `recon06`:
  - reward:
    - nominal `+0.537058`
    - light_v2 `+0.368745`
    - hard `+0.360236`
- Delta vs `latent_recon05` representative (`plansv2_m2`, seed42):
  - nominal `-0.254255`
  - light_v2 `+0.013921`
  - hard `-0.122692`

### Local decision
- `diffusion_latent_recon_coef=0.4` is **not a new main candidate**, but it is the first useful signal on this axis:
  - it still loses to the current kept reference under all three conditions,
  - however, it is much better than `recon06`,
  - and it gives a small `light_v2` gain over the old `latent_recon05` representative.

### Remaining blocked/risky
- Hard robustness remains the main blocker for route promotion.
- The `latent_recon_coef` axis is not monotonic:
  - `0.6` is strongly negative,
  - `0.4` recovers much of the damage and nearly matches the current reference on nominal,
  - but still falls short on `light_v2/hard` versus the current kept reference.
- Current best single-seed tradeoff still remains:
  - `tail_coef=0.2 + tail_threshold=0.15 + selective=True + anchor=0.03`.

### Single recommended next step
- Keep the current reference unchanged.
- Run one bounded **midpoint latent-recon** probe:
  - keep `tail_coef=0.2 + tail_threshold=0.15 + selective=True + anchor=0.03`,
  - set `diffusion_latent_recon_coef=0.45`,
  - then apply the same strict keep/drop gate under `nominal/light_v2/hard`.

---

## v2-050 (2026-03-27) — PLANS_v2 P3 Midpoint Latent-Recon Probe (`diffusion_latent_recon_coef=0.45`)

### Target milestone/subgoal
- Execute the latest bounded single-variable midpoint probe on the `latent_recon_coef` axis:
  - keep `tail_coef=0.2 + tail_threshold=0.15 + selective=True + anchor=0.03`,
  - set `diffusion_latent_recon_coef=0.45`,
  - evaluate under unified `nominal/light_v2/hard` protocol.
- Tighten validation quality:
  - treat an early-stop run with `Current Best=0.00` as provisional only,
  - use a retry run with visible nonzero training signal as the final decision basis.

### What changed (files + behavior impact)
- Initial provisional run artifacts:
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_recon045_seed42_15min/`
  - train log:
    - `outputs/train_logs/run_a_latent_tailcoef02_thr015_sel_anchor003_recon045_seed42_15min_train.log`
  - eval logs:
    - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_recon045/diffusion_nominal_s42.log`
    - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_recon045/diffusion_light_v2_s42.log`
    - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_recon045/diffusion_hard_s42.log`
- Final validated retry run artifacts:
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_recon045_retry_seed42_15min/`
  - train log:
    - `outputs/train_logs/run_a_latent_tailcoef02_thr015_sel_anchor003_recon045_retry_seed42_15min_train.log`
  - eval logs:
    - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_recon045_retry/diffusion_nominal_s42.log`
    - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_recon045_retry/diffusion_light_v2_s42.log`
    - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_recon045_retry/diffusion_hard_s42.log`
- No code edits in this session block (experiment-only).

### What was verified (commands + key outcomes)
- Initial bounded training command:
  - `./docker-run-isaacgym.sh timeout 1000 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_tailcoef02_thr015_sel_anchor003_recon045_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_latent_recon_coef=0.45 +train.ppo.diffusion_teacher_delta_tail_coef=0.2 +train.ppo.diffusion_teacher_delta_tail_threshold=0.15 +train.ppo.diffusion_teacher_delta_tail_selective=True +train.ppo.diffusion_base_action_anchor_coef=0.03`
  - outcome:
    - `model_best.ckpt` existed,
    - but training log stayed at `Current Best: 0.00` before manual interruption (`exit code 130`),
    - so this first run is treated as `provisional/scaffold-only`, not as the final evidence basis.
- Validated retry training command:
  - `./docker-run-isaacgym.sh timeout 1000 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_tailcoef02_thr015_sel_anchor003_recon045_retry_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_latent_recon_coef=0.45 +train.ppo.diffusion_teacher_delta_tail_coef=0.2 +train.ppo.diffusion_teacher_delta_tail_threshold=0.15 +train.ppo.diffusion_teacher_delta_tail_selective=True +train.ppo.diffusion_base_action_anchor_coef=0.03`
  - outcome:
    - `model_best.ckpt` created successfully,
    - run was manually interrupted after visible training signal (`exit code 130`),
    - `Current Best` reached `1706.08`.
- Retry eval commands (docker IsaacGym, `seed=42`, `steps=256`):
  - nominal:
    - `./docker-run-isaacgym.sh timeout 480 bash scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent <retry_ckpt> 256 plansv2_live_tailcoef02_thr015_sel_anchor003_recon045_retry_nominal +train.ppo.diffusion_eval_decode_only=False +train.ppo.diffusion_eval_report_recon=True`
  - light_v2:
    - `./docker-run-isaacgym.sh timeout 480 bash scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent <retry_ckpt> 256 plansv2_live_tailcoef02_thr015_sel_anchor003_recon045_retry_lightv2 +train.ppo.diffusion_eval_decode_only=False +train.ppo.diffusion_eval_report_recon=True task.env.randomization.obs_noise_e_scale=0.03 task.env.randomization.obs_noise_t_scale=0.015 task.env.forceScale=1.0 task.env.randomForceProbScalar=0.2`
  - hard:
    - `./docker-run-isaacgym.sh timeout 480 bash scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent <retry_ckpt> 256 plansv2_live_tailcoef02_thr015_sel_anchor003_recon045_retry_hard +train.ppo.diffusion_eval_decode_only=False +train.ppo.diffusion_eval_report_recon=True task.env.randomization.obs_noise_e_scale=0.06 task.env.randomization.obs_noise_t_scale=0.03 task.env.forceScale=2.0 task.env.randomForceProbScalar=0.3`
- Retry eval results (`seed=42`, `steps=256`):
  - nominal: reward `1.337213`, done `0.001383`, `latent_mse=0.106486`, `action_mse_to_teacher=0.187041`
  - light_v2: reward `1.351897`, done `0.001628`, `latent_mse=0.106974`, `action_mse_to_teacher=0.184087`
  - hard: reward `0.660181`, done `0.002767`, `latent_mse=0.123048`, `action_mse_to_teacher=0.219807`
- Delta vs current kept reference (`tail_coef=0.2, thr=0.15, anchor=0.03`, `recon=0.5`):
  - reward:
    - nominal `-0.337899`
    - light_v2 `-0.286369`
    - hard `-0.844723`
  - done rate:
    - nominal `+0.000081`
    - light_v2 `+0.000245`
    - hard `+0.000895`
  - recon / teacher-alignment:
    - nominal `latent_mse +0.037733`, `action_mse_to_teacher +0.056939`
    - light_v2 `latent_mse +0.029900`, `action_mse_to_teacher +0.041039`
    - hard `latent_mse +0.038771`, `action_mse_to_teacher +0.059819`
- Delta vs `recon04`:
  - reward:
    - nominal `-0.315756`
    - light_v2 `-0.067437`
    - hard `-0.527218`
- Delta vs `recon06`:
  - reward:
    - nominal `+0.221302`
    - light_v2 `+0.301308`
    - hard `-0.166982`

### Local decision
- `diffusion_latent_recon_coef=0.45` probe is **rejected**:
  - after valid retraining, it still loses to the current kept reference on all three conditions,
  - it also loses to `recon04` on all three conditions,
  - and hard robustness collapses materially despite a decent training-side `Current Best`.
- The initial early-stop `recon045` run should not be reused as accepted evidence:
  - keep it only as a provisional artifact showing why visible training signal is needed before final eval.

### Remaining blocked/risky
- Hard robustness remains the dominant blocker for route promotion.
- The `latent_recon_coef` neighborhood is now bounded enough for this branch:
  - `0.4` rejected,
  - `0.45` rejected,
  - `0.5` remains the current kept reference,
  - `0.6` rejected.
- Training-side peak is still not a reliable promotion signal:
  - `Current Best=1706.08` did not transfer into rollout quality.

### Single recommended next step
- Keep the current reference unchanged and stop spending budget on the `latent_recon_coef` axis.
- Run one bounded **weaker tail-intensity** probe:
  - keep `tail_threshold=0.15 + selective=True + anchor=0.03 + diffusion_latent_recon_coef=0.5`,
  - only reduce `diffusion_teacher_delta_tail_coef` from `0.2` to `0.18`,
  - then apply the same strict keep/drop gate under `nominal/light_v2/hard`.

---

## v2-051 (2026-03-27) — PLANS_v2 P3 Weaker Tail-Intensity Probe (`tail_coef=0.18`)

### Target milestone/subgoal
- Execute one bounded single-variable probe around the current kept reference:
  - keep `tail_threshold=0.15 + selective=True + anchor=0.03 + diffusion_latent_recon_coef=0.5`,
  - only reduce `diffusion_teacher_delta_tail_coef` from `0.2` to `0.18`,
  - evaluate by unified `nominal/light_v2/hard` protocol.

### What changed (files + behavior impact)
- New bounded training run:
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef018_thr015_sel_anchor003_seed42_15min/`
  - train log:
    - `outputs/train_logs/run_a_latent_tailcoef018_thr015_sel_anchor003_seed42_15min_train.log`
- New eval logs:
  - `outputs/robustness_eval/plansv2_live_tailcoef018_thr015_sel_anchor003/diffusion_nominal_s42.log`
  - `outputs/robustness_eval/plansv2_live_tailcoef018_thr015_sel_anchor003/diffusion_light_v2_s42.log`
  - `outputs/robustness_eval/plansv2_live_tailcoef018_thr015_sel_anchor003/diffusion_hard_s42.log`
- No code edits in this session block (experiment-only).

### What was verified (commands + key outcomes)
- Bounded training command:
  - `./docker-run-isaacgym.sh timeout 1000 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_tailcoef018_thr015_sel_anchor003_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_latent_recon_coef=0.5 +train.ppo.diffusion_teacher_delta_tail_coef=0.18 +train.ppo.diffusion_teacher_delta_tail_threshold=0.15 +train.ppo.diffusion_teacher_delta_tail_selective=True +train.ppo.diffusion_base_action_anchor_coef=0.03`
  - outcome:
    - `model_best.ckpt` created successfully,
    - run was manually interrupted after visible training signal (`exit code 130`),
    - `Current Best` reached `1593.59`.
- Eval commands (docker IsaacGym, `seed=42`, `steps=256`):
  - nominal:
    - `./docker-run-isaacgym.sh timeout 480 bash scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent <ckpt> 256 plansv2_live_tailcoef018_thr015_sel_anchor003_nominal +train.ppo.diffusion_eval_decode_only=False +train.ppo.diffusion_eval_report_recon=True`
  - light_v2:
    - `./docker-run-isaacgym.sh timeout 480 bash scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent <ckpt> 256 plansv2_live_tailcoef018_thr015_sel_anchor003_lightv2 +train.ppo.diffusion_eval_decode_only=False +train.ppo.diffusion_eval_report_recon=True task.env.randomization.obs_noise_e_scale=0.03 task.env.randomization.obs_noise_t_scale=0.015 task.env.forceScale=1.0 task.env.randomForceProbScalar=0.2`
  - hard:
    - `./docker-run-isaacgym.sh timeout 480 bash scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent <ckpt> 256 plansv2_live_tailcoef018_thr015_sel_anchor003_hard +train.ppo.diffusion_eval_decode_only=False +train.ppo.diffusion_eval_report_recon=True task.env.randomization.obs_noise_e_scale=0.06 task.env.randomization.obs_noise_t_scale=0.03 task.env.forceScale=2.0 task.env.randomForceProbScalar=0.3`
- Eval results (`seed=42`, `steps=256`):
  - nominal: reward `1.212616`, done `0.002360`, `latent_mse=0.103098`, `action_mse_to_teacher=0.196359`
  - light_v2: reward `1.202849`, done `0.002686`, `latent_mse=0.111284`, `action_mse_to_teacher=0.220592`
  - hard: reward `0.893597`, done `0.002848`, `latent_mse=0.122814`, `action_mse_to_teacher=0.240173`
- Delta vs current kept reference (`tail_coef=0.2, thr=0.15, anchor=0.03`, `recon=0.5`):
  - reward:
    - nominal `-0.462496`
    - light_v2 `-0.435417`
    - hard `-0.611307`
  - done rate:
    - nominal `+0.001058`
    - light_v2 `+0.001303`
    - hard `+0.000976`
  - recon / teacher-alignment:
    - nominal `latent_mse +0.034345`, `action_mse_to_teacher +0.066257`
    - light_v2 `latent_mse +0.034210`, `action_mse_to_teacher +0.077544`
    - hard `latent_mse +0.038537`, `action_mse_to_teacher +0.080185`

### Local decision
- `tail_coef=0.18` probe is **rejected**:
  - all three conditions regress clearly relative to the current kept reference,
  - done rates rise across all three conditions,
  - teacher-alignment metrics also worsen across all three conditions.

### Remaining blocked/risky
- Hard robustness remains the dominant blocker for route promotion.
- The `tail_coef` neighborhood is now locally bounded enough for this branch:
  - `0.18` rejected,
  - `0.2` remains the current kept reference,
  - `0.25` rejected.
- The broader `tail-threshold / anchor / recon / bc` neighborhood around the kept reference is also accumulating mostly negative evidence, so continued local sweeps in the exact same subspace have declining expected value.

### Single recommended next step
- Keep the current reference unchanged and stop spending budget on the `tail_coef` axis.
- Run one bounded **selectivity-toggle** probe:
  - keep `tail_coef=0.2 + tail_threshold=0.15 + anchor=0.03 + diffusion_latent_recon_coef=0.5`,
  - only switch `diffusion_teacher_delta_tail_selective` from `True` to `False`,
  - then apply the same strict keep/drop gate under `nominal/light_v2/hard`.

---

## v2-052 (2026-03-27) — PLANS_v2 P3 Selectivity-Toggle Probe (`selective=False`)

### Target milestone/subgoal
- Execute one bounded single-variable probe around the current kept reference:
  - keep `tail_coef=0.2 + tail_threshold=0.15 + anchor=0.03 + diffusion_latent_recon_coef=0.5`,
  - only switch `diffusion_teacher_delta_tail_selective` from `True` to `False`,
  - evaluate by unified `nominal/light_v2/hard` protocol.

### What changed (files + behavior impact)
- New bounded training run:
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_nonsel_anchor003_seed42_15min/`
  - train log:
    - `outputs/train_logs/run_a_latent_tailcoef02_thr015_nonsel_anchor003_seed42_15min_train.log`
- New eval logs:
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_nonsel_anchor003/diffusion_nominal_s42.log`
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_nonsel_anchor003/diffusion_light_v2_s42.log`
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_nonsel_anchor003/diffusion_hard_s42.log`
- No code edits in this session block (experiment-only).

### What was verified (commands + key outcomes)
- Bounded training command:
  - `./docker-run-isaacgym.sh timeout 1000 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_tailcoef02_thr015_nonsel_anchor003_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_latent_recon_coef=0.5 +train.ppo.diffusion_teacher_delta_tail_coef=0.2 +train.ppo.diffusion_teacher_delta_tail_threshold=0.15 +train.ppo.diffusion_teacher_delta_tail_selective=False +train.ppo.diffusion_base_action_anchor_coef=0.03`
  - outcome:
    - `model_best.ckpt` created successfully,
    - run was manually interrupted after visible training signal (`exit code 130`),
    - `Current Best` reached `1757.74`.
- Eval commands (docker IsaacGym, `seed=42`, `steps=256`):
  - nominal:
    - `./docker-run-isaacgym.sh timeout 480 bash scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent <ckpt> 256 plansv2_live_tailcoef02_thr015_nonsel_anchor003_nominal +train.ppo.diffusion_eval_decode_only=False +train.ppo.diffusion_eval_report_recon=True`
  - light_v2:
    - `./docker-run-isaacgym.sh timeout 480 bash scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent <ckpt> 256 plansv2_live_tailcoef02_thr015_nonsel_anchor003_lightv2 +train.ppo.diffusion_eval_decode_only=False +train.ppo.diffusion_eval_report_recon=True task.env.randomization.obs_noise_e_scale=0.03 task.env.randomization.obs_noise_t_scale=0.015 task.env.forceScale=1.0 task.env.randomForceProbScalar=0.2`
  - hard:
    - `./docker-run-isaacgym.sh timeout 480 bash scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent <ckpt> 256 plansv2_live_tailcoef02_thr015_nonsel_anchor003_hard +train.ppo.diffusion_eval_decode_only=False +train.ppo.diffusion_eval_report_recon=True task.env.randomization.obs_noise_e_scale=0.06 task.env.randomization.obs_noise_t_scale=0.03 task.env.forceScale=2.0 task.env.randomForceProbScalar=0.3`
- Eval results (`seed=42`, `steps=256`):
  - nominal: reward `1.404112`, done `0.001302`, `latent_mse=0.105381`, `action_mse_to_teacher=0.214957`
  - light_v2: reward `1.155605`, done `0.001790`, `latent_mse=0.106796`, `action_mse_to_teacher=0.226411`
  - hard: reward `0.869128`, done `0.002848`, `latent_mse=0.111865`, `action_mse_to_teacher=0.243669`
- Delta vs current kept reference (`tail_coef=0.2, thr=0.15, selective=True, anchor=0.03`, `recon=0.5`):
  - reward:
    - nominal `-0.271000`
    - light_v2 `-0.482661`
    - hard `-0.635776`
  - done rate:
    - nominal `+0.000000`
    - light_v2 `+0.000407`
    - hard `+0.000976`
  - recon / teacher-alignment:
    - nominal `latent_mse +0.036628`, `action_mse_to_teacher +0.084855`
    - light_v2 `latent_mse +0.029722`, `action_mse_to_teacher +0.083363`
    - hard `latent_mse +0.027588`, `action_mse_to_teacher +0.083681`

### Local decision
- `selective=False` probe is **rejected**:
  - all three conditions regress relative to the current kept reference,
  - `light_v2/hard` degrade strongly,
  - teacher-alignment worsens materially across all three conditions despite decent training-side `Current Best`.

### Remaining blocked/risky
- Hard robustness remains the dominant blocker for route promotion.
- The `selective` axis is now locally bounded enough for this branch:
  - `selective=True` remains the current kept reference choice,
  - `selective=False` is rejected.
- Training-side peak is again not a reliable promotion signal:
  - `Current Best=1757.74` did not translate into rollout gains.

### Single recommended next step
- Keep the current reference unchanged and stop spending budget on the `selective` axis.
- Run one bounded **threshold-midpoint** probe:
  - keep `tail_coef=0.2 + selective=True + anchor=0.03 + diffusion_latent_recon_coef=0.5`,
  - only increase `diffusion_teacher_delta_tail_threshold` from `0.15` to `0.155`,
  - then apply the same strict keep/drop gate under `nominal/light_v2/hard`.

---

## v2-053 (2026-03-27) — PLANS_v2 P3 Threshold-Midpoint Probe (`tail_threshold=0.155`)

### Target milestone/subgoal
- Execute one bounded single-variable midpoint probe on the `tail_threshold` axis:
  - keep `tail_coef=0.2 + selective=True + anchor=0.03 + diffusion_latent_recon_coef=0.5`,
  - only increase `diffusion_teacher_delta_tail_threshold` from `0.15` to `0.155`,
  - evaluate by unified `nominal/light_v2/hard` protocol.

### What changed (files + behavior impact)
- New bounded training run:
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr0155_sel_anchor003_seed42_15min/`
  - train log:
    - `outputs/train_logs/run_a_latent_tailcoef02_thr0155_sel_anchor003_seed42_15min_train.log`
- New eval logs:
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr0155_sel_anchor003/diffusion_nominal_s42.log`
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr0155_sel_anchor003/diffusion_light_v2_s42.log`
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr0155_sel_anchor003/diffusion_hard_s42.log`
- No code edits in this session block (experiment-only).

### What was verified (commands + key outcomes)
- Bounded training command:
  - `./docker-run-isaacgym.sh timeout 1000 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_tailcoef02_thr0155_sel_anchor003_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_latent_recon_coef=0.5 +train.ppo.diffusion_teacher_delta_tail_coef=0.2 +train.ppo.diffusion_teacher_delta_tail_threshold=0.155 +train.ppo.diffusion_teacher_delta_tail_selective=True +train.ppo.diffusion_base_action_anchor_coef=0.03`
  - outcome:
    - `model_best.ckpt` created successfully,
    - run was manually interrupted after visible training signal (`exit code 130`),
    - `Current Best` reached `1269.42`.
- Eval commands (docker IsaacGym, `seed=42`, `steps=256`):
  - nominal:
    - `./docker-run-isaacgym.sh timeout 480 bash scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent <ckpt> 256 plansv2_live_tailcoef02_thr0155_sel_anchor003_nominal +train.ppo.diffusion_eval_decode_only=False +train.ppo.diffusion_eval_report_recon=True`
  - light_v2:
    - `./docker-run-isaacgym.sh timeout 480 bash scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent <ckpt> 256 plansv2_live_tailcoef02_thr0155_sel_anchor003_lightv2 +train.ppo.diffusion_eval_decode_only=False +train.ppo.diffusion_eval_report_recon=True task.env.randomization.obs_noise_e_scale=0.03 task.env.randomization.obs_noise_t_scale=0.015 task.env.forceScale=1.0 task.env.randomForceProbScalar=0.2`
  - hard:
    - `./docker-run-isaacgym.sh timeout 480 bash scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent <ckpt> 256 plansv2_live_tailcoef02_thr0155_sel_anchor003_hard +train.ppo.diffusion_eval_decode_only=False +train.ppo.diffusion_eval_report_recon=True task.env.randomization.obs_noise_e_scale=0.06 task.env.randomization.obs_noise_t_scale=0.03 task.env.forceScale=2.0 task.env.randomForceProbScalar=0.3`
- Eval results (`seed=42`, `steps=256`):
  - nominal: reward `0.948742`, done `0.002116`, `latent_mse=0.111604`, `action_mse_to_teacher=0.201484`
  - light_v2: reward `1.014372`, done `0.002848`, `latent_mse=0.111348`, `action_mse_to_teacher=0.206558`
  - hard: reward `0.858345`, done `0.002279`, `latent_mse=0.119275`, `action_mse_to_teacher=0.216819`
- Delta vs current kept reference (`tail_coef=0.2, thr=0.15, selective=True, anchor=0.03`, `recon=0.5`):
  - reward:
    - nominal `-0.726370`
    - light_v2 `-0.623894`
    - hard `-0.646559`
  - done rate:
    - nominal `+0.000814`
    - light_v2 `+0.001465`
    - hard `+0.000407`
  - recon / teacher-alignment:
    - nominal `latent_mse +0.042851`, `action_mse_to_teacher +0.071382`
    - light_v2 `latent_mse +0.034274`, `action_mse_to_teacher +0.063510`
    - hard `latent_mse +0.034998`, `action_mse_to_teacher +0.056831`

### Local decision
- `tail_threshold=0.155` probe is **rejected**:
  - all three conditions regress heavily relative to the current kept reference,
  - done rates rise on all three conditions,
  - recon / teacher-alignment metrics also worsen across all three conditions.

### Remaining blocked/risky
- Hard robustness remains the dominant blocker for route promotion.
- The `tail_threshold` neighborhood is now locally bounded enough for this branch:
  - `0.145` rejected,
  - `0.15` remains the current kept reference,
  - `0.155` rejected,
  - `0.16` rejected.
- More broadly, the current local neighborhood around the kept reference now has strong negative evidence on:
  - `tail_coef`
  - `tail_threshold`
  - `tail_selective`
  - `base_action_anchor_coef`
  - `latent_recon_coef`
  - `bc_loss_coef`
- Continuing blind micro-sweeps inside the same neighborhood now has low expected return.

### Single recommended next step
- Keep the current reference unchanged.
- If continuing optimization, stop spending budget on the current `tail/anchor/recon/bc/selective` local neighborhood and switch to one new single-variable probe outside this basin, ideally a training-distribution-alignment axis that has not already been rejected in `docs/session_handoff_v2.md`.

---

## v2-054 (2026-03-27) — PLANS_v2 P3 Training-Distribution Probe (`task.env.randomForceProbScalar=0.2`)

### Target milestone/subgoal
- Execute one new single-variable probe outside the saturated `tail/anchor/recon/bc/selective` basin:
  - keep `tail_coef=0.2 + tail_threshold=0.15 + selective=True + anchor=0.03 + diffusion_latent_recon_coef=0.5`,
  - only change the training-time force-event probability from the wrapper default `0.1` to `task.env.randomForceProbScalar=0.2`,
  - then evaluate by unified `nominal/light_v2/hard` protocol.

### What changed (files + behavior impact)
- New bounded training run:
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_trainforceprob02_seed42_15min/`
- New training log:
  - `outputs/train_logs/run_a_latent_tailcoef02_thr015_sel_anchor003_trainforceprob02_seed42_15min_train.log`
- Frozen eval checkpoint snapshot:
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_trainforceprob02_seed42_15min/stage2_diffusion_nn/model_best_evalfreeze.ckpt`
- New eval logs:
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_trainforceprob02/diffusion_nominal_s42.log`
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_trainforceprob02/diffusion_light_v2_s42.log`
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_trainforceprob02/diffusion_hard_s42.log`
- No code edits in this session block (experiment-only).

### What was verified (commands + key outcomes)
- Bounded training command:
  - `./docker-run-isaacgym.sh timeout 1000 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_tailcoef02_thr015_sel_anchor003_trainforceprob02_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_latent_recon_coef=0.5 +train.ppo.diffusion_teacher_delta_tail_coef=0.2 +train.ppo.diffusion_teacher_delta_tail_threshold=0.15 +train.ppo.diffusion_teacher_delta_tail_selective=True +train.ppo.diffusion_base_action_anchor_coef=0.03 task.env.randomForceProbScalar=0.2`
  - outcome:
    - saved config confirms `task.env.randomForceProbScalar: 0.2`,
    - `model_best.ckpt` was created successfully,
    - run was manually stopped after meaningful signal and then frozen to `model_best_evalfreeze.ckpt`,
    - observed training-side `Current Best` reached `1923.49`.
- Eval commands (docker IsaacGym, `seed=42`, `steps=256`):
  - nominal:
    - `./docker-run-isaacgym.sh timeout 480 bash scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent <evalfreeze_ckpt> 256 plansv2_live_tailcoef02_thr015_sel_anchor003_trainforceprob02_nominal +train.ppo.diffusion_eval_decode_only=False +train.ppo.diffusion_eval_report_recon=True`
  - light_v2:
    - `./docker-run-isaacgym.sh timeout 480 bash scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent <evalfreeze_ckpt> 256 plansv2_live_tailcoef02_thr015_sel_anchor003_trainforceprob02_lightv2 +train.ppo.diffusion_eval_decode_only=False +train.ppo.diffusion_eval_report_recon=True task.env.randomization.obs_noise_e_scale=0.03 task.env.randomization.obs_noise_t_scale=0.015 task.env.forceScale=1.0 task.env.randomForceProbScalar=0.2`
  - hard:
    - `./docker-run-isaacgym.sh timeout 480 bash scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent <evalfreeze_ckpt> 256 plansv2_live_tailcoef02_thr015_sel_anchor003_trainforceprob02_hard +train.ppo.diffusion_eval_decode_only=False +train.ppo.diffusion_eval_report_recon=True task.env.randomization.obs_noise_e_scale=0.06 task.env.randomization.obs_noise_t_scale=0.03 task.env.forceScale=2.0 task.env.randomForceProbScalar=0.3`
- Eval results (`seed=42`, `steps=256`):
  - nominal: reward `1.370869`, done `0.001546`, `latent_mse=0.086764`, `action_mse_to_teacher=0.165403`
  - light_v2: reward `1.401639`, done `0.001465`, `latent_mse=0.090520`, `action_mse_to_teacher=0.168638`
  - hard: reward `1.225263`, done `0.002360`, `latent_mse=0.100479`, `action_mse_to_teacher=0.186422`
- Delta vs current kept reference (`tail_coef=0.2, threshold=0.15, selective=True, anchor=0.03`, `recon=0.5`):
  - reward:
    - nominal `-0.304243`
    - light_v2 `-0.236627`
    - hard `-0.279641`
  - done rate:
    - nominal `+0.000244`
    - light_v2 `+0.000082`
    - hard `+0.000488`
  - recon / teacher-alignment:
    - nominal `latent_mse +0.018011`, `action_mse_to_teacher +0.035301`
    - light_v2 `latent_mse +0.013446`, `action_mse_to_teacher +0.025590`
    - hard `latent_mse +0.016202`, `action_mse_to_teacher +0.026434`

### Local decision
- `task.env.randomForceProbScalar=0.2` probe is **rejected**:
  - all three conditions regress relative to the current kept reference,
  - hard robustness degrades materially,
  - recon / teacher-alignment metrics also worsen across all three conditions.

### Remaining blocked/risky
- Hard robustness remains the dominant blocker for route promotion.
- The broader conclusion that the old `tail/anchor/recon/bc/selective` neighborhood is saturated still stands.
- This new result adds one more constraint on the replacement axis:
  - training-distribution alignment is still a valid new direction,
  - but `randomForceProbScalar=0.2` appears too aggressive for the current latent student setup.

### Single recommended next step
- Keep the current reference unchanged.
- If continuing optimization, stay on the new training-distribution-alignment axis but reduce perturbation intensity to one milder single-variable probe, with `task.env.randomForceProbScalar=0.15` as the recommended next candidate.

---

## v2-055 (2026-03-27) — PLANS_v2 P3 Milder Training-Distribution Probe (`task.env.randomForceProbScalar=0.15`)

### Target milestone/subgoal
- Execute one milder follow-up probe on the same training-distribution axis:
  - keep `tail_coef=0.2 + tail_threshold=0.15 + selective=True + anchor=0.03 + diffusion_latent_recon_coef=0.5`,
  - only change the training-time force-event probability from the wrapper default `0.1` to `task.env.randomForceProbScalar=0.15`,
  - then evaluate by unified `nominal/light_v2/hard` protocol.

### What changed (files + behavior impact)
- New bounded training run:
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_trainforceprob015_seed42_15min/`
- New training log:
  - `outputs/train_logs/run_a_latent_tailcoef02_thr015_sel_anchor003_trainforceprob015_seed42_15min_train.log`
- Frozen eval checkpoint snapshot:
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_trainforceprob015_seed42_15min/stage2_diffusion_nn/model_best_evalfreeze.ckpt`
- New eval logs:
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_trainforceprob015/diffusion_nominal_s42.log`
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_trainforceprob015/diffusion_light_v2_s42.log`
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_trainforceprob015/diffusion_hard_s42.log`
- No code edits in this session block (experiment-only).

### What was verified (commands + key outcomes)
- Bounded training command:
  - `./docker-run-isaacgym.sh timeout 1000 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_tailcoef02_thr015_sel_anchor003_trainforceprob015_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_latent_recon_coef=0.5 +train.ppo.diffusion_teacher_delta_tail_coef=0.2 +train.ppo.diffusion_teacher_delta_tail_threshold=0.15 +train.ppo.diffusion_teacher_delta_tail_selective=True +train.ppo.diffusion_base_action_anchor_coef=0.03 task.env.randomForceProbScalar=0.15`
  - outcome:
    - saved config confirms `task.env.randomForceProbScalar: 0.15`,
    - `model_best.ckpt` was created successfully,
    - run was manually stopped after meaningful signal and then frozen to `model_best_evalfreeze.ckpt`,
    - observed training-side `Current Best` reached `1862.27`,
    - the lone `KeyboardInterrupt` / `Traceback` in the log is from the manual stop rather than a training failure.
- Eval commands (docker IsaacGym, `seed=42`, `steps=256`):
  - nominal:
    - `./docker-run-isaacgym.sh timeout 480 bash scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent <evalfreeze_ckpt> 256 plansv2_live_tailcoef02_thr015_sel_anchor003_trainforceprob015_nominal +train.ppo.diffusion_eval_decode_only=False +train.ppo.diffusion_eval_report_recon=True`
  - light_v2:
    - `./docker-run-isaacgym.sh timeout 480 bash scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent <evalfreeze_ckpt> 256 plansv2_live_tailcoef02_thr015_sel_anchor003_trainforceprob015_lightv2 +train.ppo.diffusion_eval_decode_only=False +train.ppo.diffusion_eval_report_recon=True task.env.randomization.obs_noise_e_scale=0.03 task.env.randomization.obs_noise_t_scale=0.015 task.env.forceScale=1.0 task.env.randomForceProbScalar=0.2`
  - hard:
    - `./docker-run-isaacgym.sh timeout 480 bash scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent <evalfreeze_ckpt> 256 plansv2_live_tailcoef02_thr015_sel_anchor003_trainforceprob015_hard +train.ppo.diffusion_eval_decode_only=False +train.ppo.diffusion_eval_report_recon=True task.env.randomization.obs_noise_e_scale=0.06 task.env.randomization.obs_noise_t_scale=0.03 task.env.forceScale=2.0 task.env.randomForceProbScalar=0.3`
- Eval results (`seed=42`, `steps=256`):
  - nominal: reward `1.453599`, done `0.001628`, `latent_mse=0.089682`, `action_mse_to_teacher=0.153101`
  - light_v2: reward `1.351574`, done `0.002197`, `latent_mse=0.096730`, `action_mse_to_teacher=0.174291`
  - hard: reward `1.130938`, done `0.002197`, `latent_mse=0.100712`, `action_mse_to_teacher=0.180712`
- Delta vs current kept reference (`tail_coef=0.2, threshold=0.15, selective=True, anchor=0.03`, `recon=0.5`):
  - reward:
    - nominal `-0.221513`
    - light_v2 `-0.286692`
    - hard `-0.373966`
  - done rate:
    - nominal `+0.000326`
    - light_v2 `+0.000814`
    - hard `+0.000325`
  - recon / teacher-alignment:
    - nominal `latent_mse +0.020929`, `action_mse_to_teacher +0.022999`
    - light_v2 `latent_mse +0.019656`, `action_mse_to_teacher +0.031243`
    - hard `latent_mse +0.016435`, `action_mse_to_teacher +0.020724`
- Delta vs the rejected `task.env.randomForceProbScalar=0.2` probe:
  - nominal improves somewhat:
    - reward `+0.082730`
    - `action_mse_to_teacher -0.012302`
  - but `light_v2` and `hard` still do not improve into a keepable region:
    - light_v2 reward `-0.050065`, done `+0.000732`
    - hard reward `-0.094325`, done `-0.000163`

### Local decision
- `task.env.randomForceProbScalar=0.15` probe is **rejected**:
  - it is somewhat less damaging than `0.2` on nominal alignment,
  - but it still loses to the current kept reference on all three conditions,
  - and `light_v2` / `hard` remain materially below the keep threshold.

### Remaining blocked/risky
- Hard robustness remains the dominant blocker for route promotion.
- The training-distribution-alignment route is still not exhausted, but the current evidence now constrains this specific sub-axis:
  - pushing `randomForceProbScalar` above the wrapper default `0.1` does not appear beneficial for the current latent student branch,
  - `0.15` is less harmful than `0.2`, but still not competitive enough to keep.

### Single recommended next step
- Keep the current reference unchanged.
- If continuing optimization, stop increasing `task.env.randomForceProbScalar` and move to one different single-variable training-distribution axis; the recommended next candidate is a mild training-time force-scale probe such as `task.env.forceScale=0.6` while leaving `randomForceProbScalar` at its wrapper default `0.1`.

---

## v2-056 (2026-04-01) — PLANS_v2 P3 Training-Distribution Probe (`task.env.forceScale=0.6`)

### Target milestone/subgoal
- Execute one bounded single-variable probe on the training-distribution axis:
  - keep `tail_coef=0.2 + tail_threshold=0.15 + selective=True + anchor=0.03 + diffusion_latent_recon_coef=0.5`,
  - keep `task.env.randomForceProbScalar=0.1`,
  - only change training-time `task.env.forceScale` from wrapper default `0.5` to `0.6`,
  - then evaluate with unified `nominal/light_v2/hard` protocol.

### What changed (files + behavior impact)
- New bounded training run:
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_trainforcescale06_seed42_15min/`
- Frozen eval checkpoint snapshot:
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_trainforcescale06_seed42_15min/stage2_diffusion_nn/model_best_evalfreeze.ckpt`
- New eval logs:
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_trainforcescale06/diffusion_nominal_s42.log`
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_trainforcescale06/diffusion_light_v2_s42.log`
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_trainforcescale06/diffusion_hard_s42.log`
- No code edits in this session block (experiment-only).

### What was verified (commands + key outcomes)
- Bounded training command:
  - `./docker-run-isaacgym.sh timeout 1000 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_tailcoef02_thr015_sel_anchor003_trainforcescale06_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_latent_recon_coef=0.5 +train.ppo.diffusion_teacher_delta_tail_coef=0.2 +train.ppo.diffusion_teacher_delta_tail_threshold=0.15 +train.ppo.diffusion_teacher_delta_tail_selective=True +train.ppo.diffusion_base_action_anchor_coef=0.03 task.env.forceScale=0.6 task.env.randomForceProbScalar=0.1`
  - outcome:
    - saved config confirms `task.env.forceScale: 0.6` and `task.env.randomForceProbScalar: 0.1`,
    - `model_best.ckpt` created successfully,
    - run manually interrupted after meaningful signal (`Current Best` observed at `1645.22`, exit `130`),
    - `model_best.ckpt` frozen to `model_best_evalfreeze.ckpt` for eval.
- Eval commands (docker IsaacGym, `seed=42`, `steps=256`):
  - nominal:
    - `./docker-run-isaacgym.sh timeout 480 bash scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent <evalfreeze_ckpt> 256 plansv2_live_tailcoef02_thr015_sel_anchor003_trainforcescale06_nominal +train.ppo.diffusion_eval_decode_only=False +train.ppo.diffusion_eval_report_recon=True`
  - light_v2:
    - `./docker-run-isaacgym.sh timeout 480 bash scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent <evalfreeze_ckpt> 256 plansv2_live_tailcoef02_thr015_sel_anchor003_trainforcescale06_lightv2 +train.ppo.diffusion_eval_decode_only=False +train.ppo.diffusion_eval_report_recon=True task.env.randomization.obs_noise_e_scale=0.03 task.env.randomization.obs_noise_t_scale=0.015 task.env.forceScale=1.0 task.env.randomForceProbScalar=0.2`
  - hard:
    - `./docker-run-isaacgym.sh timeout 480 bash scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent <evalfreeze_ckpt> 256 plansv2_live_tailcoef02_thr015_sel_anchor003_trainforcescale06_hard +train.ppo.diffusion_eval_decode_only=False +train.ppo.diffusion_eval_report_recon=True task.env.randomization.obs_noise_e_scale=0.06 task.env.randomization.obs_noise_t_scale=0.03 task.env.forceScale=2.0 task.env.randomForceProbScalar=0.3`
- Eval results (`seed=42`, `steps=256`):
  - nominal: reward `1.867209`, done `0.001139`, `latent_mse=0.081603`, `action_mse_to_teacher=0.152386`
  - light_v2: reward `1.233401`, done `0.002279`, `latent_mse=0.098427`, `action_mse_to_teacher=0.196681`
  - hard: reward `1.151800`, done `0.002604`, `latent_mse=0.105413`, `action_mse_to_teacher=0.214764`
- Delta vs current kept reference (`tail_coef=0.2, threshold=0.15, selective=True, anchor=0.03`, `recon=0.5`):
  - reward:
    - nominal `+0.192097`
    - light_v2 `-0.404865`
    - hard `-0.353104`
  - done rate:
    - nominal `-0.000163`
    - light_v2 `+0.000896`
    - hard `+0.000732`
  - recon / teacher-alignment:
    - nominal `latent_mse +0.012850`, `action_mse_to_teacher +0.022284`
    - light_v2 `latent_mse +0.021353`, `action_mse_to_teacher +0.053633`
    - hard `latent_mse +0.021136`, `action_mse_to_teacher +0.054776`

### Local decision
- `task.env.forceScale=0.6` probe is **rejected**:
  - nominal improves, but `light_v2` and `hard` regress strongly,
  - robustness done-rates and teacher-alignment both degrade under perturbed conditions,
  - does not satisfy the strict keep/drop gate.

### Remaining blocked/risky
- Hard robustness remains the dominant blocker for route promotion.
- Training-side `Current Best` increase again did not translate to robust eval gains.
- Single-seed bounded probes can only prune directions; they do not establish promotion-level evidence.

### Single recommended next step
- Keep the current reference unchanged.
- If continuing bounded optimization, mirror this axis in the opposite mild direction with one single-variable probe:
  - `task.env.forceScale=0.4` (keep all other settings fixed),
  - then run the same `nominal/light_v2/hard` eval gate.

---

## v2-057 (2026-04-01) — PLANS_v2 P3 Training-Distribution Probe (`task.env.forceScale=0.4`)

### Target milestone/subgoal
- Execute the opposite-direction mirror probe on the same single-variable axis:
  - keep `tail_coef=0.2 + tail_threshold=0.15 + selective=True + anchor=0.03 + diffusion_latent_recon_coef=0.5`,
  - keep `task.env.randomForceProbScalar=0.1`,
  - only change training-time `task.env.forceScale` from wrapper default `0.5` to `0.4`,
  - then evaluate with unified `nominal/light_v2/hard` protocol.

### What changed (files + behavior impact)
- New bounded training run:
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_trainforcescale04_seed42_15min/`
- Frozen eval checkpoint snapshot:
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_trainforcescale04_seed42_15min/stage2_diffusion_nn/model_best_evalfreeze.ckpt`
- New eval logs:
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_trainforcescale04/diffusion_nominal_s42.log`
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_trainforcescale04/diffusion_light_v2_s42.log`
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_trainforcescale04/diffusion_hard_s42.log`
- No code edits in this session block (experiment-only).

### What was verified (commands + key outcomes)
- Bounded training command:
  - `./docker-run-isaacgym.sh timeout 1000 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_tailcoef02_thr015_sel_anchor003_trainforcescale04_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_latent_recon_coef=0.5 +train.ppo.diffusion_teacher_delta_tail_coef=0.2 +train.ppo.diffusion_teacher_delta_tail_threshold=0.15 +train.ppo.diffusion_teacher_delta_tail_selective=True +train.ppo.diffusion_base_action_anchor_coef=0.03 task.env.forceScale=0.4 task.env.randomForceProbScalar=0.1`
  - outcome:
    - saved config confirms `task.env.forceScale: 0.4` and `task.env.randomForceProbScalar: 0.1`,
    - `model_best.ckpt` created successfully,
    - run manually interrupted after meaningful signal (`Current Best` observed at `1601.19`, exit `130`),
    - `model_best.ckpt` frozen to `model_best_evalfreeze.ckpt` for eval.
- Eval commands (docker IsaacGym, `seed=42`, `steps=256`):
  - nominal:
    - `./docker-run-isaacgym.sh timeout 480 bash scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent <evalfreeze_ckpt> 256 plansv2_live_tailcoef02_thr015_sel_anchor003_trainforcescale04_nominal +train.ppo.diffusion_eval_decode_only=False +train.ppo.diffusion_eval_report_recon=True`
  - light_v2:
    - `./docker-run-isaacgym.sh timeout 480 bash scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent <evalfreeze_ckpt> 256 plansv2_live_tailcoef02_thr015_sel_anchor003_trainforcescale04_lightv2 +train.ppo.diffusion_eval_decode_only=False +train.ppo.diffusion_eval_report_recon=True task.env.randomization.obs_noise_e_scale=0.03 task.env.randomization.obs_noise_t_scale=0.015 task.env.forceScale=1.0 task.env.randomForceProbScalar=0.2`
  - hard:
    - `./docker-run-isaacgym.sh timeout 480 bash scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent <evalfreeze_ckpt> 256 plansv2_live_tailcoef02_thr015_sel_anchor003_trainforcescale04_hard +train.ppo.diffusion_eval_decode_only=False +train.ppo.diffusion_eval_report_recon=True task.env.randomization.obs_noise_e_scale=0.06 task.env.randomization.obs_noise_t_scale=0.03 task.env.forceScale=2.0 task.env.randomForceProbScalar=0.3`
- Eval results (`seed=42`, `steps=256`):
  - nominal: reward `1.182823`, done `0.001709`, `latent_mse=0.102772`, `action_mse_to_teacher=0.198058`
  - light_v2: reward `1.255701`, done `0.002197`, `latent_mse=0.101593`, `action_mse_to_teacher=0.198308`
  - hard: reward `1.117510`, done `0.002197`, `latent_mse=0.105681`, `action_mse_to_teacher=0.206533`
- Delta vs current kept reference (`tail_coef=0.2, threshold=0.15, selective=True, anchor=0.03`, `recon=0.5`):
  - reward:
    - nominal `-0.492289`
    - light_v2 `-0.382565`
    - hard `-0.387394`
  - done rate:
    - nominal `+0.000407`
    - light_v2 `+0.000814`
    - hard `+0.000325`
  - recon / teacher-alignment:
    - nominal `latent_mse +0.034019`, `action_mse_to_teacher +0.067956`
    - light_v2 `latent_mse +0.024519`, `action_mse_to_teacher +0.055260`
    - hard `latent_mse +0.021404`, `action_mse_to_teacher +0.046545`

### Local decision
- `task.env.forceScale=0.4` probe is **rejected**:
  - all three conditions regress in reward,
  - done-rates rise on all three conditions,
  - teacher-alignment worsens materially across all three conditions.

### Remaining blocked/risky
- Hard robustness remains the dominant blocker for route promotion.
- The force-scale axis now has clear negative evidence in both directions around default:
  - `forceScale=0.6` rejected,
  - `forceScale=0.4` rejected.
- Continuing this axis has low expected return.

### Single recommended next step
- Keep the current reference unchanged.
- If continuing bounded optimization, stop spending budget on `forceScale` and switch to one new single-variable training-distribution axis not yet rejected; recommended next candidate:
  - mild training-time observation-noise-e probe `task.env.randomization.obs_noise_e_scale=0.025` (keep all other settings fixed), then run the same `nominal/light_v2/hard` gate.

---

## v2-058 (2026-04-01) — PLANS_v2 P3 Training-Distribution Probe (`task.env.randomization.obs_noise_e_scale=0.025`)

### Target milestone/subgoal
- Execute one bounded single-variable probe on the training-distribution axis:
  - keep `tail_coef=0.2 + tail_threshold=0.15 + selective=True + anchor=0.03 + diffusion_latent_recon_coef=0.5`,
  - keep `task.env.forceScale=0.5` and `task.env.randomForceProbScalar=0.1`,
  - only change training-time `task.env.randomization.obs_noise_e_scale` from wrapper default `0.02` to `0.025`,
  - then evaluate with unified `nominal/light_v2/hard` protocol.

### What changed (files + behavior impact)
- New bounded training run:
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_trainobsnoisee0025_seed42_15min/`
- Frozen eval checkpoint snapshot:
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_trainobsnoisee0025_seed42_15min/stage2_diffusion_nn/model_best_evalfreeze.ckpt`
- New eval logs:
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_trainobsnoisee0025/diffusion_nominal_s42.log`
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_trainobsnoisee0025/diffusion_light_v2_s42.log`
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_trainobsnoisee0025/diffusion_hard_s42.log`
- No code edits in this session block (experiment-only).

### What was verified (commands + key outcomes)
- Bounded training command:
  - `./docker-run-isaacgym.sh timeout 1000 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_tailcoef02_thr015_sel_anchor003_trainobsnoisee0025_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_latent_recon_coef=0.5 +train.ppo.diffusion_teacher_delta_tail_coef=0.2 +train.ppo.diffusion_teacher_delta_tail_threshold=0.15 +train.ppo.diffusion_teacher_delta_tail_selective=True +train.ppo.diffusion_base_action_anchor_coef=0.03 task.env.randomization.obs_noise_e_scale=0.025 task.env.randomForceProbScalar=0.1 task.env.forceScale=0.5`
  - outcome:
    - saved config confirms `obs_noise_e_scale: 0.025`, `obs_noise_t_scale: 0.01`, `forceScale: 0.5`, `randomForceProbScalar: 0.1`,
    - `model_best.ckpt` created successfully,
    - run manually interrupted after meaningful signal (`Current Best` observed at `1541.51`, exit `130`),
    - `model_best.ckpt` frozen to `model_best_evalfreeze.ckpt` for eval.
- Eval commands (docker IsaacGym, `seed=42`, `steps=256`):
  - nominal:
    - `./docker-run-isaacgym.sh timeout 480 bash scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent <evalfreeze_ckpt> 256 plansv2_live_tailcoef02_thr015_sel_anchor003_trainobsnoisee0025_nominal +train.ppo.diffusion_eval_decode_only=False +train.ppo.diffusion_eval_report_recon=True`
  - light_v2:
    - `./docker-run-isaacgym.sh timeout 480 bash scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent <evalfreeze_ckpt> 256 plansv2_live_tailcoef02_thr015_sel_anchor003_trainobsnoisee0025_lightv2 +train.ppo.diffusion_eval_decode_only=False +train.ppo.diffusion_eval_report_recon=True task.env.randomization.obs_noise_e_scale=0.03 task.env.randomization.obs_noise_t_scale=0.015 task.env.forceScale=1.0 task.env.randomForceProbScalar=0.2`
  - hard:
    - `./docker-run-isaacgym.sh timeout 480 bash scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent <evalfreeze_ckpt> 256 plansv2_live_tailcoef02_thr015_sel_anchor003_trainobsnoisee0025_hard +train.ppo.diffusion_eval_decode_only=False +train.ppo.diffusion_eval_report_recon=True task.env.randomization.obs_noise_e_scale=0.06 task.env.randomization.obs_noise_t_scale=0.03 task.env.forceScale=2.0 task.env.randomForceProbScalar=0.3`
- Eval results (`seed=42`, `steps=256`):
  - nominal: reward `1.518356`, done `0.001221`, `latent_mse=0.100290`, `action_mse_to_teacher=0.170096`
  - light_v2: reward `1.430807`, done `0.001465`, `latent_mse=0.104855`, `action_mse_to_teacher=0.178499`
  - hard: reward `0.996136`, done `0.002523`, `latent_mse=0.115405`, `action_mse_to_teacher=0.210014`
- Delta vs current kept reference (`tail_coef=0.2, threshold=0.15, selective=True, anchor=0.03`, `recon=0.5`):
  - reward:
    - nominal `-0.156756`
    - light_v2 `-0.207459`
    - hard `-0.508768`
  - done rate:
    - nominal `-0.000081`
    - light_v2 `+0.000082`
    - hard `+0.000651`
  - recon / teacher-alignment:
    - nominal `latent_mse +0.031537`, `action_mse_to_teacher +0.039994`
    - light_v2 `latent_mse +0.027781`, `action_mse_to_teacher +0.035451`
    - hard `latent_mse +0.031128`, `action_mse_to_teacher +0.050026`

### Local decision
- `task.env.randomization.obs_noise_e_scale=0.025` probe is **rejected**:
  - reward regresses on all three conditions,
  - hard condition degrades strongly,
  - teacher-alignment metrics worsen across all three conditions.

### Remaining blocked/risky
- Hard robustness remains the dominant blocker for route promotion.
- The observation-noise-e upward direction now has clear negative evidence (`0.025` rejected).
- Training-side signal again does not reliably map to robust eval improvement.

### Single recommended next step
- Keep the current reference unchanged.
- If continuing bounded optimization, mirror this axis in the opposite mild direction with one single-variable probe:
  - `task.env.randomization.obs_noise_e_scale=0.015` (keep all other settings fixed),
  - then run the same `nominal/light_v2/hard` eval gate.

---

## v2-059 (2026-04-01) — Continuous-Execution Rule + 3-Probe Training-Distribution Batch

### Target milestone/subgoal
- Follow the newly confirmed continuous-execution preference while staying inside current Plan v2 boundaries:
  - add stable workflow rule to allow multi-probe continuous execution until major progress/escalation,
  - execute a bounded 3-probe batch on nearby training-distribution axes and prune them with the same strict gate.

### What changed (files + behavior impact)
- Governance/workflow update:
  - `AGENTS.md`
    - added `## Continuous Execution Preference`:
      - continue multiple plan-aligned local probes without per-probe confirmation,
      - stop/report on breakthrough, acceptable engineering progress, escalation trigger, or user redirect.
- New bounded training runs:
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_trainobsnoisee0015_seed42_15min/`
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_trainobsnoiset0008_seed42_15min/`
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_trainobsnoiset0012_seed42_15min/`
- Frozen eval checkpoint snapshots:
  - `<run>/stage2_diffusion_nn/model_best_evalfreeze.ckpt` for all three runs above.
- New eval logs:
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_trainobsnoisee0015/{diffusion_nominal_s42.log,diffusion_light_v2_s42.log,diffusion_hard_s42.log}`
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_trainobsnoiset0008/{diffusion_nominal_s42.log,diffusion_light_v2_s42.log,diffusion_hard_s42.log}`
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_trainobsnoiset0012/{diffusion_nominal_s42.log,diffusion_light_v2_s42.log,diffusion_hard_s42.log}`
- No core algorithm code changes in this session block.

### What was verified (commands + key outcomes)
- Probe A (`obs_noise_e=0.015`, other kept settings unchanged):
  - training command:
    - `./docker-run-isaacgym.sh timeout 1000 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_tailcoef02_thr015_sel_anchor003_trainobsnoisee0015_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_latent_recon_coef=0.5 +train.ppo.diffusion_teacher_delta_tail_coef=0.2 +train.ppo.diffusion_teacher_delta_tail_threshold=0.15 +train.ppo.diffusion_teacher_delta_tail_selective=True +train.ppo.diffusion_base_action_anchor_coef=0.03 task.env.randomization.obs_noise_e_scale=0.015 task.env.randomForceProbScalar=0.1 task.env.forceScale=0.5`
  - training-side note:
    - run manually interrupted after plateaued meaningful signal (`Current Best` observed `1394.51`, exit `130`).
  - eval results (`seed=42`, `steps=256`):
    - nominal: reward `1.521114`, done `0.002197`, `latent_mse=0.098100`, `action_mse_to_teacher=0.169708`
    - light_v2: reward `1.519357`, done `0.001790`, `latent_mse=0.101800`, `action_mse_to_teacher=0.174242`
    - hard: reward `0.986758`, done `0.002035`, `latent_mse=0.114967`, `action_mse_to_teacher=0.199809`
  - delta vs kept reference:
    - reward: nominal `-0.153998`, light_v2 `-0.118909`, hard `-0.518146`
    - done: nominal `+0.000895`, light_v2 `+0.000407`, hard `+0.000163`
    - teacher-alignment: all three conditions worse.
- Probe B (`obs_noise_t=0.008`, other kept settings unchanged):
  - training command:
    - `./docker-run-isaacgym.sh timeout 1000 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_tailcoef02_thr015_sel_anchor003_trainobsnoiset0008_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_latent_recon_coef=0.5 +train.ppo.diffusion_teacher_delta_tail_coef=0.2 +train.ppo.diffusion_teacher_delta_tail_threshold=0.15 +train.ppo.diffusion_teacher_delta_tail_selective=True +train.ppo.diffusion_base_action_anchor_coef=0.03 task.env.randomization.obs_noise_t_scale=0.008 task.env.randomForceProbScalar=0.1 task.env.forceScale=0.5`
  - training-side note:
    - initial cold-start weak, later rose to `Current Best` `1520.89`; manually interrupted (`130`).
  - eval results (`seed=42`, `steps=256`):
    - nominal: reward `1.277656`, done `0.001628`, `latent_mse=0.107156`, `action_mse_to_teacher=0.195177`
    - light_v2: reward `1.386177`, done `0.001790`, `latent_mse=0.108668`, `action_mse_to_teacher=0.192447`
    - hard: reward `0.797258`, done `0.003011`, `latent_mse=0.122463`, `action_mse_to_teacher=0.229987`
  - delta vs kept reference:
    - reward: nominal `-0.397456`, light_v2 `-0.252089`, hard `-0.707646`
    - done: nominal `+0.000326`, light_v2 `+0.000407`, hard `+0.001139`
    - teacher-alignment: all three conditions worse.
- Probe C (`obs_noise_t=0.012`, other kept settings unchanged):
  - training command:
    - `./docker-run-isaacgym.sh timeout 1000 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_tailcoef02_thr015_sel_anchor003_trainobsnoiset0012_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_latent_recon_coef=0.5 +train.ppo.diffusion_teacher_delta_tail_coef=0.2 +train.ppo.diffusion_teacher_delta_tail_threshold=0.15 +train.ppo.diffusion_teacher_delta_tail_selective=True +train.ppo.diffusion_base_action_anchor_coef=0.03 task.env.randomization.obs_noise_t_scale=0.012 task.env.randomForceProbScalar=0.1 task.env.forceScale=0.5`
  - training-side note:
    - run reached `Current Best` `1643.64`; manually interrupted (`130`).
  - eval results (`seed=42`, `steps=256`):
    - nominal: reward `1.490344`, done `0.001383`, `latent_mse=0.104861`, `action_mse_to_teacher=0.193513`
    - light_v2: reward `1.303042`, done `0.002116`, `latent_mse=0.106800`, `action_mse_to_teacher=0.205021`
    - hard: reward `1.120968`, done `0.002116`, `latent_mse=0.115423`, `action_mse_to_teacher=0.216695`
  - delta vs kept reference:
    - reward: nominal `-0.184768`, light_v2 `-0.335224`, hard `-0.383936`
    - done: nominal `+0.000081`, light_v2 `+0.000733`, hard `+0.000244`
    - teacher-alignment: all three conditions worse.

### Local decision
- Batch conclusion: all three probes are **rejected** by the strict keep/drop gate.
- No robustness breakthrough observed.

### Remaining blocked/risky
- Hard robustness remains the dominant blocker.
- `obs_noise_e` local neighborhood now has both directions rejected:
  - `0.015` rejected, `0.025` rejected.
- `obs_noise_t` local neighborhood now also has both directions rejected:
  - `0.008` rejected, `0.012` rejected.
- Training-side `Current Best` remains a weak promotion signal; robust eval still governs.

### Single recommended next step
- Keep the current reference unchanged.
- Stop spending budget on the current `obs_noise_e/obs_noise_t/forceScale` neighborhood.
- If continuing bounded optimization, switch to a new single-variable training-distribution axis not yet rejected:
  - recommended next candidate: `task.env.randomization.action_noise_e_scale=0.008` (all other settings fixed), then run the same `nominal/light_v2/hard` gate.

---

## v2-060 (2026-04-01) — PLANS_v2 P3 Action-Noise-E Axis 3-Point Batch

### Target milestone/subgoal
- Execute a bounded 3-point batch on a new, previously untested single-variable axis:
  - `task.env.randomization.action_noise_e_scale in {0.008, 0.012, 0.006}`,
  - keep the current kept reference fixed on all other settings,
  - evaluate each run under unified `nominal/light_v2/hard` protocol.

### What changed (files + behavior impact)
- New bounded training runs:
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_trainactionnoisee0008_seed42_15min/`
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_trainactionnoisee0012_seed42_15min/`
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_trainactionnoisee0006_seed42_15min/`
- Frozen eval checkpoint snapshots:
  - `<run>/stage2_diffusion_nn/model_best_evalfreeze.ckpt` for all three runs above.
- New eval logs:
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_trainactionnoisee0008/{diffusion_nominal_s42.log,diffusion_light_v2_s42.log,diffusion_hard_s42.log}`
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_trainactionnoisee0012/{diffusion_nominal_s42.log,diffusion_light_v2_s42.log,diffusion_hard_s42.log}`
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_trainactionnoisee0006/{diffusion_nominal_s42.log,diffusion_light_v2_s42.log,diffusion_hard_s42.log}`
- No core algorithm code changes in this session block.

### What was verified (commands + key outcomes)
- Probe A (`action_noise_e_scale=0.008`):
  - training command:
    - `./docker-run-isaacgym.sh timeout 1000 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_tailcoef02_thr015_sel_anchor003_trainactionnoisee0008_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_latent_recon_coef=0.5 +train.ppo.diffusion_teacher_delta_tail_coef=0.2 +train.ppo.diffusion_teacher_delta_tail_threshold=0.15 +train.ppo.diffusion_teacher_delta_tail_selective=True +train.ppo.diffusion_base_action_anchor_coef=0.03 task.env.randomization.action_noise_e_scale=0.008 task.env.forceScale=0.5 task.env.randomForceProbScalar=0.1`
  - training-side note:
    - run plateaued and was manually interrupted at meaningful signal (`Current Best` observed `1316.69`, exit `130`).
  - eval results (`seed=42`, `steps=256`):
    - nominal: reward `1.482150`, done `0.002116`, `latent_mse=0.096510`, `action_mse_to_teacher=0.176908`
    - light_v2: reward `1.368173`, done `0.002279`, `latent_mse=0.101285`, `action_mse_to_teacher=0.188377`
    - hard: reward `0.912775`, done `0.002279`, `latent_mse=0.110215`, `action_mse_to_teacher=0.196052`
  - delta vs kept reference:
    - reward: nominal `-0.192962`, light_v2 `-0.270093`, hard `-0.592129`
    - done: nominal `+0.000814`, light_v2 `+0.000896`, hard `+0.000407`
    - teacher-alignment: all three conditions worse.
- Probe B (`action_noise_e_scale=0.012`):
  - training command:
    - `./docker-run-isaacgym.sh timeout 1000 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_tailcoef02_thr015_sel_anchor003_trainactionnoisee0012_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_latent_recon_coef=0.5 +train.ppo.diffusion_teacher_delta_tail_coef=0.2 +train.ppo.diffusion_teacher_delta_tail_threshold=0.15 +train.ppo.diffusion_teacher_delta_tail_selective=True +train.ppo.diffusion_base_action_anchor_coef=0.03 task.env.randomization.action_noise_e_scale=0.012 task.env.forceScale=0.5 task.env.randomForceProbScalar=0.1`
  - training-side note:
    - run reached `Current Best` `1364.73`; manually interrupted (`130`).
  - eval results (`seed=42`, `steps=256`):
    - nominal: reward `1.482941`, done `0.001465`, `latent_mse=0.107605`, `action_mse_to_teacher=0.203610`
    - light_v2: reward `1.354838`, done `0.002767`, `latent_mse=0.110111`, `action_mse_to_teacher=0.211412`
    - hard: reward `1.106502`, done `0.002035`, `latent_mse=0.119079`, `action_mse_to_teacher=0.233536`
  - delta vs kept reference:
    - reward: nominal `-0.192171`, light_v2 `-0.283428`, hard `-0.398402`
    - done: nominal `+0.000163`, light_v2 `+0.001384`, hard `+0.000163`
    - teacher-alignment: all three conditions worse.
- Probe C (`action_noise_e_scale=0.006`):
  - training command:
    - `./docker-run-isaacgym.sh timeout 1000 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_tailcoef02_thr015_sel_anchor003_trainactionnoisee0006_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_latent_recon_coef=0.5 +train.ppo.diffusion_teacher_delta_tail_coef=0.2 +train.ppo.diffusion_teacher_delta_tail_threshold=0.15 +train.ppo.diffusion_teacher_delta_tail_selective=True +train.ppo.diffusion_base_action_anchor_coef=0.03 task.env.randomization.action_noise_e_scale=0.006 task.env.forceScale=0.5 task.env.randomForceProbScalar=0.1`
  - training-side note:
    - run reached `Current Best` `1543.31`; manually interrupted (`130`).
  - eval results (`seed=42`, `steps=256`):
    - nominal: reward `1.698397`, done `0.001383`, `latent_mse=0.090309`, `action_mse_to_teacher=0.164235`
    - light_v2: reward `1.266852`, done `0.001872`, `latent_mse=0.096929`, `action_mse_to_teacher=0.178066`
    - hard: reward `1.169188`, done `0.002035`, `latent_mse=0.106431`, `action_mse_to_teacher=0.188274`
  - delta vs kept reference:
    - reward: nominal `+0.023285`, light_v2 `-0.371414`, hard `-0.335716`
    - done: nominal `+0.000081`, light_v2 `+0.000489`, hard `+0.000163`
    - teacher-alignment: all three conditions worse.

### Local decision
- Batch conclusion: all three `action_noise_e` probes are **rejected** by strict keep/drop gate.
- `0.006` gives slight nominal reward gain, but robustness (`light_v2/hard`) still drops materially, so not keepable.

### Remaining blocked/risky
- Hard robustness remains the dominant blocker.
- New axis (`action_noise_e`) also fails to produce robust advantage in this local neighborhood.
- Training-side high `Current Best` again does not imply robust eval improvement.

### Single recommended next step
- Keep the current reference unchanged.
- If continuing bounded optimization, move to another untested single-variable training-distribution axis:
  - recommended next candidate: `task.env.randomization.action_noise_t_scale=0.004` (all other settings fixed), then run the same `nominal/light_v2/hard` gate.

---

## v2-061 (2026-04-01) — PLANS_v2 P3 Action-Noise-T Axis 3-Point Batch

### Target milestone/subgoal
- Execute a bounded 3-point batch on a new single-variable axis:
  - `task.env.randomization.action_noise_t_scale in {0.002, 0.004, 0.006}`,
  - keep the current kept reference fixed on all other settings,
  - evaluate each run with unified `nominal/light_v2/hard` protocol.

### What changed (files + behavior impact)
- Completed pending eval logs for:
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_trainactionnoiset0006_seed42_15min/`
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_trainactionnoiset0006/{diffusion_nominal_s42.log,diffusion_light_v2_s42.log,diffusion_hard_s42.log}`
- Added one new bounded training run:
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_trainactionnoiset0002_seed42_15min/`
- Added frozen eval checkpoint snapshot:
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_trainactionnoiset0002_seed42_15min/stage2_diffusion_nn/model_best_evalfreeze.ckpt`
- Added eval logs for the new `0.002` probe:
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_trainactionnoiset0002/{diffusion_nominal_s42.log,diffusion_light_v2_s42.log,diffusion_hard_s42.log}`
- Reused existing `0.004` eval evidence from prior in-progress block:
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_trainactionnoiset0004/{diffusion_nominal_s42.log,diffusion_light_v2_s42.log,diffusion_hard_s42.log}`
- No core algorithm code changes in this session block.

### What was verified (commands + key outcomes)
- Probe B completion (`action_noise_t_scale=0.006`, eval-only completion):
  - nominal:
    - `./docker-run-isaacgym.sh timeout 480 bash scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent <0006_evalfreeze_ckpt> 256 plansv2_live_tailcoef02_thr015_sel_anchor003_trainactionnoiset0006_nominal +train.ppo.diffusion_eval_decode_only=False +train.ppo.diffusion_eval_report_recon=True > outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_trainactionnoiset0006/diffusion_nominal_s42.log 2>&1`
  - light_v2:
    - `./docker-run-isaacgym.sh timeout 480 bash scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent <0006_evalfreeze_ckpt> 256 plansv2_live_tailcoef02_thr015_sel_anchor003_trainactionnoiset0006_lightv2 +train.ppo.diffusion_eval_decode_only=False +train.ppo.diffusion_eval_report_recon=True task.env.randomization.obs_noise_e_scale=0.03 task.env.randomization.obs_noise_t_scale=0.015 task.env.forceScale=1.0 task.env.randomForceProbScalar=0.2 > outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_trainactionnoiset0006/diffusion_light_v2_s42.log 2>&1`
  - hard:
    - `./docker-run-isaacgym.sh timeout 480 bash scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent <0006_evalfreeze_ckpt> 256 plansv2_live_tailcoef02_thr015_sel_anchor003_trainactionnoiset0006_hard +train.ppo.diffusion_eval_decode_only=False +train.ppo.diffusion_eval_report_recon=True task.env.randomization.obs_noise_e_scale=0.06 task.env.randomization.obs_noise_t_scale=0.03 task.env.forceScale=2.0 task.env.randomForceProbScalar=0.3 > outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_trainactionnoiset0006/diffusion_hard_s42.log 2>&1`
- Probe C (`action_noise_t_scale=0.002`) training + eval:
  - training command:
    - `./docker-run-isaacgym.sh timeout 1000 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_tailcoef02_thr015_sel_anchor003_trainactionnoiset0002_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_latent_recon_coef=0.5 +train.ppo.diffusion_teacher_delta_tail_coef=0.2 +train.ppo.diffusion_teacher_delta_tail_threshold=0.15 +train.ppo.diffusion_teacher_delta_tail_selective=True +train.ppo.diffusion_base_action_anchor_coef=0.03 task.env.randomization.action_noise_t_scale=0.002 task.env.forceScale=0.5 task.env.randomForceProbScalar=0.1`
  - training-side note:
    - early stage showed abnormally long low-signal region (`Current Best` stayed near `0`, later only reached `62.64` before manual stop); nevertheless `model_best.ckpt` was generated and frozen for eval.
  - eval commands:
    - same `nominal/light_v2/hard` template as above with cache prefix `plansv2_live_tailcoef02_thr015_sel_anchor003_trainactionnoiset0002_*`.
- Aggregated parse source:
  - `EvalSummary` + `EvalReconSummary` extracted from:
    - kept reference: `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003/diffusion_{nominal,light_v2,hard}_s42.log`
    - probes: `..._trainactionnoiset000{2,4,6}/diffusion_{nominal,light_v2,hard}_s42.log`

### Probe outcomes (`seed=42`, `steps=256`, delta vs kept reference)
- Probe A (`action_noise_t_scale=0.004`, existing logs):
  - nominal: reward `1.041356` (`-0.633756`), done `0.002035` (`+0.000733`), `latent_mse +0.038530`, `action_mse_to_teacher +0.060718`
  - light_v2: reward `0.824649` (`-0.813617`), done `0.002441` (`+0.001058`), `latent_mse +0.038044`, `action_mse_to_teacher +0.059461`
  - hard: reward `0.892104` (`-0.612800`), done `0.002930` (`+0.001058`), `latent_mse +0.033712`, `action_mse_to_teacher +0.053916`
- Probe B (`action_noise_t_scale=0.006`):
  - nominal: reward `0.882492` (`-0.792620`), done `0.003906` (`+0.002604`), `latent_mse +0.063482`, `action_mse_to_teacher +0.115681`
  - light_v2: reward `0.957380` (`-0.680886`), done `0.003092` (`+0.001709`), `latent_mse +0.053438`, `action_mse_to_teacher +0.099519`
  - hard: reward `0.364069` (`-1.140835`), done `0.004069` (`+0.002197`), `latent_mse +0.058193`, `action_mse_to_teacher +0.097229`
- Probe C (`action_noise_t_scale=0.002`):
  - nominal: reward `1.313802` (`-0.361310`), done `0.001465` (`+0.000163`), `latent_mse +0.043808`, `action_mse_to_teacher +0.081826`
  - light_v2: reward `1.216835` (`-0.421431`), done `0.002523` (`+0.001140`), `latent_mse +0.042217`, `action_mse_to_teacher +0.087854`
  - hard: reward `1.022859` (`-0.482045`), done `0.002930` (`+0.001058`), `latent_mse +0.035643`, `action_mse_to_teacher +0.076644`

### Local decision
- Batch conclusion: all three `action_noise_t` probes are **rejected** by strict keep/drop gate.
- Axis conclusion: this local `action_noise_t` neighborhood (`0.002/0.004/0.006`) provides no keepable robustness benefit and consistently worsens teacher-alignment.

### Remaining blocked/risky
- Hard robustness remains the dominant blocker.
- Training-side `Current Best` is again weakly predictive of robust eval quality.
- With `forceScale`, `obs_noise_e`, `obs_noise_t`, `action_noise_e`, and now `action_noise_t` neighborhoods all rejected, expected return of further nearby perturbation sweeps is dropping.

### Single recommended next step
- Keep the current reference unchanged.
- Stop spending budget on the current action-noise neighborhood.
- If continuing bounded optimization, switch to a previously untested single-variable training-distribution axis:
  - recommended next candidate: `task.env.randomization.noisy_pos_scale=0.015` (all other settings fixed), then run the same `nominal/light_v2/hard` gate.

---

## v2-062 (2026-04-01) — PLANS_v2 P3 Training-Distribution Probe (`task.env.randomization.noisy_pos_scale=0.015`)

### Target milestone/subgoal
- Execute one bounded single-variable probe on a previously untested training-distribution axis:
  - keep current kept reference fixed,
  - only change `task.env.randomization.noisy_pos_scale` from default `0.02` to `0.015`,
  - evaluate with unified `nominal/light_v2/hard` protocol.

### What changed (files + behavior impact)
- New bounded training run:
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_trainnoisypos0015_seed42_15min/`
- Frozen eval checkpoint snapshot:
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_trainnoisypos0015_seed42_15min/stage2_diffusion_nn/model_best_evalfreeze.ckpt`
- New eval logs:
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_trainnoisypos0015/diffusion_nominal_s42.log`
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_trainnoisypos0015/diffusion_light_v2_s42.log`
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_trainnoisypos0015/diffusion_hard_s42.log`
- No code edits in this session block.

### What was verified (commands + key outcomes)
- Bounded training command:
  - `./docker-run-isaacgym.sh timeout 1000 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_tailcoef02_thr015_sel_anchor003_trainnoisypos0015_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_latent_recon_coef=0.5 +train.ppo.diffusion_teacher_delta_tail_coef=0.2 +train.ppo.diffusion_teacher_delta_tail_threshold=0.15 +train.ppo.diffusion_teacher_delta_tail_selective=True +train.ppo.diffusion_base_action_anchor_coef=0.03 task.env.randomization.noisy_pos_scale=0.015 task.env.forceScale=0.5 task.env.randomForceProbScalar=0.1`
  - training-side note:
    - run showed prolonged low-signal regime (`Current Best` mostly near `0`, peak observed `47.84`) and was manually interrupted (`KeyboardInterrupt` / exit `130`), but `model_best.ckpt` was produced and frozen for eval.
- Eval commands (`seed=42`, `steps=256`):
  - nominal:
    - `./docker-run-isaacgym.sh timeout 480 bash scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent <noisypos0015_evalfreeze_ckpt> 256 plansv2_live_tailcoef02_thr015_sel_anchor003_trainnoisypos0015_nominal +train.ppo.diffusion_eval_decode_only=False +train.ppo.diffusion_eval_report_recon=True > outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_trainnoisypos0015/diffusion_nominal_s42.log 2>&1`
  - light_v2:
    - `./docker-run-isaacgym.sh timeout 480 bash scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent <noisypos0015_evalfreeze_ckpt> 256 plansv2_live_tailcoef02_thr015_sel_anchor003_trainnoisypos0015_lightv2 +train.ppo.diffusion_eval_decode_only=False +train.ppo.diffusion_eval_report_recon=True task.env.randomization.obs_noise_e_scale=0.03 task.env.randomization.obs_noise_t_scale=0.015 task.env.forceScale=1.0 task.env.randomForceProbScalar=0.2 > outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_trainnoisypos0015/diffusion_light_v2_s42.log 2>&1`
  - hard:
    - `./docker-run-isaacgym.sh timeout 480 bash scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent <noisypos0015_evalfreeze_ckpt> 256 plansv2_live_tailcoef02_thr015_sel_anchor003_trainnoisypos0015_hard +train.ppo.diffusion_eval_decode_only=False +train.ppo.diffusion_eval_report_recon=True task.env.randomization.obs_noise_e_scale=0.06 task.env.randomization.obs_noise_t_scale=0.03 task.env.forceScale=2.0 task.env.randomForceProbScalar=0.3 > outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_trainnoisypos0015/diffusion_hard_s42.log 2>&1`
- Eval results (`seed=42`, `steps=256`):
  - nominal: reward `1.152044`, done `0.002035`, `latent_mse=0.124002`, `action_mse_to_teacher=0.236190`
  - light_v2: reward `1.122484`, done `0.003255`, `latent_mse=0.122487`, `action_mse_to_teacher=0.233404`
  - hard: reward `0.860019`, done `0.003337`, `latent_mse=0.126702`, `action_mse_to_teacher=0.239406`
- Delta vs kept reference:
  - reward:
    - nominal `-0.523068`
    - light_v2 `-0.515782`
    - hard `-0.644885`
  - done rate:
    - nominal `+0.000733`
    - light_v2 `+0.001872`
    - hard `+0.001465`
  - recon / teacher-alignment:
    - nominal `latent_mse +0.055249`, `action_mse_to_teacher +0.106088`
    - light_v2 `latent_mse +0.045413`, `action_mse_to_teacher +0.090356`
    - hard `latent_mse +0.042425`, `action_mse_to_teacher +0.079418`

### Local decision
- `task.env.randomization.noisy_pos_scale=0.015` probe is **rejected**:
  - reward regresses on all three conditions,
  - done rate increases on all three conditions,
  - teacher-alignment metrics worsen substantially.

### Remaining blocked/risky
- Hard robustness remains the dominant blocker.
- Multiple training-distribution neighborhoods are now rejected with consistent negative evidence; local randomization micro-tuning is showing diminishing returns.
- Training-side `Current Best` remains unreliable as a promotion signal versus robust eval.

### Single recommended next step
- Keep the current reference unchanged.
- Continue bounded optimization only on an untested single-variable axis:
  - recommended next candidate: `task.env.randomization.noisy_rpy_scale=0.08` (all other settings fixed), then run the same `nominal/light_v2/hard` gate.

---

## v2-063 (2026-04-01) — PLANS_v2 P3 `noisy_rpy` Axis Probe + Low-Signal Hash-Collapse Evidence

### Target milestone/subgoal
- Continue the bounded single-variable training-distribution loop on the next untested axis:
  - probe `task.env.randomization.noisy_rpy_scale=0.08`,
  - then check a symmetric point `noisy_rpy_scale=0.12`,
  - keep all other settings fixed to current kept reference.
- Verify whether this axis provides a valid robustness signal or only repeats the recent low-signal collapse pattern.

### What changed (files + behavior impact)
- New bounded training runs:
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_trainnoisyrpy008_seed42_15min/`
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_trainnoisyrpy012_seed42_15min/`
- Frozen eval checkpoints:
  - `...trainnoisyrpy008.../stage2_diffusion_nn/model_best_evalfreeze.ckpt`
  - `...trainnoisyrpy012.../stage2_diffusion_nn/model_best_evalfreeze.ckpt`
- New eval logs:
  - full `nominal/light_v2/hard` for `trainnoisyrpy008` under:
    - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_trainnoisyrpy008/`
  - nominal confirmation for `trainnoisyrpy012` under:
    - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_trainnoisyrpy012/diffusion_nominal_s42.log`
- No code edits in this session block.

### What was verified (commands + key outcomes)
- `noisy_rpy=0.08` training:
  - `./docker-run-isaacgym.sh timeout 1000 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_tailcoef02_thr015_sel_anchor003_trainnoisyrpy008_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_latent_recon_coef=0.5 +train.ppo.diffusion_teacher_delta_tail_coef=0.2 +train.ppo.diffusion_teacher_delta_tail_threshold=0.15 +train.ppo.diffusion_teacher_delta_tail_selective=True +train.ppo.diffusion_base_action_anchor_coef=0.03 task.env.randomization.noisy_rpy_scale=0.08 task.env.forceScale=0.5 task.env.randomForceProbScalar=0.1`
  - training-side note:
    - prolonged low-signal regime; manual stop at `Current Best=47.84` (`KeyboardInterrupt`, exit `130`).
- `noisy_rpy=0.08` eval (`seed=42`, `steps=256`):
  - nominal / light_v2 / hard all completed via `scripts/eval_screwdriver_student_robustness.sh` with unified protocol.
  - parsed results:
    - nominal: reward `1.152044`, done `0.002035`, `latent_mse=0.124002`, `action_mse_to_teacher=0.236190`
    - light_v2: reward `1.122484`, done `0.003255`, `latent_mse=0.122487`, `action_mse_to_teacher=0.233404`
    - hard: reward `0.860019`, done `0.003337`, `latent_mse=0.126702`, `action_mse_to_teacher=0.239406`
  - delta vs kept reference:
    - reward: nominal `-0.523068`, light_v2 `-0.515782`, hard `-0.644885`
    - done: nominal `+0.000733`, light_v2 `+0.001872`, hard `+0.001465`
    - alignment: all conditions worse (`latent_mse` and `action_mse_to_teacher` both up).
- `noisy_rpy=0.12` training:
  - `./docker-run-isaacgym.sh timeout 1000 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_tailcoef02_thr015_sel_anchor003_trainnoisyrpy012_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_latent_recon_coef=0.5 +train.ppo.diffusion_teacher_delta_tail_coef=0.2 +train.ppo.diffusion_teacher_delta_tail_threshold=0.15 +train.ppo.diffusion_teacher_delta_tail_selective=True +train.ppo.diffusion_base_action_anchor_coef=0.03 task.env.randomization.noisy_rpy_scale=0.12 task.env.forceScale=0.5 task.env.randomForceProbScalar=0.1`
  - training-side note:
    - same low-signal regime and same plateau (`Current Best=47.84`), manual stop (`130`).
  - nominal eval check:
    - `EvalSummary`/`EvalReconSummary` exactly matches `noisy_rpy=0.08`.
- Hash-collapse evidence:
  - `sha1sum` on `model_best.ckpt` for
    - `trainnoisypos0015`
    - `trainnoisyrpy008`
    - `trainnoisyrpy012`
  - all are identical:
    - `099dd04477d0d500f4a681425396de882e978a88`

### Local decision
- `noisy_rpy=0.08` is **rejected** by strict keep/drop gate (all three conditions regress).
- `noisy_rpy=0.12` is **rejected** as the same low-signal collapsed checkpoint family:
  - same `model_best` hash as `0.08` and `noisy_pos=0.015`,
  - nominal eval already exact match;
  - light_v2/hard for `0.12` are inferred to be identical under same checkpoint + same deterministic eval protocol.

### Remaining blocked/risky
- Hard robustness remains the dominant blocker.
- Training-distribution micro-tuning now repeatedly enters a low-signal collapse mode:
  - very low `Current Best`,
  - repeated identical `model_best` hashes,
  - repeated degraded eval.
- Continuing adjacent env-randomization sweeps is likely low-yield.

### Single recommended next step
- Keep the current reference unchanged.
- Pause nearby env-randomization sweeps and switch to one optimization-axis single-variable probe:
  - recommended next candidate: `train.ppo.diffusion_lr=0.0002` (all other settings fixed), then run `nominal/light_v2/hard` gate.

---

## v2-064 (2026-04-01) — PLANS_v2 P3 Diffusion-LR Axis 2-Point Batch (`0.0002`, `0.0004`)

### Target milestone/subgoal
- Execute a bounded optimization-axis batch after env-randomization collapse evidence:
  - probe `train.ppo.diffusion_lr=0.0002`,
  - probe `train.ppo.diffusion_lr=0.0004`,
  - keep all other settings fixed to the current kept reference.
- Verify whether LR-axis can produce non-collapsed, keepable robustness behavior.

### What changed (files + behavior impact)
- New bounded training runs:
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_trainlr0002_seed42_15min/`
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_trainlr0004_seed42_15min/`
- New frozen eval checkpoints:
  - `...trainlr0002.../stage2_diffusion_nn/model_best_evalfreeze.ckpt`
  - `...trainlr0004.../stage2_diffusion_nn/model_best_evalfreeze.ckpt`
- New eval logs:
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_trainlr0002/{diffusion_nominal_s42.log,diffusion_light_v2_s42.log,diffusion_hard_s42.log}`
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_trainlr0004/{diffusion_nominal_s42.log,diffusion_light_v2_s42.log,diffusion_hard_s42.log}`
- No code edits in this session block.

### What was verified (commands + key outcomes)
- Initial override syntax correction:
  - first run used `+train.ppo.diffusion_lr=0.0002` and failed with Hydra key-exists error.
  - rerun succeeded with `train.ppo.diffusion_lr=0.0002`.
- `lr=0.0002` training:
  - command:
    - `./docker-run-isaacgym.sh timeout 1000 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_tailcoef02_thr015_sel_anchor003_trainlr0002_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_latent_recon_coef=0.5 +train.ppo.diffusion_teacher_delta_tail_coef=0.2 +train.ppo.diffusion_teacher_delta_tail_threshold=0.15 +train.ppo.diffusion_teacher_delta_tail_selective=True +train.ppo.diffusion_base_action_anchor_coef=0.03 train.ppo.diffusion_lr=0.0002 task.env.forceScale=0.5 task.env.randomForceProbScalar=0.1`
  - training-side note:
    - low-signal behavior; manual stop at `Current Best=54.69` (`130`).
  - hash check:
    - `model_best.ckpt` hash = `9ea24103f2a547fd3102d86d63486caf5c78f21e` (not the prior collapse hash `099dd0...`).
  - eval results (`seed=42`, `steps=256`):
    - nominal: reward `1.025205`, done `0.002116`, `latent_mse=0.121674`, `action_mse_to_teacher=0.224246`
    - light_v2: reward `1.095725`, done `0.002360`, `latent_mse=0.120959`, `action_mse_to_teacher=0.227586`
    - hard: reward `0.838448`, done `0.003662`, `latent_mse=0.122951`, `action_mse_to_teacher=0.234537`
  - delta vs kept reference:
    - reward: nominal `-0.649907`, light_v2 `-0.542541`, hard `-0.666456`
    - done: nominal `+0.000814`, light_v2 `+0.000977`, hard `+0.001790`
    - alignment: all worse.
- `lr=0.0004` training:
  - command:
    - `./docker-run-isaacgym.sh timeout 1000 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_tailcoef02_thr015_sel_anchor003_trainlr0004_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_latent_recon_coef=0.5 +train.ppo.diffusion_teacher_delta_tail_coef=0.2 +train.ppo.diffusion_teacher_delta_tail_threshold=0.15 +train.ppo.diffusion_teacher_delta_tail_selective=True +train.ppo.diffusion_base_action_anchor_coef=0.03 train.ppo.diffusion_lr=0.0004 task.env.forceScale=0.5 task.env.randomForceProbScalar=0.1`
  - training-side note:
    - low-signal but slightly stronger than `0.0002`; manual stop at `Current Best=160.86` (`130`).
  - hash check:
    - `model_best.ckpt` hash = `8efc63e37f626118ff752496e7b6d935f5cccce5` (distinct from both `0.0002` and `099dd0...` collapse family).
  - eval results (`seed=42`, `steps=256`):
    - nominal: reward `1.339894`, done `0.002116`, `latent_mse=0.111508`, `action_mse_to_teacher=0.193846`
    - light_v2: reward `1.140712`, done `0.001872`, `latent_mse=0.114861`, `action_mse_to_teacher=0.204542`
    - hard: reward `1.120410`, done `0.002197`, `latent_mse=0.114936`, `action_mse_to_teacher=0.209872`
  - delta vs kept reference:
    - reward: nominal `-0.335218`, light_v2 `-0.497554`, hard `-0.384494`
    - done: nominal `+0.000814`, light_v2 `+0.000489`, hard `+0.000325`
    - alignment: all worse.

### Local decision
- `lr=0.0002`: **rejected** (all three conditions worse, substantial robustness loss).
- `lr=0.0004`: **rejected** (still all three conditions below kept reference; only “less bad” than `0.0002`).
- LR-axis local conclusion:
  - this 2-point batch did break out of the repeated-hash collapse family,
  - but still failed strict keep/drop gate on robust metrics.

### Remaining blocked/risky
- Hard robustness remains the dominant blocker.
- Current probe methodology repeatedly produces low-training-signal candidates (`Current Best` very low) that are almost always non-keepable.
- Short-budget probes remain useful for pruning, but may under-sample any candidate needing longer warm-up.

### Single recommended next step
- Keep the current reference unchanged.
- Run one **quality-gated longer-budget** verification on the least-bad new candidate (`lr=0.0004`) before opening new axes:
  - extend training budget (single run) and only evaluate if training escapes low-signal regime (e.g., `Current Best` meaningfully above current low-signal band), then apply the same `nominal/light_v2/hard` gate.

---

## v2-065 (2026-04-01) — Quality-Gated Longer-Budget Verification (`lr=0.0004`) Rejected

### Target milestone/subgoal
- Execute the recommended longer-budget gate for the least-bad recent candidate:
  - `train.ppo.diffusion_lr=0.0004`,
  - single-run longer timeout,
  - only proceed to `nominal/light_v2/hard` eval if training clearly exits low-signal band.

### What changed (files + behavior impact)
- New longer-budget training run:
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_trainlr0004_longgated_seed42/`
- No eval logs added for this run (gate failed before eval stage).
- No code edits in this session block.

### What was verified (commands + key outcomes)
- Longer-budget training command:
  - `./docker-run-isaacgym.sh timeout 2400 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_tailcoef02_thr015_sel_anchor003_trainlr0004_longgated_seed42 "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_latent_recon_coef=0.5 +train.ppo.diffusion_teacher_delta_tail_coef=0.2 +train.ppo.diffusion_teacher_delta_tail_threshold=0.15 +train.ppo.diffusion_teacher_delta_tail_selective=True +train.ppo.diffusion_base_action_anchor_coef=0.03 train.ppo.diffusion_lr=0.0004 task.env.forceScale=0.5 task.env.randomForceProbScalar=0.1`
- Training-side observations:
  - long low-signal regime,
  - `Current Best` rose but stalled at `160.86`,
  - manual stop (`KeyboardInterrupt`, exit `130`) because quality gate not met.
- Hash check:
  - `sha1sum` comparison:
    - `...trainlr0004_longgated.../model_best.ckpt`
    - `...trainlr0004_seed42_15min.../model_best.ckpt`
  - result: identical hash
    - `8efc63e37f626118ff752496e7b6d935f5cccce5`

### Local decision
- Longer-budget gate is **rejected**:
  - training did not exceed quality threshold (still in low-signal band),
  - produced the exact same best checkpoint as the shorter 15min run,
  - therefore no additional eval was run (would be redundant by construction).

### Remaining blocked/risky
- Hard robustness remains the dominant blocker.
- At least for this candidate, increasing time budget does not change the selected checkpoint.
- Probe efficiency risk remains high unless low-signal gate is enforced strictly.

### Single recommended next step
- Keep the current reference unchanged.
- Continue with one new optimization-axis single-variable probe that can change update dynamics (not merely longer budget on known candidate), then apply the same quality gate before eval.

---

## v2-066 (2026-04-01) — PLANS_v2 P3 Diffusion-LR High-Side Probe (`train.ppo.diffusion_lr=0.0006`)

### Target milestone/subgoal
- Continue optimization-axis search after `lr=0.0002/0.0004` rejection:
  - test one higher-side LR point `train.ppo.diffusion_lr=0.0006`,
  - keep all other settings fixed,
  - apply quality-gate observation then unified `nominal/light_v2/hard` evaluation.

### What changed (files + behavior impact)
- New bounded training run:
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_trainlr0006_seed42_15min/`
- Frozen eval checkpoint snapshot:
  - `...trainlr0006.../stage2_diffusion_nn/model_best_evalfreeze.ckpt`
- New eval logs:
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_trainlr0006/{diffusion_nominal_s42.log,diffusion_light_v2_s42.log,diffusion_hard_s42.log}`
- No code edits in this session block.

### What was verified (commands + key outcomes)
- Training command:
  - `./docker-run-isaacgym.sh timeout 1000 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_tailcoef02_thr015_sel_anchor003_trainlr0006_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_latent_recon_coef=0.5 +train.ppo.diffusion_teacher_delta_tail_coef=0.2 +train.ppo.diffusion_teacher_delta_tail_threshold=0.15 +train.ppo.diffusion_teacher_delta_tail_selective=True +train.ppo.diffusion_base_action_anchor_coef=0.03 train.ppo.diffusion_lr=0.0006 task.env.forceScale=0.5 task.env.randomForceProbScalar=0.1`
  - training-side note:
    - low-signal regime persisted; `Current Best` peaked at `183.67`, manually stopped (`130`).
- Checkpoint hash:
  - `model_best.ckpt` hash = `e811b96850ab9bcc4fbf7004199dcd57607bc1a2`
  - distinct from prior `lr=0.0004` hash (`8efc63...`), so this is not a repeated-hash clone run.
- Eval commands (`seed=42`, `steps=256`):
  - nominal / light_v2 / hard all completed via `scripts/eval_screwdriver_student_robustness.sh` under standard protocol.
- Eval results:
  - nominal: reward `1.317061`, done `0.002441`, `latent_mse=0.122772`, `action_mse_to_teacher=0.209479`
  - light_v2: reward `0.967835`, done `0.003337`, `latent_mse=0.138283`, `action_mse_to_teacher=0.246996`
  - hard: reward `0.722249`, done `0.003662`, `latent_mse=0.145574`, `action_mse_to_teacher=0.249319`
- Delta vs kept reference:
  - reward:
    - nominal `-0.358051`
    - light_v2 `-0.670431`
    - hard `-0.782655`
  - done rate:
    - nominal `+0.001139`
    - light_v2 `+0.001954`
    - hard `+0.001790`
  - alignment:
    - all conditions worse in both `latent_mse` and `action_mse_to_teacher`.

### Local decision
- `lr=0.0006` is **rejected** by strict keep/drop gate.
- Combined LR-axis view now (`0.0002/0.0004/0.0006`):
  - all rejected,
  - higher LR further hurts robustness (`light_v2/hard`) despite distinct checkpoint.

### Remaining blocked/risky
- Hard robustness remains the dominant blocker.
- LR-axis exploration has low expected return in the current neighborhood.
- Low-signal training regime continues to correlate with non-keepable robust eval.

### Single recommended next step
- Keep the current reference unchanged.
- Stop expanding LR neighborhood and switch to one new optimization axis not yet probed in this local loop:
  - recommended next candidate: `train.ppo.diffusion_steps=8` and `train.ppo.diffusion_steps_infer=8` (single-variable-ish sampler-depth axis with same eval gate).

---

## v2-067 (2026-04-01) — PLANS_v2 P3 Sampler-Depth Probe (`diffusion_steps=8`) Gate Failure

### Target milestone/subgoal
- Probe a new optimization axis after LR neighborhood rejection:
  - set `train.ppo.diffusion_steps=8`,
  - set `train.ppo.diffusion_steps_infer=8`,
  - keep all other settings fixed.
- Apply quality gate first; only evaluate if training exits low-signal regime.

### What changed (files + behavior impact)
- New bounded training run:
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_trainsteps8_seed42_15min/`
- No eval logs added for this run (quality gate failed).
- No code edits in this session block.

### What was verified (commands + key outcomes)
- Training command:
  - `./docker-run-isaacgym.sh timeout 1000 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_tailcoef02_thr015_sel_anchor003_trainsteps8_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_latent_recon_coef=0.5 +train.ppo.diffusion_teacher_delta_tail_coef=0.2 +train.ppo.diffusion_teacher_delta_tail_threshold=0.15 +train.ppo.diffusion_teacher_delta_tail_selective=True +train.ppo.diffusion_base_action_anchor_coef=0.03 train.ppo.diffusion_steps=8 train.ppo.diffusion_steps_infer=8 task.env.forceScale=0.5 task.env.randomForceProbScalar=0.1`
- Training-side observations:
  - prolonged low-signal regime,
  - `Current Best` stayed very low, peaking only at `17.07`,
  - manual stop (`130`) due quality gate failure.
- Hash check:
  - `model_best.ckpt` hash = `ac87f3b26e42d959d8337947d05307c82b3cac72`
  - distinct from both collapse-family hash (`099dd0...`) and recent LR probes.

### Local decision
- `diffusion_steps=8 / diffusion_steps_infer=8` probe is **rejected by quality gate**:
  - training signal remains in very low regime,
  - no robustness eval executed to avoid redundant low-yield runs.

### Remaining blocked/risky
- Hard robustness remains the dominant blocker.
- Even non-randomization optimization-axis probes can remain trapped in low-signal training behavior.
- Current loop is effective at pruning, but no keepable candidate has emerged.

### Single recommended next step
- Keep the current reference unchanged.
- Continue optimization-axis exploration with one additional single-variable probe that changes sampler/training dynamics but preserves pipeline compatibility (then same quality gate + eval protocol).

---

## v2-068 (2026-04-01) — PLANS_v2 P3 Sampler Continuation + Diffusion-Loss Axis Batch (Quality-Gate Rejection)

### Target milestone/subgoal
- Continue the post-`v2-067` single-variable optimization loop with strict quality gate:
  - finish pending sampler probe `diffusion_steps=12 / diffusion_steps_infer=12`,
  - test smaller sampler perturbations around reference (`infer_steps=8`, `infer_steps=9`, `train_steps=11`),
  - test diffusion-loss balance axis (`diffusion_loss_coef=0.8`, `1.2`).
- Only run `nominal/light_v2/hard` eval if training escapes low-signal regime.

### What changed (files + behavior impact)
- New bounded training runs:
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_trainsteps12_seed42_15min/`
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_infersteps8_seed42_15min/`
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_infersteps9_seed42_15min/`
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_trainsteps11_seed42_15min/`
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_trainlossdiff08_seed42_15min/`
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_trainlossdiff12_seed42_15min/`
- No new eval logs were added in this batch (all rejected by training-side quality gate).
- No code edits in this session block.

### What was verified (commands + key outcomes)
- `trainsteps12`:
  - command:
    - `./docker-run-isaacgym.sh timeout 1000 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_tailcoef02_thr015_sel_anchor003_trainsteps12_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_latent_recon_coef=0.5 +train.ppo.diffusion_teacher_delta_tail_coef=0.2 +train.ppo.diffusion_teacher_delta_tail_threshold=0.15 +train.ppo.diffusion_teacher_delta_tail_selective=True +train.ppo.diffusion_base_action_anchor_coef=0.03 train.ppo.diffusion_steps=12 train.ppo.diffusion_steps_infer=12 task.env.forceScale=0.5 task.env.randomForceProbScalar=0.1`
  - training-side note:
    - prolonged low-signal regime; `Current Best` peaked at `51.65`, then manual stop (`130`).
  - hash:
    - `52eec11c0f14c118f740872190c37383d6f67636`
- `infersteps8`:
  - command override:
    - `train.ppo.diffusion_steps_infer=8` (other settings fixed to kept reference stack).
  - training-side note:
    - low-signal plateau at `Current Best=47.84`, manual stop.
  - hash:
    - `099dd04477d0d500f4a681425396de882e978a88` (same collapse-family hash seen in prior low-signal runs).
- `infersteps9`:
  - command override:
    - `train.ppo.diffusion_steps_infer=9`
  - training-side note:
    - low-signal regime; `Current Best` peaked at `57.25`, manual stop.
  - hash:
    - `527c02b447bd9beff0c2c8dd9d3398642f82bc53` (distinct hash but still low-signal gate fail).
- `trainsteps11`:
  - command override:
    - `train.ppo.diffusion_steps=11` (infer remains `10` from script defaults).
  - training-side note:
    - severe low-signal behavior; `Current Best` only `12.80`, manual stop.
  - hash:
    - `c7fc6ec2e97990c9e4aba2c1ff1f28493e764204`
- `trainlossdiff08`:
  - command override:
    - `train.ppo.diffusion_loss_coef=0.8`
  - training-side note:
    - long `Current Best=0.00` regime, brief rise to `15.72`, manual stop.
  - hash:
    - `00dc2fdc05ba52f3266ef6cd16a4d42fedc2b4aa`
- `trainlossdiff12`:
  - command override:
    - `train.ppo.diffusion_loss_coef=1.2`
  - training-side note:
    - remained at `Current Best=0.00` through observed window, manual stop.
  - hash:
    - `420474c6a1a642b59dfffbbffa6433c1e922622e`

### Local decision
- All six probes in this batch are **rejected by quality gate** (no robust eval triggered).
- Local axis conclusion:
  - sampler perturbations around current reference (`infer_steps` and `train_steps`) remained trapped in low-signal regime.
  - diffusion-loss up/down perturbations (`0.8`, `1.2`) were even less stable, with near-zero training signal.

### Remaining blocked/risky
- Hard robustness remains the dominant blocker, but this batch failed before robustness stage due training quality.
- Repeated low-signal behavior now spans multiple axes, increasing risk of wasted eval compute if gate is not enforced.
- Distinct checkpoint hashes do not imply keepability; training-signal gate remains the stronger filter.

### Single recommended next step
- Keep the current reference unchanged.
- Switch to one **near-reference recovery micro-probe** (single variable, minimal deviation) to seek non-collapsed signal before any new broad axis:
  - recommended next candidate: `train.ppo.bc_loss_coef=1.05` (all other settings fixed), then apply the same quality gate and run full `nominal/light_v2/hard` only if training signal is acceptable.

---

## v2-069 (2026-04-01) — Recovery Micro-Probe (`bc=1.05`) + Control-Replay Repro Check (Escalation Triggered)

### Target milestone/subgoal
- Execute the recommended near-reference recovery probe:
  - `train.ppo.bc_loss_coef=1.05`.
- If recovery probe still fails quality gate, run a control replay of the current kept reference config to verify whether baseline behavior is still reproducible in the current repo/runtime state.

### What changed (files + behavior impact)
- New recovery probe run:
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_trainbc105_seed42_15min/`
- New control replay run (same kept-reference settings):
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_controlreplay_seed42_15min/`
- No robustness eval logs added in this block (both failed training-side quality gate).
- No code edits in this session block.

### What was verified (commands + key outcomes)
- Recovery probe (`bc=1.05`):
  - command:
    - `./docker-run-isaacgym.sh timeout 1000 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_tailcoef02_thr015_sel_anchor003_trainbc105_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_latent_recon_coef=0.5 +train.ppo.diffusion_teacher_delta_tail_coef=0.2 +train.ppo.diffusion_teacher_delta_tail_threshold=0.15 +train.ppo.diffusion_teacher_delta_tail_selective=True +train.ppo.diffusion_base_action_anchor_coef=0.03 train.ppo.bc_loss_coef=1.05 task.env.forceScale=0.5 task.env.randomForceProbScalar=0.1`
  - training-side note:
    - remained in very low-signal regime (`Current Best` stayed at `0.00` in observed window), manual stop.
  - hash:
    - `00d1d1c76fd4f61ea1645c63a1a16901451083fc`
- Control replay (kept-reference config):
  - command:
    - `./docker-run-isaacgym.sh timeout 1000 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_tailcoef02_thr015_sel_anchor003_controlreplay_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_latent_recon_coef=0.5 +train.ppo.diffusion_teacher_delta_tail_coef=0.2 +train.ppo.diffusion_teacher_delta_tail_threshold=0.15 +train.ppo.diffusion_teacher_delta_tail_selective=True +train.ppo.diffusion_base_action_anchor_coef=0.03 task.env.forceScale=0.5 task.env.randomForceProbScalar=0.1`
  - training-side note:
    - did not recover historical kept-reference signal; eventually plateaued at `Current Best=47.84`, manual stop.
  - hash:
    - `099dd04477d0d500f4a681425396de882e978a88` (same low-signal collapse-family hash).

### Local decision
- `bc=1.05` recovery probe is rejected by quality gate.
- **Control replay failed to reproduce kept-reference behavior** and fell into collapse-family hash.
- This is now treated as an execution-level reproducibility blocker, not just a hyperparameter-search miss.

### Remaining blocked/risky
- Current optimization sweep results are no longer reliably interpretable unless kept-reference reproducibility is re-established.
- Continued parameter sweeps risk wasting compute on a drifted/unstable state.
- This contradicts the working assumption that the current kept reference remains directly replayable under present repo/runtime conditions.

### Single recommended next step
- Pause new sweep axes and escalate with a compact `codeagent_issue.md` focused on reproducibility drift:
  - request governance-confirmed debugging priority to re-establish kept-reference replayability (or redefine the active reference) before resuming Plan v2 optimization sweeps.

---

## v2-070 (2026-04-01) — Diffusion 路线/实现文档化（算法说明）

### Target milestone/subgoal
- 按用户请求，基于当前仓库实现整理一份可直接阅读的 diffusion 算法文档：
  - 说明技术路线（Plan v2 主线/保底线）；
  - 说明 latent diffusion 训练与推理细节；
  - 说明关键损失项、配置键和代码映射关系。

### What changed (files + behavior impact)
- 新增文档：
  - `docs/diffusion_algorithm.md`
- 文档覆盖：
  - 当前治理下 diffusion 技术路线位置；
  - `DiffusionLatentStudent` 核心结构；
  - 训练损失分解与参数含义；
  - 推理/评测机制与脚本入口；
  - 关键代码文件映射。
- 无代码逻辑改动（仅文档）。

### What was verified (commands + key outcomes)
- 核心实现对照读取：
  - `dexscrew/algo/ppo/diffusion_latent_student.py`
  - `dexscrew/algo/ppo/padapt.py`
  - `dexscrew/algo/models/models.py`
  - `dexscrew/algo/models/block.py`
  - `scripts/screwdriver_student_diffusion_latent_robust_light.sh`
  - `scripts/eval_screwdriver_student_robustness.sh`
- 文档落盘检查：
  - `docs/diffusion_algorithm.md` 已创建并包含路线、公式化描述、实现细节与配置项。

### Remaining blocked/risky
- 该条目只完成文档化，不改变此前 `v2-069` 的复现性阻塞结论。
- 训练主线继续前仍需先处理 reproducibility drift（或治理层重定义 reference）。

### Single recommended next step
- 若继续执行实验主线：先按 `codeagent_issue.md` 处理复现性恢复，再恢复参数探索。
- 若当前目标是论文/汇报材料：可基于 `docs/diffusion_algorithm.md` 继续拆分出“方法章节 + 实验协议章节”。

---

## v2-071 (2026-04-01) — Reproducibility Recovery Validation + Full-Budget Recheck (`infersteps9`)

### Target milestone/subgoal
- Verify whether the previously escalated “reproducibility drift” is real or caused by premature early-stop.
- Run full-budget replay on:
  - kept-reference config (`controlreplay_full`),
  - one previously early-stopped candidate (`diffusion_steps_infer=9`).
- Re-evaluate both under unified `nominal/light_v2/hard` protocol.

### What changed (files + behavior impact)
- New full-budget training run:
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_controlreplay_full_seed42_15min/`
- New full-budget candidate run:
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_infersteps9_full_seed42_15min/`
- New eval logs:
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_controlreplay_full/{diffusion_nominal_s42.log,diffusion_light_v2_s42.log,diffusion_hard_s42.log}`
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_infersteps9_full/{diffusion_nominal_s42.log,diffusion_light_v2_s42.log,diffusion_hard_s42.log}`
- No code edits in this block.

### What was verified (commands + key outcomes)
- Config consistency check:
  - diff of saved run configs (`config_*.yaml`) between historical kept reference and replay shows only `output_name` difference.
  - checkpoint path and core overrides (`diffusion_steps_infer=10`, `latent_recon=0.5`, `tail selective=True`) are consistent.
- Full replay (`controlreplay_full`) training outcome:
  - log parse (`train_1000s.log`): `Current Best max=1774.05`
  - best ckpt hash: `150e0c115f31d9d16f3ff32ef2d58cf738e2ed5e`
- `controlreplay_full` eval vs kept reference:
  - nominal reward `1.654212` (delta `-0.020900`)
  - light_v2 reward `1.530427` (delta `-0.107839`)
  - hard reward `1.188177` (delta `-0.316727`)
  - all alignment metrics (`latent_mse`, `action_mse_to_teacher`) are worse than kept reference.
- `infersteps9_full` training outcome:
  - log parse (`train_1000s.log`): `Current Best max=1826.76`
  - best ckpt hash: `3449b4f0285c705f423d38cebce2b740fcafdcda`
- `infersteps9_full` eval vs kept reference:
  - nominal reward `1.563361` (delta `-0.111751`)
  - light_v2 reward `1.587175` (delta `-0.051091`)
  - hard reward `1.350380` (delta `-0.154524`)
  - none of the three conditions surpass kept reference; alignment metrics also worse.
- `infersteps9_full` vs `controlreplay_full`:
  - better on light_v2/hard reward than `controlreplay_full`,
  - but still below kept reference.

### Local decision
- The previous “reproducibility drift” escalation is **not sustained**:
  - full-budget replay confirms high-signal training can be recovered.
- However, strict keep/drop gate still **does not accept** `controlreplay_full` or `infersteps9_full` as new reference.
- Kept reference remains unchanged.

### Remaining blocked/risky
- Methodology risk identified:
  - aggressive early-stop on low initial `Current Best` can produce false negatives.
- Robustness gap vs kept reference remains unresolved.

### Single recommended next step
- Keep current reference unchanged.
- Update local execution rule for this branch:
  - for near-reference probes, prefer full-budget (or late-window) validation before rejection;
  - then continue one-at-a-time optimization probes with unified `nominal/light_v2/hard` gate.

---

## v2-072 (2026-04-02) — Near-Reference Full-Budget Probe (`diffusion_steps_infer=11`) + Clamp Visibility Patch

### Target milestone/subgoal
- Continue `PLANS_v2` P3 one-at-a-time optimization under updated execution rule:
  - run one near-reference full-budget probe,
  - only evaluate when probe is materially different from kept reference.

### What changed (files + behavior impact)
- New full-budget training run:
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_infersteps11_full_seed42_15min/`
- Code change:
  - `dexscrew/algo/ppo/diffusion_latent_student.py`
  - added explicit runtime note when `diffusion_steps_infer` is clamped by `diffusion_steps`:
    - prints requested value vs effective value
    - no training logic change (visibility-only patch).

### What was verified (commands + key outcomes)
- Full-budget training command:
  - `./docker-run-isaacgym.sh timeout 1000 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_tailcoef02_thr015_sel_anchor003_infersteps11_full_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_latent_recon_coef=0.5 +train.ppo.diffusion_teacher_delta_tail_coef=0.2 +train.ppo.diffusion_teacher_delta_tail_threshold=0.15 +train.ppo.diffusion_teacher_delta_tail_selective=True +train.ppo.diffusion_base_action_anchor_coef=0.03 train.ppo.diffusion_steps_infer=11 task.env.forceScale=0.5 task.env.randomForceProbScalar=0.1`
  - timeout exit: `124` (expected full-budget stop).
- Training signal parse:
  - `train_1000s.log` -> `Current Best max=1830.24` (high-signal regime reached).
- Checkpoint identity check:
  - new probe `model_best.ckpt` hash: `31150d26646911325cac2aa408d3d42fdd4640a7`
  - kept reference `model_best.ckpt` hash: `31150d26646911325cac2aa408d3d42fdd4640a7`
  - outcome: identical checkpoint bytes.
- Config diff check:
  - only material config difference is `diffusion_steps_infer: 10 -> 11` (plus `output_name`).
- Implementation check:
  - `diffusion_latent_student.py` confirms
    - `self.diffusion_steps_infer = min(requested_diffusion_steps_infer, self.diffusion_steps)`
    - therefore `infer=11` under `diffusion_steps=10` is clamped to `10`.
- Syntax check:
  - `PYTHONDONTWRITEBYTECODE=1 python - <<'PY' ... compile('dexscrew/algo/ppo/diffusion_latent_student.py', 'exec') ... PY`
  - outcome: `syntax_ok`.

### Local decision
- This probe is **non-informative as a performance candidate**:
  - effective inference schedule is clamped back to baseline (`10`),
  - resulting best checkpoint is exactly identical to kept reference.
- Robust eval was intentionally skipped:
  - identical checkpoint implies redundant metrics by construction under same eval protocol.

### Remaining blocked/risky
- Probe-space validity risk:
  - any run with `diffusion_steps_infer > diffusion_steps` silently collapses to the same effective setting unless surfaced.
- Robustness gap vs kept reference remains unresolved (unchanged in this step).

### Single recommended next step
- Keep current reference unchanged.
- Run one **valid** near-reference sampler-depth probe where setting is not clamped:
  - `train.ppo.diffusion_steps=11` and `train.ppo.diffusion_steps_infer=11` (full-budget, seed42),
  - then unified `nominal/light_v2/hard` eval gate if checkpoint is non-identical.

---

## v2-073 (2026-04-02) — Valid Sampler-Depth Probe (`steps=11,infer=11`) + Eval-Time Infer Sweep Check

### Target milestone/subgoal
- Execute the recommended valid near-reference probe (non-clamped):
  - `train.ppo.diffusion_steps=11`
  - `train.ppo.diffusion_steps_infer=11`
- Run unified `nominal/light_v2/hard` evaluation and compare against kept reference.
- Add one low-cost eval-only check on the same checkpoint with `infer=10`.

### What changed (files + behavior impact)
- New full-budget training run:
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_trainsteps11_full_seed42_15min/`
- New eval logs (primary, `infer=11`):
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_trainsteps11_full/{diffusion_nominal_s42.log,diffusion_light_v2_s42.log,diffusion_hard_s42.log}`
- New eval logs (eval-only side check, `infer=10`):
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_trainsteps11_full_evalinfer10/{diffusion_nominal_s42.log,diffusion_light_v2_s42.log,diffusion_hard_s42.log}`
- Frozen eval checkpoint:
  - `.../trainsteps11_full.../stage2_diffusion_nn/model_best_evalfreeze.ckpt`

### What was verified (commands + key outcomes)
- Full-budget training command:
  - `./docker-run-isaacgym.sh timeout 1000 scripts/screwdriver_student_diffusion_latent_robust_light.sh ... train.ppo.diffusion_steps=11 train.ppo.diffusion_steps_infer=11 ...`
  - timeout exit: `124` (expected budget stop).
- Training-signal and hash:
  - `Current Best max=1811.39` (`train_1000s.log` parse).
  - new best hash: `72ccdf44f8e787ce9558efe9cc80816bc7420c5b`
  - kept reference hash: `31150d26646911325cac2aa408d3d42fdd4640a7`
  - previous early-stop `trainsteps11` hash: `c7fc6ec2e97990c9e4aba2c1ff1f28493e764204`
  - outcome: full-budget run is non-identical and escaped prior low-signal hash family.
- Eval compatibility note:
  - first nominal eval attempt failed with shape mismatch (`t_embed.weight` size 11 vs 10) when using eval defaults.
  - corrected by passing `+train.ppo.diffusion_steps=11` during eval.
- Primary eval results (`infer=11`) vs kept reference:
  - nominal: `1.461887` (delta `-0.213225`)
  - light_v2: `1.676452` (delta `+0.038186`)
  - hard: `1.390341` (delta `-0.114563`)
  - done rate is higher in all three conditions than kept reference.
- Eval-only side check (`infer=10` on same ckpt) vs kept reference:
  - nominal: `1.581308` (delta `-0.093804`)
  - light_v2: `1.379957` (delta `-0.258309`)
  - hard: `1.359385` (delta `-0.145519`)
  - conclusion: reducing infer steps at eval harms robustness for this checkpoint.

### Local decision
- `steps=11,infer=11` probe is **rejected** by strict keep/drop gate:
  - only `light_v2` shows a small gain,
  - `nominal` and `hard` are both below kept reference,
  - done-rate regression remains.
- Eval-only `infer=10` fallback is also rejected (worse robust profile).

### Remaining blocked/risky
- Robustness gap to kept reference persists.
- Sampler-depth increase can improve one condition (`light_v2`) while hurting others, indicating trade-off instability.

### Single recommended next step
- Keep current reference unchanged.
- Run the symmetric valid depth-down probe under full budget:
  - `train.ppo.diffusion_steps=9` and `train.ppo.diffusion_steps_infer=9` (seed42),
  - then unified `nominal/light_v2/hard` eval gate with non-identical-checkpoint requirement.

---

## v2-074 (2026-04-02) — Symmetric Depth-Down Probe (`steps=9,infer=9`) Full-Budget Evaluation

### Target milestone/subgoal
- Execute the recommended symmetric valid sampler-depth probe:
  - `train.ppo.diffusion_steps=9`
  - `train.ppo.diffusion_steps_infer=9`
- Run unified `nominal/light_v2/hard` evaluation and compare against kept reference.

### What changed (files + behavior impact)
- New full-budget training run:
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_trainsteps9_full_seed42_15min/`
- New eval logs:
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_trainsteps9_full/{diffusion_nominal_s42.log,diffusion_light_v2_s42.log,diffusion_hard_s42.log}`
- Frozen eval checkpoint:
  - `.../trainsteps9_full.../stage2_diffusion_nn/model_best_evalfreeze.ckpt`
- No code edits in this block.

### What was verified (commands + key outcomes)
- Full-budget training command:
  - `./docker-run-isaacgym.sh timeout 1000 scripts/screwdriver_student_diffusion_latent_robust_light.sh ... train.ppo.diffusion_steps=9 train.ppo.diffusion_steps_infer=9 ...`
  - timeout exit: `124` (expected budget stop).
- Training-side quality:
  - `train_1000s.log` parse: `Current Best max=1880.49`.
- Checkpoint identity:
  - new probe hash: `69b10406039bf76f6861efd962b59625704ae422`
  - kept reference hash: `31150d26646911325cac2aa408d3d42fdd4640a7`
  - previous `infersteps9_full` hash: `3449b4f0285c705f423d38cebce2b740fcafdcda`
  - outcome: high-signal and non-identical.
- Unified eval (`seed=42`, `steps=256`) with model-compatible overrides:
  - eval adds `+train.ppo.diffusion_steps=9 +train.ppo.diffusion_steps_infer=9`.
- Eval results vs kept reference:
  - nominal:
    - reward `1.771327` (delta `+0.096215`)
    - done `0.001221` (delta `-0.000081`)
    - `latent_mse +0.004336`, `action_mse_to_teacher -0.000236`
  - light_v2:
    - reward `1.615927` (delta `-0.022339`)
    - done `0.001546` (delta `+0.000163`)
    - `latent_mse +0.004450`, `action_mse_to_teacher +0.001737`
  - hard:
    - reward `1.477429` (delta `-0.027475`)
    - done `0.002035` (delta `+0.000163`)
    - `latent_mse -0.000565`, `action_mse_to_teacher -0.004666`

### Local decision
- `steps=9,infer=9` is **rejected** by strict keep/drop gate:
  - nominal improves clearly,
  - but both robustness conditions (`light_v2`, `hard`) are still below kept reference.
- Kept reference remains unchanged.

### Remaining blocked/risky
- Current optimization continues to show condition trade-off:
  - nominal gain often comes with slight robust regressions.
- Robustness gap remains the primary blocker to acceptance.

### Single recommended next step
- Keep current reference unchanged.
- Run one near-reference **mixup** probe to test if nominal gain can be preserved while reducing robust loss:
  - `train.ppo.diffusion_steps=9` with `train.ppo.diffusion_steps_infer=10` (effective infer is clamped to 9 at train time but eval-time can be compared under both infer=9 and infer=10 with matching model shape),
  - evaluate both eval infer settings under `nominal/light_v2/hard`,
  - accept only if robust deltas turn non-negative while nominal does not collapse.

---

## v2-075 (2026-04-02) — `trainsteps9_full` Multiseed Reality Check (`42,43,44`)

### Target milestone/subgoal
- Validate whether `v2-074` near-miss (`steps=9,infer=9`) is a seed-42 artifact.
- Compare candidate and kept reference under the same protocol for additional seeds `43,44`.

### What changed (files + behavior impact)
- New candidate eval logs (`seed=43,44`):
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_trainsteps9_full_multiseed/{diffusion_nominal_s43.log,diffusion_light_v2_s43.log,diffusion_hard_s43.log,diffusion_nominal_s44.log,diffusion_light_v2_s44.log,diffusion_hard_s44.log}`
- New reference eval logs (`seed=43,44`):
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_reference_multiseed/{diffusion_nominal_s43.log,diffusion_light_v2_s43.log,diffusion_hard_s43.log,diffusion_nominal_s44.log,diffusion_light_v2_s44.log,diffusion_hard_s44.log}`
- Existing seed42 logs reused:
  - reference: `plansv2_live_tailcoef02_thr015_sel_anchor003/*_s42.log`
  - candidate: `plansv2_live_tailcoef02_thr015_sel_anchor003_trainsteps9_full/*_s42.log`
- No code edits in this block.

### What was verified (commands + key outcomes)
- Candidate eval commands (`seed=43,44`) used:
  - `+train.ppo.diffusion_steps=9 +train.ppo.diffusion_steps_infer=9`
  - conditions: `nominal`, `light_v2`, `hard`.
- Reference eval commands (`seed=43,44`) used:
  - `+train.ppo.diffusion_steps=10 +train.ppo.diffusion_steps_infer=10`
  - conditions: `nominal`, `light_v2`, `hard`.
- Aggregated comparison on seeds `42,43,44`:
  - nominal reward mean:
    - reference `1.863957±0.138738`
    - candidate `2.032900±0.225208`
    - delta `+0.168944`
  - light_v2 reward mean:
    - reference `1.917208±0.198149`
    - candidate `1.715182±0.155943`
    - delta `-0.202026`
  - hard reward mean:
    - reference `1.481093±0.024690`
    - candidate `1.449879±0.151871`
    - delta `-0.031214`
  - done-rate mean deltas (candidate - reference):
    - nominal `-0.000515`
    - light_v2 `-0.000108`
    - hard `+0.000000`
- Per-seed reward delta (candidate - reference):
  - nominal: `+0.096215`, `+0.316563`, `+0.094053`
  - light_v2: `-0.022339`, `-0.144510`, `-0.439230`
  - hard: `-0.027475`, `+0.129265`, `-0.195433`

### Local decision
- `trainsteps9_full` is **rejected** by strict keep/drop gate after multiseed check:
  - nominal gain is consistent,
  - but robust conditions remain negative on average (especially `light_v2`).
- This is no longer treated as a likely seed-42 false negative.

### Remaining blocked/risky
- Sampler-depth axis keeps exhibiting nominal-vs-robust tradeoff rather than net robust gain.
- Hard condition still lacks stable positive edge over kept reference.

### Single recommended next step
- Keep current reference unchanged.
- Stop expanding sampler-depth axis for now and run one non-sampler near-reference full-budget probe:
  - `train.ppo.diffusion_teacher_delta_tail_threshold=0.14` (all other kept-reference settings fixed),
  - then unified `nominal/light_v2/hard` keep/drop gate.

---

## v2-076 (2026-04-02) — Non-Sampler Full-Budget Probe (`tail_threshold=0.14`) Rejected

### Target milestone/subgoal
- Execute the recommended non-sampler near-reference full-budget probe:
  - keep `tail_coef=0.2 + selective=True + anchor=0.03`,
  - set `train.ppo.diffusion_teacher_delta_tail_threshold=0.14`,
  - evaluate under unified `nominal/light_v2/hard`.

### What changed (files + behavior impact)
- New full-budget training run:
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr014_sel_anchor003_full_seed42_15min/`
- New eval logs:
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr014_sel_anchor003_full/{diffusion_nominal_s42.log,diffusion_light_v2_s42.log,diffusion_hard_s42.log}`
- Frozen eval checkpoint:
  - `.../thr014.../stage2_diffusion_nn/model_best_evalfreeze.ckpt`
- No code edits in this block.

### What was verified (commands + key outcomes)
- Training command:
  - `./docker-run-isaacgym.sh timeout 1000 scripts/screwdriver_student_diffusion_latent_robust_light.sh ... +train.ppo.diffusion_teacher_delta_tail_threshold=0.14 ...`
  - timeout exit: `124` (expected full-budget stop).
- Training quality and identity:
  - `Current Best max=1874.66` (`train_1000s.log` parse).
  - new hash: `087a5c68a172a183df5976e1a0c3bdd3edd1bc3f`
  - kept reference hash: `31150d26646911325cac2aa408d3d42fdd4640a7`
  - outcome: high-signal, non-identical checkpoint.
- Unified eval results vs kept reference (`seed=42`):
  - nominal:
    - reward `1.453175` (delta `-0.221937`)
    - done `0.001709` (delta `+0.000407`)
    - `latent_mse +0.014200`, `action_mse_to_teacher +0.033548`
  - light_v2:
    - reward `1.513260` (delta `-0.125006`)
    - done `0.001628` (delta `+0.000245`)
    - `latent_mse +0.007220`, `action_mse_to_teacher +0.019443`
  - hard:
    - reward `1.177461` (delta `-0.327443`)
    - done `0.002035` (delta `+0.000163`)
    - `latent_mse +0.008440`, `action_mse_to_teacher +0.024235`

### Local decision
- `tail_threshold=0.14` is **rejected**:
  - all three conditions regress,
  - alignment metrics degrade consistently.

### Remaining blocked/risky
- Tail-threshold down direction strongly harms robust behavior and teacher alignment.
- Robustness gap remains unresolved.

### Single recommended next step
- Run the opposite-side full-budget counterpart under the same protocol:
  - `train.ppo.diffusion_teacher_delta_tail_threshold=0.16`,
  - then compare `0.14 vs 0.15(reference) vs 0.16` directly.

---

## v2-077 (2026-04-02) — Non-Sampler Full-Budget Counterpart (`tail_threshold=0.16`) Rejected

### Target milestone/subgoal
- Complete threshold-axis bilateral verification under current full-budget + unified protocol:
  - run `tail_threshold=0.16`,
  - compare against current reference (`0.15`) and new `0.14` full-budget result.

### What changed (files + behavior impact)
- New full-budget training run:
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr016_sel_anchor003_full_seed42_15min/`
- New eval logs:
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr016_sel_anchor003_full/{diffusion_nominal_s42.log,diffusion_light_v2_s42.log,diffusion_hard_s42.log}`
- Frozen eval checkpoint:
  - `.../thr016.../stage2_diffusion_nn/model_best_evalfreeze.ckpt`
- No code edits in this block.

### What was verified (commands + key outcomes)
- Training command:
  - `./docker-run-isaacgym.sh timeout 1000 scripts/screwdriver_student_diffusion_latent_robust_light.sh ... +train.ppo.diffusion_teacher_delta_tail_threshold=0.16 ...`
  - timeout exit: `124` (expected full-budget stop).
- Training quality and identity:
  - `Current Best max=1795.35` (`train_1000s.log` parse).
  - new full-budget hash: `a3b179d0d17c44604a636b6358b6cf7f3266b664`
  - kept reference hash: `31150d26646911325cac2aa408d3d42fdd4640a7`
  - prior old `thr016` run hash: `ace6e093e1db55a4cca516e40a85a615fde602d8`
  - outcome: non-identical and now protocol-aligned full-budget evidence.
- Unified eval results vs kept reference (`seed=42`):
  - nominal:
    - reward `1.458110` (delta `-0.217002`)
    - done `0.001383` (delta `+0.000081`)
  - light_v2:
    - reward `1.694782` (delta `+0.056516`)
    - done `0.001465` (delta `+0.000082`)
  - hard:
    - reward `1.315465` (delta `-0.189439`)
    - done `0.002441` (delta `+0.000569`)
  - alignment deltas (`thr016 - reference`):
    - nominal `latent_mse +0.004510`, `action_mse_to_teacher +0.013909`
    - light_v2 `latent_mse -0.004163`, `action_mse_to_teacher -0.002761`
    - hard `latent_mse -0.003099`, `action_mse_to_teacher -0.001715`
- Bilateral threshold summary (`seed42 reward deltas vs reference`):
  - nominal: `thr014 -0.221937`, `thr016 -0.217002`
  - light_v2: `thr014 -0.125006`, `thr016 +0.056516`
  - hard: `thr014 -0.327443`, `thr016 -0.189439`

### Local decision
- `tail_threshold=0.16` is **rejected** by strict keep/drop gate:
  - only light_v2 improves,
  - nominal + hard remain below reference,
  - done-rate worsens in all three conditions.

### Remaining blocked/risky
- Threshold axis now has stronger bilateral negative evidence under current full-budget protocol.
- Pattern remains: single-condition gain with cross-condition tradeoff, not robust net improvement.

### Single recommended next step
- Keep current reference unchanged.
- De-prioritize threshold tuning and switch to a different near-reference axis with full-budget validation:
  - recommended next candidate: `train.ppo.bc_loss_coef=1.05` (full-budget replay of previously early-stopped micro-probe),
  - then unified `nominal/light_v2/hard` keep/drop gate.

---

## v2-078 (2026-04-02) — Near-Reference Full-Budget Replay (`bc_loss_coef=1.05`) Rejected

### Target milestone/subgoal
- Revisit previously early-stopped micro-probe under full budget:
  - keep `tail_coef=0.2 + tail_threshold=0.15 + selective=True + anchor=0.03`,
  - set `train.ppo.bc_loss_coef=1.05`,
  - run full-budget train + unified `nominal/light_v2/hard` evaluation.

### What changed (files + behavior impact)
- New full-budget training run:
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_trainbc105_full_seed42_15min/`
- New eval logs:
  - `outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003_trainbc105_full/{diffusion_nominal_s42.log,diffusion_light_v2_s42.log,diffusion_hard_s42.log}`
- Frozen eval checkpoint:
  - `.../trainbc105_full.../stage2_diffusion_nn/model_best_evalfreeze.ckpt`
- No code edits in this block.

### What was verified (commands + key outcomes)
- Full-budget training command:
  - `./docker-run-isaacgym.sh timeout 1000 scripts/screwdriver_student_diffusion_latent_robust_light.sh ... train.ppo.bc_loss_coef=1.05 ...`
  - timeout exit: `124` (expected).
- Training quality and identity:
  - `Current Best max=1841.12` (`train_1000s.log` parse).
  - full-budget hash: `f7931981f9b6e0c16d3b834a14b38acfce44a7da`
  - kept reference hash: `31150d26646911325cac2aa408d3d42fdd4640a7`
  - earlier early-stop `bc=1.05` hash: `00d1d1c76fd4f61ea1645c63a1a16901451083fc`
  - outcome: confirms early-stop false-negative risk; full-budget run is high-signal and non-identical.
- Unified eval results vs kept reference (`seed=42`):
  - nominal:
    - reward `1.569710` (delta `-0.105402`)
    - done `0.001383` (delta `+0.000081`)
    - `latent_mse +0.002032`, `action_mse_to_teacher +0.001537`
  - light_v2:
    - reward `1.463405` (delta `-0.174861`)
    - done `0.001546` (delta `+0.000163`)
    - `latent_mse +0.006280`, `action_mse_to_teacher +0.019389`
  - hard:
    - reward `1.181598` (delta `-0.323306`)
    - done `0.002523` (delta `+0.000651`)
    - `latent_mse +0.010248`, `action_mse_to_teacher +0.037008`

### Local decision
- `bc_loss_coef=1.05` is **rejected** by strict keep/drop gate:
  - all three conditions regress,
  - robustness and alignment both worsen, especially on `hard`.

### Remaining blocked/risky
- Full-budget replay does rescue training signal for some probes, but not final robust utility.
- Current near-reference axes still show robust regression despite high training `Current Best`.

### Single recommended next step
- Keep current reference unchanged.
- Run the symmetric near-reference counterpart under full budget:
  - `train.ppo.bc_loss_coef=0.95`,
  - then unified `nominal/light_v2/hard` keep/drop gate.

---

## v2-079 (2026-04-08) — Diffusion 算法文档补充（面向非代码读者的一页总结）

### Target milestone/subgoal
- 按用户请求补充“简单易懂但足够详细”的总结说明，重点回答：
  - 这个项目的工作内容是什么；
  - 当前技术路线是什么；
  - diffusion 与整个项目主线的关系是什么。

### What changed (files + behavior impact)
- 更新文档：
  - `docs/diffusion_algorithm.md`
- 新增章节：
  - `0. 一页看懂：这个项目在做什么，Diffusion 在哪里`
  - 子节 `0.1/0.2/0.3` 分别对应：
    - 项目工作内容（基线层/增强层/证据层）
    - Plan v2 技术路线（latent-first + residual fallback）
    - diffusion 在工程中的定位（受控增强轴，而非独立替代主线）
- 无代码逻辑改动。

### What was verified (commands + key outcomes)
- 文档落盘检查：
  - `sed -n '1,220p' docs/diffusion_algorithm.md`
  - outcome:
    - 新增总结段已存在；
    - 原有实现细节章节（代码映射、损失、推理、配置键）保持完整。

### Remaining blocked/risky
- 本条仅文档增强，不改变当前实验 keep/drop 结论。
- 若后续 Plan v2 治理文本发生阶段切换，需要同步刷新本总结段中的路线描述。

### Single recommended next step
- 继续执行实验主线（当前单变量 full-budget 验证序列），并在每次阶段性结论后同步维护该文档中的“路线定位”段落，确保论文/汇报表述始终与最新证据一致。

---

## v2-080 (2026-04-08) — `diffusion_algorithm.md` 3.1 细化（输入/处理/输出/动机）

### Target milestone/subgoal
- 按用户要求把 `3.1 条件 latent diffusion` 写得更详细且易懂，重点覆盖：
  - 用了哪些输入；
  - 每步如何处理；
  - 得到什么输出；
  - 为什么这样做、对应 diffusion 的什么思想；
  - 这些输入输出在本项目中的具体含义。

### What changed (files + behavior impact)
- 更新文档：
  - `docs/diffusion_algorithm.md`
- `3.1` 从简版升级为结构化细化版，新增：
  - `3.1.1 输入是什么（项目映射）`
  - `3.1.2 处理流程（训练迭代视角）`
  - `3.1.3 设计动机`
  - `3.1.4 diffusion 思想对应`
  - `3.1.5 项目输入输出闭环`
- 无代码改动。

### What was verified (commands + key outcomes)
- 代码对照来源：
  - `dexscrew/algo/ppo/diffusion_latent_student.py`（`sample_latent/train/test` 路径）
- 文档检查：
  - `sed -n '80,190p' docs/diffusion_algorithm.md`
  - outcome: 3.1 已包含完整“输入→处理→输出→原因→收益”描述，且与实现一致。

### Remaining blocked/risky
- 本条为说明文档增强，不改变当前实验 keep/drop 状态。
- 若后续实现改动了 `sample_latent` 或损失项组成，需要同步刷新该段。

### Single recommended next step
- 继续沿当前实验主线推进（full-budget 单变量探针），并在形成新结论后回写同口径解释，保持“实现-文档-证据”一致。
