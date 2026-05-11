# Cloud Session Handoff

Scope: live cloud execution state for `/root/code/dexscrew-repro`.

Use this file as the first stop when switching between Ubuntu-side and Windows-side Codex sessions. It records what the cloud machine has already done and what should happen next.

---

## 2026-05-06 -- CoDriveThesis Extra Paper Eval Completed

### Current Cloud State
- No active `tmux` sessions after completion.
- No active `python train.py` process after completion.
- GPU after completion: RTX 4090 D idle, about `1 MiB / 24564 MiB`, `0%`.

### Eval Run
- Additional eval run:
  `outputs/paper_eval_codrive_thesis_extra_s42_s43_s44_20260506_161503/`
- Previous clean baseline eval reused for combined tables:
  `outputs/paper_eval_codrive_thesis_cloud_s42_s43_s44_20260506_092231/`
- Protocol:
  - task: `Dexh13HoraLightbulbSim2RealTwoFingerCoDriveThesis`
  - seeds: `42,43,44`
  - fixed steps: `2048`
  - num envs: `48`
  - termination/noise aligned with previous paper eval.
- Startup note:
  - An initial tmux launch used `./run_eval.sh` and resolved `RUN_DIR` to the repo root, so it exited before starting real eval.
  - The eval was relaunched with an absolute script path and completed cleanly.

### Validation
- `validation_summary.txt`:
  - `raw_rows=18`
  - `expected_rows=18`
  - `all_ok=True`
  - `combined_raw_rows=30`
  - `combined_expected_rows=30`
  - `combined_all_ok=True`
- Error scan over logs found no:
  `Traceback`, `RuntimeError`, `Error executing job`, `CUDA out of memory`,
  `FileNotFoundError`, `Missing key`, `Unexpected key`, or `size mismatch`.

### Added Methods
| Method | Fixed-Step Reward mean/std | Done Rate mean/std | Status |
|---|---:|---:|---|
| `padapt` | `4.831464 +/- 0.077221` | `0.001061 +/- 0.000025` | OK |
| `diffusion_latent` | `3.504940 +/- 0.152013` | `0.001193 +/- 0.000006` | OK |
| `consistency_latent` | `5.310003 +/- 0.030259` | `0.001038 +/- 0.000011` | OK |
| `flow_matching` | `5.078809 +/- 0.134603` | `0.001078 +/- 0.000041` | OK |
| `diffusion_action_chunk` | `-1.231769 +/- 0.053342` | `0.016602 +/- 0.000000` | OK run, poor result |
| `purebc` | `4.969141 +/- 0.056104` | `0.001065 +/- 0.000038` | OK |

### Combined Table Artifacts
- Additional raw:
  `outputs/paper_eval_codrive_thesis_extra_s42_s43_s44_20260506_161503/fixed_eval_raw.csv`
- Additional aggregate:
  `outputs/paper_eval_codrive_thesis_extra_s42_s43_s44_20260506_161503/fixed_eval_aggregate.csv`
- Combined raw:
  `outputs/paper_eval_codrive_thesis_extra_s42_s43_s44_20260506_161503/fixed_eval_combined_raw.csv`
- Combined aggregate:
  `outputs/paper_eval_codrive_thesis_extra_s42_s43_s44_20260506_161503/fixed_eval_combined_aggregate.csv`

### Local Conclusion
- The uploaded current CoDriveThesis PAdapt/diffusion/PureBC checkpoints are evaluable under the same paper protocol.
- `consistency_latent` is the strongest student in this fixed-step table and is close to teacher PPO.
- `flow_matching` and `purebc` are also strong.
- `diffusion_latent` is valid but below DOTPG/PAdapt/PureBC/DAgger in this metric.
- `diffusion_action_chunk` is a valid run but fails behaviorally under this protocol and should not be presented as competitive without explanation.

### Recommended Next Cloud Action
- Treat `fixed_eval_combined_aggregate.csv` as the current cloud-side main-table source.
- Before final paper export, decide whether to include `diffusion_action_chunk` in the main table as a negative result or move it to an ablation/appendix table.
- Do not rerun these 30 rows unless the paper protocol changes.

## 2026-05-06 -- Windows Handoff: Eval Checkpoints Verified On Cloud

### Current Cloud State
- No new eval/training was launched.
- Final checkpoint availability check was run on cloud before switching to Windows-side development.
- GPU at check time: RTX 4090 D idle, about `1 MiB / 24564 MiB`, `0%`.

### Verified Eval Checkpoints Present
All checkpoint paths needed for the next unified CoDriveThesis paper eval are present on cloud:

- PPO teacher:
  `sim2real/codrive_thesis/best_reward_3655.17.pth`
- LatentBC:
  `outputs/Dexh13HoraLightbulb_student_bc_codrive_thesis/codrive_thesis_formal_latentbc_s42/bc_nn/model_best_student_eval.ckpt`
- DAgger:
  `outputs/Dexh13HoraLightbulb_student_dagger_codrive_thesis/codrive_thesis_formal_dagger_pure_replay_s42/dagger_nn/model_best_student_eval.ckpt`
- DOTPG:
  `outputs/Dexh13HoraLightbulb_student_dotpg_codrive_thesis/codrive_thesis_formal_dotpg_dual_bc5_s42/student_output/dotpg_nn/model_best.ckpt`
- PAdapt:
  `outputs/Dexh13HoraLightbulb_student_padapt_codrive_thesis/codrive_thesis_continue2h_rising_s42_20260505_from3h/stage2_nn/model_best_train.ckpt`
- diffusion latent:
  `outputs/Dexh13HoraLightbulb_student_diffusion_latent_codrive_thesis/codrive_thesis_teacher3655_s42_3h_final_20260504/stage2_diffusion_nn/model_best_train.ckpt`
- consistency latent:
  `outputs/Dexh13HoraLightbulb_student_consistency_codrive_thesis/codrive_thesis_teacher3655_s42_3h_final_20260504/stage2_consistency_nn/model_best_train.ckpt`
- flow matching:
  `outputs/Dexh13HoraLightbulb_student_flow_matching_codrive_thesis/codrive_thesis_continue2h_rising_s42_20260505_from3h/stage2_flow_nn/model_best_train.ckpt`
- diffusion action chunk:
  `outputs/Dexh13HoraLightbulb_student_diffusion_action_chunk_codrive_thesis/codrive_thesis_teacher3655_s42_3h_final_20260504/stage2_diffusion_action_chunk_nn/model_best_train.ckpt`
- PureBC:
  `outputs/Dexh13HoraLightbulb_student_purebc_codrive_thesis/codrive_thesis_continue2h_rising_s42_20260505_from3h/stage2_bc_nn/model_best.ckpt`

### Existing Clean Eval
- Existing paper eval:
  `outputs/paper_eval_codrive_thesis_cloud_s42_s43_s44_20260506_092231/`
- `validation_summary.txt`:
  `raw_rows=12`, `expected_rows=12`, `all_ok=True`
- This clean table covers:
  PPO teacher, LatentBC, DAgger, DOTPG.

### Recommended Next Cloud Action
- From Windows Codex, first read this file.
- Extend the existing paper eval pipeline to add:
  PAdapt, diffusion latent, consistency latent, flow matching, diffusion action chunk, and PureBC.
- Keep protocol aligned with the existing clean run unless intentionally changing the paper protocol:
  `Dexh13HoraLightbulbSim2RealTwoFingerCoDriveThesis`, seeds `42,43,44`, `2048` steps.

---

## 2026-05-06 -- DOTPG/Diffusion Local Files Synced, Eval Deferred

### Current Cloud State
- User explicitly deferred new eval execution; no eval or training was launched in this step.
- Cloud-visible handoff file is now present at:
  `docs/cloud_session_handoff.md`
- `AGENTS.md` now requires agents to read cloud-side docs as needed before cloud execution or preparing cloud commands. At minimum, reachable cloud sessions should check this file first.

### Synced From Local To Cloud
- Workflow/source:
  - `AGENTS.md`
  - `train.py`
  - `dexscrew/dotpg/`
  - `dexscrew/algo/eval_select.py`
  - diffusion-class student files under `dexscrew/algo/ppo/`
  - student wrappers/baseline files under `dexscrew/algo/student/`
- Scripts:
  - `scripts/cloud_codrive_diffusion4_student_1h.sh`
  - `scripts/cloud_codrive_diffusion4_student_continue2h.sh`
  - diffusion/DOTPG launch and visualizer scripts
- Config/package:
  - `configs/task/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveThesis.yaml`
  - `configs/train/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveThesis.yaml`
  - `sim2real/codrive_thesis/`
- Docs/theory:
  - `docs/diffusion_algorithm.md`
  - `docs/dotpg_codrive_optimization.md`
  - `docs/dotpg_env.md`
  - `thesis_reference/DOTPG-draft.md`
- Selected local CoDriveThesis ckpts now available on cloud:
  - PAdapt continued:
    `outputs/Dexh13HoraLightbulb_student_padapt_codrive_thesis/codrive_thesis_continue2h_rising_s42_20260505_from3h/stage2_nn/model_best_train.ckpt`
  - diffusion latent 3h:
    `outputs/Dexh13HoraLightbulb_student_diffusion_latent_codrive_thesis/codrive_thesis_teacher3655_s42_3h_final_20260504/stage2_diffusion_nn/model_best_train.ckpt`
  - consistency latent 3h:
    `outputs/Dexh13HoraLightbulb_student_consistency_codrive_thesis/codrive_thesis_teacher3655_s42_3h_final_20260504/stage2_consistency_nn/model_best_train.ckpt`
  - flow matching continued:
    `outputs/Dexh13HoraLightbulb_student_flow_matching_codrive_thesis/codrive_thesis_continue2h_rising_s42_20260505_from3h/stage2_flow_nn/model_best_train.ckpt`
  - diffusion action chunk 3h:
    `outputs/Dexh13HoraLightbulb_student_diffusion_action_chunk_codrive_thesis/codrive_thesis_teacher3655_s42_3h_final_20260504/stage2_diffusion_action_chunk_nn/model_best_train.ckpt`
  - PureBC continued:
    `outputs/Dexh13HoraLightbulb_student_purebc_codrive_thesis/codrive_thesis_continue2h_rising_s42_20260505_from3h/stage2_bc_nn/model_best.ckpt`

### Existing Paper Eval To Reuse
- Existing cloud paper-eval run:
  `outputs/paper_eval_codrive_thesis_cloud_s42_s43_s44_20260506_092231/`
- It already has clean `s42,s43,s44`, `2048`-step fixed eval for:
  - PPO teacher
  - LatentBC
  - DAgger
  - DOTPG
- `validation_summary.txt` reports:
  `all_ok=True`

### Recommended Next Cloud Action
- From Windows or Ubuntu, start by reading this file.
- Do not rerun the already-clean teacher/LatentBC/DAgger/DOTPG table unless the protocol changes.
- Extend the existing paper eval pipeline to add PAdapt, diffusion latent, consistency latent, flow matching, diffusion action chunk, and PureBC under the same task/seeds/steps protocol.

---

## 2026-05-06 -- CoDriveThesis Classic Baselines Completed, Eval Needs Repair

### Current Cloud State
- Cloud host: `cloud-training` / `/root/code/dexscrew-repro`
- GPU at last check: RTX 4090 D idle, about `1 MiB / 24564 MiB`, `0%`
- No active `tmux` sessions and no active `train.py` process at last check.

### Latest Pipeline
- Pipeline:
  `outputs/cloud_pipeline_codrive_thesis_classic/codrive_thesis_classic_s42_20260505_140934/`
- Phase:
  `done`
- Purpose:
  Train missing CoDriveThesis classic baselines from the frozen teacher:
  `sim2real/codrive_thesis/best_reward_3655.17.pth`
- Task:
  `Dexh13HoraLightbulbSim2RealTwoFingerCoDriveThesis`
- Selected cloud resource profile:
  `aggressive`

### Training Status
All formal training phases reached expected wall-clock timeout:

| Phase | status | note |
|---|---:|---|
| `formal_latentbc` | `124` | expected 3h timeout |
| `formal_dagger` | `124` | expected 3h timeout |
| `formal_dotpg` | `124` | expected 12h timeout |

Training summary from `training_summary.tsv`:

| Algorithm | Training signal |
|---|---|
| `LatentBC` | `best_student_eval=1009.37`, `last_student_eval=673.23` |
| `DAgger pure_replay` | `best_student_eval=1358.47`, `last_student_eval=1100.66` |
| `DOTPG dual_bc5` | `Current Best=2774.05` |

### Existing Fixed Eval Result
Current `fixed_eval_summary.csv` is only partially valid:

| Checkpoint | status | avg_reward | avg_done_rate | validity |
|---|---:|---:|---:|---|
| `latentbc_model_last` | `0` | `4.068575` | `0.000163` | valid |
| `dagger_model_last` | `0` | `3.808516` | `0.000651` | valid |
| `latentbc_model_best_deploy` | `1` | `NA` | `NA` | invalid: checkpoint missing |
| `dagger_model_best_deploy` | `1` | `NA` | `NA` | invalid: checkpoint missing |
| `dotpg_model_best` | `1` | `NA` | `NA` | invalid eval command: missing DOTPG restore overrides |
| `dotpg_model_last` | `1` | `NA` | `NA` | invalid: checkpoint missing |

DOTPG eval failed because the eval command restored a `teacher_actor`/`dual` checkpoint with default DOTPG policy settings. Re-eval must include:

```text
+train.dotpg.state_mode=student
+train.dotpg.dynamic_state=True
+train.dotpg.policy_arch=teacher_actor
+train.dotpg.policy_output_mode=clamp
+train.dotpg.policy_loss_mode=dual
```

### Relevant Checkpoints
- LatentBC:
  `outputs/Dexh13HoraLightbulb_student_bc_codrive_thesis/codrive_thesis_formal_latentbc_s42/bc_nn/model_last.ckpt`
- DAgger:
  `outputs/Dexh13HoraLightbulb_student_dagger_codrive_thesis/codrive_thesis_formal_dagger_pure_replay_s42/dagger_nn/model_last.ckpt`
- DOTPG:
  `outputs/Dexh13HoraLightbulb_student_dotpg_codrive_thesis/codrive_thesis_formal_dotpg_dual_bc5_s42/student_output/dotpg_nn/model_best.ckpt`

### Known Pitfalls
- `outputs/local_pipeline_codrive_thesis_students_6x3h/.../summary.tsv` contains polluted `5001.07` values; use `strict_summary.tsv` instead.
- CoDriveThesis diffusion/PAdapt/PureBC training rewards are not final paper numbers; they still need identical fixed-step eval.
- DOTPG `model_last.ckpt` is absent in the latest classic cloud run; use `model_best.ckpt`.
- LatentBC/DAgger `model_best_deploy.ckpt` were absent in the latest classic cloud run; use `model_last.ckpt` unless a later selector run creates deploy aliases.

### Recommended Next Cloud Action
Run a unified CoDriveThesis paper-eval pipeline on the cloud for:

- PPO teacher
- PAdapt continued
- diffusion latent 3h
- consistency latent 3h
- flow matching continued
- diffusion action chunk 3h
- PureBC continued
- LatentBC cloud `model_last`
- DAgger cloud `model_last`
- DOTPG cloud `model_best`

Use one fixed protocol first:

```text
task = Dexh13HoraLightbulbSim2RealTwoFingerCoDriveThesis
seed = 42
num_envs = 48 or 256
steps = 256
termination enabled
obs_noise_t_scale = 0.01
obs_noise_e_scale = 0.02
```

Then extend to multiseed/robustness only after the single-seed table is clean.

## 2026-05-06 -- CoDriveThesis Full Paper Experiment Started

- Run dir: `outputs/paper_codrive_thesis_full_20260506_210423`
- tmux/session owner: `paper_codrive_thesis_20260506_210423`
- Plan: formal 5-method 3-train-seed suite, unified eval, NFE/latency, representation ablation, robustness.
- GPU: NVIDIA GeForce RTX 4090 D, 24564 MiB
- Status file: `outputs/paper_codrive_thesis_full_20260506_210423/status/phase.txt`

## 2026-05-06 -- CoDriveThesis Full Paper Experiment Started

- Run dir: `outputs/paper_codrive_thesis_full_20260506_210536`
- tmux/session owner: `paper_codrive_thesis_20260506_210536`
- Plan: formal 5-method 3-train-seed suite, unified eval, NFE/latency, representation ablation, robustness.
- GPU: NVIDIA GeForce RTX 4090 D, 24564 MiB
- Status file: `outputs/paper_codrive_thesis_full_20260506_210536/status/phase.txt`

### Correction: 20260506_210423 launch aborted
- Run dir: `outputs/paper_codrive_thesis_full_20260506_210423`
- Reason: initial launch did not attach to tmux due shell quoting; processes were stopped and `status/phase.txt` was set to `aborted_bad_tmux_launch`.
- Active run is `outputs/paper_codrive_thesis_full_20260506_210536` in tmux `paper_codrive_thesis_20260506_210536`.


### Correction: 20260506_210536 launch aborted
- Run dir: `outputs/paper_codrive_thesis_full_20260506_210536`
- Reason: evaluator restore commands were missing `train.ppo.proprio_adapt=True` for student checkpoints; script was fixed before any long training completed.
- This run was stopped and `status/phase.txt` was set to `aborted_eval_restore_arg_fix`.

## 2026-05-06 -- CoDriveThesis Full Paper Experiment Started

- Run dir: `outputs/paper_codrive_thesis_full_20260506_210723`
- tmux/session owner: `paper_codrive_thesis_20260506_210723`
- Plan: formal 5-method 3-train-seed suite, unified eval, NFE/latency, representation ablation, robustness.
- GPU: NVIDIA GeForce RTX 4090 D, 24564 MiB
- Status file: `outputs/paper_codrive_thesis_full_20260506_210723/status/phase.txt`

## 2026-05-07 -- Checkpoint Selection Hotfix

- Run dir: outputs/paper_codrive_thesis_full_20260506_210723
- Added checkpoint aliaser session: paper_codrive_ckpt_aliaser_20260506_210723
- Reason: active 5h timeout runs do not reach eval_select 20M interval, so deploy checkpoints were not emitted.
- Action: model_best_deploy.ckpt symlinked to model_best_train.ckpt; see status/checkpoint_selection_hotfix.txt and status/checkpoint_aliases.tsv.

## 2026-05-07 -- Active Guard Started

- Session: paper_codrive_active_guard_20260506_210723
- Interval: 300 seconds
- Status: outputs/paper_codrive_thesis_full_20260506_210723/status/active_guard_status.txt
- Progress snapshot: outputs/paper_codrive_thesis_full_20260506_210723/status/training_progress.tsv
- Control alert: outputs/paper_codrive_thesis_full_20260506_210723/status/CONTROL_REQUIRED.txt

## 2026-05-08 -- CoDriveThesis Full Paper Experiment Pipeline Finished

- Run dir: `outputs/paper_codrive_thesis_full_20260506_210723`
- Validation: `outputs/paper_codrive_thesis_full_20260506_210723/validation_summary.txt`
- Main aggregate: `outputs/paper_codrive_thesis_full_20260506_210723/aggregate_csv/main_aggregate.csv`
