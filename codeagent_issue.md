# codeagent_issue.md

## 2026-08-04 Resolved Escalation: M24 Pasini PPO Migration

- status: `resolved_authorized`
- resolved_on: `2026-08-04`
- resolution: the user explicitly authorized changing the diffusion branch
  objective to `XHandPasiniM24NutBolt -> PPO teacher` and updating governance.

### Task Background
- The user paused MuJoCo sim2sim work, saved commit `159d994` on the
  `sim2sim` branch, and switched the active worktree to `diffusion`.
- The requested next task is to migrate the M24x3, 160 mm nut-and-bolt PPO
  candidate from `sim2sim` into `diffusion` and prepare local PPO training.
- The M24 task uses the 16-DOF Paxini/Pasini hand and a user-confirmed keyboard
  init pose.

### Current Blocker
- The `diffusion` branch `AGENTS.md` defines
  `XHandHoraScrewDriver -> PPO teacher -> current student -> evaluation/export`
  as the canonical path.
- The same file requires codeagent to stop and write `codeagent_issue.md` when
  “Pasini or external repos become necessary for the current stage.”
- A correct M24 migration necessarily modifies the Pasini runtime, task map,
  task/train configs, and assets. It cannot be implemented as an asset-only
  copy while preserving the saved init-pose semantics.

### Evidence
- Source snapshot: `sim2sim` commit `159d994`, already pushed to
  `origin/sim2sim`.
- Required new M24 files include:
  - `assets/meshes/m24_hex_bolt_head_36x15.stl`
  - `assets/meshes/m24x3_bolt_thread_160.stl`
  - `assets/meshes/m24x3_hex_nut_36x21p5.stl`
  - `assets/screw/m24hex/0000_m24x3_160.{urdf,npy}`
  - `configs/task/XHandPasini{,M24}NutBolt.yaml`
  - `configs/train/XHandPasini{,M24}NutBolt.yaml`
  - M24 asset generator and keyboard-tuner launchers.
- Required shared runtime changes are visible in
  `diffusion..sim2sim`:
  - `dexscrew/tasks/__init__.py`: register `XHandPasiniNutBolt` and
    `XHandPasiniM24NutBolt`;
  - `dexscrew/tasks/xhand_pasini.py`: 123 changed lines so
    `asset.handInitPose`, `handRootPos`, and `handRootRPY` are consumed during
    actor creation and reset;
  - `scripts/tune_dexh13_lightbulb_initpose.py`: generic Paxini-compatible
    numeric keyboard controls.
- The active `diffusion` worktree was clean before this issue entry and is at
  commit `2864631`, synchronized with `origin/diffusion`.

### What Has Been Tried
- Updated and pushed the full sim2sim snapshot without losing moved DOTPG
  checkpoints.
- Switched to the clean `diffusion` branch and pulled it fast-forward.
- Read the branch-specific `AGENTS.md`, `PLANS_v2.md`, current session handoff,
  and stage acceptance summary.
- Audited the exact M24 files and shared runtime diff against `sim2sim`.
- Did not copy M24 files, modify Pasini runtime, or launch PPO after confirming
  the escalation boundary.

### Local Conclusion
- The migration is technically bounded and the source artifacts are preserved,
  but it changes the active diffusion stage from the canonical Hora student
  path to a new Pasini M24 PPO teacher task.
- This is a governance/plan transition, not a missing-asset or implementation
  blocker. Continuing without revising the active plan would violate the
  branch's explicit escalation rule.

### Recommended Next Action
1. Explicitly authorize a new training-stage objective on `diffusion` for
   `XHandPasiniM24NutBolt -> PPO teacher` and update the active plan/governance
   files accordingly.
2. Then selectively import only the M24 assets/configs, Pasini pose support,
   task registration, and tuner utilities from commit `159d994`; do not import
   `sim2sim/` or MuJoCo documentation.
3. Run compile/Hydra/one-env GPU PhysX checks, followed by a monitored
   8192-environment allocation probe on the local RTX 4080 SUPER before
   starting the long PPO run.

### Resolution Evidence
- `AGENTS.md` now names M24 Pasini PPO as the canonical path.
- `PLANS_m24_ppo.md` defines migration, smoke, capacity, and persistent-run
  milestones.
- The selective import contains no `sim2sim/` directory or MuJoCo files.
- Static checks and a one-environment Docker GPU PhysX smoke passed with 16
  hand DOFs, three object bodies, one nut DOF, finite reset state, and the
  user-saved M24 init pose.

## Status
- status_now: `open`
- opened_on: `2026-04-16`
- last_updated_on: `2026-04-17`
- trigger: `flow_after_v8_not_continue_worthy_under_current_scope`

## Task Background
- 当前执行计划已推进到：`PLANS_v8.md`。
- V8 的定位不是替代 `padapt`，而是：
  - 给 `FlowMatchingLatentStudent` 做一次 bounded recovery sprint；
  - 先判断它是否有资格继续进入扩时或 multiseed，而不是直接冲 accepted branch。
- 已完成：
  - `V8-M0` flow 工程闭环补丁（latent init / rollout helper / restore-save closure）
  - `V8-M1` 旧 flow artifact 的 `infer2 / infer4` eval-only probe
  - `V8-M2` 5 个 bounded single-seed fresh candidates
- 统一评测口径：`seed=42`, `steps=256`, `nominal + light_v2 + hard`。

## Current Blocker
- 在 V8 的 recovery sprint 与 bounded sweep 内，仍没有任何 flow candidate 达到“继续投入”的 single-seed continue gate：
  - `nominal_reward >= 1.95`
  - `light_v2_reward >= 1.70`
  - `hard_reward >= 1.55`
  - `hard_done <= 0.002372`
- 因此：
  - `V8-M2.5` 30min 扩时未触发；
  - `V8-M3` multiseed 未触发；
  - 当前范围内仍无法证明 Flow Matching diffusion 值得继续投入。

## Evidence
- `PLANS_v8.md`
- `docs/plansv8_final_verdict.md`
- `docs/stage_acceptance_summary.md`
- 关键结果（seed42, steps256）：
  - M1 eval-only probes:
    - `infer2`: nominal `1.805502`, light_v2 `1.681982`, hard `1.428555`
    - `infer4`: nominal `1.885794`, light_v2 `1.640372`, hard `1.189338`
  - M2 fresh candidates:
    - `zeroinit_bc12`: nominal `0.992769`, light_v2 `0.921589`, hard `1.020256`
    - `align2_rollout`: nominal `1.082436`, light_v2 `1.108353`, hard `0.750190`
    - `align2_anchor`: nominal `1.000670`, light_v2 `0.923009`, hard `0.953265`
    - `align2_bcheavy`: nominal `1.124580`, light_v2 `1.055061`, hard `1.053348`
    - `align4_rollout`: nominal `1.114895`, light_v2 `1.152107`, hard `1.079496`
- 参考 baseline：
  - current flow baseline:
    - nominal `1.782656`
    - light_v2 `1.587523`
    - hard `1.367413`
  - strongest accepted diffusion reference:
    - `V5.5 consistency_boundary_bc_tuned`

## What Has Been Tried
- V8 已尝试：
  - `flow_train_init_mode`
  - train-time rollout BC (`flow_train_align_infer + flow_rollout_bc_coef`)
  - `infer_steps=2 / 4`
  - heavier BC weighting
  - small base action anchor
  - restore/save closure for `sa_mean_std` and `agent_steps`
- 每个候选均完成必要评测，并记录 run_dir / ckpt sha1 / verdict 文档。

## Local Conclusion
- 在当前 V8 范围与预算下，Flow Matching 已完成 recovery sprint，但仍未证明自己具备继续投入的价值。
- 按 AGENTS escalation 边界，“diffusion 无法展现超越当前 student baseline 的价值”条件继续成立。
- 当前最合理的本地定位是：
  - `padapt` 继续作为主线 baseline；
  - `V5.5 consistency_boundary_bc_tuned` 保留为最强已接受 diffusion 参考；
  - `V8` 作为 flow recovery sprint 的负结果收口。

## Recommended Next Action
1. 治理层确认：接受 `PLANS_v8 completed_conclude`，冻结当前 flow matching 扩张线。
2. 若继续：必须新开计划，并明确授权**超出 V8 边界**的新假设；仅重复当前 bounded sweep、继续加时长、或在 `infer2/infer4 + rollout BC` family 内换小系数，不建议再做。
3. 若不继续：保留 `padapt` 为主线，`V5.5 consistency_boundary_bc_tuned` 为 diffusion 次优参考，`V8` 作为论文中的负结果与边界说明。
