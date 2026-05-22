# Session Handoff v2

Scope: Plan v2 execution log (`PLANS_v2.md`) only.  
Start date: 2026-03-24.

## v2-2026-05-18 -- Build Middle/Small PAdapt-DOTPG Deploy Packs

### Target milestone/subgoal
- Package the current Middle1 and Small1 PAdapt/DOTPG deployment artifacts under `sim2real/deploy` using the existing deploy directory format.

### What changed (files + behavior impact)
- Added Middle1 deploy pack:
  - `sim2real/deploy/codrive_middle/padapt_deploy/`
  - `sim2real/deploy/codrive_middle/dotpg_deploy/`
- Added Small deploy pack:
  - `sim2real/deploy/codrive_small/padapt_deploy/`
  - `sim2real/deploy/codrive_small/dotpg_deploy/`
- Each deploy subdirectory contains exactly:
  - the task YAML used by the corresponding training/distillation run
  - the train YAML used by the corresponding training/distillation run
  - the PPO teacher `best_reward_*.pth`
  - the student `model_best.ckpt`

### What was verified (commands + key outcomes)
- Verified every deploy subdirectory has the expected four-file layout.
- Middle1 deploy artifacts use:
  - PPO teacher `best_reward_3799.45.pth`
  - PAdapt `Current Best: 3236.33`
  - DOTPG `Current Best: 2901.99`
- Small deploy artifacts use the already-distilled Small PPO3985 run:
  - PPO teacher `best_reward_3985.40.pth`
  - PAdapt `Current Best: 3464.36`
  - DOTPG `Current Best: 3048.75`
- Small deploy uses the saved `*.yaml.used` from the PPO3985 student pipeline, because the live `configs/task/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveSmall.yaml` was modified later for a different PPO-only run.

### Remaining blocked/risky
- The later Small PPO-only checkpoint `best_reward_3838.03.pth` has no corresponding PAdapt/DOTPG students yet, so it was not used for this deploy pack.

### Single recommended next step
- Use the deploy packs for local visualization/deployment checks, starting with `codrive_middle/padapt_deploy` and `codrive_small/padapt_deploy`.

---

## v2-2026-05-18 -- Sync Middle1 PPO3799 Students Locally

### Target milestone/subgoal
- Sync the completed cloud PAdapt/DOTPG students distilled from the latest `middle1_keyboardlatest_noinitnoise` PPO teacher to Ubuntu local and prepare headed visualization commands.

### What changed (files + behavior impact)
- Synced cloud PAdapt output locally:
  `outputs/Dexh13HoraLightbulb_student_padapt_codrive_middle1/middle1_keyboardlatest_noinitnoise_students_s42_20260518_160530_padapt1h_from_ppo3799/`.
- Synced cloud DOTPG output locally:
  `outputs/Dexh13HoraLightbulb_student_dotpg_codrive_middle1/middle1_keyboardlatest_noinitnoise_students_s42_20260518_160530_dotpg1h_from_ppo3799_dual_bc5/`.
- Synced cloud pipeline logs/status locally:
  `outputs/cloud_pipeline_codrive_middle1_students_after_noinitnoise1h/middle1_keyboardlatest_noinitnoise_students_s42_20260518_160530/`.

### What was verified (commands + key outcomes)
- Local checkpoint files exist:
  - PPO teacher:
    `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_middle1/middle1_keyboardlatest_noinitnoise_s42_20260518_111917_ppo1h/stage1_nn/best_reward_3799.45.pth`.
  - PAdapt student:
    `outputs/Dexh13HoraLightbulb_student_padapt_codrive_middle1/middle1_keyboardlatest_noinitnoise_students_s42_20260518_160530_padapt1h_from_ppo3799/stage2_nn/model_best.ckpt`.
  - DOTPG student:
    `outputs/Dexh13HoraLightbulb_student_dotpg_codrive_middle1/middle1_keyboardlatest_noinitnoise_students_s42_20260518_160530_dotpg1h_from_ppo3799_dual_bc5/student_output/dotpg_nn/model_best.ckpt`.
- Synced status confirms:
  - PAdapt `Current Best: 3236.33`, `exit_status=124`.
  - DOTPG `Current Best: 2901.99`, `exit_status=124`.

### Remaining blocked/risky
- The two synced student checkpoints still need headed visualization to judge behavior quality.

### Single recommended next step
- Run the three headed visualization commands for Middle1 PPO/PAdapt/DOTPG and compare policy behavior.

---

## v2-2026-05-18 -- Middle1 PPO3799 Students Cloud Complete

### Target milestone/subgoal
- Check completion status for the cloud PAdapt/DOTPG distillation jobs launched from the latest `middle1_keyboardlatest_noinitnoise` PPO teacher.

### What changed (files + behavior impact)
- Updated `docs/cloud_session_handoff.md` to mark the cloud jobs complete and record final checkpoints.

### What was verified (commands + key outcomes)
- Cloud pipeline:
  `outputs/cloud_pipeline_codrive_middle1_students_after_noinitnoise1h/middle1_keyboardlatest_noinitnoise_students_s42_20260518_160530/`.
- `status/phase.txt=done`.
- Cloud GPU idle after completion:
  `NVIDIA GeForce RTX 4090 D, 1 MiB / 24564 MiB, 0%`.
- PAdapt:
  - `padapt_exit_status=124`
  - max parsed `Current Best: 3236.33`
  - checkpoint:
    `outputs/Dexh13HoraLightbulb_student_padapt_codrive_middle1/middle1_keyboardlatest_noinitnoise_students_s42_20260518_160530_padapt1h_from_ppo3799/stage2_nn/model_best.ckpt`.
- DOTPG:
  - `dotpg_exit_status=124`
  - max parsed `Current Best: 2901.99`
  - checkpoint:
    `outputs/Dexh13HoraLightbulb_student_dotpg_codrive_middle1/middle1_keyboardlatest_noinitnoise_students_s42_20260518_160530_dotpg1h_from_ppo3799_dual_bc5/student_output/dotpg_nn/model_best.ckpt`.

### Remaining blocked/risky
- The new student checkpoints are still on the cloud and have not yet been synced locally for headed visualization.

### Single recommended next step
- Sync the new PAdapt/DOTPG Middle1 student output dirs locally and run headed visualization.

---

## v2-2026-05-18 -- CoDriveSmall Saved InitPose PPO1h Local Complete

### Target milestone/subgoal
- Apply the latest keyboard-saved `CoDriveSmall` init pose to the task YAML and run a local 1h PPO teacher training.

### What changed (files + behavior impact)
- Updated `configs/task/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveSmall.yaml` from:
  `outputs/initpose_tuning/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveSmall_current.yaml`.
  - `handRootPos: [0.100000, 0.036000, 0.247000]`
  - `handRootRPY: [3.141500, 0.439627, 3.141500]`
  - `right_thumb_joint_2: 0.5600000024`
  - `right_thumb_joint_3: 0.3919662833`
- Added local PPO pipeline script:
  `outputs/local_pipeline_codrive_small_ppo1h/codrive_small_keyboardlatest_y036_thumb056_s42_20260518_161335/run_codrive_small_ppo1h_local.sh`.
- PPO output:
  `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_small/codrive_small_keyboardlatest_y036_thumb056_s42_20260518_161335_ppo1h/`.

### What was verified (commands + key outcomes)
- Local 1h PPO command completed under `timeout 3600`:
  - `ppo_exit_status=124`
  - `timeout_status=expected_1h_timeout`
  - resource settings: `task.env.numEnvs=8192`, `train.ppo.num_actors=8192`, `train.ppo.minibatch_size=16384`, `num_threads=16`
- Best checkpoint:
  `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_small/codrive_small_keyboardlatest_y036_thumb056_s42_20260518_161335_ppo1h/stage1_nn/best_reward_3838.03.pth`.
- End-of-run process check showed no remaining local `codrive_small_keyboardlatest_y036_thumb056` training process; GPU memory returned to about `1272 MiB / 16376 MiB`.

### Remaining blocked/risky
- This PPO has not yet been headed-visualized.
- It is slightly below the previous `CoDriveSmall` PPO best `3985.40`, so visual behavior should decide whether the new hand-root/thumb pose is preferable.

### Single recommended next step
- Headed-visualize:
  `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_small/codrive_small_keyboardlatest_y036_thumb056_s42_20260518_161335_ppo1h/stage1_nn/best_reward_3838.03.pth`.

---

## v2-2026-05-18 -- CoDriveSmall InitPose Tuner Inspection

### Target milestone/subgoal
- Open the keyboard-controlled init-pose tuner for `Dexh13HoraLightbulbSim2RealTwoFingerCoDriveSmall` so the user can inspect/adjust the Small PPO initial pose.

### What changed (files + behavior impact)
- The tuner saved a new snippet to:
  `outputs/initpose_tuning/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveSmall_current.yaml`.
- The task YAML was not overwritten in this step.

### What was verified (commands + key outcomes)
- Ran:
  `./docker-run-isaacgym.sh python scripts/tune_dexh13_lightbulb_initpose.py --task Dexh13HoraLightbulbSim2RealTwoFingerCoDriveSmall --gpu 0 --seed 42 --out outputs/initpose_tuning/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveSmall_current.yaml`.
- The tuner forced clean one-env inspection and disabled init/noise randomization for viewing.
- Final saved values:
  - `handRootPos: [0.100000, 0.036000, 0.247000]`
  - `handRootRPY: [3.141500, 0.439627, 3.141500]`
  - `right_thumb_joint_2: 0.5600000024`
  - `right_thumb_joint_3: 0.3919662833`

### Remaining blocked/risky
- The saved snippet has not yet been applied to `configs/task/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveSmall.yaml`.

### Single recommended next step
- If this pose looked correct, apply the saved snippet to the Small task YAML before any new Small PPO/student training.

---

## v2-2026-05-18 -- Launch Middle1 PPO3799 Student Distillation On Cloud

### Target milestone/subgoal
- Correct the missing student follow-up for the latest cloud Middle1 NoInitNoise PPO teacher by launching PAdapt and DOTPG distillation in parallel on the cloud.

### What changed (files + behavior impact)
- Added and synced cloud launch script:
  `outputs/cloud_pipeline_codrive_middle1_students_after_noinitnoise1h/middle1_keyboardlatest_noinitnoise_students_s42_20260518_160530/run_middle1_students_after_noinitnoise1h_cloud.sh`.
- Updated `docs/cloud_session_handoff.md` with the active tmux session and cloud output paths.

### What was verified (commands + key outcomes)
- Confirmed local/cloud Middle1 task and train YAML SHA256 match.
- Confirmed teacher checkpoint exists on cloud:
  `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_middle1/middle1_keyboardlatest_noinitnoise_s42_20260518_111917_ppo1h/stage1_nn/best_reward_3799.45.pth`.
- Launched cloud tmux session:
  `middle1_students_ppo3799_20260518_160530`.
- Startup check showed both `train.py` jobs active:
  - PAdapt: `task.env.numEnvs=512`, `train.ppo.minibatch_size=6144`
  - DOTPG: `task.env.numEnvs=1024`, `train.ppo.minibatch_size=12288`
- Initial GPU after launch:
  `NVIDIA GeForce RTX 4090 D, 9162 MiB / 24564 MiB, 95%`.

### Remaining blocked/risky
- Jobs are still running. Expected successful wall-clock completion is `timeout` exit code `124` for each phase.

### Single recommended next step
- After about 1h, check `status/padapt_exit_status.txt` and `status/dotpg_exit_status.txt`; if both are `124`, sync the two student output dirs locally and visualize.

---

## v2-2026-05-18 -- Sync Six Middle/Small Visualization Checkpoints

### Target milestone/subgoal
- Sync/verify the six current `middle1` and `small` PPO/student checkpoints locally and prepare headed visualization commands.

### What changed (files + behavior impact)
- Synced final cloud `middle1_keyboardlatest_noinitnoise` PPO output locally:
  `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_middle1/middle1_keyboardlatest_noinitnoise_s42_20260518_111917_ppo1h/`.
- Synced its cloud pipeline status/log folder locally:
  `outputs/cloud_pipeline_codrive_middle1_ppo1h/middle1_keyboardlatest_noinitnoise_s42_20260518_111917/`.

### What was verified (commands + key outcomes)
- Verified the six local checkpoints exist:
  - Middle1 PPO:
    `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_middle1/middle1_keyboardlatest_noinitnoise_s42_20260518_111917_ppo1h/stage1_nn/best_reward_3799.45.pth`.
  - Middle1 PAdapt:
    `outputs/Dexh13HoraLightbulb_student_padapt_codrive_middle1/middle1_padapt_s42_20260515_131253_from_ppo1h/stage2_nn/model_best.ckpt`.
  - Middle1 DOTPG:
    `outputs/Dexh13HoraLightbulb_student_dotpg_codrive_middle1/middle1_dotpg_s42_20260515_131253_dual_bc5_from_ppo1h/student_output/dotpg_nn/model_best.ckpt`.
  - Small PPO:
    `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_small/codrive_small_keyboardlatest_s42_20260518_120558_ppo1h/stage1_nn/best_reward_3985.40.pth`.
  - Small PAdapt:
    `outputs/Dexh13HoraLightbulb_student_padapt_codrive_small/codrive_small_students_from_ppo3985_s42_20260518_131156_padapt1h/stage2_nn/model_best.ckpt`.
  - Small DOTPG:
    `outputs/Dexh13HoraLightbulb_student_dotpg_codrive_small/codrive_small_students_from_ppo3985_s42_20260518_131156_dotpg1h_dual_bc5/student_output/dotpg_nn/model_best.ckpt`.

### Remaining blocked/risky
- Middle1 PAdapt/DOTPG are from the older `middle1_ppo1h_s42_20260515_120420` teacher (`best_reward_3783.10.pth`), not from the latest `middle1_keyboardlatest_noinitnoise` teacher (`best_reward_3799.45.pth`).

### Single recommended next step
- Run the headed visualization commands for the six checkpoints and decide whether the latest Middle1 teacher also needs fresh PAdapt/DOTPG distillation.

---

## v2-2026-05-18 -- Status Check Middle1 And Small

### Target milestone/subgoal
- Confirm whether the recent `middle1` and `small` train/distill jobs have completed and whether cloud is still busy.

### What changed (files + behavior impact)
- Updated `docs/cloud_session_handoff.md` with a new top status entry correcting stale cloud "active" notes.

### What was verified (commands + key outcomes)
- Local process check:
  - no active local `train.py` / `timeout 3600` training processes remained.
- Local `CoDriveSmall`:
  - PPO done:
    `ppo_exit_status=124`, `timeout_status=expected_1h_timeout`, best:
    `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_small/codrive_small_keyboardlatest_s42_20260518_120558_ppo1h/stage1_nn/best_reward_3985.40.pth`.
  - PAdapt done:
    max parsed `Current Best: 3464.36`, ckpt:
    `outputs/Dexh13HoraLightbulb_student_padapt_codrive_small/codrive_small_students_from_ppo3985_s42_20260518_131156_padapt1h/stage2_nn/model_best.ckpt`.
  - DOTPG done:
    max parsed `Current Best: 3048.75`, ckpt:
    `outputs/Dexh13HoraLightbulb_student_dotpg_codrive_small/codrive_small_students_from_ppo3985_s42_20260518_131156_dotpg1h_dual_bc5/student_output/dotpg_nn/model_best.ckpt`.
- Cloud:
  - GPU idle: `NVIDIA GeForce RTX 4090 D, 1 MiB / 24564 MiB, 0%`.
  - no active cloud `train.py` / `timeout 3600` processes found.
  - `Middle1 KeyboardLatest NoInitNoise PPO1h` done on cloud:
    `ppo_exit_status=124`, best:
    `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_middle1/middle1_keyboardlatest_noinitnoise_s42_20260518_111917_ppo1h/stage1_nn/best_reward_3799.45.pth`.

### Remaining blocked/risky
- Local only has the mid-run synced `middle1_keyboardlatest_noinitnoise` checkpoint `best_reward_2807.89.pth`; the final cloud `best_reward_3799.45.pth` has not yet been synced back locally.
- The latest `middle1_keyboardlatest_noinitnoise` PPO did not automatically launch PAdapt/DOTPG student distillation; the older Middle1 PAdapt/DOTPG artifacts are from the earlier `middle1_ppo1h_s42_20260515_120420` teacher.

### Single recommended next step
- Sync the final cloud `middle1_keyboardlatest_noinitnoise` PPO output locally if it should be visualized or used for new student distillation.

---

## v2-2026-05-18 -- CoDriveSmall PAdapt/DOTPG Student1h Local Complete

### Target milestone/subgoal
- Distill the local `CoDriveSmall` PPO teacher `best_reward_3985.40.pth` into PAdapt and DOTPG students, each for a full 1h wall-clock window.

### What changed (files + behavior impact)
- Added local student pipeline script:
  `outputs/local_pipeline_codrive_small_students1h/codrive_small_students_from_ppo3985_s42_20260518_131156/run_codrive_small_students1h_local.sh`.
- Teacher checkpoint used:
  `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_small/codrive_small_keyboardlatest_s42_20260518_120558_ppo1h/stage1_nn/best_reward_3985.40.pth`.
- PAdapt output:
  `outputs/Dexh13HoraLightbulb_student_padapt_codrive_small/codrive_small_students_from_ppo3985_s42_20260518_131156_padapt1h/`.
- DOTPG output:
  `outputs/Dexh13HoraLightbulb_student_dotpg_codrive_small/codrive_small_students_from_ppo3985_s42_20260518_131156_dotpg1h_dual_bc5/`.
  - DOTPG variant: `policy_arch=teacher_actor`, `policy_init_from_teacher=True`, `policy_loss_mode=dual`, `bc_coef=5.0`, `bc_pretrain_steps=3000`, GPU fp16 buffers.

### What was verified (commands + key outcomes)
- Sequential local run completed:
  - `padapt_exit_status=124`, `padapt_timeout_status=expected_1h_timeout`
  - `dotpg_exit_status=124`, `dotpg_timeout_status=expected_1h_timeout`
  - `status/phase.txt=done`
- PAdapt:
  - max parsed `Current Best: 3464.36`
  - checkpoint:
    `outputs/Dexh13HoraLightbulb_student_padapt_codrive_small/codrive_small_students_from_ppo3985_s42_20260518_131156_padapt1h/stage2_nn/model_best.ckpt`
  - GPU memory about `4.0-4.2GB`.
- DOTPG:
  - max parsed `Current Best: 3048.75`
  - checkpoint:
    `outputs/Dexh13HoraLightbulb_student_dotpg_codrive_small/codrive_small_students_from_ppo3985_s42_20260518_131156_dotpg1h_dual_bc5/student_output/dotpg_nn/model_best.ckpt`
  - GPU memory about `7.2-7.4GB`.

### Remaining blocked/risky
- These are training-reward selections only. Headed visualization is still needed to judge actual two-finger behavior and deployment suitability.

### Single recommended next step
- Headed-visualize both student checkpoints on `Dexh13HoraLightbulbSim2RealTwoFingerCoDriveSmall`, starting with PAdapt because it reached the higher training best.

---

## v2-2026-05-18 -- CoDriveSmall KeyboardLatest PPO1h Local Complete

### Target milestone/subgoal
- Apply the latest keyboard-saved `CoDriveSmall` init pose and run a local 1h PPO teacher probe.

### What changed (files + behavior impact)
- Updated `configs/task/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveSmall.yaml` from:
  `outputs/initpose_tuning/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveSmall_current.yaml`.
  - `handRootPos: [0.100000, 0.026000, 0.247000]`
  - `handRootRPY: [3.141500, 0.439627, 3.141500]`
  - index joints: `[0.3499999940, 0.9252794981, 0.1628971249, 0.2492995113]`
  - thumb joints: `[0.1299999952, 1.5700000525, 0.4800000787, 0.4119662941]`
- The `CoDriveSmall` task now uses the RealBulb-style small-object scale setup:
  - `env.object.type: screw_realbulb`
  - `baseObjScale: 1.00`
  - scale randomization window `0.95-1.05`
  - no init pose noise: `object.init_pos_noise: [0.0, 0.0, 0.0]`, `handRootPosNoise: [0.0, 0.0, 0.0]`, `handRootPosZScaleComp: 0.0`
- Added local PPO pipeline output:
  `outputs/local_pipeline_codrive_small_ppo1h/codrive_small_keyboardlatest_s42_20260518_120558/`.

### What was verified (commands + key outcomes)
- Local 1h PPO command completed under `timeout 3600`:
  - `ppo_exit_status=124`
  - `timeout_status=expected_1h_timeout`
  - `status/phase.txt=done`
- Training used local RTX 4080 SUPER with:
  - `task.env.numEnvs=8192`
  - `train.ppo.num_actors=8192`
  - `train.ppo.minibatch_size=16384`
  - observed GPU memory about `11.5GB / 16GB`, FPS about `21.8k`
- Final best teacher checkpoint:
  `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_small/codrive_small_keyboardlatest_s42_20260518_120558_ppo1h/stage1_nn/best_reward_3985.40.pth`.

### Remaining blocked/risky
- The reward is close to the historical pure CoDrive 4000-range PPO, but behavior still needs headed visualization to verify whether the index finger assists rotation.

### Single recommended next step
- Visualize `best_reward_3985.40.pth` for `Dexh13HoraLightbulbSim2RealTwoFingerCoDriveSmall` and decide whether to launch student distillation from this PPO.

---

## v2-2026-05-18 -- CoDriveSmall YAML Derived From Middle1

### Target milestone/subgoal
- Create a `CoDriveSmall` task variant from the current no-noise Middle1 YAML, changing only the loaded bulb asset to the RealBulb asset path/type.

### What changed (files + behavior impact)
- Added `configs/task/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveSmall.yaml`.
  - Derived from current `configs/task/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle1.yaml`.
  - `eval_cache_name: sim2real_twofinger_codrive_small`.
  - `env.object.type: screw_realbulb`.
  - All other Middle1 settings are preserved, including:
    `baseObjScale=1.20`, scale randomization `1.15-1.25`, no init-pose noise, current keyboard-saved hand pose, reward settings, and joint limits.
- Added `configs/train/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveSmall.yaml`.
  - Byte-identical to the Middle1 train YAML.

### What was verified (commands + key outcomes)
- `diff -u configs/task/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle1.yaml configs/task/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveSmall.yaml`
  - Only `eval_cache_name` and `env.object.type` differ.
- `diff -u configs/train/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle1.yaml configs/train/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveSmall.yaml`
  - No differences.

### Remaining blocked/risky
- `screw_realbulb` is the real-size URDF asset used by the previous RealBulb task. Keeping Middle1's `baseObjScale=1.20` and `randomizeScale=1.15-1.25` is a literal "only change asset" variant, but it may not match the older RealBulb task's intended `baseObjScale=1.00` and `0.95-1.05` scale window.

### Single recommended next step
- Run a headed one-env check for `Dexh13HoraLightbulbSim2RealTwoFingerCoDriveSmall` and decide whether the asset-only variant or the RealBulb-style scale window is the desired experiment.

---

## v2-2026-05-18 -- Middle1 KeyboardLatest NoInitNoise PPO1h Cloud Active

### Target milestone/subgoal
- Apply the latest keyboard-saved Middle1 init pose, force init-pose noise/Z scale compensation to zero, and train a 1h PPO teacher on cloud.

### What changed (files + behavior impact)
- Updated `configs/task/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle1.yaml` from:
  `outputs/initpose_tuning/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle1_current.yaml`.
  - `handRootPos: [0.092000, 0.020000, 0.245000]`
  - `handRootRPY: [3.141500, 0.439627, 3.141500]`
  - index joints: `[0.3499999940, 0.9052795172, 0.1628971249, 0.4492996037]`
  - thumb joints: `[0.1299999952, 1.5700000525, 0.2999999821, 0.5719662905]`
  - `object.init_pos_noise: [0.0, 0.0, 0.0]`
  - `asset.handRootPosNoise: [0.0, 0.0, 0.0]`
  - `asset.handRootPosZScaleComp: 0.0`
- Added cloud PPO script:
  `outputs/cloud_pipeline_codrive_middle1_ppo1h/middle1_keyboardlatest_noinitnoise_s42_20260518_111917/run_middle1_keyboardlatest_noinitnoise_ppo1h_cloud.sh`.
  - Output:
    `Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_middle1/middle1_keyboardlatest_noinitnoise_s42_20260518_111917_ppo1h`.
  - Uses `timeout 3600`, `task.env.numEnvs=12288`, `train.ppo.num_actors=12288`, `train.ppo.minibatch_size=24576`, `num_threads=22`.

### What was verified (commands + key outcomes)
- Read cloud handoff before cloud execution.
- Pure CoDrive reprobe was stopped to free the GPU:
  - last observed partial best:
    `best_reward_2848.31.pth`.
- The first `middle1_keyboardlatest_s42_20260518_111658` cloud run was stopped/restarted because init-pose noise was not zero.
- Local and cloud YAML were verified for the restarted run:
  - `object.init_pos_noise: [0.0, 0.0, 0.0]`
  - `asset.handRootPosNoise: [0.0, 0.0, 0.0]`
  - `asset.handRootPosZScaleComp: 0.0`
- Synced task/train YAML, latest tuner output, and cloud script to `/root/code/dexscrew-repro`.
- Launched cloud tmux session:
  `middle1_keyboardlatest_noinitnoise_ppo1h_20260518_111917`.
- Startup health verified:
  - `status/phase.txt`: `ppo`
  - Python active under `timeout 3600`
  - GPU around `14313 MiB / 24564 MiB`
  - first best checkpoint:
    `best_reward_61.34.pth` at about 2.6 minutes elapsed.
- Mid-run sync for visualization:
  - Cloud still active: `phase=ppo`.
  - GPU around `14313 MiB / 24564 MiB`.
  - Latest synced checkpoint:
    `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_middle1/middle1_keyboardlatest_noinitnoise_s42_20260518_111917_ppo1h/stage1_nn/best_reward_2807.89.pth`.

### Remaining blocked/risky
- Training is still running; final `ppo_exit_status` and final best reward are not known yet.
- Need final sync/visualization after completion.

### Single recommended next step
- Let this no-noise Middle1 PPO run to the expected 1h timeout, then sync:
  `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_middle1/middle1_keyboardlatest_noinitnoise_s42_20260518_111917_ppo1h/`
  and visualize the final best checkpoint.

---

## v2-2026-05-18 -- Pure CoDrive Reprobe PPO1h Cloud Active

### Target milestone/subgoal
- Stop the weak-looking Middle1 `Index112 NoInitNoise` isolation PPO and re-run the original/pure CoDrive PPO for 1h to test whether the historical CoDrive behavior/reward is reproducible.

### What changed (files + behavior impact)
- Added cloud PPO script:
  `outputs/cloud_pipeline_codrive_reprobe_ppo1h/codrive_reprobe_s42_20260518_105210/run_codrive_reprobe_ppo1h_cloud.sh`.
  - Task: `Dexh13HoraLightbulbSim2RealTwoFingerCoDrive`.
  - Output:
    `Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive/codrive_reprobe_s42_20260518_105210_ppo1h`.
  - Uses `timeout 3600`, `task.env.numEnvs=12288`, `train.ppo.num_actors=12288`, `train.ppo.minibatch_size=24576`, `num_threads=22`.

### What was verified (commands + key outcomes)
- User-requested stop of the active `Index112 NoInitNoise` run completed.
  - Cloud GPU returned to idle before launching pure CoDrive.
  - Stop status:
    `outputs/cloud_pipeline_codrive_middle1_index112_noinitnoise_ppo1h/middle1_index112_noinitnoise_s42_20260518_101206/status/phase.txt = stopped_manual`.
  - Last observed best before stop:
    `best_reward_3360.56.pth`.
- Local deployment copy check:
  - `configs/codrive/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.task.yaml` is byte-identical to `configs/task/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.yaml`.
  - `configs/codrive/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.train.yaml` is byte-identical to `configs/train/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.yaml`.
  - Task SHA256:
    `019876eaf3de4c8bbc2180886fd90ec406b753a8cdbfce72a198c31b790c6e62`.
  - Train SHA256:
    `07f2903dd9aa511c44e358757a322cf361323b2876a5c2e15f507e627ce2a627`.
- Synced pure CoDrive task/train YAML and cloud script to `/root/code/dexscrew-repro`.
- Launched cloud tmux session:
  `codrive_reprobe_ppo1h_20260518_105210`.
- Startup health verified:
  - `status/phase.txt`: `ppo`
  - Python active under `timeout 3600`
  - GPU around `14329 MiB / 24564 MiB`
  - first best checkpoint appeared:
    `best_reward_65.72.pth` at about 3 minutes elapsed.

### Remaining blocked/risky
- Pure CoDrive reprobe is still running; final `ppo_exit_status` and final best reward are not known yet.
- Need final sync/visualization after completion.

### Single recommended next step
- Let pure CoDrive run to the expected 1h timeout, then sync:
  `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive/codrive_reprobe_s42_20260518_105210_ppo1h/`
  and visualize the final `stage1_nn/best_reward_*.pth`.

---

## v2-2026-05-18 -- CoDriveMiddle1 Index112 NoInitNoise PPO1h Cloud Active

### Target milestone/subgoal
- Run an isolation PPO teacher experiment on cloud for the Middle1 contact geometry:
  close initial pose noise and set `right_index_joint_3=1.12`, then inspect behavior around the 30min mark.

### What changed (files + behavior impact)
- Added isolated task YAML:
  `configs/task/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle1Index112NoInitNoise.yaml`.
  - Derived from `Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle1.yaml`.
  - `object.init_pos_noise: [0.0, 0.0, 0.0]`.
  - `asset.handRootPosNoise: [0.0, 0.0, 0.0]`.
  - `right_index_joint_3: 1.1200000000`.
  - Preserves `asset.handRootPosZScaleComp: 0.0`.
  - Preserves scale/mass/friction/PD/reward/action settings from Middle1.
- Added matching train YAML:
  `configs/train/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle1Index112NoInitNoise.yaml`.
- Added cloud PPO script:
  `outputs/cloud_pipeline_codrive_middle1_index112_noinitnoise_ppo1h/middle1_index112_noinitnoise_s42_20260518_101206/run_middle1_index112_noinitnoise_ppo1h_cloud.sh`.
  - Output:
    `Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_middle1_index112_noinitnoise/middle1_index112_noinitnoise_s42_20260518_101206_ppo1h`.
  - Uses `timeout 3600`, `task.env.numEnvs=12288`, `train.ppo.num_actors=12288`, `train.ppo.minibatch_size=24576`, `num_threads=22`.

### What was verified (commands + key outcomes)
- Read local `docs/session_handoff_v2.md` and `docs/stage_acceptance_summary.md`.
- Read cloud `/root/code/dexscrew-repro/docs/cloud_session_handoff.md` before launch.
- Cloud was idle before launch: RTX 4090 D around `1 MiB / 24564 MiB`.
- Synced task/train YAML and cloud script to `/root/code/dexscrew-repro/`.
- Launched cloud tmux session:
  `middle1_index112_noinitnoise_ppo1h_20260518_101206`.
- Startup health verified:
  - `status/phase.txt`: `ppo`
  - Python active under `timeout 3600`
  - GPU around `14339 MiB / 24564 MiB`
  - first best checkpoint appeared and advanced to about:
    `best_reward_117.90.pth` at about 3.6 minutes elapsed.
- 30min check:
  - `phase=ppo`, cloud training still active.
  - best at about 20min: `best_reward_2818.47.pth`.
  - best at about 30.5min: `best_reward_3237.63.pth`.
  - Synced active PPO output/pipeline back to local.
  - Local latest synced checkpoint:
    `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_middle1_index112_noinitnoise/middle1_index112_noinitnoise_s42_20260518_101206_ppo1h/stage1_nn/best_reward_3255.81.pth`.
- Updated cloud-visible handoff:
  `docs/cloud_session_handoff.md`.

### Remaining blocked/risky
- Training is still running; final `ppo_exit_status` is not known yet.
- The 30min visual check requires syncing the active output directory and launching a local headed viewer against the newest/best checkpoint.

### Single recommended next step
- Headed-visualize the 30min synced checkpoint:
  `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_middle1_index112_noinitnoise/middle1_index112_noinitnoise_s42_20260518_101206_ppo1h/stage1_nn/best_reward_3255.81.pth`.

---

## v2-2026-05-18 -- Git Ignore Tightened For Experiment Artifacts

### Target milestone/subgoal
- Prevent generated baseline/deploy/model artifacts from being accidentally included in normal git commits.

### What changed (files + behavior impact)
- Updated `.gitignore`.
  - Ignore `baseline_results/` and baseline result zip artifacts.
  - Ignore `sim2real/deploy/` and deploy zip artifacts.
  - Ignore generated model/checkpoint binaries: `*.pth`, `*.ckpt`, `*.pt`.
  - Ignore TensorBoard event dumps: `events.out.tfevents*`, `*.tfevents*`.

### What was verified (commands + key outcomes)
- `git status --short --untracked-files=all`
  - Large untracked deploy/baseline CSV/PTH/CKPT artifacts no longer appear.
- `git check-ignore -v ...`
  - Confirmed examples under `baseline_results/`, `sim2real/deploy/`, `baseline_results.zip`, and `sim2real/dotpg_deployv1.zip` are now ignored.

### Remaining blocked/risky
- `.gitignore` does not affect files already tracked by Git. Existing tracked artifacts such as `sim2real/codrive.zip` and some historical `sim2real/**/*.pth/*.ckpt/*.pt` remain tracked unless explicitly removed with `git rm --cached`.
- Current status still shows `D sim2real/codrive.zip`; decide whether to commit that deletion or restore it before pushing.

### Single recommended next step
- Before committing, run `git status --short` and decide whether `sim2real/codrive.zip` should stay deleted or be restored with `git restore sim2real/codrive.zip`.

---

## v2-2026-05-15 -- CoDriveMiddle1 Latest Initpose PPO1h Cloud Completed/Synced

### Target milestone/subgoal
- Train a 1h PPO teacher on cloud for the latest keyboard-saved `Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle1` init pose.

### What changed (files + behavior impact)
- Updated `configs/task/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle1.yaml` from the latest tuner save:
  `outputs/initpose_tuning/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle1_current.yaml`.
  - `handRootPos: [0.080000, 0.020000, 0.239000]`
  - `handRootRPY: [3.141500, 0.439627, 3.141500]`
  - index joints: `[0.3499999940, 0.9852794409, 0.2628971040, 0.5692995787]`
  - thumb joints: `[0.1299999952, 1.5700000525, 0.0399999991, 0.5719662905]`
  - Preserved Middle1 noise/reward settings.
- Added cloud PPO pipeline script:
  `outputs/cloud_pipeline_codrive_middle1_ppo1h/middle1_latestinit_s42_20260515_150457/run_middle1_latestinit_ppo1h_cloud.sh`.
  - Output name:
    `Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_middle1/middle1_latestinit_s42_20260515_150457_ppo1h`
  - Uses `timeout 3600`, `task.env.numEnvs=12288`, `train.ppo.minibatch_size=24576`, `num_threads=22`.

### What was verified (commands + key outcomes)
- Read cloud handoff before execution.
- Cloud was idle before launch: RTX 4090 D around `1 MiB / 24564 MiB`.
- Synced to cloud:
  - `configs/task/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle1.yaml`
  - `configs/train/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle1.yaml`
  - `outputs/initpose_tuning/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle1_current.yaml`
  - cloud pipeline script.
- Removed a temporary cloud sync directory accidentally created during the first rsync attempt:
  `/root/code/dexscrew-repro/configs/__tmp_should_not_use`.
- Launched cloud tmux session:
  `middle1_latestinit_ppo1h_20260515_150457`.
- Startup health was verified:
  - Python process active under `timeout 3600`
  - GPU around `14273 MiB / 24564 MiB`
  - first best checkpoint appeared early.
- Completion verified:
  - `status/phase.txt`: `done`
  - `ppo_exit_status=124`
  - `timeout_status=expected_1h_timeout`
  - cloud GPU idle after completion: RTX 4090 D around `1 MiB / 24564 MiB`
  - final best checkpoint:
    `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_middle1/middle1_latestinit_s42_20260515_150457_ppo1h/stage1_nn/best_reward_3309.63.pth`
- Synced cloud results back to Ubuntu local:
  - `outputs/cloud_pipeline_codrive_middle1_ppo1h/middle1_latestinit_s42_20260515_150457/`
  - `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_middle1/middle1_latestinit_s42_20260515_150457_ppo1h/`

### Remaining blocked/risky
- None for artifact sync. Behavioral quality still needs local headed visualization.

### Single recommended next step
- Visualize:
  `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_middle1/middle1_latestinit_s42_20260515_150457_ppo1h/stage1_nn/best_reward_3309.63.pth`
  and compare index/thumb cooperation against the previous Middle1 PPO (`best_reward_3783.10.pth`) and original CoDrive (`best_reward_4159.37.pth`).

---

## v2-2026-05-15 -- CoDriveMiddle1 Initpose Updated From Latest Tuner Save

### Target milestone/subgoal
- Solidify the latest keyboard-tuned `Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle1` init pose before another visual/training comparison.

### What changed (files + behavior impact)
- Updated `configs/task/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle1.yaml` from:
  `outputs/initpose_tuning/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle1_current.yaml`.
- New asset pose:
  - `handRootPos: [0.080000, 0.022000, 0.239000]`
  - `handRootRPY: [3.141500, 0.439627, 3.141500]`
- New two-finger init joints:
  - index: `[0.3499999940, 0.9852794409, 0.2628971040, 0.5692995787]`
  - thumb: `[0.1299999952, 1.5700000525, 0.0399999991, 0.5719662905]`
- Preserved Middle1 training randomization/noise/reward settings, including:
  - `object.init_pos_noise: [0.0075, 0.0075, 0.0]`
  - `asset.handRootPosNoise: [0.001, 0.001, 0.001]`
  - `asset.handRootPosZScaleComp: 0.0`

### What was verified (commands + key outcomes)
- `sed -n '280,306p' configs/task/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle1.yaml`
  confirmed the saved pose values are present in the task YAML.

### Remaining blocked/risky
- Host-side `omegaconf` import is unavailable, so the quick host YAML parse check did not run.
- The next useful validation is visual, because this change is contact-geometry sensitive.

### Single recommended next step
- Open a deterministic headed Middle1 PPO/student viewer with noise disabled and check whether the index fingertip now starts close enough to contribute tangential rotation.

---

## v2-2026-05-15 -- Middle1 PAdapt Local And DOTPG Cloud Distillation Completed/Synced

### Target milestone/subgoal
- Use the Middle1 PPO teacher checkpoint as the baseline for two 1h student distillation runs:
  - local PAdapt for 1h
  - cloud DOTPG dual-BC5 for 1h

### What changed (files + behavior impact)
- Added local PAdapt pipeline script:
  `outputs/local_pipeline_codrive_middle1_padapt1h/middle1_padapt_s42_20260515_131253/run_middle1_padapt1h_local.sh`.
  - Teacher checkpoint:
    `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_middle1/middle1_ppo1h_s42_20260515_120420/stage1_nn/best_reward_3783.10.pth`.
  - Output:
    `outputs/Dexh13HoraLightbulb_student_padapt_codrive_middle1/middle1_padapt_s42_20260515_131253_from_ppo1h/`.
  - Uses `timeout 3600`, `task.env.numEnvs=512`, `train.ppo.minibatch_size=6144`.
- Added cloud DOTPG pipeline script:
  `outputs/cloud_pipeline_codrive_middle1_dotpg1h/middle1_dotpg_s42_20260515_131253/run_middle1_dotpg1h_cloud.sh`.
  - Teacher checkpoint synced to cloud under the same relative `outputs/.../best_reward_3783.10.pth` path.
  - Output:
    `outputs/Dexh13HoraLightbulb_student_dotpg_codrive_middle1/middle1_dotpg_s42_20260515_131253_dual_bc5_from_ppo1h/`.
  - Uses `timeout 3600`, `task.env.numEnvs=1024`, `train.ppo.minibatch_size=12288`.
  - DOTPG variant: `policy_arch=teacher_actor`, `policy_loss_mode=dual`, `bc_coef=5.0`, `bc_alpha_max=20.0`.
- Synced Middle1 task/train YAML, Middle1 PPO teacher pth, and cloud DOTPG script to:
  `/root/code/dexscrew-repro/`.

### What was verified (commands + key outcomes)
- Read cloud handoff before launching.
- Cloud was idle before launch: RTX 4090 D around `1 MiB / 24564 MiB`, no active training process.
- Local first PAdapt background attempt did not persist; root cause was the Docker command receiving a host absolute checkpoint path. Fixed script to pass the container-visible relative `outputs/.../best_reward_3783.10.pth`, removed the stale created container, and relaunched.
- Local PAdapt no longer has an active training process and produced a usable best checkpoint:
  `outputs/Dexh13HoraLightbulb_student_padapt_codrive_middle1/middle1_padapt_s42_20260515_131253_from_ppo1h/stage2_nn/model_best.ckpt`
  - checkpoint timestamp: `2026-05-15 14:16`
  - checkpoint size: `1,301,474` bytes
  - TensorBoard event:
    `outputs/Dexh13HoraLightbulb_student_padapt_codrive_middle1/middle1_padapt_s42_20260515_131253_from_ppo1h/stage2_tb/events.out.tfevents.1778822198.wbz-ubuntu22-pc`
- Cloud DOTPG completed with the expected wall-clock timeout:
  - `status/phase.txt`: `done`
  - `dotpg_exit_status=124`
  - latest best:
    `outputs/Dexh13HoraLightbulb_student_dotpg_codrive_middle1/middle1_dotpg_s42_20260515_131253_dual_bc5_from_ppo1h/student_output/dotpg_nn/model_best.ckpt`
  - observed final current best plateau around `2397.86`.
- Synced cloud DOTPG results back to local:
  - `outputs/cloud_pipeline_codrive_middle1_dotpg1h/middle1_dotpg_s42_20260515_131253/`
  - `outputs/Dexh13HoraLightbulb_student_dotpg_codrive_middle1/middle1_dotpg_s42_20260515_131253_dual_bc5_from_ppo1h/`
- Cloud after completion is idle:
  RTX 4090 D around `1 MiB / 24564 MiB`, `0%`.

### Remaining blocked/risky
- Local PAdapt wrapper did not write `status/summary.txt` or `padapt_exit_status.txt`; its `status/phase.txt` still says `padapt`.
  Treat the produced `stage2_nn/model_best.ckpt` as the usable artifact, but do not treat local wrapper metadata as complete.
- Local PAdapt stdout log is heavily buffered and only contains startup lines; use the TensorBoard event/checkpoint for artifact evidence.

### Single recommended next step
- Visualize and compare the two Middle1 student checkpoints locally:
  - PAdapt:
    `outputs/Dexh13HoraLightbulb_student_padapt_codrive_middle1/middle1_padapt_s42_20260515_131253_from_ppo1h/stage2_nn/model_best.ckpt`
  - DOTPG:
    `outputs/Dexh13HoraLightbulb_student_dotpg_codrive_middle1/middle1_dotpg_s42_20260515_131253_dual_bc5_from_ppo1h/student_output/dotpg_nn/model_best.ckpt`

---

## v2-2026-05-15 -- Middle1/Middle2 PPO1h Completed And Synced

### Target milestone/subgoal
- Run and collect a one-hour PPO comparison between two CoDrive init-pose variants:
  - local `Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle1`
  - cloud `Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle2`

### What changed (files + behavior impact)
- Added local pipeline script:
  `outputs/local_pipeline_codrive_middle1_ppo1h/middle1_s42_20260515_120420/run_middle1_ppo1h.sh`.
  - Runs `timeout 3600` PPO for Middle1.
  - Records command, metadata, exit status, and latest best checkpoint under the same pipeline directory.
- Added cloud pipeline script:
  `outputs/cloud_pipeline_codrive_middle2_ppo1h/middle2_s42_20260515_120420/run_middle2_ppo1h_cloud.sh`.
  - Runs `timeout 3600` PPO for Middle2.
  - Uses the cloud conda IsaacGym environment instead of Docker.
  - Records command, metadata, exit status, and latest best checkpoint under the cloud pipeline directory.
- Synced Middle2 task/train YAMLs and cloud script to:
  `/root/code/dexscrew-repro/`.
- Synced cloud Middle2 PPO results back to local:
  - `outputs/cloud_pipeline_codrive_middle2_ppo1h/middle2_s42_20260515_120420/`
  - `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_middle2/middle2_s42_20260515_120420_ppo1h/`

### What was verified (commands + key outcomes)
- Stopped the previous headed Middle2 viewer/training process before starting long runs.
- Local Middle1 launched in background with `nohup` because local `tmux` is unavailable:
  - pipeline: `outputs/local_pipeline_codrive_middle1_ppo1h/middle1_s42_20260515_120420/`
  - command uses `task.env.numEnvs=8192`, `train.ppo.minibatch_size=16384`
  - GPU observed around `11.45GB / 16GB`, Python process active.
- Cloud SSH host key was refreshed for the confirmed target:
  `ssh -p 22222 root@180.184.47.96`.
- Cloud handoff was read before execution.
- Cloud Middle2 first Docker launch failed because the new cloud does not have image `dexscrew:ig20-py38`.
- Cloud script was corrected to source the existing `dexscrew-ig` conda/IsaacGym environment, matching previous cloud pipelines.
- Cloud Middle2 relaunched in tmux:
  - session: `middle2_ppo1h_20260515_120420_retry`
  - pipeline: `outputs/cloud_pipeline_codrive_middle2_ppo1h/middle2_s42_20260515_120420/`
  - command uses `task.env.numEnvs=12288`, `train.ppo.minibatch_size=24576`, `num_threads=22`
  - GPU observed around `14.3GB / 24GB`, Python process active.
  - Later confirmed PPO progress and completion:
    `ppo_exit_status=124`, expected one-hour timeout.
  - Best cloud checkpoint synced locally:
    `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_middle2/middle2_s42_20260515_120420_ppo1h/stage1_nn/best_reward_3503.45.pth`.
- Local Middle1 checkpoint exists:
  `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_middle1/middle1_ppo1h_s42_20260515_120420/stage1_nn/best_reward_3783.10.pth`.
- Cloud process check after completion:
  - no active Middle2 training process
  - GPU idle around `1 MiB / 24564 MiB`

### Remaining blocked/risky
- Local Middle1 generated a valid best checkpoint and TensorBoard event file, but its wrapper did not write `status/summary.txt` or `ppo_exit_status.txt`; treat the checkpoint as usable, but do not treat the local wrapper metadata as complete.
- Cloud Middle2 metadata is complete and records expected timeout status `124`.

### Single recommended next step
- Visualize both PPO checkpoints locally and compare behavior:
  - Middle1: `best_reward_3783.10.pth`
  - Middle2: `best_reward_3503.45.pth`

---

## v2-2026-05-15 -- CoDriveMiddle2 Latest Initpose Variant Added

### Target milestone/subgoal
- Create a second CoDriveMiddle init-pose variant from `middle1` using the newest keyboard-saved hand/root pose, then open a headed PPO viewer for inspection.

### What changed (files + behavior impact)
- Added `configs/task/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle2.yaml`.
  - Based on `Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle1.yaml`.
  - `eval_cache_name: sim2real_twofinger_codrive_middle2`.
  - Latest saved tuner source:
    `outputs/initpose_tuning/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle1_current.yaml`.
  - Applied `asset.handRootPos: [0.086000, 0.038000, 0.233000]`.
  - Applied `asset.handRootRPY: [3.141500, 0.439627, 3.141500]`.
  - Applied updated index/thumb init joints:
    index joint1 `0.8252795935`, thumb joint3 `0.5319663286`.
  - Inherited `middle1` randomization/noise, including object init noise `[0.0075, 0.0075, 0.0]`, hand root noise `[0.001, 0.001, 0.001]`, and `handRootPosZScaleComp: 0.0`.
- Added `configs/train/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle2.yaml`.
  - Exact copy of the `middle1` train yaml.

### What was verified (commands + key outcomes)
- `diff -u configs/task/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle1.yaml configs/task/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle2.yaml`
  - Only experiment name and the new saved hand/root init-pose fields differ.
- `diff -q configs/train/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle1.yaml configs/train/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle2.yaml`
  - Train YAMLs are identical.
- Launched headed local viewer/training:
  - `./docker-run-isaacgym.sh python train.py task=Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle2 headless=False seed=42 task.env.numEnvs=10 train.ppo.minibatch_size=120 train.ppo.output_name=Dexh13HoraLightbulb_teacher/vis_codrive_middle2_headed_s42 wandb_activate=False graphics_device_id=0`
  - Startup config confirmed `numEnvs=10`, `handRootPos=[0.086,0.038,0.233]`, `handRootPosZScaleComp=0.0`, and the inherited noise/randomization settings.

### Remaining blocked/risky
- The headed run is for visual inspection only; no long training/eval conclusion yet.

### Single recommended next step
- Inspect the current Middle2 headed viewer. If the pose looks right, use this YAML as the next PPO candidate; if not, return to the keyboard tuner and save another init-pose.

---

## v2-2026-05-15 -- Initpose Tuner HandRoot Noise Disabled

### Target milestone/subgoal
- Ensure keyboard init-pose tuning always uses the YAML center pose, not a random hand-root sample.

### What changed (files + behavior impact)
- Updated `scripts/tune_dexh13_lightbulb_initpose.py`.
  - Added default override `task.env.asset.handRootPosNoise=[0.0,0.0,0.0]`.
  - The tuner already disabled object init-position noise, scale randomization, mass/COM/friction/PD randomization, and random force.
- Updated `tele_readme.md`.
  - Documented that hand-root position noise is also forced off during keyboard tuning.

### What was verified (commands + key outcomes)
- `python -m py_compile scripts/tune_dexh13_lightbulb_initpose.py`
  - Passed.
- `rg -n "handRootPosNoise" scripts/tune_dexh13_lightbulb_initpose.py tele_readme.md`
  - Confirmed the override appears in both script and docs.

### Remaining blocked/risky
- None for the tuner override. Training YAMLs can still keep hand-root noise; only the keyboard tuning entrypoint forces it off.

### Single recommended next step
- Use the normal tuner command without extra overrides; it now opens a clean center-pose view even for noisy task YAMLs.

---

## v2-2026-05-15 -- CoDriveMiddle1 Initpose/Noise Variant Added

### Target milestone/subgoal
- Create a CoDrive-derived `middle1` task using the latest keyboard-tuned DexH13 hand/root pose, then inspect it under XHandHora-style init-position noise with a headed PPO viewer.

### What changed (files + behavior impact)
- Added `configs/task/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle1.yaml`.
  - Based on `Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.yaml`.
  - `eval_cache_name: sim2real_twofinger_codrive_middle1`.
  - Uses latest saved tuner values from `outputs/initpose_tuning/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveExper_current.yaml`.
  - `asset.handRootPos: [0.094000, 0.026000, 0.229000]`.
  - `asset.handRootRPY: [3.1415, 0.352360, 3.1415]`.
  - Two-finger joint init pose updated from the saved snippet.
  - Added XHandHora-style init-position noise:
    - `object.init_pos_noise: [0.0075, 0.0075, 0.0]`
    - `asset.handRootPosNoise: [0.001, 0.001, 0.001]`
  - Kept original CoDrive scale compensation: `asset.handRootPosZScaleComp: 0.06`.
- Added `configs/train/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle1.yaml`.
  - Exact copy of the original CoDrive train YAML.

### What was verified (commands + key outcomes)
- First headed run was stopped because it mistakenly used `handRootPos: [0.0, 0.0, 0.21]`.
- Corrected headed run launched:
  - `./docker-run-isaacgym.sh python train.py task=Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle1 headless=False seed=42 task.env.numEnvs=10 train.ppo.minibatch_size=120 train.ppo.output_name=Dexh13HoraLightbulb_teacher/vis_codrive_middle1_headed_s42 wandb_activate=False graphics_device_id=0`
- Runtime config print confirmed:
  - `handRootPos: [0.094, 0.026, 0.229]`
  - `handRootPosNoise: [0.001, 0.001, 0.001]`
  - `object.init_pos_noise: [0.0075, 0.0075, 0.0]`
  - `numEnvs: 10`
- Reran the keyboard tuner for Middle1:
  - `./docker-run-isaacgym.sh python scripts/tune_dexh13_lightbulb_initpose.py --task Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle1 --gpu 0 --seed 42 --out outputs/initpose_tuning/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle1_current.yaml`
  - Final saved candidate from the latest user run:
    - `handRootPos: [0.082000, 0.030000, 0.239000]`
    - `handRootRPY: [3.141500, 0.439627, 3.141500]`
    - `right_index_joint_0: 0.3499999940`
    - `right_index_joint_1: 0.8852795362`
    - `right_thumb_joint_0: 0.1299999952`
    - `right_thumb_joint_2: 0.0000000000`
    - `right_thumb_joint_3: 0.5519663095`
  - Applied this latest snippet back to `configs/task/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle1.yaml`, preserving `object.init_pos_noise` and `asset.handRootPosNoise`.
- Later updated `configs/task/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle1.yaml`:
  - `asset.handRootPosZScaleComp: 0.0`
  - This removes the extra scale-dependent z lift during headed training and future PPO runs while keeping object/hand-root noise enabled.

### Remaining blocked/risky
- The headed viewer is for human visual inspection only; the negative random-PPO reward during this run is not meaningful for policy quality.
- The latest saved `middle1` init pose and `handRootPosZScaleComp: 0.0` have been applied back to the task YAML, but still need headed inspection under the added init-position noise.

### Single recommended next step
- Rerun the 10-env headed PPO view for `Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle1` to inspect the updated pose under object/hand-root noise.

---

## v2-2026-05-15 -- RealBulb Headed PPO Visualization Check

### Target milestone/subgoal
- Verify local RealBulb task files exist and launch a headed PPO viewer for visual inspection of the real-size bulb asset and inherited CoDrive init pose.

### What changed (files + behavior impact)
- No source/config files were changed in this step.
- Generated temporary visualization output under:
  - `outputs/Dexh13HoraLightbulb_teacher/vis_realbulb_ppo_headed_tmp/`

### What was verified (commands + key outcomes)
- Confirmed local files exist:
  - `assets/screw/realbulb/0000_lightbulb.urdf`
  - `configs/task/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveRealBulb.yaml`
  - `configs/train/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveRealBulb.yaml`
- Confirmed RealBulb task values:
  - `object.type: screw_realbulb`
  - `baseObjScale: 1.00`
  - `randomizeScaleList: [0.975, 1.025]`
  - `object.init_pos_noise: [0.0, 0.0, 0.0]`
  - `asset.handRootPosNoise: [0.0, 0.0, 0.0]`
- Launched headed PPO viewer:
  - `./docker-run-isaacgym.sh python train.py task=Dexh13HoraLightbulbSim2RealTwoFingerCoDriveRealBulb headless=False seed=42 train.algo=PPO wandb_activate=False task.env.numEnvs=10 train.ppo.num_actors=10 train.ppo.minibatch_size=120 train.ppo.max_agent_steps=1000000 train.ppo.output_name=Dexh13HoraLightbulb_teacher/vis_realbulb_ppo_headed_tmp graphics_device_id=0 task.env.randomization.randomizeScale=False task.env.object.init_pos_noise=[0.0,0.0,0.0] task.env.termination.log=True`
  - Outcome: task loaded `screw_realbulb`, created 10 envs, started PPO with viewer, then exited cleanly.

### Remaining blocked/risky
- This was only a visualization/probe run. It fixed `randomizeScale=False` to inspect the base real-size bulb, so it does not show the full `0.95-1.05` training scale distribution.

### Single recommended next step
- If the base RealBulb geometry/init pose looks acceptable, rerun the same headed PPO with `task.env.randomization.randomizeScale=True` or open the keyboard tuner for RealBulb-specific init-pose adjustment.

---

## v2-2026-05-15 -- CoDriveExper Initpose Tuning Variant Added

### Target milestone/subgoal
- Create an isolated CoDrive-derived task YAML for hand init-pose tuning without modifying the frozen original CoDrive deployment/training config.

### What changed (files + behavior impact)
- Added `configs/task/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveExper.yaml`.
  - Copied from `Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.yaml`.
  - Only changed `eval_cache_name` to `sim2real_twofinger_codrive_exper`.
- Added `configs/train/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveExper.yaml`.
  - Exact copy of the original CoDrive train YAML so Hydra can resolve `train: ${task}`.
- Ran the keyboard init-pose tuner once on the new task and saved a candidate snippet to:
  - `outputs/initpose_tuning/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveExper_current.yaml`
- The saved candidate was not applied back to the task YAML yet.

### What was verified (commands + key outcomes)
- Verified task diff against original CoDrive is only `eval_cache_name`; train YAML has no diff.
- Tuner launch command succeeded:
  - `./docker-run-isaacgym.sh python scripts/tune_dexh13_lightbulb_initpose.py --task Dexh13HoraLightbulbSim2RealTwoFingerCoDriveExper --gpu 0 --seed 42 --out outputs/initpose_tuning/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveExper_current.yaml`
- Confirmed the new task loads `screw_contactviz` and starts an IsaacGym viewer.
- Current tuner setup is a clean one-env view: object init noise and hand root noise are zero in the YAML, and the tuner also disables mass/COM/friction/scale/PD-gain randomization and external random force through overrides.

### Remaining blocked/risky
- The user wants to rerun the tuner locally to watch terminal print output directly before choosing the final `handRootPos`, `handRootRPY`, and `handInitPose`.

### Single recommended next step
- Rerun the tuner command locally, press `C`/`O` to inspect/save values, then apply the chosen snippet under `env.asset` in `Dexh13HoraLightbulbSim2RealTwoFingerCoDriveExper.yaml`.

---

## v2-2026-05-13 -- CoDrive BC/DAgger Baseline Curve Pack

### Target milestone/subgoal
- Collect the original non-noise CoDrive BC/LatentBC and DAgger student training evidence for baseline comparison, including W&B-like reward curves and reproducibility parameters.

### What changed (files + behavior impact)
- Added local comparison pack:
  - `sim2real/deploy/codrive/baseline_curves/`
- Also copied the same pack to the top-level results area:
  - `baseline_results/codrive_bc_dagger_baseline_curves/`
  - added `baseline_results/README.md` as the baseline result index
- Added four per-baseline CoDrive key-metric CSVs under `baseline_results/codrive/`, matching the existing `run,tag,step,value,wall_time` format:
  - `bc_key_metrics.csv`
  - `dagger_key_metrics.csv`
  - `padapt_key_metrics.csv`
  - `dotpg_key_metrics.csv`
  - `baseline_metric_sources.tsv`
  - `README.md`
- Pack contents:
  - copied TensorBoard event files under `raw/`
  - exported scalar CSVs under `csv/`
  - report-ready PNG plots under `plots/`
  - full run configs and CoDrive task/train YAMLs under `configs/`
  - compact `summary.csv`, `key_training_params.csv`, `manifest.json`, and `README.md`
- Selected runs:
  - BC/LatentBC: `outputs/Dexh13HoraLightbulb_student_bc_codrive/codrive_bc_opt_iter3_evalsel_latent_evalsel_s42`
  - DAgger: `outputs/Dexh13HoraLightbulb_student_dagger_codrive/codrive_dagger_opt_iter3_evalsel_pure_replay_s42`

### What was verified (commands + key outcomes)
- Parsed TensorBoard scalars with `tensorboard.backend.event_processing.event_accumulator`.
- Generated plots:
  - `plots/codrive_bc_dagger_report_panels.png`
  - `plots/codrive_bc_dagger_train_reward.png`
  - `plots/codrive_bc_dagger_student_eval_reward.png`
  - `plots/codrive_bc_dagger_eval_select_score.png`
  - `plots/codrive_bc_dagger_eval_select_avg_reward.png`
  - `plots/codrive_bc_dagger_total_loss.png`
- `python -m json.tool sim2real/deploy/codrive/baseline_curves/manifest.json` passed.
- Headline metrics in `summary.csv`:
  - BC/LatentBC: TB train return max `852.115`, eval-select best score `2.541`.
  - DAgger: TB eval reward max `1384.226`, eval-select best score `3.763`.
- Per-baseline key-metric row counts:
  - BC/LatentBC: `47,746`
  - DAgger: `58,516`
  - PAdapt: `399,776`
  - DOTPG: `20`

### Remaining blocked/risky
- These are training/eval-select curves, not a unified multi-seed deployment eval table. Use them for training-process and baseline-method comparison, then pair them with fixed-step eval results for final paper ranking.
- DOTPG's full TensorBoard training curve was not available in the current synced local artifacts; its key-metric CSV is converted from selected `dual_bc5` train/eval TSV summaries.

### Single recommended next step
- Use `baseline_results/codrive/` for four-baseline CSV analysis and `baseline_results/codrive_bc_dagger_baseline_curves/plots/codrive_bc_dagger_report_panels.png` for the quick BC-vs-DAgger visual entry point.

---

## v2-2026-05-13 -- RealBulb Cloud PPO/PAdapt/DOTPG Pipeline Started

### Target milestone/subgoal
- Train the RealBulb CoDrive task on cloud with a 2h PPO teacher phase followed by 2h parallel PAdapt and DOTPG student distillation.

### What changed (files + behavior impact)
- Added cloud pipeline script:
  - `outputs/cloud_pipeline_realbulb_ppo2h_padapt_dotpg2h/run_realbulb_ppo2h_padapt_dotpg2h.sh`
- Synced to cloud:
  - `assets/screw/realbulb/0000_lightbulb.urdf`
  - `configs/task/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveRealBulb.yaml`
  - `configs/train/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveRealBulb.yaml`
  - the pipeline script above
- Pipeline behavior:
  - PPO teacher uses `timeout 7200` with `task.env.numEnvs=12288`, `train.ppo.minibatch_size=24576`.
  - After PPO exits with expected timeout `124`, the script selects the best `stage1_nn/best_reward_*.pth`.
  - PAdapt and DOTPG run in parallel, each under its own `timeout 7200`.
  - Start/end timestamps, exact commands, selected teacher path, and exit statuses are written under the pipeline status directory.

### What was verified (commands + key outcomes)
- Cloud RealBulb smoke passed:
  - `timeout 180 python train.py task=Dexh13HoraLightbulbSim2RealTwoFingerCoDriveRealBulb ... task.env.numEnvs=4 train.ppo.max_agent_steps=24`
  - Outcome: cloud loaded `screw_realbulb`, generated initial poses at scales `0.975` and `1.025`, and exited with `max steps achieved`.
- Active cloud tmux session launched:
  - `realbulb_ppo2h_students2h_20260513_031105`
  - Pipeline path:
    `outputs/cloud_pipeline_realbulb_ppo2h_padapt_dotpg2h/realbulb_s42_20260513_031105/`
  - Convenience link:
    `outputs/cloud_pipeline_realbulb_ppo2h_padapt_dotpg2h/latest`
- PPO phase confirmed live:
  - `status/phase.txt = ppo`
  - GPU about `14359 MiB / 24564 MiB`, around `80%` utilization.
  - PPO log reached real training lines, e.g. `Agent Steps: 0005M ... Current Best: 57.18`.

### Remaining blocked/risky
- PPO is still running; no teacher checkpoint has been selected yet.
- Student phases have not started yet; they should start automatically after PPO reaches the 2h timeout.
- SSH to this cloud endpoint intermittently returns `kex_exchange_identification`, so status checks may need a retry.

### Single recommended next step
- Monitor `outputs/cloud_pipeline_realbulb_ppo2h_padapt_dotpg2h/latest/status/phase.txt` until PPO exits, then confirm PAdapt and DOTPG both start and later finish with expected timeout status `124`.

---

## v2-2026-05-12 -- Real-Size CoDrive Bulb Asset Added

### Target milestone/subgoal
- Add a real-size lightbulb asset aligned to the measured physical bulb and create a CoDrive task variant with small 0.95-1.05 object-scale randomization.

### What changed (files + behavior impact)
- Added `assets/screw/realbulb/0000_lightbulb.urdf`.
  - Uses existing `contact0.stl` and `contact1.stl` for both visual and collision geometry.
  - The viewer now shows the actual low-poly contact surface, matching the contactviz convention.
  - Meshes are anisotropically scaled so the bulb is about `140 mm` long and `60 mm` max diameter before actor-scale randomization.
  - Object type is selectable as `task.env.object.type=screw_realbulb`.
- Added `configs/task/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveRealBulb.yaml`.
  - Inherits the original CoDrive task structure.
  - `eval_cache_name: sim2real_twofinger_codrive_realbulb`
  - `object.type: screw_realbulb`
  - `baseObjScale: 1.00`
  - `randomizeScaleList: [0.975, 1.025]`
  - `randomizeScaleMin/Max` and `randomizeScaleLower/Upper`: `0.95/1.05`
- Added `configs/train/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveRealBulb.yaml`.
  - Exact copy of the original CoDrive train YAML so Hydra can resolve `train: ${task}` for the new task.

### What was verified (commands + key outcomes)
- XML/YAML parse check passed for the new URDF/task/train files.
- Geometry calculation confirmed:
  - visual/collision contact-pair scaled size: about `140.0 x 60.0 x 60.0 mm`
- Local IsaacGym smoke passed:
  - `./docker-run-isaacgym.sh timeout 180 scripts/run_with_cleanup.sh python train.py task=Dexh13HoraLightbulbSim2RealTwoFingerCoDriveRealBulb headless=True seed=42 train.algo=PPO wandb_activate=False task.env.numEnvs=4 train.ppo.minibatch_size=12 train.ppo.max_agent_steps=24 train.ppo.output_name=Dexh13HoraLightbulb_teacher/smoke_codrive_realbulb_tmp task.env.termination.log=True`
  - Outcome: loaded `screw_realbulb`, generated initial poses for scales `0.975` and `1.025`, and exited with `max steps achieved`.
- `git diff --check` passed for the new asset/config files and handoff docs.

### Remaining blocked/risky
- The real-size bulb is taller than the original simulated bulb, so the inherited CoDrive hand root/init pose may need visual tuning before long training or deployment.
- The real-size visual is intentionally low-poly because it now matches collision geometry exactly; use it for contact inspection rather than presentation renders.
- The new `screw_realbulb` asset currently has no precomputed `.npy` point cloud, matching the previous `screw_contactviz` behavior that falls back to an approximate cylinder point cloud.

### Single recommended next step
- Open a headed init-pose or short visual rollout with `task=Dexh13HoraLightbulbSim2RealTwoFingerCoDriveRealBulb` to confirm finger placement against the taller 140 mm bulb before launching a long PPO/student run.

---

## v2-2026-05-12 -- Original CoDrive Student Deploy Packages Completed

### Target milestone/subgoal
- Package original non-noise CoDrive student distillation checkpoints under `sim2real/deploy/codrive`, all aligned to the frozen roughly-4000-reward PPO teacher.

### What changed (files + behavior impact)
- Created/updated original CoDrive deploy folders:
  - `sim2real/deploy/codrive/padapt_deploy/`
  - `sim2real/deploy/codrive/dotpg_deployv1/`
  - `sim2real/deploy/codrive/diffusion_latent_deploy/`
  - `sim2real/deploy/codrive/consistency_latent_deploy/`
  - `sim2real/deploy/codrive/flow_matching_deploy/`
  - `sim2real/deploy/codrive/diffusion_action_chunk_deploy/`
  - `sim2real/deploy/codrive/bc_latentbc_deploy/`
  - `sim2real/deploy/codrive/dagger_deploy/`
- Every package contains the same four-file deploy format:
  - `Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.task.yaml`
  - `Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.train.yaml`
  - `best_reward_4159.37.pth`
  - `model_best.ckpt`

### What was verified (commands + key outcomes)
- Confirmed all deploy packages have exactly four files.
- Confirmed every package's teacher/YAML files hash-match the frozen original CoDrive package under `sim2real/codrive`.
- All packages use teacher `sim2real/codrive/best_reward_4159.37.pth`, the original CoDrive PPO baseline with max reward `4159.37`.
- Student checkpoint selection:
  - PAdapt: `outputs/Dexh13HoraLightbulb_student_padapt_sim2real_twofinger_codrive/sim2real_twofinger_codrive_padapt_s42_2h/stage2_nn/model_best.ckpt`
  - DOTPG: `sim2real/codrive/dotpg_bc5/model_best.ckpt`
  - Diffusion latent: continue-2h `model_best.ckpt`
  - Consistency latent: 1h `model_best.ckpt` because short deploy eval was stronger than continue-2h
  - Flow matching: 1h `model_best.ckpt` because short deploy eval was stronger than continue-2h
  - Diffusion action chunk: continue-2h `model_best_student_reward.ckpt`, copied as deploy `model_best.ckpt`
  - BC/LatentBC: opt-iter3 eval-select `model_best_deploy.ckpt`
  - DAgger: opt-iter3 pure-replay eval-select `model_best_deploy.ckpt`

### Remaining blocked/risky
- These are packaged deployment candidates, not final ranked real-robot selections. Visual checks and a unified fixed-step eval should be used before choosing one.

### Single recommended next step
- For original bulb CoDrive deployment, visualize `consistency_latent_deploy`, `flow_matching_deploy`, `padapt_deploy`, and `dotpg_deployv1` first.

---

## v2-2026-05-12 -- Deploy Packages Grouped By Task

### Target milestone/subgoal
- Reorganize `sim2real/deploy` so deployment packages are grouped by task family rather than all living at the top level.

### What changed (files + behavior impact)
- Moved the non-thesis CoDrive deploy package to:
  - `sim2real/deploy/codrive/dotpg_deployv1/`
- Moved the CoDriveThesis deploy packages to:
  - `sim2real/deploy/codrive_thesis/bc_latentbc_deploy/`
  - `sim2real/deploy/codrive_thesis/dagger_deploy/`
  - `sim2real/deploy/codrive_thesis/dotpg_deployv2/`
  - `sim2real/deploy/codrive_thesis/padapt_deploy/`
  - `sim2real/deploy/codrive_thesis/diffusion_latent_deploy/`
  - `sim2real/deploy/codrive_thesis/consistency_latent_deploy/`
  - `sim2real/deploy/codrive_thesis/flow_matching_deploy/`
  - `sim2real/deploy/codrive_thesis/diffusion_action_chunk_deploy/`
  - `sim2real/deploy/codrive_thesis/purebc_deploy/`
- Left the CoDriveNoise deploy packages grouped under:
  - `sim2real/deploy/codrive_noise/`

### What was verified (commands + key outcomes)
- Top-level `sim2real/deploy` now contains only task-family folders:
  - `codrive`
  - `codrive_thesis`
  - `codrive_noise`
- Every nested deploy package still contains exactly four files.

### Remaining blocked/risky
- Any old commands pointing directly at `sim2real/deploy/padapt_deploy` or similar need the new `codrive_thesis/` or `codrive_noise/` prefix.

### Single recommended next step
- Use the task-family path when selecting deployment artifacts: `sim2real/deploy/codrive`, `sim2real/deploy/codrive_thesis`, or `sim2real/deploy/codrive_noise`.

---

## v2-2026-05-12 -- CoDriveNoise Student Deploy Packages Grouped

### Target milestone/subgoal
- Sync all CoDriveNoise student distillation checkpoints from cloud/local outputs and package them for deployment under `sim2real/deploy/codrive_noise`.

### What changed (files + behavior impact)
- Synced cloud CoDriveNoise student outputs locally:
  - `outputs/Dexh13HoraLightbulb_student_diffusion_latent_codrive_noise/`
  - `outputs/Dexh13HoraLightbulb_student_consistency_codrive_noise/`
  - `outputs/Dexh13HoraLightbulb_student_flow_matching_codrive_noise/`
  - `outputs/Dexh13HoraLightbulb_student_diffusion_action_chunk_codrive_noise/`
  - `outputs/Dexh13HoraLightbulb_student_purebc_codrive_noise/`
  - `outputs/Dexh13HoraLightbulb_student_dagger_codrive_noise/`
  - existing local/cloud-synced `padapt`, `dotpg`, teacher, and YAML files were also verified.
- Added deploy-format folders, each containing exactly task YAML, train YAML, PPO teacher pth, and student ckpt:
  - `sim2real/deploy/codrive_noise/codrive_noise_padapt_deploy/`
  - `sim2real/deploy/codrive_noise/codrive_noise_dotpg_deploy/`
  - `sim2real/deploy/codrive_noise/codrive_noise_diffusion_latent_deploy/`
  - `sim2real/deploy/codrive_noise/codrive_noise_consistency_latent_deploy/`
  - `sim2real/deploy/codrive_noise/codrive_noise_flow_matching_deploy/`
  - `sim2real/deploy/codrive_noise/codrive_noise_diffusion_action_chunk_deploy/`
  - `sim2real/deploy/codrive_noise/codrive_noise_purebc_deploy/`
  - `sim2real/deploy/codrive_noise/codrive_noise_dagger_deploy/`
  - `sim2real/deploy/codrive_noise/codrive_noise_bc_latentbc_deploy/`

### What was verified (commands + key outcomes)
- Source checkpoint existence was verified for all nine student variants:
  - PAdapt, DOTPG, diffusion latent, consistency latent, flow matching, diffusion action chunk, PureBC, DAgger, and BC/LatentBC.
- Deploy package check confirmed each `sim2real/deploy/codrive_noise/codrive_noise_*_deploy` directory has 4 files:
  - `Dexh13HoraLightbulbSim2RealTwoFingerCoDriveNoise.task.yaml`
  - `Dexh13HoraLightbulbSim2RealTwoFingerCoDriveNoise.train.yaml`
  - `best_reward_4076.32.pth`
  - `model_best.ckpt`
- Verified no `sim2real/deploy/codrive_noise_*_deploy` directories remain directly under `sim2real/deploy`.

### Remaining blocked/risky
- These are packaged from training-selected `model_best.ckpt` files. Final deployment ranking should still use the unified fixed-step eval/visualization protocol.

### Single recommended next step
- Run or sync a unified fixed-step eval table for these nine CoDriveNoise deploy packages before choosing the final real-robot candidate.

---

## v2-2026-05-12 -- CoDriveNoise Remaining Baselines Completed

### Target milestone/subgoal
- Check completion of the CoDriveNoise remaining-baseline 3h distillation batch.

### What changed (files + behavior impact)
- No code/config changes in this status check.
- Recorded final cloud and local checkpoint status for the remaining CoDriveNoise baselines.

### What was verified (commands + key outcomes)
- Cloud batch completed:
  - `outputs/cloud_pipeline_codrive_noise_remaining_baselines_3h/codrive_noise_remaining_s42_3h_20260512_013000/`
  - `phase.txt = done`
  - no active `python train.py`
  - RTX 4090 D idle at about `1 MiB / 24564 MiB`
  - exit statuses all `124`, expected for 3h timeout:
    `diffusion_latent`, `consistency_latent`, `flow_matching_latent`, `diffusion_action_chunk`, `purebc`, `dagger`
- Cloud parsed training rewards:
  - diffusion latent: `4019.67`
  - consistency latent: `3840.09`
  - flow matching: `3850.96`
  - diffusion action chunk: `1341.73`
  - purebc: `3305.94`
  - DAgger: train return last `1655.33`, best student eval `979.06`
- Cloud checkpoint folders exist:
  - `outputs/Dexh13HoraLightbulb_student_diffusion_latent_codrive_noise/codrive_noise_remaining_s42_3h_20260512_013000/stage2_diffusion_nn/model_best.ckpt`
  - `outputs/Dexh13HoraLightbulb_student_consistency_codrive_noise/codrive_noise_remaining_s42_3h_20260512_013000/stage2_consistency_nn/model_best.ckpt`
  - `outputs/Dexh13HoraLightbulb_student_flow_matching_codrive_noise/codrive_noise_remaining_s42_3h_20260512_013000/stage2_flow_nn/model_best.ckpt`
  - `outputs/Dexh13HoraLightbulb_student_diffusion_action_chunk_codrive_noise/codrive_noise_remaining_s42_3h_20260512_013000/stage2_diffusion_action_chunk_nn/`
  - `outputs/Dexh13HoraLightbulb_student_purebc_codrive_noise/codrive_noise_remaining_s42_3h_20260512_013000/stage2_bc_nn/model_best.ckpt`
  - `outputs/Dexh13HoraLightbulb_student_dagger_codrive_noise/codrive_noise_remaining_s42_3h_20260512_013000_dagger_pure_replay/dagger_nn/`
- Local BC/LatentBC completed and produced checkpoints:
  - `outputs/Dexh13HoraLightbulb_student_bc_codrive_noise/codrive_noise_remaining_s42_3h_20260512_013000_latentbc/bc_nn/model_best.ckpt`
  - `model_best_deploy.ckpt`, `model_best_eval.ckpt`, `model_best_student_eval.ckpt`, and `model_last.ckpt` also exist.
- Local BC eval-select history:
  - max `avg_reward = 3.47085`
  - max `score = 2.98257`

### Remaining blocked/risky
- The local wrapper status file remained stale at `running` because the Docker container exited after timeout without the wrapper writing summary/status. Checkpoint timestamps and absence of Docker/train processes confirm local BC finished.
- Training reward is not sufficient for final deployment selection. Use fixed-step eval and visualization before ranking.

### Single recommended next step
- Sync the six cloud baseline outputs locally, then run a unified fixed-step eval over all CoDriveNoise students: PPO teacher, PAdapt, DOTPG, diffusion latent, consistency, flow matching, action chunk, PureBC, DAgger, and BC/LatentBC.

---

## v2-2026-05-12 -- CoDriveNoise Remaining Baselines Started

### Target milestone/subgoal
- Use the CoDriveNoise PPO teacher to distill the remaining student baselines while respecting a 10h wall-clock budget.

### What changed (files + behavior impact)
- Added cloud launcher:
  - `outputs/cloud_pipeline_codrive_noise_remaining_baselines_3h/run_cloud_diffusion_purebc_3h.sh`
  - runs `DiffusionLatentStudent`, `ConsistencyLatentStudent`, `FlowMatchingLatentStudent`, `DiffusionActionChunkStudent`, and `PureBC` in parallel for 3h each.
- Added cloud DAgger launcher:
  - `outputs/cloud_pipeline_codrive_noise_remaining_baselines_3h/run_cloud_dagger_3h.sh`
- Added local launcher:
  - `outputs/local_pipeline_codrive_noise_remaining_baselines_3h/run_local_bc_dagger_3h.sh`
- Synced the CoDriveNoise PPO teacher locally:
  - `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_noise/codrive_noise_s42_20260511_094738_ppo3h/stage1_nn/best_reward_4076.32.pth`

### What was verified (commands + key outcomes)
- Syntax checks passed:
  - `bash -n outputs/cloud_pipeline_codrive_noise_remaining_baselines_3h/run_cloud_diffusion_purebc_3h.sh`
  - `bash -n outputs/cloud_pipeline_codrive_noise_remaining_baselines_3h/run_cloud_dagger_3h.sh`
  - `bash -n outputs/local_pipeline_codrive_noise_remaining_baselines_3h/run_local_bc_dagger_3h.sh`
- Active run tag:
  - `codrive_noise_remaining_s42_3h_20260512_013000`
- Cloud active tmux sessions:
  - `codrive_noise_remaining_s42_3h_20260512_013000_cloud`
  - `codrive_noise_remaining_s42_3h_20260512_013000_dagger`
- Cloud startup check:
  - RTX 4090 D about `17026 MiB / 24564 MiB`, about `98%`
  - all six cloud jobs had logs and no tail `Traceback`, OOM, missing-key/size-mismatch, or segfault patterns.
- Local startup check:
  - RTX 4080 SUPER about `4.0GB / 16GB`
  - local BC/LatentBC Docker job entered training and reported teacher sanity reward around `4224.80`, student pretrain/eval alive, and no fatal error.

### Remaining blocked/risky
- Local `docker-run-isaacgym.sh` did not cleanly allow two concurrent local Docker training containers. The local launcher started BC, but DAgger was moved to cloud to keep wall-clock efficiency.
- Local BC output is best monitored via `docker logs` while the Docker container is active; the parent `nohup` wrapper exited after launching the Docker process, so final status may need to be inferred from the timeout process/checkpoint files.
- Current training should finish after 3h wall-clock unless interrupted; status files/checkpoints still need final verification.

### Single recommended next step
- Let the active CoDriveNoise remaining-baseline jobs complete, then parse summaries, sync cloud outputs locally, and run a unified fixed-step eval before choosing deployment candidates.

---

## v2-2026-05-12 -- CoDriveNoise Student Checkpoints Synced Local

### Target milestone/subgoal
- Sync the CoDriveNoise PAdapt and DOTPG student checkpoints from cloud to local for visualization.

### What changed (files + behavior impact)
- Synced cloud outputs to the local workspace:
  - `outputs/Dexh13HoraLightbulb_student_padapt_codrive_noise/codrive_noise_s42_20260511_094738_padapt3h_from_ppo3h/`
  - `outputs/Dexh13HoraLightbulb_student_dotpg_codrive_noise/codrive_noise_s42_20260511_094738_dotpg3h_from_ppo3h_dual_bc5/`
  - `configs/task/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveNoise.yaml`
  - `configs/train/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveNoise.yaml`
  - `outputs/cloud_pipeline_codrive_noise_ppo3h_padapt_dotpg3h/codrive_noise_s42_20260511_094738/selected_teacher_ckpt.txt`

### What was verified (commands + key outcomes)
- Local checkpoint paths now exist:
  - PAdapt:
    `outputs/Dexh13HoraLightbulb_student_padapt_codrive_noise/codrive_noise_s42_20260511_094738_padapt3h_from_ppo3h/stage2_nn/model_best.ckpt`
  - DOTPG:
    `outputs/Dexh13HoraLightbulb_student_dotpg_codrive_noise/codrive_noise_s42_20260511_094738_dotpg3h_from_ppo3h_dual_bc5/student_output/dotpg_nn/model_best.ckpt`
- Local CoDriveNoise YAML contains:
  - `eval_cache_name: sim2real_twofinger_codrive_noise`
  - `object.init_pos_noise: [0.005, 0.005, 0.0]`
  - `asset.handRootPosNoise: [0.001, 0.001, 0.001]`

### Remaining blocked/risky
- These checkpoints have not yet been locally visualized in this sync step.

### Single recommended next step
- Run headed local visualization for PAdapt and DOTPG on `Dexh13HoraLightbulbSim2RealTwoFingerCoDriveNoise`.

---

## v2-2026-05-12 -- CoDriveNoise Cloud Pipeline Completed

### Target milestone/subgoal
- Check completion status for the CoDriveNoise cloud pipeline:
  3h PPO teacher, then 3h PAdapt and 3h DOTPG students from that PPO.

### What changed (files + behavior impact)
- No code/config behavior changed in this status check.
- Updated cloud/local handoff records with final completion status, selected checkpoints, and parsed rewards.

### What was verified (commands + key outcomes)
- Cloud status:
  - `status/phase.txt = done`
  - no active `python train.py`
  - RTX 4090 D idle, about `1 MiB / 24564 MiB`, `0%`
- Exit statuses:
  - PPO `124`
  - PAdapt `124`
  - DOTPG `124`
  - all are expected timeout completions for the requested 3h phases.
- Selected/result checkpoints:
  - PPO:
    `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_noise/codrive_noise_s42_20260511_094738_ppo3h/stage1_nn/best_reward_4076.32.pth`
  - PAdapt:
    `outputs/Dexh13HoraLightbulb_student_padapt_codrive_noise/codrive_noise_s42_20260511_094738_padapt3h_from_ppo3h/stage2_nn/model_best.ckpt`
  - DOTPG:
    `outputs/Dexh13HoraLightbulb_student_dotpg_codrive_noise/codrive_noise_s42_20260511_094738_dotpg3h_from_ppo3h_dual_bc5/student_output/dotpg_nn/model_best.ckpt`
- Parsed actual training progress lines:
  - PPO max/current best: `4076.32`
  - PAdapt max/current best: `3501.40`
  - DOTPG max/current best: `2631.82`
- Tail scans of final log sections found no runtime/OOM/missing-key/size-mismatch/segfault patterns.

### Remaining blocked/risky
- The logs include startup dirty-git-diff text from before `DEXSCREW_SKIP_GIT_DIFF=1` was added; broad grep can find unrelated historical reward/error text. Real rewards above were parsed only from actual `Agent Steps: ... Current Best` progress lines.
- These are training rewards only; for paper/deploy comparison they still need fixed-step eval and/or local visualization.

### Single recommended next step
- Sync the PPO/PAdapt/DOTPG CoDriveNoise checkpoints and YAMLs to local, then visualize the PAdapt and DOTPG students against the noised PPO teacher.

---

## v2-2026-05-11 -- CoDriveNoise Cloud PPO/PAdapt/DOTPG Pipeline Launched

### Target milestone/subgoal
- Add a separate CoDrive noise ablation YAML and launch cloud training:
  PPO teacher for 3h, then PAdapt and DOTPG students for 3h each from that PPO.

### What changed (files + behavior impact)
- Restored the base CoDrive YAML to no init-pose noise:
  - `configs/task/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.yaml`
  - `object.init_pos_noise: [0.0, 0.0, 0.0]`
  - `asset.handRootPosNoise: [0.0, 0.0, 0.0]`
- Added the new noise variant:
  - `configs/task/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveNoise.yaml`
  - `configs/train/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveNoise.yaml`
  - `eval_cache_name: sim2real_twofinger_codrive_noise`
  - `object.init_pos_noise: [0.005, 0.005, 0.0]`
  - `asset.handRootPosNoise: [0.001, 0.001, 0.001]`
- Added cloud pipeline script:
  - `outputs/cloud_pipeline_codrive_noise_ppo3h_padapt_dotpg3h/run_codrive_noise_ppo3h_padapt_dotpg3h.sh`
  - PPO is wrapped in its own `timeout 10800`.
  - PAdapt and DOTPG are each wrapped in their own `timeout 10800`.
  - The script selects the PPO teacher checkpoint explicitly from `stage1_nn/best_reward_*.pth` before starting students.
  - The script records commands, exit statuses, GPU usage, and selected checkpoint under the pipeline directory.

### What was verified (commands + key outcomes)
- Local YAML smoke passed:
  - `./docker-run-isaacgym.sh timeout 180 scripts/run_with_cleanup.sh python train.py task=Dexh13HoraLightbulbSim2RealTwoFingerCoDriveNoise ... task.env.numEnvs=4 train.ppo.max_agent_steps=24`
  - Outcome: completed with `max steps achieved`.
- Synced config/script/source to cloud and verified cloud-side YAML values:
  - base CoDrive remained zero-noise
  - CoDriveNoise had object noise `[0.005, 0.005, 0.0]` and hand root noise `[0.001, 0.001, 0.001]`
- Cloud smoke passed after sourcing the cloud IsaacGym activation:
  - same task, `numEnvs=4`, `max_agent_steps=24`
  - Outcome: `smoke_status=0`.
- Launched active cloud tmux session:
  - `codrive_noise_ppo3h_students3h_20260511_174738`
  - Pipeline path:
    `outputs/cloud_pipeline_codrive_noise_ppo3h_padapt_dotpg3h/codrive_noise_s42_20260511_094738/`
- PPO startup was confirmed:
  - phase: `ppo`
  - GPU: about `14327 MiB / 24564 MiB`, about `67%` utilization
  - latest startup check observed checkpoint:
    `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_noise/codrive_noise_s42_20260511_094738_ppo3h/stage1_nn/best_reward_70.57.pth`

### Remaining blocked/risky
- The 6h total pipeline is still running on cloud; PPO has not yet reached its 3h timeout in this handoff.
- The currently active run was launched before the script was patched with `DEXSCREW_SKIP_GIT_DIFF=1`, so the PPO log has startup git-diff noise. The running training itself is unaffected.
- Need verify after PPO exits that `selected_teacher_ckpt.txt` points to the intended best teacher, then that both PAdapt and DOTPG start.

### Single recommended next step
- Monitor `outputs/cloud_pipeline_codrive_noise_ppo3h_padapt_dotpg3h/latest/status/` on cloud until PPO exits, then confirm PAdapt and DOTPG complete their own 3h timeouts and sync the best checkpoints back for visualization.

---

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

## v2-202 (2026-04-30) -- DOTPG 2h Distillation From Thumbstable PPO Teacher Completed

### Target milestone/subgoal
- Distill a DOTPG student for 2h from the sim2real two-finger thumbstable PPO teacher:
  `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger/sim2real_twofinger_thumbstable/stage1_nn/best_reward_6103.04.pth`

### What changed (files + behavior impact)
- Updated `scripts/dexh13_lightbulb_student_dotpg_sim2real_twofinger.sh`.
  - Converted the DOTPG defaults from smoke-test settings to real distillation settings:
    - `train.dotpg.warmup_steps=5000`
    - `train.dotpg.adapt_warmup_steps=1000`
    - `train.dotpg.bc_coef=2.5`
    - `train.dotpg.bc_pretrain_steps=2000`
    - `train.dotpg.bc_pretrain_lr=0.0003`
    - `train.dotpg.bc_batch_size=512`
    - `train.dotpg.online_expert=True`
  - Rationale: the first attempted run inherited `bc_pretrain_steps=0`, so it behaved like a near-random DOTPG policy and reached only `Current Best: 1.76` early.

### What was verified (commands + key outcomes)
- Static checks:
  - `bash -n scripts/dexh13_lightbulb_student_dotpg_sim2real_twofinger.sh`
    - Outcome: pass.
  - `git diff --check -- scripts/dexh13_lightbulb_student_dotpg_sim2real_twofinger.sh dexscrew/dotpg/dotpg.py train.py dexscrew/algo/student/__init__.py dexscrew/dotpg/__init__.py`
    - Outcome: pass.
- Final 2h DOTPG command:
  - Cache: `sim2real_twofinger_thumbstable_dotpg_bc_s42_2h`
  - Output dir:
    `outputs/Dexh13HoraLightbulb_student_dotpg_sim2real_twofinger/sim2real_twofinger_thumbstable_dotpg_bc_s42_2h`
  - Teacher checkpoint:
    `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger/sim2real_twofinger_thumbstable/stage1_nn/best_reward_6103.04.pth`
  - Result:
    - `exit_status=124`, expected from the 7200s timeout.
    - Adapt warmup completed: `loss: 0.0287`.
    - Expert buffer collected: `80000` samples.
    - BC pretrain completed: loss decreased from about `0.0751` to about `0.0382`.
    - Final observed DOTPG `Current Best`: `1426.51`.
    - No Traceback, CUDA OOM, FileNotFoundError, or segfault was observed in the final log scan.
- Produced files:
  - `student_output/dotpg_nn/model_best.ckpt`
  - `student_output/dotpg_tb/events.out.tfevents...`
  - `expert_buffer_student_raw.pt`
  - `config_043004_21e8eff.yaml`
  - `train_2h.log`
- Cleanup:
  - No residual process specific to `sim2real_twofinger_thumbstable_dotpg_bc_s42_2h` remained after timeout.
  - A separate user/visualization process for `Dexh13HoraLightbulbSim2RealTwoFingerCoDrive` was still active on GPU, so no additional DOTPG restore/test was launched to avoid interfering with the viewer.

### Local conclusion
- DOTPG is now runnable for the requested sim2real thumbstable teacher and produces a valid student checkpoint.
- With BC/adapt warmup, DOTPG improved steadily from near zero to `1426.51`, but plateaued far below:
  - PPO teacher: `6103.04`
  - previous PAdapt student: around `4261` final observed best in the 3.5h local run.
- Current DOTPG result should be treated as an algorithm-support/ablation checkpoint, not as the preferred sim2real deployment student.

### Remaining blocked/risky
- The DOTPG `model_best.ckpt` has not yet been visually restored because another GPU visualization process was active at completion.
- DOTPG sample efficiency is currently much lower than PAdapt in this task.
- If DOTPG is pursued further, the next tuning target is not environment reward; it is DOTPG-specific stability:
  - longer/stronger BC or TD3+BC regularization,
  - larger expert buffer,
  - `state_mode=teacher` diagnostic run to separate student adapter difficulty from DOTPG policy learning difficulty.

### Single recommended next step
- After the current visualization process exits, run the DOTPG visualizer on:
  `outputs/Dexh13HoraLightbulb_student_dotpg_sim2real_twofinger/sim2real_twofinger_thumbstable_dotpg_bc_s42_2h/student_output/dotpg_nn/model_best.ckpt`
  and compare behavior against the PAdapt student before spending more time on DOTPG tuning.

---

## v2-202 (2026-04-30) -- Sim2Real TwoFinger CoDrive YAML Smoke-Validated

### Target milestone/subgoal
- Pause the current YAML direction and create a new config based on the stable `Dexh13HoraLightbulbSim2RealTwoFinger` task that encourages true index+thumb cooperative rotation instead of index-only normal bracing plus thumb-only drive.

### What changed (files + behavior impact)
- Added task config:
  - `configs/task/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.yaml`
- Added matching train config:
  - `configs/train/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.yaml`
- The new task preserves the stable Sim2Real two-finger setup, including index+thumb action mask, bulb scale randomization around `1.15-1.25`, termination, and thumb slip guard.
- Co-drive changes:
  - `rotate_reward_scale: 3.0`, reduced from `4.5` so pure object rotation is less dominant.
  - `fingertip_tangent_reward_scale: 1.0`, increased from `0.6`.
  - `fingertip_torque_reward_scale: 2.0`, increased from `0.3`.
  - `fingertip_torque_reward.torque_clip: 4.0`, down from `8.0`, so early positive index+thumb torque is rewarded sooner.
  - two-finger contact force thresholds raised to `1.2-3.5`.
  - `active_two_finger_contact.penalty_scale: -2.5`, stronger penalty when active rotation occurs without both active contacts.
  - `opposition_grip_reward.reward_scale: 0.7`, modestly stronger stable opposed grip support.

### What was verified (commands + key outcomes)
- Bootstrap context read:
  - `docs/session_handoff_v2.md`
  - `docs/stage_acceptance_summary.md`
- Current visualizer session was stopped cleanly before changes.
- Static check:
  - `git diff --check -- configs/task/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.yaml configs/train/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.yaml`
  - Outcome: pass.
- Local IsaacGym smoke:
  - `./docker-run-isaacgym.sh timeout 180 scripts/run_with_cleanup.sh python train.py task=Dexh13HoraLightbulbSim2RealTwoFingerCoDrive headless=True seed=42 sim_device=cuda:0 rl_device=cuda:0 graphics_device_id=0 train.algo=PPO wandb_activate=False task.env.numEnvs=4 train.ppo.minibatch_size=12 train.ppo.max_agent_steps=96 train.ppo.output_name=Dexh13HoraLightbulb_teacher/smoke_codrive_tmp task.env.termination.log=True`
  - Outcome: Hydra resolved the new task/train config, the environment built with 4 envs, PPO entered the loop, and exited via `max steps achieved`.

### Local conclusion
- The new CoDrive YAML is technically runnable.
- The design intentionally uses reward shaping rather than hard trajectory forcing:
  - strict simultaneous turn-and-release may be physically over-constrained for an ellipsoid bulb;
  - min-aggregated positive torque over index+thumb directly targets the current failure mode where index only presses normally.

### Remaining blocked/risky
- The smoke validates startup only, not behavior.
- Stronger co-drive reward may reduce scalar reward or make learning slower if the policy cannot discover coordinated torque early.
- If this still learns index bracing, the next step should be a code-level coactive torque gate/penalty that explicitly multiplies positive rotate reward by min positive index+thumb torque.

### Single recommended next step
- Run a 20-30 minute PPO probe with `task=Dexh13HoraLightbulbSim2RealTwoFingerCoDrive`, then visualize whether index fingertip produces visible tangential drive instead of only normal support.

---

## v2-203 (2026-04-30) -- Cloud Sim2Real TwoFinger CoDrive PPO 2h Launched

### Target milestone/subgoal
- Sync the new CoDrive YAML to the GPU cloud machine and launch a 2h PPO teacher run.

### What changed (files + behavior impact)
- Added cloud launch script:
  - `outputs/cloud_pipeline_sim2real_twofinger_codrive_ppo2h/run_ppo_2h_aggressive.sh`
- Synced local repo inputs to:
  - `cloud-training:/root/code/dexscrew-repro/`
- Synced the pipeline script separately under:
  - `outputs/cloud_pipeline_sim2real_twofinger_codrive_ppo2h/`
- Remote run task:
  - `Dexh13HoraLightbulbSim2RealTwoFingerCoDrive`
- Remote output:
  - `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive/sim2real_twofinger_codrive_s42_2h/`

### What was verified (commands + key outcomes)
- Local checks:
  - `bash -n outputs/cloud_pipeline_sim2real_twofinger_codrive_ppo2h/run_ppo_2h_aggressive.sh`
  - `git diff --check -- configs/task/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.yaml configs/train/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.yaml outputs/cloud_pipeline_sim2real_twofinger_codrive_ppo2h/run_ppo_2h_aggressive.sh`
  - Outcome: pass.
- Remote preflight:
  - SSH target: `cloud-training` (`root@180.184.47.96:22222`).
  - `tmux` exists.
  - GPU visible: `NVIDIA GeForce RTX 4090 D`, `24564 MiB`.
  - No matching `train.py` process was running before launch.
- Remote launch:
  - tmux session:
    - `sim2real_twofinger_codrive_ppo2h`
  - command uses:
    - `timeout 7200`
    - `task.env.numEnvs=12288`
    - `train.ppo.minibatch_size=24576`
    - `num_threads=22`
    - `task.env.termination.log=True`
  - pipeline log:
    - `outputs/cloud_pipeline_sim2real_twofinger_codrive_ppo2h/latest.log`
- Startup checks:
  - Hydra resolved `Dexh13HoraLightbulbSim2RealTwoFingerCoDrive`.
  - Logged key reward settings:
    - `rotate_reward_scale: 3.0`
    - `fingertip_tangent_reward_scale: 1.0`
    - `fingertip_torque_reward_scale: 2.0`
    - `fingertip_torque_reward.torque_clip: 4.0`
  - Environment generated scale caches for `1.175` and `1.225`.
  - GPU after allocation: about `14329 MiB / 24564 MiB`.
  - Checkpoints started updating:
    - `best_reward_58.56.pth`
    - then `best_reward_73.33.pth`

### Local conclusion
- The requested 2h cloud PPO run is active and using the intended CoDrive YAML.
- The aggressive `12288/24576` resource setting allocated successfully on the 24GB cloud GPU.
- The log is noisy because `train.py` prints the dirty git diff at startup; use anchored runtime greps or checkpoint timestamps rather than plain `grep Agent Steps`.

### Remaining blocked/risky
- The 2h run has not yet completed.
- Behavior quality is unknown until the final/best checkpoint is synced and visualized.
- CoDrive rewards may initially learn slower than the previous stable two-finger YAML because thumb-only rotation is less rewarded.

### Single recommended next step
- After the expected timeout around `2026-04-30 14:06 CST`, verify `ppo_exit_status=124`, sync:
  - `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive/sim2real_twofinger_codrive_s42_2h/`
  - `outputs/cloud_pipeline_sim2real_twofinger_codrive_ppo2h/`
  then visualize the best PPO checkpoint.

---

## v2-204 (2026-04-30) -- Cloud CoDrive PPO 2h Completed, Synced, Viewer Opened

### Target milestone/subgoal
- Confirm the cloud `Dexh13HoraLightbulbSim2RealTwoFingerCoDrive` 2h PPO run finished, sync artifacts locally, and open local headed visualization.

### What changed (files + behavior impact)
- Synced cloud output to local:
  - `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive/sim2real_twofinger_codrive_s42_2h/`
  - `outputs/cloud_pipeline_sim2real_twofinger_codrive_ppo2h/`
- No source/config behavior change in this step.

### What was verified (commands + key outcomes)
- Remote completion:
  - `phase.txt`: `done`
  - `ppo_exit_status`: `124`
  - pipeline ended at `2026-04-30T06:06:33+00:00`.
- Final checkpoints synced locally:
  - `stage1_nn/best_reward_4159.37.pth`
  - `stage1_nn/ep_1000_step_0147m_reward_4059.04.pth`
  - `stage1_nn/ep_500_step_0073m_reward_3729.03.pth`
  - `stage1_nn/last.pth`
- Final observed training status:
  - around `195M` agent steps before timeout.
  - best reward `4159.37`.
- Local viewer launched with:
  - `task=Dexh13HoraLightbulbSim2RealTwoFingerCoDrive`
  - `checkpoint=outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive/sim2real_twofinger_codrive_s42_2h/stage1_nn/best_reward_4159.37.pth`
  - `headless=False`
  - `task.env.numEnvs=1`
  - action/obs noise and random force disabled.

### Local conclusion
- The requested 2h cloud PPO completed normally and artifacts are local.
- The local viewer is running against the correct CoDrive best checkpoint.

### Remaining blocked/risky
- User visual inspection is still needed to decide whether the index fingertip visibly participates in tangential rotation rather than mostly bracing.
- A separate local DOTPG student training process was already running and left untouched; the viewer still launched with enough GPU memory headroom.

### Single recommended next step
- Inspect the current local viewer. If the index still only braces, add a harder code-level coactive torque gate/penalty instead of only increasing YAML reward weights.

---

## v2-205 (2026-04-30) -- Cloud CoDrive PAdapt Distillation 2h Launched

### Target milestone/subgoal
- Distill the visually acceptable 2h CoDrive PPO teacher using `ProprioAdapt` / PAdapt.

### What changed (files + behavior impact)
- Added cloud PAdapt launch script:
  - `outputs/cloud_pipeline_sim2real_twofinger_codrive_padapt2h/run_padapt_2h.sh`
- Added remote stop watcher after user requested a quick 30min validation:
  - `outputs/cloud_pipeline_sim2real_twofinger_codrive_padapt2h/stop_at_30m.sh`
  - It stops the active tmux run at `30min` from launch and preserves `stage2_nn/model_best_30m.ckpt`.
- The script uses the CoDrive task directly rather than the older non-CoDrive student script default:
  - `task=Dexh13HoraLightbulbSim2RealTwoFingerCoDrive`
- Teacher checkpoint:
  - `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive/sim2real_twofinger_codrive_s42_2h/stage1_nn/best_reward_4159.37.pth`
- Student output:
  - `outputs/Dexh13HoraLightbulb_student_padapt_sim2real_twofinger_codrive/sim2real_twofinger_codrive_padapt_s42_2h/`

### What was verified (commands + key outcomes)
- Local script checks:
  - `bash -n outputs/cloud_pipeline_sim2real_twofinger_codrive_padapt2h/run_padapt_2h.sh`
  - `git diff --check -- outputs/cloud_pipeline_sim2real_twofinger_codrive_padapt2h/run_padapt_2h.sh`
  - Outcome: pass.
- Cloud preflight:
  - GPU visible and idle before launch:
    - `NVIDIA GeForce RTX 4090 D`, `24564 MiB`, `0%` utilization.
  - CoDrive teacher checkpoint exists on cloud.
  - CoDrive task/train configs exist on cloud.
- Remote tmux launch:
  - session: `sim2real_twofinger_codrive_padapt2h`
  - timeout: `7200` seconds.
  - resource: `task.env.numEnvs=48`, `train.ppo.minibatch_size=576`.
- Startup log verified:
  - `train.algo=ProprioAdapt`
  - `train.ppo.proprio_adapt=True`
  - actual task resolved as `Dexh13HoraLightbulbSim2RealTwoFingerCoDrive`
  - actual checkpoint resolved to `best_reward_4159.37.pth`
  - CoDrive reward settings are present in the runtime config.
- Initial training status:
  - `phase.txt`: `padapt`
  - `model_best.ckpt` has already been created under `stage2_nn/`.
  - Current best in early logs reached about `3033.24`.
  - GPU use around `2703 MiB / 24564 MiB`.
- 30min watcher status:
  - scheduled at `2026-04-30T06:49:16+00:00`
  - deadline: `2026-04-30T07:12:44+00:00` (`2026-04-30 15:12:44 CST`)
  - initial `sleep_sec=1408`.

### Local conclusion
- The requested CoDrive PAdapt distillation is running correctly on cloud and is now scheduled to stop at the 30min quick-validation point.
- Based on the previous 3.5h PAdapt run, rough line-fraction estimates were:
  - about `3986` by 30m,
  - about `4062` by 60m,
  - about `4138` by 90m,
  - about `4186` by 120m,
  - about `4261` by 210m.
- Practical recommendation: `1h` is enough for a quick probe, `1.5h` is a good cost/performance point, and `2h` is the better default when the teacher behavior is worth preserving. This launch uses `2h`.

### Remaining blocked/risky
- The PAdapt run has not reached the 30min stop point yet.
- Student visual behavior may still underperform the PPO teacher even if scalar reward is good; visual inspection is required after sync.

### Single recommended next step
- After the scheduled stop around `2026-04-30 15:12:44 CST`, verify `phase.txt=stopped_30m`, sync:
  - `outputs/Dexh13HoraLightbulb_student_padapt_sim2real_twofinger_codrive/sim2real_twofinger_codrive_padapt_s42_2h/`
  - `outputs/cloud_pipeline_sim2real_twofinger_codrive_padapt2h/`
  then visualize `stage2_nn/model_best_30m.ckpt` or latest `model_best.ckpt`.

---

## v2-206 (2026-04-30) -- CoDrive PAdapt 30m Synced and Viewer Opened

### Target milestone/subgoal
- Use the 30min quick-validation PAdapt snapshot for the CoDrive student and open local visualization.

### What changed (files + behavior impact)
- Synced cloud student output locally:
  - `outputs/Dexh13HoraLightbulb_student_padapt_sim2real_twofinger_codrive/sim2real_twofinger_codrive_padapt_s42_2h/`
- Synced cloud pipeline logs locally:
  - `outputs/cloud_pipeline_sim2real_twofinger_codrive_padapt2h/`
- No source/config behavior change.

### What was verified (commands + key outcomes)
- Remote 30min stop completed:
  - `phase.txt`: `stopped_30m`
  - `manual_stop_reason`: `manual_stop_30m`
  - `padapt_exit_status`: `120` from manual interrupt/timeout wrapper behavior after watcher stop.
  - stop log preserved:
    - `stage2_nn/model_best_30m.ckpt`
- Final remote PAdapt reward from stdout:
  - `Current Best: 3448.38`
- Synced local checkpoints:
  - `stage2_nn/model_best.ckpt`
  - `stage2_nn/model_best_30m.ckpt`
- Local viewer launched with:
  - `task=Dexh13HoraLightbulbSim2RealTwoFingerCoDrive`
  - `train.algo=ProprioAdapt`
  - `train.ppo.proprio_adapt=True`
  - `checkpoint=outputs/Dexh13HoraLightbulb_student_padapt_sim2real_twofinger_codrive/sim2real_twofinger_codrive_padapt_s42_2h/stage2_nn/model_best_30m.ckpt`
  - `headless=False`
  - `task.env.numEnvs=1`
  - obs/action noise and random force disabled.
- Viewer process restored the checkpoint and entered rollout, printing `Step 1...`.

### Local conclusion
- The 30min CoDrive PAdapt snapshot is local and currently being visualized.
- This is a quick-validation student; its scalar reward is below the PPO teacher (`3448.38` vs teacher `4159.37`), but it may still preserve the visually important co-drive behavior.

### Remaining blocked/risky
- User visual inspection is required to decide whether 30min PAdapt is good enough or should continue to 1h/2h.
- Manual-stop exit status is nonzero by design; artifact preservation was successful.

### Single recommended next step
- Inspect the current local student viewer. If behavior is acceptable, keep `model_best_30m.ckpt` as the quick student candidate; otherwise resume/launch a longer 1h or 2h PAdapt run from the same CoDrive teacher.

---

## v2-207 (2026-04-30) -- CoDrive Sim2Real Bundle Created

### Target milestone/subgoal
- Package the selected CoDrive PPO teacher and 30min PAdapt student artifacts into a compact `sim2real/codrive/` folder.

### What changed (files + behavior impact)
- Created/updated bundle directory:
  - `sim2real/codrive/`
- Bundle contents:
  - `sim2real/codrive/model_best_30m.ckpt`
  - `sim2real/codrive/best_reward_4159.37.pth`
  - `sim2real/codrive/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.task.yaml`
  - `sim2real/codrive/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.train.yaml`
- No training/source behavior changed.

### What was verified (commands + key outcomes)
- Re-synced remote student and teacher checkpoint outputs from `cloud-training`.
- Verified bundle sizes:
  - student ckpt: `1301474 bytes`
  - teacher pth: `1179649 bytes`
  - task YAML: `9828 bytes`
  - train YAML: `856 bytes`
- Verified SHA1 equality with source artifacts:
  - `model_best_30m.ckpt`: `843bc3868a45fce5e8e9bcc4823fe90b67d0d8c5`
  - `best_reward_4159.37.pth`: `7d028b3db549c9422f268cfd391a968976988b6a`
- Verified bundled YAMLs match current repo configs via `diff -q`.

### Local conclusion
- `sim2real/codrive/` now contains the minimal CoDrive sim2real handoff bundle requested by the user.

### Remaining blocked/risky
- The bundle intentionally does not include TensorBoard logs or full output directories.
- If future export/JIT uses a different expected filename, update the export command to point to `sim2real/codrive/model_best_30m.ckpt`.

### Single recommended next step
- Use the bundled student checkpoint for export/real-world inference validation, keeping the PPO teacher `.pth` only as provenance and fallback visualization reference.

---

## v2-204 (2026-04-30) -- SeeTaCloud 4090 Remote Environment Prepared

### Target milestone/subgoal
- Set up the newly opened SeeTaCloud SSH target so it can train the current DexScrew/IsaacGym project like the previous cloud machine.

### What changed (files + behavior impact)
- Synced the current local project to:
  - `root@connect.bjb2.seetacloud.com:22411:/root/code/dexscrew-repro/`
- Synced IsaacGym Preview4 to:
  - `/root/Codefield/third_party/isaacgym_preview4/`
- Reused existing remote conda env:
  - `/root/miniconda3/envs/dexscrew`
- Installed/confirmed Python dependencies from:
  - `requirements.txt`
- Installed IsaacGym into the `dexscrew` conda env:
  - `pip install -e /root/Codefield/third_party/isaacgym_preview4/isaacgym/python`
- Configured Git safe directory to avoid root/dubious-ownership failures after rsync:
  - `git config --global --add safe.directory /root/code/dexscrew-repro`
- Installed remote `tmux` through apt for long-running jobs.
- Added activation helper:
  - `outputs/cloud_setup_seetacloud/activate_dexscrew_ig.sh`

### What was verified (commands + key outcomes)
- Remote preflight:
  - SSH target works:
    - `ssh -p 22411 root@connect.bjb2.seetacloud.com`
  - Host:
    - `autodl-container-fvufu5jw9a-733711b2`
  - GPU:
    - `NVIDIA GeForce RTX 4090`, `24564 MiB`
  - Memory:
    - about `1.0 TiB`
  - Disk:
    - about `20 GiB` free after sync/install.
- Conda/Python:
  - env: `dexscrew`
  - Python: `3.8.20`
  - Torch: `2.4.1+cu121`
  - `torch.cuda.is_available() == True`
  - NumPy: `1.22.4`
- IsaacGym:
  - import path:
    - `/root/Codefield/third_party/isaacgym_preview4/isaacgym/python/isaacgym/__init__.py`
  - binding loaded:
    - `gym_38.so`
- Project:
  - remote git HEAD: `21e8eff`
  - `configs/task/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.yaml` exists.
  - `configs/train/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.yaml` exists.
- Training smoke:
  - command used `Dexh13HoraLightbulbSim2RealTwoFingerCoDrive`, `task.env.numEnvs=4`, `train.ppo.minibatch_size=12`, `train.ppo.max_agent_steps=24`.
  - outcome:
    - IsaacGym torch extension compiled/loaded;
    - CUDA PhysX environment built;
    - generated initial pose caches for scales `1.175` and `1.225`;
    - PPO reached `max steps achieved`.
- Long-run tooling:
  - `tmux 3.0a` installed and available.

### Local conclusion
- The new SeeTaCloud 4090 machine is ready for DexScrew IsaacGym training.
- Use:
  - `source /root/code/dexscrew-repro/outputs/cloud_setup_seetacloud/activate_dexscrew_ig.sh`
  before launching training commands on that machine.

### Remaining blocked/risky
- No long training job has been launched on the new SeeTaCloud target yet.
- Free root disk is only about `20 GiB`; avoid syncing large local `outputs/` or TensorBoard event archives unless needed.
- The smoke validates technical readiness, not policy behavior.

### Single recommended next step
- Launch the next requested PPO/student run on this new target from `/root/code/dexscrew-repro` using the activation helper and `tmux`, with explicit wall-clock `timeout` for reproducibility.

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

## v2-109 (2026-04-22) — Local Codex Model Effort Config Update

### Target milestone/subgoal
- Local execution environment adjustment requested by user: set project Codex default reasoning effort to `xhigh`.

### What changed (files + behavior impact)
- `.config/codex/config.toml`
  - Kept `model = "gpt-5.4"` unchanged.
  - Changed `model_reasoning_effort` from `medium` to `xhigh`.
  - Behavior impact: newly started Codex sessions in this project should default to `gpt-5.4` with `xhigh` reasoning effort; the current already-running session does not hot-switch itself.

### What was verified (commands + key outcomes)
- `sed -n '1,220p' .config/codex/config.toml`
  - Outcome before edit: `model = "gpt-5.4"`, `model_reasoning_effort = "medium"`.
- `sed -n '1,40p' .config/codex/config.toml`
  - Outcome after edit: `model = "gpt-5.4"`, `model_reasoning_effort = "xhigh"`.
- `rg -n "model|reasoning|effort|xhigh|5\\.4" .config/codex -g '!*.sqlite*' -g '!sessions/**' -g '!history.jsonl'`
  - Outcome: active project config source confirmed as `.config/codex/config.toml`.

### Remaining blocked/risky
- Current live session remains on the reasoning level it was started with; a new session/restart is required for the `xhigh` default to take effect.
- Provider-side limits or fallback behavior, if any, are external to repo config and were not exercised here.

### Single recommended next step
- Restart or open a new Codex session in this project and confirm the session context reports `model=gpt-5.4` and `reasoning_effort=xhigh`.

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

---

## v2-081 (2026-04-14) — Plan v3 草案创建（治理切换提案）

### Target milestone/subgoal
- 按用户请求起草 `Plan v3`，用于回应 v2 后段“连续 reject、边际收益下降”的执行现实。

### What changed (files + behavior impact)
- 新增治理草案：
  - `PLANS_v3.md`
- 草案内容包含：
  - v3 生效说明（草案/待确认）
  - 为什么需要 v3（从局部调参转向假设驱动）
  - v3 目标、范围边界、hard-first gate 与止损规则
  - Mainline A/B、里程碑 V3-M0~M4
  - 首批实验包（方向级）
  - 退出条件
- 无代码逻辑改动。

### What was verified (commands + key outcomes)
- 文件存在检查：
  - `ls -1 | rg '^PLANS.*\\.md$'`
  - outcome: `PLANS_v3.md` 已创建。
- 内容检查：
  - 草案已明确“确认前仍以 v2 为主计划”的边界，避免与现行执行权威冲突。

### Remaining blocked/risky
- `PLANS_v3.md` 当前是草案，不应被自动视为已生效主计划。
- 若要正式切换，需要用户/治理明确确认。

### Single recommended next step
- 由用户确认是否将 `PLANS_v3.md` 升格为当前执行主计划；确认后再把 v3 的 M0（reference 冻结 + 首批方向实验）落为具体命令清单并开始执行。

---

## v2-082 (2026-04-14) — Plan v3 M1 / Mainline A 三候选完成并触发方向冻结

### Target milestone/subgoal
- 按 `PLANS_v3.md` 执行 `V3-M1`（Mainline A，最多 3 候选）：
  - `A1` staged_te（light 对齐）；
  - `A2` staged_hold_te（hard 对齐）；
  - `A3` linear（mid target，平滑过渡）。
- 使用统一 single-seed gate（`seed=42`, `steps=256`, `nominal/light_v2/hard`）判定是否入围 multiseed。

### What changed (files + behavior impact)
- 治理与冻结文档（已落盘）：
  - `PLANS_v3.md`（active 状态）
  - `docs/plansv3_m0_freeze.md`（reference hash + eval protocol freeze）
- 新增训练产物：
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_v3a1_curriculum_stagedte_seed42_full/`
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_v3a2_curriculum_stagedhold_hardtarget_seed42_full/`
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_v3a3_curriculum_linear_midtarget_seed42_full/`
- 新增 single-seed eval 日志：
  - `outputs/robustness_eval/plansv3_m1_a1_curriculum_stagedte_seed42/`
  - `outputs/robustness_eval/plansv3_m1_a2_curriculum_stagedhold_hardtarget_seed42/`
  - `outputs/robustness_eval/plansv3_m1_a3_curriculum_linear_midtarget_seed42/`
- 代码逻辑无修改；本次为执行与证据推进。

### What was verified (commands + key outcomes)
- 统一 reference（frozen）：
  - ckpt hash: `31150d26646911325cac2aa408d3d42fdd4640a7`
  - reference seed42: nominal `1.675112/0.001302`, light_v2 `1.638266/0.001383`, hard `1.504904/0.001872`（reward/done）
- `A1`（hash `f9f26a54be974c01052c6f75f12feb19111113cb`, Current Best max `1873.51`）：
  - reward delta vs ref: nominal `+0.190914`, light_v2 `+0.131123`, hard `-0.152663`
  - hard done delta: `+0.000569`
  - gate: **FAIL**（hard reward / hard done 超阈）
- `A2`（hash `137c7fed4bc18085f6b44cc2477a4df615be5270`, Current Best max `1795.56`）：
  - reward delta vs ref: nominal `-0.181735`, light_v2 `-0.253708`, hard `-0.315135`
  - hard done delta: `+0.000488`
  - gate: **FAIL**（hard/light reward 超阈）
- `A3`（hash `6aae8c6cafb7945f205c575f2c1e159a1a7b847b`, Current Best max `1846.47`）：
  - reward delta vs ref: nominal `-0.030013`, light_v2 `-0.045582`, hard `-0.051875`
  - hard done delta: `+0.000244`
  - gate: **FAIL**（hard reward 仅超阈 `0.001875`）
- 方向判定（v3 止损）：
  - Mainline A 连续 3 候选 single-seed gate 失败，按规则触发 **方向 A 冻结**，不进入 multiseed。

### Local decision
- `V3-M1`（Mainline A）完成，且结果为 **freeze**：
  - A1/A2/A3 均未通过 single-seed hard-first gate。
- 当前不存在“可进入 multiseed 的 A 候选”。

### Remaining blocked/risky
- 现有课程策略方向仍复现“训练信号可高、统一 robust gate 不过”的模式。
- 若继续在 A 方向重复细调，预期边际收益低且 reject 风险高。

### Single recommended next step
- 按 `PLANS_v3.md` 切换至 `Mainline B`（Residual-Corrective Re-entry）执行 `B1`：
  - 单候选、full-budget、同 gate 的最小验证，
  - 仅启用小规模 residual corrective（不扩展大架构），
  - 若仍失败则累计 B 方向止损计数。

---

## v2-083 (2026-04-14) — Plan v3 Mainline B 三候选完成并触发方向冻结（v3 退出条件满足）

### Target milestone/subgoal
- 执行 `PLANS_v3.md` 的 Mainline B（Residual-Corrective Re-entry），完成最多 3 个候选：
  - `B1`: residual `scale=0.5`（保留 near-reference tail/anchor）
  - `B2`: residual `scale=1.0`（保留 near-reference tail/anchor）
  - `B3`: residual `scale=1.0` + 去除 tail/anchor（`tail_coef=0.0`, `anchor=0.0`）
- 对每个候选执行 full-budget 训练 + `seed42` 三条件 gate（`nominal/light_v2/hard`, `steps=256`）。

### What changed (files + behavior impact)
- 新增训练产物：
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_v3b1_residual_scale05_seed42_full/`
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_v3b2_residual_scale10_seed42_full/`
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_v3b3_residual_notail_seed42_full/`
- 新增 single-seed eval 日志：
  - `outputs/robustness_eval/plansv3_m2_b1_residual_scale05_seed42/`
  - `outputs/robustness_eval/plansv3_m2_b2_residual_scale10_seed42/`
  - `outputs/robustness_eval/plansv3_m2_b3_residual_notail_seed42/`
- 代码逻辑未改动；本条为实验执行与证据推进。

### What was verified (commands + key outcomes)
- 统一 frozen reference：
  - hash: `31150d26646911325cac2aa408d3d42fdd4640a7`
  - seed42: nominal `1.675112/0.001302`, light_v2 `1.638266/0.001383`, hard `1.504904/0.001872`
- `B1`（hash `924ecaa75b42b70a86c70c68cd3fc34ad12f073f`, Current Best max `1785.28`）：
  - deltas vs reference:
    - nominal `-0.284072` / done `+0.000407`
    - light_v2 `-0.391142` / done `+0.000082`
    - hard `-0.547112` / done `+0.000569`
  - gate: **FAIL**（hard/light reward 与 hard done 全不满足）
- `B2`（hash `f2c724215cb47d76608d8d47e5d647de59d08d25`, Current Best max `1925.01`）：
  - deltas vs reference:
    - nominal `-0.342622` / done `+0.000814`
    - light_v2 `-0.417906` / done `+0.000407`
    - hard `-0.352470` / done `+0.000732`
  - gate: **FAIL**（hard/light reward 与 hard done 全不满足）
- `B3`（hash `807e716754054cbaa31b586132f635ecade44a2b`, Current Best max `1796.88`）：
  - deltas vs reference:
    - nominal `-0.035265` / done `+0.000733`
    - light_v2 `+0.151464` / done `+0.000000`
    - hard `-0.161701` / done `+0.000244`
  - gate: **FAIL**（hard reward 不满足）
- 方向判定（v3 止损）：
  - Mainline B 连续 3 候选 single-seed gate 失败，触发 **方向 B 冻结**。

### Local decision
- Mainline A（v2-082）已冻结 + Mainline B（本条）已冻结。
- `PLANS_v3` 退出条件 2 已满足：
  - “Mainline A/B 均触发止损冻结且无可接受候选”。

### Remaining blocked/risky
- 在当前仓库与当前 teacher-student 约束下，diffusion 扩张方向未出现可通过统一 gate 的替代候选。
- 继续在同类局部空间重复搜索，预期收益低、计算成本高。

### Single recommended next step
- 进入 `V3-M4` 阶段收敛与治理回切：
  - 维持 frozen reference（不替换），
  - 输出面向 GPT 的治理问题单（是否暂停 diffusion 扩张并转论文收敛/对照结论）。

---

## v2-084 (2026-04-14) — Plan v4 M0 完整执行（bug-free baseline）并触发硬止损

### Target milestone/subgoal
- 按 `PLANS_v4.md` 执行 `V4-M0` 最小闭环：
  - 语法可用性检查（bug-fix 代码）
  - 使用 V3-M0 frozen config（seed=42, 15min budget）重训 bug-free baseline
  - 统一协议评测 `nominal/light_v2/hard`（`steps=256`）
  - 计算相对 V3-M0 与 PAdapt 的 delta，并做 single-seed gate 判定。

### What changed (files + behavior impact)
- 新增训练运行产物：
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_v4m0_bugfix_seed42_15min/`
- 新增训练日志：
  - `outputs/v4_logs/v4_m0_bugfix_baseline_train_seed42.log`
- 新增评测日志：
  - `outputs/v4_logs/v4_m0_bugfix_eval_nominal_seed42.log`
  - `outputs/v4_logs/v4_m0_bugfix_eval_lightv2_seed42.log`
  - `outputs/v4_logs/v4_m0_bugfix_eval_hard_seed42.log`
- 新增评测 cache 运行目录：
  - `outputs/XHandHoraScrewDriver_eval_robustness/plansv4_m0_bugfix_baseline_nominal_seed42/`
  - `outputs/XHandHoraScrewDriver_eval_robustness/plansv4_m0_bugfix_baseline_lightv2_seed42/`
  - `outputs/XHandHoraScrewDriver_eval_robustness/plansv4_m0_bugfix_baseline_hard_seed42/`
- 代码逻辑未新增改动；本条为执行与证据推进。

### What was verified (commands + key outcomes)
- 语法检查（避免 `__pycache__` 写权限问题，使用 `compile()` 无落盘方式）：
  - `python - <<'PY' ... compile(src, file, 'exec') ... PY`
  - outcome: `syntax_ok 4`
- 训练命令（容器内，15min budget via `timeout 1000`）：
  - `./docker-run-isaacgym.sh bash -lc '... timeout 1000 python train.py ...'`
  - 关键 override：
    - `train.algo=DiffusionLatentStudent`
    - `train.ppo.output_name=XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_v4m0_bugfix_seed42_15min`
    - `+train.ppo.diffusion_teacher_delta_tail_coef=0.2`
    - `+train.ppo.diffusion_teacher_delta_tail_threshold=0.15`
    - `+train.ppo.diffusion_teacher_delta_tail_selective=True`
    - `+train.ppo.diffusion_base_action_anchor_coef=0.03`
  - outcome:
    - 训练按预算超时结束（`exit_code=124`，非崩溃）
    - `Current Best` 峰值约 `1891.38`
    - best ckpt: `.../stage2_diffusion_nn/model_best.ckpt`
- 关键 artifact 指纹：
  - commit: `1f8d373fd695c04b52657ba82514ba41873903a0`
  - config snapshot: `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_v4m0_bugfix_seed42_15min/config_041415_1f8d373.yaml`
  - ckpt sha1: `83bc317790da32d78273db13d9acb65da8249023`
- 评测命令（三条件）：
  - `bash scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent <ckpt> 256 plansv4_m0_bugfix_baseline_nominal_seed42`
  - `bash scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent <ckpt> 256 plansv4_m0_bugfix_baseline_lightv2_seed42 task.env.randomization.obs_noise_e_scale=0.03 task.env.randomization.obs_noise_t_scale=0.015 task.env.forceScale=1.0 task.env.randomForceProbScalar=0.2`
  - `bash scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent <ckpt> 256 plansv4_m0_bugfix_baseline_hard_seed42 task.env.randomization.obs_noise_e_scale=0.05 task.env.randomization.obs_noise_t_scale=0.025 task.env.forceScale=1.5 task.env.randomForceProbScalar=0.3`
- seed42 指标（reward / done）：
  - nominal: `1.738804 / 0.001465`
  - light_v2: `1.588133 / 0.001790`
  - hard: `1.367269 / 0.002279`
- delta vs V3-M0 reference（primary gate）：
  - nominal reward: `+0.063692`
  - light_v2 reward: `-0.050133`
  - hard reward: `-0.137635`
  - hard done: `+0.000407`
- delta vs PAdapt baseline（secondary）：
  - nominal: `-0.429016`
  - light_v2: `-0.490941`
  - hard: `-0.470956`
- single-seed gate 判定：
  - `delta_hard >= -0.05` → **FAIL**
  - `delta_light_v2 >= -0.08` → PASS
  - `delta_done_hard <= +0.0005` → PASS
  - overall: **FAIL**
- V4 硬止损检查：
  - `hard delta < -0.10` → **TRUE**（`-0.137635`）

### Local decision
- `V4-M0` 完成并得到 clean evidence。
- 结果满足 `PLANS_v4` 硬止损规则（bug-fixed baseline hard 严重恶化），应直接从 `M0` 进入 `V4-M3`，不再继续 `M1/M2`。

### Remaining blocked/risky
- bug-fix 后 baseline 在 hard 条件显著回退，说明当前 diffusion 路径在统一 gate 下竞争力不足。
- 若继续做同类局部消融，存在较高“额外计算投入但难改结论”的风险。

### Single recommended next step
- 执行 `V4-M3`：汇总 V3+V4 证据并输出 final verdict（`Accept/Suspend/Inconclusive` 三选一），建议当前证据优先走 `Suspend` 候选结论草案。

---

## v2-085 (2026-04-15) — Plan v4 M3 Final Verdict 完成（结论: Suspend）

### Target milestone/subgoal
- 在 `V4-M0` 触发硬止损后，按 `PLANS_v4.md` 执行 `V4-M3`：
  - 汇总 V3 + V4 全部候选证据
  - 产出 final verdict evidence block
  - 给出 `Accept/Suspend/Inconclusive` 三选一结论与论文呈现建议。

### What changed (files + behavior impact)
- 新增 final verdict 文档：
  - `docs/plansv4_m3_final_verdict.md`
- 更新治理问题单（与 M3 结果对齐）：
  - `codeagent_issue.md`
- 代码与训练/eval 入口无改动；本条为证据收敛与治理收口。

### What was verified (commands + key outcomes)
- 读取计划约束并确认 M3 完成条件：
  - `sed -n '200,360p' PLANS_v4.md`
- 覆盖候选日志存在性检查：
  - `find outputs/robustness_eval -maxdepth 2 -type d | rg 'plansv3_m1_a[123]|plansv3_m2_b[123]'`
  - `ls -1 outputs/v4_logs | rg 'v4_m0_bugfix_eval_(nominal|lightv2|hard)_seed42\.log'`
- 自动抽取 `EvalSummary` 并统一计算 gate/delta：
  - `python - <<'PY' ... parse v3/v4 logs ... PY`
  - outcome: V3 6 候选 + V4-M0 共 7 条记录全部 single-seed gate FAIL。
- 生成并复核 M3 文档：
  - `python - <<'PY' ... write docs/plansv4_m3_final_verdict.md ... PY`
  - `sed -n '1,260p' docs/plansv4_m3_final_verdict.md`
- 关键结论（来自 final verdict）：
  - `V4-M0 hard delta = -0.137635 < -0.10`（命中硬止损）
  - 无任一候选通过 single-seed gate
  - 三选一结论：`Suspend`

### Local decision
- `V4-M3` 已完成，`PLANS_v4` 达到退出条件 2：
  - “V4-M3 触发且 final verdict 完成 → 结论：Suspend 或 Inconclusive”。
- 本地建议结论：`Suspend`（冻结 diffusion 扩张，保留 baseline/reference，不再在当前 scope 内追加训练候选）。

### Remaining blocked/risky
- 当前剩余为治理确认风险而非技术不确定性：
  - 若不接受 `Suspend`，必须明确批准超出 V4 scope 的新路线（否则会重复低收益探索）。

### Single recommended next step
- 由治理层确认采纳 `Suspend`，随后将 `docs/plansv4_m3_final_verdict.md` 作为论文负结果/对照证据的主引用，并切换到报告整理任务。

---

## v2-086 (2026-04-15) — Governance 确认完成（Suspend 正式采纳）

### Target milestone/subgoal
- 在 `V4-M3` 完成后执行治理确认落盘：
  - 将 `Suspend` 从“本地建议结论”升级为“已确认结论”
  - 同步计划状态、verdict 文档与 issue 状态。

### What changed (files + behavior impact)
- 更新主计划状态：
  - `PLANS_v4.md`
  - `status: pending_activation -> completed_suspend`
  - 新增 `completed_on` 与 `final_decision` 字段
- 更新 final verdict 文档：
  - `docs/plansv4_m3_final_verdict.md`
  - 新增 `Governance Confirmation` 区块（记录确认日期与确认来源）
- 更新治理问题单：
  - `codeagent_issue.md`
  - 新增 `Status=resolved`，并将 blocker 改为已解除、下一步改为写作证据整理

### What was verified (commands + key outcomes)
- 内容核验：
  - `sed -n '1,60p' PLANS_v4.md`
  - `sed -n '1,220p' codeagent_issue.md`
  - `sed -n '1,260p' docs/plansv4_m3_final_verdict.md`
- outcome:
  - `PLANS_v4` 状态与结论一致（completed + suspend）
  - `codeagent_issue.md` 已转 resolved 归档态
  - `final_verdict` 已包含治理确认区块

### Local decision
- `PLANS_v4` 执行生命周期已闭环：技术证据完成 + 治理确认完成。
- 当前 diffusion 扩张线在既定 scope 内正式冻结（Suspend）。

### Remaining blocked/risky
- 无当前未决技术 blocker。
- 唯一风险是后续写作阶段若脱离证据文档口径，可能造成结论不一致。

### Single recommended next step
- 进入论文产出模式：以 `docs/plansv4_m3_final_verdict.md` 为主证据，整理结果段落与附录表格（V3+V4 汇总）。

---

## v2-087 (2026-04-15) — Thesis 证据整理包落地（V4 收敛后写作交付）

### Target milestone/subgoal
- 按 `v2-086` 推荐下一步，进入论文产出模式：
  - 生成可直接引用的 V4 thesis result bundle
  - 生成 subsection draft 与附录表格（CSV/LaTeX）
  - 将 V4 闭环结论写入 stage acceptance 快照。

### What changed (files + behavior impact)
- 新增 V4 thesis bundle 文档：
  - `docs/plansv4_thesis_result_bundle.md`
- 新增 V4 thesis 结果小节草稿：
  - `docs/plansv4_thesis_results_subsection_draft.md`
- 新增 V4 数据表（可用于图表/附录/LaTeX）：
  - `docs/data/plansv4_thesis_candidate_table.csv`
  - `docs/data/plansv4_thesis_delta_table.csv`
  - `docs/data/plansv4_thesis_tables.tex`
- 更新 stage 接受摘要：
  - `docs/stage_acceptance_summary.md`（追加 `PLANS_v4 Closure Snapshot (2026-04-15)`）
- 代码逻辑无改动；本条为论文证据资产化。

### What was verified (commands + key outcomes)
- 对齐既有 thesis 文档格式：
  - `sed -n '1,260p' docs/plansv2_thesis_results_subsection_draft.md`
  - `sed -n '1,260p' docs/plansv2_thesis_result_bundle.md`
- 自动解析日志并生成数据包：
  - `python - <<'PY' ... parse v3/v4 EvalSummary logs; write csv/tex/md ... PY`
  - outcome: 5 个目标文件全部生成
- 内容抽样核验：
  - `sed -n '1,260p' docs/plansv4_thesis_result_bundle.md`
  - `sed -n '1,240p' docs/plansv4_thesis_results_subsection_draft.md`
  - `sed -n '1,60p' docs/data/plansv4_thesis_candidate_table.csv`
  - `sed -n '1,60p' docs/data/plansv4_thesis_delta_table.csv`
- 结论一致性核验：
  - 生成文件中的候选数量、gate 失败数、`v4m0 delta_hard=-0.137635` 与 `docs/plansv4_m3_final_verdict.md` 一致。

### Local decision
- V4 已从“实验收敛”推进到“写作可用资产”阶段：
  - 结论、表格、附录素材与 narrative 草稿已齐备。

### Remaining blocked/risky
- 当前主要风险为写作一致性：
  - 若后续手工改写数值，可能与自动生成表格不一致。

### Single recommended next step
- 在论文主文中引用 `docs/plansv4_thesis_results_subsection_draft.md` 作为结果段落初稿，并直接复用 `docs/data/plansv4_thesis_tables.tex` 作为附录表格起点。

---

## v2-088 (2026-04-15) — V4 thesis bundle 生成脚本化（可重复构建）

### Target milestone/subgoal
- 将 V4 论文数据包从“一次性命令”升级为“可重复执行脚本”，降低后续写作阶段的人为转录风险。

### What changed (files + behavior impact)
- 新增脚本：
  - `scripts/build_plansv4_thesis_bundle.py`
- 脚本功能：
  - 解析 V3/V4 既有 `EvalSummary` 日志
  - 生成 `docs/data/plansv4_thesis_candidate_table.csv`
  - 生成 `docs/data/plansv4_thesis_delta_table.csv`
  - 生成 `docs/data/plansv4_thesis_tables.tex`
  - 生成/刷新 `docs/plansv4_thesis_result_bundle.md`
  - 生成/刷新 `docs/plansv4_thesis_results_subsection_draft.md`

### What was verified (commands + key outcomes)
- 脚本执行：
  - `python scripts/build_plansv4_thesis_bundle.py`
  - outcome: 5 个目标产物全部成功写出
- 语法检查：
  - `python -m py_compile scripts/build_plansv4_thesis_bundle.py`
  - outcome: pass
- 脚本内容抽检：
  - `sed -n '1,120p' scripts/build_plansv4_thesis_bundle.py`
  - outcome: 输入日志路径、reference 常量、输出文件列表与 V4 证据口径一致

### Local decision
- V4 写作资产已具备“可重复重建”能力。
- 后续若补充或替换日志，只需重跑该脚本即可同步更新论文表格与文本草稿。

### Remaining blocked/risky
- 若日志路径变更或文件被移动，脚本会在解析阶段报错，需要同步更新路径常量。

### Single recommended next step
- 在论文仓或主文编辑阶段，固定使用 `python scripts/build_plansv4_thesis_bundle.py` 作为 V4 表格/草稿刷新入口，避免手工维护分叉。

---

## v2-089 (2026-04-15) — 按顺序推进：主文终稿段落 + 附录接入 + 刷新流程

### Target milestone/subgoal
- 按用户确认的顺序继续推进写作交付：
  1) 产出可直接贴入主文的 V4 结果段（非 draft）
  2) 产出附录接入模板（可复制 LaTeX）
  3) 固化一键刷新流程并在 README 暴露入口。

### What changed (files + behavior impact)
- 新增主文可用结果段（final 版）：
  - `docs/plansv4_thesis_results_subsection_final.md`
- 新增附录接入模板：
  - `docs/plansv4_thesis_appendix_integration.md`
- 新增刷新流程文档：
  - `docs/plansv4_thesis_refresh_workflow.md`
- 更新 README 入口：
  - `README.md` 新增 `PLANS_v4 Thesis Assets` 区块

### What was verified (commands + key outcomes)
- 内容抽检：
  - `sed -n '1,240p' docs/plansv4_thesis_results_subsection_final.md`
  - `sed -n '1,260p' docs/plansv4_thesis_appendix_integration.md`
  - `sed -n '1,220p' docs/plansv4_thesis_refresh_workflow.md`
- README 入口检查：
  - `rg -n 'PLANS_v4 Thesis Assets|plansv4_thesis_results_subsection_final|plansv4_thesis_appendix_integration|plansv4_thesis_refresh_workflow' README.md`
  - outcome: 入口项均可检索到。
- 生成脚本可用性保持：
  - `python scripts/build_plansv4_thesis_bundle.py`
  - `python -m py_compile scripts/build_plansv4_thesis_bundle.py`
  - outcome: 产物可重建、脚本语法通过。

### Local decision
- 这三步已全部完成，当前论文交付链路从“结果结论”到“附录接入”到“可重复刷新”已闭环。

### Remaining blocked/risky
- 无当前技术阻塞。
- 风险仅在写作执行层：若后续手工改数字且不重跑生成脚本，可能与证据表脱节。

### Single recommended next step
- 直接把 `docs/plansv4_thesis_results_subsection_final.md` 合入论文主文，并按 `docs/plansv4_thesis_appendix_integration.md` 插入附录表格。

---

## v2-090 (2026-04-15) — PLANS_v5 收敛执行：M1/M2 gate 完成 + M4 final verdict

### Target milestone/subgoal
- 以 `PLANS_v5.md` 为执行主线完成当前闭环：
  - 收尾 `V5-M1`（Consistency 候选 3 的最终验证）
  - 确认 `V5-M2`（Flow 方向是否继续扩展）
  - 在无初筛通过候选时落地 `V5-M4` final verdict。

### What changed (files + behavior impact)
- 新增 V5 final verdict 文档：
  - `docs/plansv5_m4_final_verdict.md`
- 更新 V5 主计划状态：
  - `PLANS_v5.md`（`status -> completed_conclude`，补 `activated_on/completed_on/final_decision`）
- 更新 stage acceptance 快照：
  - `docs/stage_acceptance_summary.md`（追加 `PLANS_v5 Closure Snapshot (2026-04-15)`）
- 代码逻辑与 train/eval/export 入口未改动；本次为实验执行与证据收敛。

### What was verified (commands + key outcomes)
- 训练完成性与错误检查：
  - `ps -p 1016219 -o pid=,etime=,cmd=`
  - `python - <<'PY' ... parse train_900s.log Current Best/error keywords ... PY`
  - outcome: `v5_m1_consistency_tuned_step2_seed42_15min` 训练完成，`Current Best max=1685.56`，无 Traceback。
- ckpt 指纹核验：
  - `sha1sum .../v5_m1_consistency_baseline.../model_best.ckpt .../v5_m1_consistency_tuned_step2.../model_best.ckpt .../v5_m1_consistency_no_bc.../model_best.ckpt .../v5_m2_flow_baseline.../model_best.ckpt`
  - outcome:
    - consistency_baseline = `2209a84dc29d356c77275281085a30dfe8d36c91`
    - consistency_tuned_step2 = `2209a84dc29d356c77275281085a30dfe8d36c91`（与 baseline 相同）
    - consistency_no_bc = `4686896b23e6f7e8f5df152abaab967ded2f8960`
    - flow_baseline = `f790820138c7a9483d572aeebf18c626dc149f0f`
- 三条件评测（tuned_step2）：
  - `docker run ... bash scripts/eval_screwdriver_student_robustness.sh ... v5_m1_consistency_tuned_step2_seed42_{nominal,light_v2,hard}`
  - logs:
    - `outputs/robustness_eval/v5_m1_consistency_tuned_step2_seed42/consistency_nominal.log`
    - `outputs/robustness_eval/v5_m1_consistency_tuned_step2_seed42/consistency_light_v2.log`
    - `outputs/robustness_eval/v5_m1_consistency_tuned_step2_seed42/consistency_hard.log`
  - outcome: 与 `consistency_baseline` 指标一致（同 hash 导致同结果）。
- 2-step 推理本体验证（eval-only）：
  - `docker run ... bash scripts/eval_screwdriver_student_robustness.sh ... +train.ppo.consistency_infer_steps=2 ...`
  - logs:
    - `outputs/robustness_eval/v5_m1_consistency_infer2_evalonly_seed42/consistency_nominal.log`
    - `outputs/robustness_eval/v5_m1_consistency_infer2_evalonly_seed42/consistency_light_v2.log`
    - `outputs/robustness_eval/v5_m1_consistency_infer2_evalonly_seed42/consistency_hard.log`
  - outcome:
    - nominal/light_v2/hard = `1.896259 / 1.980752 / 1.520207`
    - hard_done = `0.003337`
    - delta_hard vs V3-M0 = `+0.015303`，但 `delta_done_hard=+0.001465`（gate fail）。
- 全候选 gate 汇总解析：
  - `python - <<'PY' ... parse candidate logs and compute gate ... PY`
  - outcome:
    - `consistency_baseline`: reward gate pass, done gate fail
    - `consistency_no_bc`: reward gate fail
    - `consistency_infer2_evalonly`: done gate fail
    - `flow_baseline`: hard reward + done gate fail
    - 初筛通过数 `0`，`V5-M3` 不触发。

### Local decision
- `PLANS_v5` 在当前 scope 内已完成到 `M4`：
  - diffusion 新形式（Consistency/Flow）均未通过 single-seed gate。
  - 结论落地为 `Conclude`（见 `docs/plansv5_m4_final_verdict.md`）。
- 记录到位的附加事实：
  - Consistency 相对 DDPM 历史 evidence 有 reward 改善信号，但未满足 done-rate 约束，故不可接受为新基线。

### Remaining blocked/risky
- 无当前技术 blocker。
- 若要继续扩展 diffusion，需要新的治理计划（例如单独针对 done-rate 的可证伪改进路线）；否则会重复低收益试探。

### Single recommended next step
- 冻结 `PLANS_v5` 扩张线并进入写作/汇总：以 `docs/plansv5_m4_final_verdict.md` 作为 V5 主证据，将“reward 改善但 gate 未过”的结论纳入论文负结果讨论。

---

## v2-091 (2026-04-15) — V5.5 优化计划草案落地（待激活）

### Target milestone/subgoal
- 在 `PLANS_v5` 已收敛后，按用户意图准备 `Plan 5.5` 草案文件，供后续优化执行参考。

### What changed (files + behavior impact)
- 新增计划文件：
  - `PLANS_v5_5.md`
- 影响：
  - 无代码执行行为变化；
  - 提供一份可直接激活的 `Consistency hard_done` 定向优化执行蓝图。

### What was verified (commands + key outcomes)
- 现有 V5 收敛文档存在性检查：
  - `rg --files | rg -i 'PLANS_v5|plansv5_m4_final_verdict|session_handoff_v2|stage_acceptance_summary'`
- 读取 V5 主计划与 final verdict 对齐基线：
  - `sed -n '1,260p' PLANS_v5.md`
  - `sed -n '1,260p' docs/plansv5_m4_final_verdict.md`
- 参数可执行性抽检（Consistency 可用 override）：
  - `rg -n 'consistency_|bc_loss_coef|base_action_anchor_coef|consistency_action_l2_coef|consistency_num_scales|consistency_boundary_coef|consistency_stochastic_infer|consistency_infer_steps' dexscrew/algo/ppo/consistency_latent_student.py`
- outcome:
  - `PLANS_v5_5.md` 中候选参数均来自当前代码可识别项；
  - 计划结构包含里程碑、止损、有效性约束与证据记录规范。

### Local decision
- `PLANS_v5_5.md` 已完成“可执行草案”状态，等待治理确认后可切换为 active plan 执行。

### Remaining blocked/risky
- 当前 blocker 为治理层激活决策（非技术阻塞）。
- 未激活前不应直接消耗训练预算执行 V5.5 候选。

### Single recommended next step
- 由用户确认激活 `PLANS_v5_5.md`，随后从 `V5.5-M0` 开始按候选矩阵连续推进。

---

## v2-092 (2026-04-15) — PLANS_v5_5 验收口径升级：多指标防退化 + hard_done 突破并重

### Target milestone/subgoal
- 响应用户要求：V5.5 不仅关注 `hard_done`，还必须防止其它关键指标退化。

### What changed (files + behavior impact)
- 更新计划文件：
  - `PLANS_v5_5.md`
- 主要新增：
  - `4.4 Anti-regression Guardrails`
  - 明确基线锚点（consistency_baseline 的 reward/done）
  - 新增 reward 与 done 的防退化阈值
  - 明确 hard_done 突破的绝对目标（`<=0.002372`）与相对改善幅度（`>=0.000314`）
  - 将 guardrails 纳入 M1/M2/M4 判定逻辑（不再“只要 hard_done 好就算通过”）。

### What was verified (commands + key outcomes)
- 计划文件修改后复核：
  - `sed -n '1,260p' PLANS_v5_5.md`
- outcome:
  - V5.5 已升级为“primary gate + anti-regression 双重验收”；
  - hard_done 仍是核心突破点，但 reward/done 各条件退化会触发 FAIL。

### Local decision
- V5.5 现在是更稳健的执行框架：避免“单指标优化导致整体变差”的风险。

### Remaining blocked/risky
- 当前仍待治理激活（`draft_pending_activation`）。
- 阈值为工程化约束，后续若发现过严/过松，需要在激活后首轮结果上再微调一次。

### Single recommended next step
- 激活 `PLANS_v5_5.md`，从 `V5.5-M0` 开始执行，并在首个候选评测后检查阈值是否需要轻微校准。

---

## v2-093 (2026-04-15) — PLANS_v5_5 全流程执行完成：Consistency 优化达成 Accept

### Target milestone/subgoal
- 按用户指令执行 `PLANS_v5_5`，完成 Consistency 优化与阶段验收：
  - M0 baseline 对齐
  - M1 三候选训练+三条件评测
  - M2 top-1 multiseed 验证
  - M3 final verdict 落盘。

### What changed (files + behavior impact)
- 更新计划状态：
  - `PLANS_v5_5.md` -> `status=completed_accept`
- 新增最终结论文档：
  - `docs/plansv5_5_final_verdict.md`
- 更新 stage 快照：
  - `docs/stage_acceptance_summary.md`（新增 `PLANS_v5_5 Closure Snapshot`）
- 无 train/eval 入口代码改动；本次为实验执行与证据更新。

### What was verified (commands + key outcomes)
- M0 baseline 锁定：
  - `sha1sum outputs/XHandHoraScrewDriver_student_consistency/v5_m1_consistency_baseline_seed42_15min/stage2_consistency_nn/model_best.ckpt`
  - baseline hash: `2209a84dc29d356c77275281085a30dfe8d36c91`
  - baseline eval: nominal `2.201198/0.000651`, light_v2 `1.936892/0.001628`, hard `1.714672/0.002686`.

- M1 candidate A (`action_l2_stable`)：
  - train run:
    - `outputs/XHandHoraScrewDriver_student_consistency/v5_5_m1_action_l2_stable_seed42_15min/`
    - best ckpt sha1: `3cefcea0b4b12a1a6dfa062fc36456332a0523e0`
    - `Current Best max=1715.43`
  - eval logs:
    - `outputs/robustness_eval/v5_5_m1_action_l2_stable_seed42/consistency_{nominal,light_v2,hard}.log`
  - result:
    - primary gate: PASS
    - anti-regression: FAIL（`hard_reward_vs_base=-0.183468 < -0.15`）

- M1 candidate B (`anchor_l2_combo`)：
  - train run:
    - `outputs/XHandHoraScrewDriver_student_consistency/v5_5_m1_anchor_l2_combo_seed42_15min/`
    - best ckpt sha1: `ad73e00539772aa3dac8088c07691eba12c1da48`
    - `Current Best max=1687.33`
  - eval logs:
    - `outputs/robustness_eval/v5_5_m1_anchor_l2_combo_seed42/consistency_{nominal,light_v2,hard}.log`
  - result:
    - primary gate: FAIL（`light_v2` 不达标）
    - anti-regression: FAIL（多项）

- M1 candidate C (`boundary_bc_tuned`)：
  - train run:
    - `outputs/XHandHoraScrewDriver_student_consistency/v5_5_m1_boundary_bc_tuned_seed42_15min/`
    - best ckpt sha1: `45d785763f385bfb0a5326866eb31c9cbcbb3215`
    - `Current Best max=1655.81`
  - eval logs:
    - `outputs/robustness_eval/v5_5_m1_boundary_bc_tuned_seed42/consistency_{nominal,light_v2,hard}.log`
  - result:
    - nominal/light_v2/hard = `2.221939 / 1.902549 / 1.639399`
    - done = `0.000732 / 0.001546 / 0.002035`
    - primary gate: PASS
    - anti-regression: PASS
    - selected as M2 top-1。

- M2 multiseed (candidate C, seeds=42/43/44):
  - logs root:
    - `outputs/robustness_eval/v5_5_m2_boundary_bc_tuned_multiseed/`
  - aggregates:
    - nominal mean `2.336284` (delta vs V3-M0 `+0.661172`)
    - light_v2 mean `2.044725` (delta `+0.406459`)
    - hard mean `1.714471` (delta `+0.209567`)
    - hard done mean `0.001601` (`<=0.002372`)
  - acceptance gate:
    - `hard_mean_delta>=0`: PASS
    - `light_v2_mean_delta>=0`: PASS
    - `nominal_mean_delta>=-0.10`: PASS
    - `hard_done_mean<=0.002372`: PASS

### Local decision
- `PLANS_v5_5` 结论：`Accept`。
- 通过候选：`consistency_boundary_bc_tuned`（`boundary=0.8, num_scales=16, bc=1.2`）。
- 本轮已完成从优化到验证到落盘的闭环。

### Remaining blocked/risky
- 无当前技术 blocker。
- 风险点：accepted candidate 目前以单次训练 run 为主证据，后续若需论文级稳健性可加一次同配置重跑确认方差。

### Single recommended next step
- 将 `consistency_boundary_bc_tuned` 升级为新的 consistency reference（固定 ckpt 与配置），并执行一次同配置复现实验（fresh run id）用于最终报告定稿。

---

## v2-094 (2026-04-16) — PLANS_v5_5d 总结文档落地（对齐 consistency vs latent vs padapt）

### Target milestone/subgoal
- 按用户要求先产出 `planv5.5d` 总结文档，明确当前方法相对 `latent` 与 `padapt` 的真实位置，为后续优化执行提供单点入口。

### What changed (files + behavior impact)
- 新增：
  - `docs/plansv5_5d_summary.md`
- 影响：
  - 无代码与训练/评测行为改动；
  - 增加一份可直接引用的阶段总结，统一口径说明“已领先部分”和“未追回部分”。

### What was verified (commands + key outcomes)
- 文件与上下文检索：
  - `ls -1`
  - `rg --files | rg -n "PLANS_v5|plansv5|v5_5|v5\\.5|v5_5d|v5\\.5d|handoff|stage_acceptance"`
- bootstrap 必读文件检查：
  - `sed -n '1,220p' AGENTS.md`
  - `tail -n 220 docs/session_handoff_v2.md`
  - `sed -n '360,460p' docs/stage_acceptance_summary.md`
  - `sed -n '1,240p' docs/plansv5_5_final_verdict.md`
- outcome:
  - `v5.5d` 命名文件此前不存在，已新增 summary 文档；
  - 文档内已使用统一 multiseed 协议数据给出 `consistency / latent_diffusion / padapt` 并排对比及 delta 结论；
  - 结论明确：`consistency` 已全面领先旧 latent 参考，但尚未全面超过 `padapt`（差距主要在 `hard`）。

### Local decision
- 本次会话完成文档侧里程碑：`PLANS_v5_5d` 执行前的事实对齐与总结落地已就绪。

### Remaining blocked/risky
- 无技术 blocker。
- 若进入下一轮训练，主要风险是为追 `hard` reward 导致 done 或 light_v2 退化，需继续沿用 anti-regression 约束。

### Single recommended next step
- 以 `consistency_boundary_bc_tuned` 为唯一起点，启动 `V5.5d` 的首个“小步 hard-targeted”候选（保持统一三条件评测与多指标防退化门槛）。

---

## v2-095 (2026-04-16) — PLANS_v6 全流程执行完成：M1+M3 全候选未达 hard 门槛，结论 Conclude

### Target milestone/subgoal
- 按用户指令完整执行 `PLANS_v6`，目标是让 Consistency 尽可能逼近并超越 `padapt`。
- 执行路径：`M0 -> M1 -> (触发止损) -> M3 -> M5`。

### What changed (files + behavior impact)
- 计划状态更新：
  - `PLANS_v6.md`（`status=completed_conclude`，补 `completed_on/final_decision`）
- 新增最终结论文档：
  - `docs/plansv6_final_verdict.md`
- 更新 stage 快照：
  - `docs/stage_acceptance_summary.md`（新增 `PLANS_v6 Closure Snapshot (2026-04-16)`）
- 代码级改动（M3 方向探索，均在 consistency 路线内）：
  - `dexscrew/algo/ppo/consistency_latent_student.py`
    - 新增 `consistency_obs_noise_curriculum*`
    - 新增 `consistency_train_align_infer`
    - 新增 `consistency_use_ema_target` / `consistency_ema_decay` 与 EMA target 路径
- 治理升级记录：
  - `codeagent_issue.md`（按 AGENTS 边界，登记“diffusion 未超越当前 baseline”阻塞）

### What was verified (commands + key outcomes)
- M0 baseline 锁定：
  - `sha1sum outputs/XHandHoraScrewDriver_student_consistency/v5_5_m1_boundary_bc_tuned_seed42_15min/stage2_consistency_nn/model_best.ckpt`
  - baseline hash: `45d785763f385bfb0a5326866eb31c9cbcbb3215`
  - PAdapt reference（来自 `docs/stage_acceptance_summary.md`）:
    - nominal `2.167820`, light_v2 `2.079074`, hard `1.838225`

- M1 probe 1 (`capacity_boost`, hidden_dim=512, 15min)：
  - train run:
    - `outputs/XHandHoraScrewDriver_student_consistency/v6_m1_capacity_boost_seed42_15min/`
    - ckpt sha1: `b07d4bbfa11b04ccaeca366ec0c1df40ab98fdb2`
    - `Current Best max=1713.12`
  - eval root:
    - `outputs/robustness_eval/v6_m1_capacity_boost_seed42/`
  - result:
    - nominal `1.451814/0.001383`
    - light_v2 `1.668441/0.000977`
    - hard `1.255473/0.001628`

- M1 probe 2 (`longer_train`, 30min)：
  - train run:
    - `outputs/XHandHoraScrewDriver_student_consistency/v6_m1_longer_train_seed42_30min/`
    - ckpt sha1: `da3fbf8d4c0ff940b8500c801c374bcdcb78b65c`
    - `Current Best max=1765.68`
  - eval root:
    - `outputs/robustness_eval/v6_m1_longer_train_seed42/`
  - result:
    - nominal `2.064455/0.000977`
    - light_v2 `1.858506/0.001546`
    - hard `1.415326/0.001953`

- M1 probe 3 (`lr_schedule`, consistency_lr=1e-4, 15min)：
  - train run:
    - `outputs/XHandHoraScrewDriver_student_consistency/v6_m1_lr_schedule_seed42_15min/`
    - ckpt sha1: `572ff6d91d8482f6543b5dc35f9b9da67e4fa50d`
    - `Current Best max=1641.83`
  - eval root:
    - `outputs/robustness_eval/v6_m1_lr_schedule_seed42/`
  - result:
    - nominal `2.291926/0.000814`
    - light_v2 `2.055887/0.001058`
    - hard `1.335743/0.001628`

- M1 gate decision:
  - 三个探针 `hard` 均低于 `1.780`；且相对 V5.5 seed42 baseline 的 hard 增益均 `< +0.02`。
  - 触发 `PLANS_v6` 规则：跳过 `M2`，进入 `M3`。

- M3 direction A (`obs_noise_curriculum`)：
  - 代码开关：
    - `+train.ppo.consistency_obs_noise_curriculum=True`
    - `+train.ppo.consistency_obs_noise_e_target=0.05`
    - `+train.ppo.consistency_obs_noise_t_target=0.025`
  - train run:
    - `outputs/XHandHoraScrewDriver_student_consistency/v6_m3_obs_noise_curriculum_seed42_15min/`
    - ckpt sha1: `2fd9f89b94f9814fbe039315afdcf911dadc71c0`
    - `Current Best max=1614.48`
  - eval root:
    - `outputs/robustness_eval/v6_m3_obs_noise_curriculum_seed42/`
  - result:
    - nominal `2.022649/0.001221`
    - light_v2 `1.830568/0.001953`
    - hard `1.387970/0.002441`

- M3 direction B (`infer2_align`)：
  - 代码开关：
    - `+train.ppo.consistency_infer_steps=2`
    - `+train.ppo.consistency_train_align_infer=True`
  - train run:
    - `outputs/XHandHoraScrewDriver_student_consistency/v6_m3_infer2_align_seed42_15min/`
    - ckpt sha1: `19d2c9afc45eeb28c69f7531736c73565b216da6`
    - `Current Best max=1728.99`
  - eval root:
    - `outputs/robustness_eval/v6_m3_infer2_align_seed42/`
  - result:
    - nominal `1.980866/0.001058`
    - light_v2 `1.786391/0.001546`
    - hard `1.622737/0.001302`

- M3 direction C (`ema_target`)：
  - 代码开关：
    - `+train.ppo.consistency_use_ema_target=True`
    - `+train.ppo.consistency_ema_decay=0.995`
  - train run:
    - `outputs/XHandHoraScrewDriver_student_consistency/v6_m3_ema_target_seed42_15min/`
    - ckpt sha1: `b9b8d2c1b3ccf522ce937f852ebe97280f230ffe`
    - `Current Best max=1698.92`
  - eval root:
    - `outputs/robustness_eval/v6_m3_ema_target_seed42/`
  - result:
    - nominal `1.736832/0.001383`
    - light_v2 `2.077133/0.001058`
    - hard `1.723353/0.002116`

- M3 gate decision:
  - 三方向 hard 分别为 `1.387970 / 1.622737 / 1.723353`，均 `< 1.780`。
  - 未满足进入 `M4` multiseed 前置条件。

### Local decision
- `PLANS_v6` 结论：`Conclude`。
- 主因：在 `M1+M3` 预算内无候选达到 V6 hard 门槛，无法触发 multiseed 验收。
- baseline 位置保持：`padapt` 主线不变。

### Remaining blocked/risky
- 治理级阻塞：在当前 V6 边界（局部超参 + 轻量 consistency 代码改动）下，未能证明 diffusion 路线可超越当前主线 baseline。
- 已按 AGENTS 规则更新 `codeagent_issue.md`。

### Single recommended next step
- 治理确认 `V6 Conclude` 并冻结当前扩张线；若继续，应新开计划并授权超出 V6 范围的改动（例如训练分布/数据策略层面），否则将重复低收益试探。

---

## v2-096 (2026-04-16) — V6 执行审阅：代码审阅与证据化结论复核

### Target milestone/subgoal
- 按用户要求对当前 `diffusion / consistency / plan` 执行状态做一次审阅：
  - 审阅 `dexscrew/algo/ppo/consistency_latent_student.py` 中 V6 新增逻辑（obs noise curriculum / train-infer alignment / EMA target）
  - 复核 `PLANS_v6.md`、`docs/plansv6_final_verdict.md`、`docs/session_handoff_v2.md`（`v2-093`~`v2-095`）、`docs/stage_acceptance_summary.md`、`codeagent_issue.md`
  - 判断当前 `V6 Conclude` 结论是否证据充分、记录是否自洽、是否仍有明显遗漏/风险

### What changed (files + behavior impact)
- 更新：
  - `docs/session_handoff_v2.md`
- 行为影响：
  - 无训练/评测入口代码改动；
  - 增加一条执行审阅记录，明确当前结论的支持度、已确认风险、以及唯一建议复核点。

### What was verified (commands + key outcomes)
- bootstrap 必读文件：
  - `sed -n '1,240p' AGENTS.md`
  - `sed -n '1,260p' PLANS_v6.md`
  - `sed -n '1,260p' docs/plansv6_final_verdict.md`
  - `sed -n '5092,5365p' docs/session_handoff_v2.md`
  - `sed -n '340,470p' docs/stage_acceptance_summary.md`
  - `sed -n '1,240p' codeagent_issue.md`
  - `nl -ba dexscrew/algo/ppo/consistency_latent_student.py | sed -n '1,680p'`

- 代码与接线复核：
  - `nl -ba dexscrew/algo/ppo/diffusion_latent_student.py | sed -n '1,360p'`
  - `nl -ba train.py | sed -n '1,240p'`
  - `nl -ba student_eval.py | sed -n '1,240p'`
  - `rg -n "consistency_(obs_noise|train_align_infer|use_ema_target|ema_decay|infer_steps)|stage2_consistency_nn|consistency_latent_student|student_mode.*consistency|consistency" train.py student_eval.py dexscrew/algo/ppo -g '!**/__pycache__/**'`

- 产物与记录一致性复核：
  - `find outputs/robustness_eval -maxdepth 2 -type f | rg 'v6_(m1|m3).*consistency_(nominal|light_v2|hard)\\.log$' | sort`
  - `rg -n "EvalSummary|EvalReconSummary|load_path|consistency_|seed|test_num_steps|randomForceProbScalar|obs_noise" outputs/robustness_eval/v6_m3_{obs_noise_curriculum,infer2_align,ema_target}_seed42/*.log`
  - outcome:
    - `v2-095` / `plansv6_final_verdict` / stage summary 中记录的 V6 M3 指标与现有 eval logs 一致；
    - 统一评测口径仍为 `seed=42`, `steps=256`, `nominal + light_v2 + hard`。

- 关键实现级发现：
  - EMA 权重确实被保存，但当前 eval/infer 路径未使用：
    - `python - <<'PY' ... torch.load(v6_m3_ema_target ...); print(sorted(obj.keys()))`
    - outcome: `v6_m3_ema_target` ckpt 含 `consistency_ema_model`
  - EMA 与在线权重并不相同：
    - `python - <<'PY' ... print(avg_mean_abs_diff, max_abs_diff) ...`
    - outcome: `avg_mean_abs_diff=0.001333...`, `max_abs_diff=0.023499...`
  - 当前环境无法直接做 Isaac Gym 最小复核：
    - `python -V` -> `Python 3.12.7`
    - `source scripts/_ensure_isaacgym_env.sh && ensure_isaacgym_env`
    - outcome: 本地 shell 不满足 `Python 3.8 + isaacgym importable`，因此本会话未追加 live re-eval。

- 计划门槛与配置复核：
  - `nl -ba PLANS_v6.md | sed -n '90,320p'`
  - `nl -ba docs/plansv6_final_verdict.md | sed -n '1,140p'`
  - `nl -ba docs/stage_acceptance_summary.md | sed -n '108,130p;420,440p'`
  - `rg -n "load_path:|checkpoint:" outputs/XHandHoraScrewDriver_student_consistency/v6_*/*yaml`
  - outcome:
    - 文档多处将 `1.780` 写成 `PAdapt hard mean - 1σ`，但 `stage_acceptance_summary` 中 `padapt hard = 1.838225 ± 0.105458`，按数值应约为 `1.732767`；
    - 尽管该算术口径写错，现有最佳 `ema_target hard = 1.723353` 仍低于修正后的 `~1.733`，所以该问题单独不足以推翻 `Conclude`；
    - V6 实际训练配置 `train.load_path` 指向 teacher ckpt，而不是 V5.5 accepted consistency ckpt，说明执行上采用的是“accepted config 复跑/变体”而非“accepted ckpt 续训”。

### Local decision
- 本次审阅未发现足以直接推翻当前 `V6 Conclude` 的实现级问题。
- 当前更准确的本地判断为：`support_but_with_risks`
  - 支持点：
    - 现有 V6 结果日志、handoff、final verdict、stage summary 主体数字一致；
    - 即便修正 `PAdapt - 1σ` 算术，当前最佳 `ema_target` 仍未达到 corrected hard 门槛。
  - 风险点：
    - `ema_target` 方向的现有评估没有实际测试保存下来的 EMA 权重本身，只测试了在线权重；
    - `obs_noise_curriculum` 候选在动态训练难度下仍用原始训练 reward 选 `model_best`，存在 checkpoint 选择偏差；
    - `PLANS_v6` 方向 B 文案写的是“训练时也用 2-step consistency loss”，但当前实现只把 rollout 对齐接入 BC 分支，不是完整的 2-step consistency loss 版本；
    - `PLANS_v6` 中 “V5.5 accepted config + checkpoint 作为 baseline” 与实际 V6 run 从 teacher ckpt 起训存在表述/执行偏差。

### Remaining blocked/risky
- 方法学风险：
  - `ema_target` 是最接近门槛的 M3 候选，但由于 eval 未覆盖 EMA 权重本身，其负结论仍留有一个最小复核缺口。
- 文档自洽风险：
  - `1.780 = PAdapt hard mean - 1σ` 的表述与 acceptance summary 数值不自洽，后续若直接引用到论文或答辩材料，容易被追问。
- 环境风险：
  - 当前 shell 不具备本地 Isaac Gym 运行条件，若要补做最小复核，需要进入仓库既有 `docker-run-isaacgym.sh` / Python 3.8 环境。

### Single recommended next step
- 若要在最终冻结前补一个且只补一个复核点，优先在 Isaac Gym 3.8 环境中对 `v6_m3_ema_target` 做一次 **hard-only EMA-weight eval**（将保存的 `consistency_ema_model` 临时作为 `consistency_model` 评测）；若仍低于 corrected hard 1σ 门槛（约 `1.733`），则可放心接受当前 `V6 Conclude`。

---

## v2-097 (2026-04-16) — V7 M2 Candidate A Closed, Candidate B Started

### Target milestone/subgoal
- 继续执行 `PLANS_v7`：
  - 先确认 `V7-M2` 候选 A (`ema_obs_combo_seed42_15min`) 的训练产物状态；
  - 按统一协议完成其 `nominal + light_v2 + hard` 三条件评测；
  - 若失败，则按计划顺序启动候选 B (`ema_target_seed42_30min`)。

### What changed (files + behavior impact)
- 更新：
  - `PLANS_v7.md`
  - `docs/plansv7_final_verdict.md`
  - `docs/session_handoff_v2.md`
- 行为影响：
  - 无训练/评测代码改动；
  - 候选 A 已完成判定并淘汰；
  - 候选 B 已起训，后续会在相同协议下继续评测。

### What was verified (commands + key outcomes)
- 候选 A 训练状态与产物确认：
  - `ls -la outputs/XHandHoraScrewDriver_student_consistency/v7_m2_ema_obs_combo_seed42_15min`
  - `find outputs/XHandHoraScrewDriver_student_consistency/v7_m2_ema_obs_combo_seed42_15min -maxdepth 2 -type f -printf '%TY-%Tm-%Td %TH:%TM:%TS %p\n' | sort | tail -n 20`
  - `sha1sum outputs/XHandHoraScrewDriver_student_consistency/v7_m2_ema_obs_combo_seed42_15min/stage2_consistency_nn/model_best.ckpt`
  - outcome:
    - 旧训练会话已结束于外层 `timeout`
    - `model_best.ckpt` 有效存在，sha1=`1c7c75577496a00234b1007e89fb2715ea391d7c`

- 候选 A 三条件 eval：
  - nominal:
    - `./docker-run-isaacgym.sh timeout 480 bash scripts/eval_screwdriver_student_robustness.sh 0 42 ConsistencyLatentStudent outputs/XHandHoraScrewDriver_student_consistency/v7_m2_ema_obs_combo_seed42_15min/stage2_consistency_nn/model_best.ckpt 256 v7_m2_ema_obs_combo_seed42_nominal +train.ppo.consistency_use_ema_target=True +train.ppo.consistency_ema_decay=0.995 +train.ppo.consistency_infer_use_ema=True +train.ppo.consistency_boundary_coef=0.8 +train.ppo.consistency_num_scales=16 +train.ppo.bc_loss_coef=1.2`
    - result: `EvalSummary steps=256 avg_reward=0.211057 avg_done_rate=0.001953`
    - recon: `latent_mse=0.228280 latent_l1=0.379646 action_mse_to_teacher=0.229339`
  - light_v2:
    - `./docker-run-isaacgym.sh timeout 480 bash scripts/eval_screwdriver_student_robustness.sh 0 42 ConsistencyLatentStudent outputs/XHandHoraScrewDriver_student_consistency/v7_m2_ema_obs_combo_seed42_15min/stage2_consistency_nn/model_best.ckpt 256 v7_m2_ema_obs_combo_seed42_light_v2 +train.ppo.consistency_use_ema_target=True +train.ppo.consistency_ema_decay=0.995 +train.ppo.consistency_infer_use_ema=True +train.ppo.consistency_boundary_coef=0.8 +train.ppo.consistency_num_scales=16 +train.ppo.bc_loss_coef=1.2 task.env.randomization.obs_noise_e_scale=0.03 task.env.randomization.obs_noise_t_scale=0.015 task.env.forceScale=1.0 task.env.randomForceProbScalar=0.2`
    - result: `EvalSummary steps=256 avg_reward=0.296649 avg_done_rate=0.001709`
    - recon: `latent_mse=0.226521 latent_l1=0.378764 action_mse_to_teacher=0.216359`
  - hard:
    - `./docker-run-isaacgym.sh timeout 480 bash scripts/eval_screwdriver_student_robustness.sh 0 42 ConsistencyLatentStudent outputs/XHandHoraScrewDriver_student_consistency/v7_m2_ema_obs_combo_seed42_15min/stage2_consistency_nn/model_best.ckpt 256 v7_m2_ema_obs_combo_seed42_hard +train.ppo.consistency_use_ema_target=True +train.ppo.consistency_ema_decay=0.995 +train.ppo.consistency_infer_use_ema=True +train.ppo.consistency_boundary_coef=0.8 +train.ppo.consistency_num_scales=16 +train.ppo.bc_loss_coef=1.2 task.env.randomization.obs_noise_e_scale=0.05 task.env.randomization.obs_noise_t_scale=0.025 task.env.forceScale=1.5 task.env.randomForceProbScalar=0.3`
    - result: `EvalSummary steps=256 avg_reward=0.355474 avg_done_rate=0.002279`
    - recon: `latent_mse=0.228203 latent_l1=0.380401 action_mse_to_teacher=0.226538`

- 候选 A gate decision：
  - outcome:
    - `nominal/light_v2/hard` 全部远低于 `V7` single-seed entry gate
    - 不满足候选 D 触发条件（最佳结果未达到 `hard>=1.700` 且 `nominal>=2.100`）
    - 候选 A 淘汰

- 候选 B 启动：
  - train command:
    - `./docker-run-isaacgym.sh timeout 1800 python train.py task=XHandHoraScrewDriver headless=True seed=42 sim_device=cuda:0 rl_device=cuda:0 graphics_device_id=7 train.algo=ConsistencyLatentStudent train.ppo.proprio_adapt=True train.ppo.output_name=XHandHoraScrewDriver_student_consistency/v7_m2_ema_target_seed42_30min checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_1550.49.pth wandb_activate=False +train.ppo.consistency_boundary_coef=0.8 +train.ppo.consistency_num_scales=16 +train.ppo.bc_loss_coef=1.2 +train.ppo.consistency_use_ema_target=True +train.ppo.consistency_ema_decay=0.995 +train.ppo.consistency_infer_use_ema=True`
  - early outcome:
    - 训练已正常启动，run dir 存在：
      - `outputs/XHandHoraScrewDriver_student_consistency/v7_m2_ema_target_seed42_30min/`
    - 早期 restore warning:
      - `Checkpoint missing sa_mean_std during train restore; using current defaults.`
      - `Checkpoint missing agent_steps during train restore; resume will start from current counter.`
    - 该 warning 与从 teacher ckpt fresh 起训一致，不构成异常。

### Remaining blocked/risky
- 候选 B 仍在训练中，尚未产生可用于结论的最终 eval 结果。
- 若候选 B 仍明显低于门槛，则 `V7-M2` 基本只剩候选 C 可复核；候选 D 已大概率失去触发可能。
- 候选 A 的异常低 deploy 表现提示：`EMA + obs_noise_curriculum` 组合在当前实现下可能存在显著训练/部署失配，不建议在没有新证据前继续加预算。

### Single recommended next step
- 等候选 B (`ema_target_seed42_30min`) 训练结束后，立即按 `nominal + light_v2 + hard` 协议完成评测；若仍未接近 `hard >= 1.733`，则进入候选 C (`ema_alignfix_seed42_15min`)，否则再决定是否需要 multiseed。

---

## v2-098 (2026-04-16) — PLANS_v7 完整收口：B/C 完成评测，D/M3 未触发，结论 completed_conclude

### Target milestone/subgoal
- 完成 `PLANS_v7` 的剩余执行与验收：
  - 确认候选 B (`ema_target_seed42_30min`) 的最终产物与 gate 结果；
  - 完成候选 C (`ema_alignfix_seed42_15min`) 的 `hard` 评测并据此判断是否触发候选 D；
  - 收口 `PLANS_v7.md`、`docs/plansv7_final_verdict.md`、`docs/stage_acceptance_summary.md`。

### What changed (files + behavior impact)
- 更新：
  - `PLANS_v7.md`
  - `docs/plansv7_final_verdict.md`
  - `docs/stage_acceptance_summary.md`
  - `docs/session_handoff_v2.md`
- 行为影响：
  - 无新的训练/评测入口代码改动；
  - `PLANS_v7` 已从执行态切换为 `completed_conclude`；
  - `ConsistencyLatentStudent` 在当前 V7 bounded scope 下未获得可进入 multiseed 的 single-seed 候选。

### What was verified (commands + key outcomes)
- bootstrap 对齐：
  - `tail -n 220 docs/session_handoff_v2.md`
  - `sed -n '1,220p' docs/stage_acceptance_summary.md`

- 候选 C hard eval 完成：
  - `write_stdin(session_id=69143, chars=\"\")`
  - result:
    - `EvalSummary steps=256 avg_reward=1.135807 avg_done_rate=0.001953`
    - `EvalReconSummary steps=256 mode=consistency latent_mse=0.119058 latent_l1=0.227084 action_mse_to_teacher=0.226580`

- V7 候选 artifact hash 对齐：
  - `sha1sum outputs/XHandHoraScrewDriver_student_consistency/v7_m2_ema_obs_combo_seed42_15min/stage2_consistency_nn/model_best.ckpt outputs/XHandHoraScrewDriver_student_consistency/v7_m2_ema_target_seed42_30min/stage2_consistency_nn/model_best.ckpt outputs/XHandHoraScrewDriver_student_consistency/v7_m2_ema_alignfix_seed42_15min/stage2_consistency_nn/model_best.ckpt`
  - outcome:
    - candidate A sha1 = `1c7c75577496a00234b1007e89fb2715ea391d7c`
    - candidate B sha1 = `1c7c75577496a00234b1007e89fb2715ea391d7c`
    - candidate C sha1 = `024409e9f765232775f33f139493e04c09ae29d2`
  - implication:
    - B 的 `model_best.ckpt` 与 A 完全相同，30min EMA-only 训练没有产生新的最优 artifact

- 候选 B 产物时间戳确认：
  - `find outputs/XHandHoraScrewDriver_student_consistency/v7_m2_ema_target_seed42_30min -maxdepth 2 -type f -printf '%TY-%Tm-%Td %TH:%TM:%TS %p\n' | sort | tail -n 20`
  - outcome:
    - `model_best.ckpt` 停留在 `2026-04-16 16:31:14`
    - 后续只有 tb event 继续更新，符合“训练继续但 best 未刷新”的状态

- 文档收口更新前查看：
  - `sed -n '1,260p' PLANS_v7.md`
  - `sed -n '1,260p' docs/plansv7_final_verdict.md`
  - `tail -n 120 docs/stage_acceptance_summary.md`

### Local decision
- `PLANS_v7 = completed_conclude`
- 关键原因：
  - `V7-M1` EMA recheck 失败：`hard=1.608308 < 1.733`
  - `V7-M2-A/B` 均停留在同一 best ckpt，且三条件 reward 全部极低
  - `V7-M2-C` 虽训练信号改善，但最终 deploy 指标仅：
    - nominal `1.124038/0.002035`
    - light_v2 `1.177704/0.002279`
    - hard `1.135807/0.001953`
  - 因而 A/B/C 全部未达到：
    - `hard >= 1.733`
    - `nominal >= 2.100`
    - `light_v2 >= 1.900`
- 由此：
  - 候选 D 不触发
  - `V7-M3` multiseed 不触发
  - `padapt` 继续保持主线 baseline 地位

### Remaining blocked/risky
- 当前阻塞已不是实现级风险，而是方法边界：
  - V7 已关闭 EMA 推理/restore 相关残余风险，但在不扩 scope 的前提下，Consistency 仍无法跨过 corrected hard single-seed gate。
- 若继续尝试，需要新的计划边界和新的优化假设；在当前 scope 内继续加时长/重复 sweep 的收益很低。

### Single recommended next step
- 接受 `PLANS_v7 completed_conclude`，冻结当前 consistency 扩张线；若后续要继续，只在新计划中引入**超出 V7 范围**的明确新假设，否则不建议重复当前 bounded sweep。

---

## v2-099 (2026-04-16) — V7 文档收尾：补计划结束元信息并将 escalation issue 升级到 V7

### Target milestone/subgoal
- 完成 `PLANS_v7` 相关收尾：
  - 给计划文档补齐标准结束元信息；
  - 将 `codeagent_issue.md` 从 `V6` 语境更新到当前 `V7 completed_conclude` 状态。

### What changed (files + behavior impact)
- 更新：
  - `PLANS_v7.md`
  - `codeagent_issue.md`
  - `docs/session_handoff_v2.md`
- 行为影响：
  - 无训练/评测/代码路径改动；
  - 文档层面的计划状态与 escalation 语境已与 `V7` 最终结论对齐。

### What was verified (commands + key outcomes)
- closure 对齐检查：
  - `sed -n '1,240p' codeagent_issue.md`
  - `sed -n '1,120p' PLANS_v7.md`
  - `sed -n '1,40p' PLANS_v6.md`
  - `sed -n '1,40p' PLANS_v5_5.md`
  - `git status --short`
- outcome:
  - `PLANS_v6` / `PLANS_v5_5` 均包含 `completed_on` 与 `final_decision`，因此对 `PLANS_v7` 做了同口径补齐；
  - 原 `codeagent_issue.md` 仍停留在 `V6` 的 `1.780` gate 叙述，已更新为 `V7` corrected gate 与 bounded-sweep 结论；
  - 仓库本身存在较多既有未提交改动，本次只追加最小文档补丁，未触碰其他工作树内容。

### Local decision
- `PLANS_v7` 的收尾文档现在已完整：
  - `status=completed_conclude`
  - `completed_on=2026-04-16`
  - `final_decision=conclude_no_single_seed_breakthrough_under_v7_scope`
- `codeagent_issue.md` 继续保持 `open`，但其阻塞语境已升级为：
  - `diffusion_cannot_show_value_beyond_current_student_baseline_under_v7_scope`

### Remaining blocked/risky
- 当前剩余事项不再是执行层问题，而是治理层选择：
  - 接受 `V7` 收口并冻结 consistency 扩张线；
  - 或者显式新开一个超出 `V7` 边界的新计划。

### Single recommended next step
- 若本轮只做收尾，到此即可；下一次若继续推进，应直接基于 `codeagent_issue.md` 的建议决定“冻结”还是“新开超范围计划”，而不是回到 `V7` 内重复 sweep。

---

## v2-100 (2026-04-17) — V8 flow sprint: M0/M1 completed, M2 candidate 1/2/5 rejected, candidate 4 training in progress

### Target milestone/subgoal
- 执行 `PLANS_v8`：
  - 完成 flow matching 的 M0 工程闭环补丁；
  - 做旧 flow artifact 的 M1 eval-only probe；
  - 推进 M2 fresh seed42 sweep，优先清理 `infer2` family。

### What changed (files + behavior impact)
- 代码更新：
  - `dexscrew/algo/ppo/flow_matching_latent_student.py`
- 文档更新：
  - `PLANS_v8.md`
  - `docs/plansv8_final_verdict.md`
  - `docs/session_handoff_v2.md`
- 行为影响：
  - 新增 flow 训练/推理对齐与恢复闭环能力：
    - `_init_latent(...)`
    - `_sample_latent_rollout(...)`
    - `flow_train_init_mode`
    - `flow_train_align_infer`
    - `flow_rollout_bc_coef`
    - `save()/restore_*()` 的 `agent_steps` / `sa_mean_std` handling
  - 旧 `v5_m2_flow_baseline` 在默认 1-step eval 下数值保持不变，说明 M0 默认行为兼容。

### What was verified (commands + key outcomes)
- static / compatibility:
  - `PYTHONPYCACHEPREFIX=/tmp/codex_pycache python -m py_compile dexscrew/algo/ppo/flow_matching_latent_student.py train.py student_eval.py`
  - `python - <<'PY' ... torch.load('outputs/XHandHoraScrewDriver_student_flow_matching/v5_m2_flow_baseline_seed42_15min/stage2_flow_nn/model_best.ckpt') ... PY`
  - `./docker-run-isaacgym.sh timeout 480 bash scripts/eval_screwdriver_student_robustness.sh 0 42 FlowMatchingLatentStudent outputs/XHandHoraScrewDriver_student_flow_matching/v5_m2_flow_baseline_seed42_15min/stage2_flow_nn/model_best.ckpt 256 v8_m0_flow_baseline_compat_seed42 +train.ppo.flow_stochastic_infer=False`
- M1 eval-only probes:
  - `infer2`:
    - nominal `1.805502 / 0.001546`
    - light_v2 `1.681982 / 0.001953`
    - hard `1.428555 / 0.002604`
  - `infer4`:
    - nominal `1.885794 / 0.001302`
    - light_v2 `1.640372 / 0.002523`
    - hard `1.189338 / 0.002930`
  - conclusion:
    - `infer2` 是整体更强的 multistep 方向；
    - `infer4` 只提升 nominal，但 robust 更差。
- M2 candidate 1 `v8_m2_flow_zeroinit_bc12_seed42_15min`:
  - train:
    - `Current Best` 持续卡在 `-1.62`，~`4M` agent steps 提前停止
  - eval:
    - nominal `0.992769 / 0.001872`
    - light_v2 `0.921589 / 0.002523`
    - hard `1.020256 / 0.002604`
  - decision:
    - 明确低于当前 flow baseline，淘汰
- M2 candidate 2 `v8_m2_flow_align2_rollout_seed42_15min`:
  - train:
    - `Current Best` 在 ~`2M` agent steps 内始终未刷新，`model_best.ckpt` 时间戳不变
  - eval:
    - nominal `1.082436 / 0.002035`
    - light_v2 `1.108353 / 0.002116`
    - hard `0.750190 / 0.003174`
  - decision:
    - `infer2 + rollout BC` 失败，淘汰
- M2 candidate 5 `v8_m2_flow_align2_bcheavy_seed42_15min`:
  - train:
    - `Current Best` 在 ~`1M` agent steps 内始终未刷新，提前停止
  - eval:
    - nominal `1.124580 / 0.001872`
    - light_v2 `1.055061 / 0.002441`
    - hard `1.053348 / 0.002279`
  - decision:
    - 更重 BC 仅小幅拉回 nominal，仍显著低于 baseline，淘汰
- M2 candidate 4 `v8_m2_flow_align2_anchor_seed42_15min`:
  - train:
    - 已启动并观察到 ~`1M` agent steps
    - `Current Best` 依旧始终停在 `-1.62`
    - 已按与 candidate 2/5 相同标准提前停止，避免无效占用 GPU
  - eval:
    - 尚未执行；仅差补三条件评测作为证据收口

### Local decision
- `V8-M0` 完成且兼容旧 flow baseline。
- `V8-M1` 已补齐并锁定：`infer2 > infer4`（robust 维度）。
- `V8-M2` 目前已明确拒绝：
  - candidate 1 `zeroinit_bc12`
  - candidate 2 `align2_rollout`
  - candidate 5 `align2_bcheavy`
- 当前最强观察不是“哪个候选接近 continue gate”，而是：
  - 整个 `infer2` family 在 fresh-train 下都表现出非常相似的失败训练模式；
  - 即使部署端略有差异，也远低于 single-seed continue gate。

### Remaining blocked/risky
- `V8` 仍未正式收口，因为：
  - candidate 4 `align2_anchor` 训练已确认复现相同失败模式，但还缺 eval 结果；
  - candidate 3 `align4_rollout` 尚未执行。
- 当前主要风险不是实现错误，而是方法边界：
  - 若 candidate 4 继续复现 `Current Best = -1.62` 的模式，则 `infer2` family 基本可判定已被扫清；
  - `infer4` 在 M1 已显示 robust 更差，后续更像形式化收口验证，而非高期望救火线。

### Single recommended next step
- 下一步先补完 candidate 4 `align2_anchor` 的 `nominal + light_v2 + hard` 三条件 eval；
  - 若其结果继续低于当前 flow baseline，则 `infer2` family 可视为正式扫清；
  - 之后只需决定是否对 candidate 3 `align4_rollout` 做最小收口验证，再判断 `V8 conclude`。

---

## v2-101 (2026-04-17) — V8 completed_conclude: candidate 4/3 closed out, no flow continue signal

### Target milestone/subgoal
- 完成 `PLANS_v8` 收口：
  - 补完 candidate 4 `align2_anchor` 的三条件 eval；
  - 完成最后一个 M2 fresh candidate `align4_rollout`；
  - 判断 `M2.5 / M3` 是否触发并写出最终 verdict。

### What changed (files + behavior impact)
- 更新：
  - `PLANS_v8.md`
  - `docs/plansv8_final_verdict.md`
  - `docs/stage_acceptance_summary.md`
  - `docs/session_handoff_v2.md`
  - `codeagent_issue.md`
- 行为影响：
  - 无代码路径新增改动；
  - `PLANS_v8` 已正式切换到 `completed_conclude`；
  - 仓库的 flow matching 方向在当前 scope 下被标记为“不值得继续投入”。

### What was verified (commands + key outcomes)
- candidate 4 eval:
  - `./docker-run-isaacgym.sh timeout 480 bash scripts/eval_screwdriver_student_robustness.sh ... v8_m2_flow_align2_anchor_seed42_nominal ...`
  - `./docker-run-isaacgym.sh timeout 480 bash scripts/eval_screwdriver_student_robustness.sh ... v8_m2_flow_align2_anchor_seed42_light_v2 ...`
  - `./docker-run-isaacgym.sh timeout 480 bash scripts/eval_screwdriver_student_robustness.sh ... v8_m2_flow_align2_anchor_seed42_hard ...`
  - outcome:
    - nominal `1.000670 / 0.002035`
    - light_v2 `0.923009 / 0.002360`
    - hard `0.953265 / 0.002686`
    - `infer2` family 至此证据化扫清
- candidate 3 train/eval:
  - train:
    - `./docker-run-isaacgym.sh timeout 1000 python train.py ... output_name=XHandHoraScrewDriver_student_flow_matching/v8_m2_flow_align4_rollout_seed42_15min ... +train.ppo.flow_infer_steps=4 ...`
    - 观测到 ~`1M` agent steps 内 `Current Best` 始终卡在 `-1.62`，提前停止
  - eval:
    - `./docker-run-isaacgym.sh timeout 480 bash scripts/eval_screwdriver_student_robustness.sh ... v8_m2_flow_align4_rollout_seed42_nominal ...`
    - `./docker-run-isaacgym.sh timeout 480 bash scripts/eval_screwdriver_student_robustness.sh ... v8_m2_flow_align4_rollout_seed42_light_v2 ...`
    - `./docker-run-isaacgym.sh timeout 480 bash scripts/eval_screwdriver_student_robustness.sh ... v8_m2_flow_align4_rollout_seed42_hard ...`
  - outcome:
    - nominal `1.114895 / 0.001790`
    - light_v2 `1.152107 / 0.001872`
    - hard `1.079496 / 0.002848`
    - best fresh candidate, but still clearly below current flow baseline
- closeout checks:
  - `sha1sum` / `stat` on candidate 3 and 4 `model_best.ckpt`
  - 文档对齐更新到：
    - `PLANS_v8.md`
    - `docs/plansv8_final_verdict.md`
    - `docs/stage_acceptance_summary.md`
    - `codeagent_issue.md`

### Local decision
- `PLANS_v8` 最终结论：`completed_conclude`
- 理由：
  - `M2` 的 5 个 fresh seed42 candidates 全部失败
  - best fresh candidate `align4_rollout` 仍远低于 V8 continue gate：
    - nominal `1.114895 < 1.95`
    - light_v2 `1.152107 < 1.70`
    - hard `1.079496 < 1.55`
    - hard_done `0.002848 > 0.002372`
  - `M2.5` 不触发
  - `M3` 不触发

### Remaining blocked/risky
- 当前剩余的不再是实现风险，而是 scope 边界：
  - 在 V8 所覆盖的 flow-native 小改动范围内，Flow Matching 没有显示出继续投入的价值。
- 若还要继续推进 flow，需要新的计划边界和新的方法假设；在当前 family 内重复换小系数或继续加时长，收益预期很低。

### Single recommended next step
- 接受 `PLANS_v8 completed_conclude`，冻结当前 flow matching 扩张线；若后续仍要推进 diffusion，请新开一个**超出 V8 边界**的新计划，而不是继续重复当前 bounded sweep。

---

## v2-102 (2026-04-17) — Algorithm doc expanded into repo-wide student algorithm reference

### Target milestone/subgoal
- 把现有 `docs/diffusion_algorithm.md` 从单一 diffusion 路线说明，扩写成一份覆盖当前仓库主算法族的统一说明文档，便于：
  - 查实现入口
  - 对齐实验结论
  - 直接支持论文/汇报写作

### What changed (files + behavior impact)
- 更新：
  - `docs/diffusion_algorithm.md`
- 行为影响：
  - 无训练/评测/导出逻辑变更；
  - 仅新增一份更完整的算法说明文档，覆盖：
    - `teacher_ppo`
    - `padapt`
    - `purebc`
    - `diffusion_latent`
    - `consistency_latent`
    - `flow_matching_latent`
    - `diffusion_action_chunk`
  - 文档中补充了：
    - 各算法定位
    - 输入/输出
    - 公共 backbone 结构与层数
    - 每层作用
    - 项目内表现与优劣势
    - 当前为何仍以 `padapt` 为主线的总结

### What was verified (commands + key outcomes)
- bootstrap/read:
  - `sed -n '1,220p' docs/session_handoff_v2.md`
  - `sed -n '1,220p' docs/stage_acceptance_summary.md`
  - outcome:
    - 确认当前仓库结论仍是：
      - `padapt` 为主线 baseline
      - `V5.5 consistency_boundary_bc_tuned` 为最强 accepted diffusion reference
      - `V8` flow 已 conclude
- code/doc cross-check:
  - `sed -n '1,260p' dexscrew/algo/ppo/padapt.py`
  - `sed -n '1,240p' dexscrew/algo/ppo/pure_bc.py`
  - `sed -n '1,620p' dexscrew/algo/ppo/diffusion_latent_student.py`
  - `sed -n '1,760p' dexscrew/algo/ppo/consistency_latent_student.py`
  - `sed -n '1,760p' dexscrew/algo/ppo/flow_matching_latent_student.py`
  - `sed -n '1,320p' dexscrew/algo/ppo/diffusion_action_chunk_student.py`
  - `sed -n '1,260p' dexscrew/algo/models/models.py`
  - `sed -n '1,260p' dexscrew/algo/models/block.py`
  - outcome:
    - 文档中关于输入/输出、层数、共享骨架、loss 逻辑、trainable 范围的描述已与实现对齐
- config/result spot checks:
  - `sed -n '170,235p' outputs/XHandHoraScrewDriver_student_padapt/run_a/config_032010_8bf90ec.yaml`
  - `sed -n '170,250p' outputs/XHandHoraScrewDriver_student_consistency/v5_5_m1_boundary_bc_tuned_seed42_15min/config_041515_1f8d373.yaml`
  - `sed -n '170,250p' outputs/XHandHoraScrewDriver_student_flow_matching/v5_m2_flow_baseline_seed42_15min/config_041508_1f8d373.yaml`
  - `sed -n '360,520p' docs/stage_acceptance_summary.md`
  - `sed -n '1,220p' docs/plansv5_5_final_verdict.md`
  - `sed -n '1,220p' docs/plansv8_final_verdict.md`
  - outcome:
    - 文档中引用的 backbone 层宽、consistency accepted 指标、flow conclude 结论与现有 artifact 一致
- final review:
  - `git diff -- docs/diffusion_algorithm.md`
  - outcome:
    - 确认本次只改动算法说明文档，没有误触训练代码

### Local decision
- 当前仓库已经有一份可以直接作为“算法说明 / 结果定位 / 实现索引”入口使用的统一文档。
- 若后续用户要写 thesis results、related work 对照、方法章节实现解析，可以优先复用这份文档，而不必再从多个 `plan/final verdict` 文档里手工拼接。

### Remaining blocked/risky
- 当前风险主要不是实现，而是文档边界：
  - `docs/diffusion_algorithm.md` 现在已经覆盖了主要 student family，但它不是逐行代码注释式文档；
  - 若后续需要论文级方法图、公式版 appendix、或英文版 methods subsection，仍需要基于该文档再裁剪一次。

### Single recommended next step
- 若接下来进入写作阶段，优先从 `docs/diffusion_algorithm.md` 拆一版“论文方法小节 + 结果分析小节”，而不是重新从实验 handoff 里拼材料。

---

## v2-103 (2026-04-17) — Asset compatibility check: `dexh13_hand` vs current Pasini hand, `lightbulb` vs current `bulb`

### Target milestone/subgoal
- 在不启动新实验的前提下，先确认新加入的 hand/object asset 与当前 `XHandPasini` 路线的兼容性和差异量级：
  - `assets/dexh13_hand/*`
  - `assets/lightbulb/*`

### What changed (files + behavior impact)
- 更新：
  - `docs/session_handoff_v2.md`
- 行为影响：
  - 无代码/配置/训练逻辑改动；
  - 仅新增一次 asset 勘查结论，方便后续决定是否新开“换 hand / 换 object”的验证计划。

### What was verified (commands + key outcomes)
- 当前 `XHandPasini` hand asset 定位：
  - `sed -n '150,210p' configs/task/XHandPasiniScrewDriver.yaml`
  - `sed -n '120,170p' configs/task/XHandPasiniBulb.yaml`
  - outcome:
    - 当前 Pasini 路线实际使用的是
      - `assets/dexh13_right_description/urdf/dexh13_right_fix_path.urdf`
    - 不是 `xhand_left`
- current Pasini hand body assumptions:
  - `sed -n '2270,2335p' dexscrew/tasks/xhand_pasini.py`
  - outcome:
    - fingertip body 名字硬编码为：
      - `right_index_link_3`
      - `right_middle_link_3`
      - `right_ring_link_3`
      - `right_thumb_link_3`
- hand URDF topology comparison:
  - `diff -q assets/dexh13_right_description/urdf/dexh13_right_fix_path.urdf assets/dexh13_hand/urdf/dexh13_hand_right.urdf`
  - Python/XML grep summary on:
    - `assets/dexh13_right_description/urdf/dexh13_right_fix_path.urdf`
    - `assets/dexh13_hand/urdf/dexh13_hand_right.urdf`
    - `assets/dexh13_hand/urdf/dexh13_hand_right_sim.urdf`
    - `assets/dexh13_hand/urdf/dexh13_hand_right_with_tips.urdf`
  - outcome:
    - current Pasini hand vs `dexh13_hand_right.urdf`:
      - link name set: equal
      - joint name set: equal
      - `28 links / 27 joints / 16 revolute / 11 fixed`
      - therefore kinematic skeleton 基本同源，DOF 级兼容性很高
    - but files are not byte-identical:
      - mesh path style differs (`../meshes/...` vs `meshes/...`)
      - collision organization differs
      - current file contains some parent-link tactile collision composition / palm collision scale handling not identical to new file
    - `dexh13_hand_right_sim.urdf` and `..._with_tips.urdf`:
      - `33 links / 32 joints / 16 revolute / 16 fixed`
      - extra `base` and 4 explicit tip links
      - not a strict 1:1 replacement for the current Pasini hand asset
- current Pasini bulb mapping:
  - `sed -n '2140,2195p' dexscrew/tasks/xhand_pasini.py`
  - outcome:
    - current `bulb` object maps to:
      - `assets/bulb/0000bulb.urdf`
- object URDF comparison:
  - `diff -u assets/bulb/0000bulb.urdf assets/lightbulb/0000_lightbulb.urdf | sed -n '1,260p'`
  - structured parse on both files
  - outcome:
    - both are screw-like 1DOF objects:
      - `3 links / 2 joints`
      - `1 revolute + 1 fixed`
      - same limit range `0 .. 628.3185`
    - but geometry representation differs a lot:
      - current `bulb`:
        - primitive geometry
        - links `base / shaft / handle`
        - sphere + cylinders
      - new `lightbulb`:
        - mesh-based geometry
        - links `base / bolt / nut`
        - STL visual + contact meshes
        - smaller base cylinder
        - different inertial simplification
        - mesh orientation compensated by `rpy`

### Local decision
- `dexh13_hand_right.urdf` 与当前 Pasini hand **不是逐字一样**，但 **骨架和关节拓扑几乎等价**；
  - 如果目标是“保持同一只手的大体控制维度，只换另一份更规范/更新的 dexh13 资产”，这是可行方向。
- `dexh13_hand_right_sim.urdf` / `with_tips` 则不应视为当前 Pasini hand 的 1:1 替身；
  - 它们引入了额外 fixed tip bodies，更适合作为“显式 tip/contact 建模”的新 hand 变体。
- `lightbulb` 和当前 `bulb` 在任务语义上同类，都是单自由度旋拧物体；
  - 但在几何、碰撞、质量和 link naming 上差别明显，足以算一个有意义的新 object robustness 验证项，而不是简单换皮。

### Remaining blocked/risky
- 若后续真的切到 `lightbulb`：
  - 当前 `XHandPasini` 还没有 `lightbulb` 的 object type 映射；
  - 需要新加 asset path mapping 和对应 task yaml；
  - 初始位姿、接触阈值、终止条件可能都要重新校一下。
- 若后续切到 `dexh13_hand_right_sim` / `with_tips`：
  - 需要重新审查 rigid body handles、接触体数量、point cloud / termination 是否受额外 tip body 影响。

### Single recommended next step
- 若目标是低风险验证，优先顺序建议：
  - 先试 `dexh13_hand_right.urdf` 对 current Pasini hand 的替换可行性；
  - 再单独开 `lightbulb` object 分支，因为它对任务几何与接触分布的改变更大。

---

## v2-104 (2026-04-17) — `PLANS_v9` wiring implementation: `Dexh13HoraLightbulb` + `XHandHoraLightbulb` + core student packaging

### Target milestone/subgoal
- 落地 `PLANS_v9` 的工程接入部分：
  - 新 hand family：`Dexh13Hora`
  - 新 lightbulb object 路线：`screw_lightbulb`
  - core student algo 统一导出入口
- 在不启动长训练的前提下完成最小 smoke：
  - env instantiation + reset
  - core student ctor smoke
  - old baseline regression ctor smoke

### What changed (files + behavior impact)
- 代码：
  - `dexscrew/tasks/xhand_hora.py`
    - 增加 hand/object 可配置兼容层：
      - `env.apply_action_mask`
      - `env.action_mask_indices`
      - `env.asset.fingertipBodies`
      - `env.asset.handInitPose`
      - `env.asset.handRootPos`
      - `env.asset.handRootQuat`
      - `env.asset.handRootRPY`
      - `env.asset.handRootPosNoise`
      - `env.asset.handRootPosZScaleComp`
      - `env.asset.dofLowerLimits`
      - `env.asset.dofUpperLimits`
      - `env.asset.dofEffortLimits`
      - `env.asset.dofVelocityLimits`
      - `env.object.init_pos`
      - `env.object.init_pos_noise`
    - 默认行为保持兼容旧 `XHandHoraScrewDriver`
  - `dexscrew/tasks/dexh13_hora.py`
    - 新增 `Dexh13Hora(XHandHora)`，提供 DexH13 默认 fingertip / DOF / init pose 约定
  - `dexscrew/tasks/__init__.py`
    - 注册：
      - `XHandHoraLightbulb`
      - `Dexh13HoraLightbulb`
  - `train.py`
  - `student_eval.py`
    - student 导入切换到 `dexscrew.algo.student`
    - `student_eval.py` 不再写死 `student_dim=24`
- 资产：
  - `assets/screw/lightbulb/0000_lightbulb.urdf`
    - 新增 runtime 使用的 `screw_lightbulb` 资产
    - 使用 primitive geometry，避免原始 draft `lightbulb` 缺少 STL mesh 时直接卡死
- 配置：
  - `configs/task/XHandHoraLightbulb.yaml`
  - `configs/task/Dexh13HoraLightbulb.yaml`
  - `configs/train/XHandHoraLightbulb.yaml`
  - `configs/train/Dexh13HoraLightbulb.yaml`
- student packaging：
  - `dexscrew/algo/student/__init__.py`
  - `dexscrew/algo/student/padapt.py`
  - `dexscrew/algo/student/purebc.py`
  - `dexscrew/algo/student/diffusion_latent.py`
  - `dexscrew/algo/student/consistency_latent.py`
  - `dexscrew/algo/student/flow_matching_latent.py`
- 脚本：
  - `scripts/xhand_lightbulb_teacher.sh`
  - `scripts/dexh13_lightbulb_teacher.sh`
  - `scripts/vis_xhand_lightbulb_teacher.sh`
  - `scripts/vis_dexh13_lightbulb_teacher.sh`
  - `scripts/xhand_lightbulb_student_padapt.sh`
  - `scripts/xhand_lightbulb_student_purebc.sh`
  - `scripts/xhand_lightbulb_student_diffusion_latent.sh`
  - `scripts/xhand_lightbulb_student_consistency.sh`
  - `scripts/xhand_lightbulb_student_flow_matching.sh`
  - `scripts/dexh13_lightbulb_student_padapt.sh`
  - `scripts/dexh13_lightbulb_student_purebc.sh`
  - `scripts/dexh13_lightbulb_student_diffusion_latent.sh`
  - `scripts/dexh13_lightbulb_student_consistency.sh`
  - `scripts/dexh13_lightbulb_student_flow_matching.sh`
- 文档：
  - `PLANS_v9.md`

### What was verified (commands + key outcomes)
- 静态源码 compile：
  - `python - <<'PY' ... compile(src, path, 'exec') ... PY`
  - outcome:
    - `train.py`
    - `student_eval.py`
    - `dexscrew/tasks/xhand_hora.py`
    - `dexscrew/tasks/dexh13_hora.py`
    - `dexscrew/tasks/__init__.py`
    - `dexscrew/algo/student/*`
    - 全部语法通过
- 脚本 shell 语法：
  - `bash -n scripts/xhand_lightbulb_teacher.sh ... scripts/dexh13_lightbulb_student_flow_matching.sh`
  - outcome:
    - 所有新增 `lightbulb` 脚本通过 `bash -n`
- 资产路径存在性：
  - `python - <<'PY' ... os.path.exists(...) ... PY`
  - outcome:
    - `assets/screw/lightbulb/0000_lightbulb.urdf = True`
    - `assets/dexh13_hand/urdf/dexh13_hand_right.urdf = True`
    - `assets/xhand_left/urdf/xhand_left.urdf = True`
- Docker Isaac Gym env reset smoke：
  - `./docker-run-isaacgym.sh timeout 240 bash -lc 'python - <<\"PY\" ... task_name=\"XHandHoraLightbulb\" ... env.reset() ... PY'`
  - outcome:
    - `XHandHoraLightbulb obs (1, 96) acts (12,)`
  - `./docker-run-isaacgym.sh timeout 240 bash -lc 'python - <<\"PY\" ... task_name=\"Dexh13HoraLightbulb\" ... env.reset() ... PY'`
  - outcome:
    - `Dexh13HoraLightbulb obs (1, 96) acts (16,)`
- Core student ctor smoke, XHand route：
  - `./docker-run-isaacgym.sh timeout 240 bash -lc 'python - <<\"PY\" ... TASK=\"XHandHoraLightbulb\" ... ProprioAdapt/PureBC/DiffusionLatentStudent/ConsistencyLatentStudent/FlowMatchingLatentStudent ... PY'`
  - outcome:
    - 5 个 core student ctor 全通过
    - `XHandHoraLightbulb ProprioAdapt ctor_ok 12`
    - `XHandHoraLightbulb PureBC ctor_ok 12`
    - `XHandHoraLightbulb DiffusionLatentStudent ctor_ok 12`
    - `XHandHoraLightbulb ConsistencyLatentStudent ctor_ok 12`
    - `XHandHoraLightbulb FlowMatchingLatentStudent ctor_ok 12`
- Core student ctor smoke, DexH13 route：
  - `./docker-run-isaacgym.sh timeout 240 bash -lc 'python - <<\"PY\" ... TASK=\"Dexh13HoraLightbulb\" ... ProprioAdapt/PureBC/DiffusionLatentStudent/ConsistencyLatentStudent/FlowMatchingLatentStudent ... PY'`
  - outcome:
    - 5 个 core student ctor 全通过
    - `Dexh13HoraLightbulb ProprioAdapt ctor_ok 16 pdim 32`
    - `Dexh13HoraLightbulb PureBC ctor_ok 16 pdim 32`
    - `Dexh13HoraLightbulb DiffusionLatentStudent ctor_ok 16 pdim 32`
    - `Dexh13HoraLightbulb ConsistencyLatentStudent ctor_ok 16 pdim 32`
    - `Dexh13HoraLightbulb FlowMatchingLatentStudent ctor_ok 16 pdim 32`
- Old-path regression：
  - `./docker-run-isaacgym.sh timeout 240 bash -lc 'python - <<\"PY\" ... TASK=\"XHandHoraScrewDriver\" ... ProprioAdapt ... env.reset() ... PY'`
  - outcome:
    - `XHandHoraScrewDriver obs (1, 96) acts (12,) pdim 24`
    - 说明 `train.py/student_eval.py` 的 student 入口重组没有破坏当前 Hora 主线 baseline

### Local decision
- `PLANS_v9` 的工程接入部分已经达到“可继续 smoke”的状态：
  - 新 task 名、asset path、task registry、core student import surface、脚本入口都已接通
  - 两条新 task 都能在 Isaac Gym Docker 中实例化并 reset
  - 两条新 task 上 5 个 core student 类都能成功构造
  - 旧 `XHandHoraScrewDriver + ProprioAdapt` spot-check 仍正常
- 当前还不能说实验矩阵已完成；
  - 现在只是完成了 “engineering wiring accept / runtime env reset accept / core student ctor accept”
  - 还没进入 `teacher 2~5min smoke -> 1min teacher ckpt -> student smoke -> ranking compare`

### Remaining blocked/risky
- `teacher smoke` 还没跑：
  - 尚未确认当前 `lightbulb` init pose 是否足够稳定进入可训区间
  - 尤其 `Dexh13HoraLightbulb` 的 `handRootPos / handRootRPY / handInitPose` 仍可能需要 viewer 下微调
- `assets/lightbulb/0000_lightbulb.urdf` 原 draft 仍引用缺失 STL：
  - 本次 runtime 已切到 `assets/screw/lightbulb/0000_lightbulb.urdf`
  - 如果后续要追求更高保真视觉/接触 mesh，需要补完整 mesh 资产再替换
- `student smoke` 还没跑：
  - 当前只验证到 ctor，不代表 `restore_train/train/save/restore_test/test` 全链已过

### Single recommended next step
- 严格按 `PLANS_v9` 的 smoke 顺序往下走，不要直接开全算法矩阵：
  1. `scripts/vis_xhand_lightbulb_teacher.sh` 和 `scripts/vis_dexh13_lightbulb_teacher.sh` 做 1-env viewer sanity
  2. 对两条 task 各跑一次 `2~5 min PPO teacher smoke`
  3. 若有大面积 reset / 无接触 / reward 死平，只调：
     - `env.object.init_pos`
     - `env.object.init_pos_noise`
     - `env.asset.handRootPos`
     - `env.asset.handRootQuat` / `handRootRPY`
     - `env.asset.handInitPose`
     - `reset_dist_threshold`
     - 少量 reward scale
  4. 拿到可训 init pose 后，再跑 `1 min teacher ckpt` 进入 5 个 core student 的完整 smoke

## v2-105 (2026-04-17) — `PLANS_v9` smoke execution: XHand pass, DexH13 blocked on pose stability

### Targeted milestone/subgoal
- 推进 `PLANS_v9` 从“engineering wiring accept”进入真实 smoke 阶段：
  - `1-env viewer sanity`
  - `2~5 min teacher smoke`
  - `1 route student smoke`

### What changed
- runtime bug fix:
  - `dexscrew/tasks/xhand_hora.py`
  - 在 `compute_observations()` 里为
    - `self.nut_dof_vel_cf[at_reset_env_ids] = self.nut_dof_vel[at_reset_env_ids]`
    - 增加空 `at_reset_env_ids` guard
  - 影响：
    - 修复 `XHandHoraLightbulb` / `Dexh13HoraLightbulb` 在 `1-env` PPO rollout 中的 shape mismatch
    - 该错误此前会在 viewer sanity / small-env teacher smoke 的第一轮 rollout 内直接中断
- student script fix:
  - `scripts/xhand_lightbulb_student_*.sh`
  - `scripts/dexh13_lightbulb_student_*.sh`
  - 为所有 `task.env.numEnvs=48` 的新 lightbulb student 脚本补充：
    - `train.ppo.minibatch_size=576`
  - 影响：
    - 修复新 student 脚本的 PPO batch divisibility 问题
    - 避免进入 student smoke 时因 `48 * 12 = 576` 与默认 `16384` 不整除而直接报错
- docs / plan state:
  - `PLANS_v9.md`
  - 更新为 `active_partial_smoke`

### What was verified
- viewer sanity rerun after bug fix:
  - `./docker-run-isaacgym.sh timeout 45 scripts/vis_xhand_lightbulb_teacher.sh 0 42 v9_xhand_viewer_sanity_rerun`
  - `./docker-run-isaacgym.sh timeout 45 scripts/vis_dexh13_lightbulb_teacher.sh 0 42 v9_dexh13_viewer_sanity_rerun`
  - outcome:
    - 两条新 task 都不再触发 `nut_dof_vel_cf` shape mismatch
    - 两条线都能进入 PPO rollout
- `XHandHoraLightbulb` teacher smoke:
  - `./docker-run-isaacgym.sh timeout 180 scripts/xhand_lightbulb_teacher.sh 0 42 v9_xhand_teacher_smoke_seed42_3min True task.env.numEnvs=32 train.ppo.minibatch_size=384`
  - outputs:
    - `outputs/XHandHoraLightbulb_teacher/v9_xhand_teacher_smoke_seed42_3min/`
    - ckpt: `stage1_nn/best_reward_0.00.pth`
  - outcome:
    - reward 从约 `-655` 稳定改善到约 `-46.5`
    - 无大面积 crash / 早停 / reset-storm 证据
    - 当前可视为 `XHandHoraLightbulb` 的可训 teacher 起点
- `Dexh13HoraLightbulb` teacher smoke, default config:
  - `./docker-run-isaacgym.sh timeout 180 scripts/dexh13_lightbulb_teacher.sh 0 42 v9_dexh13_teacher_smoke_seed42_3min True task.env.numEnvs=32 train.ppo.minibatch_size=384`
  - outputs:
    - `outputs/Dexh13HoraLightbulb_teacher/v9_dexh13_teacher_smoke_seed42_3min/`
    - ckpt: `stage1_nn/best_reward_0.00.pth`
  - outcome:
    - reward 从约 `-5604` 劣化到约 `-8531`
    - 默认 pose 不满足 smoke accept
- reset geometry probes:
  - `XHandHoraLightbulb` default:
    - fingertip->nut `min=0.0635`, `mean=0.0859`
  - `Dexh13HoraLightbulb` default:
    - fingertip->nut `min=0.1040`, `mean=0.1151`
  - `Dexh13HoraLightbulb` rootpos A:
    - override `task.env.asset.handRootPos=[0.14,0.102,0.137]`
    - fingertip->nut `min=0.0786`, `mean=0.0874`
  - `Dexh13HoraLightbulb` rootpos B:
    - override `task.env.asset.handRootPos=[0.14,0.092,0.127]`
    - fingertip->nut `min=0.0717`, `mean=0.0795`
  - interpretation:
    - `Dexh13` 默认 pose 初始接触区明显比 `XHand` 更远、更散
    - 但“仅把距离压近”不能保证 teacher 稳定
- `Dexh13HoraLightbulb` short root-pos probes:
  - candidate A:
    - `./docker-run-isaacgym.sh timeout 90 scripts/dexh13_lightbulb_teacher.sh 0 42 v9_dexh13_teacher_probe_rootpos_a_90s True task.env.numEnvs=32 train.ppo.minibatch_size=384 task.env.asset.handRootPos=[0.14,0.102,0.137]`
  - candidate B:
    - `./docker-run-isaacgym.sh timeout 90 scripts/dexh13_lightbulb_teacher.sh 0 42 v9_dexh13_teacher_probe_rootpos_b_90s True task.env.numEnvs=32 train.ppo.minibatch_size=384 task.env.asset.handRootPos=[0.14,0.092,0.127]`
  - outcome:
    - A 早期 reward 仍持续劣化到 `-12k` 量级
    - B 虽改善初始几何距离，但 90s probe 也劣化到 `-12k` 量级
    - 说明 `Dexh13` 当前问题不只是“整体离得远”，还涉及更细的姿态/接触稳定性
- `XHandHoraLightbulb + ProprioAdapt` student smoke:
  - train:
    - `./docker-run-isaacgym.sh timeout 60 scripts/xhand_lightbulb_student_padapt.sh 0 42 v9_xhand_padapt_smoke checkpoint=outputs/XHandHoraLightbulb_teacher/v9_xhand_teacher_smoke_seed42_3min/stage1_nn/best_reward_*.pth`
    - output:
      - `outputs/XHandHoraLightbulb_student_padapt/v9_xhand_padapt_smoke/stage2_nn/model_best.ckpt`
    - outcome:
      - teacher ckpt restore + student train loop + save 全部通过
  - eval:
    - `./docker-run-isaacgym.sh timeout 120 bash -lc 'python train.py task=XHandHoraLightbulb headless=True test=True seed=42 train.algo=ProprioAdapt train.ppo.proprio_adapt=True train.ppo.output_name=XHandHoraLightbulb_student_padapt_eval/v9_xhand_padapt_smoke task.env.numEnvs=1 ... checkpoint=outputs/XHandHoraLightbulb_student_padapt/v9_xhand_padapt_smoke/stage2_nn/model_best.ckpt +test_num_steps=64'`
    - output:
      - `outputs/XHandHoraLightbulb_student_padapt_eval/v9_xhand_padapt_smoke/eval_64.log`
    - outcome:
      - `EvalSummary steps=64 avg_reward=-5.448445 avg_done_rate=0.015625`
      - 说明 `restore_test -> test` 也已通过

### Local decision
- `PLANS_v9` 现已从纯 wiring 阶段推进到“partial smoke pass”：
  - `XHandHoraLightbulb`:
    - teacher smoke = pass
    - `ProprioAdapt` student smoke train/test = pass
  - `Dexh13HoraLightbulb`:
    - default teacher smoke = fail
    - 仅靠 root translation 的两个 probe 也 fail
- 因此当前最合理的执行策略是分流：
  - 继续把 `XHandHoraLightbulb` 路线扩到剩余 4 个 core student smoke
  - 暂停 `Dexh13HoraLightbulb` student 矩阵，先解决 teacher 可训起点

### Remaining blocked/risky
- `Dexh13HoraLightbulb` 仍未达到 teacher smoke accept：
  - 当前证据显示不是简单的“距离太远”单因子问题
  - 下一步需要更明确的：
    - `handRootRPY / handRootQuat`
    - `handInitPose`
    - `object.init_pos`
    - `reset_dist_threshold`
    - 少量 reward / termination
    - 的联动微调
- `XHandHoraLightbulb` 目前只完成了 `ProprioAdapt` 的完整 smoke：
  - `PureBC`
  - `DiffusionLatentStudent`
  - `ConsistencyLatentStudent`
  - `FlowMatchingLatentStudent`
  - 仍待补齐

### Single recommended next step
- 先不再盲扫 `Dexh13` 的平移 root pose。
- 优先继续 `XHandHoraLightbulb` 上剩余 4 个 core student smoke，保持当前 teacher ckpt 不变。
- `Dexh13HoraLightbulb` 单独开下一轮局部调试时，优先顺序改为：
  1. `env.asset.handRootRPY` / `handRootQuat`
  2. `env.asset.handInitPose`
  3. `env.object.init_pos`
  4. `reset_dist_threshold`
  5. 少量 reward / termination

## v2-106 (2026-04-17) — `PLANS_v9` smoke acceptance closure: both lightbulb task families pass teacher/student smoke

### Target milestone/subgoal
- 将 `PLANS_v9` 从 `active_partial_smoke` 推进到完整 smoke 验收：
  - `XHandHoraLightbulb`
  - `Dexh13HoraLightbulb`
  - 两条线上 5 个 core student 都要完成 `train + official eval`

### What changed
- `configs/task/Dexh13HoraLightbulb.yaml`
  - 为 `Dexh13 + lightbulb` 固化 smoke 级 reward 微调：
    - `pose_diff_penalty_scale: -0.01`
    - `torque_penalty_scale: -0.5`
    - `work_penalty_scale: -0.001`
    - `rotate_penalty_scale: -0.2`
  - 影响：
    - 将 `Dexh13HoraLightbulb` teacher smoke 从默认的 `-5.6k -> -8.5k` 崩盘区拉回到可训 smoke 区间
- `dexscrew/algo/models/models.py`
  - stage2 `TemporalConv` 输入维度不再写死为 `24`
  - 改为使用 `proprio_dim`
- `dexscrew/algo/ppo/padapt.py`
  - 向 `ActorCritic` 显式传递 `proprio_dim`
  - 影响：
    - 修复 `Dexh13` student smoke 的 `mat1 and mat2 shapes cannot be multiplied (...32 and 24...)`
    - 保持 `XHand` 路线语义不变（仍为 `24-dim`)
- docs:
  - `PLANS_v9.md`
  - `docs/plansv9_final_verdict.md`
  - `docs/stage_acceptance_summary.md`

### What was verified
- `Dexh13HoraLightbulb` 零动作诊断：
  - default `lightbulb_inclined`：
    - `thumb_dist mean=0.1235`
    - `index_dist mean=0.1037`
    - `step_all_reward mean16=-74.674170`
    - 主导负项来自 `work_done` 与 `torques`
  - `screwdriver_inclined` 更差，不采用
- `Dexh13HoraLightbulb` recovered teacher smoke:
  - `./docker-run-isaacgym.sh timeout 180 scripts/dexh13_lightbulb_teacher.sh 0 42 v9_dexh13_teacher_smoke_rewardrelax_seed42_3min True task.env.numEnvs=32 train.ppo.minibatch_size=384`
  - output:
    - `outputs/Dexh13HoraLightbulb_teacher/v9_dexh13_teacher_smoke_rewardrelax_seed42_3min/stage1_nn/best_reward_0.00.pth`
  - outcome:
    - around `2.2 min` 时 reward 稳定在 `-705 ~ -769`
    - 作为 smoke teacher 起点 accepted
- `Dexh13HoraLightbulb` student smoke matrix:
  - `ProprioAdapt`
    - train output:
      - `outputs/Dexh13HoraLightbulb_student_padapt/v9_dexh13_padapt_smoke/stage2_nn/model_best.ckpt`
    - eval:
      - `EvalSummary steps=64 avg_reward=-9.051320 avg_done_rate=0.015625`
  - `PureBC`
    - train output:
      - `outputs/Dexh13HoraLightbulb_student_purebc/v9_dexh13_purebc_smoke/stage2_bc_nn/model_best.ckpt`
    - eval:
      - `EvalSummary steps=64 avg_reward=-9.049654 avg_done_rate=0.015625`
  - `DiffusionLatentStudent`
    - train output:
      - `outputs/Dexh13HoraLightbulb_student_diffusion_latent/v9_dexh13_diffusion_smoke/stage2_diffusion_nn/model_best.ckpt`
    - eval:
      - `EvalSummary steps=64 avg_reward=-9.163227 avg_done_rate=0.015625`
      - `EvalReconSummary steps=64 mode=diffusion latent_mse=0.111387 latent_l1=0.279704 action_mse_to_teacher=0.001462`
  - `ConsistencyLatentStudent`
    - train output:
      - `outputs/Dexh13HoraLightbulb_student_consistency/v9_dexh13_consistency_smoke/stage2_consistency_nn/model_best.ckpt`
    - eval:
      - `EvalSummary steps=64 avg_reward=-9.187637 avg_done_rate=0.015625`
      - `EvalReconSummary steps=64 mode=consistency latent_mse=0.102859 latent_l1=0.261397 action_mse_to_teacher=0.001217`
  - `FlowMatchingLatentStudent`
    - train output:
      - `outputs/Dexh13HoraLightbulb_student_flow_matching/v9_dexh13_flow_smoke/stage2_flow_nn/model_best.ckpt`
    - eval:
      - `EvalSummary steps=64 avg_reward=-9.202663 avg_done_rate=0.015625`
      - `EvalReconSummary steps=64 mode=flow_matching latent_mse=0.113891 latent_l1=0.277006 action_mse_to_teacher=0.001581`
- `XHandHoraLightbulb` remaining 4 student smokes also补齐:
  - `PureBC`: `EvalSummary steps=64 avg_reward=-5.379851 avg_done_rate=0.015625`
  - `DiffusionLatentStudent`: `EvalSummary steps=64 avg_reward=-5.226565 avg_done_rate=0.015625`
  - `ConsistencyLatentStudent`: `EvalSummary steps=64 avg_reward=-5.652328 avg_done_rate=0.015625`
  - `FlowMatchingLatentStudent`: `EvalSummary steps=64 avg_reward=-5.256454 avg_done_rate=0.015625`
- common-model regression spot-check after the `proprio_dim` fix:
  - `./docker-run-isaacgym.sh timeout 90 bash -lc 'python train.py task=XHandHoraLightbulb ... checkpoint=outputs/XHandHoraLightbulb_student_padapt/v9_xhand_padapt_smoke/stage2_nn/model_best.ckpt +test_num_steps=16'`
  - outcome:
    - `EvalSummary steps=16 avg_reward=-7.725555 avg_done_rate=0.000000`
    - confirms old `XHand` 24-dim path still runs

### Local decision
- `PLANS_v9` 的 smoke acceptance 已完成：
  - `teacher_smoke_accept = yes`
  - `student_train_test_smoke_accept = yes`
  - `ranking_comparison_accept = smoke_protocol_yes`
- 本轮结论应写成：
  - `completed_smoke_accept`
  - 而不是 longer-run ranking accept

### Remaining blocked/risky
- 当前比较仍是 smoke 协议，不是 `nominal / light / hard` 长训练结论
- `Dexh13HoraLightbulb` 的 reward 微调是 smoke 级修复，后续 longer-run 若继续要单独评估是否仍合适
- 旧任务没有做完整 retrain regression；目前只做了最小运行回归

### Single recommended next step
- 不要再在 `PLANS_v9` 内继续追加 smoke。
- 如果要继续，应新开一个 longer-run comparison 计划，目标固定为：
  1. `XHandHoraLightbulb` 与 `Dexh13HoraLightbulb` teacher longer-run
  2. 5 个 core student 的 longer-run / official evaluation matrix
  3. 与 screw 任务上的算法排序变化做正式对照

---

## v2-107 (2026-04-20) — Original XHand Padapt Wandb Smoke Probe

### Target milestone/subgoal
- 非计划内诊断：验证“原版 `XHandHoraScrewDriver + ProprioAdapt` student 训练是否能正常把 TensorBoard 指标同步到 wandb”。

### What changed (files + behavior impact)
- `docs/session_handoff_v2.md`
  - 追加本次 wandb 诊断记录。
- 代码与配置未改动。

### What was verified (commands + key outcomes)
- 在 Isaac Gym Docker 内直接运行一个带 wandb 凭据挂载的 180s smoke：
  - `docker run --rm ... -v "$HOME/.netrc:/tmp/.netrc:ro" -v "$HOME/.config/wandb:/tmp/.config/wandb:ro" ... timeout 180 python train.py task=XHandHoraScrewDriver ... train.algo=ProprioAdapt ... train.ppo.output_name=XHandHoraScrewDriver_student_padapt/wandb_probe_xhand_seed42_180s experiment=padapt_cmp_v1 wandb_activate=True ... checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_1550.49.pth`
- Wandb init / sync 关键日志：
  - `wandb.login()] Loaded credentials for https://api.wandb.ai from /tmp/.netrc`
  - `wandb: Syncing run wandb_probe_xhand_seed42_180s_2026-04-20_12-03-41`
  - project URL:
    - `https://wandb.ai/3319963854-south-china-university-of-technology/dexscrew`
  - run URL:
    - `https://wandb.ai/3319963854-south-china-university-of-technology/dexscrew/runs/fud6pz25`
- Local wandb metadata / TB callback:
  - local run dir:
    - `wandb/run-20260420_120342-fud6pz25/`
  - metadata confirms args include:
    - `train.algo=ProprioAdapt`
    - `experiment=padapt_cmp_v1`
    - `wandb_activate=True`
  - `wandb/run-20260420_120342-fud6pz25/logs/debug.log` contains:
    - `tensorboard callback: outputs/XHandHoraScrewDriver_student_padapt/wandb_probe_xhand_seed42_180s/stage2_tb, True`
- Smoke artifacts created:
  - `outputs/XHandHoraScrewDriver_student_padapt/wandb_probe_xhand_seed42_180s/stage2_tb/events.out.tfevents.1776686625.wbz-ubuntu22-pc`
  - `outputs/XHandHoraScrewDriver_student_padapt/wandb_probe_xhand_seed42_180s/stage2_nn/model_best.ckpt`
- Training signal during the 180s probe:
  - stdout `Current Best` improved from `0.00` to about `1385.81` before timeout.

### Local decision
- 原版 `XHandHoraScrewDriver + ProprioAdapt` student 训练本身可以正常连接 wandb。
- 这次 probe 也明确证明：
  - wandb 端收到了 student run；
  - wandb backend 已接上对应 `stage2_tb` 事件目录；
  - 因此“wandb 完全收不到 student 指标”这一假设不成立。
- 当前更可能的问题是：
  - wandb `Charts` 页只自动展开了少量面板；
  - 或远端查看的不是这次 run / 不是同一代码版本。

### Remaining blocked/risky
- 本次只验证了“wandb 连通 + TensorBoard callback 接通”，没有在本地完整枚举远端 UI 上所有 tag。
- 仓库当前本地 Python 环境缺少 `tensorboard` 包，因此没有进一步直接解析 event 文件做 tag 清单。
- 如果后续仍只在 wandb `Charts` 里看到少量 loss，优先检查：
  - 是否看的就是 run `fud6pz25`
  - 是否切到了 wandb 顶部 `Tensorboard` 标签
  - 是否存在自定义 workspace / panel 过滤

### Single recommended next step
- 直接在 wandb 打开 run `fud6pz25`，优先检查顶部 `Tensorboard` 页而不是 `Charts` 默认页；若仍缺 `env/...` 面板，再做一次远端 tag 枚举而不是继续怀疑 student 没上报。

---

## v2-108 (2026-04-21) — Mesh lightbulb cutover + Hora/Pasini teacher initPose tuning

### Target milestone/subgoal
- 将 `lightbulb` 主资产从 primitive runtime 近似版切到 STL mesh 主线。
- 新建 `XHandPasiniLightbulb`，并完成：
  - `XHandHoraLightbulb` baseline + 5 次 5min teacher initPose tuning + final confirm
  - `XHandPasiniLightbulb` baseline + 5 次 5min teacher initPose tuning + final confirm
- 对 `Dexh13HoraLightbulb` 做 mesh cutover 回归，但不进入 5 轮 pose 主调参。

### What changed (files + behavior impact)
- `assets/screw/lightbulb/0000_lightbulb.urdf`
  - 从 primitive runtime 近似版切到 mesh-STL runtime canonical 版。
  - 现在直接引用：
    - `../../lightbulb/lightbulb_head.stl`
    - `../../lightbulb/lightbulb_socket.stl`
    - `../../lightbulb/contact0.stl`
    - `../../lightbulb/contact1.stl`
- `assets/lightbulb/0000_lightbulb.urdf`
  - 原始 draft URDF 同步改为本地可用相对路径：
    - `lightbulb_head.stl`
    - `lightbulb_socket.stl`
    - `contact0.stl`
    - `contact1.stl`
  - 解决了此前 `object_sim/lightbulb/*.stl` 缺失导致的 Viewer/runtime 不一致。
- `dexscrew/tasks/__init__.py`
  - 新增 `XHandPasiniLightbulb -> XHandPasini` 映射。
- `dexscrew/tasks/xhand_pasini.py`
  - 接受 `env.initPose=lightbulb_inclined`，初始 joint seed alias `bulb_inclined`。
  - 支持从 YAML 读取 `env.object.init_pos / init_pos_noise`。
  - `dump_current_pose()` 提示改为当前 task YAML 通用文案。
- 新增：
  - `configs/task/XHandPasiniLightbulb.yaml`
  - `configs/train/XHandPasiniLightbulb.yaml`
  - `scripts/pasini_lightbulb_teacher.sh`
  - `scripts/vis_pasini_lightbulb_teacher.sh`
- 最终固化的 tuned defaults：
  - `configs/task/XHandHoraLightbulb.yaml`
    - `env.asset.handRootPos: [0.0, 0.004, 0.206]`
  - `configs/task/XHandPasiniLightbulb.yaml`
    - `env.object.init_pos: [0.009, 0.058, 0.0]`
    - `env.object.init_pos_noise: [0.002, 0.002, 0.0]`

### What was verified (commands + key outcomes)
- Static / wiring:
  - `rg -n "object_sim/lightbulb" assets/lightbulb/0000_lightbulb.urdf assets/screw/lightbulb/0000_lightbulb.urdf`
    - outcome: no remaining stale mesh path references.
  - `PYTHONDONTWRITEBYTECODE=1 python - <<'PY' ... compile('dexscrew/tasks/xhand_pasini.py', ...) ... compile('train.py', ...) ... PY`
    - outcome: `syntax_ok`.
  - `bash -n scripts/pasini_lightbulb_teacher.sh scripts/vis_pasini_lightbulb_teacher.sh ...`
    - outcome: pass.
- Runtime mesh sanity:
  - `XHandHoraLightbulb` runtime sanity:
    - `./docker-run-isaacgym.sh timeout 90 bash -lc 'python train.py task=XHandHoraLightbulb ... task.env.numEnvs=16 train.ppo.minibatch_size=192'`
    - outcome: 1.2 min 内 `mean_rewards` 约 `-251.85 -> -53.54`，mesh runtime train path 正常。
  - `Dexh13HoraLightbulb` mesh regression:
    - `./docker-run-isaacgym.sh timeout 60 bash -lc 'python train.py task=Dexh13HoraLightbulb ... task.env.numEnvs=16 train.ppo.minibatch_size=192'`
    - outcome: mesh 切换后链路未炸；但 0.7 min 区间约 `-776.63 -> -998.12`，仍不适合本轮直接推进 teacher 5 轮 pose 调参。
  - `XHandPasiniLightbulb` runtime sanity:
    - `./docker-run-isaacgym.sh timeout 60 bash scripts/pasini_lightbulb_teacher.sh ... task.env.numEnvs=16 train.ppo.minibatch_size=192`
    - outcome: 新 task 成功实例化并训练，0.7 min 区间约 `-224.20 -> -597.83`，确认链路通。
  - 1-env viewer launch sanity:
    - `scripts/vis_xhand_lightbulb_teacher.sh`
    - `scripts/vis_dexh13_lightbulb_teacher.sh`
    - `scripts/vis_pasini_lightbulb_teacher.sh`
    - outcome: headless=False 路径均能进入环境创建/rollout；本次为非交互会话，只验证无 missing mesh / bad asset 硬错误。
- `XHandHoraLightbulb` 5min tuning:
  - `r0_baseline`:
    - `outputs/XHandHoraLightbulb_teacher/mesh_r0_baseline_5min/`
    - last `mean_rewards = -24.163656`
  - `r1_object_align`:
    - `outputs/XHandHoraLightbulb_teacher/mesh_r1_object_align_5min/`
    - last `mean_rewards = -36.816711`
  - `r2_root_translation`:
    - `outputs/XHandHoraLightbulb_teacher/mesh_r2_root_translation_5min/`
    - override: `handRootPos=[0.0,0.004,0.206]`
    - last `mean_rewards = -12.503010`
    - current best
  - `r3_root_orientation`:
    - `outputs/XHandHoraLightbulb_teacher/mesh_r3_root_orientation_5min/`
    - last `mean_rewards = -285.108433`
  - `r4_finger_pose`:
    - `outputs/XHandHoraLightbulb_teacher/mesh_r4_finger_pose_5min/`
    - last `mean_rewards = -234.202718`
  - `r5_threshold_reward`:
    - `outputs/XHandHoraLightbulb_teacher/mesh_r5_threshold_reward_5min/`
    - last `mean_rewards = -14.965856`
  - final confirm:
    - `outputs/XHandHoraLightbulb_teacher/mesh_hora_final_confirm_5min/`
    - last `mean_rewards = -12.503010`
    - outcome: `r2` 可复现。
  - 60s smoke teacher ckpt:
    - `outputs/XHandHoraLightbulb_teacher/mesh_hora_teacher_smoke_ckpt_60s/`
    - 1.2 min tail `mean_rewards ≈ -61.53`
- `XHandPasiniLightbulb` tuning:
  - `r0_baseline`:
    - `outputs/XHandPasiniLightbulb_teacher/mesh_r0_baseline_5min/`
    - last `mean_rewards = -192.035345`
  - `r1_object_align`:
    - `outputs/XHandPasiniLightbulb_teacher/mesh_r1_object_align_5min/`
    - override: `init_pos=[0.009,0.058,0.0]`, `init_pos_noise=[0.002,0.002,0.0]`
    - last `mean_rewards = -69.932446`
    - current best
  - `r2_viewer_dump_pose` fact capture:
    - `./docker-run-isaacgym.sh timeout 180 bash -lc 'python - <<\"PY\" ... env.reset(); env.dump_current_pose(...) ... PY'`
    - artifact:
      - `outputs/pose_dumps/pasini_pose_env0_1776763821720.json`
    - dump gives nonzero `hand_dof_pos` and a reusable `env.customInitDofPos` template.
  - `r3_joint_refine`:
    - `outputs/XHandPasiniLightbulb_teacher/mesh_r3_joint_refine_5min/`
    - last `mean_rewards = -127.851759`
  - `r4_object_threshold`:
    - `outputs/XHandPasiniLightbulb_teacher/mesh_r4_object_threshold_5min/`
    - last `mean_rewards = -108.341273`
  - `r5_reward_touchup`:
    - `outputs/XHandPasiniLightbulb_teacher/mesh_r5_reward_touchup_5min/`
    - last `mean_rewards = -93.696232`
  - final confirm:
    - `outputs/XHandPasiniLightbulb_teacher/mesh_pasini_final_confirm_5min/`
    - last `mean_rewards = -73.043506`
    - outcome: `r1` best 区间可复现，但比首轮 best 略差，说明稳定性一般。
  - 60s smoke teacher ckpt:
    - `outputs/XHandPasiniLightbulb_teacher/mesh_pasini_teacher_smoke_ckpt_60s/`
    - 1.2 min tail `mean_rewards ≈ -271.08`

### Local decision
- mesh lightbulb cutover 已完成，runtime 主线已不再依赖 primitive `assets/screw/lightbulb` 近似几何。
- `2026-04-21` 之前的 `PLANS_v9` lightbulb smoke 结果应视为：
  - `primitive_lightbulb_runtime`
- 本次之后的新结果应视为：
  - `mesh_stl_lightbulb_runtime`
- 二者不应直接混为同一套 teacher/student 结论。
- 当前 teacher-side mesh tuned defaults:
  - `XHandHoraLightbulb`: keep `handRootPos=[0.0,0.004,0.206]`
  - `XHandPasiniLightbulb`: keep `init_pos=[0.009,0.058,0.0]`, `init_pos_noise=[0.002,0.002,0.0]`
- `Dexh13HoraLightbulb`:
  - mesh cutover regression pass
  - but still not chosen for this round’s 5-run pose tuning mainline

### Remaining blocked/risky
- 本次 viewer sanity 是非交互式启动检查；真正的窗口下接触姿态仍建议在你本机再看一遍。
- `XHandPasiniLightbulb` 的 best config 可复现到同一量级，但相较首轮 best 仍有一定 run-to-run 波动。
- `mesh_pasini_teacher_smoke_ckpt_60s` 的短预算表现较弱，只适合作为链路 smoke 输入，不代表 longer-run teacher 质量。
- 尚未在 mesh lightbulb teacher best 上展开新的 student smoke / ranking matrix。

### Single recommended next step
- 用本次固化后的 mesh tuned config，在本机 viewer 下各检查一次：
  - `scripts/vis_xhand_lightbulb_teacher.sh`
  - `scripts/vis_pasini_lightbulb_teacher.sh`
- 若视觉接触正常，则直接进入下一轮计划：
  - `XHandHoraLightbulb` 与 `XHandPasiniLightbulb` 的 mesh teacher-student smoke / longer-run ranking matrix。

---

## v2-Local (2026-04-22) — Project-Scoped Codex API Config Isolation

### Target milestone/subgoal
- Local tooling setup only: isolate Codex authentication/config for this repo without changing the training/eval/export pipeline.

### What changed (files + behavior impact)
- `.config/codex/config.toml`
  - Added project-local Codex config using `model_provider = "gac"`, `model = "gpt-5.1-codex-max"`, `wire_api = "responses"`, and file-based local credential storage.
  - Added `shell_environment_policy` with `inherit = "core"` and a project-local provider token environment variable.
  - Added `env_key = "ANTHROPIC_AUTH_TOKEN"` under `[model_providers.gac]` so Codex CLI knows which process environment variable to use for provider auth.
  - Behavior impact: launching Codex with `CODEX_HOME="$PWD/.config/codex"` uses the isolated project config instead of global `~/.codex`.
- `start_codex_local.sh`
  - Added a project-root launcher that:
    - sets `CODEX_HOME="$PWD/.config/codex"`,
    - reads the provider env var name and token from `.config/codex/config.toml`,
    - exports that env var for the current process,
    - starts `codex` with any forwarded CLI args.
  - Behavior impact: avoids accidental fallback to global `~/.codex` for normal project launches.
- `.gitignore`
  - Added `.envrc` to keep optional direnv-based local auto-export files untracked.
- `.gitignore`
  - Added `.config/codex/` so the local config/auth cache is not committed.

### What was verified (commands + key outcomes)
- TOML structure check:
  - `python3 -c 'import pathlib,tomllib; ... tomllib.loads(...) ...'`
  - Outcome: `TOML OK: isolated Codex config parsed successfully`.
- Ignore and permission check:
  - `git check-ignore -v .config/codex/config.toml`
  - Outcome: ignored by `.gitignore`.
  - `stat -c '%a %n' .config .config/codex .config/codex/config.toml`
  - Outcome: directories `700`, config file `600`.
- API connectivity checks:
  - Direct `curl` to `https://gaccode.com/codex/v1/responses` with project token and `model=gpt-5.4`
  - Outcome: streaming response completed with `OK`.
  - Direct `curl` with `model=gpt-5.4-mini`
  - Outcome: streaming response completed with `OK`.
  - `CODEX_HOME="$PWD/.config/codex"` plus exported `ANTHROPIC_AUTH_TOKEN`, `codex exec -m gpt-5.4 'Reply exactly: OK'`
  - Outcome: Codex CLI completed and returned `OK`.
  - `./start_codex_local.sh exec --skip-git-repo-check --dangerously-bypass-approvals-and-sandbox -m gpt-5.4 'Reply exactly: OK'`
  - Outcome: launcher path completed and returned `OK`.
  - `codex exec` using config default `model=gpt-5.1-codex-max`
  - Outcome: did not complete; direct HTTP returned provider error that this model is not supported for the current account route.

### Remaining blocked/risky
- The config contains local credentials by design; keep `.config/codex/` ignored and do not paste or commit the file contents.
- `shell_environment_policy.set` does not provide environment variables to Codex CLI itself; launch Codex with `ANTHROPIC_AUTH_TOKEN` already exported, or use a local wrapper that exports it before invoking `codex`.
- Current default `model = "gpt-5.1-codex-max"` is not usable on this provider/account route based on the API check; `gpt-5.4` and `gpt-5.4-mini` are reachable.

### Single recommended next step
- Use `./start_codex_local.sh` as the default way to launch Codex in this repo; if fully automatic per-directory activation is desired, add a local `.envrc` and load it with `direnv`.

---

## v2-110 (2026-04-22) — Bootstrap Context Map For Upcoming Environment Test Experiments

### Target milestone/subgoal
- Recover the current executable state before environment-side test adjustments.
- Build a minimal project map for upcoming environment experiments without changing the existing teacher-student pipeline.

### What changed (files + behavior impact)
- `docs/session_handoff_v2.md`
  - Added this bootstrap/context entry.
  - Behavior impact: no runtime/code change; clarifies the current experiment surface for the next environment-adjustment session.

### What was verified (commands + key outcomes)
- Required bootstrap docs:
  - `sed -n '1,220p' docs/session_handoff_v2.md`
  - `sed -n '1,220p' docs/stage_acceptance_summary.md`
  - `sed -n '1,240p' PLANS_v2.md`
  - Outcome:
    - historical `PLANS_v2` diffusion route is already locally closed as baseline-first;
    - current executable environment work has moved to `PLANS_v9` / mesh-lightbulb transfer validation.
- Current active task/config surface:
  - `sed -n '1,260p' train.py`
  - `sed -n '1,260p' configs/config.yaml`
  - `find configs -maxdepth 2 -type f | sort`
  - Outcome:
    - single Hydra entrypoint remains `train.py`;
    - env selection is `task=<...>` + `configs/task/*.yaml`;
    - algo selection is `train.algo=<...>` + `configs/train/*.yaml`.
- Environment wiring spot checks:
  - `sed -n '1,220p' dexscrew/tasks/__init__.py`
  - `rg -n "initPose|forceScale|randomForceProbScalar|obs_noise|lightbulb|sampleProb|object_tilt|dump_current_pose|evalObjectType|reward|privInfo|randomization" dexscrew/tasks/xhand_hora.py dexscrew/tasks/xhand_pasini.py dexscrew/tasks/dexh13_hora.py`
  - Outcome:
    - primary edit points for env experiments are task YAMLs first;
    - shared runtime semantics live mainly in `dexscrew/tasks/xhand_hora.py`, with hand-family overrides in `xhand_pasini.py` and `dexh13_hora.py`.
- Lightbulb plan/state recovery:
  - `sed -n '6440,6695p' docs/session_handoff_v2.md`
  - `sed -n '518,610p' docs/stage_acceptance_summary.md`
  - `sed -n '1,260p' PLANS_v9.md`
  - Outcome:
    - mesh runtime is now the active boundary for lightbulb experiments;
    - tuned teacher defaults currently frozen as:
      - `XHandHoraLightbulb`: `handRootPos=[0.0,0.004,0.206]`
      - `XHandPasiniLightbulb`: `init_pos=[0.009,0.058,0.0]`, `init_pos_noise=[0.002,0.002,0.0]`
    - `Dexh13HoraLightbulb` is smoke-stable but not the current main tuning path.
- Algo pipeline spot checks:
  - `sed -n '1,220p' dexscrew/algo/ppo/padapt.py`
  - `sed -n '1,240p' dexscrew/algo/ppo/diffusion_latent_student.py`
  - `sed -n '1,200p' dexscrew/algo/models/models.py`
  - `rg -n "class |def test|def train|EvalSummary|EvalReconSummary|restore_test|restore_train|sample_latent|sample_action_chunk|bc_loss|latent_loss|diffusion_loss|consistency|flow" dexscrew/algo/ppo/*.py`
  - Outcome:
    - teacher/student still share the same actor backbone;
    - `padapt` remains the stable baseline;
    - diffusion/consistency/flow remain packaged and runnable, but are not the current execution mainline.

### Remaining blocked/risky
- Documentation layers span `PLANS_v2` through `PLANS_v9`; without checking dates, it is easy to confuse the closed screwdriver diffusion stage with the current lightbulb environment-validation stage.
- Some narrative docs (for example algorithm summary text) still reflect earlier diffusion-facing conclusions and should not override the newer handoff/acceptance records.
- The next environment experiments should preserve the current mesh-lightbulb tuned defaults unless there is a clear hypothesis; repeating old primitive-runtime or already-rejected settings would create avoidable drift.

### Single recommended next step
- Open the next execution step as a bounded lightbulb environment experiment, starting from the frozen mesh defaults:
  - prefer `XHandHoraLightbulb` first,
  - use `configs/task/*.yaml` overrides for init pose / object init / reward / randomization changes,
  - only touch task Python files if the experiment needs new semantics rather than new values.

---

## v2-111 (2026-04-22) — XHand Lightbulb PPO Peak-Time Probe With Wandb + Early Stop

### Target milestone/subgoal
- Run a real headless `XHandHoraLightbulb` PPO teacher training with:
  - wandb enabled for live metric observation,
  - reward-based early stop,
  - explicit measurement of when the current config reaches a practical peak.

### What changed (files + behavior impact)
- `train.py`
  - Added a robust `import_wandb_package()` helper so a local repo `wandb/` run directory does not shadow the real wandb package.
  - `wandb_group` / `wandb_entity` are now actually forwarded into `wandb.init(...)` when set.
  - Behavior impact: wandb logging can be used reliably from the repo root.
- `dexscrew/algo/ppo/ppo.py`
  - Added optional PPO early-stop controls:
    - `train.ppo.early_stop_patience`
    - `train.ppo.early_stop_min_improvement`
    - `train.ppo.early_stop_min_agent_steps`
  - Added best-reward tracking fields:
    - best epoch
    - best agent steps
    - best wallclock minutes
    - epochs since improvement
  - Added `EarlyStopSummary ...` terminal summary on stop.
  - Fixed a measurement bug: best-reward / early-stop logic now ignores the invalid pre-episode `mean_rewards=0` bootstrap phase and only starts after real completed-episode stats exist.
  - Behavior impact: PPO teacher runs can now stop on a plateau and report a meaningful peak time.
- `docker-run-isaacgym.sh`
  - Auto-mounts host wandb auth files into the container when present:
    - `~/.netrc -> /tmp/.netrc`
    - `~/.config/wandb -> /tmp/.config/wandb`
  - Behavior impact: containerized Isaac Gym training can reuse the user’s existing wandb login without manual re-auth.

### What was verified (commands + key outcomes)
- Syntax / wiring:
  - `python - <<'PY' ... compile('train.py', ...) ... compile('dexscrew/algo/ppo/ppo.py', ...) ... PY`
  - `bash -n docker-run-isaacgym.sh`
  - Outcome: pass.
- Container wandb auth reuse:
  - `./docker-run-isaacgym.sh bash -lc 'python - <<\"PY\" ... print((Path.home()/\".netrc\").exists()) ... import wandb ... PY'`
  - `./docker-run-isaacgym.sh bash -lc 'python - <<\"PY\" ... wandb.login(relogin=False) ... PY'`
  - Outcome:
    - container sees `/tmp/.netrc`;
    - wandb login succeeds from mounted host credentials.
- First probe exposed and confirmed the invalid-bootstrap-best bug:
  - run: `XHandHoraLightbulb_teacher/peakscan_wandb_earlystop`
  - Outcome:
    - trainer initially treated bootstrap `mean_rewards=0` as best before any completed episode;
    - early stop was therefore anchored to an invalid `0.00` reference;
    - code patched immediately after confirmation.
- Corrected peak-time probe:
  - Command:
    - `./docker-run-isaacgym.sh bash scripts/xhand_lightbulb_teacher.sh 0 42 peakscan_wandb_earlystop_v2 task.env.numEnvs=32 train.ppo.minibatch_size=384 wandb_activate=True wandb_group=lightbulb_teacher +train.ppo.early_stop_patience=150 +train.ppo.early_stop_min_improvement=0.10 +train.ppo.early_stop_min_agent_steps=150000 train.ppo.max_agent_steps=1000000`
  - Outcome:
    - run completed by trainer-side early stop;
    - `EarlyStopSummary reason=no_improvement patience=150 min_improvement=0.100000 best_reward=-10.806959 best_epoch=389 best_agent_steps=149760 best_elapsed_min=4.10`
    - wandb run:
      - `https://wandb.ai/3319963854-south-china-university-of-technology/dexscrew/runs/9grs5meo`
    - local run root:
      - `outputs/XHandHoraLightbulb_teacher/peakscan_wandb_earlystop_v2/`

### Remaining blocked/risky
- The current probe is single-seed (`seed=42`) and uses a bounded plateau heuristic, so it is suitable for “peak arrival time” diagnosis, not for final ranking claims.
- The early-stop conclusion is tied to this exact setup:
  - `XHandHoraLightbulb`
  - mesh runtime
  - `32 env`
  - `minibatch_size=384`
  - current reward/randomization defaults
- If the user changes randomization, reward scales, or teacher init geometry, the measured peak time can shift materially.

### Single recommended next step
- Keep this run as the reference headless PPO teacher peak probe for the current `XHandHoraLightbulb(mesh)` config, then choose one follow-up axis:
  - rerun multiseed with the same early-stop settings to test timing stability,
  - or open a new bounded env-yaml probe (for example object init / reward / randomization) and compare how the peak time and best reward move relative to `peakscan_wandb_earlystop_v2`.

---

## v2-Local (2026-04-22) - Current Codex Auth Mode Verification

### Target milestone/subgoal
- Local tooling check only: verify whether the current repo-scoped Codex session is using API-key auth or account login.

### What changed (files + behavior impact)
- `docs/session_handoff_v2.md`
  - Added this auth-mode verification entry.
  - Behavior impact: no runtime/code change; records the currently active repo-scoped Codex auth path for future sessions.

### What was verified (commands + key outcomes)
- Required bootstrap docs:
  - `sed -n '1,220p' docs/session_handoff_v2.md`
  - `sed -n '1,220p' docs/stage_acceptance_summary.md`
  - Outcome: bootstrap requirements satisfied before local inspection.
- Current process environment:
  - `env`
  - Outcome: current Codex process environment contains `CODEX_HOME=/home/wbz-ubuntu22-pc/Codefield/py/dexscrew-repro/.config/codex` and an exported provider token environment variable `ANTHROPIC_AUTH_TOKEN`.
- Repo-scoped Codex config:
  - `rg -n -S "^model_provider|^model |=^wire_api|env_key|shell_environment_policy|oauth|login|token|auth" .config/codex/config.toml`
  - Outcome:
    - `model_provider = "gac"`
    - `preferred_auth_method = "apikey"`
    - `cli_auth_credentials_store = "file"`
    - provider `env_key = "ANTHROPIC_AUTH_TOKEN"`
    - repo-local config also currently contains the token value under `shell_environment_policy.set`.
- Repo launcher behavior:
  - `rg -n -S "CODEX_HOME|config.toml|env_key|ANTHROPIC_AUTH_TOKEN|export|codex" start_codex_local.sh`
  - Outcome: launcher reads the repo-local config, exports the provider token env var, then executes `codex`.
- Active session metadata:
  - `rg -n -S "model_provider|source|cli_version|gac|openai|oauth|login" .config/codex/sessions/2026/04/22/rollout-2026-04-22T13-44-15-019db3b7-b7b5-7d02-a0df-fc5eb47e3a43.jsonl`
  - Outcome: current session metadata reports `source="cli"` and `model_provider="gac"`, consistent with repo-scoped token/provider auth rather than interactive account login.

### Remaining blocked/risky
- The current repo-scoped setup is not account-login based; it is explicitly configured for API-key/token auth.
- A live provider token is stored in the repo-local ignored config and exported into the process environment; if this secret scope is broader than intended, it should be rotated and moved to a safer injection path.
- Because the auth path is provider `gac` plus `ANTHROPIC_AUTH_TOKEN`, users should not assume this session is using first-party OpenAI account login semantics.

### Single recommended next step
- Keep using `./start_codex_local.sh` if repo-scoped token auth is intended; otherwise switch the repo back to account login or move the token out of `.config/codex/config.toml` and rotate the current secret.

---

## v2-Local (2026-04-22) - Repo-Local Codex Secret Removal

### Target milestone/subgoal
- Local tooling hardening only: remove the repo-local plain-text provider token and require external environment-variable injection.

### What changed (files + behavior impact)
- `.config/codex/config.toml`
  - Removed `[shell_environment_policy.set]` token injection.
  - Behavior impact: repo-local Codex config no longer stores the provider secret in plain text.
- `start_codex_local.sh`
  - Simplified launcher behavior so it reads only the provider env var name from config.
  - Added an explicit guard that exits if the required env var is not already exported.
  - Behavior impact: launcher no longer loads the secret from disk; users must inject `ANTHROPIC_AUTH_TOKEN` from the outside environment before launch.
- `docs/session_handoff_v2.md`
  - Added this hardening entry.
  - Behavior impact: no runtime change beyond documenting the safer launch contract.

### What was verified (commands + key outcomes)
- Static checks:
  - `bash -n start_codex_local.sh`
  - Outcome: pass.
- Config inspection:
  - `sed -n '1,80p' .config/codex/config.toml`
  - Outcome: config still declares `env_key = "ANTHROPIC_AUTH_TOKEN"` but no longer contains a plain-text token value.
- Launcher inspection:
  - `sed -n '1,80p' start_codex_local.sh`
  - Outcome: launcher reads env var name only and checks `${!TOKEN_ENV_NAME:-}` instead of loading a token from config.
- Runtime behavior:
  - `./start_codex_local.sh --help`
  - Outcome: pass when `ANTHROPIC_AUTH_TOKEN` is present in the current shell environment.
  - `env -u ANTHROPIC_AUTH_TOKEN ./start_codex_local.sh --help`
  - Outcome: fast failure with `Missing required environment variable: ANTHROPIC_AUTH_TOKEN`.

### Remaining blocked/risky
- `.config/codex/` being ignored prevents normal Git tracking, but ignore alone is not a secret-management control; any local process with workspace read access can still read files under that path.
- The provider token had already existed in plain text during earlier local sessions, so rotation is still recommended if the credential scope matters.
- The current shell environment still carries `ANTHROPIC_AUTH_TOKEN` for active sessions; this is safer than disk-storing it in repo config, but it is still a live credential in process environment space.

### Single recommended next step
- Rotate the current provider token if you want a clean post-hardening state, then inject the new token only via shell environment (for example a local untracked `.envrc` or manual `export ANTHROPIC_AUTH_TOKEN=...`) before using `./start_codex_local.sh`.

---

## v2-Local (2026-04-22) - Project-Local Token Persistence Via .envrc

### Target milestone/subgoal
- Local tooling convenience only: make repo-scoped Codex token injection persistent without putting the secret back into tracked config.

### What changed (files + behavior impact)
- `.envrc`
  - Created a repo-root local env file exporting `ANTHROPIC_AUTH_TOKEN`.
  - Behavior impact: the project now has a persistent local secret file that is already ignored by Git.
- `start_codex_local.sh`
  - Added loading of repo-root `.envrc` before validating the provider env var.
  - Behavior impact: `./start_codex_local.sh` works in new shells even when `ANTHROPIC_AUTH_TOKEN` is not manually exported, as long as `.envrc` exists.
- `docs/session_handoff_v2.md`
  - Added this persistence entry.
  - Behavior impact: no runtime change beyond documenting the launch contract.

### What was verified (commands + key outcomes)
- Preconditions:
  - `printf '%s\n' "${ANTHROPIC_AUTH_TOKEN:+set}"`
  - Outcome: current shell still had a live token available for one-time persistence.
  - `command -v direnv`
  - Outcome: not installed, so persistence was implemented via launcher-loaded `.envrc` rather than shell hook automation.
- File creation and permissions:
  - `.envrc` created from the current shell token with restrictive permissions.
  - `stat -c '%a %n' .envrc`
  - Outcome: `600 .envrc`.
- Ignore and content-shape checks:
  - `git check-ignore -v .envrc`
  - Outcome: ignored by `.gitignore`.
  - `sed -n '1p' .envrc | sed 's/=.*/=<redacted>/'`
  - Outcome: file shape is `export ANTHROPIC_AUTH_TOKEN=<redacted>`.
- Launcher behavior:
  - `bash -n start_codex_local.sh`
  - Outcome: pass.
  - `env -u ANTHROPIC_AUTH_TOKEN ./start_codex_local.sh --help`
  - Outcome: pass; launcher successfully sourced `.envrc` and started Codex without a manual shell export.

### Remaining blocked/risky
- `.envrc` is still a plain-text local secret file; Git ignore prevents normal commits but does not protect against local file disclosure, backups, or manual sharing.
- This persistence is repo-scoped, not system-wide; other projects or shells outside this repo will not automatically inherit the token.
- Because the token previously lived in repo-local config and now also exists in `.envrc`, rotation is still the cleanest follow-up if you want to fully retire the earlier exposure path.

### Single recommended next step
- Keep using `./start_codex_local.sh` as the standard entrypoint for this repo; if you later want shell-wide persistence instead of repo-scoped persistence, move the export to `~/.bashrc` or `~/.profile` and delete `.envrc`.

---

## v2-112 (2026-04-22) — `dotpg_env.md` vs live `Dexh13HoraLightbulb` wiring audit

### Target milestone/subgoal
- Clarify whether `docs/dotpg_env.md` still matches the runnable `Dexh13HoraLightbulb` environment before creating a DOTPG-specific task variant.

### What changed (files + behavior impact)
- `docs/session_handoff_v2.md`
  - Added this audit entry.
  - Behavior impact: no runtime change; records which `dotpg_env.md` claims are live versus stale.

### What was verified (commands + key outcomes)
- Doc/config/code inspection:
  - `sed -n '1,260p' docs/dotpg_env.md`
  - `sed -n '1,260p' configs/task/Dexh13HoraLightbulb.yaml`
  - `sed -n '1,260p' dexscrew/tasks/dexh13_hora.py`
  - `sed -n '1,360p' dexscrew/tasks/xhand_hora.py`
  - `sed -n '1,220p' configs/train/Dexh13HoraLightbulb.yaml`
  - `sed -n '1,220p' scripts/dexh13_lightbulb_teacher.sh`
  - `sed -n '1,220p' configs/config.yaml`
  - Outcome: live runnable path is still `task=Dexh13HoraLightbulb` -> `task.name=Dexh13HoraLightbulb` -> `isaacgym_task_map["Dexh13HoraLightbulb"] = Dexh13Hora`.
- Key reconciliation outcome:
  - Implemented/live in YAML + code: `apply_action_mask=False`, DexH13 16-DOF hand asset/fingertips/init pose, `reset_dist_threshold`, `proximity_reward`, `object_tilt=False`, object/lightbulb asset switch, current reward scales, current randomization flags, Hydra-resolved GPU sim flags.
  - Stale in doc: `termination` block and script overrides, `normalize_penalties_by_num_actions`, `two_finger_gate`, old fingertip/body names, old hand root pose/URDF path, old reward scales, old `reset_dist_threshold=0.20`, old scale-randomization range, claims that mass/COM/friction randomization remain enabled, and claim that GPU flags are hard-coded.

### Remaining blocked/risky
- `docs/dotpg_env.md` currently reads like a historical design note, not executable config; using it directly to author a new env would reintroduce stale termination/reward assumptions.
- A new task YAML with a new Hydra name will not auto-run unless either:
  - a matching train config is added under `configs/train/`, or
  - the launch command explicitly overrides `train=Dexh13HoraLightbulb`.
- Existing DexH13 scripts hard-code `task=Dexh13HoraLightbulb`, so they will not pick up a new task YAML unless the script or CLI invocation is changed.

### Single recommended next step
- For the fastest runnable probe, clone `configs/task/Dexh13HoraLightbulb.yaml`, keep `name: Dexh13HoraLightbulb`, and launch it directly with `python train.py task=<new_task_yaml> train=Dexh13HoraLightbulb ...`; only add a new `configs/train/*.yaml` or dedicated script if the variant will become a repeated experiment path.

---

## v2-113 (2026-04-22) — Runnable `DotpgEnv` DexH13 lightbulb variant + smoke probe

### Target milestone/subgoal
- Turn `docs/dotpg_env.md` into a directly runnable DexH13 lightbulb experiment path and measure whether the documented env/reward settings materially change PPO teacher behavior.

### What changed (files + behavior impact)
- `configs/task/Dexh13HoraLightbulbDotpgEnv.yaml`
  - Added a runnable DOTPG-env-inspired task variant.
  - Mapped only the `dotpg_env.md` fields that current task code actually supports:
    - `reset_dist_threshold=0.20`
    - harsher reward scales (`2.5 / -1.5 / -30 / -0.15 / -0.3 / -1.0 / 2.0`)
    - broader object randomization (`mass/COM/friction/scale` enabled with scale list `[1.0, 1.05, 1.10, 1.15]`)
  - Intentionally kept the current stable DexH13 mesh geometry (`handAsset`, fingertip bodies, root pose, init pose) so this variant isolates env/reward changes instead of mixing in stale geometry from the doc.
  - Added inline comments that `termination.*`, `normalize_penalties_by_num_actions`, and `two_finger_gate` remain doc-only and are not implemented in current code.
- `configs/train/Dexh13HoraLightbulbDotpgEnv.yaml`
  - Added matching train config so `task=Dexh13HoraLightbulbDotpgEnv` works without extra Hydra overrides.
- `scripts/dexh13_lightbulb_dotpg_teacher.sh`
  - Added a dedicated headless teacher launcher for the DOTPG-env variant.
- `scripts/vis_dexh13_lightbulb_dotpg_teacher.sh`
  - Added a dedicated viewer launcher for init-pose / contact inspection under the DOTPG-env variant.

### What was verified (commands + key outcomes)
- Static checks:
  - `bash -n scripts/dexh13_lightbulb_dotpg_teacher.sh`
  - `bash -n scripts/vis_dexh13_lightbulb_dotpg_teacher.sh`
  - Outcome: pass.
- Runnable smoke PPO teacher probe:
  - `./docker-run-isaacgym.sh timeout 300 bash scripts/dexh13_lightbulb_dotpg_teacher.sh 0 42 dotpg_env_smoke True task.env.numEnvs=16 train.ppo.minibatch_size=192 train.ppo.max_agent_steps=3072 wandb_activate=False`
  - Outcome:
    - Hydra resolved `task=Dexh13HoraLightbulbDotpgEnv` correctly.
    - Isaac Gym env instantiated successfully with 4 object scales (`1.0/1.05/1.10/1.15`).
    - PPO finished the bounded smoke run and saved:
      - `outputs/Dexh13HoraLightbulbDotpg_teacher/dotpg_env_smoke/stage1_nn/best_reward_-42803.18.pth`
      - `outputs/Dexh13HoraLightbulbDotpg_teacher/dotpg_env_smoke/stage1_tb/events.out.tfevents.1776849623.wbz-ubuntu22-pc`
    - Best observed smoke reward was about `-42803.18`, far below the current reward-relaxed DexH13 lightbulb smoke regime (`~ -705 ~ -769`), indicating this DOTPG-env-inspired setting materially increases training difficulty.

### Remaining blocked/risky
- This new variant is only a runnable subset of `docs/dotpg_env.md`; the document’s `termination` switches, `normalize_penalties_by_num_actions`, and `two_finger_gate` still do not exist in live task code.
- The very negative smoke reward suggests the doc-style reward/randomization package is not a drop-in replacement for the current stable DexH13 mesh teacher config.
- Because the geometry was intentionally kept at current mesh-stable values, this probe isolates env/reward/randomization impact; it is not a full reproduction of every stale geometry value in the doc.

### Single recommended next step
- Use the new viewer script first to inspect contact quality under the harsher DOTPG-env reward/randomization package, then run one bounded headless comparison with wandb enabled against the current `Dexh13HoraLightbulb` teacher to decide whether this variant is worth longer-run ranking.

---

## v2-114 (2026-04-22) — Full `dotpg_env.md` migration into live `Dexh13HoraLightbulb`

### Target milestone/subgoal
- Replace the previous partial DOTPG-env subset with a full migration of the documented DexH13 lightbulb env semantics into the active `Dexh13HoraLightbulb` task path, so later teacher/student probes measure the real env impact rather than a reward-only approximation.

### What changed (files + behavior impact)
- `dexscrew/tasks/xhand_hora.py`
  - Added live support for `env.termination.*`:
    - `grace_steps`
    - `enable_finger_dist`
    - `enable_nut_stagnation`
    - `enable_no_contact`
    - `enable_screw_limit`
    - `log`
  - Added live support for `env.normalize_penalties_by_num_actions`.
  - Added live support for `env.two_finger_gate.*`:
    - target selection (`nut_pos` / `object_pos`)
    - scale-aware offset / near / far
    - optional fingertip contact-force weighting
    - positive-velocity-only gating
    - extra no-grasp penalty term
  - Added per-env object scale tracking so `two_finger_gate.scale_with_object=True` works with randomized object scale.
  - Behavior impact:
    - the previously doc-only DexH13 reward/termination semantics now actually execute inside the shared Hora task code.
- `configs/task/Dexh13HoraLightbulb.yaml`
  - Replaced the earlier reward-relaxed mesh-smoke config with the full DOTPG-env-style config:
    - `initPose=screwdriver_inclined`
    - `reset_dist_threshold=0.20`
    - `normalize_penalties_by_num_actions=True`
    - `forceScale=2.0`, `randomForceProbScalar=0.25`
    - debug-friendly default termination block
    - full `two_finger_gate` block
    - heavier reward scales (`2.5 / -1.5 / -30 / -0.15 / -0.3 / -1.0 / 2.0`)
    - randomization restored (`mass/COM/friction/scale=True`, scale list `[1.0, 1.05, 1.10, 1.15]`)
    - DexH13 geometry switched to the documented `right_sim` asset + `*_tip` fingertip bodies + documented root pose
  - Behavior impact:
    - the active `task=Dexh13HoraLightbulb` path now corresponds to the DOTPG-env document instead of the older smoke-recovery config.
- `scripts/dexh13_lightbulb_teacher.sh`
  - Added strict teacher termination overrides:
    - `grace_steps=150`
    - all four non-max termination checks enabled
- `scripts/dexh13_lightbulb_student_padapt.sh`
- `scripts/dexh13_lightbulb_student_purebc.sh`
- `scripts/dexh13_lightbulb_student_diffusion_latent.sh`
- `scripts/dexh13_lightbulb_student_consistency.sh`
- `scripts/dexh13_lightbulb_student_flow_matching.sh`
  - Added strict student termination overrides:
    - `grace_steps=0`
    - all four non-max termination checks enabled
  - Behavior impact:
    - the DexH13 teacher/student script family now matches the termination semantics described in `dotpg_env.md`.
- Removed temporary partial-subset artifacts created during the earlier audit/probe:
  - `configs/task/Dexh13HoraLightbulbDotpgEnv.yaml`
  - `configs/train/Dexh13HoraLightbulbDotpgEnv.yaml`
  - `scripts/dexh13_lightbulb_dotpg_teacher.sh`
  - `scripts/vis_dexh13_lightbulb_dotpg_teacher.sh`
  - Behavior impact:
    - there is no longer a confusing second DexH13 lightbulb DOTPG path; the active task itself now carries the migrated config.

### What was verified (commands + key outcomes)
- Static checks:
  - `python - <<'PY' ... compile('dexscrew/tasks/xhand_hora.py', 'exec') ... PY`
  - `bash -n scripts/dexh13_lightbulb_teacher.sh`
  - `for f in scripts/dexh13_lightbulb_student_*.sh; do bash -n "$f"; done`
  - Outcome: pass.
- Bounded headless teacher smoke on the active migrated task:
  - `./docker-run-isaacgym.sh timeout 300 bash scripts/dexh13_lightbulb_teacher.sh 0 42 dotpg_migrated_smoke True task.env.numEnvs=8 train.ppo.minibatch_size=96 train.ppo.max_agent_steps=768 wandb_activate=False`
  - Outcome:
    - Hydra resolved the migrated active task with:
      - `normalize_penalties_by_num_actions=True`
      - strict teacher termination overrides (`grace_steps=150`, all checks enabled)
      - `two_finger_gate.enable=True`
      - `right_sim` hand asset + `right_*_tip` fingertip bodies
    - Isaac Gym env instantiated successfully.
    - PPO entered rollout/training without runtime errors.
    - This bounded smoke ended before any full episodes completed, so `mean_rewards` stayed `nan`; however the env/reward/reset loop executed and the migrated task path is runtime-valid.

### Remaining blocked/risky
- The active `Dexh13HoraLightbulb` task is now much harsher than the previous reward-relaxed smoke config, so old DexH13 smoke expectations (`~ -705 ~ -769`) no longer apply.
- The short smoke only proves runtime validity; it does not yet characterize the new reward scale or peak time.
- Because the active task YAML changed substantially, any later DexH13 comparison against older runs must be labeled as pre-migration vs post-migration.

### Single recommended next step
- Run one bounded 5-minute headless PPO teacher comparison on the migrated active task with wandb enabled, then compare its reward/time curve against the old reward-relaxed DexH13 teacher baseline to quantify how much the full DOTPG env changes optimization difficulty.

---

## v2-115 (2026-04-24) — Codebase Orientation With Subagents

### Target milestone/subgoal
- Bootstrap a new execution session for future algorithm development and training continuation.
- Target was read-only orientation, not a new experiment or code implementation.

### What changed (files + behavior impact)
- `docs/session_handoff_v2.md`
  - Added this orientation entry.
  - Behavior impact: no runtime behavior changed; records the current codebase map and latest actionable next step for future sessions.

### What was verified (commands + key outcomes)
- Required bootstrap reads:
  - `sed -n '1,240p' docs/session_handoff_v2.md`
  - `sed -n '1,240p' docs/stage_acceptance_summary.md`
  - `tail -n 220 docs/session_handoff_v2.md`
  - Outcome: latest real execution next step is v2-114, not older local tooling entries: run one bounded 5-minute headless PPO teacher comparison on migrated `Dexh13HoraLightbulb`.
- Codebase orientation:
  - `sed -n '1,260p' train.py`
  - `sed -n '1,260p' configs/config.yaml`
  - `sed -n '1,220p' student_eval.py`
  - `sed -n '1,220p' dexscrew/tasks/__init__.py`
  - `sed -n '1,280p' configs/train/XHandHoraScrewDriver.yaml`
  - `sed -n '1,260p' configs/task/XHandHoraScrewDriver.yaml`
  - `sed -n '1,240p' configs/task/Dexh13HoraLightbulb.yaml`
  - `sed -n '1,220p' configs/train/Dexh13HoraLightbulb.yaml`
  - Outcome: confirmed Hydra entrypoint, task map, default Hora screwdriver path, DexH13 lightbulb path, and student export limitation.
- Algorithm/environment survey with subagents:
  - Subagent B mapped train/eval/export entrypoints, Docker wrapper constraints, task names, and `train.algo` names.
  - Subagent C mapped PPO teacher, `padapt`, `purebc`, latent/action diffusion, consistency, flow matching, `XHandHora`, `Dexh13Hora`, `XHandPasini`, and asset/URDF adaptation points.
  - Subagent A initially failed due model capacity and was respawned on a smaller model; its route summary was useful but identified an older next-step entry, so the local `tail` check above is authoritative.
- Worktree state:
  - `git status --short`
  - Outcome: repository already contains many modified/untracked files across code, configs, docs, scripts, and assets; treat them as existing user/session work and do not revert.
- Handoff edit check:
  - `git diff --check -- docs/session_handoff_v2.md`
  - Outcome: pass.

### Remaining blocked/risky
- The worktree is dirty; future code edits should carefully scope diffs and avoid overwriting existing uncommitted changes.
- `PLANS_v2.md` remains the governance anchor, but later docs such as `docs/diffusion_algorithm.md` and later plan verdict files summarize additional branch conclusions; future agents should reconcile these before changing algorithm direction.
- Diffusion/consistency/flow variants exist in code, but current accepted path is still baseline-first unless a future task explicitly reopens generative student development.

### Single recommended next step
- Preserve the v2-114 experimental next step: run one bounded 5-minute headless PPO teacher comparison on the migrated active `Dexh13HoraLightbulb` task with wandb enabled, then compare its reward/time curve against the old reward-relaxed DexH13 teacher baseline.

---

## v2-116 (2026-04-24) — DexH13 Lightbulb Middle-Finger Gate Probe Config

### Target milestone/subgoal
- Create a small, reversible probe to test whether the DexH13 middle finger can learn to participate in lightbulb contact before changing geometry or adding a full three-finger gate.

### What changed (files + behavior impact)
- `configs/task/Dexh13HoraLightbulbMiddleProbe.yaml`
  - Added a task variant cloned from `Dexh13HoraLightbulb`.
  - Keeps runtime `task.name: Dexh13HoraLightbulb` so it uses the existing `Dexh13Hora` class.
  - Changes only the diagnostic gate intent:
    - `eval_cache_name: middle_probe`
    - `env.two_finger_gate.other_fingertip_indices: [1]`
  - Behavior impact: the gate now rewards/penalizes thumb + middle-finger participation instead of thumb + best-of(index, middle).
- `configs/train/Dexh13HoraLightbulbMiddleProbe.yaml`
  - Added train config cloned from `Dexh13HoraLightbulb` so `task=Dexh13HoraLightbulbMiddleProbe` resolves through Hydra without extra `train=` overrides.
- `scripts/dexh13_lightbulb_teacher_middle_probe.sh`
  - Added teacher launcher for the middle-finger gate probe.
- `scripts/vis_dexh13_lightbulb_teacher_middle_probe.sh`
  - Added viewer launcher for middle-probe PPO checkpoints.

### What was verified (commands + key outcomes)
- Static script checks:
  - `bash -n scripts/dexh13_lightbulb_teacher_middle_probe.sh`
  - `bash -n scripts/vis_dexh13_lightbulb_teacher_middle_probe.sh`
  - Outcome: pass.
- Config diff check:
  - `diff -u configs/task/Dexh13HoraLightbulb.yaml configs/task/Dexh13HoraLightbulbMiddleProbe.yaml | sed -n '1,80p'`
  - Outcome: confirmed the task variant differs only in `eval_cache_name` and `two_finger_gate.other_fingertip_indices`.
- Train config equivalence check:
  - `diff -u configs/train/Dexh13HoraLightbulb.yaml configs/train/Dexh13HoraLightbulbMiddleProbe.yaml || true`
  - Outcome: no diff; train hyperparameters match the active DexH13 lightbulb baseline.
- Patch hygiene:
  - `git diff --check -- configs/task/Dexh13HoraLightbulbMiddleProbe.yaml configs/train/Dexh13HoraLightbulbMiddleProbe.yaml scripts/dexh13_lightbulb_teacher_middle_probe.sh scripts/vis_dexh13_lightbulb_teacher_middle_probe.sh docs/session_handoff_v2.md`
  - Outcome: pass.

### Remaining blocked/risky
- No IsaacGym training or smoke run was executed in this session; the next run should validate Hydra/runtime behavior.
- This is not a final three-finger grasp reward. It only tests whether thumb + middle can become a viable contact pair.
- If the middle finger still does not approach/contact the lightbulb, the next likely bottleneck is hand/object geometry or initial pose, not just the gate.

### Single recommended next step
- Run a bounded PPO teacher probe on `Dexh13HoraLightbulbMiddleProbe`, then visualize the best checkpoint and compare whether the middle finger shows contact tendency relative to the current `dexh13_lightbulb_ppo` baseline.

---

## v2-117 (2026-04-24) — Middle-Probe Visual Diagnosis

### Target milestone/subgoal
- Interpret the unexpectedly strong `Dexh13HoraLightbulbMiddleProbe` reward and user-observed viewer behavior before changing to a full three-finger gate.

### What changed (files + behavior impact)
- `docs/session_handoff_v2.md`
  - Added this diagnostic note.
  - Behavior impact: no runtime behavior changed.

### What was verified (commands + key outcomes)
- Live middle-probe training status:
  - `find outputs/Dexh13HoraLightbulb_teacher_middle_probe/middle_probe_s42/stage1_nn ...`
  - Outcome: best checkpoint advanced from early negative reward to `best_reward_3107.56.pth`.
- TensorBoard event parse:
  - Parsed `outputs/Dexh13HoraLightbulb_teacher_middle_probe/middle_probe_s42/stage1_tb/events.out.tfevents.*` inside the IsaacGym Docker image.
  - Outcome:
    - `episode_rewards/step max ~= 3065.77` in one snapshot, then checkpoint advanced to `3107.56`.
    - `episode_lengths/step` increased to roughly `779`, so the run is no longer dying immediately after `grace_steps=150`.
    - `two_finger/other_contact_w` remains near `1.0`, but viewer inspection reports the visible middle finger still has little/no contact.
- Reward/code inspection:
  - `configs/task/Dexh13HoraLightbulbMiddleProbe.yaml`
  - `dexscrew/tasks/xhand_hora.py`
  - Outcome: middle-probe changed only `two_finger_gate.other_fingertip_indices` to `[1]`; `proximity_reward` and `finger_dist` termination still use thumb + index.
- Lightbulb asset inspection:
  - `assets/screw/lightbulb/0000_lightbulb.urdf`
  - mesh bounding-box script over:
    - `assets/lightbulb/lightbulb_head.stl`
    - `assets/lightbulb/lightbulb_socket.stl`
    - `assets/lightbulb/contact0.stl`
    - `assets/lightbulb/contact1.stl`
  - Outcome:
    - visual uses `lightbulb_head.stl` + `lightbulb_socket.stl`;
    - collision uses simplified `contact0.stl` + `contact1.stl`;
    - head collision and visual are broadly aligned but not identical, and collision meshes are very low-poly approximations.

### Remaining blocked/risky
- The high middle-probe reward does not yet prove true visible middle-finger participation.
- Existing reward/termination still contains thumb+index assumptions, so the policy can retain thumb+index rotation while satisfying or exploiting the modified gate indirectly.
- Visual/collision mismatch can make contact appear offset in the viewer; tactile/contact metrics may refer to the collision mesh, not the visible bulb surface.

### Single recommended next step
- Before implementing a full three-finger gate, add explicit per-finger diagnostics for index/middle/thumb distance and contact force, then run/visualize a short probe to confirm whether the middle finger is actually contacting the collision geometry or whether the current reward is still dominated by thumb+index behavior.

---

## v2-118 (2026-04-24) — Middle-Finger Proximity/Finger-Dist ContactViz Probe

### Target milestone/subgoal
- Make the DexH13 lightbulb middle-finger probe internally consistent before the next short training run:
  - gate uses thumb + middle,
  - proximity reward uses thumb + middle,
  - finger-distance termination uses thumb + middle,
  - viewer shows the same geometry used for collision.

### What changed (files + behavior impact)
- `dexscrew/tasks/xhand_hora.py`
  - Added `env.finger_object_contact` config parsing with backward-compatible defaults:
    - default thumb index: last fingertip,
    - default other fingertip: index finger (`[0]`).
  - Rewired `proximity_reward` distance calculation to use configured thumb + configured other fingertip(s) instead of hardcoded thumb + index.
  - Rewired `finger_dist` termination to use the same configured fingertip set instead of hardcoded thumb + index.
  - Behavior impact: existing tasks without `finger_object_contact` keep old thumb+index behavior; the new probe can explicitly test thumb+middle.
- `assets/screw/contactviz/0000_lightbulb.urdf`
  - Added a debug lightbulb asset where visual geometry is exactly the same STL geometry as collision geometry (`contact0.stl`, `contact1.stl`).
  - Behavior impact: physics collision remains the same low-poly contact mesh as the active lightbulb asset, but viewer contact inspection no longer uses the mismatched high-detail visual head/socket meshes.
- `configs/task/Dexh13HoraLightbulbMiddleContactViz.yaml`
  - Added task variant cloned from the lightbulb middle probe.
  - Sets `env.object.type: screw_contactviz`.
  - Sets `env.finger_object_contact.thumb_fingertip_index: 3`.
  - Sets `env.finger_object_contact.other_fingertip_indices: [1]`.
  - Keeps `env.two_finger_gate.other_fingertip_indices: [1]`.
- `configs/train/Dexh13HoraLightbulbMiddleContactViz.yaml`
  - Added matching train config so `task=Dexh13HoraLightbulbMiddleContactViz` resolves directly.
- `scripts/dexh13_lightbulb_teacher_middle_contactviz.sh`
  - Added teacher launcher for the contact-visualized, thumb+middle-consistent probe.
- `scripts/vis_dexh13_lightbulb_teacher_middle_contactviz.sh`
  - Added viewer launcher for the contactviz probe checkpoints.

### What was verified (commands + key outcomes)
- Code syntax:
  - `PYTHONDONTWRITEBYTECODE=1 python - <<'PY' ... compile('dexscrew/tasks/xhand_hora.py', 'exec') ... PY`
  - Outcome: `syntax_ok`.
- Script syntax:
  - `bash -n scripts/dexh13_lightbulb_teacher_middle_contactviz.sh`
  - `bash -n scripts/vis_dexh13_lightbulb_teacher_middle_contactviz.sh`
  - Outcome: `scripts_ok`.
- URDF XML parse:
  - `python - <<'PY' ... ET.parse('assets/screw/contactviz/0000_lightbulb.urdf') ... PY`
  - Outcome: `xml_ok`.
- Patch hygiene:
  - `git diff --check -- dexscrew/tasks/xhand_hora.py configs/task/Dexh13HoraLightbulbMiddleContactViz.yaml configs/train/Dexh13HoraLightbulbMiddleContactViz.yaml scripts/dexh13_lightbulb_teacher_middle_contactviz.sh scripts/vis_dexh13_lightbulb_teacher_middle_contactviz.sh assets/screw/contactviz/0000_lightbulb.urdf docs/session_handoff_v2.md`
  - Outcome: pass.
- Runtime process check:
  - `docker inspect --format '{{.Name}} {{.Config.Cmd}}' laughing_wiles`
  - Outcome: `laughing_wiles` is the old `dexh13_lightbulb_teacher_middle_probe.sh 0 42 middle_probe_s42` run, not the new contactviz run.

### Remaining blocked/risky
- No IsaacGym smoke/training run has been executed for `Dexh13HoraLightbulbMiddleContactViz` yet.
- Contactviz visual geometry is intentionally low-poly because it mirrors collision meshes; it is for contact debugging, not final presentation.
- If the middle finger still does not participate after this probe, the likely next variable is initial object pose/hand geometry clearance rather than reward index mismatch.

### Single recommended next step
- Stop the old `middle_probe_s42` container if using the same GPU, then run a bounded 30-minute PPO teacher probe on `Dexh13HoraLightbulbMiddleContactViz` and visualize the latest `best_reward_*.pth` checkpoint.

---

## v2-119 (2026-04-24) — Enlarged Lightbulb Position Probe

### Target milestone/subgoal
- Add a small geometry/initial-position ablation after the contactviz thumb+middle probe still showed no middle-finger contact.
- Test whether a larger bulb plus a slight object offset gives the middle finger more usable contact opportunity.

### What changed (files + behavior impact)
- `configs/task/Dexh13HoraLightbulbMiddleScalePos.yaml`
  - Added task variant cloned from `Dexh13HoraLightbulbMiddleContactViz`.
  - Keeps thumb+middle `finger_object_contact` and `two_finger_gate` settings.
  - Keeps `env.object.type: screw_contactviz`, so visual geometry still matches collision geometry.
  - Sets fixed enlarged object scale:
    - `env.baseObjScale: 1.20`
    - `randomizeScale: False`
    - `randomizeScaleList: [1.20]`
  - Sets a fixed small object offset:
    - `env.object.init_pos: [0.010, 0.0, 0.0]`
    - `env.object.init_pos_noise: [0.0, 0.0, 0.0]`
  - Raises hand z from `0.195` to `0.207`, matching the existing `0.06 * (scale - 1.0)` compensation rule for scale `1.20`.
- `configs/train/Dexh13HoraLightbulbMiddleScalePos.yaml`
  - Added matching train config so the new task resolves directly through Hydra.
- `scripts/dexh13_lightbulb_teacher_middle_scalepos.sh`
  - Added teacher launcher for the enlarged/offset probe.
- `scripts/vis_dexh13_lightbulb_teacher_middle_scalepos.sh`
  - Added viewer launcher for the enlarged/offset probe checkpoints.

### What was verified (commands + key outcomes)
- Script syntax:
  - `bash -n scripts/dexh13_lightbulb_teacher_middle_scalepos.sh`
  - `bash -n scripts/vis_dexh13_lightbulb_teacher_middle_scalepos.sh`
  - Outcome: pass.
- Patch hygiene:
  - `git diff --check -- configs/task/Dexh13HoraLightbulbMiddleScalePos.yaml configs/train/Dexh13HoraLightbulbMiddleScalePos.yaml scripts/dexh13_lightbulb_teacher_middle_scalepos.sh scripts/vis_dexh13_lightbulb_teacher_middle_scalepos.sh docs/session_handoff_v2.md`
  - Outcome: pass.
- Runtime process check:
  - `docker ps --format ...`
  - Outcome: only `agitated_shannon` viewer container was running; no active training container was observed in this check.

### Remaining blocked/risky
- No IsaacGym runtime smoke/training run has been executed for `Dexh13HoraLightbulbMiddleScalePos` yet.
- The default offset uses `+x 1cm` as the first thumb-side hypothesis; if visualization shows this moves the bulb the wrong way, override `task.env.object.init_pos` on the command line rather than editing the config.
- Because the object scale is fixed at `1.20`, compare this run against contactviz as a geometry probe, not as a final robustness setting.

### Single recommended next step
- Run a bounded 30-minute PPO teacher probe on `Dexh13HoraLightbulbMiddleScalePos`, visualize the latest best checkpoint, and compare whether the middle finger now approaches or contacts the bulb.

---

## v2-120 (2026-04-24) — MiddleScalePos Hand Z Raise

### Target milestone/subgoal
- Adjust the enlarged lightbulb position probe's initial hand height after visual inspection.

### What changed (files + behavior impact)
- `configs/task/Dexh13HoraLightbulbMiddleScalePos.yaml`
  - Raised `env.asset.handRootPos` z by 1 cm:
    - before: `[0.11, 0.020, 0.207]`
    - after: `[0.11, 0.020, 0.217]`
  - Behavior impact: the DexH13 hand starts 1 cm higher for this scalepos probe only. Bulb scale (`1.20`) and object offset (`[0.010, 0.0, 0.0]`) are unchanged.

### What was verified (commands + key outcomes)
- Config spot check:
  - `rg -n "handRootPos|baseObjScale|init_pos" configs/task/Dexh13HoraLightbulbMiddleScalePos.yaml`
  - Outcome: confirmed `baseObjScale=1.20`, `init_pos=[0.010, 0.0, 0.0]`, and `handRootPos=[0.11, 0.020, 0.217]`.
- Patch hygiene:
  - `git diff --check -- configs/task/Dexh13HoraLightbulbMiddleScalePos.yaml docs/session_handoff_v2.md`
  - Outcome: pass.

### Remaining blocked/risky
- No IsaacGym viewer/training run has been executed after this z-height change.

### Single recommended next step
- Open the scalepos viewer/training visualization again and check whether the raised hand reduces initial crowding while preserving reachable thumb/index/middle contact.

---

## v2-121 (2026-04-24) — MiddleScalePos Bulb Y Centering

### Target milestone/subgoal
- Adjust the enlarged lightbulb probe so the bulb center aligns better with the DexH13 palm/middle-finger centerline after visual inspection showed coordinated four-finger motion.

### What changed (files + behavior impact)
- `configs/task/Dexh13HoraLightbulbMiddleScalePos.yaml`
  - Changed object initial position:
    - before: `init_pos: [0.010, 0.0, 0.0]`
    - after: `init_pos: [0.010, 0.020, 0.0]`
  - Behavior impact: the fixed enlarged bulb probe now shifts the bulb center toward the middle-finger/palm y line while preserving:
    - `baseObjScale: 1.20`
    - `handRootPos: [0.11, 0.020, 0.217]`
    - no object position noise.

### What was verified (commands + key outcomes)
- Config spot check:
  - `rg -n "baseObjScale|init_pos|handRootPos" configs/task/Dexh13HoraLightbulbMiddleScalePos.yaml`
  - Outcome: confirmed `baseObjScale=1.20`, `init_pos=[0.010, 0.020, 0.0]`, and `handRootPos=[0.11, 0.020, 0.217]`.
- Patch hygiene:
  - `git diff --check -- configs/task/Dexh13HoraLightbulbMiddleScalePos.yaml docs/session_handoff_v2.md`
  - Outcome: pass.

### Remaining blocked/risky
- No IsaacGym viewer/training run has been executed after this y-centering change.

### Single recommended next step
- Reopen the scalepos viewer to validate the centered bulb init pose, then run the next bounded 30-minute PPO probe if the initial geometry looks right.

---

## v2-122 (2026-04-24) — MiddleScalePos Soft Three-Finger Gate

### Target milestone/subgoal
- Move the DexH13 lightbulb scalepos probe from middle-finger-only contact probing to a soft three-finger objective after visual confirmation that the middle finger has a motion trend.

### What changed (files + behavior impact)
- `dexscrew/tasks/xhand_hora.py`
  - Added configurable `env.two_finger_gate.other_aggregation` with supported modes:
    - `max` (old default behavior, preserves existing tasks),
    - `mean`,
    - `min`,
    - `mean_min`.
  - Added `other_mean_weight` and `other_min_weight` for `mean_min`.
  - For `mean_min`, the non-thumb gate term becomes:
    - `(other_mean_weight * mean(other_weights) + other_min_weight * min(other_weights)) / weight_sum`.
  - Added TensorBoard extras:
    - `two_finger/other_mean_w`
    - `two_finger/other_min_w`
    - `two_finger/other_mean_dist`
    - `two_finger/other_max_dist`
  - Existing tasks without `other_aggregation` still use `max`, so `[0,1]` remains old "best of index/middle" behavior unless explicitly changed.
- `configs/task/Dexh13HoraLightbulbMiddleScalePos.yaml`
  - Changed `finger_object_contact.other_fingertip_indices` from `[1]` to `[0, 1]`.
    - Behavior impact: proximity reward and finger-distance termination now track thumb + index + middle.
  - Changed `two_finger_gate.other_fingertip_indices` from `[1]` to `[0, 1]`.
  - Added soft three-finger gate aggregation:
    - `other_aggregation: mean_min`
    - `other_mean_weight: 0.5`
    - `other_min_weight: 0.5`
  - Behavior impact: gate now uses thumb multiplied by `0.5 * mean(index,middle) + 0.5 * min(index,middle)` instead of thumb + middle only.

### What was verified (commands + key outcomes)
- Syntax:
  - `PYTHONDONTWRITEBYTECODE=1 python - <<'PY' ... compile('dexscrew/tasks/xhand_hora.py', 'exec') ... PY`
  - Outcome: `syntax_ok`.
- Config spot check:
  - `rg -n "finger_object_contact|other_fingertip_indices|other_aggregation|other_mean_weight|other_min_weight|baseObjScale|init_pos|handRootPos" configs/task/Dexh13HoraLightbulbMiddleScalePos.yaml`
  - Outcome:
    - `finger_object_contact.other_fingertip_indices: [0, 1]`
    - `two_finger_gate.other_fingertip_indices: [0, 1]`
    - `other_aggregation: mean_min`
    - `baseObjScale: 1.20`
    - current user-edited `init_pos: [0.012, 0.005, 0.0]`
    - `handRootPos: [0.11, 0.020, 0.217]`
- Patch hygiene:
  - `git diff --check -- dexscrew/tasks/xhand_hora.py configs/task/Dexh13HoraLightbulbMiddleScalePos.yaml docs/session_handoff_v2.md`
  - Outcome: pass.

### Remaining blocked/risky
- No IsaacGym runtime smoke/training run has been executed after the soft three-finger gate change.
- The log namespace is still `two_finger/...` for compatibility, even though this scalepos probe is now a soft three-finger objective.
- The soft gate is intentionally not a hard three-finger requirement; if index dominates and middle weakens again, increase `other_min_weight` or reduce `min_mult` in a follow-up probe.

### Single recommended next step
- Run a bounded 30-minute PPO teacher probe on `Dexh13HoraLightbulbMiddleScalePos` with a new cache name, then compare `two_finger/other_mean_w` and `two_finger/other_min_w` to verify that both index and middle contribute.

---

## v2-123 (2026-04-25) — MiddleScalePos Ring Action Mask Probe

### Target milestone/subgoal
- Isolate index/middle/thumb coordination by preventing the ring finger policy actions from interfering with middle-finger rotation during the soft three-finger lightbulb probe.

### What changed (files + behavior impact)
- `configs/task/Dexh13HoraLightbulbMiddleScalePos.yaml`
  - Enabled action masking:
    - `apply_action_mask: True`
    - `action_mask_indices: [8, 9, 10, 11]`
  - Behavior impact:
    - DexH13 DOF/action order is index `0:3`, middle `4:7`, ring `8:11`, thumb `12:15`;
    - policy actions for all four right-ring joints are multiplied by zero before simulation.
  - Set all four ring init joints to zero:
    - `right_ring_joint_0: 0.0`
    - `right_ring_joint_1: 0.0`
    - `right_ring_joint_2: 0.0`
    - `right_ring_joint_3: 0.0`

### What was verified (commands + key outcomes)
- Config spot check:
  - `rg -n "apply_action_mask|action_mask_indices|right_ring_joint" configs/task/Dexh13HoraLightbulbMiddleScalePos.yaml`
  - Outcome: confirmed ring action mask `[8, 9, 10, 11]` and all right-ring init joints set to `0.0`.
- Code syntax:
  - `PYTHONDONTWRITEBYTECODE=1 python - <<'PY' ... compile('dexscrew/tasks/xhand_hora.py', 'exec') ... PY`
  - Outcome: `syntax_ok`.
- Patch hygiene:
  - `git diff --check -- configs/task/Dexh13HoraLightbulbMiddleScalePos.yaml docs/session_handoff_v2.md`
  - Outcome: pass.

### Remaining blocked/risky
- No IsaacGym runtime smoke/training run has been executed after the ring action mask change.
- Because this task uses torque control, action masking prevents active policy torque on ring joints, but it does not physically weld the joints. Contact forces may still move the ring passively. If strict immobilization is needed, add a follow-up probe that tightens ring DOF lower/upper limits near zero.

### Single recommended next step
- Reopen the scalepos viewer or run a short headless=False probe to confirm the ring starts outward/neutral and no longer receives policy-driven motion, then train a new 30-minute cache if the init pose looks right.

---

## v2-124 (2026-04-25) — MiddleScalePos Ring DOF Limit Lock

### Target milestone/subgoal
- Upgrade the ring-mask probe from action-only freezing to a near-physical ring joint lock.

### What changed (files + behavior impact)
- `configs/task/Dexh13HoraLightbulbMiddleScalePos.yaml`
  - Kept ring policy action masking:
    - `apply_action_mask: True`
    - `action_mask_indices: [8, 9, 10, 11]`
  - Kept all ring init joints at `0.0`.
  - Tightened ring DOF lower/upper limits:
    - ring joint 0 lower/upper: `[-0.001, 0.001]`
    - ring joints 1/2/3 lower/upper: `[0.0, 0.001]`
  - Behavior impact:
    - ring starts at zero,
    - policy cannot actively command ring actions,
    - IsaacGym receives near-zero joint limits for ring DOFs, making passive ring motion much more constrained than action masking alone.

### What was verified (commands + key outcomes)
- Config spot check:
  - `nl -ba configs/task/Dexh13HoraLightbulbMiddleScalePos.yaml | sed -n '6,16p;188,202p'`
  - `rg -n "dofLowerLimits|dofUpperLimits|action_mask_indices|right_ring_joint" configs/task/Dexh13HoraLightbulbMiddleScalePos.yaml`
  - Outcome: confirmed ring action mask, ring init zeros, and tightened ring limits.
- Patch hygiene:
  - `git diff --check -- configs/task/Dexh13HoraLightbulbMiddleScalePos.yaml docs/session_handoff_v2.md`
  - Outcome: pass.

### Remaining blocked/risky
- No IsaacGym runtime smoke/training run has been executed after the ring DOF limit lock.
- Limits are near-zero rather than exactly equal to avoid possible physics/import issues from identical lower/upper values.

### Single recommended next step
- Visualize the scalepos init pose with `headless=False` and confirm ring remains effectively fixed before starting a new 30-minute training cache.

---

## v2-125 (2026-04-25) — MiddleScalePos GPU OOM Cleanup

### Target milestone/subgoal
- Restore runnable state for the `Dexh13HoraLightbulbMiddleScalePos` 8192-env PPO teacher probe after a PhysX GPU allocator crash.

### What changed (files + behavior impact)
- No repo files or configs were changed for the runtime issue.
- Stopped the stale Docker training container `festive_hermann`, which was still running the old `initpose_test3` cache and occupying about 10.5 GiB of GPU memory.

### What was verified (commands + key outcomes)
- GPU/process check:
  - `nvidia-smi`
  - `docker ps`
  - `ps -fp 2572386`
  - Outcome before cleanup: old `python train.py ... output_name=Dexh13HoraLightbulb_teacher_middle_scalepos/initpose_test3 ...` occupied about 10560 MiB on the RTX 4080 SUPER.
- Cleanup:
  - `docker stop festive_hermann && nvidia-smi`
  - Outcome after cleanup: GPU memory dropped to about 1869 MiB, leaving enough free memory to rerun the 8192-env scalepos training probe.

### Remaining blocked/risky
- The failed run was a runtime GPU memory exhaustion, not a config syntax or reward-code issue.
- If another viewer or training process is open, 8192 envs can still fail close to env creation. Use a lower env count such as `task.env.numEnvs=4096 train.ppo.minibatch_size=8192` as a fallback.

### Single recommended next step
- Rerun the intended 30-minute `Dexh13HoraLightbulbMiddleScalePos` training command with a fresh cache name, then inspect TensorBoard/W&B contact and termination metrics before deciding whether to keep the ring-lock variant.

---

## v2-126 (2026-04-25) — MiddleScalePos Softer Thumb-Dominance Reduction

### Target milestone/subgoal
- Reduce the current thumb-dominant lightbulb rotation strategy without adding new reward code, by making the existing soft three-finger gate depend more strongly on both index and middle participation.

### What changed (files + behavior impact)
- `configs/task/Dexh13HoraLightbulbMiddleScalePos.yaml`
  - Kept `two_finger_gate.other_aggregation: mean_min`.
  - Changed index/middle aggregation weights:
    - `other_mean_weight: 0.5 -> 0.2`
    - `other_min_weight: 0.5 -> 0.8`
  - Lowered the residual positive-rotation reward multiplier when the gate is poor:
    - `min_mult: 0.20 -> 0.05`
  - Kept contact-force gating enabled and softened its lower threshold:
    - `use_contact_force: True`
    - `contact_force_min: 0.5 -> 0.3`
    - `contact_force_max: 2.0`
  - Behavior impact:
    - thumb-only rotation should receive much less positive rotation reward;
    - index and middle both need to be close/contacting for near-full rotate reward;
    - this is still a config-only probe, not the more aggressive fingertip tangential-contribution reward.

### What was verified (commands + key outcomes)
- Config spot check:
  - `sed -n '70,92p' configs/task/Dexh13HoraLightbulbMiddleScalePos.yaml`
  - Outcome: confirmed the updated soft-three-finger gate weights, `min_mult`, and contact-force threshold.
- Patch hygiene:
  - `git diff --check -- configs/task/Dexh13HoraLightbulbMiddleScalePos.yaml`
  - Outcome: pass.

### Remaining blocked/risky
- This may initially lower scalar reward because it removes an easy thumb-dominant shortcut.
- It still does not directly reward fingertip tangential velocity/work contribution; if index/middle remain light passive contacts, add an explicit contribution/alignment bonus in code as the next, more aggressive probe.

### Single recommended next step
- Run a fresh 30-minute cache for `Dexh13HoraLightbulbMiddleScalePos`, then visualize and compare `two_finger/other_min_w`, `two_finger/other_mean_w`, `two_finger/gate`, and rotation reward against the previous `initpose_test4` behavior.

---

## v2-127 (2026-04-25) — Middle Tangential Contribution Reward

### Target milestone/subgoal
- Move beyond "middle/index are present" gating by adding an explicit middle-finger tangential-motion bonus for the lightbulb task.

### What changed (files + behavior impact)
- `dexscrew/tasks/xhand_hora.py`
  - Added `env.fingertip_tangent_reward` parsing with validation.
  - Added a contact- and distance-gated fingertip tangent reward:
    - computes radial vector from configured target point to fingertip;
    - computes positive tangent direction as `rotation_axis x radial`;
    - projects fingertip linear velocity onto that tangent direction;
    - clips/normalizes the positive projection;
    - multiplies by distance and contact-force weights.
  - Added TensorBoard/W&B extras:
    - `fingertip_tangent/reward`
    - `fingertip_tangent/tangent_vel`
    - `fingertip_tangent/positive_vel`
    - `fingertip_tangent/dist_w`
    - `fingertip_tangent/contact_w`
- `configs/task/Dexh13HoraLightbulbMiddleScalePos.yaml`
  - Added `reward.fingertip_tangent_reward_scale: 0.6`.
  - Enabled `fingertip_tangent_reward` for middle fingertip only (`fingertip_indices: [1]`) with:
    - target `nut_pos + [0.0, 0.0, 0.04]`;
    - distance window `near: 0.08`, `far: 0.13`;
    - `velocity_clip: 0.5`;
    - object-scale-aware distances;
    - contact-force weighting from `0.3` to `2.0`.

### What was verified (commands + key outcomes)
- Syntax check without writing pyc:
  - `PYTHONDONTWRITEBYTECODE=1 python - <<'PY' ... compile(...) ... PY`
  - Outcome: `syntax_ok`.
- Config spot check:
  - `nl -ba configs/task/Dexh13HoraLightbulbMiddleScalePos.yaml | sed -n '58,116p'`
  - Outcome: confirmed tangent reward scale and middle-only tangent reward block.
- Patch hygiene:
  - `git diff --check -- dexscrew/tasks/xhand_hora.py configs/task/Dexh13HoraLightbulbMiddleScalePos.yaml`
  - Outcome: pass.
- Runtime status check:
  - `docker ps`
  - `docker top naughty_almeida -eo pid,ppid,stat,etime,cmd`
  - Outcome: `naughty_almeida` is still running `init_pose_test5`; that already-running process will not pick up the new code/config until restarted.

### Remaining blocked/risky
- No IsaacGym runtime smoke run has been executed after adding the tangent reward.
- The new reward uses fingertip velocity, distance, and contact-force magnitude; it encourages middle tangential motion but still does not directly measure signed contact torque on the bulb.
- If the middle finger gets dragged passively by the already-rotating bulb, the velocity term may still become positive. The contact and distance gates reduce empty motion, but a future force-torque contribution reward may be needed if passive riding appears.

### Single recommended next step
- Stop any old `MiddleScalePos` training container, then run a fresh cache for the tangent-reward variant and monitor `fingertip_tangent/*` together with visual middle-finger behavior.

---

## v2-128 (2026-04-26) — Persistent IsaacGym Shell and Ctrl-C Cleanup Wrapper

### Target milestone/subgoal
- Reduce Docker start/stop friction and make `Ctrl+C` reliably stop IsaacGym training subprocesses without leaving orphaned `python train.py` jobs consuming GPU memory.

### What changed (files + behavior impact)
- `docker-run-isaacgym.sh`
  - Added Docker `--init` so container PID 1 uses Docker's init process for cleaner signal forwarding and child reaping.
- `docker-shell-isaacgym.sh`
  - New helper to enter or start a named interactive IsaacGym container.
  - Default container name: `dexscrew_isaacgym_shell`.
  - If the named container is already running, the helper uses `docker exec -it ... bash`.
- `scripts/run_with_cleanup.sh`
  - New in-container wrapper for training commands.
  - Starts the requested command in a new process group via `setsid`.
  - On `Ctrl+C`, `TERM`, or `HUP`, stops the entire process group, escalating `INT -> TERM -> KILL` if needed.

### What was verified (commands + key outcomes)
- Bash syntax:
  - `bash -n docker-run-isaacgym.sh docker-shell-isaacgym.sh scripts/run_with_cleanup.sh`
  - Outcome: pass.
- Ctrl-C cleanup simulation:
  - `timeout -s INT 1s scripts/run_with_cleanup.sh bash -c 'sleep 1000' ; status=$?; echo status:${status}; pgrep -af 'sleep 1000' || true`
  - Outcome: wrapper emitted process-group cleanup messages and no standalone `sleep 1000` child remained.
- Patch hygiene:
  - `git diff --check -- docker-run-isaacgym.sh docker-shell-isaacgym.sh scripts/run_with_cleanup.sh`
  - Outcome: pass.

### Remaining blocked/risky
- This helper was validated with a dummy process, not a full IsaacGym training run.
- A user can still start multiple training jobs inside the same container and hit GPU OOM; the wrapper solves cleanup, not scheduling.

### Single recommended next step
- Use `./docker-shell-isaacgym.sh` once, then launch the next `MiddleScalePos` probe inside the container with `scripts/run_with_cleanup.sh bash scripts/dexh13_lightbulb_teacher_middle_scalepos.sh ...`; after `Ctrl+C`, check `nvidia-smi` once to confirm no large `python` process remains.

---

## v2-129 (2026-04-26) — Middle Tangent 30-Minute PPO Probe

### Target milestone/subgoal
- Run the first PPO teacher probe after adding the middle-finger tangential contribution reward and the user's updated init pose.

### What changed (files + behavior impact)
- No source/config files were changed during this execution step.
- Stopped the stale `naughty_almeida` container running `init_pose_test5`, freeing about 10.6 GiB of GPU memory.
- Launched a fresh run:
  - cache: `middle_tangent_s42_30m`
  - task: `Dexh13HoraLightbulbMiddleScalePos`
  - seed: `42`
  - timeout: `1800s`
  - W&B run: `middle_tangent_s42_30m_2026-04-26_05-43-08`

### What was verified (commands + key outcomes)
- Pre-run config/runtime check:
  - `rg -n "fingertip_tangent_reward|fingertip_tangent_reward_scale|other_mean_weight|other_min_weight|min_mult|init_pos|handRootPos|right_middle_joint" configs/task/Dexh13HoraLightbulbMiddleScalePos.yaml`
  - Outcome: confirmed tangent reward enabled, soft three-finger gate weights, current `init_pos: [0.012, -0.010, 0.0]`, and `handRootPos: [0.11, 0.020, 0.217]`.
- Training command:
  - `docker exec -w /workspace/dexscrew-repro dexscrew_isaacgym_shell timeout 1800 scripts/run_with_cleanup.sh bash scripts/dexh13_lightbulb_teacher_middle_scalepos.sh 0 42 middle_tangent_s42_30m True wandb_activate=True task.env.termination.log=True`
  - Outcome: run completed via timeout cleanup; no large training `python` process remained afterward.
- Best checkpoint:
  - `outputs/Dexh13HoraLightbulb_teacher_middle_scalepos/middle_tangent_s42_30m/stage1_nn/best_reward_2193.02.pth`
- TensorBoard metric spot check:
  - event: `outputs/Dexh13HoraLightbulb_teacher_middle_scalepos/middle_tangent_s42_30m/stage1_tb/events.out.tfevents.1777182202.wbz-ubuntu22-pc`
  - `episode_rewards/step`: last/max `2192.43`, min `-66.57`.
  - `episode_lengths/step`: last/max `686.87`.
  - `two_finger/gate`: last `0.9983`, max `0.9997`.
  - `two_finger/other_min_w`: last `0.9987`, max `0.9999`.
  - `fingertip_tangent/reward`: last `0.0442`, max `0.1966`.
  - `fingertip_tangent/positive_vel`: last `0.0222`, max `0.0995`.
  - termination fractions were low at the end (`finger_dist_frac=0.000122`, `nut_stagnant_frac=0`, `no_contact_frac=0`).
- Post-run GPU cleanup:
  - `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits`
  - Outcome: no 10+ GiB training `python` process remained.

### Remaining blocked/risky
- The scalar result is healthy, but middle-finger behavior has not yet been visually inspected.
- `fingertip_tangent/reward` is active but relatively small by the end; if visualization still shows passive middle contact, a stronger force-torque contribution reward or a different bulb edge placement may be needed.

### Single recommended next step
- Visualize `middle_tangent_s42_30m` and inspect whether the middle finger actively pushes tangentially or merely rides along while thumb/index still dominate.

---

## v2-130 (2026-04-26) — MiddleScalePos Y-Offset And Torque Contribution Probe

### Target milestone/subgoal
- Continue DexH13 lightbulb middle-finger task-space probing after the user moved the enlarged bulb to `init_pos: [0.012, -0.014, 0.0]`.
- Compare the geometry-only tangent reward run against a new middle-fingertip contact-torque contribution probe.

### What changed (files + behavior impact)
- `configs/task/Dexh13HoraLightbulbMiddleScalePos.yaml`
  - Current user-edited bulb position used for both runs:
    - `object.init_pos: [0.012, -0.014, 0.0]`
  - Added and enabled `reward.fingertip_torque_reward_scale: 0.4`.
  - Added `env.fingertip_torque_reward` for middle fingertip only (`fingertip_indices: [1]`):
    - target: `nut_pos + [0.0, 0.0, 0.04]`
    - distance window: `near: 0.08`, `far: 0.13`
    - `torque_clip: 8.0`
    - `force_sign: -1.0` to approximate fingertip-on-bulb reaction force from Isaac net contact force on the fingertip.
- `dexscrew/tasks/xhand_hora.py`
  - Added `env.fingertip_torque_reward` parsing and validation.
  - Added middle fingertip torque reward:
    - `torque = cross(fingertip_pos - target_pos, force_on_bulb) dot rotation_axis`
    - positive signed torque is clipped/normalized, distance-gated, and contact-gated.
  - Added TensorBoard/W&B extras:
    - `fingertip_torque/reward`
    - `fingertip_torque/signed_torque`
    - `fingertip_torque/positive_torque`
    - `fingertip_torque/negative_torque`
    - `fingertip_torque/abs_torque`
    - `fingertip_torque/dist_w`
    - `fingertip_torque/contact_w`

### What was verified (commands + key outcomes)
- Syntax and patch hygiene:
  - `PYTHONDONTWRITEBYTECODE=1 python - <<'PY' ... compile('dexscrew/tasks/xhand_hora.py', 'exec') ... PY`
  - `git diff --check -- dexscrew/tasks/xhand_hora.py configs/task/Dexh13HoraLightbulbMiddleScalePos.yaml`
  - Outcome: pass.
- Runtime smoke:
  - `docker exec -w /workspace/dexscrew-repro dexscrew_isaacgym_shell timeout 240 scripts/run_with_cleanup.sh bash scripts/dexh13_lightbulb_teacher_middle_scalepos.sh 0 42 torque_smoke_s42 True wandb_activate=False task.env.termination.log=True num_envs=64 train.ppo.minibatch_size=768`
  - Outcome: training loop entered successfully and wrote `fingertip_torque/*` scalars.
  - Smoke showed `torque_clip: 0.04` would saturate immediately, so it was changed to `8.0` and scale lowered to `0.4` before the formal run.
- Geometry-only tangent run:
  - command:
    - `docker exec -w /workspace/dexscrew-repro dexscrew_isaacgym_shell timeout 1800 scripts/run_with_cleanup.sh bash scripts/dexh13_lightbulb_teacher_middle_scalepos.sh 0 42 middle_tangent_yneg014_s42_30m True wandb_activate=True task.env.termination.log=True`
  - best checkpoint:
    - `outputs/Dexh13HoraLightbulb_teacher_middle_scalepos/middle_tangent_yneg014_s42_30m/stage1_nn/best_reward_2274.37.pth`
  - key metrics:
    - `episode_rewards/step` last/max `2273.8`
    - `episode_lengths/step` last/max `698.99`
    - `fingertip_tangent/positive_vel` last `0.0358`, max `0.0655`
    - `fingertip_tangent/reward` last `0.0715`, max `0.1292`
    - `two_finger/gate` last `0.9995`
    - termination fractions remained low.
- Torque reward run:
  - command:
    - `docker exec -w /workspace/dexscrew-repro dexscrew_isaacgym_shell timeout 1800 scripts/run_with_cleanup.sh bash scripts/dexh13_lightbulb_teacher_middle_scalepos.sh 0 42 middle_torqueclip8_yneg014_s42_30m True wandb_activate=True task.env.termination.log=True`
  - W&B run:
    - `middle_torqueclip8_yneg014_s42_30m_2026-04-26_08-45-18`
    - run id: `3oslhmxg`
  - best checkpoint:
    - `outputs/Dexh13HoraLightbulb_teacher_middle_scalepos/middle_torqueclip8_yneg014_s42_30m/stage1_nn/best_reward_2222.42.pth`
  - key metrics:
    - `episode_rewards/step` last/max `2219.1` (stdout best `2222.42`)
    - `episode_lengths/step` last/max `693.65`
    - `screw/angular_velocity` last `0.8883`, max `1.2829`
    - `fingertip_tangent/positive_vel` last `0.0624`, max `0.0693`
    - `fingertip_tangent/reward` last `0.1248`, max `0.1386`
    - `fingertip_torque/reward` last `0.7026`, max `0.7827`
    - `fingertip_torque/signed_torque` last `5.6233`, max `6.2865`
    - `fingertip_torque/negative_torque` stayed near zero.
    - `two_finger/gate` last `0.9994`
    - termination fractions remained low.
- Post-run GPU cleanup:
  - `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits`
  - Outcome: no large IsaacGym training `python` process remained.

### Remaining blocked/risky
- Scalar metrics cannot prove the middle finger is visually doing useful work; the torque run must be visualized.
- `fingertip_torque/*` is based on net fingertip contact force, so it is still a proxy for finger-on-bulb torque and may include non-bulb contacts if the geometry changes.
- The torque run slightly lowers scalar best reward (`2222.42` vs `2274.37`) but improves middle-related signal:
  - tangent positive velocity `0.0624` vs `0.0358`
  - tangent reward `0.1248` vs `0.0715`

### Single recommended next step
- Visualize `middle_torqueclip8_yneg014_s42_30m` and compare it against `middle_tangent_yneg014_s42_30m`.
- If middle visibly contributes, keep the torque reward but consider reducing scale to `0.2` for a longer run.
- If middle still passively rides along, set `fingertip_torque_reward_scale: 0.0` and keep `fingertip_torque/*` as diagnostics while continuing geometry/init-pose search.

---

## v2-131 (2026-04-26) — MiddleScalePos Torque Scale Closed-Loop Probe

### Target milestone/subgoal
- Continue the DexH13 lightbulb middle-finger task-space probe with the user's current bulb pose.
- Sweep middle fingertip torque-reward strength to balance scalar PPO reward against explicit middle-finger contribution metrics.

### What changed (files + behavior impact)
- `configs/task/Dexh13HoraLightbulbMiddleScalePos.yaml`
  - Changed `reward.fingertip_torque_reward_scale` from `0.4` to `0.2`, ran a 30-minute probe, then changed it to `0.3` and ran a second 30-minute probe.
  - Current checked-in config is `fingertip_torque_reward_scale: 0.3`.
  - Kept all other active scalepos settings fixed:
    - `object.init_pos: [0.012, -0.014, 0.0]`
    - `baseObjScale: 1.20`
    - `handRootPos: [0.11, 0.020, 0.217]`
    - soft three-finger gate with `mean_min`, `other_mean_weight: 0.2`, `other_min_weight: 0.8`, `min_mult: 0.05`
    - middle-only tangent and torque reward blocks
    - ring action mask plus near-zero ring DOF limits.

### What was verified (commands + key outcomes)
- Syntax and patch hygiene:
  - `PYTHONDONTWRITEBYTECODE=1 python - <<'PY' ... compile('dexscrew/tasks/xhand_hora.py', 'exec') ... PY`
  - `git diff --check -- dexscrew/tasks/xhand_hora.py configs/task/Dexh13HoraLightbulbMiddleScalePos.yaml`
  - Outcome: pass.
- Torque scale `0.2` run:
  - command:
    - `docker exec -w /workspace/dexscrew-repro dexscrew_isaacgym_shell timeout 1800 scripts/run_with_cleanup.sh bash scripts/dexh13_lightbulb_teacher_middle_scalepos.sh 0 42 middle_torque02_yneg014_s42_30m True wandb_activate=True task.env.termination.log=True`
  - W&B run:
    - `middle_torque02_yneg014_s42_30m_2026-04-26_09-47-36`
    - run id: `affubsbp`
  - best checkpoint:
    - `outputs/Dexh13HoraLightbulb_teacher_middle_scalepos/middle_torque02_yneg014_s42_30m/stage1_nn/best_reward_2275.70.pth`
  - key TensorBoard metrics:
    - `episode_rewards/step` last/max `2274.99`
    - `episode_lengths/step` last/max `704.65`
    - `fingertip_tangent/positive_vel` last `0.0490`, max `0.0542`
    - `fingertip_tangent/reward` last `0.0979`, max `0.1063`
    - `fingertip_torque/signed_torque` last `5.7763`, max `6.1914`
    - `fingertip_torque/reward` last `0.7217`, max `0.7709`
    - `two_finger/gate` last `0.9988`
    - termination fractions remained low.
- Torque scale `0.3` run:
  - command:
    - `docker exec -w /workspace/dexscrew-repro dexscrew_isaacgym_shell timeout 1800 scripts/run_with_cleanup.sh bash scripts/dexh13_lightbulb_teacher_middle_scalepos.sh 0 42 middle_torque03_yneg014_s42_30m True wandb_activate=True task.env.termination.log=True`
  - W&B run:
    - `middle_torque03_yneg014_s42_30m_2026-04-26_10-18-54`
    - run id: `mhm3z12p`
  - best checkpoint:
    - `outputs/Dexh13HoraLightbulb_teacher_middle_scalepos/middle_torque03_yneg014_s42_30m/stage1_nn/best_reward_2325.61.pth`
  - key TensorBoard metrics:
    - `episode_rewards/step` last/max `2325.05`
    - `episode_lengths/step` last/max `704.93`
    - `fingertip_tangent/positive_vel` last `0.0389`, max `0.0441`
    - `fingertip_tangent/reward` last `0.0768`, max `0.0873`
    - `fingertip_torque/signed_torque` last `5.5558`, max `6.2639`
    - `fingertip_torque/reward` last `0.6913`, max `0.7782`
    - `two_finger/gate` last `0.9956`
    - termination fractions remained low.
- Baseline comparison from this sweep:
  - tangent-only y=-0.014 run: best checkpoint `2274.37`, TB reward last/max `2273.8`.
  - torque `0.4` run: best checkpoint `2222.42`, TB reward last/max `2219.1`, but strongest middle tangent metrics.
  - torque `0.3` gives best scalar reward so far.
  - torque `0.2` gives a better middle-signal compromise than `0.3` while almost matching tangent-only scalar reward.
- Post-run GPU cleanup:
  - `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits`
  - Outcome: no large IsaacGym training `python` process remained; only small Chrome/ToDesk GPU processes were listed.

### Remaining blocked/risky
- Scalar reward alone still does not prove middle-finger active contribution.
- `0.3` is the best scalar checkpoint, but its terminal middle tangent metrics are weaker than `0.2` and much weaker than `0.4`.
- `0.2` may be the better visual-behavior candidate if the goal is specifically to see middle-finger tangential participation instead of only maximizing short-run reward.

### Single recommended next step
- Visualize `middle_torque03_yneg014_s42_30m` first because it is the best scalar checkpoint.
- Compare against `middle_torque02_yneg014_s42_30m`; if `0.3` still looks thumb-dominant while `0.2` shows more middle participation, revert config to `fingertip_torque_reward_scale: 0.2` before longer training.

---

## v2-132 (2026-04-27) — IsaacGym Wrapper In-Container Guard

### Target milestone/subgoal
- Fix the user's visualization launch error when running `./docker-run-isaacgym.sh ...` from an already-open IsaacGym container shell at `/workspace/dexscrew-repro`.

### What changed (files + behavior impact)
- `docker-run-isaacgym.sh`
  - Added an in-container guard:
    - detects `/.dockerenv` plus `/opt/isaacgym/.../gym_38.so`;
    - exports `ISAACGYM_DIR=/opt/isaacgym`, `ISAACGYM_PATH=/opt/isaacgym`, and IsaacGym `PYTHONPATH`;
    - directly executes the requested command instead of trying to start a nested Docker container.
  - Behavior impact:
    - From host: wrapper behavior is unchanged.
    - From inside the persistent container: `./docker-run-isaacgym.sh bash scripts/vis_...` now works as a forgiving alias for direct execution.

### What was verified (commands + key outcomes)
- Syntax:
  - `bash -n docker-run-isaacgym.sh`
  - Outcome: pass.
- In-container wrapper path:
  - `docker exec -w /workspace/dexscrew-repro dexscrew_isaacgym_shell bash -lc './docker-run-isaacgym.sh bash -lc "echo ISAACGYM_DIR=\\$ISAACGYM_DIR; python -c \\"import isaacgym; print(\\\\\\"isaacgym_ok\\\\\\")\\""'`
  - Outcome: printed `ISAACGYM_DIR=/opt/isaacgym` and imported `/opt/isaacgym/.../gym_38.so` successfully.

### Remaining blocked/risky
- The fix resolves the IsaacGym binding path error only.
- Viewer launch can still depend on X11/display forwarding; if a later error mentions `DISPLAY`, `XAUTHORITY`, or graphics device, handle that separately.

### Single recommended next step
- Rerun the original visualization command from the current shell:
  - `./docker-run-isaacgym.sh bash scripts/vis_dexh13_lightbulb_teacher_middle_scalepos.sh 0 42 middle_torque03_yneg014_s42_30m`

---

## v2-133 (2026-04-27) -- MiddleScalePos Six-Run Middle Contribution Matrix

### Target milestone/subgoal
- Execute the requested closed-loop DexH13 lightbulb middle-finger matrix:
  - compare `y=-0.018` vs `y=-0.020`;
  - compare middle torque reward scale `0.3` vs `0.2`;
  - test aggressive index joint0 interference limit `upper=0.1`;
  - add per-finger diagnostics before changing reward further.

### What changed (files + behavior impact)
- `dexscrew/tasks/xhand_hora.py`
  - Added diagnostic-only per-finger scalar logging. Reward behavior is unchanged by these new logs.
  - New scalar groups:
    - `finger_torque/index|middle|thumb/signed`
    - `finger_torque/index|middle|thumb/positive`
    - `finger_torque/index|middle|thumb/ratio_positive`
    - `finger_tangent/index|middle|thumb/positive_vel`
    - `finger_contact/index|middle|thumb/force_w`
    - `finger_dist/index|middle|thumb`
    - `finger_motion/middle_joint_vel_abs`
    - `finger_motion/middle_joint0_sign_flip_rate`
    - `finger_motion/index_middle_tip_dist`
- Main task config was not permanently moved for the scan; y offsets, torque scales, and index joint0 upper limit were applied through Hydra overrides per run.

### What was verified (commands + key outcomes)
- Syntax and patch hygiene:
  - `PYTHONDONTWRITEBYTECODE=1 python - <<'PY' ... compile(Path('dexscrew/tasks/xhand_hora.py').read_text(), 'dexscrew/tasks/xhand_hora.py', 'exec') ... PY`
  - `git diff --check -- dexscrew/tasks/xhand_hora.py configs/task/Dexh13HoraLightbulbMiddleScalePos.yaml docker-run-isaacgym.sh docs/session_handoff_v2.md`
  - Outcome: pass.
- 64-env smoke run:
  - `docker exec -w /workspace/dexscrew-repro dexscrew_isaacgym_shell timeout 240 scripts/run_with_cleanup.sh bash scripts/dexh13_lightbulb_teacher_middle_scalepos.sh 0 42 diag_smoke_s42 True wandb_activate=False task.env.termination.log=True num_envs=64 train.ppo.minibatch_size=768`
  - Outcome: completed under timeout with best reward `98.06`.
  - TensorBoard scalar parse confirmed all new `finger_*` diagnostic tags were written.
- Six requested 30-minute runs, all launched through `scripts/run_with_cleanup.sh`, seed `42`, 8192 envs, W&B enabled, termination logging enabled:

| Group | Cache | y | torque scale | index joint0 upper | best ckpt reward | reward last | middle pos torque | middle ratio | middle tangent vel | middle joint vel | tip dist | any reset frac | Notes |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A1 | `middle_yneg018_torque03_diag_s42_30m` | -0.018 | 0.3 | unchanged | 2291.35 | 2291.35 | 6.7000 | 0.8374 | 0.0562 | 1.1461 | 0.0351 | 0.000122 | Best scalar reward |
| A2 | `middle_yneg020_torque03_diag_s42_30m` | -0.020 | 0.3 | unchanged | 2265.80 | 2265.80 | 7.0730 | 0.8267 | 0.0680 | 1.4239 | 0.0365 | 0.000488 | Best middle torque/tangent among good-reward runs |
| A3 | `middle_yneg018_torque03_idxlim_s42_30m` | -0.018 | 0.3 | 0.1 | 2190.39 | 2190.39 | 6.4346 | 0.8596 | 0.0602 | 1.3598 | 0.0331 | 0.000366 | Highest middle ratio, but reward lower and tips closer |
| B1 | `middle_yneg018_torque02_diag_s42_30m` | -0.018 | 0.2 | unchanged | 2232.55 | 2231.32 | 6.0284 | 0.8269 | 0.0539 | 1.1764 | 0.0407 | 0.000488 | Best anti-interference proxy |
| B2 | `middle_yneg020_torque02_diag_s42_30m` | -0.020 | 0.2 | unchanged | 2206.53 | 2116.48 | 6.5251 | 0.8099 | 0.0328 | 0.9704 | 0.0363 | 0.000488 | Weak middle tangent despite checkpoint passing |
| B3 | `middle_yneg018_torque02_idxlim_s42_30m` | -0.018 | 0.2 | 0.1 | 1833.89 | 1833.50 | 6.4275 | 0.7576 | 0.0636 | 1.3548 | 0.0311 | 0.000488 | Failed reward threshold; thumb contribution increased |

- Best checkpoints:
  - `outputs/Dexh13HoraLightbulb_teacher_middle_scalepos/middle_yneg018_torque03_diag_s42_30m/stage1_nn/best_reward_2291.35.pth`
  - `outputs/Dexh13HoraLightbulb_teacher_middle_scalepos/middle_yneg020_torque03_diag_s42_30m/stage1_nn/best_reward_2265.80.pth`
  - `outputs/Dexh13HoraLightbulb_teacher_middle_scalepos/middle_yneg018_torque03_idxlim_s42_30m/stage1_nn/best_reward_2190.39.pth`
  - `outputs/Dexh13HoraLightbulb_teacher_middle_scalepos/middle_yneg018_torque02_diag_s42_30m/stage1_nn/best_reward_2232.55.pth`
  - `outputs/Dexh13HoraLightbulb_teacher_middle_scalepos/middle_yneg020_torque02_diag_s42_30m/stage1_nn/best_reward_2206.53.pth`
  - `outputs/Dexh13HoraLightbulb_teacher_middle_scalepos/middle_yneg018_torque02_idxlim_s42_30m/stage1_nn/best_reward_1833.89.pth`
- Post-run GPU cleanup:
  - `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits`
  - Outcome after each run: no large IsaacGym training `python` process remained; only small desktop/browser GPU processes were present.

### Local conclusion
- The best scalar candidate is A1 (`middle_yneg018_torque03_diag_s42_30m`).
- The best middle-contribution candidate is A2 (`middle_yneg020_torque03_diag_s42_30m`):
  - highest middle positive torque among reward-passing runs;
  - highest middle tangent positive velocity among reward-passing runs.
- B1 (`middle_yneg018_torque02_diag_s42_30m`) is the most conservative anti-interference backup:
  - largest index-middle fingertip distance;
  - lower middle joint0 flip maximum than the torque-0.3 variants;
  - scalar reward still above the short-run acceptance floor.
- Aggressive index joint0 upper limit `0.1` is not a good default yet:
  - A3 reduces index contribution and raises middle ratio, but lowers reward and brings index/middle tips closer.
  - B3 fails the reward threshold and shifts load toward thumb.
  - If index limiting is revisited, try a milder `upper=0.2` before using `0.1`.

### Remaining blocked/risky
- These diagnostics are fingertip net-force proxies; they strongly suggest middle contribution but still need viewer confirmation that contact is actually on the bulb and not from incidental geometry.
- A2 may have more useful middle torque but also higher middle joint velocity, so it could still look like visible oscillation.
- Because the six-run matrix already meets the proxy middle-ratio threshold, the next decision should be based on visual behavior, not scalar reward alone.

### Single recommended next step
- Visualize A2 and A1:
  - A2 first for middle contribution: `./docker-run-isaacgym.sh bash scripts/vis_dexh13_lightbulb_teacher_middle_scalepos.sh 0 42 middle_yneg020_torque03_diag_s42_30m`
  - A1 second for scalar baseline: `./docker-run-isaacgym.sh bash scripts/vis_dexh13_lightbulb_teacher_middle_scalepos.sh 0 42 middle_yneg018_torque03_diag_s42_30m`
- If A2 visually shows middle active pushing without severe oscillation, continue the y scan at `y=-0.022` and `y=-0.024` with torque scale `0.3`.
- If A2 still oscillates, compare B1 visually before adding any stronger reward; B1 is the best lower-jitter backup.

---

## v2-134 (2026-04-27) -- MiddleScalePos 2h Index Upper 0.15 Probe

### Target milestone/subgoal
- Run the user's requested 2-hour DexH13 lightbulb PPO probe:
  - `object.init_pos: [0.012, -0.018, 0.0]`
  - `fingertip_torque_reward_scale: 0.3`
  - index joint0 upper limit relaxed from the failed aggressive `0.1` probe to `0.15`.

### What changed (files + behavior impact)
- No source or task config files were changed for the experiment.
- The index joint0 limit was applied only as a Hydra override:
  - `task.env.asset.dofUpperLimits=[0.15,1.57,1.57,1.57,0.35,1.57,1.57,1.57,0.001,0.001,0.001,0.001,0.35,1.57,1.57,1.57]`

### What was verified (commands + key outcomes)
- Bootstrap context:
  - Read `docs/session_handoff_v2.md` and `docs/stage_acceptance_summary.md`.
- Training command:
  - `docker exec -w /workspace/dexscrew-repro dexscrew_isaacgym_shell timeout 7200 scripts/run_with_cleanup.sh bash scripts/dexh13_lightbulb_teacher_middle_scalepos.sh 0 42 middle_yneg018_torque03_idxlim015_s42_2h True wandb_activate=True task.env.termination.log=True 'task.env.object.init_pos=[0.012,-0.018,0.0]' 'task.env.reward.fingertip_torque_reward_scale=0.3' 'task.env.asset.dofUpperLimits=[0.15,1.57,1.57,1.57,0.35,1.57,1.57,1.57,0.001,0.001,0.001,0.001,0.35,1.57,1.57,1.57]'`
  - Outcome: timeout ended normally after about `Collect 105.1min + Train RL 14.5min ~= 119.6min`.
  - No OOM or segmentation fault occurred.
- Best checkpoint:
  - `outputs/Dexh13HoraLightbulb_teacher_middle_scalepos/middle_yneg018_torque03_idxlim015_s42_2h/stage1_nn/best_reward_2947.48.pth`
- TensorBoard event:
  - `outputs/Dexh13HoraLightbulb_teacher_middle_scalepos/middle_yneg018_torque03_idxlim015_s42_2h/stage1_tb/events.out.tfevents.1777247662.wbz-ubuntu22-pc`
- Key scalar metrics:
  - `episode_rewards/step`: last `2870.07`, max `2947.48`
  - `episode_lengths/step`: last `789.31`, max `792.36`
  - `screw/angular_velocity`: last `0.5601`, max `1.2571`
  - `screw/positive_vel_ratio`: last `0.7545`, max `0.8195`
  - `fingertip_tangent/positive_vel`: last `0.0450`, max `0.0740`
  - `fingertip_torque/signed_torque`: last `6.8288`, max `6.9272`
  - `fingertip_torque/reward`: last `0.8484`, max `0.8638`
  - `finger_torque/index/positive`: last `1.1856`, max `1.3200`
  - `finger_torque/middle/positive`: last `6.8288`, max `6.9272`
  - `finger_torque/thumb/positive`: last `0.0401`, max `2.2561`
  - `finger_torque/index/ratio_positive`: last `0.1465`, max `0.1579`
  - `finger_torque/middle/ratio_positive`: last `0.8500`, max `0.8980`
  - `finger_torque/thumb/ratio_positive`: last `0.0035`, max `0.1697`
  - `finger_tangent/index/positive_vel`: last `0.0146`, max `0.0381`
  - `finger_tangent/middle/positive_vel`: last `0.0450`, max `0.0740`
  - `finger_tangent/thumb/positive_vel`: last `0.0684`, max `0.1299`
  - `finger_motion/middle_joint_vel_abs`: last `0.9608`, max `1.2410`
  - `finger_motion/middle_joint0_sign_flip_rate`: last `0.0`, max `0.8209`
  - `finger_motion/index_middle_tip_dist`: last `0.0347`, max `0.0382`
  - `term/any_reset_frac`: last `0.000122`, max `0.002563`
  - `term/no_contact_frac`: last/max `0`
- Post-run cleanup:
  - `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits`
  - `pgrep -af 'train.py|dexh13_lightbulb_teacher_middle_scalepos|middle_yneg018_torque03_idxlim015' || true`
  - Outcome: no training Python process remained; only desktop/browser GPU processes were listed.

### Local conclusion
- Reward-wise, index joint0 upper `0.15` is much better than the previous aggressive `0.1` probe:
  - previous A3 30min `upper=0.1`: best `2190.39`
  - current 2h `upper=0.15`: best `2947.48`
- It also surpasses the no-index-limit short probes:
  - A1 `y=-0.018`, torque `0.3`, no index limit: best `2291.35`
  - A2 `y=-0.020`, torque `0.3`, no index limit: best `2265.80`
- The new diagnostics strongly favor middle contribution:
  - middle positive torque last `6.83`
  - middle positive torque ratio last `0.85`
  - index ratio last `0.15`
  - thumb ratio last `0.0035`
- The caveat remains important: `finger_torque/*` is a fingertip net-force proxy, so visual confirmation is required before concluding that middle truly rotates the bulb.

### Remaining blocked/risky
- The best reward checkpoint appears around the first half of the 2h run; the last scalar reward is lower (`2870.07`), so the best checkpoint should be used for visualization.
- Thumb tangent positive velocity remains nontrivial (`0.0684` last), even though thumb positive torque ratio is tiny; visualization should check whether thumb is still driving motion through another contact geometry.
- Index-middle tip distance is not larger than B1, so the index-limit improvement may be coming from reduced index torque dominance rather than clean spatial separation.

### Single recommended next step
- Visualize the new best checkpoint:
  - `./docker-run-isaacgym.sh bash scripts/vis_dexh13_lightbulb_teacher_middle_scalepos.sh 0 42 middle_yneg018_torque03_idxlim015_s42_2h`
- Compare visually against A2 (`middle_yneg020_torque03_diag_s42_30m`).
- If middle visibly pushes the bulb edge in the new run, make `index joint0 upper=0.15` the active mild index-limit setting for the next geometry sweep or long run.

---

## v2-135 (2026-04-27) -- Smooth Lightbulb Head Collision Probe

### Target milestone/subgoal
- Test the user's hypothesis that replacing the faceted lightbulb head collision with a smoother ellipsoid-like collision may reduce the thumb's late-rotation slip/launch behavior.
- Keep the strongest current middle-finger setup fixed:
  - `object.init_pos: [0.012, -0.018, 0.0]`
  - `fingertip_torque_reward_scale: 0.3`
  - index joint0 upper limit `0.15`
  - ring action mask plus near-zero ring DOF lock.

### What changed (files + behavior impact)
- `assets/lightbulb/smooth_head_collision.stl`
  - Added a generated smooth ellipsoid collision mesh for the lightbulb head.
  - It matches the previous `contact0.stl` bounding box exactly:
    - center approximately `[0.021264, -0.0000615, -0.0000615]`
    - radii approximately `[0.034592, 0.0315595, 0.0315595]`
  - Mesh size is intentionally modest: `960` triangles, `48084` bytes.
- `assets/screw/smoothbulb/0000_lightbulb.urdf`
  - Added a new object asset type path instead of overwriting `screw_contactviz`.
  - Visual meshes use the high-fidelity lightbulb head/socket meshes.
  - Head collision uses `../../lightbulb/smooth_head_collision.stl`.
  - Socket collision keeps the existing `../../lightbulb/contact1.stl`.
  - This can be selected with Hydra override `task.env.object.type=screw_smoothbulb`.

### What was verified (commands + key outcomes)
- Bootstrap context:
  - Read `docs/session_handoff_v2.md` and `docs/stage_acceptance_summary.md`.
- Asset validation:
  - Parsed `assets/screw/smoothbulb/0000_lightbulb.urdf` with `xml.etree.ElementTree`.
  - Verified the new STL exists and its bounding box matches the previous head collision dimensions.
- Syntax and patch hygiene:
  - `PYTHONDONTWRITEBYTECODE=1 python - <<'PY' ... compile(Path('dexscrew/tasks/xhand_hora.py').read_text(), 'dexscrew/tasks/xhand_hora.py', 'exec') ... PY`
  - `git diff --check -- dexscrew/tasks/xhand_hora.py configs/task/Dexh13HoraLightbulbMiddleScalePos.yaml assets/screw/smoothbulb/0000_lightbulb.urdf docs/session_handoff_v2.md`
  - Outcome: pass.
- Resource cleanup before training:
  - Found an old visualization container for `middle_yneg018_torque03_idxlim015_s42_2h` using about `3.1GB` GPU memory.
  - Stopped that stale visualization container before the 8192-env training run.
- 64-env smoke:
  - command:
    - `./docker-run-isaacgym.sh timeout 240 scripts/run_with_cleanup.sh bash scripts/dexh13_lightbulb_teacher_middle_scalepos.sh 0 42 smooth_smoke_s42 True wandb_activate=False task.env.termination.log=True task.env.object.type=screw_smoothbulb 'task.env.asset.dofUpperLimits=[0.15,1.57,1.57,1.57,0.35,1.57,1.57,1.57,0.001,0.001,0.001,0.001,0.35,1.57,1.57,1.57]' num_envs=64 train.ppo.minibatch_size=768`
  - Outcome: environment built, training loop entered, and timeout cleanup worked.
  - Best smoke checkpoint:
    - `outputs/Dexh13HoraLightbulb_teacher_middle_scalepos/smooth_smoke_s42/stage1_nn/best_reward_-100.83.pth`
- 30-minute formal training:
  - command:
    - `./docker-run-isaacgym.sh timeout 1800 scripts/run_with_cleanup.sh bash scripts/dexh13_lightbulb_teacher_middle_scalepos.sh 0 42 smoothbulb_yneg018_torque03_idxlim015_s42_30m True wandb_activate=True task.env.termination.log=True task.env.object.type=screw_smoothbulb 'task.env.asset.dofUpperLimits=[0.15,1.57,1.57,1.57,0.35,1.57,1.57,1.57,0.001,0.001,0.001,0.001,0.35,1.57,1.57,1.57]'`
  - W&B run:
    - `smoothbulb_yneg018_torque03_idxlim015_s42_30m_2026-04-27_05-09-20`
    - run id: `v1zyalrq`
  - Outcome: timeout ended normally at about `Collect 26.0min + Train RL 3.5min ~= 29.5min`; no OOM or segmentation fault.
  - Best checkpoint:
    - `outputs/Dexh13HoraLightbulb_teacher_middle_scalepos/smoothbulb_yneg018_torque03_idxlim015_s42_30m/stage1_nn/best_reward_2019.50.pth`
  - TensorBoard event:
    - `outputs/Dexh13HoraLightbulb_teacher_middle_scalepos/smoothbulb_yneg018_torque03_idxlim015_s42_30m/stage1_tb/events.out.tfevents.1777266574.wbz-ubuntu22-pc`
- Key scalar metrics:
  - `episode_rewards/step`: last/max `2018.00`
  - `episode_lengths/step`: last/max `670.48`
  - `screw/angular_velocity`: last `0.4918`, max `1.1309`
  - `screw/positive_vel_ratio`: last `0.7351`, max `0.8011`
  - `fingertip_tangent/positive_vel`: last `0.0686`, max `0.0700`
  - `fingertip_torque/signed_torque`: last `6.6157`, max `6.8753`
  - `fingertip_torque/reward`: last `0.8266`, max `0.8545`
  - `finger_torque/index/positive`: last `1.1135`, max `1.4664`
  - `finger_torque/middle/positive`: last `6.6157`, max `6.8753`
  - `finger_torque/thumb/positive`: last `0.0837`, max `2.2897`
  - `finger_torque/index/ratio_positive`: last `0.1420`, max `0.1764`
  - `finger_torque/middle/ratio_positive`: last `0.8506`, max `0.8595`
  - `finger_torque/thumb/ratio_positive`: last `0.0074`, max `0.1774`
  - `finger_tangent/index/positive_vel`: last `0.0166`, max `0.0300`
  - `finger_tangent/middle/positive_vel`: last `0.0686`, max `0.0700`
  - `finger_tangent/thumb/positive_vel`: last `0.0616`, max `0.1100`
  - `finger_motion/middle_joint_vel_abs`: last/max `1.6011`
  - `finger_motion/middle_joint0_sign_flip_rate`: last `0.0`, max `0.8413`
  - `finger_motion/index_middle_tip_dist`: last `0.0346`, max `0.0403`
  - `term/any_reset_frac`: last `0.000488`, max `0.002563`
  - `term/no_contact_frac`: last/max `0`
- Post-run cleanup:
  - `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits`
  - `pgrep -af 'train.py|dexh13_lightbulb_teacher_middle_scalepos|smoothbulb_yneg018_torque03_idxlim015' || true`
  - Outcome: no training Python process remained; only the desktop/browser GPU process was listed.

### Local conclusion
- Smooth head collision did not crash and remains trainable, but it is weaker than the previous contactviz collision under the same 30-minute-style comparison:
  - smoothbulb 30m: best `2019.50`
  - previous no-index-limit y=-0.018 30m A1: best `2291.35`
  - previous 2h index upper `0.15`: best `2947.48`
- The smooth run preserved the desired diagnostic middle dominance:
  - middle torque ratio last `0.8506`
  - index ratio last `0.1420`
  - thumb ratio last `0.0074`
- The likely tradeoff is contact stability vs usable geometry:
  - smooth collision may reduce edge-induced thumb pop-out;
  - but it also removes faceted contact affordances and lowers reward/episode length.
- Middle joint velocity is higher than the 2h contactviz run (`1.60` vs `0.96` last), so this asset may increase middle activity but not necessarily reduce visible jitter.

### Remaining blocked/risky
- The key hypothesis is visual: scalars cannot confirm whether the thumb still shoots out at the end of rotation.
- Reward is below the short-run acceptance floor of about `2090`, so this should not replace `screw_contactviz` as the default unless visualization clearly improves stability.
- The socket collision is still the old faceted `contact1.stl`; if the thumb pop-out happens near the socket/head transition, smoothing only the head may be insufficient.

### Single recommended next step
- Visualize the smoothbulb checkpoint and compare directly against the previous best contactviz checkpoint:
  - smooth candidate:
    - `./docker-run-isaacgym.sh bash scripts/vis_dexh13_lightbulb_teacher_middle_scalepos.sh 0 42 smoothbulb_yneg018_torque03_idxlim015_s42_30m task.env.object.type=screw_smoothbulb 'task.env.asset.dofUpperLimits=[0.15,1.57,1.57,1.57,0.35,1.57,1.57,1.57,0.001,0.001,0.001,0.001,0.35,1.57,1.57,1.57]'`
  - contactviz comparison:
    - `./docker-run-isaacgym.sh bash scripts/vis_dexh13_lightbulb_teacher_middle_scalepos.sh 0 42 middle_yneg018_torque03_idxlim015_s42_2h`
- If smoothbulb visibly fixes thumb launch, run a 2h smoothbulb continuation before deciding.
- If thumb still launches, revert to contactviz for training and instead test a smaller local collision smoothing/chamfer around the bulb shoulder rather than a fully smooth head.

---

## v2-136 (2026-04-27) -- Sim2sim Readiness Codebase Survey

### Target milestone/subgoal
- Prepare for sim2sim policy validation by mapping the current teacher/student/export path, reusable runtime pieces, and DexH13 lightbulb asset/config risks.
- No training or simulator run was started in this survey.

### What changed (files + behavior impact)
- `docs/session_handoff_v2.md`
  - Added this survey handoff entry only.
- No source code, task config, asset, checkpoint, or script behavior was changed.

### What was verified (commands + key outcomes)
- Bootstrap context:
  - Read `docs/session_handoff_v2.md` and `docs/stage_acceptance_summary.md`.
  - Outcome: latest documented execution step before this survey was still the smoothbulb/contactviz visual comparison from `v2-135`.
- Codebase survey:
  - Inspected `train.py`, `student_eval.py`, `instruction_docs/repo_strategy_map.md`, `xhand-deploy/xhand_deploy.py`, `configs/task/Dexh13HoraLightbulbMiddleScalePos.yaml`, `configs/train/Dexh13HoraLightbulbMiddleScalePos.yaml`, DexH13 lightbulb scripts, and contactviz/smoothbulb URDFs.
  - Searched for `sim2sim`, `mujoco`, `genesis`, `isaaclab`, `deploy`, `export`, `onnx`, and TorchScript-related paths.
  - Outcome: no direct sim2sim rollout harness exists yet; repo has IsaacGym train/eval, ProprioAdapt TorchScript export, XHand deploy runtime logic, and MuJoCo/DexH13 assets/docs.
- Subagent survey:
  - Teacher/student/export path mapped.
  - Sim2sim harness gaps mapped.
  - DexH13 lightbulb contactviz/smoothbulb asset/config risks mapped.
- Additional artifact check:
  - Found an unlogged later smoothbulb run:
    - `outputs/Dexh13HoraLightbulb_teacher_middle_scalepos/smoothbulb_visualmatch_yneg018_torque03_idxlim015_s42_2h/stage1_nn/best_reward_2849.55.pth`
    - Its config snapshot confirms `task.env.object.type=screw_smoothbulb` and index joint0 upper limit `0.15`.

### Local conclusion
- The shortest low-risk sim2sim path is still `XHandHoraScrewDriver` + `ProprioAdapt` TorchScript:
  - export/runtime pieces already exist;
  - policy input is `obs`, `proprio_hist`, `point_cloud_info`;
  - output is `mu`;
  - external runtime must reproduce normalization, 3-frame obs history, 30-frame proprio history, joint order, action mask, and `target = prev_target + action_scale * mu`.
- Diffusion/consistency/flow students are evaluable inside this repo, but `student_eval.py` explicitly does not support JIT export for diffusion-family students yet.
- DexH13 lightbulb is possible for sim2sim, but it is not the first harness target:
  - current useful policies are teacher checkpoints, not mature exported students;
  - current best checkpoints depend on exact object type and index limit overrides;
  - contact geometry and thumb launch still need visual confirmation.

### Remaining blocked/risky
- There is no existing MuJoCo/Genesis/IsaacLab rollout harness to run a policy end to end.
- `scripts/convert_student_jit.sh` is hardcoded around `XHandHoraScrewDriver` and should be used with explicit `train.load_path=...`.
- TorchScript wrapper stores normalization stats but does not normalize inside `forward`; the caller must normalize like `xhand-deploy/xhand_deploy.py`.
- For DexH13 lightbulb, forgetting `task.env.object.type=screw_smoothbulb` or index joint0 upper `0.15` when using matching checkpoints will cause policy/env mismatch.
- `screw_contactviz` and `screw_smoothbulb` have no `.npy` point cloud, so the current code falls back to a cylinder point cloud while `use_point_cloud_info=True`.

### Single recommended next step
- Before starting DexH13/lightbulb sim2sim, visually compare:
  - contactviz best: `middle_yneg018_torque03_idxlim015_s42_2h`
  - smoothbulb later candidate: `smoothbulb_visualmatch_yneg018_torque03_idxlim015_s42_2h`
- Use explicit object type and index-limit overrides during visualization; only after that choose the DexH13 asset/checkpoint for sim2sim. If the immediate goal is a generic sim2sim harness instead, start with `XHandHoraScrewDriver` + `ProprioAdapt` TorchScript rather than DexH13.

---

## v2-137 (2026-04-27) -- Low-Reward Output Cleanup

### Target milestone/subgoal
- Organize `outputs/` by removing low-value run directories whose directory-level best checkpoint reward is below `1000`.
- Keep current high-value DexH13 lightbulb / screwdriver artifacts intact.

### What changed (files + behavior impact)
- Deleted 51 run directories under `outputs/`.
  - Selection rule: run directory contains `stage*_nn/best_reward_*.pth`, and the maximum parsed `best_reward` in that run directory is `< 1000`.
  - Deletion was directory-level, not checkpoint-level, so good runs with early low intermediate checkpoints were not removed.
- Removed empty top-level output directories left by the cleanup:
  - `outputs/outputs_tmp`
  - `outputs/XHandPasiniLightbulb_teacher`
  - `outputs/Dexh13HoraLightbulbDotpg_teacher`
- `docs/session_handoff_v2.md`
  - Added this cleanup entry.

### What was verified (commands + key outcomes)
- Bootstrap context:
  - Read `docs/session_handoff_v2.md` and `docs/stage_acceptance_summary.md`.
- Pre-cleanup scan:
  - `du -sh outputs`
    - Outcome: `7.1G`.
  - Python scan over `outputs/**/best_reward_*.pth`.
    - Outcome: `total_reward_named_runs=76`.
    - Candidate rule identified 51 directories with max best reward `< 1000`.
    - Approximate candidate size: `209.1 MiB`.
- Process check:
  - `pgrep -af 'train.py|dexh13_lightbulb|screwdriver_teacher|student_|vis_' || true`
  - Outcome: an active `smoothbulb_visualmatch_yneg018_torque03_idxlim015_s42_2h` training process was running; it was not a low-reward candidate and was not touched.
- Post-cleanup verification:
  - Python rescan over `outputs/**/best_reward_*.pth`.
    - Outcome: `reward_named_runs=25`, `remaining_low_reward_runs=0`.
  - `du -sh outputs`
    - Outcome: `6.9G`.
  - `git status --short`
    - Outcome: source/config dirty state remains the pre-existing working tree plus this handoff edit; output deletions are untracked filesystem cleanup.

### Local conclusion
- Low-reward reward-named run directories were cleaned successfully.
- The remaining reward-named output runs all have directory-level max best reward `>= 1000`.
- The active smoothbulb training output was preserved.

### Remaining blocked/risky
- This cleanup only used checkpoint filename rewards. Student `.ckpt` runs without `best_reward_*.pth` were not classified or deleted.
- Historical docs may still mention some deleted smoke/viewer/probe runs; this cleanup intentionally prioritized output disk hygiene over retaining every low-score artifact.
- The `smoothbulb_visualmatch_yneg018_torque03_idxlim015_s42_2h` run was still active during cleanup and should be checked after completion before any further pruning.

### Single recommended next step
- Let the active `smoothbulb_visualmatch_yneg018_torque03_idxlim015_s42_2h` training finish, then summarize its final TensorBoard metrics and decide whether to visualize it against `middle_yneg018_torque03_idxlim015_s42_2h`.

---

## v2-137 (2026-04-27) -- Smoothbulb Visual/Collision Match And 2h Result

### Target milestone/subgoal
- Fix the smoothbulb viewer mismatch reported by the user:
  - visual mesh and collision mesh did not match, making fingertip contact inspection misleading.
- Test whether the observed occasional large thumb slip was mainly due to the previous smoothbulb run being only about 30 minutes.

### What changed (files + behavior impact)
- `assets/screw/smoothbulb/0000_lightbulb.urdf`
  - Updated the `nut` visual meshes to match collision meshes exactly:
    - visual head now uses `../../lightbulb/smooth_head_collision.stl`
    - visual socket now uses `../../lightbulb/contact1.stl`
    - both visual origins/rpy match the collision origins/rpy: `xyz="0 0 0.06" rpy="0 -1.57079632679 0"`
  - Behavior impact:
    - Physics is unchanged from the previous smoothbulb collision probe.
    - Viewer inspection is now direct: visible bulb geometry is the same geometry PhysX collides with.

### What was verified (commands + key outcomes)
- Bootstrap context:
  - Read `docs/session_handoff_v2.md` and `docs/stage_acceptance_summary.md`.
- URDF validation:
  - Parsed `assets/screw/smoothbulb/0000_lightbulb.urdf` with `xml.etree.ElementTree`.
  - Confirmed visual and collision mesh/origin pairs for `nut` are aligned:
    - visual/collision `smooth_head_collision.stl`
    - visual/collision `contact1.stl`
- Patch hygiene:
  - `git diff --check -- assets/screw/smoothbulb/0000_lightbulb.urdf`
  - Outcome: pass.
- Freed GPU memory:
  - Found the user's active smoothbulb viewer using about `3.1GB` GPU memory.
  - Stopped that viewer container before the 8192-env long run.
- 2h smoothbulb visual-match training:
  - command:
    - `./docker-run-isaacgym.sh timeout 7200 scripts/run_with_cleanup.sh bash scripts/dexh13_lightbulb_teacher_middle_scalepos.sh 0 42 smoothbulb_visualmatch_yneg018_torque03_idxlim015_s42_2h True wandb_activate=True task.env.termination.log=True task.env.object.type=screw_smoothbulb 'task.env.asset.dofUpperLimits=[0.15,1.57,1.57,1.57,0.35,1.57,1.57,1.57,0.001,0.001,0.001,0.001,0.35,1.57,1.57,1.57]'`
  - W&B run:
    - `smoothbulb_visualmatch_yneg018_torque03_idxlim015_s42_2h_2026-04-27_06-14-37`
    - run id: `ytc7nx5p`
  - Outcome: timeout ended normally after about `Collect 105.5min + Train RL 13.7min ~= 119.2min`; no OOM or segmentation fault.
  - Best checkpoint:
    - `outputs/Dexh13HoraLightbulb_teacher_middle_scalepos/smoothbulb_visualmatch_yneg018_torque03_idxlim015_s42_2h/stage1_nn/best_reward_2849.55.pth`
  - TensorBoard event:
    - `outputs/Dexh13HoraLightbulb_teacher_middle_scalepos/smoothbulb_visualmatch_yneg018_torque03_idxlim015_s42_2h/stage1_tb/events.out.tfevents.1777270493.wbz-ubuntu22-pc`
- Key scalar metrics:
  - `episode_rewards/step`: last `2796.32`, max `2849.55`
  - `episode_lengths/step`: last `767.88`, max `770.04`
  - `screw/angular_velocity`: last `0.7845`, max `1.3215`
  - `screw/positive_vel_ratio`: last `0.6931`, max `0.8011`
  - `fingertip_tangent/positive_vel`: last `0.0877`, max `0.0894`
  - `fingertip_torque/signed_torque`: last `6.5941`, max `6.8753`
  - `fingertip_torque/reward`: last `0.8242`, max `0.8545`
  - `finger_torque/index/positive`: last `1.1887`, max `1.4664`
  - `finger_torque/middle/positive`: last `6.5931`, max `6.8753`
  - `finger_torque/thumb/positive`: last `0.0827`, max `2.2897`
  - `finger_torque/index/ratio_positive`: last `0.1469`, max `0.1764`
  - `finger_torque/middle/ratio_positive`: last `0.8455`, max `0.8595`
  - `finger_torque/thumb/ratio_positive`: last `0.0076`, max `0.1774`
  - `finger_tangent/index/positive_vel`: last `0.0154`, max `0.0300`
  - `finger_tangent/middle/positive_vel`: last `0.0872`, max `0.0894`
  - `finger_tangent/thumb/positive_vel`: last `0.0608`, max `0.1100`
  - `finger_motion/middle_joint_vel_abs`: last `1.7994`, max `1.9467`
  - `finger_motion/middle_joint0_sign_flip_rate`: last `0.0`, max `0.8413`
  - `finger_motion/index_middle_tip_dist`: last `0.0384`, max `0.0403`
  - `term/any_reset_frac`: last `0.000366`, max `0.002563`
  - `term/no_contact_frac`: last/max `0`
- Post-run cleanup:
  - `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits`
  - `pgrep -af 'train.py|dexh13_lightbulb_teacher_middle_scalepos|smoothbulb_visualmatch' || true`
  - Outcome: no training Python process remained; only the desktop/browser GPU process was listed.

### Local conclusion
- Issue 1 is fixed for the smoothbulb asset:
  - visual and collision now use the same mesh geometry for both bulb head and socket.
- Training time was a significant part of the observed instability:
  - 30m smoothbulb best: `2019.50`
  - 2h smoothbulb visual-match best: `2849.55`
  - previous 2h contactviz index-upper-0.15 best: `2947.48`
- Smoothbulb is now close to the original contactviz reward but still slightly lower.
- The best reward appeared around the middle of the 2h run; the last reward was lower (`2796.32`), so visualization should use the best checkpoint.
- Remaining thumb slip cannot be diagnosed from scalars alone:
  - thumb positive torque ratio remains tiny at the end (`0.0076`);
  - thumb tangent velocity is still nonzero (`0.0608`);
  - if the best checkpoint still has rare large thumb ejection, the likely cause is smooth surface slip after contact loss rather than insufficient PPO time alone.

### Remaining blocked/risky
- The final behavioral decision still needs viewer confirmation on the best checkpoint.
- If rare large thumb slip persists in the 2h best checkpoint, the next URDF direction should not be a fully smooth ellipsoid. Prefer local shoulder/chamfer smoothing or a slightly less smooth collision mesh that keeps some contact affordance.
- Middle joint velocity is higher than the contactviz 2h run (`1.80` last vs `0.96` last), so visual inspection should also check whether middle participation became more active or simply more jittery.

### Single recommended next step
- Visualize the 2h smoothbulb best checkpoint:
  - `./docker-run-isaacgym.sh bash scripts/vis_dexh13_lightbulb_teacher_middle_scalepos.sh 0 42 smoothbulb_visualmatch_yneg018_torque03_idxlim015_s42_2h task.env.object.type=screw_smoothbulb 'task.env.asset.dofUpperLimits=[0.15,1.57,1.57,1.57,0.35,1.57,1.57,1.57,0.001,0.001,0.001,0.001,0.35,1.57,1.57,1.57]'`
- Compare against:
  - `./docker-run-isaacgym.sh bash scripts/vis_dexh13_lightbulb_teacher_middle_scalepos.sh 0 42 middle_yneg018_torque03_idxlim015_s42_2h`
- If smoothbulb best clearly reduces thumb launch, keep it as a stability candidate; otherwise revert training to contactviz and test local collision smoothing near the bulb shoulder/socket transition.

---

## v2-138 (2026-04-27) -- XHandPasiniBulb Output Lookup

### Target milestone/subgoal
- Locate any existing teacher PPO checkpoint for the old `XHandPasiniBulb` task, which uses `assets/bulb/0000bulb.urdf`.

### What changed (files + behavior impact)
- `docs/session_handoff_v2.md`
  - Added this lookup entry only.
- No code, config, asset, checkpoint, or output file was changed.

### What was verified (commands + key outcomes)
- Bootstrap context:
  - Read `docs/session_handoff_v2.md` and `docs/stage_acceptance_summary.md`.
- Existing task wiring:
  - `configs/task/XHandPasiniBulb.yaml` exists.
  - `configs/train/XHandPasiniBulb.yaml` exists.
  - `scripts/pasini_bulb_teacher.sh` and `scripts/vis_pasini_bulb_teacher.sh` exist.
  - `dexscrew/tasks/xhand_pasini.py` maps `object.type: bulb` to `assets/bulb/0000bulb.urdf`.
- Output lookup:
  - Searched current `outputs/` and repository files for `XHandPasiniBulb`, `PasiniBulb`, `pasini_bulb`, and `XHandPasiniBulb_teacher`.
  - Outcome: no current `outputs/XHandPasiniBulb_teacher/.../stage1_nn/best_reward_*.pth` checkpoint was found.
  - Existing lightbulb outputs are for `XHandHoraLightbulb` or DexH13 lightbulb variants, not `XHandPasiniBulb`.

### Local conclusion
- The old task/config path exists and is wired correctly.
- No usable teacher PPO checkpoint for `XHandPasiniBulb` is currently present under `outputs/`.
- If such a checkpoint existed historically, it is not in the current output tree.

### Remaining blocked/risky
- There is no current `XHandPasiniBulb_teacher/<cache>` run to visualize.
- Do not confuse `XHandPasiniBulb` (`object.type: bulb`, `assets/bulb/0000bulb.urdf`) with `XHandPasiniLightbulb` (`object.type: screw_lightbulb`, `assets/screw/lightbulb/*.urdf`).

### Single recommended next step
- If this path is needed again, rerun a bounded teacher probe with `scripts/pasini_bulb_teacher.sh`, then visualize it with `scripts/vis_pasini_bulb_teacher.sh`.

---

## v2-139 (2026-04-27) -- XHandPasiniBulb Middle/Ring Frozen Task YAML

### Target milestone/subgoal
- Create a new Pasini bulb task YAML based on the old `XHandPasiniBulb` config, but aligned with the latest lightbulb middle-scale-position probe style:
  - ring action mask and near-zero DOF limits;
  - middle gets the same near-zero DOF limits as ring;
  - middle also gets action-masked.

### What changed (files + behavior impact)
- `configs/task/XHandPasiniBulbMiddleScalePos.yaml`
  - New task config with `name: XHandPasiniBulb`, so it reuses the existing `XHandPasini` task class.
  - Keeps the old `XHandPasiniBulb` object/config surface (`object.type: bulb`, `assets/bulb/0000bulb.urdf` through task code).
  - Adds:
    - `env.apply_action_mask: True`
    - `env.action_mask_indices: [4, 5, 6, 7, 8, 9, 10, 11]`
    - middle/ring near-zero `env.asset.dofLowerLimits` and `env.asset.dofUpperLimits`
    - `env.customInitDofPos` with middle/ring reset pose at zero so the frozen fingers start inside their new limits.
- `configs/train/XHandPasiniBulbMiddleScalePos.yaml`
  - New matching train config copied from `XHandPasiniBulb`, allowing `task=XHandPasiniBulbMiddleScalePos` to compose without explicitly overriding `train=`.
- `dexscrew/tasks/xhand_pasini.py`
  - Added YAML-driven support for:
    - `env.apply_action_mask`
    - `env.action_mask_indices`
    - `env.asset.dofLowerLimits`
    - `env.asset.dofUpperLimits`
    - optional `env.asset.dofEffortLimits`
    - optional `env.asset.dofVelocityLimits`
  - Default behavior remains compatible with the old Pasini bulb path: if no custom mask is given, non-screwdriver poses still mask ring actions `[8:12]`.

### What was verified (commands + key outcomes)
- Syntax:
  - `PYTHONDONTWRITEBYTECODE=1 python - <<'PY' ... compile('dexscrew/tasks/xhand_pasini.py', 'exec') ... PY`
  - Outcome: `xhand_pasini_syntax_ok`.
- Patch hygiene:
  - `git diff --check -- dexscrew/tasks/xhand_pasini.py configs/task/XHandPasiniBulbMiddleScalePos.yaml configs/train/XHandPasiniBulbMiddleScalePos.yaml`
  - Outcome: pass.
- Docker Hydra compose:
  - `./docker-run-isaacgym.sh bash -lc "python - <<'PY' ... compose(task=XHandPasiniBulbMiddleScalePos, num_envs=4) ... PY"`
  - Outcome:
    - `task_name= XHandPasiniBulb`
    - `train_algo= PPO`
    - `customInit_middle= [0.0, 0.0, 0.0, 0.0]`
    - `customInit_ring= [0.0, 0.0, 0.0, 0.0]`
    - `action_mask_indices= [4, 5, 6, 7, 8, 9, 10, 11]`
    - middle/ring lower and upper limits match.
- Docker runtime smoke:
  - Built a 2-env headless `XHandPasiniBulbMiddleScalePos` env and called `reset()`.
  - Outcome:
    - `env_smoke_ok`
    - `apply_action_mask=True`
    - `custom_action_mask_indices=[4, 5, 6, 7, 8, 9, 10, 11]`
    - middle/ring lower tensors are `[-0.001, 0.0, 0.0, 0.0]`
    - middle/ring upper tensors are `[0.001, 0.001, 0.001, 0.001]`
    - reset middle/ring positions stay near the new zero limits.

### Local conclusion
- The new Pasini bulb task YAML is ready to train/evaluate as a fresh config.
- Because the policy action space remains 16-D, this is not compatible with old policies unless they are evaluated under the same new mask/limit semantics intentionally.

### Remaining blocked/risky
- No teacher training has been run for this new task yet.
- The old `scripts/pasini_bulb_teacher.sh` still points at `task=XHandPasiniBulb`; use a direct Hydra command or add a dedicated script before longer runs.

### Single recommended next step
- Run a bounded smoke teacher probe for `task=XHandPasiniBulbMiddleScalePos`, then visualize only if reward/contact behavior is promising.

---

## v2-140 (2026-04-27) -- XHandPasiniBulbMiddleScalePos 30m Teacher Probe

### Target milestone/subgoal
- Train the new `XHandPasiniBulbMiddleScalePos` teacher PPO probe for about 30 minutes using the default 8192-env scale, then check whether the environment/config is trainable enough to visualize.

### What changed (files + behavior impact)
- New output artifacts only:
  - `outputs/XHandPasiniBulbMiddleScalePos_teacher/middle_ring_frozen_s42_30m/config_042708_21e8eff.yaml`
  - `outputs/XHandPasiniBulbMiddleScalePos_teacher/middle_ring_frozen_s42_30m/gitdiff.patch`
  - `outputs/XHandPasiniBulbMiddleScalePos_teacher/middle_ring_frozen_s42_30m/stage1_nn/best_reward_592.93.pth`
  - `outputs/XHandPasiniBulbMiddleScalePos_teacher/middle_ring_frozen_s42_30m/stage1_tb/events.out.tfevents.1777280293.wbz-ubuntu22-pc`
  - `outputs/XHandPasiniBulbMiddleScalePos_teacher/middle_ring_frozen_s42_30m/train_30m.log`
- `docs/session_handoff_v2.md`
  - Added this execution entry.
- No source/config changes were made during this training run.

### What was verified (commands + key outcomes)
- Pre-run checks:
  - Read latest `docs/session_handoff_v2.md`.
  - Inspected current `configs/task/XHandPasiniBulbMiddleScalePos.yaml`.
  - Checked GPU/process state:
    - no active `train.py` process;
    - only browser GPU process present.
- Training command:
  - `./docker-run-isaacgym.sh bash -lc 'set -o pipefail; CACHE=middle_ring_frozen_s42_30m; OUT=outputs/XHandPasiniBulbMiddleScalePos_teacher/${CACHE}; mkdir -p "${OUT}"; timeout 1800 python train.py task=XHandPasiniBulbMiddleScalePos headless=True seed=42 experiment=rl train.algo=PPO wandb_activate=True train.ppo.output_name=XHandPasiniBulbMiddleScalePos_teacher/${CACHE} 2>&1 | tee "${OUT}/train_30m.log"'`
  - Outcome: command exited with code `124` from the intended `timeout 1800`, after about `Collect 23.6min + Train RL 5.9min ~= 29.5min`.
- W&B:
  - Run name: `middle_ring_frozen_s42_30m_2026-04-27_08-57-59`
  - Run id: `52msq4m6`
- Output/checkpoint check:
  - Best checkpoint: `best_reward_592.93.pth`.
  - Log and TensorBoard event were written.
- TensorBoard scalar summary:
  - `episode_rewards/step`: last `591.86`, max `592.93`
  - `episode_lengths/step`: last/max `592.82`
  - `info/best_reward`: last/max `592.93`
  - `info/best_reward_step`: `51707904`
  - `info/best_reward_elapsed_min`: `28.75`
  - `screw/angular_velocity`: last `0.01946`, max `0.20146`
  - `screw/angular_position`: last `3.0550`, max `3.1694`
  - `screw/positive_vel_ratio`: last `0.4902`, max `0.5211`
  - `rotation_reward`: last `0.1450`, max `0.1917`
  - `step_all_reward`: last `1.0863`, max `1.3345`
  - `info/kl`: last `0.02286`, max `0.18290`
  - `info/last_lr`: last `0.000293`, max `0.002222`
- Post-run checks:
  - `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits`
  - `pgrep -af 'train.py|XHandPasiniBulbMiddleScalePos|middle_ring_frozen_s42_30m' || true`
  - Outcome: no training Python process remained; only browser GPU process was listed.

### Local conclusion
- The new `XHandPasiniBulbMiddleScalePos` config is trainable at 8192 env scale.
- Early PPO was very unstable:
  - reward first climbed to about `69.58`,
  - fell back to the `20-30` range,
  - then escaped and rose steadily to `592.93`.
- The 30m result is not strong compared with mature teacher runs, but it is good enough for a quick viewer check of whether the learned behavior and frozen middle/ring mechanics look sane.
- Scalar behavior suggests low rotation speed/positive velocity ratio, so the policy may be making slow or partial progress rather than a strong twisting behavior.

### Remaining blocked/risky
- The run ended by timeout, not by a clean trainer shutdown; this is expected for the 30m probe, but W&B final sync may be partial.
- No visual check has been done yet.
- There is no dedicated `scripts/vis_*` wrapper for `XHandPasiniBulbMiddleScalePos`; use a direct `train.py test=True` command.

### Single recommended next step
- Visualize the best checkpoint:
  - `./docker-run-isaacgym.sh python train.py task=XHandPasiniBulbMiddleScalePos headless=False seed=42 sim_device=cuda:0 rl_device=cuda:0 graphics_device_id=7 task.env.numEnvs=6 test=True train.algo=PPO wandb_activate=False train.ppo.output_name=XHandPasiniBulbMiddleScalePos_teacher/middle_ring_frozen_s42_30m "checkpoint=outputs/XHandPasiniBulbMiddleScalePos_teacher/middle_ring_frozen_s42_30m/stage1_nn/best_reward_*.pth"`

---

## v2-141 (2026-04-27) -- DexH13 Lightbulb Thesis Two-Finger Config

### Target milestone/subgoal
- Stop the middle-finger collaboration branch for thesis closure.
- Create a clean DexH13 lightbulb teacher PPO config that uses only index + thumb as active fingers, while middle and ring are frozen near zero.

### What changed (files + behavior impact)
- `configs/task/Dexh13HoraLightbulbThesisTwoFinger.yaml`
  - New task config based on the current scaled/positioned DexH13 lightbulb setup.
  - Active actions/DOFs are index `0:3` and thumb `12:15`.
  - Middle `4:7` and ring `8:11` are action-masked.
  - Middle/ring init pose values are all `0.0`.
  - Middle/ring DOF limits are locked near zero:
    - lower `[-0.001, 0.0, 0.0, 0.0]`
    - upper `[0.001, 0.001, 0.001, 0.001]`
  - Finger contact, two-finger gate, fingertip tangent reward, and fingertip torque reward now target index + thumb instead of middle/three-finger collaboration.
- `configs/train/Dexh13HoraLightbulbThesisTwoFinger.yaml`
  - Added matching PPO train config.
- `scripts/dexh13_lightbulb_teacher_thesis_twofinger.sh`
  - Added training wrapper writing to `outputs/Dexh13HoraLightbulb_teacher_thesis_twofinger/<cache>`.
- `scripts/vis_dexh13_lightbulb_teacher_thesis_twofinger.sh`
  - Added visualization wrapper loading the best checkpoint from the thesis two-finger output path.

### What was verified (commands + key outcomes)
- Script syntax:
  - `bash -n scripts/dexh13_lightbulb_teacher_thesis_twofinger.sh`
  - `bash -n scripts/vis_dexh13_lightbulb_teacher_thesis_twofinger.sh`
  - Outcome: pass.
- Config structure in Docker:
  - `./docker-run-isaacgym.sh python -c "... OmegaConf.load('configs/task/Dexh13HoraLightbulbThesisTwoFinger.yaml') ..."`
  - Outcome: `docker_omegaconf_thesis_twofinger_ok`.
- Hydra composition:
  - `./docker-run-isaacgym.sh python train.py task=Dexh13HoraLightbulbThesisTwoFinger train.algo=PPO num_envs=1 headless=True --cfg job`
  - Outcome: task/train config composes successfully and includes `eval_cache_name: thesis_twofinger`, action mask, index-only gate, and locked DOF limits.
- Patch hygiene:
  - `git diff --check -- configs/task/Dexh13HoraLightbulbThesisTwoFinger.yaml configs/train/Dexh13HoraLightbulbThesisTwoFinger.yaml scripts/dexh13_lightbulb_teacher_thesis_twofinger.sh scripts/vis_dexh13_lightbulb_teacher_thesis_twofinger.sh`
  - Outcome: pass.

### Local conclusion
- The new thesis two-finger configuration is ready for a fresh teacher PPO run.
- This config intentionally abandons middle participation; middle/ring should remain physically near zero and policy-inactive.
- Because the action space is still 16-D but masked/limited, policies trained under previous middle configs should not be compared as if they used the same action semantics.

### Remaining blocked/risky
- No PPO training has been run yet for this new thesis two-finger config.
- Thumb ejection/stability risk may still exist because this config does not yet alter the global thumb pose penalty mask or controller gains.
- If final visualization still shows thumb instability, the next thesis-safe fix should be controller damping/torque reduction or a small thumb stability penalty, not reintroducing middle collaboration.

### Single recommended next step
- Run a 30-minute thesis two-finger teacher probe:
  - `./docker-run-isaacgym.sh timeout 1800 scripts/run_with_cleanup.sh bash scripts/dexh13_lightbulb_teacher_thesis_twofinger.sh 0 42 thesis_twofinger_s42_30m True wandb_activate=True task.env.termination.log=True`

---

## v2-142 (2026-04-27) -- Thesis Thumb Pose Penalty

### Target milestone/subgoal
- Add the previously identified thumb posture constraint for the final DexH13 lightbulb thesis two-finger branch.
- Reduce the risk that the policy learns a high-force thumb shortcut that rotates the bulb briefly, loses friction, and sends the thumb far away from its comfortable pose.

### What changed (files + behavior impact)
- `dexscrew/tasks/xhand_hora.py`
  - Replaced the hardcoded thumb pose-diff mask behavior with a configurable value:
    - `env.pose_diff_penalty.thumb_weight`
    - default remains `0.0`, preserving legacy Hora configs unless they opt in.
  - Added thumb pose deviation diagnostics:
    - `pose_diff_penalty/thumb_raw`
    - `pose_diff_penalty/thumb_weighted`
- `configs/task/Dexh13HoraLightbulbThesisTwoFinger.yaml`
  - Enabled the final branch thumb posture penalty:
    - `env.pose_diff_penalty.thumb_weight: 1.0`

### What was verified (commands + key outcomes)
- Syntax:
  - `PYTHONDONTWRITEBYTECODE=1 python - <<'PY' ... compile('dexscrew/tasks/xhand_hora.py', 'exec') ... PY`
  - Outcome: `xhand_hora_compile_ok`.
- Patch hygiene:
  - `git diff --check -- dexscrew/tasks/xhand_hora.py configs/task/Dexh13HoraLightbulbThesisTwoFinger.yaml`
  - Outcome: pass.
- Hydra composition:
  - `./docker-run-isaacgym.sh python train.py task=Dexh13HoraLightbulbThesisTwoFinger train.algo=PPO num_envs=1 headless=True --cfg job`
  - Outcome: composed config includes `pose_diff_penalty.thumb_weight: 1.0`.
- Runtime smoke:
  - `./docker-run-isaacgym.sh timeout 180 python train.py task=Dexh13HoraLightbulbThesisTwoFinger headless=True seed=42 num_envs=4 train.algo=PPO train.ppo.minibatch_size=12 train.ppo.max_agent_steps=48 wandb_activate=False train.ppo.output_name=Dexh13HoraLightbulb_teacher_thesis_twofinger/smoke_posepenalty_tmp`
  - Outcome: environment built, reward path executed, and trainer ended with `max steps achieved`.
  - Temporary smoke output was removed afterward.

### Local conclusion
- The final thesis two-finger task now penalizes thumb deviation from the configured initial pose.
- This is the right first stabilization step before lowering controller authority, because it targets the exact bad behavior while keeping the learned two-finger twisting objective intact.
- Smoothbulb vs original contactviz decision:
  - current evidence favors original `screw_contactviz` for the final branch.
  - previous 2h contactviz best: `2947.48`.
  - previous 2h smooth visual-match best: `2849.55`.
  - smooth helped some small contact artifacts but still showed larger occasional thumb slip; the original faceted/contactviz geometry provides more contact affordance and better reward.

### Remaining blocked/risky
- Thumb pose penalty may reduce reward at first; this is expected if it removes the old high-force shortcut.
- If thumb still ejects after training this version, the next thesis-safe fixes are:
  - lower `controller.torque_limit` and `action_scale`;
  - increase `controller.dgain`;
  - add a small thumb tip velocity / contact-continuity penalty.
- Do not switch the final thesis branch to smoothbulb without an apples-to-apples two-finger run showing better visual stability and comparable reward.

### Single recommended next step
- Run the 30-minute thesis two-finger probe with thumb pose penalty enabled:
  - `./docker-run-isaacgym.sh timeout 1800 scripts/run_with_cleanup.sh bash scripts/dexh13_lightbulb_teacher_thesis_twofinger.sh 0 42 thesis_twofinger_posepen_s42_30m True wandb_activate=True task.env.termination.log=True`

---

## v2-143 (2026-04-27) -- Thesis Thumb Pose Penalty Relaxed

### Target milestone/subgoal
- Relax the thesis two-finger thumb pose penalty after reconsidering that the original Hora path intentionally left thumb unpenalized because thumb motion amplitude is naturally large.

### What changed (files + behavior impact)
- `configs/task/Dexh13HoraLightbulbThesisTwoFinger.yaml`
  - Changed `env.pose_diff_penalty.thumb_weight` from `1.0` to `0.1`.
  - Behavior impact:
    - index remains fully covered by the base pose-diff penalty;
    - thumb now receives only a light pose-diff penalty, intended to discourage extreme ejection without suppressing normal large thumb motion.

### What was verified (commands + key outcomes)
- Patch hygiene:
  - `git diff --check -- configs/task/Dexh13HoraLightbulbThesisTwoFinger.yaml`
  - Outcome: pass.
- Hydra composition:
  - `./docker-run-isaacgym.sh python train.py task=Dexh13HoraLightbulbThesisTwoFinger train.algo=PPO num_envs=1 headless=True --cfg job`
  - Outcome: composed config includes `pose_diff_penalty.thumb_weight: 0.1`.

### Local conclusion
- `thumb_weight=0.1` is a better first thesis setting than `1.0`: it keeps a stabilizing signal while respecting the thumb's larger natural workspace.
- If thumb still ejects after short training, the next adjustment should be controller-side damping/authority rather than immediately increasing thumb pose penalty sharply.

### Remaining blocked/risky
- No training run has been completed with the relaxed `0.1` value yet.
- The init pose should be visually checked before the next 30-minute run.

### Single recommended next step
- Open a viewer training run to inspect the thesis two-finger init pose:
  - `./docker-run-isaacgym.sh bash scripts/dexh13_lightbulb_teacher_thesis_twofinger.sh 0 42 thesis_twofinger_initpose_vis False wandb_activate=False task.env.randomization.randomizePDGains=False task.env.randomization.action_noise_e_scale=0.0 task.env.randomization.action_noise_t_scale=0.0 task.env.randomization.obs_noise_e_scale=0.0 task.env.randomization.obs_noise_t_scale=0.0 task.env.randomization.noisy_rpy_scale=0.0 task.env.randomization.noisy_pos_scale=0.0 task.env.forceScale=0.0 task.env.randomForceProbScalar=0.0`

---

## v2-144 (2026-04-27) -- Thesis Contact Material Stabilization Probe

### Target milestone/subgoal
- Improve final DexH13 lightbulb thesis two-finger stability after visualization showed occasional thumb slip/ejection.
- Test a contact/material-side fix before changing reward more aggressively or reducing controller authority.

### What changed (files + behavior impact)
- `dexscrew/tasks/xhand_hora.py`
  - Made object restitution randomization configurable through:
    - `env.randomization.randomizeRestitutionLower`
    - `env.randomization.randomizeRestitutionUpper`
  - Default remains `[0.0, 1.0]`, preserving legacy behavior for configs that do not opt in.
- `configs/task/Dexh13HoraLightbulbThesisTwoFinger.yaml`
  - Increased object friction randomization range to `[1.0, 5.0]`.
  - Restricted restitution to `[0.0, 0.05]` to reduce bounce-like contact response.
  - Set `max_depenetration_velocity: 10.0` to reduce hard contact correction impulses.
  - Kept `pose_diff_penalty.thumb_weight: 0.1`.

### What was verified (commands + key outcomes)
- Syntax:
  - `PYTHONDONTWRITEBYTECODE=1 python - <<'PY' ... compile('dexscrew/tasks/xhand_hora.py', 'exec') ... PY`
  - Outcome: `xhand_hora_compile_ok`.
- Patch hygiene:
  - `git diff --check -- dexscrew/tasks/xhand_hora.py configs/task/Dexh13HoraLightbulbThesisTwoFinger.yaml`
  - Outcome: pass.
- Hydra composition:
  - `./docker-run-isaacgym.sh python train.py task=Dexh13HoraLightbulbThesisTwoFinger train.algo=PPO num_envs=1 headless=True --cfg job`
  - Outcome: composed config includes `thumb_weight: 0.1`, friction `[1.0, 5.0]`, restitution `[0.0, 0.05]`, and `max_depenetration_velocity: 10.0`.
- Runtime smoke:
  - `./docker-run-isaacgym.sh timeout 180 python train.py task=Dexh13HoraLightbulbThesisTwoFinger headless=True seed=42 num_envs=4 train.algo=PPO train.ppo.minibatch_size=12 train.ppo.max_agent_steps=48 wandb_activate=False train.ppo.output_name=Dexh13HoraLightbulb_teacher_thesis_twofinger/smoke_matstable_tmp`
  - Outcome: environment built and trainer ended with `max steps achieved`; temporary smoke output was removed.
- 30-minute PPO probe:
  - `./docker-run-isaacgym.sh timeout 1800 scripts/run_with_cleanup.sh bash scripts/dexh13_lightbulb_teacher_thesis_twofinger.sh 0 42 thesis_twofinger_matstable_s42_30m True wandb_activate=True task.env.termination.log=True`
  - Outcome: timeout ended cleanly, no training process remained afterward.
  - Checkpoint: `outputs/Dexh13HoraLightbulb_teacher_thesis_twofinger/thesis_twofinger_matstable_s42_30m/stage1_nn/best_reward_2845.32.pth`.

### Training metrics
- `episode_rewards/step`: last/max `2844.83`.
- `episode_lengths/step`: last/max `722.43`.
- `screw/angular_velocity`: last `0.8560`, max `2.0802`.
- `screw/positive_vel_ratio`: last `0.8412`, max `0.9104`.
- `two_finger/gate`: last `0.99898`.
- `two_finger/thumb_contact_w`: last `0.99926`.
- `two_finger/other_contact_w`: last `0.99972`.
- `term/any_reset_frac`: last `0.000366`, max `0.002319`.
- `term/no_contact_frac`: last/max `0`.
- `pose_diff_penalty/thumb_raw`: last `0.0441`, max `0.0517`.
- `fingertip_torque/positive_torque`: last `1.1963`, max `1.5904`.

### Local conclusion
- The contact/material-stable version kept reward essentially equal to the previous relaxed thumb-pose run (`2845.32` vs about `2847.96`), so it did not break the learned two-finger behavior.
- Contact diagnostics remained very strong and no-contact resets stayed at zero.
- This is a good candidate to visualize immediately; scalar logs alone cannot prove whether the rare thumb ejection improved.

### Remaining blocked/risky
- Visual confirmation is still required. The key question is whether the lower restitution and depenetration limit reduced the large thumb slip/ejection events.
- If ejection persists, the next thesis-safe step should be controller-side damping/authority tuning, for example testing lower `torque_limit` or higher `dgain`, rather than switching back to middle collaboration.

### Single recommended next step
- Visualize the material-stable 30-minute checkpoint:
  - `./docker-run-isaacgym.sh bash scripts/vis_dexh13_lightbulb_teacher_thesis_twofinger.sh 0 42 thesis_twofinger_matstable_s42_30m`

---

## v2-145 (2026-04-27) -- Thesis Smooth Geometry A/B Probe

### Target milestone/subgoal
- Test whether replacing the current faceted/contactviz lightbulb collision with the smooth visual=collision bulb reduces thumb slip/ejection without changing the thesis two-finger reward, controller, or material stabilization settings.

### What changed (files + behavior impact)
- New output artifacts only:
  - `outputs/Dexh13HoraLightbulb_teacher_thesis_twofinger/thesis_twofinger_smooth_matstable_s42_30m/config_042715_21e8eff.yaml`
  - `outputs/Dexh13HoraLightbulb_teacher_thesis_twofinger/thesis_twofinger_smooth_matstable_s42_30m/gitdiff.patch`
  - `outputs/Dexh13HoraLightbulb_teacher_thesis_twofinger/thesis_twofinger_smooth_matstable_s42_30m/stage1_nn/best_reward_2067.97.pth`
  - `outputs/Dexh13HoraLightbulb_teacher_thesis_twofinger/thesis_twofinger_smooth_matstable_s42_30m/stage1_tb/events.out.tfevents.1777305240.wbz-ubuntu22-pc`
- No source/config edits were made during this A/B run.
- The run used override `task.env.object.type=screw_smoothbulb`, keeping the current thesis two-finger config otherwise unchanged.

### What was verified (commands + key outcomes)
- Pre-run bootstrap:
  - Read `docs/session_handoff_v2.md` and `docs/stage_acceptance_summary.md`.
  - Confirmed `assets/screw/smoothbulb/0000_lightbulb.urdf`, `assets/lightbulb/smooth_head_collision.stl`, and thesis train/vis scripts exist.
  - Confirmed the output cache did not already exist.
  - Confirmed no active training process was running.
- Training command:
  - `./docker-run-isaacgym.sh timeout 1800 scripts/run_with_cleanup.sh bash scripts/dexh13_lightbulb_teacher_thesis_twofinger.sh 0 42 thesis_twofinger_smooth_matstable_s42_30m True wandb_activate=True task.env.termination.log=True task.env.object.type=screw_smoothbulb`
  - Outcome: command exited with code `124` from the intended 30-minute timeout; cleanup completed and no training process remained.
- W&B:
  - Run id: `87py0g6g`
  - Run name: `thesis_twofinger_smooth_matstable_s42_30m_2026-04-27_15-53-41`
- Post-run checks:
  - Checkpoint exists: `best_reward_2067.97.pth`.
  - GPU process check showed no training Python process, only browser/ToDesk processes.

### Training metrics
- `episode_rewards/step`: TensorBoard last/max `2065.62`; checkpoint best `2067.97`.
- `episode_lengths/step`: last/max `670.99`.
- `screw/angular_velocity`: last `0.7393`, max `1.5490`.
- `screw/positive_vel_ratio`: last `0.8147`, max `0.8635`.
- `two_finger/gate`: last `0.99766`, max `0.99907`.
- `two_finger/thumb_contact_w`: last `0.99768`.
- `two_finger/other_contact_w`: last `0.99997`.
- `term/any_reset_frac`: last `0.000977`, max `0.003052`.
- `term/no_contact_frac`: last/max `0`.
- `pose_diff_penalty/thumb_raw`: last `0.01036`, max `0.02446`.
- `fingertip_torque/positive_torque`: last `1.1306`, max `1.7340`.

### Local conclusion
- Smooth geometry is trainable and stable, but it does not meet the short-run scalar acceptance threshold:
  - target `episode_rewards/step >= 2700`;
  - observed checkpoint best `2067.97`.
- Compared with `thesis_twofinger_matstable_s42_30m` on contactviz:
  - reward is much lower (`2067.97` vs `2845.32`);
  - angular velocity is lower (`0.7393` last vs `0.8560` last; `1.5490` max vs `2.0802` max);
  - two-finger contact remains strong and no-contact reset remains zero.
- Current interpretation:
  - the smoother bulb likely removes some edge/patch discontinuity, but also removes useful contact affordance for fast twisting;
  - do not switch the thesis final default to `screw_smoothbulb` based on scalar evidence alone.

### Remaining blocked/risky
- Visual confirmation is still needed. The scalar result says smooth is slower, but only viewer inspection can answer whether thumb ejection is actually reduced.
- If smooth visually removes thumb ejection but is too slow, the next experiment should combine the better visual/contact geometry idea with controller tuning rather than simply replacing the final asset.

### Single recommended next step
- Visualize the smooth A/B checkpoint:
  - `./docker-run-isaacgym.sh bash scripts/vis_dexh13_lightbulb_teacher_thesis_twofinger.sh 0 42 thesis_twofinger_smooth_matstable_s42_30m task.env.object.type=screw_smoothbulb`
- Then compare against current contactviz candidate:
  - `./docker-run-isaacgym.sh bash scripts/vis_dexh13_lightbulb_teacher_thesis_twofinger.sh 0 42 thesis_twofinger_matstable_s42_30m`

---

## v2-146 (2026-04-28) -- Thesis Two-Finger Thumb Slip Diagnostics And Controller/Pose Ablation

### Target milestone/subgoal
- Keep the thesis final path on the original contactviz bulb (`task.env.object.type=screw_contactviz`) and identify the main lever behind occasional thumb slip/ejection.
- Compare torque limit, damping, and thumb pose regularization without changing the reward formula or URDF.

### What changed (files + behavior impact)
- `dexscrew/tasks/xhand_hora.py`
  - Added TensorBoard/W&B-only thumb slip diagnostics:
    - `thumb_slip/contact_drop_frac`
    - `thumb_slip/far_frac`
    - `thumb_slip/active_detach_frac`
    - `thumb_slip/active_far_frac`
    - `thumb_slip/ejection_frac`
    - `thumb_slip/tip_speed_mean`
    - `thumb_slip/tip_speed_p95`
    - `thumb_slip/joint_vel_abs_mean`
    - `thumb_slip/joint_vel_abs_p95`
    - `thumb_slip/dist_p95`
    - `thumb_slip/contact_w_p05`
    - `thumb_slip/active_screw_frac`
    - `thumb_slip/score`
  - These metrics are diagnostic only and do not participate in reward.
- `configs/task/Dexh13HoraLightbulbThesisTwoFinger.yaml`
  - Added `env.thumb_slip_diagnostics` thresholds:
    - `contact_drop_w: 0.2`
    - `far_dist: 0.09`
    - `high_tip_speed: 0.25`
    - `active_screw_vel: 0.2`
- `scripts/summarize_thesis_twofinger_runs.py`
  - Added a small TensorBoard summary helper for thesis two-finger ablations.

### What was verified (commands + key outcomes)
- Syntax:
  - `PYTHONDONTWRITEBYTECODE=1 python - <<'PY' ... compile('dexscrew/tasks/xhand_hora.py', 'exec') ... PY`
  - Outcome: `xhand_hora_compile_ok`.
- Patch hygiene:
  - `git diff --check -- dexscrew/tasks/xhand_hora.py configs/task/Dexh13HoraLightbulbThesisTwoFinger.yaml`
  - Outcome: pass.
  - `PYTHONDONTWRITEBYTECODE=1 python -m py_compile scripts/summarize_thesis_twofinger_runs.py`
  - Outcome: pass.
- Hydra composition:
  - Confirmed `env.thumb_slip_diagnostics`, default `task.env.object.type=screw_contactviz`, default `torque_limit: 300.0`, and default `dgain: 0.01`.
- Runtime smoke:
  - `./docker-run-isaacgym.sh timeout 300 python train.py task=Dexh13HoraLightbulbThesisTwoFinger headless=True seed=42 num_envs=64 train.algo=PPO train.ppo.minibatch_size=128 train.ppo.max_agent_steps=2048 wandb_activate=False train.ppo.output_name=Dexh13HoraLightbulb_teacher_thesis_twofinger/smoke_thumbslip_diag_tmp task.env.termination.log=True`
  - Outcome: smoke completed and TensorBoard contained the new `thumb_slip/*` tags.
- Process hygiene:
  - After each 1-hour run, checkpoint existence was checked.
  - `pgrep` showed no residual `train.py` process.
  - `nvidia-smi` showed no residual training Python process.

### Training metrics
| run | best ckpt | reward | vel | gate | slip score | active far | active detach | ejection | tip p95 | dist p95 | contact p05 | reset | no contact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `thesis_twofinger_tlim200_diag_s42_1h` | 3488.70 | 3488.70 | 0.9002 | 0.9990 | 0.00470 | 0.00300 | 0.00000 | 0.00171 | 0.4321 | 0.0638 | 1.0000 | 0.00037 | 0.00000 |
| `thesis_twofinger_tlim180_diag_s42_1h` | 3525.85 | 3525.67 | 1.1236 | 0.9990 | 0.00544 | 0.00262 | 0.00000 | 0.00171 | 0.4553 | 0.0636 | 1.0000 | 0.00037 | 0.00000 |
| `thesis_twofinger_dgain015_diag_s42_1h` | 3695.92 | 3695.76 | 1.1292 | 0.9985 | 0.00626 | 0.00569 | 0.00000 | 0.00366 | 0.3933 | 0.0624 | 1.0000 | 0.00085 | 0.00000 |
| `thesis_twofinger_thumbpose02_diag_s42_1h` | 3171.29 | 3171.25 | 0.6086 | 0.9989 | 0.00183 | 0.00134 | 0.00000 | 0.00049 | 0.2725 | 0.0625 | 1.0000 | 0.00012 | 0.00000 |
| `thesis_twofinger_thumbpose02_tlim200_diag_s42_1h` | 3171.29 | 3171.25 | 0.6086 | 0.9989 | 0.00183 | 0.00134 | 0.00000 | 0.00049 | 0.2725 | 0.0625 | 1.0000 | 0.00012 | 0.00000 |
| `thesis_twofinger_thumbpose02_dgain015_diag_s42_1h` | 3558.70 | 3555.97 | 0.8850 | 0.9988 | 0.00425 | 0.00482 | 0.00000 | 0.00269 | 0.3500 | 0.0620 | 1.0000 | 0.00037 | 0.00000 |

### Local conclusion
- The strongest single lever for reducing thumb ejection is `pose_diff_penalty.thumb_weight=0.2`.
  - It reduced slip score from `0.00470-0.00626` down to `0.00183`.
  - It reduced ejection from `0.00171-0.00366` down to `0.00049`.
  - It also reduced thumb tip speed p95 to `0.2725`, but slowed screw angular velocity to `0.6086`.
- `dgain=0.015` alone produced the highest reward (`3695.92`) and high angular velocity, but it also had the worst slip score and ejection rate.
- `torque_limit=180/200` did not solve the slip issue by itself.
- `thumb_weight=0.2 + torque_limit=200` matched pure `thumb_weight=0.2` almost exactly, suggesting `torque_limit=200` is not binding in this trained policy.
- `thumb_weight=0.2 + dgain=0.015` is the best balanced follow-up candidate:
  - reward improved to `3558.70`;
  - angular velocity recovered to `0.8850`;
  - slip score stayed below the raw controller-only runs, but is not as clean as pure `thumb_weight=0.2`.

### Remaining blocked/risky
- Visual confirmation is still required. The scalar winner by reward (`dgain015`) and the scalar winner by slip (`thumbpose02`) are different.
- The balanced candidate (`thumbpose02_dgain015`) may still show visible late-rotation thumb ejection because its `active_far` and `ejection` are higher than pure `thumbpose02`.
- The current diagnostics show thumb contact weight p05 remains `1.0`; the observed visual "脱手" is better captured by distance/speed/ejection than by contact-drop alone.

### Single recommended next step
- Visualize the balanced candidate first:
  - `./docker-run-isaacgym.sh bash scripts/vis_dexh13_lightbulb_teacher_thesis_twofinger.sh 0 42 thesis_twofinger_thumbpose02_dgain015_diag_s42_1h`
- Then visualize the lowest-slip stable candidate:
  - `./docker-run-isaacgym.sh bash scripts/vis_dexh13_lightbulb_teacher_thesis_twofinger.sh 0 42 thesis_twofinger_thumbpose02_diag_s42_1h`
- If the balanced candidate still visibly throws the thumb out, use pure `thumbpose02` as thesis-stable default and consider a smaller damping step (`dgain=0.0125`) rather than further reducing torque limit.

---

## v2-147 (2026-04-28) -- Thesis Two-Finger Teacher PPO Selection For Student Distillation

### Target milestone/subgoal
- Lock the final thesis two-finger teacher PPO checkpoint for subsequent student imitation/distillation.

### What changed (files + behavior impact)
- Documentation only.
- User visually compared:
  - `thesis_twofinger_thumbpose02_dgain015_diag_s42_1h`
  - `thesis_twofinger_thumbpose02_diag_s42_1h`
- Decision: use the slower but visibly more stable pure `thumbpose02` run as the teacher PPO checkpoint for student distillation.

### What was verified (commands + key outcomes)
- Confirmed selected checkpoint exists:
  - `ls -lh outputs/Dexh13HoraLightbulb_teacher_thesis_twofinger/thesis_twofinger_thumbpose02_diag_s42_1h/stage1_nn/best_reward_3171.29.pth`
  - Outcome: file exists, about `1.2M`.
- Confirmed the faster comparison checkpoint also exists:
  - `outputs/Dexh13HoraLightbulb_teacher_thesis_twofinger/thesis_twofinger_thumbpose02_dgain015_diag_s42_1h/stage1_nn/best_reward_3558.70.pth`

### Local conclusion
- Selected teacher PPO checkpoint:
  - `outputs/Dexh13HoraLightbulb_teacher_thesis_twofinger/thesis_twofinger_thumbpose02_diag_s42_1h/stage1_nn/best_reward_3171.29.pth`
- Rationale:
  - `thumbpose02_dgain015` is faster, but visual slip/ejection is larger.
  - pure `thumbpose02` rotates slower, but is the more thesis-safe stable teacher for distillation.
- Do not use the higher-reward `dgain015` or `thumbpose02_dgain015` as the default distillation teacher unless the goal changes from stable final behavior to speed-first behavior.

### Remaining blocked/risky
- Existing generic DexH13 lightbulb student scripts default to:
  - `task=Dexh13HoraLightbulb`
  - `checkpoint=outputs/Dexh13HoraLightbulb_teacher/${CACHE}/stage1_nn/best_reward_*.pth`
- For thesis two-finger student distillation, override both task and checkpoint explicitly, or add thesis-specific student scripts before long runs.

### Single recommended next step
- Start student distillation with the explicit selected teacher checkpoint:
  - `task=Dexh13HoraLightbulbThesisTwoFinger`
  - `checkpoint=outputs/Dexh13HoraLightbulb_teacher_thesis_twofinger/thesis_twofinger_thumbpose02_diag_s42_1h/stage1_nn/best_reward_3171.29.pth`

---

## v2-148 (2026-04-28) -- Thesis Two-Finger ProprioAdapt Student Script

### Target milestone/subgoal
- Provide a dedicated project-native ProprioAdapt script for distilling the selected thesis two-finger PPO teacher.

### What changed (files + behavior impact)
- `scripts/dexh13_lightbulb_student_padapt_thesis_twofinger.sh`
  - New executable script.
  - Uses `task=Dexh13HoraLightbulbThesisTwoFinger`.
  - Uses `train.algo=ProprioAdapt` and `train.ppo.proprio_adapt=True`.
  - Writes to `outputs/Dexh13HoraLightbulb_student_padapt_thesis_twofinger/${CACHE}`.
  - Defaults checkpoint to the selected stable teacher:
    - `outputs/Dexh13HoraLightbulb_teacher_thesis_twofinger/thesis_twofinger_thumbpose02_diag_s42_1h/stage1_nn/best_reward_3171.29.pth`
  - Keeps the existing DexH13 student defaults:
    - `task.env.numEnvs=48`
    - `train.ppo.minibatch_size=576`
    - observation noise `obs_noise_t_scale=0.01`, `obs_noise_e_scale=0.02`
    - thesis termination checks enabled with `grace_steps=0`

### What was verified (commands + key outcomes)
- Static checks:
  - `chmod +x scripts/dexh13_lightbulb_student_padapt_thesis_twofinger.sh`
  - `bash -n scripts/dexh13_lightbulb_student_padapt_thesis_twofinger.sh`
  - `git diff --check -- scripts/dexh13_lightbulb_student_padapt_thesis_twofinger.sh`
  - Outcome: pass.
- Checkpoint:
  - `ls -lh outputs/Dexh13HoraLightbulb_teacher_thesis_twofinger/thesis_twofinger_thumbpose02_diag_s42_1h/stage1_nn/best_reward_3171.29.pth`
  - Outcome: selected teacher checkpoint exists.
- Runtime smoke:
  - `./docker-run-isaacgym.sh timeout 240 bash scripts/dexh13_lightbulb_student_padapt_thesis_twofinger.sh 0 42 smoke_padapt_thesis_tmp task.env.numEnvs=4 train.ppo.minibatch_size=48 train.ppo.max_agent_steps=48 wandb_activate=False`
  - Outcome:
    - Hydra composed `task=Dexh13HoraLightbulbThesisTwoFinger`.
    - `train.algo=ProprioAdapt`.
    - `train.load_path` resolved to the selected teacher checkpoint.
    - environment built successfully with the thesis two-finger config.
    - teacher checkpoint loaded successfully.
    - `ProprioAdapt trainable patterns: ['adapt_tconv'] | trainable params: 25544`.
    - no tensor/action/observation dimension mismatch occurred.
    - smoke was stopped by timeout code `124` after proving the path; temporary smoke output was removed.
- Process hygiene:
  - No residual student smoke training process remained.
  - An unrelated existing viewer process was still running:
    - PID `2998181`, `test=True`, visualizing `thesis_twofinger_thumbpose02_dgain015_diag_s42_1h`.

### Local conclusion
- The built-in `ProprioAdapt` path is compatible with the thesis two-finger PPO teacher.
- The correct production command should use the new thesis-specific script rather than the older generic `dexh13_lightbulb_student_padapt.sh`, because the older script defaults to `task=Dexh13HoraLightbulb` and the old teacher output root.

### Remaining blocked/risky
- The smoke used only 4 envs and was timeout-bounded. It confirms wiring/compatibility, not final distillation quality.
- The user should stop unrelated viewer processes before launching a long student run if GPU memory becomes tight.

### Single recommended next step
- Start a first bounded ProprioAdapt thesis student run:
  - `./docker-run-isaacgym.sh timeout 1800 scripts/run_with_cleanup.sh bash scripts/dexh13_lightbulb_student_padapt_thesis_twofinger.sh 0 42 thesis_twofinger_padapt_s42_30m wandb_activate=True`

---

## v2-149 (2026-04-28) -- Thesis Two-Finger ProprioAdapt Student Visualization Script

### Target milestone/subgoal
- Visualize the first 30-minute ProprioAdapt thesis two-finger student checkpoint.

### What changed (files + behavior impact)
- `scripts/vis_dexh13_lightbulb_student_padapt_thesis_twofinger.sh`
  - New executable visualization script for the thesis two-finger ProprioAdapt student.
  - Uses:
    - `task=Dexh13HoraLightbulbThesisTwoFinger`
    - `train.algo=ProprioAdapt`
    - `train.ppo.proprio_adapt=True`
    - `test=True`
    - deterministic visual settings with action/obs noise and force disturbance disabled.
  - Loads:
    - `outputs/Dexh13HoraLightbulb_student_padapt_thesis_twofinger/${CACHE}/stage2_nn/model_best.ckpt`

### What was verified (commands + key outcomes)
- Confirmed the 30-minute student checkpoint exists:
  - `outputs/Dexh13HoraLightbulb_student_padapt_thesis_twofinger/thesis_twofinger_padapt_s42_30m/stage2_nn/model_best.ckpt`
- Static checks:
  - `chmod +x scripts/vis_dexh13_lightbulb_student_padapt_thesis_twofinger.sh`
  - `bash -n scripts/vis_dexh13_lightbulb_student_padapt_thesis_twofinger.sh`
  - `git diff --check -- scripts/vis_dexh13_lightbulb_student_padapt_thesis_twofinger.sh`
  - Outcome: pass.
- Process check:
  - No active Isaac Gym `test=True` viewer process was found.

### Local conclusion
- The 30-minute ProprioAdapt student can now be visualized with a short cache-based command.

### Remaining blocked/risky
- Visual quality has not yet been inspected by the user.
- If the student visually lags the teacher but keeps stable two-finger contact, continue distillation from the same teacher with a longer run.

### Single recommended next step
- Visualize the 30-minute ProprioAdapt thesis student:
  - `./docker-run-isaacgym.sh bash scripts/vis_dexh13_lightbulb_student_padapt_thesis_twofinger.sh 0 42 thesis_twofinger_padapt_s42_30m`

---

## v2-150 (2026-04-28) -- Thesis Two-Finger ProprioAdapt Sim2Real Handoff Package

### Target milestone/subgoal
- Preserve the first usable 30-minute ProprioAdapt thesis two-finger student for a teammate's sim2real deployment work.

### What changed (files + behavior impact)
- Created `sim2real/thesis_twofinger_padapt_s42_30m/`.
- Added deployment/handoff artifacts:
  - `teacher_ppo_best_reward_3171.29.pth`
    - Stable thesis two-finger PPO teacher selected for reproduction and distillation.
  - `student_policy.pt`
    - TorchScript export from the ProprioAdapt student.
  - `model_best.ckpt`
    - Original project checkpoint copied from the run output.
  - `train_config.yaml`
    - Full Hydra config from the student training run.
  - `task_config_Dexh13HoraLightbulbThesisTwoFinger.yaml`
    - Task YAML snapshot.
  - `vis_student.sh`
    - Convenience visualization script snapshot.
  - `README.md`
    - Source paths, checksums, interface notes, and deployment caveat.

### What was verified (commands + key outcomes)
- Source checkpoint inspection:
  - `torch.load(.../model_best.ckpt, map_location='cpu')`
  - Outcome: checkpoint keys include `model`, `running_mean_std`, `sa_mean_std`, `priv_mean_std`, and `point_cloud_mean_std`.
- TorchScript export:
  - `JIT_OUTPUT_NAME=../sim2real/thesis_twofinger_padapt_s42_30m/student_policy.pt python student_eval.py task=Dexh13HoraLightbulbThesisTwoFinger ... train.algo=ProprioAdapt train.load_path=sim2real/thesis_twofinger_padapt_s42_30m/model_best.ckpt`
  - Outcome: export completed and wrote `student_policy.pt`.
- TorchScript load check:
  - `torch.jit.load('sim2real/thesis_twofinger_padapt_s42_30m/student_policy.pt', map_location='cpu')`
  - Outcome: load succeeded; normalization buffers are present.
- File checks:
  - `teacher_ppo_best_reward_3171.29.pth`: SHA1 `9a3b8d587dc2aecec6d0cdb8dcf98f0199c1cbdd`
  - `model_best.ckpt`: SHA1 `a92dd01eaf90302b68a1008129d36023cde1373d`
  - `student_policy.pt`: SHA1 `e0dd6a2f6b6d91256ba4ea5bed4947d9b4345b06`
  - `train_config.yaml`: SHA1 `9b28596ac4dd540cbec0f87e5d32c84e33dc4f25`
  - `task_config_Dexh13HoraLightbulbThesisTwoFinger.yaml`: SHA1 `50616aa953ec7841760ee44e2c29f93975554b53`
- Patch hygiene:
  - `git diff --check -- sim2real/thesis_twofinger_padapt_s42_30m/README.md`
  - Outcome: pass.
- Process hygiene:
  - No residual `student_eval.py` or thesis training process remained.

### Local conclusion
- The primary file to keep for exact repo reproduction and continued distillation is:
  - `sim2real/thesis_twofinger_padapt_s42_30m/model_best.ckpt`
- The primary teacher PPO file to keep for reproducing teacher visualization is:
  - `sim2real/thesis_twofinger_padapt_s42_30m/teacher_ppo_best_reward_3171.29.pth`
- The primary lightweight deployment candidate is:
  - `sim2real/thesis_twofinger_padapt_s42_30m/student_policy.pt`
- The package is intentionally small and does not include TensorBoard logs.

### Remaining blocked/risky
- The current TorchScript export path stores normalization buffers but follows the repo's existing traced forward convention. A real-robot runtime must confirm input schema and normalization before hardware execution.
- The user observed the policy is usable but still has occasional thumb slip/ejection, so this should be treated as a deploy candidate rather than a final safety-certified controller.

### Single recommended next step
- Hand the directory `sim2real/thesis_twofinger_padapt_s42_30m/` to the sim2real teammate, and have them first validate `student_policy.pt` in an offline replay or dry-run runtime before commanding the real hand.

---

## v2-151 (2026-04-28) -- Rounded-Contact Lightbulb Geometry Probe

### Target milestone/subgoal
- Create a conservative rounded-contact bulb URDF variant for visual inspection, without changing the thesis two-finger task config, reward, controller, or selected teacher/student checkpoints.

### What changed (files + behavior impact)
- Added `assets/lightbulb/rounded_contact_head.stl`.
  - Smooth lathed bulb-head mesh intended to reduce local faceted contact-normal changes.
  - Keeps the original `contact0.stl` x length/bounds and only slightly increases y/z radius.
- Added `assets/screw/roundedcontact/0000_lightbulb.urdf`.
  - New object type: `screw_roundedcontact`.
  - Visual and collision both use `../../lightbulb/rounded_contact_head.stl` for the head.
  - Socket visual/collision remains `../../lightbulb/contact1.stl`.
  - Main thesis default remains `screw_contactviz`; this is an override-only probe.

### What was verified (commands + key outcomes)
- XML and mesh dimension check:
  - `python - <<'PY' ... ET.parse('assets/screw/roundedcontact/0000_lightbulb.urdf') ... PY`
  - Outcome: URDF XML parse succeeded.
  - `contact0.stl`: `tris=124`, bbox size `[0.069184, 0.063119, 0.063119]`.
  - `rounded_contact_head.stl`: `tris=5904`, bbox size `[0.069184, 0.0644956, 0.0644956]`.
- Patch hygiene:
  - `git diff --check -- assets/screw/roundedcontact/0000_lightbulb.urdf docs/session_handoff_v2.md`
  - Outcome: pass.
- Isaac Gym asset smoke:
  - `./docker-run-isaacgym.sh timeout 180 python train.py task=Dexh13HoraLightbulbThesisTwoFinger task.env.object.type=screw_roundedcontact headless=True seed=42 num_envs=1 train.algo=PPO train.ppo.minibatch_size=12 train.ppo.max_agent_steps=24 wandb_activate=False train.ppo.output_name=Dexh13HoraLightbulb_teacher_thesis_twofinger/smoke_roundedcontact_tmp`
  - Outcome: task printed `Primitive List ['screw_roundedcontact']`, `using 1 training objects`, and `env 0 object_asset id: 0`; the asset loads and builds.
  - Tiny one-env smoke produced `mean_rewards: nan`, which is expected to be non-diagnostic for this short asset-load probe.
- Cleanup:
  - Removed `outputs/Dexh13HoraLightbulb_teacher_thesis_twofinger/smoke_roundedcontact_tmp`.
  - No `screw_roundedcontact` smoke training process remains.

### Local conclusion
- The rounded-contact geometry is available as a safe A/B asset via `task.env.object.type=screw_roundedcontact`.
- Because this keeps the old placement and nearly the old scale, it is a cleaner geometry probe than the previous smoothbulb attempt.

### Remaining blocked/risky
- User visual inspection is still needed.
- If the thumb still contacts with the side rather than fingertip, the next fix should likely be hand/object pose tuning (`handRootPos` or bulb y/z), not further mesh smoothing alone.
- Two unrelated one-env ProprioAdapt visualization/test processes were present on GPU during the check; they were not killed.

### Single recommended next step
- Visualize the selected thesis teacher checkpoint with the rounded-contact asset override:
  - `./docker-run-isaacgym.sh bash scripts/vis_dexh13_lightbulb_teacher_thesis_twofinger.sh 0 42 thesis_twofinger_thumbpose02_diag_s42_1h task.env.object.type=screw_roundedcontact`

---

## v2-152 (2026-04-28) -- Viewer Process Cleanup For Geometry A/B

### Target milestone/subgoal
- Clear stale one-env visualization/test processes before comparing `screw_contactviz` and `screw_roundedcontact` under the same thesis PPO checkpoint.

### What changed (files + behavior impact)
- No code/config behavior changed.
- Stopped stale `test=True` viewer/test processes:
  - `3032424`
  - `3035450`
  - `3102342`

### What was verified (commands + key outcomes)
- Process/GPU check:
  - `pgrep -af 'train.py task=Dexh13HoraLightbulbThesisTwoFinger.*test=True' || true`
  - `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits || true`
  - Outcome: no remaining thesis `train.py ... test=True` process; only the Chrome GPU process remained visible.

### Local conclusion
- The GPU is clear of the stale Isaac Gym visualization/test runs.

### Remaining blocked/risky
- The rounded-contact visual/collision gap still needs manual side-by-side inspection against `screw_contactviz`.

### Single recommended next step
- Run the same PPO viewer once with `task.env.object.type=screw_contactviz`, then once with `task.env.object.type=screw_roundedcontact`, and compare the apparent finger-to-bulb visual gap.

---

## v2-153 (2026-04-28) -- Cloud Conda/IsaacGym Bootstrap For Thesis Two-Finger PPO Teacher

### Target milestone/subgoal
- User-directed remote execution setup:
  - configure a feasible conda/IsaacGym environment on `cloud-training`,
  - prepare to run the stable thesis two-finger PPO teacher training with default training parameters and no time limit.

### What changed (files + behavior impact)
- Remote system/environment changes on `cloud-training`:
  - Synced local Isaac Gym Preview 4 to `/root/Codefield/third_party/isaacgym_preview4`.
  - Installed NVIDIA user-space packages matching the visible kernel module version:
    - `libnvidia-compute-535=535.154.05-0ubuntu1`
    - `nvidia-utils-535=535.154.05-0ubuntu1`
  - Created conda env `dexscrew-ig` by cloning remote `base`.
  - Installed project Python requirements into `dexscrew-ig`.
  - Installed Isaac Gym Python package editable from `/root/Codefield/third_party/isaacgym_preview4/isaacgym/python`.
  - Set conda env vars for `dexscrew-ig`:
    - `ISAACGYM_DIR=/root/Codefield/third_party/isaacgym_preview4`
    - `ISAACGYM_PATH=/root/Codefield/third_party/isaacgym_preview4`
    - `PYTHONPATH=/root/Codefield/third_party/isaacgym_preview4/isaacgym/python`
    - `LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64`
- No local training code behavior changed.

### What was verified (commands + key outcomes)
- Remote GPU/driver discovery:
  - `/proc/driver/nvidia/gpus/0000:65:01.0/information`
  - Outcome: GPU exists at host level: `NVIDIA GeForce RTX 4090 D`, kernel module `535.154.05`.
- Device access probe:
  - `dd if=/dev/nvidiactl of=/dev/null bs=1 count=0`
  - `dd if=/dev/nvidia0 of=/dev/null bs=1 count=0`
  - `dd if=/dev/nvidia-uvm of=/dev/null bs=1 count=0`
  - Outcome: all fail with `Operation not permitted`.
- Python import probe in `dexscrew-ig`:
  - `import isaacgym`
  - `from isaacgym import gymapi`
  - `import torch`
  - `import hydra, omegaconf, gym, trimesh, wandb`
  - Outcome: imports pass in IsaacGym-first order; `torch.cuda.is_available()` remains `False`.
- Minimal thesis PPO teacher startup probe:
  - `HYDRA_FULL_ERROR=1 timeout 180 python train.py task=Dexh13HoraLightbulbThesisTwoFinger ... task.env.numEnvs=1 train.ppo.minibatch_size=12 train.ppo.max_agent_steps=24 ...`
  - Log: `outputs/cloud_smoke/thesis_twofinger_teacher_env_probe.log` on remote.
  - Outcome: Isaac Gym binding loads, Hydra config resolves, task starts building, then fails at CUDA allocation with `RuntimeError: No CUDA GPUs are available`.

### Local conclusion
- The remote conda/IsaacGym environment is prepared enough for imports and task startup.
- Formal PPO teacher training was not started because the cloud runtime's Kubernetes/cgroup device policy denies access to `/dev/nvidia*`, despite the host-level 4090D being visible in `/proc`.

### Remaining blocked/risky
- This blocker cannot be fixed from conda or repo code. The cloud platform must expose GPU device access to the user container/session.
- Until `nvidia-smi` and `torch.cuda.is_available()` work inside the SSH session, Isaac Gym PPO training cannot run.

### Single recommended next step
- Fix cloud GPU device exposure from the platform side, then verify:
  - `nvidia-smi`
  - `python - <<'PY' ... torch.cuda.is_available() ... PY`
  - `dd if=/dev/nvidia0 of=/dev/null bs=1 count=0`
- Once those pass, start the real no-time-limit teacher run in `tmux`:
  - `source ~/miniconda3/etc/profile.d/conda.sh && conda activate dexscrew-ig && cd /root/code/dexscrew-repro && bash scripts/dexh13_lightbulb_teacher_thesis_twofinger.sh 0 42 thesis_twofinger_cloud_s42_unlimited True`

---

## v2-154 (2026-04-28) -- Cloud Training Readiness Recheck

### Target milestone/subgoal
- Ensure the remote thesis two-finger PPO teacher training environment is complete and confirm whether the current no-time-limit teacher training task is running stably.

### What changed (files + behavior impact)
- No training code/config behavior changed.
- No formal PPO teacher training process was launched because GPU access still fails before CUDA allocation.

### What was verified (commands + key outcomes)
- Remote process/session check:
  - `tmux ls`
  - `pgrep -af "train.py|dexh13_lightbulb_teacher_thesis_twofinger"`
  - Outcome: no active thesis teacher training tmux session or training process.
- Remote GPU access check:
  - `nvidia-smi`
  - Outcome: `Failed to initialize NVML: Unknown Error`.
  - `dd if=/dev/nvidiactl of=/dev/null bs=1 count=0`
  - `dd if=/dev/nvidia0 of=/dev/null bs=1 count=0`
  - `dd if=/dev/nvidia-uvm of=/dev/null bs=1 count=0`
  - Outcome: all fail with `Operation not permitted`.
- Remote conda/IsaacGym environment check:
  - `conda activate dexscrew-ig`
  - IsaacGym-first import order:
    - `import isaacgym`
    - `from isaacgym import gymapi, gymtorch`
    - `import torch`
    - `import hydra, omegaconf, gym, trimesh, wandb, tensorboardX`
  - Outcome: imports pass; `gymtorch` extension loads from cache; `torch.cuda.is_available()` remains `False`, `torch.cuda.device_count()` remains `0`.
- Remote training script check:
  - `bash -n scripts/dexh13_lightbulb_teacher_thesis_twofinger.sh`
  - Outcome: pass.
- Docker/GPU runtime check:
  - `docker info --format 'Runtimes={{json .Runtimes}} Default={{.DefaultRuntime}}'`
  - Outcome: only `runc` runtimes are available; no NVIDIA runtime is configured, so Docker cannot bypass the current GPU device-access blocker.
- Disk check:
  - Project: `93M`
  - Isaac Gym: `847M`
  - conda env `dexscrew-ig`: `4.7G`
  - Root filesystem: `20G` total, `13G` available.

### Local conclusion
- Training environment is complete at the Python/IsaacGym/project dependency layer.
- Current PPO teacher training is not running and cannot be made stable until the cloud platform exposes GPU device access to this SSH session/container.
- Starting the no-time-limit training now would immediately fail with the same CUDA error seen in the smoke probe.

### Remaining blocked/risky
- Hard platform blocker: cgroup/device policy denies opening `/dev/nvidia*`.
- The cloud UI/instance configuration must enable GPU device access for the current development container/session.

### Single recommended next step
- Fix GPU device exposure in the cloud platform, then re-run:
  - `nvidia-smi`
  - `dd if=/dev/nvidia0 of=/dev/null bs=1 count=0`
  - `source ~/miniconda3/etc/profile.d/conda.sh && conda activate dexscrew-ig && python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"`
- Once all pass, launch:
  - `tmux new -s thesis_teacher`
  - `source ~/miniconda3/etc/profile.d/conda.sh && conda activate dexscrew-ig && cd /root/code/dexscrew-repro && bash scripts/dexh13_lightbulb_teacher_thesis_twofinger.sh 0 42 thesis_twofinger_cloud_s42_unlimited True`

---

## v2-155 (2026-04-28) -- GPU Cloud Thesis Two-Finger PPO Teacher 1h Run

### Target milestone/subgoal
- Re-run cloud setup on the new SSH target `root@180.184.47.96:22222` and train the stable thesis two-finger PPO teacher for one hour.

### What changed (files + behavior impact)
- Remote-only execution/environment changes on `cloud-training`:
  - Accepted the new host key for the reused SSH endpoint after the cloud host changed to `di-20260428205452-zqhv8`.
  - Installed remote tools: `rsync`, `tmux`, `git`, `build-essential`, `ninja-build`.
  - Synced the current repo to `/root/code/dexscrew-repro`, excluding large local `outputs/`, `wandb/`, `.config/`, `.claude/`, caches, while keeping `.git`, code, configs, scripts, assets, docs, and sim2real files.
  - Synced Isaac Gym Preview 4 to `/root/Codefield/third_party/isaacgym_preview4`.
  - Created `dexscrew-ig` by cloning the remote `base` env so it inherits working `torch 2.1.0+cu121`.
  - Installed project requirements and Isaac Gym editable into `dexscrew-ig`.
  - Set conda env vars:
    - `ISAACGYM_DIR=/root/Codefield/third_party/isaacgym_preview4`
    - `ISAACGYM_PATH=/root/Codefield/third_party/isaacgym_preview4`
    - `PYTHONPATH=/root/Codefield/third_party/isaacgym_preview4/isaacgym/python`
    - `LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64`
  - Intentionally removed `/usr/local/cuda/compat` from `LD_LIBRARY_PATH` because that path contained old `libcuda.so.530.30.02` and caused Torch CUDA error 803 against driver `535.154.05`.
- Synced the completed 1h run artifacts back to local:
  - `outputs/Dexh13HoraLightbulb_teacher_thesis_twofinger/thesis_twofinger_cloud_s42_1h/`.
- No local training code/config behavior changed.

### What was verified (commands + key outcomes)
- New remote GPU access:
  - `nvidia-smi`
  - Outcome: `NVIDIA GeForce RTX 4090 D`, driver `535.154.05`, CUDA `12.2`, GPU idle before training.
  - `dd if=/dev/nvidiactl of=/dev/null bs=1 count=0`
  - `dd if=/dev/nvidia0 of=/dev/null bs=1 count=0`
  - `dd if=/dev/nvidia-uvm of=/dev/null bs=1 count=0`
  - Outcome: all succeeded.
- Conda/IsaacGym import probe in `dexscrew-ig`:
  - `import isaacgym`
  - `from isaacgym import gymapi, gymtorch`
  - `import torch, hydra, omegaconf, gym, trimesh, wandb, tensorboardX`
  - Outcome: pass; `torch.cuda.is_available() == True`, `torch.cuda.device_count() == 1`, device `NVIDIA GeForce RTX 4090 D`.
- Short thesis PPO smoke:
  - `HYDRA_FULL_ERROR=1 timeout 300 python train.py task=Dexh13HoraLightbulbThesisTwoFinger ... task.env.numEnvs=64 train.ppo.minibatch_size=768 train.ppo.max_agent_steps=1536 ...`
  - Outcome: environment built, PPO path ran, and exited with `max steps achieved`.
- 1h teacher launch:
  - `timeout 3600s bash scripts/dexh13_lightbulb_teacher_thesis_twofinger.sh 0 42 thesis_twofinger_cloud_s42_1h True`
  - Run dir: `outputs/Dexh13HoraLightbulb_teacher_thesis_twofinger/thesis_twofinger_cloud_s42_1h/`.
  - Outcome: completed by timeout with expected `EXIT_STATUS=124`; no training process remained afterward; GPU returned to idle.
- Final TensorBoard scalar snapshot:
  - `performance/RLTrainFPS`: `228120.828125` at step `95649792`
  - `performance/EnvStepFPS`: `30359.53125` at step `95649792`
  - `episode_rewards/step`: `3656.80126953125` at step `95551488`
  - `episode_lengths/step`: `798.2957763671875` at step `95551488`
  - `info/best_reward`: `3656.80126953125` at step `95551488`
  - `info/kl`: `0.021039489656686783`
  - `losses/actor_loss`: `0.0008476140792481601`
  - `losses/critic_loss`: `0.005827922839671373`
- Final artifacts:
  - `stage1_nn/best_reward_3656.80.pth`
  - `stage1_tb/events.out.tfevents.1777381431.di-20260428205452-zqhv8`
  - `config_042813_21e8eff.yaml`
  - `gitdiff.patch`
  - `train_1h.log`
  - `train_1h.exit`

### Local conclusion
- The new cloud machine has valid GPU access and a complete working `dexscrew-ig` Isaac Gym environment.
- The thesis two-finger PPO teacher 1h run completed cleanly under the requested stable script/default training config.
- Final checkpoint:
  - `outputs/Dexh13HoraLightbulb_teacher_thesis_twofinger/thesis_twofinger_cloud_s42_1h/stage1_nn/best_reward_3656.80.pth`
- This run outperformed the earlier selected local stable teacher reward numerically, but visual stability has not yet been inspected. Do not replace the thesis-stable default teacher for distillation until visual comparison confirms it is not a speed-first/slippy behavior.

### Remaining blocked/risky
- The 1h cloud teacher has not been visually inspected.
- The run log includes a large git diff because the repo is dirty; this is expected but noisy.
- Remote environment depends on avoiding `/usr/local/cuda/compat` in `LD_LIBRARY_PATH` on this cloud image.

### Single recommended next step
- Visualize the cloud 1h teacher and compare it against the selected stable local teacher:
  - `ssh cloud-training`
  - `source ~/miniconda3/etc/profile.d/conda.sh && conda activate dexscrew-ig`
  - `cd /root/code/dexscrew-repro`
  - `bash scripts/vis_dexh13_lightbulb_teacher_thesis_twofinger.sh 0 42 thesis_twofinger_cloud_s42_1h`

---

## v2-156 (2026-04-29) -- Cloud 1h Teacher Visual Rejection

### Target milestone/subgoal
- Interpret the user's visual inspection of `thesis_twofinger_cloud_s42_1h` and compare its actual config against the previously selected stable teacher.

### What changed (files + behavior impact)
- No code/config behavior changed.
- Updated this handoff with the visual rejection and config comparison.

### What was verified (commands + key outcomes)
- Compared saved Hydra configs:
  - `outputs/Dexh13HoraLightbulb_teacher_thesis_twofinger/thesis_twofinger_thumbpose02_diag_s42_1h/config_042719_21e8eff.yaml`
  - `outputs/Dexh13HoraLightbulb_teacher_thesis_twofinger/thesis_twofinger_cloud_s42_1h/config_042813_21e8eff.yaml`
- Key difference:
  - Stable selected teacher: `pose_diff_penalty.thumb_weight: 0.2`
  - Cloud 1h teacher: `pose_diff_penalty.thumb_weight: 0.1`
- Matching core controller/reward parameters:
  - `torque_limit: 300.0`
  - `pgain: 3`
  - `dgain: 0.01`
  - `action_scale: 0.05`
  - `rotate_reward_scale: 2.5`
  - `torque_penalty_scale: -30.0`
  - `work_penalty_scale: -0.15`
  - termination gates enabled by the training script with `grace_steps: 150`.
- User visual finding:
  - `thesis_twofinger_cloud_s42_1h` is very unstable.
  - Index shakes during rotation.
  - Thumb still slips/ejects.
  - Behavior looks force/velocity driven despite high scalar reward.

### Local conclusion
- Reject `thesis_twofinger_cloud_s42_1h` as the default distillation teacher despite its higher reward.
- The most likely cause is under-constrained stability: the cloud run used weaker thumb pose penalty (`0.1`) than the selected stable teacher (`0.2`), while the slip diagnostics are logging-only and do not directly penalize the unstable behavior.
- High reward is still dominated by rotation progress; it is not a reliable proxy for thesis-safe visual stability.

### Remaining blocked/risky
- Index jitter is not currently targeted by a dedicated smoothness/contact-continuity penalty.
- Thumb slip/ejection diagnostics remain diagnostic-only; adding them to reward would need a controlled ablation.

### Single recommended next step
- Keep `thesis_twofinger_thumbpose02_diag_s42_1h` as the stable distillation teacher for now. If rerunning on cloud, reproduce the stable setting explicitly:
  - `bash scripts/dexh13_lightbulb_teacher_thesis_twofinger.sh 0 42 thesis_twofinger_cloud_thumbpose02_s42_1h True wandb_activate=True task.env.termination.log=True task.env.pose_diff_penalty.thumb_weight=0.2`

---

## v2-157 (2026-04-29) -- Dexscrew Codebase Rescan for Cloud Development

### Target milestone/subgoal
- Rebuild the current project map for cloud-side development without changing code behavior.
- Confirm the latest single recommended next step before further cloud training/eval work.

### What changed (files + behavior impact)
- No training code/config behavior changed.
- Updated this handoff with a compact codebase map for the current thesis two-finger cloud workflow.

### What was verified (commands + key outcomes)
- Read the required bootstrap docs:
  - `docs/session_handoff_v2.md`
  - `docs/stage_acceptance_summary.md`
  - Outcome: latest recommendation remains to keep `thesis_twofinger_thumbpose02_diag_s42_1h` as the stable distillation teacher and only rerun cloud teacher with `task.env.pose_diff_penalty.thumb_weight=0.2`.
- Parallel read-only scans covered:
  - `train.py`
  - `student_eval.py`
  - `configs/config.yaml`
  - `configs/task/Dexh13HoraLightbulbThesisTwoFinger.yaml`
  - `configs/train/Dexh13HoraLightbulbThesisTwoFinger.yaml`
  - `dexscrew/tasks/__init__.py`
  - `dexscrew/tasks/dexh13_hora.py`
  - `dexscrew/tasks/xhand_hora.py`
  - `dexscrew/algo/ppo/ppo.py`
  - `dexscrew/algo/ppo/padapt.py`
  - `scripts/dexh13_lightbulb_teacher_thesis_twofinger.sh`
  - `scripts/dexh13_lightbulb_student_padapt_thesis_twofinger.sh`
  - `scripts/vis_dexh13_lightbulb_teacher_thesis_twofinger.sh`
  - `scripts/vis_dexh13_lightbulb_student_padapt_thesis_twofinger.sh`
- Key mapping verified:
  - `task=Dexh13HoraLightbulbThesisTwoFinger` selects the Hydra task yaml.
  - That yaml sets `name: Dexh13HoraLightbulb`.
  - `train.py` resolves `task_name: ${task.name}` and maps `Dexh13HoraLightbulb` to `Dexh13Hora`.
  - Runtime class chain is `Dexh13Hora -> XHandHora -> VecTask`.
- Key thesis two-finger mechanics verified:
  - PPO teacher entrypoint is `scripts/dexh13_lightbulb_teacher_thesis_twofinger.sh`.
  - ProprioAdapt student entrypoint is `scripts/dexh13_lightbulb_student_padapt_thesis_twofinger.sh`.
  - The student path is imitation/distillation, not second-stage RL.
  - Two-finger behavior is mostly config-driven: 16 actions, index/thumb active, middle/ring action mask, two-finger gate, fingertip tangent/torque rewards, and thumb slip diagnostics.
  - Thumb slip diagnostics are logging-only and do not directly change reward/termination.

### Local conclusion
- The active canonical path remains valid:
  - `Dexh13HoraLightbulbThesisTwoFinger -> PPO teacher -> ProprioAdapt student -> visualization/eval/export`.
- The selected stable teacher remains:
  - `outputs/Dexh13HoraLightbulb_teacher_thesis_twofinger/thesis_twofinger_thumbpose02_diag_s42_1h/stage1_nn/best_reward_3171.29.pth`
- The cloud 1h high-reward teacher remains rejected for default distillation:
  - `outputs/Dexh13HoraLightbulb_teacher_thesis_twofinger/thesis_twofinger_cloud_s42_1h/stage1_nn/best_reward_3656.80.pth`
- For cloud reruns, the most important override is still:
  - `task.env.pose_diff_penalty.thumb_weight=0.2`

### Remaining blocked/risky
- High scalar PPO reward is not sufficient for selecting a teacher; visual stability still gates teacher acceptance.
- Index jitter and thumb slip/ejection are not fully controlled by the current reward.
- Remote cloud runs must preserve the working Isaac Gym/CUDA environment and avoid `/usr/local/cuda/compat` in `LD_LIBRARY_PATH` on the current image.

### Single recommended next step
- If continuing cloud training, rerun the stable teacher setting explicitly and visually inspect it before any student distillation:
  - `bash scripts/dexh13_lightbulb_teacher_thesis_twofinger.sh 0 42 thesis_twofinger_cloud_thumbpose02_s42_1h True wandb_activate=True task.env.termination.log=True task.env.pose_diff_penalty.thumb_weight=0.2`

---

## v2-158 (2026-04-29) -- Thesis Sim2Real Joint0 Limit Config

### Target milestone/subgoal
- Add a local sim2real-oriented thesis two-finger task yaml based on the current stable setting.

### What changed (files + behavior impact)
- Added `configs/task/Dexh13HoraLightbulbThesisSim2Real.yaml`.
  - Based on `Dexh13HoraLightbulbThesisTwoFinger.yaml`.
  - Sets `eval_cache_name: thesis_sim2real`.
  - Sets `task.env.pose_diff_penalty.thumb_weight: 0.2`.
  - Narrows index joint0 limits from `[-0.35, 0.35]` to `[-0.2, 0.2]`.
  - Narrows thumb joint0 limits from `[-0.35, 0.35]` to `[-0.2, 0.2]`.
  - Adjusts initial index/thumb joint0 values to `0.2` and `-0.2` so the init pose is inside the new limits.
- Added `configs/train/Dexh13HoraLightbulbThesisSim2Real.yaml`.
  - Copied from the thesis two-finger PPO train config so Hydra `train: ${task}` resolves cleanly.

### What was verified (commands + key outcomes)
- Docker/Hydra compose probe:
  - `./docker-run-isaacgym.sh python -c "... compose(... task=Dexh13HoraLightbulbThesisSim2Real) ..."`
  - Outcome: `eval_cache_name=thesis_sim2real`, `train_algo=PPO`, `thumb_weight=0.2`, index/thumb joint0 limits both `[-0.2, 0.2]`.
- Headless 1-step PPO checkpoint load smoke:
  - `./docker-run-isaacgym.sh python train.py task=Dexh13HoraLightbulbThesisSim2Real headless=True ... checkpoint=outputs/Dexh13HoraLightbulb_teacher_thesis_twofinger/thesis_twofinger_thumbpose02_diag_s42_1h/stage1_nn/best_reward_3171.29.pth +test_num_steps=1`
  - Outcome: environment built, checkpoint loaded, and `EvalSummary steps=1 avg_reward=0.084007 avg_done_rate=0.000000`.

### Local conclusion
- `Dexh13HoraLightbulbThesisSim2Real` is available locally for visualization/eval with the stable `thumb_weight=0.2` and tighter index/thumb joint0 limits.
- The existing stable teacher checkpoint can be loaded under this new task yaml for visual inspection.

### Remaining blocked/risky
- This is only a config-level sim2real limit test; no PPO teacher has been retrained with the narrower limits yet.
- A policy trained under the wider joint0 limits may behave differently when visually tested under the narrower limits.

### Single recommended next step
- Visualize the stable teacher under the new sim2real task yaml:
  - `./docker-run-isaacgym.sh bash scripts/vis_dexh13_lightbulb_teacher_thesis_twofinger.sh 0 42 thesis_twofinger_thumbpose02_diag_s42_1h task=Dexh13HoraLightbulbThesisSim2Real graphics_device_id=0`

---

## v2-159 (2026-04-29) -- Thesis Sim2Real Joint0 Limit Relaxation

### Target milestone/subgoal
- Adjust the local sim2real thesis task yaml joint0 range after visual/config iteration.

### What changed (files + behavior impact)
- Updated `configs/task/Dexh13HoraLightbulbThesisSim2Real.yaml`.
  - Index joint0 lower/upper limits changed from `[-0.2, 0.2]` to `[-0.34, 0.34]`.
  - Thumb joint0 lower/upper limits changed from `[-0.2, 0.2]` to `[-0.34, 0.34]`.
  - Did not change the current sim2real init pose values; index/thumb joint0 remain `0.195` and `-0.195`, which are inside the relaxed limits.

### What was verified (commands + key outcomes)
- Docker/Hydra compose probe:
  - `./docker-run-isaacgym.sh python -c "... compose(... task=Dexh13HoraLightbulbThesisSim2Real) ..."`
  - Outcome: `thumb_weight=0.2`, index joint0 limits `[-0.34, 0.34]`, thumb joint0 limits `[-0.34, 0.34]`, init joint0 values `0.195` and `-0.195`.

### Local conclusion
- `Dexh13HoraLightbulbThesisSim2Real` now keeps the stable thumb pose penalty while allowing almost the original joint0 lateral range, slightly narrower than the thesis source `[-0.35, 0.35]`.

### Remaining blocked/risky
- This is still a config-only sim2real visualization/training probe.
- A policy trained under one joint range may not transfer cleanly if evaluated or retrained under another joint range without visual checking.

### Single recommended next step
- Re-run local visualization/training visualization with `task=Dexh13HoraLightbulbThesisSim2Real` and compare thumb/index contact behavior against the previous `[-0.2, 0.2]` setting.

---

## v2-160 (2026-04-29) -- Cloud Thesis Sim2Real PPO-to-PAdapt Pipeline Launch

### Target milestone/subgoal
- Sync the current `Dexh13HoraLightbulbThesisSim2Real` task to the GPU cloud machine.
- Run a 1h PPO teacher on the sim2real yaml, then automatically launch ProprioAdapt distillation from that teacher checkpoint.

### What changed (files + behavior impact)
- Synced the sim2real task/train config and required local code/script/asset files to `cloud-training:/root/code/dexscrew-repro/`:
  - `configs/task/Dexh13HoraLightbulbThesisSim2Real.yaml`
  - `configs/train/Dexh13HoraLightbulbThesisSim2Real.yaml`
  - thesis two-finger task/train configs
  - thesis teacher/student scripts
  - active task/algo Python files
  - relevant lightbulb/contactviz assets
- Created a remote pipeline script:
  - `outputs/cloud_pipeline_thesis_sim2real/run_pipeline.sh`
- Launched remote tmux session:
  - `thesis_sim2real_pipeline`

### What was verified (commands + key outcomes)
- Remote GPU/env/config check:
  - `nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv,noheader`
  - Outcome: `NVIDIA GeForce RTX 4090 D`, driver `535.154.05`, GPU available.
  - IsaacGym-first import probe with Torch:
    - Outcome: `torch_cuda True 1 NVIDIA GeForce RTX 4090 D`.
  - Hydra compose for `task=Dexh13HoraLightbulbThesisSim2Real`:
    - `eval_cache_name thesis_sim2real`
    - `train_algo PPO`
    - `thumb_weight 0.2`
    - index joint0 limits `[-0.34, 0.34]`
    - thumb joint0 limits `[-0.34, 0.34]`
- Remote teacher process startup:
  - tmux session exists: `thesis_sim2real_pipeline`.
  - Active command:
    - `timeout 3600s bash scripts/dexh13_lightbulb_teacher_thesis_twofinger.sh 0 42 thesis_sim2real_joint034_s42_1h True task=Dexh13HoraLightbulbThesisSim2Real ...`
  - GPU compute process exists and uses about `10524 MiB`.
  - Log confirms environment build reached:
    - `Start Building the Environment`
    - `Generated 5000 random initial poses for XHand at scale 1.2`

### Local conclusion
- The cloud pipeline is launched and currently in the PPO teacher phase.
- The teacher run name is:
  - `thesis_sim2real_joint034_s42_1h`
- The teacher output root is:
  - `outputs/Dexh13HoraLightbulb_teacher_thesis_sim2real/thesis_sim2real_joint034_s42_1h/`
- After the 1h `timeout 3600s` teacher command exits with `0` or expected `124`, the remote script will select the newest:
  - `outputs/Dexh13HoraLightbulb_teacher_thesis_sim2real/thesis_sim2real_joint034_s42_1h/stage1_nn/best_reward_*.pth`
- Then it will launch ProprioAdapt with:
  - `task=Dexh13HoraLightbulbThesisSim2Real`
  - `checkpoint=<teacher best_reward ckpt>`
  - output root `outputs/Dexh13HoraLightbulb_student_padapt_thesis_sim2real/thesis_sim2real_padapt_from_joint034_ppo1h_s42/`

### Remaining blocked/risky
- The teacher has not finished its 1h run yet.
- The student distillation has not started yet; it is queued inside the pipeline script after teacher checkpoint selection.
- `train.py` is printing a large dirty git diff at startup, so `latest.log` is noisy before normal `Agent Steps` lines appear.
- The resulting teacher still needs visual inspection before treating it as a stable sim2real distillation source.

### Single recommended next step
- Monitor the cloud pipeline until teacher timeout and ProprioAdapt launch:
  - `ssh cloud-training`
  - `cd /root/code/dexscrew-repro`
  - `tmux attach -t thesis_sim2real_pipeline`
  - or `tail -f outputs/cloud_pipeline_thesis_sim2real/latest.log`

---

## v2-161 (2026-04-29) -- Cloud Thesis Sim2Real Pipeline Status and Randomization Check

### Target milestone/subgoal
- Check whether the cloud PPO-to-ProprioAdapt pipeline has finished.
- Clarify the active domain randomization in `Dexh13HoraLightbulbThesisSim2Real.yaml`.

### What changed (files + behavior impact)
- No code/config behavior changed.
- Updated this handoff with cloud run status and domain-randomization interpretation.

### What was verified (commands + key outcomes)
- Remote process/log check:
  - `tmux ls | grep thesis_sim2real_pipeline`
  - `pgrep -af "run_pipeline|train.py|dexh13_lightbulb"`
  - `grep -E "\\[pipeline\\]|teacher_exit_status|teacher_ckpt|Current Best" outputs/cloud_pipeline_thesis_sim2real/latest.log`
- PPO teacher status:
  - Finished the intended 1h timeout with expected `teacher_exit_status=124`.
  - Selected checkpoint:
    - `outputs/Dexh13HoraLightbulb_teacher_thesis_sim2real/thesis_sim2real_joint034_s42_1h/stage1_nn/best_reward_3316.34.pth`
- ProprioAdapt status:
  - Started automatically from the teacher checkpoint and is still running.
  - Active output root:
    - `outputs/Dexh13HoraLightbulb_student_padapt_thesis_sim2real/thesis_sim2real_padapt_from_joint034_ppo1h_s42/`
  - `stage2_nn/model_best.ckpt` already exists, but final student exit status has not appeared yet.
- Randomization code/config check:
  - `randomizeScale: False`, `randomizeScaleList: [1.20]`, and `baseObjScale: 1.20`, so object/URDF scale is fixed at `1.20` and not randomized.
  - `randomizeMass: True`, with mass sampled in `[0.04, 0.06]` at runtime for object rigid bodies.
  - `randomizeCOM: True`, with COM offsets sampled around the configured millimeter-scale range.
  - `randomizeFriction: True`, with friction sampled in `[1.0, 5.0]` and restitution sampled in `[0.0, 0.05]`; the sampled values are applied to hand and object rigid shapes.
  - `randomizePDGains: True`, with P gain in `[2.7, 3.3]` and D gain in `[0.009, 0.011]`.
  - Observation/action noise and random object forces are enabled in the task yaml.

### Local conclusion
- The full pipeline is not finished yet: teacher is done, ProprioAdapt distillation is in progress.
- The current task does not randomize URDF/object size; it uses a fixed scaled size.
- The current task does randomize object mass, COM, friction/restitution, PD gains, observation/action noise, and external forces.

### Remaining blocked/risky
- The newly trained teacher has not been visually inspected.
- ProprioAdapt is still running, so the final student checkpoint/exit status is not yet known.
- The teacher reward `3316.34` is scalar-only and should not be treated as stable until visual inspection.

### Single recommended next step
- Continue monitoring until ProprioAdapt exits:
  - `ssh cloud-training`
  - `cd /root/code/dexscrew-repro`
  - `tail -f outputs/cloud_pipeline_thesis_sim2real/latest.log`

---

## v2-162 (2026-04-29) -- Dexh13HoraLightbulb YAML Rebased From ThesisSim2Real With NutBolt-Style Reward

### Target milestone/subgoal
- Rework `configs/task/Dexh13HoraLightbulb.yaml` as a comparison/training config based on `Dexh13HoraLightbulbThesisSim2Real.yaml`, while reverting selected randomization and reward terms toward `Dexh13HoraLightbulb2.yaml` / original `XHandHoraNutBolt.yaml`.

### What changed (files + behavior impact)
- Updated `configs/task/Dexh13HoraLightbulb.yaml`.
- Kept ThesisSim2Real-style two-finger structure:
  - `apply_action_mask: True`
  - `action_mask_indices: [4, 5, 6, 7, 8, 9, 10, 11]`
  - `object.type: screw_contactviz`
  - hand/root/init pose and middle/ring DOF locking from the Sim2Real line.
- Changed object scale randomization to match `Dexh13HoraLightbulb2.yaml`:
  - `baseObjScale: 1.0`
  - `randomizeScale: True`
  - `randomizeScaleList: [1.0, 1.05, 1.10, 1.15]`
  - scale min/max/lower/upper set to `[1.0, 1.15]`.
- Added object init noise:
  - kept Sim2Real object center `init_pos: [0.012, -0.018, 0.0]`
  - set `init_pos_noise: [0.005, 0.005, 0.0]`.
- Changed friction randomization to match `Dexh13HoraLightbulb2.yaml` / NutBolt:
  - `randomizeFrictionLower: 0.5`
  - `randomizeFrictionUpper: 8.0`
  - removed explicit restitution bounds.
- Changed reward/penalty scales to original NutBolt-style values:
  - `angvelPenaltyThres: 10.0`
  - `rotate_reward_scale: 6.0`
  - `pose_diff_penalty_scale: -0.5`
  - `torque_penalty_scale: -0.1`
  - `work_penalty_scale: -0.01`
  - `rotate_penalty_scale: -0.3`
  - `pc_z_dist_penalty_scale: -1.0`
  - `proximity_reward_scale: 2.0`
  - `normalize_penalties_by_num_actions: False`.
- Disabled thesis extra rewards and thumb-specific pose penalty:
  - `pose_diff_penalty.thumb_weight: 0.0`
  - `fingertip_tangent_reward_scale: 0.0`, `fingertip_tangent_reward.enable: False`
  - `fingertip_torque_reward_scale: 0.0`, `fingertip_torque_reward.enable: False`.

### What was verified (commands + key outcomes)
- Patch hygiene:
  - `git diff --check -- configs/task/Dexh13HoraLightbulb.yaml`
  - Outcome: pass.
- OmegaConf load/key placement check inside Docker:
  - `./docker-run-isaacgym.sh bash -lc "python - <<'PY' ... OmegaConf.load('configs/task/Dexh13HoraLightbulb.yaml') ... PY"`
  - Outcome:
    - `normalize penalties False`
    - reward tuple `6.0 -0.5 -0.1 -0.01`
    - `thumb_weight 0.0`
    - extra reward scales `0.0 0.0`, extra enables `False False`
    - object scale randomization `1.0 True [1.0, 1.05, 1.1, 1.15]`
    - friction range `0.5 8.0`
    - no explicit `randomizeRestitutionLower`
    - object `screw_contactviz [0.012, -0.018, 0.0] [0.005, 0.005, 0.0]`.

### Local conclusion
- `Dexh13HoraLightbulb.yaml` is now a hybrid comparison config:
  - Sim2Real two-finger geometry/action mask/DOF lock.
  - Lightbulb2-style object scale and friction randomization.
  - NutBolt-style reward/penalty magnitudes and no thesis extra index/thumb rewards.
- In `xhand_hora.py`, if restitution bounds are not explicit and `randomizeFriction=True`, restitution is sampled uniformly from `[0.0, 1.0]`.

### Remaining blocked/risky
- This config has not been smoke-trained after the edit.
- The broader friction/restitution range may reintroduce contact instability relative to the more conservative Sim2Real range `[friction 1.0-5.0, restitution 0.0-0.05]`.
- Turning off extra index rewards and thumb pose weight may make behavior less thesis-stable even if it is closer to original NutBolt reward form.

### Single recommended next step
- Run a short 64-env smoke before any long training:
  - `./docker-run-isaacgym.sh timeout 300 python train.py task=Dexh13HoraLightbulb headless=True seed=42 num_envs=64 train.algo=PPO train.ppo.minibatch_size=768 train.ppo.max_agent_steps=1536 wandb_activate=False train.ppo.output_name=Dexh13HoraLightbulb_teacher/smoke_nutbolt_reward_hybrid`

---

## v2-163 (2026-04-29) -- Cloud Thesis Sim2Real Artifacts Pulled For Local Visualization

### Target milestone/subgoal
- Retrieve the latest cloud-trained ThesisSim2Real teacher/student artifacts so the user can locally visualize the learned lightbulb policy.

### What changed (files + behavior impact)
- Pulled remote output artifacts from `cloud-training:/root/code/dexscrew-repro/` into local `outputs/`:
  - `outputs/Dexh13HoraLightbulb_teacher_thesis_sim2real/thesis_sim2real_joint034_s42_1h/`
  - `outputs/Dexh13HoraLightbulb_student_padapt_thesis_sim2real/thesis_sim2real_padapt_from_joint034_ppo1h_s42/`
  - `outputs/cloud_pipeline_thesis_sim2real/`
- No source/config behavior was changed.

### What was verified (commands + key outcomes)
- Remote pipeline status check:
  - Teacher completed the intended 1h timeout with selected checkpoint:
    - `outputs/Dexh13HoraLightbulb_teacher_thesis_sim2real/thesis_sim2real_joint034_s42_1h/stage1_nn/best_reward_3316.34.pth`
  - ProprioAdapt student was still running at check time, with current best around `2785.x`.
  - Student checkpoint already exists:
    - `outputs/Dexh13HoraLightbulb_student_padapt_thesis_sim2real/thesis_sim2real_padapt_from_joint034_ppo1h_s42/stage2_nn/model_best.ckpt`
- Local artifact check:
  - Teacher `.pth`: `1179649 bytes`
  - Student `.ckpt`: `1301474 bytes`
  - Local run dirs:
    - teacher dir `6.9M`
    - student dir `577M`
    - pipeline log dir `8.3M`

### Local conclusion
- Local visualization can be run immediately against the teacher checkpoint and the current student best checkpoint.
- The student checkpoint is a live snapshot because the cloud distillation process had not reached a `student_exit_status` marker yet.

### Remaining blocked/risky
- The cloud student may continue improving after this pull; a final re-sync is recommended once the tmux pipeline exits.
- The teacher/student policies have not yet been visually inspected locally after artifact transfer.

### Single recommended next step
- Run local visualization first on the teacher, then on the current student snapshot:
  - `./docker-run-isaacgym.sh bash scripts/vis_dexh13_lightbulb_teacher_thesis_twofinger.sh 0 42 thesis_sim2real_joint034_s42_1h task=Dexh13HoraLightbulbThesisSim2Real train.ppo.output_name=Dexh13HoraLightbulb_teacher_thesis_sim2real/thesis_sim2real_joint034_s42_1h checkpoint=outputs/Dexh13HoraLightbulb_teacher_thesis_sim2real/thesis_sim2real_joint034_s42_1h/stage1_nn/best_reward_3316.34.pth graphics_device_id=0`
  - `./docker-run-isaacgym.sh bash scripts/vis_dexh13_lightbulb_student_padapt_thesis_twofinger.sh 0 42 thesis_sim2real_padapt_from_joint034_ppo1h_s42 task=Dexh13HoraLightbulbThesisSim2Real train.ppo.output_name=Dexh13HoraLightbulb_student_padapt_thesis_sim2real/thesis_sim2real_padapt_from_joint034_ppo1h_s42 checkpoint=outputs/Dexh13HoraLightbulb_student_padapt_thesis_sim2real/thesis_sim2real_padapt_from_joint034_ppo1h_s42/stage2_nn/model_best.ckpt graphics_device_id=0`

---

## v2-164 (2026-04-29) -- Lightbulb Gate Target Moved To Bulb Maximum Circumference

### Target milestone/subgoal
- Align the two-finger target/contact gate in `Dexh13HoraLightbulb.yaml` with the actual widest bulb-head cross-section so index/thumb are encouraged to contact the useful bulb surface rather than the lower neck/socket region.

### What changed (files + behavior impact)
- Updated `configs/task/Dexh13HoraLightbulb.yaml`:
  - `two_finger_gate.target_offset: [0.0, 0.0, 0.084]`
  - `fingertip_tangent_reward.target_offset: [0.0, 0.0, 0.084]`
  - `fingertip_torque_reward.target_offset: [0.0, 0.0, 0.084]`
- The tangent/torque fingertip rewards remain disabled in this config; their offsets were synchronized only for diagnostic/future consistency.

### What was verified (commands + key outcomes)
- Geometry check:
  - Parsed `assets/lightbulb/contact0.stl` under the URDF transform used by `assets/screw/contactviz/0000_lightbulb.urdf`.
  - The maximum-radius head cross-section is around world/nut-link target offset `z ~= 0.0843 m` for `baseObjScale=1.0`.
- Isaac Gym one-env pose check:
  - `nut_pos = object_pos + [0, 0, 0.005]`, confirming the gate offset is relative to the task's `nut_pos` rigid-body state, not raw object root.
- Config checks:
  - `rg -n "target_offset" configs/task/Dexh13HoraLightbulb.yaml`
  - Outcome: all three relevant offsets are `[0.0, 0.0, 0.084]`.
  - `git diff --check -- configs/task/Dexh13HoraLightbulb.yaml`
  - Outcome: pass.

### Local conclusion
- For the current `Dexh13HoraLightbulb.yaml` with `baseObjScale: 1.0`, `target_offset z=0.084` is the right first target for making both active fingers aim at the bulb's largest useful circumference.
- If returning to a `baseObjScale: 1.20` thesis config, the equivalent unscaled physical target would be about `0.101`.

### Remaining blocked/risky
- This offset change has not yet been smoke-trained or visually checked.
- Moving the gate upward may require a small hand/root/init pose adjustment if the fingertips now undershoot the target band.

### Single recommended next step
- Run the existing visual training/init check for `Dexh13HoraLightbulb` and inspect whether index/thumb now aim at the bulb head's widest cross-section before launching a long PPO run.

---

## v2-165 (2026-04-29) -- Interactive DexH13 Init-Pose Tuning Viewer

### Target milestone/subgoal
- Provide a temporary visual interface for adjusting DexH13 lightbulb initial hand pose before committing values into YAML.

### What changed (files + behavior impact)
- Added `scripts/tune_dexh13_lightbulb_initpose.py`.
- The script starts a single Isaac Gym viewer env for a chosen task and lets the user interactively tune:
  - hand root 6D pose via `handRootPos` and `handRootRPY`,
  - all 16 hand DOF values under `handInitPose`.
- It disables training/randomization disturbances for the tuning session:
  - one env,
  - no mass/COM/friction/scale/PD randomization,
  - no object init noise,
  - no random force perturbation.
- Pressing `O` or closing/quitting saves a paste-ready YAML snippet to `outputs/initpose_tuning/`.

### What was verified (commands + key outcomes)
- Syntax/hygiene:
  - `PYTHONDONTWRITEBYTECODE=1 python -m py_compile scripts/tune_dexh13_lightbulb_initpose.py`
  - `git diff --check -- scripts/tune_dexh13_lightbulb_initpose.py`
  - Outcome: pass.
- Isaac Gym smoke:
  - `./docker-run-isaacgym.sh timeout 20 python scripts/tune_dexh13_lightbulb_initpose.py --task Dexh13HoraLightbulb --gpu 0 --out outputs/initpose_tuning/smoke.yaml`
  - Outcome: env/viewer bootstrapped, printed interactive controls and accepted viewer keyboard events before the expected timeout killed the smoke process.

### Local conclusion
- The user can now tune root pose and 16DOF init pose visually without running PPO.
- The output YAML snippet is intended to be pasted under `env.asset` in the selected task YAML.

### Remaining blocked/risky
- Because this is viewer-driven, actual usability depends on the host X11/Vulkan viewer working in the Docker session.
- The smoke command used `timeout`, so it intentionally did not test the final save-on-quit path through a normal manual close.

### Single recommended next step
- Run the tuner without `timeout`, adjust the pose visually, press `O`, then paste/send the generated `outputs/initpose_tuning/*.yaml` snippet so the task YAML can be updated.

---

## v2-166 (2026-04-29) -- Diffusion Student Algorithm Summary Drafted

### Target milestone/subgoal
- Summarize the current four diffusion / generative student algorithms in the same detailed style as `output_docs/algo_adapt.md`.

### What changed (files + behavior impact)
- Wrote `output_docs/algo_diffusion.md`.
- The document now covers:
  - `DiffusionLatentStudent` / latent DDPM,
  - `ConsistencyLatentStudent`,
  - `FlowMatchingLatentStudent`,
  - `DiffusionActionChunkStudent`,
  - common teacher/student tensors and normalization,
  - mathematical objectives,
  - train-loop implementation,
  - inference/deployment paths,
  - checkpoint contents,
  - key config fields and practical differences.
- No training or source behavior was changed.

### What was verified (commands + key outcomes)
- Source inspection:
  - `output_docs/algo_adapt.md`
  - `dexscrew/algo/ppo/diffusion_latent_student.py`
  - `dexscrew/algo/ppo/consistency_latent_student.py`
  - `dexscrew/algo/ppo/flow_matching_latent_student.py`
  - `dexscrew/algo/ppo/diffusion_action_chunk_student.py`
  - relevant train config / algorithm registration files.
- Markdown hygiene:
  - `git diff --check -- output_docs/algo_diffusion.md`
  - Outcome: pass.
- Content spot checks:
  - `wc -l output_docs/algo_diffusion.md`
  - Outcome: `1830` lines.
  - `rg -n "DiffusionLatentStudent|ConsistencyLatentStudent|FlowMatchingLatentStudent|DiffusionActionChunkStudent|L_diff|L_cons|L_flow|Action Chunk|Checkpoint" output_docs/algo_diffusion.md`
  - Outcome: all four algorithms and core losses/checkpoint sections present.

### Local conclusion
- `output_docs/algo_diffusion.md` is now a standalone systematic algorithm note for the four diffusion/generative student branches, suitable for review and later condensation into thesis/report sections.

### Remaining blocked/risky
- The document is detailed and implementation-oriented; if used in a thesis, it should be condensed into method and ablation-analysis subsections rather than copied wholesale.
- It describes current code behavior, including historical exploratory branches; it does not assert that all four algorithms are accepted final baselines.

### Single recommended next step
- Review `output_docs/algo_diffusion.md` for wording/level-of-detail, then decide which parts should be compressed into the formal thesis algorithm section.

---

## v2-166 (2026-04-29) -- Thesis Sim2Real Student Snapshot Re-Synced

### Target milestone/subgoal
- Check whether the cloud ThesisSim2Real ProprioAdapt distillation finished and pull the latest available local visualization artifacts.

### What changed (files + behavior impact)
- Re-synced the live student output directory from cloud to local:
  - `outputs/Dexh13HoraLightbulb_student_padapt_thesis_sim2real/thesis_sim2real_padapt_from_joint034_ppo1h_s42/`
- Re-synced pipeline logs:
  - `outputs/cloud_pipeline_thesis_sim2real/`
- No source/config behavior was changed.

### What was verified (commands + key outcomes)
- Remote status check:
  - `tmux` session `thesis_sim2real_pipeline` still exists.
  - `train.py` ProprioAdapt process is still active.
  - No `student_exit_status` marker was present in `outputs/cloud_pipeline_thesis_sim2real/latest.log`.
  - Latest parsed `Current Best` was around `2808.04`.
- Local artifact check after rsync:
  - `outputs/Dexh13HoraLightbulb_student_padapt_thesis_sim2real/thesis_sim2real_padapt_from_joint034_ppo1h_s42/stage2_nn/model_best.ckpt`
  - size `1301474 bytes`
  - local mtime `2026-04-29 15:59:18 Asia/Shanghai`
  - local student run dir size `886M`
  - local pipeline log dir size `13M`

### Local conclusion
- Student distillation is not finished yet; the local `model_best.ckpt` is the latest pulled live snapshot.
- This snapshot is valid for local visual inspection, but it may be superseded if the cloud run continues improving.

### Remaining blocked/risky
- A final sync is still needed after the cloud pipeline exits if a final checkpoint is required.
- Visual behavior has not yet been inspected from this newer student snapshot.

### Single recommended next step
- Visualize the latest pulled student snapshot locally:
  - `./docker-run-isaacgym.sh bash scripts/vis_dexh13_lightbulb_student_padapt_thesis_twofinger.sh 0 42 thesis_sim2real_padapt_from_joint034_ppo1h_s42 task=Dexh13HoraLightbulbThesisSim2Real train.ppo.output_name=Dexh13HoraLightbulb_student_padapt_thesis_sim2real/thesis_sim2real_padapt_from_joint034_ppo1h_s42 checkpoint=outputs/Dexh13HoraLightbulb_student_padapt_thesis_sim2real/thesis_sim2real_padapt_from_joint034_ppo1h_s42/stage2_nn/model_best.ckpt graphics_device_id=0`

---

## v2-167 (2026-04-29) -- Thesis Sim2Real Student Stopped And Final Snapshot Synced

### Target milestone/subgoal
- Stop the overlong cloud ProprioAdapt distillation, preserve the best checkpoint, and sync the final available snapshot locally.

### What changed (files + behavior impact)
- Stopped the cloud ThesisSim2Real student run:
  - killed the active ProprioAdapt `train.py` process,
  - killed remaining wrapper/tmux pipeline processes,
  - verified no matching `tmux`, wrapper, or `train.py` processes remain.
- Preserved the cloud best checkpoint before termination:
  - `outputs/Dexh13HoraLightbulb_student_padapt_thesis_sim2real/thesis_sim2real_padapt_from_joint034_ppo1h_s42/stage2_nn/model_best_userstop_20260429_080630.ckpt`
- Re-synced local artifacts:
  - `outputs/Dexh13HoraLightbulb_student_padapt_thesis_sim2real/thesis_sim2real_padapt_from_joint034_ppo1h_s42/`
  - `outputs/cloud_pipeline_thesis_sim2real/`

### What was verified (commands + key outcomes)
- Remote process check after stop:
  - no `thesis_sim2real_pipeline` tmux session,
  - no matching pipeline wrapper,
  - no matching ProprioAdapt `train.py` process.
- Local checkpoint check:
  - `model_best.ckpt`: `1301474 bytes`, mtime `2026-04-29 16:06:07 Asia/Shanghai`
  - `model_best_userstop_20260429_080630.ckpt`: `1301474 bytes`, same mtime.
- Latest parsed training line before termination:
  - `Agent Steps: 0007M ... Current Best: 2808.35`
- Local directory sizes after rsync:
  - student run dir `912M`
  - pipeline log dir `13M`

### Local conclusion
- The student run is stopped and the best checkpoint is available locally.
- Because the script lacked a student-side `timeout 3600s`, the run exceeded the intended 1h distillation limit and was manually stopped.

### Remaining blocked/risky
- `latest.log` ends with `Terminated` rather than a clean `student_exit_status` marker because the process was manually stopped.
- Future cloud pipeline scripts should wrap the student command with `timeout 3600s` when the intended protocol is 1h distillation.

### Single recommended next step
- Visualize the stopped-run best checkpoint locally:
  - `./docker-run-isaacgym.sh bash scripts/vis_dexh13_lightbulb_student_padapt_thesis_twofinger.sh 0 42 thesis_sim2real_padapt_from_joint034_ppo1h_s42 task=Dexh13HoraLightbulbThesisSim2Real train.ppo.output_name=Dexh13HoraLightbulb_student_padapt_thesis_sim2real/thesis_sim2real_padapt_from_joint034_ppo1h_s42 checkpoint=outputs/Dexh13HoraLightbulb_student_padapt_thesis_sim2real/thesis_sim2real_padapt_from_joint034_ppo1h_s42/stage2_nn/model_best.ckpt graphics_device_id=0`

---

## v2-168 (2026-04-29) -- Cloud Training Execution Rule Added

### Target milestone/subgoal
- Add stable execution guidance so future cloud PPO-to-student runs honor explicit wall-clock limits and use stronger cloud resources deliberately.

### What changed (files + behavior impact)
- Updated `AGENTS.md`.
- Added a new `Cloud training execution` section requiring future cloud runs to:
  - detect cloud/remote context from user wording, SSH aliases, remote paths, or `nvidia-smi`;
  - implement requested sequential PPO teacher and student/distillation phases with exact commands or a remote script;
  - wrap each time-limited phase in its own `timeout`;
  - log start/end timestamps, exact command lines, selected checkpoints, and teacher/student exit statuses;
  - preserve and sync the best student checkpoint after completion or manual stop;
  - avoid treating large `max_agent_steps` as a substitute for wall-clock limits.
- Added cloud resource guidance:
  - prefer higher `numEnvs` and compatible minibatch sizes on the 24GB GPU/cloud CPU/RAM when the user has not pinned exact settings;
  - verify GPU/memory/FPS before committing to long runs;
  - back off on OOM risk or degraded startup performance.

### What was verified (commands + key outcomes)
- `git diff --check -- AGENTS.md`
  - Outcome: pass.
- `git diff -- AGENTS.md`
  - Outcome: reviewed the inserted section and confirmed it is stable workflow guidance rather than stage-specific experiment detail.

### Local conclusion
- Future agents should not repeat the previous failure mode where teacher had `timeout 3600s` but student ran open-ended.
- Cloud runs can now be intentionally more aggressive while still keeping launch settings explicit and reproducible.

### Remaining blocked/risky
- Existing cloud pipeline scripts are not automatically rewritten by this governance update.
- Any new cloud run still needs a per-run script/command generated from the user's requested task, duration, and checkpoint policy.

### Single recommended next step
- For the next cloud PPO-to-student run, generate an `outputs/cloud_pipeline_*` script with `timeout` around both teacher and student phases before launching `tmux`.

---

## v2-169 (2026-04-29) -- Cloud DexH13 Lightbulb PPO Aggressive 1h Launch

### Target milestone/subgoal
- Use the current local `configs/task/Dexh13HoraLightbulb.yaml` as the baseline and launch a cloud PPO teacher run for 1 hour.
- Probe a more aggressive cloud resource setting before the 1h run and record the selected setting.

### What changed (files + behavior impact)
- Added cloud pipeline script:
  - `outputs/cloud_pipeline_dexh13_lightbulb_ppo_aggressive/run_ppo1h_aggressive.sh`
- Synced current local training inputs to `cloud-training:/root/code/dexscrew-repro/`:
  - `configs/task/Dexh13HoraLightbulb.yaml`
  - `configs/train/Dexh13HoraLightbulb.yaml`
  - `scripts/dexh13_lightbulb_teacher.sh`
  - cloud pipeline script above.
- Launched remote tmux session:
  - `dexh13_lightbulb_ppo_aggressive`

### What was verified (commands + key outcomes)
- Local script syntax:
  - `bash -n outputs/cloud_pipeline_dexh13_lightbulb_ppo_aggressive/run_ppo1h_aggressive.sh`
  - Outcome: pass.
- Remote sync/path check:
  - Confirmed cloud `configs/task/Dexh13HoraLightbulb.yaml` contains current local settings:
    - `numEnvs: ${resolve_default:8192,${...num_envs}}`
    - `two_finger_gate.target_offset: [0.0, 0.0, 0.084]`
    - `object.type: screw_contactviz`
    - `handRootPos: [0.064000, 0.014000, 0.229000]`
    - `enable_nut_dof_vel: Truecc` preserved exactly as local YAML.
- Cloud resource probe:
  - Candidate order: `12288/24576`, `8192/16384`, `6144/12288` for `task.env.numEnvs/train.ppo.minibatch_size`.
  - First candidate `12288 envs / minibatch 24576` allocated and ran to the 180s probe timeout without OOM.
  - Selected setting: `task.env.numEnvs=12288`, `train.ppo.minibatch_size=24576`, `num_threads=22`.
- Formal PPO run:
  - Started at `2026-04-29T08:28:04+00:00` with `timeout 3600s`.
  - Output root:
    - `outputs/Dexh13HoraLightbulb_teacher/dexh13_lightbulb_yaml_current_aggressive_s42_1h/`
  - Active command includes:
    - `task=Dexh13HoraLightbulb`
    - `train.algo=PPO`
    - `task.env.numEnvs=12288`
    - `train.ppo.minibatch_size=24576`
    - `num_threads=22`
  - Training entered effective execution and created:
    - `stage1_tb/events.out.tfevents...`
    - `stage1_nn/best_reward_160.93.pth`
  - Observed resource use after startup:
    - GPU memory about `14.5GB / 24.6GB`
    - GPU utilization up to about `90%`.

### Local conclusion
- The aggressive 12288-env cloud setting is viable on the 24GB 4090D for this task and leaves about 10GB memory headroom.
- Startup is heavier than the previous 8192/default setting, but the run is now producing checkpoints and TensorBoard events.
- The run is still active and should stop automatically via `timeout 3600s` around `2026-04-29T09:28:04+00:00`.

### Remaining blocked/risky
- The run has not reached the 1h timeout yet.
- `latest.log` is noisy because `train.py` prints the dirty git diff at startup.
- The local YAML contains `enable_nut_dof_vel: Truecc`, which is preserved by request but is a string rather than a boolean.
- The final teacher checkpoint still needs to be synced back after the 1h timeout.

### Single recommended next step
- After `2026-04-29T09:28:04+00:00`, check the cloud tmux/log for `ppo_exit_status=124`, then rsync:
  - `outputs/Dexh13HoraLightbulb_teacher/dexh13_lightbulb_yaml_current_aggressive_s42_1h/`
  - `outputs/cloud_pipeline_dexh13_lightbulb_ppo_aggressive/`

---

## v2-170 (2026-04-29) -- AGENTS Cloud Resource Probe Order Recorded

### Target milestone/subgoal
- Preserve the resource-choice lesson from the active cloud DexH13 Lightbulb PPO run in stable agent instructions.

### What changed (files + behavior impact)
- Updated `AGENTS.md` cloud training guidance with a preferred aggressive probe order for IsaacGym Hora PPO teacher runs on the current 24GB cloud GPU:
  - `task.env.numEnvs=12288`, `train.ppo.minibatch_size=24576`
  - `task.env.numEnvs=8192`, `train.ppo.minibatch_size=16384`
  - `task.env.numEnvs=6144`, `train.ppo.minibatch_size=12288`
- Added the condition that the selected probe should allocate, reach stable startup, and still leave meaningful training time inside the requested wall-clock budget.

### What was verified (commands + key outcomes)
- `git diff --check -- AGENTS.md docs/session_handoff_v2.md`
  - Outcome: pass.

### Local conclusion
- Future cloud PPO runs should start from the measured 12288/24576 aggressive profile when the user asks to utilize cloud resources and has not pinned exact values, then fallback in the recorded order.

### Remaining blocked/risky
- The active `dexh13_lightbulb_yaml_current_aggressive_s42_1h` run is still in progress and has not reached its 1h timeout yet.
- The recorded profile is based on this cloud class and IsaacGym Hora PPO teacher workload; other algorithms/tasks still require a probe.

### Single recommended next step
- Let the active cloud PPO run reach `timeout 3600s`, then sync the teacher output and pipeline log locally.

---

## v2-171 (2026-04-29) -- Cloud DexH13 Lightbulb PPO 1h Synced Locally

### Target milestone/subgoal
- Check completion of the aggressive cloud DexH13 Lightbulb PPO teacher run and sync artifacts locally for visualization.

### What changed (files + behavior impact)
- Synced cloud teacher output to local:
  - `outputs/Dexh13HoraLightbulb_teacher/dexh13_lightbulb_yaml_current_aggressive_s42_1h/`
- Synced cloud pipeline logs to local:
  - `outputs/cloud_pipeline_dexh13_lightbulb_ppo_aggressive/`
- No source/config behavior was changed.

### What was verified (commands + key outcomes)
- Remote completion check:
  - `ppo_exit_status=124` at `2026-04-29T09:28:04+00:00`, expected because PPO was wrapped in `timeout 3600s`.
  - `tmux` session no longer exists and no matching PPO process remains.
  - Selected cloud setting:
    - `task.env.numEnvs=12288`
    - `train.ppo.minibatch_size=24576`
- Remote/log best checkpoint:
  - `outputs/Dexh13HoraLightbulb_teacher/dexh13_lightbulb_yaml_current_aggressive_s42_1h/stage1_nn/best_reward_948.07.pth`
- Last parsed training line:
  - `Agent Steps: 0099M | FPS: 28832.5 | Last FPS: 29012.6 | Collect Time: 51.7 min | Train RL Time: 5.9 min | Current Best: 938.85`
- Local artifact check:
  - `best_reward_948.07.pth`: `1179610 bytes`
  - `ep_500_step_0073m_reward_810.79.pth`: `1180181 bytes`
  - `last.pth`: `1177400 bytes`
  - teacher run dir size `7.2M`
  - pipeline log dir size `1.6M`

### Local conclusion
- The requested 1h cloud PPO teacher run completed and is synced locally.
- The visualization checkpoint to use is:
  - `outputs/Dexh13HoraLightbulb_teacher/dexh13_lightbulb_yaml_current_aggressive_s42_1h/stage1_nn/best_reward_948.07.pth`

### Remaining blocked/risky
- Policy behavior has not yet been visually inspected.
- The task YAML used for this run still contains `enable_nut_dof_vel: Truecc`; this did not crash training but remains a config-quality concern.

### Single recommended next step
- Visualize the synced teacher checkpoint locally:
  - `./docker-run-isaacgym.sh bash scripts/vis_dexh13_lightbulb_teacher.sh 0 42 dexh13_lightbulb_yaml_current_aggressive_s42_1h test=True train.ppo.output_name=Dexh13HoraLightbulb_teacher/dexh13_lightbulb_yaml_current_aggressive_s42_1h checkpoint=outputs/Dexh13HoraLightbulb_teacher/dexh13_lightbulb_yaml_current_aggressive_s42_1h/stage1_nn/best_reward_948.07.pth graphics_device_id=0`

---

## v2-171 (2026-04-29) -- Output Diffusion Algorithm Notes Completed

### Target milestone/subgoal
- Produce an `output_docs`-level systematic summary of the four current diffusion / generative student algorithms, matching the detail level of `output_docs/algo_adapt.md`.

### What changed (files + behavior impact)
- Wrote `output_docs/algo_diffusion.md`.
- Covered:
  - `DiffusionLatentStudent` / latent DDPM,
  - `ConsistencyLatentStudent`,
  - `FlowMatchingLatentStudent`,
  - `DiffusionActionChunkStudent`,
  - shared teacher-student tensor definitions,
  - mathematical objectives,
  - train/eval/deploy logic,
  - checkpoint contents and key config fields.
- No source behavior or training config was changed.

### What was verified (commands + key outcomes)
- Inspected implementation files:
  - `dexscrew/algo/ppo/diffusion_latent_student.py`
  - `dexscrew/algo/ppo/consistency_latent_student.py`
  - `dexscrew/algo/ppo/flow_matching_latent_student.py`
  - `dexscrew/algo/ppo/diffusion_action_chunk_student.py`
  - relevant train config / algorithm registration files.
- Markdown hygiene:
  - `git diff --check -- output_docs/algo_diffusion.md`
  - Outcome: pass.
- Content spot checks:
  - `wc -l output_docs/algo_diffusion.md`
  - Outcome: `1830` lines.
  - `rg -n "DiffusionLatentStudent|ConsistencyLatentStudent|FlowMatchingLatentStudent|DiffusionActionChunkStudent|L_diff|L_cons|L_flow|Action Chunk|Checkpoint" output_docs/algo_diffusion.md`
  - Outcome: all four algorithms and core loss/checkpoint sections present.

### Local conclusion
- `output_docs/algo_diffusion.md` is now a standalone implementation-oriented reference for the repository's four diffusion/generative student branches.

### Remaining blocked/risky
- This is a detailed engineering note; it should be condensed before being used as thesis prose.
- It summarizes implementation and algorithm mechanics, not final experimental acceptance status for each branch.

### Single recommended next step
- Review `output_docs/algo_diffusion.md`, then extract a shorter thesis-ready method section from it.

---

## v2-173 (2026-04-29) -- Sim2Real Strong Two-Finger Cooperative YAML Added

### Target milestone/subgoal
- Create a new DexH13 lightbulb sim2real two-finger task variant based on the stable `ThesisTwoFinger + thumbpose02` line, but with a stronger simultaneous-contact gate for real-world stability.

### What changed (files + behavior impact)
- Added `configs/task/Dexh13HoraLightbulbSim2RealTwoFinger.yaml`.
  - Based on `configs/task/Dexh13HoraLightbulbThesisTwoFinger.yaml`.
  - Sets `eval_cache_name: sim2real_twofinger`.
  - Keeps the stable thumbpose02 setting: `task.env.pose_diff_penalty.thumb_weight=0.2`.
  - Enables reset/termination checks in YAML:
    - `enable_finger_dist=True`
    - `enable_nut_stagnation=True`
    - `enable_no_contact=True`
    - `enable_screw_limit=True`
    - `log=True`
  - Strengthens the positive-rotation two-finger gate:
    - `min_mult=0.0`
    - `power=3.0`
    - `contact_force_min=0.5`
    - `no_grasp_penalty_scale=-1.5`
- Added `configs/train/Dexh13HoraLightbulbSim2RealTwoFinger.yaml`.
  - Copied from the thesis two-finger PPO train config so Hydra `train: ${task}` resolves.
- Added `scripts/dexh13_lightbulb_teacher_sim2real_twofinger.sh`.
  - PPO teacher wrapper writing to `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger/<cache>`.
  - In headed mode (`HEADLESS=False`), forces `task.env.numEnvs=1` and `train.ppo.minibatch_size=12`.

### What was verified (commands + key outcomes)
- Script/hygiene:
  - `bash -n scripts/dexh13_lightbulb_teacher_sim2real_twofinger.sh`
  - `git diff --check -- configs/task/Dexh13HoraLightbulbSim2RealTwoFinger.yaml configs/train/Dexh13HoraLightbulbSim2RealTwoFinger.yaml scripts/dexh13_lightbulb_teacher_sim2real_twofinger.sh`
  - Outcome: pass.
- Hydra compose:
  - `./docker-run-isaacgym.sh python -c "... compose(... task=Dexh13HoraLightbulbSim2RealTwoFinger) ..."`
  - Outcome:
    - `eval_cache_name=sim2real_twofinger`
    - `train_algo=PPO`
    - `thumb_weight=0.2`
    - `enable_finger_dist=True`
    - `gate_min_mult=0.0`
    - `gate_power=3.0`
    - `contact_force_min=0.5`
    - `no_grasp_penalty=-1.5`
- Isaac Gym smoke:
  - `./docker-run-isaacgym.sh timeout 180 python train.py task=Dexh13HoraLightbulbSim2RealTwoFinger headless=True seed=42 num_envs=4 train.algo=PPO train.ppo.minibatch_size=12 train.ppo.max_agent_steps=24 wandb_activate=False train.ppo.output_name=Dexh13HoraLightbulb_teacher_sim2real_twofinger/smoke_tmp`
  - Outcome: environment built with `using 1 training objects`, generated random initial poses at scale `1.2`, and exited cleanly after `max steps achieved`.

### Local conclusion
- The new strong two-finger sim2real task is ready for headed init-pose inspection and short PPO probing.
- This variant intentionally makes single-finger positive-rotation flicking much less rewarding than the earlier thesis two-finger yaml.

### Remaining blocked/risky
- This is a config-level stronger gate; it does not yet add a separate single-finger XOR penalty or a free/compliant base.
- Stronger gate may reduce early learning speed and reward; visual stability should be prioritized over raw reward for this branch.

### Single recommended next step
- Run headed training to inspect init/contact:
  - `./docker-run-isaacgym.sh bash scripts/dexh13_lightbulb_teacher_sim2real_twofinger.sh 0 42 sim2real_twofinger_initpose_vis False wandb_activate=False task.env.randomization.randomizePDGains=False task.env.randomization.action_noise_e_scale=0.0 task.env.randomization.action_noise_t_scale=0.0 task.env.randomization.obs_noise_e_scale=0.0 task.env.randomization.obs_noise_t_scale=0.0 task.env.randomization.noisy_rpy_scale=0.0 task.env.randomization.noisy_pos_scale=0.0 task.env.forceScale=0.0 task.env.randomForceProbScalar=0.0 graphics_device_id=0`

---

## v2-172 (2026-04-29) -- Local Headed DexH13 Lightbulb YAML Train Check

### Target milestone/subgoal
- Run a local headed PPO training check on the current `Dexh13HoraLightbulb.yaml` so the user can inspect the live viewer behavior and confirm the YAML init relationship.

### What changed (files + behavior impact)
- No source/config files were changed.
- Created a local PPO teacher check output:
  - `outputs/Dexh13HoraLightbulb_teacher/local_headed_lightbulb_yaml_initcheck/`

### What was verified (commands + key outcomes)
- Command:
  - `./docker-run-isaacgym.sh timeout 600 bash scripts/dexh13_lightbulb_teacher.sh 0 42 local_headed_lightbulb_yaml_initcheck False wandb_activate=False task.env.object.init_pos_noise=[0.0,0.0,0.0] task.env.randomization.randomizeScale=False task.env.randomization.randomizeScaleList=[1.0] task.env.randomization.randomizePDGains=False task.env.randomization.action_noise_e_scale=0.0 task.env.randomization.action_noise_t_scale=0.0 task.env.randomization.obs_noise_e_scale=0.0 task.env.randomization.obs_noise_t_scale=0.0 task.env.randomization.noisy_rpy_scale=0.0 task.env.randomization.noisy_pos_scale=0.0 task.env.forceScale=0.0 task.env.randomForceProbScalar=0.0 graphics_device_id=0`
- Runtime config confirmed:
  - `task=Dexh13HoraLightbulb`
  - `headless=False`
  - `task.env.numEnvs=1`
  - `object.init_pos=[0.012,-0.018,0.0]`
  - `object.init_pos_noise=[0.0,0.0,0.0]`
  - `handRootPos=[0.064,0.014,0.229]`
  - `handRootRPY=[3.1415,0.526893,3.1415]`
  - `randomizeScale=False`, `randomizeScaleList=[1.0]`
  - `forceScale=0.0`, `randomForceProbScalar=0.0`
- Outcome:
  - Command exited cleanly with status `0`.
  - Saved `outputs/Dexh13HoraLightbulb_teacher/local_headed_lightbulb_yaml_initcheck/stage1_nn/best_reward_-776.96.pth`.
  - Early headed-training reward improved from about `-1440.49` to `-776.96`; this is a startup sanity check, not a performance result.

### Local conclusion
- Local headed PPO training starts successfully on the current `Dexh13HoraLightbulb.yaml`.
- With init noise, scale randomization, PD randomization, action/obs noise, and random external force disabled, the viewer uses the intended deterministic object/hand root init settings.

### Remaining blocked/risky
- This was a short headed sanity check with `numEnvs=1`; it is not comparable to the cloud 1h PPO result.
- Current local `Dexh13HoraLightbulb.yaml` has `reset_dist_threshold=0.14`, while the previous cloud 1h run used `0.10`.
- The YAML still contains `enable_nut_dof_vel: Truecc`, which remains a config-quality issue even though it has not crashed.

### Single recommended next step
- If the visual init/contact is still not acceptable, tune `env.asset.handRootPos/handRootRPY` and `env.object.init_pos` directly with deterministic viewer overrides before launching another cloud PPO run.

---

## v2-173 (2026-04-29) -- Frozen Initpose Viewer Added For DexH13 Lightbulb

### Target milestone/subgoal
- Separate three visual states that were being conflated:
  - static task reset init pose,
  - interactive initpose tuner pose,
  - PPO checkpoint rollout after policy actions.

### What changed (files + behavior impact)
- Added `scripts/view_dexh13_lightbulb_initpose_freeze.py`.
- The script composes the normal Hydra task config and opens a headed IsaacGym viewer, but does not execute PPO actions.
- It disables init position noise, mass/COM/friction/scale/PD randomization, action/obs/noisy pose terms, and random external forces.
- It prints the exact `object.init_pos`, `object.init_pos_noise`, `handRootPos`, and `handRootRPY` used by the viewer.

### What was verified (commands + key outcomes)
- Syntax checks:
  - `python3 -m py_compile scripts/view_dexh13_lightbulb_initpose_freeze.py`
  - `git diff --check -- scripts/view_dexh13_lightbulb_initpose_freeze.py`
  - Outcome: pass.
- Frozen viewer launch:
  - `./docker-run-isaacgym.sh python scripts/view_dexh13_lightbulb_initpose_freeze.py --task Dexh13HoraLightbulb --gpu 0 --num-envs 1`
  - Outcome: viewer launched and printed:
    - `object.init_pos: [0.012, -0.018, 0.0]`
    - `object.init_pos_noise: [0.0, 0.0, 0.0]`
    - `handRootPos: [0.064, 0.014, 0.229]`
    - `handRootRPY: [3.1415, 0.526893, 3.1415]`
- PPO visualization launch:
  - Visualized `outputs/Dexh13HoraLightbulb_teacher/dexh13_lightbulb_yaml_current_aggressive_s42_1h/stage1_nn/best_reward_948.07.pth`.
  - Used deterministic init overrides plus training-matched `task.env.reset_dist_threshold=0.10` and termination flags.

### Local conclusion
- Config inspection shows no evidence that PPO visualization is using a different `handRootPos`, `handRootRPY`, `handInitPose`, or `object.init_pos` than the trained task config.
- The visible difference between frozen/tuner screenshots and the PPO screenshot is most likely from the loaded PPO policy moving the active index/thumb after reset, not from an initpose override.
- Training-time `object.init_pos_noise=[0.005,0.005,0.0]` is positive-only in the task code and `randomizeScale=True` was active, but those changes are too small to explain the large index-fingertip separation by themselves.

### Remaining blocked/risky
- The `Dexh13HoraLightbulb.yaml` PPO checkpoint only reached `Current Best` around `948`, much lower than the earlier stable ThesisSim2Real teacher around `3316`, so poor visual grasp behavior is expected.
- The current reward/gate setup has `min_mult=0.05` and disabled fingertip tangent/torque rewards, so the policy can still receive some rotation reward even with weak or transient index contact.

### Single recommended next step
- If the goal is stable two-finger bulb grasp, do not treat this PPO checkpoint as a good policy; first retune init/contact target or restore stronger contact shaping, then rerun a short PPO probe and visualize early.

---

## v2-174 (2026-04-29) -- Sim2Real Two-Finger Rotate Reward Raised

### Target milestone/subgoal
- Make the new strong two-finger cooperative sim2real task less sparse by increasing the gated rotation reward while keeping the finger-distance reset threshold at the requested value.

### What changed (files + behavior impact)
- Updated `configs/task/Dexh13HoraLightbulbSim2RealTwoFinger.yaml`.
  - `task.env.reward.rotate_reward_scale`: `2.5 -> 6.0`.
  - Confirmed `task.env.reset_dist_threshold` is already `0.15`, so no reset-threshold edit was needed.
- No other task, train, or source files were changed for this adjustment.

### What was verified (commands + key outcomes)
- `git diff --check -- configs/task/Dexh13HoraLightbulbSim2RealTwoFinger.yaml`
  - Outcome: pass.
- Hydra compose probe:
  - `./docker-run-isaacgym.sh python -c "... task=Dexh13HoraLightbulbSim2RealTwoFinger ..."`
  - Outcome:
    - `reset_dist_threshold=0.15`
    - `rotate_reward_scale=6.0`
    - `enable_finger_dist=True`
    - `gate_min_mult=0.0`
    - `gate_power=3.0`

### Local conclusion
- The strong two-finger sim2real branch now has a larger rotation incentive while preserving the strict simultaneous-contact gate and finger-distance reset.

### Remaining blocked/risky
- Higher rotation reward can speed up learning, but may also reintroduce aggressive contact or slip if the gate is insufficient; visual inspection after short training remains necessary.

### Single recommended next step
- Run a short headed or 30min headless probe on `Dexh13HoraLightbulbSim2RealTwoFinger` and compare whether thumb/index now learn simultaneous contact instead of returning to single-finger flicking.

---

## v2-175 (2026-04-29) -- DexH13 Lightbulb Finger Reset Target Aligned To Bulb Head

### Target milestone/subgoal
- Fix the mismatch where `Dexh13HoraLightbulb.yaml` rewarded/gated index+thumb contact near the bulb head but terminated/proximity-shaped finger distance against raw `nut_pos`.

### What changed (files + behavior impact)
- Updated `dexscrew/tasks/xhand_hora.py`.
  - `env.finger_object_contact` now supports:
    - `target: nut_pos | object_pos`
    - `target_offset: [x, y, z]`
    - `scale_with_object`
    - `threshold_scale_with_object`
  - Proximity reward now measures thumb/other fingertip distances to this configured finger-contact target.
  - Finger-distance termination now uses the same configured target and can scale the threshold with object scale.
  - Defaults preserve previous behavior for tasks that do not set the new keys.
- Updated `configs/task/Dexh13HoraLightbulb.yaml`.
  - `reset_dist_threshold: 0.15`.
  - `finger_object_contact.target_offset: [0.0, 0.0, 0.084]`.
  - `finger_object_contact.scale_with_object: True`.
  - `finger_object_contact.threshold_scale_with_object: True`.
  - `termination.grace_steps: 150`.
  - `two_finger_gate.min_mult: 0.02`.

### What was verified (commands + key outcomes)
- Static checks:
  - `PYTHONDONTWRITEBYTECODE=1 python3 - <<'PY' ... compile(...) ... PY`
  - Outcome: `syntax_ok`.
  - `git diff --check -- dexscrew/tasks/xhand_hora.py configs/task/Dexh13HoraLightbulb.yaml scripts/view_dexh13_lightbulb_initpose_freeze.py docs/session_handoff_v2.md`
  - Outcome: pass.
- IsaacGym smoke:
  - `./docker-run-isaacgym.sh timeout 180 python train.py task=Dexh13HoraLightbulb headless=True seed=42 num_envs=2 task.env.numEnvs=2 train.algo=PPO train.ppo.minibatch_size=24 train.ppo.max_agent_steps=24 wandb_activate=False train.ppo.output_name=Dexh13HoraLightbulb_teacher/smoke_finger_contact_target_align_tmp`
  - Outcome: environment built and exited cleanly with `max steps achieved`.
  - Runtime config confirmed:
    - `finger_object_contact.target_offset=[0.0,0.0,0.084]`
    - `finger_object_contact.scale_with_object=True`
    - `finger_object_contact.threshold_scale_with_object=True`
    - `reset_dist_threshold=0.15`
    - `two_finger_gate.min_mult=0.02`

### Local conclusion
- The current `Dexh13HoraLightbulb.yaml` no longer has the direct contradiction where gate/reward targets the bulb head but finger-distance reset measures from raw `nut_pos`.
- This should make index/thumb exploration toward the useful bulb-head contact region less likely to be killed by reset, especially when object scale randomization samples larger bulbs.

### Remaining blocked/risky
- Existing PPO checkpoints were trained under the old mismatch; they must not be used to judge the new reset target behavior.
- `enable_nut_dof_vel: Truecc` remains in the YAML and is still a config-quality issue even though smoke training tolerates it.

### Single recommended next step
- Launch a fresh short PPO probe on the updated `Dexh13HoraLightbulb.yaml`, then visualize the best checkpoint before committing to a full cloud run.

---

## v2-176 (2026-04-29) -- Cloud DexH13 Lightbulb Target-Aligned PPO Unlimited Launch

### Target milestone/subgoal
- Start a cloud PPO teacher run using the updated `Dexh13HoraLightbulb.yaml` where finger reset/proximity targets are aligned to the bulb-head gate target.
- User explicitly requested no wall-clock limit.

### What changed (files + behavior impact)
- Added cloud pipeline script:
  - `outputs/cloud_pipeline_dexh13_lightbulb_targetalign_ppo_unlimited/run_ppo_unlimited_aggressive.sh`
- Synced current local training inputs to `cloud-training:/root/code/dexscrew-repro/`:
  - `configs/task/Dexh13HoraLightbulb.yaml`
  - `configs/train/Dexh13HoraLightbulb.yaml`
  - `dexscrew/tasks/xhand_hora.py`
  - `scripts/dexh13_lightbulb_teacher.sh`
  - pipeline script above.
- Launched remote tmux session:
  - `dexh13_lightbulb_targetalign_ppo_unlimited`

### What was verified (commands + key outcomes)
- Local script checks:
  - `bash -n outputs/cloud_pipeline_dexh13_lightbulb_targetalign_ppo_unlimited/run_ppo_unlimited_aggressive.sh`
  - `git diff --check -- outputs/cloud_pipeline_dexh13_lightbulb_targetalign_ppo_unlimited/run_ppo_unlimited_aggressive.sh`
  - Outcome: pass.
- Remote sync/config check:
  - Cloud `configs/task/Dexh13HoraLightbulb.yaml` contains:
    - `reset_dist_threshold: 0.15`
    - `finger_object_contact.target_offset: [0.0, 0.0, 0.084]`
    - `finger_object_contact.threshold_scale_with_object: True`
    - `termination.grace_steps: 150`
    - `two_finger_gate.min_mult: 0.02`
  - Cloud `dexscrew/tasks/xhand_hora.py` contains the new finger-contact target helper path.
- Cloud resource probe:
  - Candidate order: `12288/24576`, `8192/16384`, `6144/12288`.
  - `12288 envs / minibatch 24576` ran to the 180s probe timeout without OOM.
  - Selected setting:
    - `task.env.numEnvs=12288`
    - `train.ppo.minibatch_size=24576`
    - `num_threads=22`
- Formal PPO run:
  - Started at `2026-04-29T10:34:13+00:00`.
  - No formal `timeout` is applied, matching the user request.
  - Output root:
    - `outputs/Dexh13HoraLightbulb_teacher/dexh13_lightbulb_targetalign_ppo_unlimited_s42/`
  - Active process:
    - tmux session `dexh13_lightbulb_targetalign_ppo_unlimited`
    - `python train.py task=Dexh13HoraLightbulb ... train.ppo.output_name=Dexh13HoraLightbulb_teacher/dexh13_lightbulb_targetalign_ppo_unlimited_s42 task.env.numEnvs=12288 train.ppo.minibatch_size=24576`
  - Runtime config file confirms:
    - `reset_dist_threshold=0.15`
    - `finger_object_contact.threshold_scale_with_object=true`
    - `two_finger_gate.min_mult=0.02`
  - Initial checkpoint saved:
    - `outputs/Dexh13HoraLightbulb_teacher/dexh13_lightbulb_targetalign_ppo_unlimited_s42/stage1_nn/best_reward_361.88.pth`
  - GPU at status check:
    - `NVIDIA GeForce RTX 4090 D`, about `14309 MiB / 24564 MiB`, utilization around `61%`.

### Local conclusion
- The requested no-time-limit PPO teacher training is running on the cloud with the updated target-aligned YAML/code.
- The run has entered effective PPO training and is already saving `best_reward_*.pth` checkpoints.

### Remaining blocked/risky
- Because there is no wall-clock timeout, this run must be stopped manually when the user wants to inspect/sync.
- `latest.log` is noisy because `train.py` prints a large dirty git diff before normal training lines.
- The YAML still contains `enable_nut_dof_vel: Truecc`, which has not crashed but remains a config-quality issue.

### Single recommended next step
- Let the cloud PPO run continue until the user wants inspection, then stop/sync:
  - `ssh cloud-training`
  - `cd /root/code/dexscrew-repro`
  - `tmux attach -t dexh13_lightbulb_targetalign_ppo_unlimited`
  - monitor `outputs/cloud_pipeline_dexh13_lightbulb_targetalign_ppo_unlimited/latest.log`

---

## v2-177 (2026-04-29) -- Sim2Real Two-Finger Coactive Contribution Reward Added

### Target milestone/subgoal
- Respond to visual evidence that the strong contact gate still produced two-stage thumb/index behavior instead of true two-finger cooperation.

### What changed (files + behavior impact)
- Updated `dexscrew/tasks/xhand_hora.py`.
  - Added `aggregation` support for `env.fingertip_tangent_reward`:
    - `mean` preserves previous behavior.
    - `min` uses the weakest tracked fingertip's reward, so one active finger cannot hide the other inactive finger.
  - Added the same `aggregation` support for `env.fingertip_torque_reward`.
  - Added TensorBoard/W&B diagnostics:
    - `fingertip_tangent/reward_min`
    - `fingertip_torque/reward_min`
- Updated `configs/task/Dexh13HoraLightbulbSim2RealTwoFinger.yaml`.
  - `fingertip_tangent_reward.fingertip_indices: [0, 3]`.
  - `fingertip_tangent_reward.aggregation: min`.
  - `fingertip_tangent_reward.contact_force_min: 0.5`.
  - `fingertip_torque_reward.fingertip_indices: [0, 3]`.
  - `fingertip_torque_reward.aggregation: min`.
  - `fingertip_torque_reward.contact_force_min: 0.5`.

### What was verified (commands + key outcomes)
- Static checks:
  - `git diff --check -- dexscrew/tasks/xhand_hora.py configs/task/Dexh13HoraLightbulbSim2RealTwoFinger.yaml`
  - Outcome: pass.
  - `python - <<'PY' ... compile(Path('dexscrew/tasks/xhand_hora.py').read_text(), ...) ... PY`
  - Outcome: `compile_ok`.
  - Note: direct `py_compile` on the host hit a root-owned `__pycache__` permission error; the no-write compile path passed.
- Hydra compose probe:
  - `./docker-run-isaacgym.sh python -c "... task=Dexh13HoraLightbulbSim2RealTwoFinger ..."`
  - Outcome:
    - `tangent_indices=[0,3]`
    - `tangent_aggregation=min`
    - `torque_indices=[0,3]`
    - `torque_aggregation=min`
    - `rotate_reward_scale=6.0`
- IsaacGym smoke:
  - `./docker-run-isaacgym.sh timeout 180 python train.py task=Dexh13HoraLightbulbSim2RealTwoFinger headless=True seed=42 num_envs=4 train.algo=PPO train.ppo.minibatch_size=12 train.ppo.max_agent_steps=24 wandb_activate=False train.ppo.output_name=Dexh13HoraLightbulb_teacher_sim2real_twofinger/smoke_coactive_tmp`
  - Outcome: env built, runtime config showed coactive `[0,3]` min aggregation, and run exited cleanly after `max steps achieved`.

### Local conclusion
- The sim2real two-finger branch now distinguishes passive simultaneous contact from real coactive tangent/torque contribution.
- Previous strong-gate checkpoints should not be used to judge this new behavior because they were trained before the coactive reward change.

### Remaining blocked/risky
- This still does not force a free/compliant bulb base; it only shapes two-finger coactive behavior under the fixed-base simulator.
- `min` aggregation is stricter and may slow learning or lower reward; visual behavior should be prioritized over reward scale.

### Single recommended next step
- Train a fresh short probe on `Dexh13HoraLightbulbSim2RealTwoFinger` and visualize the new best checkpoint; check `fingertip_tangent/reward_min`, `fingertip_torque/reward_min`, and per-finger torque logs before deciding if a separate single-finger penalty is needed.

---

## v2-178 (2026-04-29) -- DexH13 Lightbulb PPO Stopped, Synced, And Initpose Retreated

### Target milestone/subgoal
- Stop the cloud `Dexh13HoraLightbulb` PPO teacher run on user request, sync the best model locally, and inspect the visual behavior.

### What changed (files + behavior impact)
- Stopped remote tmux run:
  - `dexh13_lightbulb_targetalign_ppo_unlimited`
- Synced cloud artifacts to local:
  - `outputs/Dexh13HoraLightbulb_teacher/dexh13_lightbulb_targetalign_ppo_unlimited_s42/`
  - `outputs/cloud_pipeline_dexh13_lightbulb_targetalign_ppo_unlimited/`
- Updated `configs/task/Dexh13HoraLightbulb.yaml` after visual inspection showed the trained policy was using an over-forward hand init pose:
  - `finger_object_contact.target_offset: [0.0, 0.0, 0.04]`
  - `two_finger_gate.target_offset: [0.0, 0.0, 0.04]`
  - disabled fingertip reward target offsets also aligned to `0.04`
  - `handRootPos: [0.11, 0.020, 0.217]`
  - `handRootRPY: [3.1415, 0.3, 3.1415]`
  - restored stable two-finger index/thumb initial joint angles from the sim2real/twofinger branch
  - fixed config typo `enable_nut_dof_vel: Truecc -> True`

### What was verified (commands + key outcomes)
- Remote stop/sync:
  - `tmux send-keys -t dexh13_lightbulb_targetalign_ppo_unlimited C-c`
  - Outcome: tmux exited, no remote PPO process remained.
  - Best synced checkpoint:
    - `outputs/Dexh13HoraLightbulb_teacher/dexh13_lightbulb_targetalign_ppo_unlimited_s42/stage1_nn/best_reward_7281.51.pth`
  - Last periodic checkpoint:
    - `outputs/Dexh13HoraLightbulb_teacher/dexh13_lightbulb_targetalign_ppo_unlimited_s42/stage1_nn/ep_2000_step_0295m_reward_7213.14.pth`
- Local config checks:
  - text check confirmed `target_offset=0.04`, `handRootPos=[0.11,0.020,0.217]`, and `enable_nut_dof_vel=True`
  - `git diff --check -- configs/task/Dexh13HoraLightbulb.yaml`
  - Outcome: pass.
- Local freeze viewer:
  - First headed attempt failed because two leftover Docker/IsaacGym containers were consuming most GPU memory.
  - Stopped containers `316ef324c97f` and `ff8bb809183d`.
  - GPU freed from about `14.8GB/16.4GB` to about `1.2GB/16.4GB`.
  - Relaunched freeze viewer:
    - `./docker-run-isaacgym.sh timeout 25 python scripts/view_dexh13_lightbulb_initpose_freeze.py --task Dexh13HoraLightbulb --gpu 0 --num-envs 1`
    - Outcome: environment built and printed corrected pose:
      - `handRootPos: [0.11, 0.02, 0.217]`
      - `handRootRPY: [3.1415, 0.3, 3.1415]`

### Local conclusion
- The high-reward `best_reward_7281.51.pth` run should not be treated as a valid final teacher because it was trained with the over-forward `[0.064, 0.014, 0.229] + pitch 0.526893` hand root pose.
- The corrected YAML is now closer to the stable two-finger/sim2real geometry and gives the index finger more usable tangent-rotation workspace.

### Remaining blocked/risky
- The synced checkpoint is useful for recordkeeping only; it is geometrically mismatched with the corrected YAML.
- A fresh PPO run is needed before judging whether reward and visual behavior now agree.

### Single recommended next step
- Run a fresh short PPO probe on corrected `Dexh13HoraLightbulb.yaml`, visualize the best checkpoint, and only then launch another no-limit/full cloud PPO run.

---

## v2-179 (2026-04-29) -- DexH13 Lightbulb Scale Range Moved To 1.15-1.25

### Target milestone/subgoal
- Adjust only the object scale distribution for `Dexh13HoraLightbulb.yaml` after visual inspection suggested mismatch between fixed-scale initpose tuning and scale-randomized training.

### What changed (files + behavior impact)
- Updated `configs/task/Dexh13HoraLightbulb.yaml`.
  - `baseObjScale: 1.0 -> 1.20`.
  - `randomizeScaleList: [1.0, 1.05, 1.10, 1.15] -> [1.175, 1.225]`.
  - `randomizeScaleMin/Max` and `randomizeScaleLower/Upper`: `1.15 / 1.25`.
  - Added a short comment that task code samples each listed scale with `+/-0.025`.
- Did not modify `handRootPos`, `handRootRPY`, or `handInitPose` in this update.

### What was verified (commands + key outcomes)
- `git diff --check -- configs/task/Dexh13HoraLightbulb.yaml`
  - Outcome: pass.
- Docker/Hydra compose probe:
  - `baseObjScale=1.2`
  - `randomizeScale=True`
  - `randomizeScaleList=[1.175, 1.225]`
  - `randomizeScaleMinMax=1.15 1.25`
  - initpose fields still resolve to:
    - `handRootPos=[0.11, 0.02, 0.217]`
    - `handRootRPY=[3.1415, 0.3, 3.1415]`
    - `right_thumb_joint_0=-0.33`
    - `right_index_joint_1=0.925279522`

### Local conclusion
- Future `Dexh13HoraLightbulb` runs will train near the stable 1.20 bulb scale while still covering approximately `1.15-1.25`.
- The already-running local headed 10-env PPO process was launched before this edit and still uses its originally parsed scale config.

### Remaining blocked/risky
- The task code currently ignores `randomizeScaleMin/Max` for actual sampling and uses `randomizeScaleList[i] +/- 0.025`; the chosen centers therefore implement the intended approximate range.

### Single recommended next step
- Restart the local headed 10-env PPO if the user wants to inspect the new 1.15-1.25 scale distribution.

---

## v2-180 (2026-04-29) -- Cloud DexH13 Thesis Initpose Scale115125 PPO 1h Launch

### Target milestone/subgoal
- Restore `Dexh13HoraLightbulb.yaml` init pose to the thesis stable keyboard-tuned pose and launch a 1h cloud PPO teacher run.

### What changed (files + behavior impact)
- Updated `configs/task/Dexh13HoraLightbulb.yaml`.
  - Restored thesis stable initpose from `outputs/initpose_tuning/lightbulb_initpose.yaml`:
    - `handRootPos: [0.064000, 0.014000, 0.229000]`
    - `handRootRPY: [3.141500, 0.526893, 3.141500]`
    - `right_index_joint_1: 1.1252793074`
    - `right_index_joint_2: 0.3028971255`
    - `right_index_joint_3: 0.3092995286`
    - `right_thumb_joint_0: 0.0`
    - `right_thumb_joint_2: 0.0599999987`
    - `right_thumb_joint_3: 1.0319659710`
  - Kept the previous scale update:
    - `baseObjScale: 1.20`
    - `randomizeScaleList: [1.175, 1.225]`
    - approximate runtime scale range `1.15-1.25`.
- Added cloud launch script:
  - `outputs/cloud_pipeline_dexh13_lightbulb_thesis_initpose_scale115125_ppo1h/run_ppo_1h_aggressive.sh`
  - Uses `timeout 3600`, `task.env.numEnvs=12288`, `train.ppo.minibatch_size=24576`.
- Synced to `cloud-training:/root/code/dexscrew-repro/`:
  - `configs/task/Dexh13HoraLightbulb.yaml`
  - `configs/train/Dexh13HoraLightbulb.yaml`
  - `dexscrew/tasks/xhand_hora.py`
  - launch script above.

### What was verified (commands + key outcomes)
- Local checks:
  - `bash -n outputs/cloud_pipeline_dexh13_lightbulb_thesis_initpose_scale115125_ppo1h/run_ppo_1h_aggressive.sh`
  - `git diff --check -- configs/task/Dexh13HoraLightbulb.yaml outputs/cloud_pipeline_dexh13_lightbulb_thesis_initpose_scale115125_ppo1h/run_ppo_1h_aggressive.sh`
  - Outcome: pass.
- Local Docker/Hydra compose probe confirmed:
  - `baseObjScale=1.2`
  - `randomizeScaleList=[1.175, 1.225]`
  - `handRootPos=[0.064, 0.014, 0.229]`
  - `handRootRPY=[3.1415, 0.526893, 3.1415]`
  - index/thumb initial joints match the thesis stable snippet.
- Remote preflight:
  - No existing `Dexh13HoraLightbulb` tmux/process.
  - GPU idle before launch: `NVIDIA GeForce RTX 4090 D`, `1 MiB / 24564 MiB`.
- Remote launch:
  - tmux session:
    - `dexh13_lightbulb_thesis_initpose_scale115125_ppo1h`
  - run output:
    - `outputs/Dexh13HoraLightbulb_teacher/dexh13_lightbulb_thesis_initpose_scale115125_s42_1h/`
  - pipeline log:
    - `outputs/cloud_pipeline_dexh13_lightbulb_thesis_initpose_scale115125_ppo1h/latest.log`
  - Runtime config markers in remote log confirm:
    - `numEnvs: 12288`
    - `baseObjScale: 1.2`
    - `randomizeScaleList: [1.175, 1.225]`
    - `handRootPos: [0.064, 0.014, 0.229]`
    - `handRootRPY: [3.1415, 0.526893, 3.1415]`
    - `right_index_joint_1: 1.1252793074`
    - `right_thumb_joint_3: 1.031965971`
  - First status check:
    - process alive under `timeout 3600`
    - GPU about `14389 MiB / 24564 MiB`, util about `52%`
    - first best checkpoint saved:
      - `stage1_nn/best_reward_222.58.pth`

### Local conclusion
- The requested 1h cloud PPO run is active and has entered training with the thesis stable initpose plus scale range approximately `1.15-1.25`.

### Remaining blocked/risky
- The run is still in progress and should finish by timeout status `124`; this is expected for the requested 1h wall-clock limit.
- Visual quality still needs inspection after sync because reward alone may not capture index-finger usefulness.

### Single recommended next step
- After the 1h timeout finishes, sync `outputs/Dexh13HoraLightbulb_teacher/dexh13_lightbulb_thesis_initpose_scale115125_s42_1h/` locally and visualize the best checkpoint.

---

## v2-181 (2026-04-30) -- Sim2Real Coactive Reward Scale Restored

### Target milestone/subgoal
- Keep the visually acceptable coactive two-finger behavior, but restore the main rotation reward scale to the earlier thesis-twofinger value for future training comparability.

### What changed (files + behavior impact)
- Updated `configs/task/Dexh13HoraLightbulbSim2RealTwoFinger.yaml`.
  - `task.env.reward.rotate_reward_scale`: `6.0 -> 2.5`.
- Preserved the coactive behavior-shaping settings:
  - `fingertip_tangent_reward.fingertip_indices: [0, 3]`.
  - `fingertip_tangent_reward.aggregation: min`.
  - `fingertip_torque_reward.fingertip_indices: [0, 3]`.
  - `fingertip_torque_reward.aggregation: min`.

### What was verified (commands + key outcomes)
- `git diff --check -- configs/task/Dexh13HoraLightbulbSim2RealTwoFinger.yaml`
  - Outcome: pass.
- Docker/Hydra compose probe:
  - `rotate_reward_scale=2.5`
  - `tangent_indices=[0,3]`
  - `tangent_aggregation=min`
  - `torque_indices=[0,3]`
  - `torque_aggregation=min`
  - `gate_power=3.0`

### Local conclusion
- Future `Dexh13HoraLightbulbSim2RealTwoFinger` runs use the original rotation-reward magnitude while retaining the coactive two-finger contribution requirement.
- Existing checkpoint `sim2real_twofinger_coactive_r6_s42_30m/best_reward_6083.13.pth` was trained with `rotate_reward_scale=6.0`; it remains useful as a visual behavior reference but is not reward-scale comparable to future 2.5 runs.

### Remaining blocked/risky
- Lowering the main rotate reward may slow learning under the strict coactive min aggregation; if fresh training collapses or does not learn, compare behavior rather than raw reward first.

### Single recommended next step
- If retraining this branch, use a fresh cache name such as `sim2real_twofinger_coactive_r25_s42_30m` to avoid mixing reward-scale regimes.

---

## v2-182 (2026-04-30) -- Cloud DexH13 Thesis Initpose PPO 1h Synced And Visualized

### Target milestone/subgoal
- Confirm the cloud 1h PPO teacher run finished, sync the checkpoint locally, and inspect the headed viewer.

### What changed (files + behavior impact)
- Synced cloud artifacts locally:
  - `outputs/Dexh13HoraLightbulb_teacher/dexh13_lightbulb_thesis_initpose_scale115125_s42_1h/`
  - `outputs/cloud_pipeline_dexh13_lightbulb_thesis_initpose_scale115125_ppo1h/`
- Added local visual check captures:
  - `outputs/visual_checks/dexh13_lightbulb_thesis_initpose_scale115125_best.png`
  - `outputs/visual_checks/dexh13_lightbulb_thesis_initpose_scale115125_best_now.png`

### What was verified (commands + key outcomes)
- Remote completion check:
  - tmux session ended.
  - no `python train.py task=Dexh13HoraLightbulb` process remained.
  - pipeline ended at `2026-04-29T16:43:11+00:00`.
  - `train_exit_status=124`, expected for `timeout 3600`.
  - remote GPU returned idle.
- Best synced checkpoint:
  - `outputs/Dexh13HoraLightbulb_teacher/dexh13_lightbulb_thesis_initpose_scale115125_s42_1h/stage1_nn/best_reward_5352.78.pth`
- Local headed viewer command used:
  - `./docker-run-isaacgym.sh bash scripts/vis_dexh13_lightbulb_teacher.sh 0 42 dexh13_lightbulb_thesis_initpose_scale115125_s42_1h test=True checkpoint=outputs/Dexh13HoraLightbulb_teacher/dexh13_lightbulb_thesis_initpose_scale115125_s42_1h/stage1_nn/best_reward_5352.78.pth graphics_device_id=0 task.env.randomization.randomizeScale=False task.env.randomization.randomizeMass=False task.env.randomization.randomizeCOM=False task.env.randomization.randomizeFriction=False task.env.randomization.randomizePDGains=False task.env.object.init_pos_noise=[0.0,0.0,0.0] task.env.forceScale=0.0 task.env.randomForceProbScalar=0.0`
- Runtime config printed by the viewer confirmed:
  - `test: True`
  - checkpoint path above
  - `baseObjScale: 1.2`
  - domain randomization/noise disabled for visual inspection
  - thesis stable `handRootPos`, `handRootRPY`, index/thumb initial joints.

### Local conclusion
- The requested 1h cloud PPO run completed and the best checkpoint is available locally.
- Initial visual inspection shows the policy starts from the intended thesis stable pose and does not show the earlier gross initpose mismatch. The index still appears relatively low/left of the bulb and the behavior remains conservative, so visual quality should be judged from the live viewer rather than reward alone.

### Remaining blocked/risky
- The current 1h policy may still underuse the index finger even with the corrected initpose and scale range.
- Viewer was launched with randomization disabled to isolate policy behavior at the nominal 1.20 bulb scale.

### Single recommended next step
- Inspect the live viewer; if index contact is still weak, run the next ablation by adding explicit index contact/tangent shaping rather than changing initpose again.

---

## v2-183 (2026-04-30) -- DexH13 Lightbulb Fixed Scale No-Randomization Config

### Target milestone/subgoal
- Remove scale/domain-randomization as a confounder between keyboard init-pose tuning and PPO training.

### What changed (files + behavior impact)
- Updated `configs/task/Dexh13HoraLightbulb.yaml`.
  - Fixed object scale to the keyboard tuning value:
    - `baseObjScale: 1.20`
    - `randomizeScale: False`
    - `randomizeScaleList: [1.20]`
    - `randomizeScaleMin/Max/Lower/Upper: 1.20`
  - Disabled training perturbations:
    - `randomizeMass: False`
    - `randomizeCOM: False`
    - `randomizeFriction: False`
    - `randomizePDGains: False`
    - obs/action/noisy pose scales all `0.0`
    - `forceScale: 0.0`
    - `randomForceProbScalar: 0.0`
    - `object.init_pos_noise: [0.0, 0.0, 0.0]`
- Synced the updated task YAML to:
  - `cloud-training:/root/code/dexscrew-repro/configs/task/Dexh13HoraLightbulb.yaml`

### What was verified (commands + key outcomes)
- `git diff --check -- configs/task/Dexh13HoraLightbulb.yaml`
  - Outcome: pass.
- Docker/Hydra compose probe confirmed the training entrypoint resolves:
  - `baseObjScale=1.2`
  - `randomizeScale=False`
  - `randomizeScaleList=[1.2]`
  - `scaleLowerUpper=1.2 1.2`
  - `mass/com/friction/pd=False False False False`
  - obs/action/noisy scales all `0.0`
  - `object_init_pos_noise=[0.0, 0.0, 0.0]`
  - thesis stable hand root and thumb/index init joints unchanged.
- Remote grep check confirmed the cloud YAML has the same fixed-scale/no-randomization values.

### Local conclusion
- A fresh PPO run from this YAML should now match the keyboard init-pose tuning setup for scale and remove domain-randomization effects.
- Existing checkpoints remain trained under their original configs and should not be used to judge this fixed-scale change.

### Remaining blocked/risky
- If thumb/index still look wrong after retraining from this fixed-scale config, the likely causes move to geometry/contact/reward/action behavior rather than scale randomization.

### Single recommended next step
- Sync this config to the cloud and launch a fresh PPO teacher run with a new cache name, e.g. `dexh13_lightbulb_fixed120_nodr_s42_1h`.

---

## v2-184 (2026-04-30) -- Cloud DexH13 Fixed120 NoDR PPO 20m Launch

### Target milestone/subgoal
- Launch a short cloud PPO teacher run from the current fixed-scale/no-randomization `Dexh13HoraLightbulb.yaml` so the user can later visualize whether the pose is correct.

### What changed (files + behavior impact)
- Added cloud launch script:
  - `outputs/cloud_pipeline_dexh13_lightbulb_fixed120_nodr_ppo20m/run_ppo_20m_aggressive.sh`
- Script behavior:
  - `timeout 1200`
  - task `Dexh13HoraLightbulb`
  - `headless=True`
  - seed `42`
  - `task.env.numEnvs=12288`
  - `train.ppo.minibatch_size=24576`
  - output run:
    - `outputs/Dexh13HoraLightbulb_teacher/dexh13_lightbulb_fixed120_nodr_s42_20m/`
  - pipeline log:
    - `outputs/cloud_pipeline_dexh13_lightbulb_fixed120_nodr_ppo20m/latest.log`
- Synced to cloud:
  - `configs/task/Dexh13HoraLightbulb.yaml`
  - `configs/train/Dexh13HoraLightbulb.yaml`
  - `dexscrew/tasks/xhand_hora.py`
  - launch script above.

### What was verified (commands + key outcomes)
- Local checks:
  - `bash -n outputs/cloud_pipeline_dexh13_lightbulb_fixed120_nodr_ppo20m/run_ppo_20m_aggressive.sh`
  - `git diff --check -- ...`
  - Outcome: pass.
- Remote preflight:
  - no existing tmux session
  - no matching `Dexh13HoraLightbulb` training process
  - GPU idle: `NVIDIA GeForce RTX 4090 D`, `1 MiB / 24564 MiB`, `0%`
- Remote YAML/script checks:
  - script syntax pass
  - remote YAML confirms:
    - `baseObjScale: 1.20`
    - `randomizeScale: False`
    - `randomizeScaleList: [1.20]`
    - mass/COM/friction/PD randomization disabled
    - obs/action/noisy/object init noise disabled
    - thesis stable `handRootPos`, `handRootRPY`, index/thumb init joints.
- Remote launch:
  - tmux session:
    - `dexh13_lightbulb_fixed120_nodr_ppo20m`
  - running command:
    - `timeout 1200 python train.py task=Dexh13HoraLightbulb ...`
  - after about 2m14s:
    - GPU `14339 MiB / 24564 MiB`, util about `89%`
    - first checkpoint saved:
      - `stage1_nn/best_reward_104.19.pth`
    - startup log includes:
      - `Generated 5000 random initial poses for XHand at scale 1.2`

### Local conclusion
- The requested 20m cloud PPO run is active and is using the fixed 1.20/no-domain-randomization configuration.
- This run is the correct one to visualize for the pose sanity check; older checkpoints should not be used for this comparison.

### Remaining blocked/risky
- The 20m run is still in progress and should naturally exit with timeout status `124`.
- It is a short training run, so behavior quality may be rough; the main purpose is pose sanity rather than final policy quality.

### Single recommended next step
- After the 20m timeout completes, sync `outputs/Dexh13HoraLightbulb_teacher/dexh13_lightbulb_fixed120_nodr_s42_20m/` and visualize its best checkpoint locally.

---

## v2-183 (2026-04-30) -- Sim2Real Coactive Rotate Reward Set To 3.5

### Target milestone/subgoal
- Restore a moderate rotation reward scale for the coactive sim2real two-finger task: stronger than the original `2.5`, but less aggressive than the temporary `6.0`.

### What changed (files + behavior impact)
- Updated `configs/task/Dexh13HoraLightbulbSim2RealTwoFinger.yaml`.
  - `task.env.reward.rotate_reward_scale`: `2.5 -> 3.5`.
- Preserved the coactive two-finger settings:
  - `fingertip_tangent_reward.fingertip_indices: [0, 3]`.
  - `fingertip_tangent_reward.aggregation: min`.
  - `fingertip_torque_reward.fingertip_indices: [0, 3]`.
  - `fingertip_torque_reward.aggregation: min`.
  - `termination.enable_finger_dist: True`.

### What was verified (commands + key outcomes)
- `git diff --check -- configs/task/Dexh13HoraLightbulbSim2RealTwoFinger.yaml`
  - Outcome: pass.
- Docker/Hydra compose probe confirmed:
  - `rotate_reward_scale=3.5`.
  - `tangent_indices=[0,3]`.
  - `tangent_aggregation=min`.
  - `torque_indices=[0,3]`.
  - `torque_aggregation=min`.
  - `enable_finger_dist=True`.

### Local conclusion
- Future coactive sim2real two-finger training now uses the intended moderate rotation reward scale.
- The existing `sim2real_twofinger_coactive_r6_s42_30m` checkpoint was trained with `rotate_reward_scale=6.0`; use a fresh cache for any 3.5 run.

### Remaining blocked/risky
- Because reward scale changed again, raw reward values are not directly comparable between `r6` and future `r35` runs.

### Single recommended next step
- If retraining, use a new cache name such as `sim2real_twofinger_coactive_r35_s42_30m`.

---

## v2-185 (2026-04-30) -- Fixed120 NoDR PPO20m Synced And Visualized

### Target milestone/subgoal
- Sync the completed fixed-scale/no-randomization 20m PPO teacher checkpoint and open local headed visualization.

### What changed (files + behavior impact)
- Synced cloud artifacts locally:
  - `outputs/Dexh13HoraLightbulb_teacher/dexh13_lightbulb_fixed120_nodr_s42_20m/`
  - `outputs/cloud_pipeline_dexh13_lightbulb_fixed120_nodr_ppo20m/`
- Added local visual captures:
  - `outputs/visual_checks/dexh13_lightbulb_fixed120_nodr_best3461.png`
  - `outputs/visual_checks/dexh13_lightbulb_fixed120_nodr_best3461_t2.png`
  - `outputs/visual_checks/dexh13_lightbulb_fixed120_nodr_best3461_zoom.png`
  - `outputs/visual_checks/dexh13_lightbulb_fixed120_nodr_best3461_rotated.png`
  - `outputs/visual_checks/dexh13_lightbulb_fixed120_nodr_best3461_focus.png`

### What was verified (commands + key outcomes)
- Remote completion check:
  - no tmux session and no remote training process remained.
  - remote GPU idle.
  - `train_exit_status=124`, expected for `timeout 1200`.
  - best checkpoint:
    - `outputs/Dexh13HoraLightbulb_teacher/dexh13_lightbulb_fixed120_nodr_s42_20m/stage1_nn/best_reward_3461.19.pth`
- Local headed viewer command:
  - `./docker-run-isaacgym.sh bash scripts/vis_dexh13_lightbulb_teacher.sh 0 42 dexh13_lightbulb_fixed120_nodr_s42_20m test=True checkpoint=outputs/Dexh13HoraLightbulb_teacher/dexh13_lightbulb_fixed120_nodr_s42_20m/stage1_nn/best_reward_3461.19.pth graphics_device_id=0`
- Viewer runtime config confirmed:
  - checkpoint path above
  - `baseObjScale: 1.2`
  - `randomizeScale: False`
  - `randomizeScaleList: [1.2]`
  - mass/COM/friction/PD/noise/force/object init noise disabled
  - `Generated 5000 random initial poses for XHand at scale 1.2`

### Local conclusion
- The checkpoint is synced and the local viewer is open.
- The viewer confirms the correct fixed-scale/no-randomization config is loaded.
- The captured policy view is not visually good: the hand is visible but the bulb is not clearly in frame after rollout, suggesting the policy may separate from or lose the object quickly. This needs direct interactive inspection before deciding whether the underlying init pose is still wrong.

### Remaining blocked/risky
- A separate local `Dexh13HoraLightbulbSim2RealTwoFinger` training process was already running and was left untouched.
- The current 20m PPO is short and should be treated as pose/behavior sanity evidence, not a final policy.

### Single recommended next step
- Inspect the live viewer interactively; if the bulb is already gone immediately after reset, run a paused/zero-action init-pose viewer for this exact YAML to isolate initial pose from policy action.

---

## v2-186 (2026-04-30) -- DexH13 Current YAML Initpose Tuner Opened

### Target milestone/subgoal
- Open the keyboard-controlled init-pose tuner using the current `Dexh13HoraLightbulb.yaml`.

### What changed (files + behavior impact)
- No source/config behavior changed.
- Closed the previous local PPO teacher viewer process to avoid viewer/window confusion.
- Left the unrelated local `Dexh13HoraLightbulbSim2RealTwoFinger` training process untouched.

### What was verified (commands + key outcomes)
- Launched:
  - `./docker-run-isaacgym.sh python scripts/tune_dexh13_lightbulb_initpose.py --task Dexh13HoraLightbulb --gpu 0 --seed 42 --out outputs/initpose_tuning/Dexh13HoraLightbulb_current_fixed120.yaml`
- Tuner startup log confirms:
  - object type `screw_contactviz`
  - `Generated 5000 random initial poses for XHand at scale 1.2`
  - mode `hand`
  - `pos=[0.064000, 0.014000, 0.229000]`
  - `rpy=[3.141500, 0.526893, 3.141500]`
  - first joint `right_index_joint_0:0.000000`
- Captured current viewer image:
  - `outputs/visual_checks/dexh13_lightbulb_initpose_tuner_current_fixed120.png`

### Local conclusion
- The keyboard tuner is open and reading the current `Dexh13HoraLightbulb.yaml` fixed-1.20 settings.

### Remaining blocked/risky
- Current captured camera view still mainly shows the hand, so direct interactive camera adjustment may be needed to inspect the bulb-hand relation clearly.

### Single recommended next step
- Use the live tuner to inspect/adjust the pose; press `O` to save the YAML snippet to `outputs/initpose_tuning/Dexh13HoraLightbulb_current_fixed120.yaml`.

---

## v2-187 (2026-04-30) -- Sim2Real TwoFinger Scale115125 PPO30m Probe

### Target milestone/subgoal
- Test whether enabling bulb scale randomization around the nominal 1.20 size hurts the coactive sim2real two-finger PPO behavior.

### What changed (files + behavior impact)
- Updated `configs/task/Dexh13HoraLightbulbSim2RealTwoFinger.yaml`.
  - `randomizeScale: True`
  - `randomizeScaleList: [1.175, 1.225]`
  - `randomizeScaleMin/Max/Lower/Upper: 1.15 / 1.25`
  - Restored the intended moderate rotation reward:
    - `rotate_reward_scale: 3.5`
- An accidental short launch with cache `sim2real_twofinger_coactive_r25_scale115125_s42_30m` was stopped before the final run because it used `rotate_reward_scale=2.5`.

### What was verified (commands + key outcomes)
- Docker/Hydra compose probe confirmed:
  - `rotate_reward_scale=3.5`
  - `randomizeScale=True`
  - `randomizeScaleList=[1.175, 1.225]`
  - `randomizeScaleMin/Max/Lower/Upper=1.15/1.25`
  - `fingertip_tangent_reward.aggregation=min`
  - `fingertip_torque_reward.aggregation=min`
  - `termination.enable_finger_dist=True`
- Training command:
  - `./docker-run-isaacgym.sh timeout 1800 scripts/run_with_cleanup.sh bash scripts/dexh13_lightbulb_teacher_sim2real_twofinger.sh 0 42 sim2real_twofinger_coactive_r35_scale115125_s42_30m True wandb_activate=True task.env.termination.log=True task.env.numEnvs=4096 train.ppo.minibatch_size=8192`
- Outcome:
  - process exited with status `124`, expected for the 30-minute timeout wrapper.
  - final stdout best:
    - `Current Best: 3101.83`
  - checkpoint:
    - `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger/sim2real_twofinger_coactive_r35_scale115125_s42_30m/stage1_nn/best_reward_3101.83.pth`
  - TensorBoard event:
    - `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger/sim2real_twofinger_coactive_r35_scale115125_s42_30m/stage1_tb/events.out.tfevents.1777482697.wbz-ubuntu22-pc`
- Post-run process check:
  - no matching `Dexh13HoraLightbulbSim2RealTwoFinger` / `train.py` training process remained.
  - GPU still had the unrelated `scripts/tune_dexh13_lightbulb_initpose.py` process open; it was left untouched.
- `git diff --check -- configs/task/Dexh13HoraLightbulbSim2RealTwoFinger.yaml docs/session_handoff_v2.md`
  - Outcome before this handoff append: pass.

### Local conclusion
- Scale randomization `1.15-1.25` did not collapse short PPO training under the coactive two-finger reward.
- The 30-minute wall-clock run reached `best_reward_3101.83`, which is strong enough to justify visual inspection before deciding whether to keep this domain randomization range.

### Remaining blocked/risky
- The actual IsaacGym collect time was about 26 minutes because environment startup consumes part of the `timeout 1800` wall clock.
- Reward alone cannot confirm whether the two fingers remain visually cooperative across the randomized scale range.

### Single recommended next step
- Visualize `best_reward_3101.83.pth` from `sim2real_twofinger_coactive_r35_scale115125_s42_30m` and compare against the previous fixed-scale/coactive viewer behavior.

---

## v2-188 (2026-04-30) -- Sim2Real Scale115125 PPO Best Viewer Opened

### Target milestone/subgoal
- Open headed visualization for the `Sim2RealTwoFinger` scale-randomized PPO best checkpoint so the user can inspect behavior.

### What changed (files + behavior impact)
- No source/config behavior changed.
- Closed the old local initpose tuner process before opening the PPO viewer to avoid window/GPU confusion.

### What was verified (commands + key outcomes)
- Checkpoint exists:
  - `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger/sim2real_twofinger_coactive_r35_scale115125_s42_30m/stage1_nn/best_reward_3101.83.pth`
- Viewer command launched:
  - `./docker-run-isaacgym.sh bash scripts/dexh13_lightbulb_teacher_sim2real_twofinger.sh 0 42 sim2real_twofinger_coactive_r35_scale115125_s42_30m False test=True checkpoint=outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger/sim2real_twofinger_coactive_r35_scale115125_s42_30m/stage1_nn/best_reward_3101.83.pth graphics_device_id=0`
- Runtime config confirmed:
  - `test=True`
  - `headless=False`
  - `numEnvs=1`
  - checkpoint path above
  - `rotate_reward_scale=3.5`
  - `randomizeScale=True`
  - `randomizeScaleList=[1.175, 1.225]`
  - generated random initial poses for both scale centers `1.175` and `1.225`

### Local conclusion
- The correct scale-randomized PPO best viewer is open for direct visual inspection.

### Remaining blocked/risky
- Visual quality has not yet been judged in this handoff entry; user inspection is the source of truth for whether scale randomization improved sim2real behavior.

### Single recommended next step
- Inspect whether index/thumb stay coactive and stable across resets; if visual behavior is acceptable, keep `randomizeScale=1.15-1.25` for the next PPO/student-distillation branch.

---

## v2-189 (2026-04-30) -- Initpose Tuner Command Documented

### Target milestone/subgoal
- Give the user the local keyboard init-pose tuner command and document the key controls.

### What changed (files + behavior impact)
- Added `tele_readme.md`.
  - Documents the local command for `scripts/tune_dexh13_lightbulb_initpose.py`.
  - Lists keyboard controls and print/save behavior.
  - Notes the clean one-env no-randomization overrides used by the tuner.
- Closed existing local Isaac Gym tuner/viewer processes so the user can run the command and see terminal print output directly.

### What was verified (commands + key outcomes)
- Inspected `scripts/tune_dexh13_lightbulb_initpose.py` for exact key bindings.
- Process/window check after cleanup:
  - no active `tune_dexh13_lightbulb_initpose.py`.
  - no active Isaac Gym window.
  - no active Isaac Gym Python compute process besides normal desktop GPU users.

### Local conclusion
- The user can now run the documented command locally and view all `print()` output in their own terminal.

### Remaining blocked/risky
- None for documentation; actual pose adjustment remains manual/interactive.

### Single recommended next step
- Run the command in `tele_readme.md`, adjust the pose, press `C` to print values or `O` to save the YAML snippet.

---

## v2-190 (2026-04-30) -- DexH13 Lightbulb Initpose Updated And Freeze Viewer Opened

### Target milestone/subgoal
- Apply the user's newly tuned keyboard initpose to `Dexh13HoraLightbulb.yaml` and open a local no-policy frozen initpose viewer.

### What changed (files + behavior impact)
- Updated `configs/task/Dexh13HoraLightbulb.yaml`.
  - `handRootPos: [0.086000, 0.014000, 0.243000]`
  - `handRootRPY: [3.141500, 0.422173, 3.141500]`
  - `right_index_joint_1: 0.9852794409`
  - `right_index_joint_2: 0.2828971148`
  - `right_index_joint_3: 0.2892995179`
  - `right_middle_joint_0: 0.0010000000`
  - `right_thumb_joint_1: 1.5700000525`
  - `right_thumb_joint_2: 0.0000000000`
  - `right_thumb_joint_3: 0.9119660854`
- Closed the previous local tuner process before opening the freeze viewer.

### What was verified (commands + key outcomes)
- `git diff --check -- configs/task/Dexh13HoraLightbulb.yaml`
  - Outcome: pass.
- Docker/Hydra compose probe confirmed:
  - `handRootPos=[0.086, 0.014, 0.243]`
  - `handRootRPY=[3.1415, 0.422173, 3.1415]`
  - updated index/thumb/middle0 init joints
  - `baseObjScale=1.2`
  - `randomizeScale=False`
  - `randomizeScaleList=[1.2]`
- Launched local frozen viewer:
  - `./docker-run-isaacgym.sh python scripts/view_dexh13_lightbulb_initpose_freeze.py --task Dexh13HoraLightbulb --gpu 0 --seed 42 --num-envs 1`
- Viewer stdout confirms:
  - `Generated 5000 random initial poses for XHand at scale 1.2`
  - `object.init_pos: [0.012, -0.018, 0.0]`
  - `object.init_pos_noise: [0.0, 0.0, 0.0]`
  - `handRootPos: [0.086, 0.014, 0.243]`
  - `handRootRPY: [3.1415, 0.422173, 3.1415]`
- Captured current default-view screenshot:
  - `outputs/visual_checks/dexh13_lightbulb_new_initpose_freeze.png`

### Local conclusion
- The task YAML now contains the user's latest tuned initpose.
- A local frozen/no-policy viewer is open for inspecting the initial geometry.

### Remaining blocked/risky
- The default viewer camera is top-down and the screenshot mostly shows the hand, so rotate/zoom the live viewer to inspect the bulb-hand contact relation directly.

### Single recommended next step
- If the pose looks correct in the frozen viewer, use this YAML as the next PPO training baseline and avoid judging initpose through an already-moving PPO policy viewer.

---

## v2-191 (2026-04-30) -- DexH13 Lightbulb Domain Randomization Restored

### Target milestone/subgoal
- Keep the user's latest tuned initpose and restore the domain randomization settings that were temporarily disabled for pose inspection.

### What changed (files + behavior impact)
- Updated `configs/task/Dexh13HoraLightbulb.yaml`.
  - Preserved latest tuned initpose:
    - `handRootPos: [0.086000, 0.014000, 0.243000]`
    - `handRootRPY: [3.141500, 0.422173, 3.141500]`
    - updated index/thumb/middle0 initial joints from the user's keyboard tuner snippet.
  - Restored training perturbations:
    - `forceScale: 2.0`
    - `randomForceProbScalar: 0.25`
    - `randomizeMass: True`
    - `randomizeCOM: True`
    - `randomizeFriction: True`
    - `randomizePDGains: True`
    - obs/action/noisy pose scales restored to `0.01/0.005/0.1/0.02`
    - `object.init_pos_noise: [0.005, 0.005, 0.0]`
  - Restored bulb scale randomization:
    - `baseObjScale: 1.20`
    - `randomizeScale: True`
    - `randomizeScaleList: [1.175, 1.225]`
    - `randomizeScaleMin/Max/Lower/Upper: 1.15 / 1.25`

### What was verified (commands + key outcomes)
- `git diff --check -- configs/task/Dexh13HoraLightbulb.yaml`
  - Outcome: pass.
- Docker/Hydra compose probe confirmed:
  - latest tuned hand root and init joints are still active.
  - `baseObjScale=1.2`
  - `randomizeScale=True`
  - `randomizeScaleList=[1.175, 1.225]`
  - scale bounds `1.15 1.25 1.15 1.25`
  - `mass/com/friction/pd=True True True True`
  - obs/action/noisy scales `0.01 0.005 0.01 0.005 0.1 0.02`
  - `force=2.0 0.25`
  - `object_init_pos_noise=[0.005, 0.005, 0.0]`

### Local conclusion
- The YAML is ready for the next PPO training run with the new initpose and restored domain randomization.
- The actual runtime scale range is approximately `1.15-1.25` because task code samples each `randomizeScaleList` center with `±0.025`.

### Remaining blocked/risky
- Any already-open viewer/training process keeps its startup config; restart viewers/training to use this restored randomization.

### Single recommended next step
- Launch a fresh PPO teacher run from this YAML with a new cache name before comparing behavior against earlier fixed-scale/no-randomization checkpoints.

---

## v2-192 (2026-04-30) -- Cloud DexH13 NewInit DR115125 PPO20m Launch

### Target milestone/subgoal
- Train a fresh 20-minute PPO teacher on cloud from the latest `Dexh13HoraLightbulb.yaml`: new keyboard-tuned initpose plus restored domain randomization and bulb scale range approximately `1.15-1.25`.

### What changed (files + behavior impact)
- Added cloud launch script:
  - `outputs/cloud_pipeline_dexh13_lightbulb_newinit_dr115125_ppo20m/run_ppo_20m_aggressive.sh`
- Script behavior:
  - `timeout 1200`
  - task `Dexh13HoraLightbulb`
  - `headless=True`
  - seed `42`
  - `task.env.numEnvs=12288`
  - `train.ppo.minibatch_size=24576`
  - output run:
    - `outputs/Dexh13HoraLightbulb_teacher/dexh13_lightbulb_newinit_dr115125_s42_20m/`
  - pipeline log:
    - `outputs/cloud_pipeline_dexh13_lightbulb_newinit_dr115125_ppo20m/latest.log`
- Synced to cloud:
  - `configs/task/Dexh13HoraLightbulb.yaml`
  - `configs/train/Dexh13HoraLightbulb.yaml`
  - `dexscrew/tasks/xhand_hora.py`
  - launch script above.

### What was verified (commands + key outcomes)
- Local Hydra probe confirmed before sync:
  - `handRootPos=[0.086, 0.014, 0.243]`
  - `handRootRPY=[3.1415, 0.422173, 3.1415]`
  - updated index/thumb initial joints
  - `baseObjScale=1.2`
  - `randomizeScale=True`
  - `randomizeScaleList=[1.175, 1.225]`
  - scale bounds `1.15 1.25 1.15 1.25`
  - `mass/com/friction/pd=True True True True`
  - `force=2.0 0.25`
  - `object_init_pos_noise=[0.005, 0.005, 0.0]`
- Remote preflight:
  - no tmux session
  - no matching `Dexh13HoraLightbulb` train process
  - GPU idle: `NVIDIA GeForce RTX 4090 D`, `1 MiB / 24564 MiB`, `0%`
- Remote post-sync checks:
  - script syntax pass
  - remote YAML contains the new initpose and restored DR values.
- Remote launch:
  - tmux session:
    - `dexh13_lightbulb_newinit_dr115125_ppo20m`
  - command:
    - `timeout 1200 python train.py task=Dexh13HoraLightbulb ...`
  - runtime config log confirms:
    - new `handRootPos`, `handRootRPY`, index/thumb init joints.
    - object init noise and DR restored.
  - after about 48 seconds:
    - process alive
    - GPU `14265 MiB / 24564 MiB`, util about `64%`
    - scale init markers:
      - `Generated 5000 random initial poses for XHand at scale 1.175`
      - `Generated 5000 random initial poses for XHand at scale 1.225`

### Local conclusion
- The requested 20-minute cloud PPO run is active with the correct new initpose and restored domain randomization.

### Remaining blocked/risky
- No checkpoint had appeared at the first short status check; this is normal during early startup/initial training.
- It should exit with `train_exit_status=124` after `timeout 1200`.

### Single recommended next step
- After timeout completion, sync `outputs/Dexh13HoraLightbulb_teacher/dexh13_lightbulb_newinit_dr115125_s42_20m/` and visualize the best checkpoint locally.

---

## v2-191 (2026-04-30) -- Sim2Real TwoFinger Strong Stable-Grasp Reward Added

### Target milestone/subgoal
- Tighten the `Dexh13HoraLightbulbSim2RealTwoFinger` reward so the policy is pushed away from alternating single-finger flicks and toward stable thumb-index co-contact/co-grip.

### What changed (files + behavior impact)
- Updated `configs/task/Dexh13HoraLightbulbSim2RealTwoFinger.yaml`.
  - Stronger existing gate:
    - `two_finger_gate.contact_force_min: 0.5 -> 1.0`
    - `two_finger_gate.contact_force_max: 2.0 -> 3.0`
    - `two_finger_gate.no_grasp_penalty_scale: -1.5 -> -3.0`
  - Stronger coactive fingertip rewards:
    - `fingertip_tangent_reward.contact_force_min: 1.0`
    - `fingertip_tangent_reward.contact_force_max: 3.0`
    - `fingertip_torque_reward.contact_force_min: 1.0`
    - `fingertip_torque_reward.contact_force_max: 3.0`
  - Added positive-rotation co-contact penalty:
    - `active_two_finger_contact.enable: True`
    - Penalizes positive screw motion when thumb/index contact is weak.
  - Added opposition/inward grip reward:
    - `opposition_grip_reward.enable: True`
    - Rewards thumb/index being on opposite sides and applying inward force toward the bulb center.
- Updated `dexscrew/tasks/xhand_hora.py`.
  - Added config parsing for `active_two_finger_contact`.
  - Added config parsing for `opposition_grip_reward`.
  - Added reward contribution and diagnostics:
    - `active_two_finger/penalty`
    - `active_two_finger/active_frac`
    - `active_two_finger/pair_contact_w`
    - `active_two_finger/thumb_contact_w`
    - `active_two_finger/other_contact_w`
    - `opposition_grip/reward`
    - `opposition_grip/reward_scaled`
    - `opposition_grip/oppositeness`
    - `opposition_grip/radial_dot`
    - `opposition_grip/pair_dist_w`
    - `opposition_grip/pair_inward_w`
    - `opposition_grip/thumb_inward_force`
    - `opposition_grip/other_inward_force`

### What was verified (commands + key outcomes)
- Compile:
  - `PYTHONDONTWRITEBYTECODE=1 python - <<'PY' ... compile(Path('dexscrew/tasks/xhand_hora.py').read_text(), ...) ... PY`
  - Outcome: `compile_ok`.
- Static diff check:
  - `git diff --check -- dexscrew/tasks/xhand_hora.py configs/task/Dexh13HoraLightbulbSim2RealTwoFinger.yaml`
  - Outcome: pass.
- Docker/Hydra compose probe confirmed:
  - strong gate/contact thresholds are present.
  - both new config blocks resolve under the task YAML.
- IsaacGym smoke:
  - `./docker-run-isaacgym.sh timeout 180 python train.py task=Dexh13HoraLightbulbSim2RealTwoFinger headless=True seed=42 train.algo=PPO wandb_activate=False task.env.numEnvs=4 train.ppo.minibatch_size=12 train.ppo.max_agent_steps=24 train.ppo.output_name=Dexh13HoraLightbulb_teacher_sim2real_twofinger/smoke_strong_grasp_tmp task.env.termination.log=True`
  - Outcome: env built, config printed with `active_two_finger_contact` and `opposition_grip_reward`, and run exited normally with `max steps achieved`.
- Process check:
  - no matching smoke/training process remained.

### Local conclusion
- The current sim2real two-finger YAML now has all four stable-grasp changes enabled:
  - stronger no-grasp penalty,
  - higher effective contact threshold,
  - active positive-rotation co-contact penalty,
  - opposition/inward grip reward.

### Remaining blocked/risky
- This is a stricter reward and may slow early PPO learning; reward may start lower than the previous `best_reward_3101.83` run.
- The new reward has only smoke validation; visual behavior still requires a fresh PPO probe.

### Single recommended next step
- Train a fresh 30-minute PPO with a new cache such as `sim2real_twofinger_stronggrasp_scale115125_s42_30m`, then visualize whether index/thumb stay in stable co-contact instead of alternating single-finger phases.

---

## v2-192 (2026-04-30) -- Sim2Real TwoFinger Strong-Grasp r45 PPO Probe

### Target milestone/subgoal
- Test whether raising the main lightbulb rotation reward by `+1.0` improves the strong stable-grasp Sim2Real two-finger PPO probe without breaking the co-contact constraints.

### What changed (files + behavior impact)
- Updated `configs/task/Dexh13HoraLightbulbSim2RealTwoFinger.yaml`.
  - `reward.rotate_reward_scale: 3.5 -> 4.5`.
  - Kept the current strong-grasp settings unchanged:
    - stronger two-finger gate/contact thresholds,
    - active positive-rotation co-contact penalty,
    - opposition/inward grip reward,
    - bulb scale randomization around `1.15-1.25`.

### What was verified (commands + key outcomes)
- Static diff check:
  - `git diff --check -- configs/task/Dexh13HoraLightbulbSim2RealTwoFinger.yaml dexscrew/tasks/xhand_hora.py`
  - Outcome: pass.
- Docker/Hydra compose probe confirmed:
  - `rotate_reward_scale: 4.5`.
  - strong-grasp reward blocks still resolve.
- 20-minute PPO probe:
  - `./docker-run-isaacgym.sh timeout 1200 scripts/run_with_cleanup.sh bash scripts/dexh13_lightbulb_teacher_sim2real_twofinger.sh 0 42 sim2real_twofinger_stronggrasp_r45_scale115125_s42_20m True wandb_activate=True task.env.termination.log=True task.env.numEnvs=4096 train.ppo.minibatch_size=8192`
  - Outcome: expected timeout exit `124`; no OOM or segfault.
  - Best checkpoint: `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger/sim2real_twofinger_stronggrasp_r45_scale115125_s42_20m/stage1_nn/best_reward_3161.09.pth`.
- TensorBoard final scalar spot check:
  - `episode_rewards/step ~= 3158.44`.
  - `screw/angular_velocity ~= 0.919`.
  - `two_finger/gate ~= 0.986`.
  - `active_two_finger/penalty = 0.0`.
  - `active_two_finger/pair_contact_w = 1.0`.
  - `term/any_reset_frac ~= 0.00122`.
  - `term/no_contact_frac = 0.0`.
- Process/GPU check:
  - no residual IsaacGym `train.py`/run-with-cleanup process.
  - no training compute process left on GPU.

### Local conclusion
- Raising `rotate_reward_scale` to `4.5` learned normally over the 20-minute probe and reached a usable checkpoint.
- Scalar-level co-contact health looks acceptable; visual behavior still needs to be checked because the previous concern was specifically two-finger coordination quality, not reward alone.

### Remaining blocked/risky
- `opposition_grip/reward_scaled` and `opposition_grip/oppositeness` ended at `0.0` in the final scalar sample, so the opposition reward may be inactive or too hard under the current measured geometry.
- The policy may still visually fall back to alternating behavior despite the good gate scalar; visualization is the required next check.

### Single recommended next step
- Visualize `best_reward_3161.09.pth` for `sim2real_twofinger_stronggrasp_r45_scale115125_s42_20m` and decide whether this r45 probe is more stable than the previous r35 strong-grasp run.

---

## v2-193 (2026-04-30) -- Sim2Real TwoFinger Thumb Stability Penalty Added

### Target milestone/subgoal
- Address the user's visual finding that the current `Sim2RealTwoFinger` policy still tends to lose thumb contact at the end of each thumb rotation stroke.

### What changed (files + behavior impact)
- Updated `dexscrew/tasks/xhand_hora.py`.
  - Reused and extended the existing `env.thumb_slip_penalty` path.
  - Added per-env `prev_thumb_drive_w` state and reset handling.
  - Extended the thumb slip penalty with:
    - high-speed ejection loss,
    - after-drive detach loss,
    - terminal-ease loss when thumb joints are near their limits and still moving fast.
  - Added TensorBoard/W&B diagnostics:
    - `thumb_slip_penalty/ejection_loss`
    - `thumb_slip_penalty/after_drive_loss`
    - `thumb_slip_penalty/terminal_ease_loss`
    - `thumb_slip_penalty/thumb_tip_speed`
    - `thumb_slip_penalty/speed_weight`
    - `thumb_slip_penalty/prev_drive_w`
    - `thumb_slip_penalty/current_drive_w`
- Updated `configs/task/Dexh13HoraLightbulbSim2RealTwoFinger.yaml`.
  - Enabled `thumb_slip_penalty`.
  - Active only when screw velocity is positive above `0.2`.
  - Uses strong contact thresholds `1.0-3.0`.
  - Penalizes thumb being beyond `far_dist=0.085`, high-speed ejection above `0.18 m/s`, detach after prior drive, and high thumb velocity near joint-limit edges.

### What was verified (commands + key outcomes)
- Python compile:
  - `PYTHONDONTWRITEBYTECODE=1 python - <<'PY' ... compile(Path('dexscrew/tasks/xhand_hora.py').read_text(), ...) ... PY`
  - Outcome: `compile_ok`.
- Static diff check:
  - `git diff --check -- dexscrew/tasks/xhand_hora.py configs/task/Dexh13HoraLightbulbSim2RealTwoFinger.yaml`
  - Outcome: pass.
- Docker/Hydra compose probe confirmed:
  - `thumb_slip_penalty` resolves under `Dexh13HoraLightbulbSim2RealTwoFinger`.
  - New scales `ejection_penalty_scale=-2.0`, `after_drive_penalty_scale=-1.5`, `terminal_ease_penalty_scale=-0.2` are present.
- IsaacGym smoke:
  - `./docker-run-isaacgym.sh timeout 180 python train.py task=Dexh13HoraLightbulbSim2RealTwoFinger headless=True seed=42 train.algo=PPO wandb_activate=False task.env.numEnvs=4 train.ppo.minibatch_size=12 train.ppo.max_agent_steps=24 train.ppo.output_name=Dexh13HoraLightbulb_teacher_sim2real_twofinger/smoke_thumb_stability_tmp task.env.termination.log=True`
  - Outcome: env built with the new config and exited normally with `max steps achieved`.
- Process/GPU check:
  - no residual smoke/training process.
  - no IsaacGym compute process left on GPU.

### Local conclusion
- The current `Sim2RealTwoFinger` YAML now has a targeted reward term for the observed thumb end-of-stroke detach, instead of relying on broad pose/work penalties.
- Existing PPO checkpoints do not include this behavior change; a fresh PPO probe is required.

### Remaining blocked/risky
- This can lower early reward because thumb flicking is now explicitly punished.
- The terminal-ease term is deliberately mild (`-0.2`) so it should guide slowdown near the stroke edge without freezing useful thumb motion.

### Single recommended next step
- Run a fresh 20-30 minute PPO with a new cache such as `sim2real_twofinger_thumbstable_r45_scale115125_s42_30m`, then visualize whether thumb end-of-stroke detach is reduced.

---

## v2-193 (2026-04-30) -- Dexh13 Lightbulb New InitPose DR115-125 PPO20m Sync/Visualize

### Target milestone/subgoal
- Train and inspect the current `Dexh13HoraLightbulb.yaml` teacher PPO after restoring domain randomization and setting bulb scale randomization around `1.15-1.25`.

### What changed (files + behavior impact)
- No new code/config edit in this step; used the already-updated `configs/task/Dexh13HoraLightbulb.yaml`.
- Cloud run artifacts were synced back locally under:
  - `outputs/Dexh13HoraLightbulb_teacher/dexh13_lightbulb_newinit_dr115125_s42_20m/`
  - `outputs/cloud_pipeline_dexh13_lightbulb_newinit_dr115125_ppo20m/`

### What was verified (commands + key outcomes)
- Cloud PPO run completed by expected timeout:
  - `outputs/cloud_pipeline_dexh13_lightbulb_newinit_dr115125_ppo20m/pipeline_20260429_181636.log` ends with `[pipeline] train_exit_status=124`.
  - Best checkpoint synced locally:
    `outputs/Dexh13HoraLightbulb_teacher/dexh13_lightbulb_newinit_dr115125_s42_20m/stage1_nn/best_reward_3461.48.pth`.
- Local viewer launched:
  - `./docker-run-isaacgym.sh bash scripts/vis_dexh13_lightbulb_teacher.sh 0 42 dexh13_lightbulb_newinit_dr115125_s42_20m test=True checkpoint=outputs/Dexh13HoraLightbulb_teacher/dexh13_lightbulb_newinit_dr115125_s42_20m/stage1_nn/best_reward_3461.48.pth graphics_device_id=0`
  - Outcome: Isaac Gym viewer window is active; checkpoint loads.
- Screenshot captured:
  - `outputs/visual_checks/dexh13_lightbulb_newinit_dr115125_best3461.png`.

### Local conclusion
- Training finished and the best PPO checkpoint is available locally.
- Visual inspection from the captured frame shows thumb-side contact is closer than index-side contact; the index still appears separated from the bulb at this sampled moment/view.

### Remaining blocked/risky
- This is a 20-minute PPO probe, not a converged policy.
- Viewer script disables force/noise but leaves scale randomization active, so visualized initial geometry can still sample the `1.15-1.25` scale range.

### Single recommended next step
- User should inspect the live viewer and decide whether the index gap is acceptable or whether the next PPO probe should use a fixed `1.2` scale for visual debugging before re-enabling scale randomization.

---

## v2-194 (2026-04-30) -- Dexh13 Lightbulb Thumb-Slip Penalty Added

### Target milestone/subgoal
- Reduce the observed thumb-end slip in `Dexh13HoraLightbulb` by first implementing:
  - positive-rotation thumb contact-loss penalty,
  - stricter angular-velocity clipping/penalty to discourage hard flicks.

### What changed (files + behavior impact)
- Updated `dexscrew/tasks/xhand_hora.py`.
  - Added `env.thumb_slip_penalty` parsing.
  - Added `_compute_thumb_slip_penalty()`.
  - The penalty is active only during positive screw rotation above `active_screw_vel`.
  - Penalizes weak thumb contact force and thumb distance beyond `far_dist`.
  - Logs:
    - `thumb_slip_penalty/penalty`
    - `thumb_slip_penalty/contact_loss`
    - `thumb_slip_penalty/far_loss`
    - `thumb_slip_penalty/thumb_contact_w`
    - `thumb_slip_penalty/thumb_dist`
    - `thumb_slip_penalty/active_frac`
    - `thumb_slip_penalty/velocity_weight`
- Updated `configs/task/Dexh13HoraLightbulb.yaml`.
  - Added:
    - `thumb_slip_penalty.enable: True`
    - `active_screw_vel: 0.15`
    - `contact_force_min/max: 0.3/2.0`
    - `far_dist: 0.09`
    - `scale_with_object: True`
    - `contact_penalty_scale: -1.0`
    - `far_penalty_scale: -0.5`
  - Made rotation less flick-friendly:
    - `reward.angvelClipMax: 4.0 -> 2.5`
    - `reward.angvelPenaltyThres: 10.0 -> 2.8`
    - `reward.rotate_penalty_scale: -0.3 -> -0.8`

### What was verified (commands + key outcomes)
- Python compile:
  - `PYTHONDONTWRITEBYTECODE=1 python - <<'PY' ... compile(...) ... PY`
  - Outcome: `compile_ok`.
- Static whitespace/conflict check:
  - `git diff --check -- dexscrew/tasks/xhand_hora.py configs/task/Dexh13HoraLightbulb.yaml`
  - Outcome: pass.
- IsaacGym smoke:
  - `./docker-run-isaacgym.sh timeout 180 python train.py task=Dexh13HoraLightbulb headless=True seed=42 train.algo=PPO wandb_activate=False task.env.numEnvs=4 train.ppo.minibatch_size=12 train.ppo.max_agent_steps=24 train.ppo.output_name=Dexh13HoraLightbulb_teacher/smoke_thumbslip_penalty_tmp task.env.termination.log=True`
  - Outcome: task builds, config prints `thumb_slip_penalty` and the stricter velocity settings, then exits with `max steps achieved`.
- Process check:
  - no residual headless smoke training process remained.

### Local conclusion
- The first two requested reward changes are implemented and smoke-validated.
- This is a behavior-changing reward update; existing PPO checkpoints do not reflect it.

### Remaining blocked/risky
- The new penalty can reduce raw reward at first because slipping during active rotation is now explicitly punished.
- It still does not implement thumb action-rate/joint-velocity smoothing; that is intentionally left for the next step if this penalty alone does not fix the end-of-stroke slip.

### Single recommended next step
- Sync these two files to cloud and run a fresh PPO probe with a new output name, then compare `thumb_slip_penalty/*`, `thumb_slip/*`, and visual behavior against `dexh13_lightbulb_newinit_dr115125_s42_20m`.

---

## v2-195 (2026-04-30) -- Dexh13 Lightbulb Current YAML Compatibility Smoke

### Target milestone/subgoal
- Check whether later shared `xhand_hora.py` changes for another YAML broke the current `Dexh13HoraLightbulb.yaml` training entrypoint.

### What changed (files + behavior impact)
- No source/config edits were made for this check.
- Observed current workspace state:
  - `dexscrew/tasks/xhand_hora.py` is modified and now contains extended `thumb_slip_penalty` support.
  - `configs/task/Dexh13HoraLightbulb.yaml` is modified and uses only the basic thumb slip penalty fields.
  - `configs/task/Dexh13HoraLightbulbSim2RealTwoFinger.yaml` appears untracked in this local worktree.

### What was verified (commands + key outcomes)
- Bootstrap context read:
  - `docs/session_handoff_v2.md`
  - `docs/stage_acceptance_summary.md`
- Static checks:
  - `PYTHONDONTWRITEBYTECODE=1 python - <<'PY' ... compile(Path('dexscrew/tasks/xhand_hora.py').read_text(), ...) ... PY`
    - Outcome: `compile_ok`.
  - `git diff --check -- dexscrew/tasks/xhand_hora.py configs/task/Dexh13HoraLightbulb.yaml`
    - Outcome: pass.
- Compatibility grep:
  - The extended `thumb_slip_penalty` keys such as `ejection_penalty_scale`, `after_drive_penalty_scale`, and `terminal_ease_penalty_scale` exist in `xhand_hora.py` / `Dexh13HoraLightbulbSim2RealTwoFinger.yaml`.
  - They are not set in `Dexh13HoraLightbulb.yaml`, so this YAML uses the code defaults.
- IsaacGym smoke:
  - `./docker-run-isaacgym.sh timeout 180 python train.py task=Dexh13HoraLightbulb headless=True seed=42 train.algo=PPO wandb_activate=False task.env.numEnvs=4 train.ppo.minibatch_size=12 train.ppo.max_agent_steps=24 train.ppo.output_name=Dexh13HoraLightbulb_teacher/smoke_current_yaml_compat_tmp task.env.termination.log=True`
  - Outcome: environment built, config printed the current `Dexh13HoraLightbulb.yaml`, generated scale caches for `1.175` and `1.225`, and exited normally with `max steps achieved`.
- Process check:
  - no residual `smoke_current_yaml_compat_tmp` or headless `Dexh13HoraLightbulb` train process remained.

### Local conclusion
- The current shared `xhand_hora.py` changes do not break `Dexh13HoraLightbulb.yaml` at smoke-test level.
- The other YAML's extra thumb-stability fields can affect this YAML only if configured here or if future shared-code defaults become nonzero.

### Remaining blocked/risky
- Smoke proves startup/rollout/update compatibility, not long-run reward quality.
- Because `xhand_hora.py` is shared, future edits for another YAML can still change behavior for all Hora tasks if they alter default values or non-guarded code paths.

### Single recommended next step
- Before cloud training, sync the current `Dexh13HoraLightbulb.yaml` and `xhand_hora.py` as a pair, then run a fresh PPO with a new output name so the checkpoint is traceable to this exact reward implementation.

---

## v2-196 (2026-04-30) -- Sim2Real TwoFinger Thumbstable 7h PPO->PAdapt Pipeline Launched

### Target milestone/subgoal
- Run the user's requested 7h sequence:
  - `3.5h` PPO teacher on `Dexh13HoraLightbulbSim2RealTwoFinger`.
  - then `3.5h` `ProprioAdapt` student distillation from the teacher's selected best PPO checkpoint.

### What changed (files + behavior impact)
- Added `scripts/dexh13_lightbulb_student_padapt_sim2real_twofinger.sh`.
  - Uses task `Dexh13HoraLightbulbSim2RealTwoFinger`.
  - Uses algo `ProprioAdapt` with `train.ppo.proprio_adapt=True`.
  - Takes an explicit teacher checkpoint path as its 4th argument, avoiding the old hardcoded `ThesisTwoFinger` teacher checkpoint.
- Added `scripts/run_sim2real_twofinger_thumbstable_ppo_padapt_7h.sh`.
  - Runs teacher with `timeout 12600`.
  - Selects best `stage1_nn/best_reward_*.pth` after teacher finishes.
  - Runs student with a separate `timeout 12600`.
  - Logs command lines, exit statuses, selected checkpoint, and student artifacts under:
    `outputs/sim2real_twofinger_thumbstable_ppo_padapt_7h/`.
- Added `scripts/monitor_sim2real_twofinger_thumbstable_7h.sh`.
  - Logs stage, recent reward, best ckpt, GPU usage, and error patterns every 5 minutes.

### What was verified (commands + key outcomes)
- No local conflicting `train.py` / thumbstable process before launch.
- Script syntax:
  - `bash -n scripts/dexh13_lightbulb_student_padapt_sim2real_twofinger.sh`
  - `bash -n scripts/run_sim2real_twofinger_thumbstable_ppo_padapt_7h.sh`
  - `bash -n scripts/monitor_sim2real_twofinger_thumbstable_7h.sh`
  - Outcome: pass.
- Hydra compose probe:
  - task resolves as `Dexh13HoraLightbulbSim2RealTwoFinger`.
  - `train.algo=ProprioAdapt` resolves.
  - `action_mask_indices` and `thumb_slip_penalty` are present.
- Pipeline launched via `nohup setsid` because local `tmux` is not installed.
  - Pipeline pid: `984556`.
  - Monitor pid: `996633`.
  - Main log: `outputs/sim2real_twofinger_thumbstable_ppo_padapt_7h/latest.log`.
  - Monitor log: `outputs/sim2real_twofinger_thumbstable_ppo_padapt_7h/monitor.log`.
- Current teacher status at launch supervision:
  - stage: teacher.
  - command:
    `./docker-run-isaacgym.sh timeout 12600 scripts/run_with_cleanup.sh bash scripts/dexh13_lightbulb_teacher_sim2real_twofinger.sh 0 42 sim2real_twofinger_thumbstable True wandb_activate=True task.env.termination.log=True`
  - GPU: RTX 4080 SUPER, about `11.6GB / 16.4GB`, high utilization.
  - Teacher entered training loop, no OOM/segfault.
  - Early best checkpoint reached at least:
    `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger/sim2real_twofinger_thumbstable/stage1_nn/best_reward_73.57.pth`.

### Local conclusion
- The requested long sequence is running and supervised.
- The first phase is using the exact requested teacher cache `sim2real_twofinger_thumbstable`.
- The student phase is wired to consume the teacher phase's selected best PPO checkpoint, not a stale hardcoded teacher.

### Remaining blocked/risky
- The teacher phase is intentionally time-limited; `teacher_exit_status=124` is expected around 3.5h.
- The pipeline must be checked at the teacher->student transition to verify the selected checkpoint and student startup.
- The monitor's first failed inline attempt left harmless shell syntax text in `monitor.log`; the active monitor script was restarted and is now functioning.

### Single recommended next step
- Continue supervising until teacher timeout, confirm `selected_teacher_ckpt=...`, then confirm the `ProprioAdapt` student process starts with that checkpoint.

---

## v2-197 (2026-04-30) -- Cloud Dexh13 Lightbulb Thumbslip PPO 3.5h -> PAdapt 3.5h Supervised Launch

### Target milestone/subgoal
- Run the user's requested cloud sequence on the current `Dexh13HoraLightbulb.yaml`:
  - `3.5h` PPO teacher.
  - then `3.5h` `ProprioAdapt` / PAdapt student distillation from the selected best teacher checkpoint.

### What changed (files + behavior impact)
- Added cloud run artifacts under:
  - `outputs/cloud_pipeline_dexh13_lightbulb_thumbslip_ppo3p5h_padapt3p5h/run_ppo_padapt_7h_supervised.sh`
  - `outputs/cloud_pipeline_dexh13_lightbulb_thumbslip_ppo3p5h_padapt3p5h/watch_ppo_padapt_7h.sh`
- Synced current training inputs to `cloud-training:/root/code/dexscrew-repro/`:
  - `AGENTS.md`
  - `train.py`
  - `configs/`
  - `dexscrew/`
  - `scripts/`
  - `assets/`
  - `.git`
- Teacher resource setting:
  - `task.env.numEnvs=12288`
  - `train.ppo.minibatch_size=24576`
  - `num_threads=22`
- Student resource setting:
  - `task.env.numEnvs=512`
  - `train.ppo.minibatch_size=6144`
  - `num_threads=22`

### What was verified (commands + key outcomes)
- Remote compile:
  - `python -m py_compile dexscrew/tasks/xhand_hora.py`
  - Outcome: pass.
- Remote config presence:
  - `Dexh13HoraLightbulb.yaml` contains the current `thumb_slip_penalty` and stricter rotation velocity penalty settings.
  - `assets/screw/contactviz/0000_lightbulb.urdf` exists.
- Remote teacher smoke:
  - `timeout 180 scripts/run_with_cleanup.sh python train.py task=Dexh13HoraLightbulb ... train.algo=PPO ... task.env.numEnvs=4 ... train.ppo.max_agent_steps=24`
  - Outcome: passed after adding `/root/code/dexscrew-repro` to git `safe.directory`.
- Remote PAdapt smoke:
  - `timeout 180 scripts/run_with_cleanup.sh python train.py task=Dexh13HoraLightbulb ... train.algo=ProprioAdapt train.ppo.proprio_adapt=True checkpoint=...`
  - Outcome: loaded teacher checkpoint, built env, printed `ProprioAdapt trainable patterns: ['adapt_tconv']`, entered the training loop, then was manually cleaned up.
- Formal pipeline launched on cloud in tmux:
  - pipeline session: `dexh13_lightbulb_thumbslip_ppo_padapt_7h`
  - watchdog session: `dexh13_lightbulb_thumbslip_watchdog_7h`
  - phase file: `outputs/cloud_pipeline_dexh13_lightbulb_thumbslip_ppo3p5h_padapt3p5h/phase.txt`
  - pipeline log: `outputs/cloud_pipeline_dexh13_lightbulb_thumbslip_ppo3p5h_padapt3p5h/latest.log`
  - watchdog log: `outputs/cloud_pipeline_dexh13_lightbulb_thumbslip_ppo3p5h_padapt3p5h/watchdog_latest.log`
- Current cloud teacher status at launch supervision:
  - phase: `teacher_running`
  - command uses `timeout 12600`.
  - GPU: RTX 4090 D, about `14.3GB / 24.6GB`, active utilization.
  - early checkpoints are being refreshed, reaching at least:
    `outputs/Dexh13HoraLightbulb_teacher/dexh13_lightbulb_thumbslip_s42_ppo3p5h/stage1_nn/best_reward_127.77.pth`

### Local conclusion
- The current `Dexh13HoraLightbulb.yaml` teacher-student cloud pipeline is launched and actively supervised.
- Teacher startup and PAdapt startup were both smoke-validated before the long run.
- The formal teacher phase is actively training and producing checkpoints.

### Remaining blocked/risky
- The main stdout log is noisy because `train.py` prints a large dirty git diff at startup.
- `teacher_exit_status=124` is expected at the 3.5h timeout and should not be treated as failure.
- The critical transition to verify is teacher timeout -> best checkpoint selection -> PAdapt student startup.

### Single recommended next step
- Continue live supervision through the teacher timeout, verify `selected_teacher_ckpt.txt`, then verify the PAdapt student phase starts with that exact checkpoint.

---

## v2-198 (2026-04-30) -- Cloud Dexh13 Lightbulb Thumbslip PPO/PAdapt 7h Completed and Synced

### Target milestone/subgoal
- Finish and verify the cloud 7h sequence launched in v2-197:
  - `3.5h` PPO teacher.
  - `3.5h` PAdapt student distillation from the selected PPO teacher checkpoint.

### What changed (files + behavior impact)
- No source/config behavior was changed after launch.
- Synced completed remote artifacts back to local, excluding huge TensorBoard event dirs:
  - `outputs/Dexh13HoraLightbulb_teacher/dexh13_lightbulb_thumbslip_s42_ppo3p5h/`
  - `outputs/Dexh13HoraLightbulb_student_padapt/dexh13_lightbulb_thumbslip_s42_padapt_from_ppo3p5h/`
  - `outputs/cloud_pipeline_dexh13_lightbulb_thumbslip_ppo3p5h_padapt3p5h/`
- Saved a compact local pipeline tail:
  - `outputs/cloud_pipeline_dexh13_lightbulb_thumbslip_ppo3p5h_padapt3p5h/latest_tail_180.txt`

### What was verified (commands + key outcomes)
- Remote final status:
  - phase: `done`
  - `teacher_exit_status=124`
  - `student_exit_status=124`
  - GPU idle after completion.
  - no residual `python train.py` / `run_with_cleanup` processes.
- Teacher timing / result:
  - `teacher_duration_sec=12604`
  - selected checkpoint:
    `outputs/Dexh13HoraLightbulb_teacher/dexh13_lightbulb_thumbslip_s42_ppo3p5h/stage1_nn/best_reward_5151.36.pth`
- Student timing / result:
  - `student_duration_sec=12604`
  - final observed training metrics before timeout:
    `Agent Steps: 0077M | FPS: 6158.5 | Current Best: 5001.07`
  - best student checkpoint:
    `outputs/Dexh13HoraLightbulb_student_padapt/dexh13_lightbulb_thumbslip_s42_padapt_from_ppo3p5h/stage2_nn/model_best.ckpt`
- Local sync verification:
  - teacher checkpoints present locally:
    - `best_reward_5151.36.pth`
    - `ep_500_step_0073m_reward_4173.85.pth`
    - `ep_1000_step_0147m_reward_4523.76.pth`
    - `ep_1500_step_0221m_reward_4843.62.pth`
    - `last.pth`
  - student checkpoint present locally:
    - `stage2_nn/model_best.ckpt`
  - status files present locally:
    - `phase.txt` contains `done`
    - `teacher_exit_status` contains `124`
    - `student_exit_status` contains `124`
    - `selected_teacher_ckpt.txt` points to `best_reward_5151.36.pth`

### Local conclusion
- The requested cloud sequence completed correctly and exactly followed PPO teacher first, then PAdapt student from the selected best PPO checkpoint.
- Both phases used explicit wall-clock `timeout 12600` guards and ended with expected timeout status `124`.
- Required `.pth` / `.ckpt` artifacts are now available locally for visualization.

### Remaining blocked/risky
- No visual inspection has been performed yet on the completed teacher or student artifacts.
- High scalar reward does not guarantee the thumb slip behavior is visually solved; local viewer inspection is still required.
- Full TensorBoard event files were intentionally not synced because the student event file alone was about `981MB`.

### Single recommended next step
- Visualize the synced teacher first, then the synced PAdapt student, and compare thumb slip/stability against the previous `dexh13_lightbulb_newinit_dr115125_s42_20m` baseline.

---

## v2-198 (2026-04-30) -- Local Sim2Real TwoFinger Thumbstable PPO 3.5h -> PAdapt 3.5h Completed

### Target milestone/subgoal
- Complete the user's requested local 7h flow for `Dexh13HoraLightbulbSim2RealTwoFinger`:
  - `3.5h` PPO teacher with cache `sim2real_twofinger_thumbstable`.
  - then `3.5h` `ProprioAdapt` / PAdapt student distillation from that teacher's selected best PPO checkpoint.

### What changed (files + behavior impact)
- Added/used local pipeline scripts:
  - `scripts/run_sim2real_twofinger_thumbstable_ppo_padapt_7h.sh`
  - `scripts/dexh13_lightbulb_student_padapt_sim2real_twofinger.sh`
  - `scripts/monitor_sim2real_twofinger_thumbstable_7h.sh`
- Produced final local artifacts:
  - PPO teacher:
    `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger/sim2real_twofinger_thumbstable/stage1_nn/best_reward_6103.04.pth`
  - PAdapt student:
    `outputs/Dexh13HoraLightbulb_student_padapt_sim2real_twofinger/sim2real_twofinger_thumbstable_padapt/stage2_nn/model_best.ckpt`
  - Pipeline log:
    `outputs/sim2real_twofinger_thumbstable_ppo_padapt_7h/latest.log`

### What was verified (commands + key outcomes)
- Teacher phase:
  - Exited with `teacher_exit_status=124`, expected from the 3.5h `timeout`.
  - Selected exact teacher checkpoint:
    `best_reward_6103.04.pth`.
- Student phase:
  - Started from the selected teacher checkpoint, not a stale path.
  - Exited with `student_exit_status=124`, expected from the 3.5h `timeout`.
  - Pipeline wrote `pipeline_done`.
  - Final observed student `Current Best`: `4261.45`.
- Artifacts:
  - `model_best.ckpt` timestamp: `2026-04-30 10:06:52`.
  - Student TensorBoard event timestamp: `2026-04-30 10:06:56`.
- Cleanup/safety:
  - No local `train.py` / pipeline process remained after completion.
  - Recent pipeline log scan found no `Segmentation fault`, `PxgCudaDeviceMemoryAllocator fail`, `Traceback`, `FileNotFoundError`, `RuntimeError:`, or `CUDA out of memory`.
  - GPU returned to non-training baseline use after completion.

### Local conclusion
- The requested local 7h PPO -> PAdapt flow completed successfully.
- There are two usable strategies:
  - teacher PPO `best_reward_6103.04.pth`;
  - distilled PAdapt student `model_best.ckpt`.
- The student reward is lower than teacher as expected for a proprio/adaptation student, but the distillation run was technically healthy and produced a valid best checkpoint.

### Remaining blocked/risky
- Behavior still needs visual validation before marking it as deployment candidate; reward alone cannot prove thumb slip has been eliminated.
- The student run used `task.env.numEnvs=48` and `train.ppo.minibatch_size=576` from the student script, so its reward scale/learning curve should be compared against prior student runs rather than PPO teacher directly.

### Single recommended next step
- Visualize the PAdapt `model_best.ckpt`; if behavior is acceptable, copy the teacher PPO + student ckpt + exact output config into the sim2real handoff bundle.

---

## v2-199 (2026-04-30) -- Sim2Real TwoFinger Student Index Contact Force Probe

### Target milestone/subgoal
- Answer whether the PAdapt student index finger is applying meaningful contact force on the bulb or only lightly covering it.

### What changed (files + behavior impact)
- No source/config behavior change.
- Added a rollout analysis artifact:
  - `outputs/Dexh13HoraLightbulb_student_padapt_sim2real_twofinger/sim2real_twofinger_thumbstable_padapt/index_force_rollout.pt`

### What was verified (commands + key outcomes)
- Docker already has TensorBoard installed:
  - `tensorboard 2.14.0`
- Ran a deterministic headless rollout with:
  - checkpoint:
    `outputs/Dexh13HoraLightbulb_student_padapt_sim2real_twofinger/sim2real_twofinger_thumbstable_padapt/stage2_nn/model_best.ckpt`
  - `task.env.numEnvs=64`
  - `collect_steps=300`
  - action/obs/pose/random force noise disabled
  - point cloud saving disabled
- Rollout summary:
  - `mean_reward=7.1559`
  - `mean_done_rate=0.0000`
- Contact-force-weight metrics from rollout extras:
  - `finger_contact/index/force_w`: mean `1.000`, p05 `1.000`, min `1.000`
  - `two_finger/other_contact_w`: mean `0.998947`, p05 `0.997659`, min `0.963221`
  - `active_two_finger/other_contact_w`: mean `1.000`, p05 `1.000`, min `1.000`
  - `active_two_finger/pair_contact_w`: mean `1.000`, p05 `1.000`, min `1.000`
  - `two_finger/gate`: mean `0.991883`, p05 `0.977593`, min `0.597645`

### Local conclusion
- Existing logs store normalized force weights, not raw Newtons.
- With current thresholds `contact_force_min=1.0` and `contact_force_max=3.0`, `force_w=1.0` means the measured net fingertip contact force is at or above the `3N` saturation threshold.
- Index contact is therefore not just lightly covering the bulb in the rollout; it is consistently above the configured strong-contact threshold.

### Remaining blocked/risky
- The existing `force_w` metric is clipped and does not separate normal force from tangential force.
- Exact raw normal-force magnitude requires adding/logging raw fingertip contact vector projection, e.g. `finger_contact/index/force_raw` and `finger_contact/index/normal_force`.

### Single recommended next step
- If sim2real deployment needs force calibration, add unclipped raw/index/thumb contact force and contact-normal projection diagnostics before the next long run.

---

## v2-200 (2026-04-30) -- Dexh13 Lightbulb Initpose090020233 PPO 30m Completed

### Target milestone/subgoal
- Convert the user's latest keyboard-tuned initpose into a separate task YAML and run a 30min cloud PPO teacher probe.

### What changed (files + behavior impact)
- Added task config:
  - `configs/task/Dexh13HoraLightbulbInitpose090020233.yaml`
- Added matching train config for Hydra default resolution:
  - `configs/train/Dexh13HoraLightbulbInitpose090020233.yaml`
- The new task config is copied from current `Dexh13HoraLightbulb.yaml` and only changes the saved initpose:
  - `handRootPos: [0.090000, 0.020000, 0.233000]`
  - `handRootRPY: [3.141500, 0.422173, 3.141500]`
  - `right_index_joint_0: 0.3400000036`
- The original `configs/task/Dexh13HoraLightbulb.yaml` was not modified.

### What was verified (commands + key outcomes)
- Latest tuner save file:
  - `outputs/initpose_tuning/Dexh13HoraLightbulb_current_fixed120.yaml`
  - Confirmed final saved value was `[0.090000, 0.020000, 0.233000]`, not the earlier `[0.088000, 0.020000, 0.233000]`.
- Local checks:
  - `git diff --check -- configs/task/Dexh13HoraLightbulbInitpose090020233.yaml configs/train/Dexh13HoraLightbulbInitpose090020233.yaml`
  - Outcome: pass.
- Remote smoke:
  - `timeout 180 scripts/run_with_cleanup.sh python train.py task=Dexh13HoraLightbulbInitpose090020233 ... task.env.numEnvs=4 train.ppo.max_agent_steps=24`
  - Outcome: task resolved, env built, and exited with `max steps achieved`.
  - Remote log confirmed:
    - `handRootPos: [0.09, 0.02, 0.233]`
    - `right_index_joint_0: 0.3400000036`
- Remote 30min PPO:
  - tmux session: `dexh13_initpose090020233_ppo30m`
  - pipeline dir: `outputs/cloud_pipeline_dexh13_initpose090020233_ppo30m/`
  - command used `timeout 1800`.
  - resource setting:
    - `task.env.numEnvs=12288`
    - `train.ppo.minibatch_size=24576`
    - `num_threads=22`
  - completed with:
    - `phase=done`
    - `ppo_exit_status=124`
    - duration about `1804s`
  - final best checkpoint:
    `outputs/Dexh13HoraLightbulb_teacher/dexh13_lightbulb_initpose090020233_s42_30m/stage1_nn/best_reward_3305.03.pth`
  - final observed tail included:
    - `Agent Steps: 0044M`
    - `FPS: ~28136`
    - `Current Best: 3023.40` before the final saved `3305.03` checkpoint.
- Local sync:
  - synced teacher output excluding TensorBoard:
    `outputs/Dexh13HoraLightbulb_teacher/dexh13_lightbulb_initpose090020233_s42_30m/`
  - synced pipeline status:
    `outputs/cloud_pipeline_dexh13_initpose090020233_ppo30m/`
  - local status files:
    - `phase.txt`: `done`
    - `ppo_exit_status`: `124`

### Local conclusion
- The new initpose YAML is available and the requested 30min PPO probe completed correctly.
- The resulting local visualization checkpoint is:
  `outputs/Dexh13HoraLightbulb_teacher/dexh13_lightbulb_initpose090020233_s42_30m/stage1_nn/best_reward_3305.03.pth`

### Remaining blocked/risky
- The new policy has not yet been visually inspected.
- `right_index_joint_0` is at the configured upper joint limit `0.34`; this is intentional from the tuner save but should be watched visually for contact geometry or saturation artifacts.

### Single recommended next step
- Visualize `best_reward_3305.03.pth` locally with `task=Dexh13HoraLightbulbInitpose090020233` and compare index/thumb contact against the previous 7h `Dexh13HoraLightbulb` teacher/student runs.
## v2-200 (2026-04-30) -- Sim2Real TwoFinger Thumb No-Slip YAML and Diagnostics

### Target milestone/subgoal
- Create a separate DexH13 lightbulb two-finger config focused on reducing the remaining thumb end-of-stroke slip, without overwriting the current `Dexh13HoraLightbulbSim2RealTwoFinger` baseline.

### What changed (files + behavior impact)
- Added `configs/task/Dexh13HoraLightbulbSim2RealTwoFingerThumbNoSlip.yaml`.
  - Based on the current sim2real two-finger config.
  - Keeps index+thumb only, middle/ring action mask and DOF lock.
  - Lowers controller authority:
    - `action_scale: 0.04`
    - `torque_limit: 220.0`
    - `dgain: 0.015`
  - Strengthens anti-slip behavior:
    - `pose_diff_penalty.thumb_weight: 0.3`
    - tighter thumb slip `far_dist/high_tip_speed`
    - stronger contact/far/ejection/after-drive/terminal-ease penalties
    - stricter two-finger contact force thresholds `1.5 -> 4.0`
  - Softens first-pass random external force to `forceScale=1.0`, `randomForceProbScalar=0.15`.
- Added `configs/train/Dexh13HoraLightbulbSim2RealTwoFingerThumbNoSlip.yaml`.
  - Copied from the current sim2real two-finger train config so Hydra `train: ${task}` resolves.
- Added scripts:
  - `scripts/dexh13_lightbulb_teacher_sim2real_twofinger_thumb_noslip.sh`
  - `scripts/vis_dexh13_lightbulb_teacher_sim2real_twofinger_thumb_noslip.sh`
- Updated `dexscrew/tasks/xhand_hora.py` diagnostic logging only.
  - Added unclipped per-finger:
    - `finger_contact/{index|middle|thumb}/force_raw`
    - `finger_contact/{index|middle|thumb}/normal_force`
    - `finger_tangent/{index|middle|thumb}/positive_force`
  - Added thumb-specific:
    - `thumb_slip/force_raw_mean`
    - `thumb_slip/force_raw_p05`
    - `thumb_slip/normal_force_mean`
    - `thumb_slip/normal_force_p05`
    - `thumb_slip/tangent_vel_abs_mean`
    - `thumb_slip/tangent_vel_abs_p95`
    - `thumb_slip/active_normal_drop_frac`

### What was verified (commands + key outcomes)
- Bootstrap context read:
  - `docs/session_handoff_v2.md`
  - `docs/stage_acceptance_summary.md`
- Static checks:
  - `PYTHONDONTWRITEBYTECODE=1 python - <<'PY' ... compile('dexscrew/tasks/xhand_hora.py', 'exec') ... PY`
    - Outcome: `compile ok`.
  - `git diff --check -- dexscrew/tasks/xhand_hora.py configs/task/Dexh13HoraLightbulbSim2RealTwoFingerThumbNoSlip.yaml configs/train/Dexh13HoraLightbulbSim2RealTwoFingerThumbNoSlip.yaml scripts/dexh13_lightbulb_teacher_sim2real_twofinger_thumb_noslip.sh scripts/vis_dexh13_lightbulb_teacher_sim2real_twofinger_thumb_noslip.sh`
    - Outcome: pass.
  - Docker YAML parse for the new task/train YAML:
    - Outcome: pass.
  - Hydra compose:
    - `./docker-run-isaacgym.sh bash -lc 'python train.py task=Dexh13HoraLightbulbSim2RealTwoFingerThumbNoSlip --cfg job ...'`
    - Outcome: task/train resolve, new thresholds are present.
  - Script syntax:
    - `bash -n` on both new scripts.
    - Outcome: pass.
- IsaacGym smoke:
  - `./docker-run-isaacgym.sh timeout 120 scripts/run_with_cleanup.sh bash scripts/dexh13_lightbulb_teacher_sim2real_twofinger_thumb_noslip.sh 0 42 thumb_noslip_smoke_4env True task.env.numEnvs=4 train.ppo.minibatch_size=12 train.ppo.max_agent_steps=96 wandb_activate=False task.env.termination.log=True`
  - Outcome: environment built using `screw_contactviz`, entered loop, and exited by `max steps achieved`.
  - Note: `mean_rewards: nan` is expected for this ultra-short smoke because no meaningful episode statistics exist.
- TensorBoard tag verification:
  - `EventAccumulator` found all new tags under the smoke run, including:
    - `finger_contact/thumb/force_raw`
    - `finger_contact/thumb/normal_force`
    - `thumb_slip/normal_force_p05`
    - `thumb_slip/tangent_vel_abs_p95`
    - `thumb_slip/active_normal_drop_frac`
- Cleanup:
  - No residual local `train.py` / `run_with_cleanup` thumb no-slip process remained.

### Local conclusion
- A separate no-slip experiment path is ready.
- The working hypothesis is that the remaining slip is not mainly “index too light”; prior force-weight probes saturated index contact. The more likely cause is thumb terminal impulse: high action/torque authority plus weak terminal deceleration lets the thumb generate rotation and then leave the bulb.
- The new config attacks this by reducing authority, increasing damping, tightening terminal slip penalties, and logging raw/normal force so the next run can separate:
  - true loss of normal support,
  - excessive tangential speed,
  - distance/ejection after rotation.

### Remaining blocked/risky
- This is deliberately more conservative and may reduce scalar reward/rotation speed.
- The new `normal_force` projection assumes the existing configured `force_sign` convention; if values look inverted, use raw force and visual behavior first, then flip the diagnostic sign.
- Smoke validates startup/logging only, not behavior quality.

### Single recommended next step
- Run a 60-90 minute PPO probe with `Dexh13HoraLightbulbSim2RealTwoFingerThumbNoSlip`, then compare `thumb_slip/score`, `thumb_slip/normal_force_p05`, `thumb_slip/tangent_vel_abs_p95`, `thumb_slip_penalty/terminal_ease_loss`, and visualization against `sim2real_twofinger_thumbstable`.

---
## v2-201 (2026-04-30) -- DOTPG Student Algorithm Integrated and Smoke-Validated

### Target milestone/subgoal
- Integrate the newly added `dexscrew/dotpg` DOT-PG student distillation algorithm into the active teacher-student training entrypoint so it can be launched with `train.algo=DOTPG` / `DOTPGStudent`.

### What changed (files + behavior impact)
- Updated `dexscrew/dotpg/dotpg.py`.
  - Fixed package imports from the branch-local path:
    - `dexscrew.algo.dotpg.*` -> `dexscrew.dotpg.*`.
  - Added `train.ppo.max_agent_steps` support to the DOTPG training loop.
  - Saves `model_last.ckpt` when `max_agent_steps` is reached.
- Updated `dexscrew/dotpg/__init__.py`.
  - Fixed imports to the local `dexscrew.dotpg` package.
  - Added alias `DOTPG = DOTPGStudent`.
- Updated `dexscrew/algo/student/__init__.py`.
  - Exposes `DOTPG` and `DOTPGStudent` with the other student algorithms.
- Updated `train.py`.
  - Imports `DOTPG` and `DOTPGStudent`, so `eval(config.train.algo)` can resolve both names.
- Added scripts:
  - `scripts/dexh13_lightbulb_student_dotpg_sim2real_twofinger.sh`
  - `scripts/vis_dexh13_lightbulb_student_dotpg_sim2real_twofinger.sh`
  - The training script defaults to:
    - task `Dexh13HoraLightbulbSim2RealTwoFinger`
    - `train.algo=DOTPG`
    - `train.ppo.proprio_adapt=True`
    - output root `outputs/Dexh13HoraLightbulb_student_dotpg_sim2real_twofinger/${CACHE}`
    - checkpoint argument as the PPO teacher `.pth`.

### What was verified (commands + key outcomes)
- Bootstrap context read:
  - `docs/session_handoff_v2.md`
  - `docs/stage_acceptance_summary.md`
- Static checks:
  - Python compile for:
    - `train.py`
    - `dexscrew/dotpg/__init__.py`
    - `dexscrew/dotpg/dotpg.py`
    - `dexscrew/dotpg/networks.py`
    - `dexscrew/dotpg/buffer.py`
    - `dexscrew/algo/student/__init__.py`
    - Outcome: pass.
  - `bash -n` for both new DOTPG scripts.
    - Outcome: pass.
  - `git diff --check` on touched code/scripts.
    - Outcome: pass.
- Docker import validation:
  - Imported `train` first to respect IsaacGym import order, then imported:
    - `DOTPG`
    - `DOTPGStudent`
    - `DOTPGConfig`
  - Outcome:
    - `has train DOTPG True True`
- DOTPG training smoke:
  - Command used `Dexh13HoraLightbulbSim2RealTwoFinger` with teacher checkpoint:
    `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger/sim2real_twofinger_thumbstable/stage1_nn/best_reward_6103.04.pth`
  - Overrides:
    - `task.env.numEnvs=4`
    - `train.ppo.max_agent_steps=16`
    - `train.dotpg.warmup_steps=2`
    - `train.dotpg.batch_size=4`
    - small replay/expert buffers
  - Outcome:
    - task built using `screw_contactviz`
    - teacher PPO checkpoint loaded
    - expert buffer collected and saved
    - DOTPG wrote:
      - `student_output/dotpg_nn/model_best.ckpt`
      - `student_output/dotpg_nn/model_last.ckpt`
    - exited via `max_agent_steps reached: 20 >= 16`.
- DOTPG restore/test smoke:
  - Loaded the smoke `model_best.ckpt` with `test=True`, `test_max_steps=2`.
  - Outcome:
    - model restored
    - ran two environment steps and printed `[DOTPG][TEST] step=1/2`.
- Cleanup:
  - No residual local `train.py` / `run_with_cleanup` / `dotpg_smoke` process remained.

### Local conclusion
- DOTPG is now a supported student distillation algorithm in this repo's active training entrypoint.
- It can load the existing PPO teacher checkpoint and produce DOTPG student checkpoints.
- For normal usage, prefer `train.algo=DOTPG`; `DOTPGStudent` also resolves.

### Remaining blocked/risky
- The smoke only proves wiring and runtime compatibility; it does not validate DOTPG policy quality.
- DOTPG checkpoints are under `student_output/dotpg_nn/`, not `stage2_nn/`.
- Hydra overrides for DOTPG-specific fields that are not already in the script should use `++train.dotpg.<key>=...`.
- IsaacGym printed transient PhysX warnings during the tiny 4-env smoke; the run still completed and saved checkpoints.

### Single recommended next step
- Run a short 20-30 minute DOTPG distillation from the current sim2real PPO teacher and visualize `student_output/dotpg_nn/model_best.ckpt` before committing to a longer DOTPG run.

---
## v2-202 (2026-04-30) -- SeeTaCloud DOTPG Resume 2h Launched

### Target milestone/subgoal
- Continue the existing DOTPG student distillation on the new SeeTaCloud machine for 2 hours using:
  - teacher PPO checkpoint `best_reward_6103.04.pth`
  - DOTPG cache `sim2real_twofinger_thumbstable_dotpg_bc_s42_2h`
  - DOTPG checkpoint `student_output/dotpg_nn/model_best.ckpt`

### What changed (files + behavior impact)
- Updated `dexscrew/dotpg/dotpg.py`.
  - Added DOTPG training resume config:
    - `train.dotpg.resume_path`
    - `train.dotpg.resume_load_optimizers`
  - Training now loads a DOTPG student checkpoint before continuing distillation when `resume_path` is set.
  - Resume load covers policy/target policy, critics/target critics, dual network, adapters, point MLP, running statistics, optimizers, `agent_steps`, `total_it`, and `best_rewards`.
- Added `outputs/cloud_pipeline_seetacloud_dotpg_resume2h/run_dotpg_resume_2h.sh`.
  - Runs the 2-hour continuation under `timeout 7200`.
  - Writes phase/status/logs under `outputs/cloud_pipeline_seetacloud_dotpg_resume2h`.
  - Uses the existing cache/output directory so `model_best.ckpt` and `model_last.ckpt` stay in the same DOTPG run path.

### What was verified (commands + key outcomes)
- Bootstrap context read:
  - `docs/session_handoff_v2.md`
  - `docs/stage_acceptance_summary.md`
- Local artifact check:
  - Teacher exists:
    `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger/sim2real_twofinger_thumbstable/stage1_nn/best_reward_6103.04.pth`
  - DOTPG checkpoint exists:
    `outputs/Dexh13HoraLightbulb_student_dotpg_sim2real_twofinger/sim2real_twofinger_thumbstable_dotpg_bc_s42_2h/student_output/dotpg_nn/model_best.ckpt`
  - DOTPG expert buffer exists:
    `outputs/Dexh13HoraLightbulb_student_dotpg_sim2real_twofinger/sim2real_twofinger_thumbstable_dotpg_bc_s42_2h/expert_buffer_student_raw.pt`
- Static checks:
  - `python -m py_compile dexscrew/dotpg/dotpg.py`
    - Outcome: pass.
  - `bash -n outputs/cloud_pipeline_seetacloud_dotpg_resume2h/run_dotpg_resume_2h.sh`
    - Outcome: pass.
  - `git diff --check -- dexscrew/dotpg/dotpg.py outputs/cloud_pipeline_seetacloud_dotpg_resume2h/run_dotpg_resume_2h.sh`
    - Outcome: pass.
- Checkpoint inspection:
  - Local DOTPG `model_best.ckpt` contains:
    - `agent_steps=1842912`
    - `total_it=38388`
    - `best_rewards=1425.9762383391162`
    - optimizer states present.
- Remote sync to SeeTaCloud:
  - Synced source/code changes to `/root/code/dexscrew-repro`.
  - Synced teacher PPO checkpoint, DOTPG expert buffer, DOTPG config/logs, and DOTPG `model_best.ckpt`.
- Remote resume smoke:
  - Loaded existing expert buffer:
    `从 ... expert_buffer_student_raw.pt 加载了 80000 条专家数据`
  - Loaded DOTPG resume checkpoint:
    `DOTPG student 续训检查点加载完成: agent_steps=1842912, total_it=38388, best_rewards=1425.98`
  - Short run exited cleanly via `max_agent_steps`.
- Remote resource probe:
  - Probe with `task.env.numEnvs=4096`, `train.dotpg.batch_size=2048`, `train.dotpg.bc_batch_size=4096`, 1M CPU fp16 replay/expert buffers completed startup and short continuation without OOM.
- Formal 2-hour SeeTaCloud run:
  - Launched in tmux session `seetacloud_dotpg_resume2h`.
  - Start timestamp: `2026-04-30T14:20:24+08:00`.
  - Expected timeout completion: about `2026-04-30 16:20:24 CST`.
  - Current status checked at `2026-04-30 14:25:39 CST`:
    - tmux session exists.
    - phase file: `dotpg`.
    - GPU: RTX 4090, about 6.7GB / 24.6GB used, GPU util about 65%.
    - Run has advanced beyond 4M agent steps.
    - Current best remains `1425.98` early in the continuation.
    - No `Traceback`, CUDA OOM, or PhysX GPU allocation failure found in the checked log tail.

### Current launch parameters
- Task: `Dexh13HoraLightbulbSim2RealTwoFinger`
- Teacher checkpoint:
  `/root/code/dexscrew-repro/outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger/sim2real_twofinger_thumbstable/stage1_nn/best_reward_6103.04.pth`
- DOTPG resume checkpoint:
  `/root/code/dexscrew-repro/outputs/Dexh13HoraLightbulb_student_dotpg_sim2real_twofinger/sim2real_twofinger_thumbstable_dotpg_bc_s42_2h/student_output/dotpg_nn/model_best.ckpt`
- Resource settings:
  - `task.env.numEnvs=4096`
  - `train.ppo.minibatch_size=49152`
  - `train.dotpg.batch_size=2048`
  - `train.dotpg.bc_batch_size=4096`
  - `train.dotpg.expert_add_num_envs=256`
  - `train.dotpg.online_expert_add_num_envs=256`
  - `train.dotpg.buffer_size=1000000`
  - `train.dotpg.expert_buffer_size=1000000`
  - CPU fp16 replay/expert buffers.

### Local conclusion
- DOTPG now supports true training continuation from an existing student checkpoint, not just PPO teacher loading.
- The SeeTaCloud environment can load the teacher, expert buffer, and DOTPG checkpoint correctly.
- The formal 2-hour continuation is running under a persistent tmux/timeout script with status and logs.

### Remaining blocked/risky
- The 2-hour run is still in progress; checkpoint quality has not been visualized yet.
- Current best had not improved during the early status check, but this is too early to conclude quality.
- The log includes large git-diff text emitted by `train.py`; use checkpoint timestamps, phase/status files, and exact DOTPG resume lines rather than broad grep over the whole log.

### Single recommended next step
- After `outputs/cloud_pipeline_seetacloud_dotpg_resume2h/dotpg_exit_status` appears with `124` or `0`, sync `student_output/dotpg_nn/model_best.ckpt` and visualize the DOTPG student locally.

---
## v2-203 (2026-04-30) -- SeeTaCloud DOTPG Max-Throughput Relaunch

### Target milestone/subgoal
- Re-evaluate the DOTPG continuation settings on SeeTaCloud for actual distillation efficiency, not just VRAM fill, and relaunch the 2-hour continuation with the highest measured throughput.

### What changed (files + behavior impact)
- Updated `dexscrew/dotpg/dotpg.py`.
  - Added `train.dotpg.updates_per_env_step`.
  - The DOTPG loop can now run multiple `train_step()` updates per environment interaction step.
  - TensorBoard/direct metrics are averaged across those per-step updates.
- Added probe scripts:
  - `outputs/cloud_pipeline_seetacloud_dotpg_resume2h/probe_dotpg_aggressive.sh`
  - `outputs/cloud_pipeline_seetacloud_dotpg_resume2h/probe_dotpg_aggressive_phase2.sh`
- Added formal max-throughput launch script:
  - `outputs/cloud_pipeline_seetacloud_dotpg_resume2h/run_dotpg_resume_2h_maxthroughput.sh`
  - Writes logs/status under `outputs/cloud_pipeline_seetacloud_dotpg_resume2h_maxthroughput`.

### What was verified (commands + key outcomes)
- Static checks:
  - `python -m py_compile dexscrew/dotpg/dotpg.py`
    - Outcome: pass.
  - `bash -n` on all new/probed cloud scripts.
    - Outcome: pass.
  - `git diff --check -- dexscrew/dotpg/dotpg.py outputs/cloud_pipeline_seetacloud_dotpg_resume2h/*.sh`
    - Outcome: pass.
- Stopped the earlier conservative run:
  - Previous session `seetacloud_dotpg_resume2h` was interrupted for aggressive relaunch.
  - `model_best.ckpt` was not overwritten during that early conservative continuation.
- First probe pack:
  - Summary path:
    `outputs/cloud_pipeline_seetacloud_dotpg_resume2h/aggressive_probes_20260430_143348/summary.tsv`
  - Results:
    - `cuda8192_b4096_u1`: `grad_samples_per_s=7662.6`, `max_mem_mib=15421`, `max_util_pct=48`
    - `cuda8192_b4096_u2`: `grad_samples_per_s=13054.0`, `max_mem_mib=15421`, `max_util_pct=55`
    - `cpu12288_b4096_u1`: `grad_samples_per_s=4085.8`, `max_mem_mib=12373`, `max_util_pct=77`
    - `cuda6144_b8192_u2`: `grad_samples_per_s=35461.7`, `max_mem_mib=14667`, `max_util_pct=56`
  - Note: the `oom=1` field in the first probe summary was a false positive caused by `train.py` printing historical git-diff text containing old CUDA/OOM strings; runtime grep showed no actual OOM and all statuses were `0`.
- Second probe pack:
  - Summary path:
    `outputs/cloud_pipeline_seetacloud_dotpg_resume2h/aggressive_probes_phase2_20260430_143643/summary.tsv`
  - Results:
    - `cuda6144_b8192_u3`: `grad_samples_per_s=44838.6`, `max_mem_mib=14669`, `max_util_pct=49`, `oom=0`, `traceback=0`
    - `cuda6144_b16384_u2`: `grad_samples_per_s=61020.5`, `max_mem_mib=14671`, `max_util_pct=67`, `oom=0`, `traceback=0`
    - `cuda4096_b16384_u3`: `grad_samples_per_s=83797.8`, `max_mem_mib=12675`, `max_util_pct=100`, `oom=0`, `traceback=0`
- CPU/GPU buffer conclusion:
  - CPU fp16 buffer is safe and large, but DOTPG sampling then pays CPU->GPU transfer cost every update.
  - GPU fp16 replay/expert buffers fit on the 24GB RTX 4090 and are measurably faster here.
  - 12288 env with CPU buffer used more environment parallelism but reduced effective distillation update throughput.

### Formal max-throughput run
- Launched tmux session:
  - `seetacloud_dotpg_resume2h_maxthroughput`
- Start timestamp:
  - `2026-04-30 14:39:14 CST`
- Expected timeout completion:
  - about `2026-04-30 16:39 CST`
- Status checked at `2026-04-30 14:40:39 CST`:
  - phase: `dotpg_maxthroughput`
  - GPU: RTX 4090, about `12677 / 24564 MiB`
  - DOTPG expert buffer loaded:
    `从 ... expert_buffer_student_raw.pt 加载了 80000 条专家数据`
  - DOTPG checkpoint resumed:
    `agent_steps=1842912, total_it=38388, best_rewards=1425.98`
  - Training loop active, with runtime `Last FPS` around `6k-10k`; at the selected config this corresponds roughly to `72k-120k` gradient samples/sec.

### Current launch parameters
- Task: `Dexh13HoraLightbulbSim2RealTwoFinger`
- Output/cache:
  `outputs/Dexh13HoraLightbulb_student_dotpg_sim2real_twofinger/sim2real_twofinger_thumbstable_dotpg_bc_s42_2h`
- Teacher checkpoint:
  `/root/code/dexscrew-repro/outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger/sim2real_twofinger_thumbstable/stage1_nn/best_reward_6103.04.pth`
- DOTPG resume checkpoint:
  `/root/code/dexscrew-repro/outputs/Dexh13HoraLightbulb_student_dotpg_sim2real_twofinger/sim2real_twofinger_thumbstable_dotpg_bc_s42_2h/student_output/dotpg_nn/model_best.ckpt`
- Resource settings:
  - `task.env.numEnvs=4096`
  - `train.dotpg.batch_size=16384`
  - `train.dotpg.bc_batch_size=16384`
  - `train.dotpg.updates_per_env_step=3`
  - `train.dotpg.replay_buffer_device=cuda`
  - `train.dotpg.expert_buffer_device=cuda`
  - `train.dotpg.buffer_size=1000000`
  - `train.dotpg.expert_buffer_size=1000000`
  - `train.dotpg.online_expert_add_num_envs=512`

### Local conclusion
- On this DOTPG workload, filling all 24GB VRAM is not the correct optimization target.
- The measured bottleneck is effective gradient update throughput. The best tested setting uses less VRAM than the largest-env settings but saturates GPU compute and gives the highest estimated distillation update throughput.
- The final 2-hour run is now using all-GPU buffers and the best measured probe configuration.

### Remaining blocked/risky
- `updates_per_env_step=3` is more aggressive than prior DOTPG runs; it maximizes short-probe throughput but could change off-policy stability. Monitor `Current Best`, loss metrics, and final visualization.
- The run is still in progress; final checkpoint quality is unknown until timeout completion and local visualization.

### Single recommended next step
- When `outputs/cloud_pipeline_seetacloud_dotpg_resume2h_maxthroughput/dotpg_exit_status` appears with `124` or `0`, sync `student_output/dotpg_nn/model_best.ckpt` from SeeTaCloud and run local DOTPG student visualization.

---
## v2-204 (2026-04-30) -- SeeTaCloud DOTPG CoDrive Teacher Sweep Launched

### Target milestone/subgoal
- Restart DOTPG distillation from scratch using the newly supplied CoDrive PPO teacher and matching YAML files:
  - teacher: `sim2real/codrive/best_reward_4159.37.pth`
  - task YAML: `sim2real/codrive/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.task.yaml`
  - train YAML: `sim2real/codrive/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.train.yaml`

### What changed (files + behavior impact)
- Added `outputs/cloud_pipeline_seetacloud_dotpg_codrive_sweep/run_dotpg_codrive_sweep.sh`.
  - Uses `task=Dexh13HoraLightbulbSim2RealTwoFingerCoDrive`.
  - Uses the CoDrive teacher checkpoint `sim2real/codrive/best_reward_4159.37.pth`.
  - Does not set `train.dotpg.resume_path`; every candidate starts DOTPG from scratch.
  - Copies the user-provided CoDrive task/train YAMLs from `sim2real/codrive` into `configs/task` and `configs/train` on the cloud before training, so Hydra uses the supplied CoDrive config.
  - Runs three sequential candidates with separate `timeout`s and separate output folders.

### What was verified (commands + key outcomes)
- Local file check:
  - `sim2real/codrive/best_reward_4159.37.pth`
  - `sim2real/codrive/model_best_codrive.ckpt`
  - `sim2real/codrive/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.task.yaml`
  - `sim2real/codrive/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.train.yaml`
- YAML equivalence check:
  - `sim2real/codrive/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.task.yaml` matches `configs/task/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.yaml`.
  - `sim2real/codrive/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.train.yaml` matches `configs/train/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.yaml`.
  - Hashes:
    - teacher pth: `2c3ff7411c7da40401866569116d98de73feded6f10f5431beac64f13ecd18ec`
    - task yaml: `019876eaf3de4c8bbc2180886fd90ec406b753a8cdbfce72a198c31b790c6e62`
    - train yaml: `07f2903dd9aa511c44e358757a322cf361323b2876a5c2e15f507e627ce2a627`
- Stopped the previous old-teacher fresh sweep:
  - `seetacloud_dotpg_fresh_sweep` was stopped.
  - GPU returned to `0 / 24564 MiB`.
- Synced to SeeTaCloud:
  - `sim2real/codrive/`
  - CoDrive task/train YAMLs
  - `dexscrew/dotpg/dotpg.py`
  - new CoDrive sweep script.
- Remote checks:
  - `bash -n outputs/cloud_pipeline_seetacloud_dotpg_codrive_sweep/run_dotpg_codrive_sweep.sh`
    - Outcome: pass.
  - `python -m py_compile dexscrew/dotpg/dotpg.py`
    - Outcome: pass.
  - Remote hash check matched the local CoDrive teacher/YAML hashes.

### Current cloud run
- Launched tmux session:
  - `seetacloud_dotpg_codrive_sweep`
- Start timestamp:
  - `2026-04-30 16:49:47 CST`
- Pipeline root:
  - `outputs/cloud_pipeline_seetacloud_dotpg_codrive_sweep/codrive_dotpg_s42_20260430_164945`
- Current phase at first check:
  - `baseline48_30m`
- Confirmed active command uses:
  - `task=Dexh13HoraLightbulbSim2RealTwoFingerCoDrive`
  - `checkpoint=/root/code/dexscrew-repro/sim2real/codrive/best_reward_4159.37.pth`
  - output root `outputs/Dexh13HoraLightbulb_student_dotpg_codrive/codrive_dotpg_s42_20260430_164945_*`

### Candidate schedule
- `baseline48_30m`
  - `timeout=1800`
  - `numEnvs=48`
  - CPU fp16 replay/expert buffers
  - known-good DOTPG recipe: `warmup_steps=5000`, `adapt_warmup_steps=1000`, `bc_pretrain_steps=2000`, `bc_coef=2.5`
- `gpu512_45m`
  - `timeout=2700`
  - `numEnvs=512`
  - GPU fp16 replay/expert buffers
  - moderate scaling with adapter freeze.
- `bcstrong2048_60m`
  - `timeout=3600`
  - `numEnvs=2048`
  - GPU fp16 replay/expert buffers
  - stronger BC, larger batch, one DOTPG update per env step.

### Local conclusion
- The new CoDrive DOTPG sweep is correctly pointed at the user-supplied `best_reward_4159.37.pth`; it no longer uses the older `best_reward_6103.04.pth` teacher.
- Because there is one RTX 4090, candidates are run sequentially for comparable results and to avoid multi-process interference/OOM.

### Remaining blocked/risky
- First status check happened while `train.py` was still printing a large dirty git diff into the log; the process was active but not yet past startup noise in the visible tail.
- Candidate quality is unknown until each phase writes summary rows.

### Single recommended next step
- Monitor `outputs/cloud_pipeline_seetacloud_dotpg_codrive_sweep/codrive_dotpg_s42_20260430_164945/summary.tsv`; after the sweep completes, sync the best candidate's `student_output/dotpg_nn/model_best.ckpt` locally for visualization.

---
## v2-208 (2026-04-30) -- DOTPG CoDrive Algorithm-vs-Config Controls Queued

### Target milestone/subgoal
- Continue DOTPG CoDrive distillation controls to distinguish:
  - algorithm/implementation limitation,
  - poor training configuration,
  - student adapter/proprio-state bottleneck,
  - slow-warmup behavior.

### Current observed results before extension
- Active cloud sweep:
  - tmux: `seetacloud_dotpg_codrive_sweep`
  - pipeline: `outputs/cloud_pipeline_seetacloud_dotpg_codrive_sweep/codrive_dotpg_s42_20260430_164945`
  - current phase at `2026-04-30 18:31:14 CST`: `bcstrong2048_60m`
- Completed candidates:
  - `baseline48_30m`: `max_best=146.67`, status `124`, no runtime errors.
  - `gpu512_45m`: `max_best=494.10`, status `124`, no runtime errors.
- Running candidate:
  - `bcstrong2048_60m`
  - `numEnvs=2048`, GPU fp16 replay/expert buffers, `bc_coef=4.0`, `bc_pretrain_steps=4000`
  - GPU around `9115 / 24564 MiB` at status check.

### What changed (files + behavior impact)
- Added `outputs/cloud_pipeline_seetacloud_dotpg_codrive_extended/run_dotpg_codrive_extended.sh`.
  - Runs after the current CoDrive sweep finishes.
  - Uses the same CoDrive task/YAML and teacher:
    `sim2real/codrive/best_reward_4159.37.pth`
  - Writes outputs under:
    `outputs/cloud_pipeline_seetacloud_dotpg_codrive_extended/<RUN_ID>`
    and
    `outputs/Dexh13HoraLightbulb_student_dotpg_codrive/<RUN_ID>_*`
- Launched queue tmux:
  - `seetacloud_dotpg_codrive_extended_queue`
  - It waits for `seetacloud_dotpg_codrive_sweep` to exit, then starts the extended sweep.
  - Queue status file:
    `outputs/cloud_pipeline_seetacloud_dotpg_codrive_extended/queue_status.txt`

### Extended candidate schedule
- `teacherstate512_45m`
  - Diagnostic purpose: remove the student adapter/proprio-history bottleneck.
  - `state_mode=teacher`, `dynamic_state=false`.
  - If this works much better than student-state DOTPG, the issue is mainly student representation/adaptation, not DOTPG's distribution-matching core.
- `student_slowwarm512_90m`
  - Diagnostic purpose: test slow-warmup hypothesis.
  - Longer adapt warmup/expert collection/BC pretrain:
    - `adapt_warmup_steps=3000`
    - `warmup_steps=6000`
    - `bc_pretrain_steps=10000`
    - `bc_coef=5.0`
  - Frozen adapter after warmup, lower policy/Q/dual learning rates, lower exploration/target noise.
- `student_bcanchor2048_90m`
  - Diagnostic purpose: stronger BC anchor at larger scale without multi-update overload.
  - `numEnvs=2048`, `batch_size=2048`, `bc_batch_size=8192`, `bc_coef=8.0`, `bc_pretrain_steps=8000`.

### What was verified
- Current CoDrive sweep status checked on cloud:
  - `baseline48_30m` and `gpu512_45m` completed.
  - `bcstrong2048_60m` running.
  - No stale old-teacher DOTPG process.
- New extended script checks:
  - `bash -n outputs/cloud_pipeline_seetacloud_dotpg_codrive_extended/run_dotpg_codrive_extended.sh`
    - Outcome: pass locally and remotely.
  - `git diff --check -- outputs/cloud_pipeline_seetacloud_dotpg_codrive_extended/run_dotpg_codrive_extended.sh`
    - Outcome: pass.
- Queue launched:
  - tmux `seetacloud_dotpg_codrive_extended_queue`
  - status file begins with:
    `waiting_for_codrive_sweep_2026-04-30T18:32:43+08:00`

### Local conclusion
- Existing CoDrive DOTPG student-state results remain weak so far; the best completed control is only `494.10`.
- The next critical discriminator is `teacherstate512_45m`:
  - high teacher-state score means current weakness is likely adapter/proprio-state/config;
  - low teacher-state score means the DOTPG training objective/implementation is likely not competitive for this CoDrive task.

### Remaining blocked/risky
- Extended controls are queued but not yet running; they depend on the current `bcstrong2048_60m` phase finishing.
- Long slow-warm controls take several hours; no conclusion until `summary.tsv` appears under the extended pipeline.

### Single recommended next step
- After `seetacloud_dotpg_codrive_extended_queue` starts the extended run, monitor:
  `outputs/cloud_pipeline_seetacloud_dotpg_codrive_extended/<RUN_ID>/summary.tsv`
  and compare `teacherstate512_45m` against student-state candidates.

---
---
## v2-205 (2026-04-30) -- CoDrive Diffusion Student 4-Way 1h Sweep Launched

### Target milestone/subgoal
- Use the current most stable CoDrive PPO teacher as the fixed baseline and compare the four diffusion-style student distillation algorithms under the same task/train YAML:
  - teacher: `sim2real/codrive/best_reward_4159.37.pth`
  - task: `Dexh13HoraLightbulbSim2RealTwoFingerCoDrive`
  - algorithms: `DiffusionLatentStudent`, `ConsistencyLatentStudent`, `FlowMatchingLatentStudent`, `DiffusionActionChunkStudent`

### What changed (files + behavior impact)
- Added `scripts/cloud_codrive_diffusion4_student_1h.sh`.
  - Launches all four diffusion-class student distillation runs in parallel.
  - Uses `timeout 3600` per algorithm.
  - Uses separate output/log directories per algorithm.
  - Uses the same CoDrive task/train YAML and the same PPO teacher checkpoint for all candidates.
  - Uses the same student runtime overrides as the recent CoDrive PAdapt validation:
    - `task.env.numEnvs=48`
    - `train.ppo.minibatch_size=576`
    - `task.env.termination.grace_steps=0`
    - termination gates enabled
    - `obs_noise_t_scale=0.01`
    - `obs_noise_e_scale=0.02`

### What was verified (commands + key outcomes)
- Local static checks:
  - `bash -n scripts/cloud_codrive_diffusion4_student_1h.sh`
    - Outcome: pass.
  - `git diff --check -- scripts/cloud_codrive_diffusion4_student_1h.sh`
    - Outcome: pass.
- Synced to SeeTaCloud:
  - `train.py`
  - `dexscrew/`
  - `configs/`
  - `scripts/`
  - `sim2real/codrive/`
- Remote checks:
  - `bash -n scripts/cloud_codrive_diffusion4_student_1h.sh`
    - Outcome: pass.
  - `python -m py_compile` for the four diffusion student files and `train.py`
    - Outcome: pass.
  - Remote hashes matched local:
    - teacher pth: `2c3ff7411c7da40401866569116d98de73feded6f10f5431beac64f13ecd18ec`
    - task yaml: `019876eaf3de4c8bbc2180886fd90ec406b753a8cdbfce72a198c31b790c6e62`
    - train yaml: `07f2903dd9aa511c44e358757a322cf361323b2876a5c2e15f507e627ce2a627`
- Cloud GPU preflight:
  - RTX 4090D was idle before launch: about `1 / 24564 MiB`, `0%` util.

### Current cloud run
- tmux session:
  - `codrive_diffusion4_1h_085749`
- Pipeline latest symlink:
  - `outputs/cloud_pipeline_codrive_diffusion4_1h/latest`
- Concrete run tag:
  - `codrive_diffusion4_s42_1h_20260430_085750`
- Pipeline root:
  - `outputs/cloud_pipeline_codrive_diffusion4_1h/codrive_diffusion4_s42_1h_20260430_085750`
- Status checked around `2026-04-30 17:00 CST`:
  - phase: `running`
  - four `python train.py` processes active
  - GPU: about `10809 / 24564 MiB`, util `93-99%`
  - no actual CUDA OOM observed during startup.

### Output directories
- Diffusion latent:
  - `outputs/Dexh13HoraLightbulb_student_diffusion_latent_codrive/codrive_diffusion4_s42_1h_20260430_085750/stage2_diffusion_nn/`
- Consistency latent:
  - `outputs/Dexh13HoraLightbulb_student_consistency_codrive/codrive_diffusion4_s42_1h_20260430_085750/stage2_consistency_nn/`
- Flow matching latent:
  - `outputs/Dexh13HoraLightbulb_student_flow_matching_codrive/codrive_diffusion4_s42_1h_20260430_085750/stage2_flow_nn/`
- Diffusion action chunk:
  - `outputs/Dexh13HoraLightbulb_student_diffusion_action_chunk_codrive/codrive_diffusion4_s42_1h_20260430_085750/stage2_diffusion_action_chunk_nn/`

### Remaining blocked/risky
- The four runs are still in progress; final checkpoint quality is unknown until the 1h timeouts finish.
- `train.py` prints large dirty git diffs into each log, so broad grep for historical terms like `Traceback`/`OOM` can produce false positives from embedded docs/diff text. Prefer each log tail, status files, and `summary.tsv`.
- Four processes share one GPU. Startup memory is safe, but final throughput/quality may differ from single-process PAdapt because GPU compute is shared.

### Single recommended next step
- After `outputs/cloud_pipeline_codrive_diffusion4_1h/latest/summary.tsv` is written and all statuses are `0` or `124`, sync the four `model_best.ckpt` files locally and visualize/evaluate them under `task=Dexh13HoraLightbulbSim2RealTwoFingerCoDrive`.

---
## v2-206 (2026-04-30) -- BC / DAgger Student Baselines Integrated

### Target milestone/subgoal
- Migrate the user-supplied `algo/student/bc+dagger` code into the active teacher-student pipeline as baseline student distillation algorithms, so later experiments can compare PAdapt / DOTPG / diffusion / BC / DAgger under the same PPO teacher.

### What changed (files + behavior impact)
- Added canonical student modules:
  - `dexscrew/algo/student/bc_student.py`
  - `dexscrew/algo/student/bc_buffer.py`
  - `dexscrew/algo/student/dagger_student.py`
  - `dexscrew/algo/student/dagger_buffer.py`
- Registered the new algorithms in `dexscrew/algo/student/__init__.py` and `train.py`.
  - Hydra `train.algo=BCStudent` and `train.algo=DAggerStudent` now resolve through the normal `train.py` entrypoint.
  - Aliases `BC` and `DAgger` are also exported for compatibility.
- Added baseline scripts:
  - `scripts/dexh13_lightbulb_student_bc_sim2real_twofinger.sh`
  - `scripts/dexh13_lightbulb_student_dagger_sim2real_twofinger.sh`
  - `scripts/vis_dexh13_lightbulb_student_bc_sim2real_twofinger.sh`
  - `scripts/vis_dexh13_lightbulb_student_dagger_sim2real_twofinger.sh`
- Implementation fixes:
  - Rewired old imports from nonexistent `dexscrew.algo.BC/DAgger` packages to `dexscrew.algo.student.*`.
  - `max_agent_steps=0` now inherits `train.ppo.max_agent_steps`.
  - Save/eval intervals default to `1e5` agent steps instead of the old very large defaults, so time-limited runs are more likely to produce checkpoints.
  - BC saves `model_last` after demo collection / pretrain and runs an immediate pure-student eval after pretrain when eval is enabled.
  - DAgger saves `model_last` at startup and keeps `model_best_mixed` separate from pure-student `model_best`.
  - BC buffer save now clones sliced tensors before `torch.save`, avoiding serialization of the whole preallocated buffer capacity.

### What was verified (commands + key outcomes)
- Static checks:
  - `PYTHONDONTWRITEBYTECODE=1 python - <<'PY' ... compile(...) ... PY`
    - Outcome: `compile_ok`.
  - `bash -n` for all four new scripts.
    - Outcome: pass.
  - `git diff --check -- train.py dexscrew/algo/student/* scripts/dexh13_lightbulb_student_* scripts/vis_dexh13_lightbulb_student_*`
    - Outcome: pass.
- Docker import check:
  - `./docker-run-isaacgym.sh python -c "import train; from dexscrew.algo.student import BCStudent, DAggerStudent, BC, DAgger; ..."`
    - Outcome: `student_algos_ok BCStudent DAggerStudent BCStudent DAggerStudent`.
- BC smoke:
  - Used PPO teacher `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger/sim2real_twofinger_thumbstable/stage1_nn/best_reward_6103.04.pth`.
  - `task.env.numEnvs=4`, tiny demo/pretrain settings.
  - Outcome: loaded teacher checkpoint, collected/saved demo buffer, trained `adapt_tconv`, wrote `bc_nn/model_last.ckpt`.
- DAgger smoke:
  - Same PPO teacher and 4-env settings.
  - Outcome: loaded teacher checkpoint, rolled student/teacher mixed actions, aggregated teacher labels, trained supervised adapter, wrote `dagger_nn/model_last.ckpt` and `dagger_nn/model_best_mixed.ckpt`.
- BC buffer serialization check:
  - A synthetic 16-sample buffer with `max_size=200000` now saves as about `36KB`, confirming the clone fix works.

### Local conclusion
- BC and DAgger are now usable as first-class student distillation baselines inside the existing Hora teacher-student system.
- The original copied folder under `dexscrew/algo/student/bc+dagger代码（无best ckpt）/` remains as backup/reference only; canonical runs should use the new modules and scripts.

### Remaining blocked/risky
- Smoke tests validate wiring, checkpoint loading, buffer flow, and script entrypoints; they do not validate policy quality.
- Real BC / DAgger runs should be compared by pure-student `model_best.ckpt`, not DAgger's `model_best_mixed.ckpt`, because the mixed metric can include teacher beta actions.
- Default BC/DAgger buffer devices are CPU/fp16 to avoid VRAM pressure; throughput may be lower than all-GPU DOTPG/PAdapt.

### Single recommended next step
- Run 20-30 minute baseline probes from the current PPO teacher for both `BCStudent` and `DAggerStudent`, then visualize their `model_best.ckpt` policies and compare against the existing PAdapt and DOTPG students.
---
## v2-206 (2026-04-30) -- CoDrive Diffusion Student 4-Way 1h Sweep Completed

### Target milestone/subgoal
- Finish the 1h four-way diffusion-class student distillation sweep from the CoDrive PPO teacher and produce synchronized artifacts plus a quick numerical comparison.

### What changed (files + behavior impact)
- No additional code changes after launch.
- Synced completed cloud artifacts back to local:
  - `outputs/cloud_pipeline_codrive_diffusion4_1h/codrive_diffusion4_s42_1h_20260430_085750/`
  - `outputs/Dexh13HoraLightbulb_student_diffusion_latent_codrive/codrive_diffusion4_s42_1h_20260430_085750/`
  - `outputs/Dexh13HoraLightbulb_student_consistency_codrive/codrive_diffusion4_s42_1h_20260430_085750/`
  - `outputs/Dexh13HoraLightbulb_student_flow_matching_codrive/codrive_diffusion4_s42_1h_20260430_085750/`
  - `outputs/Dexh13HoraLightbulb_student_diffusion_action_chunk_codrive/codrive_diffusion4_s42_1h_20260430_085750/`

### What was verified (commands + key outcomes)
- Cloud run completed at `2026-04-30T09:58:50+00:00` / `2026-04-30 17:58:50 CST`.
- Cloud GPU returned to idle after completion: about `1 / 24564 MiB`, `0%`.
- `summary.tsv` statuses:
  - all four algorithms ended with status `124`, expected from the 3600s timeout.
- Training-window best rewards:
  - `DiffusionLatentStudent`: `4145.42`
  - `FlowMatchingLatentStudent`: `3975.03`
  - `ConsistencyLatentStudent`: `3837.96`
  - `DiffusionActionChunkStudent`: `3623.38`
- Local checkpoint presence verified:
  - `stage2_diffusion_nn/model_best.ckpt`
  - `stage2_consistency_nn/model_best.ckpt`
  - `stage2_flow_nn/model_best.ckpt`
  - `stage2_diffusion_action_chunk_nn/model_best.ckpt`
  - action-chunk extras: `model_best_student.ckpt`, `model_best_student_reward.ckpt`, `model_best_deploy_probe.ckpt`, `model_last.ckpt`
- Checkpoint SHA256:
  - diffusion latent `model_best.ckpt`: `9139111b8661e4536421ec19a867d9bdba4b66d999584ced46376392a042bac9`
  - consistency latent `model_best.ckpt`: `923168a082adf7b2c90d826b24b51f489ea88556adb9df8d018dc006fbede998`
  - flow matching `model_best.ckpt`: `d92f643d0d905a3afd9c81ec8be6397ed02c1b89d61ca88467ce0957ebfb7ac5`
  - action chunk `model_best.ckpt`: `877625cae129fd45b04fce2666f304872943ec2ada9ae35161f46945449a97cc`
  - action chunk `model_best_student_reward.ckpt`: `6f0831df954ad4af5c593d123a657279c207b36403d3b12aa0be5edd8aa8cab9`
- Cloud 256-step headless eval was run with the same CoDrive task and student overrides:
  - `DiffusionLatentStudent`: `avg_reward=2.855633`, `avg_done_rate=0.000895`
  - `ConsistencyLatentStudent`: `avg_reward=5.155176`, `avg_done_rate=0.000000`
  - `FlowMatchingLatentStudent`: `avg_reward=4.827848`, `avg_done_rate=0.000081`
  - `DiffusionActionChunkStudent model_best`: `avg_reward=-3.786618`, `avg_done_rate=0.015625`
  - `DiffusionActionChunkStudent model_best_student_reward`: `avg_reward=0.178947`, `avg_done_rate=0.007975`

### Local conclusion
- Parallel 4-way diffusion distillation is feasible on the 24GB RTX 4090D for this CoDrive student setup:
  - startup/runtime memory stayed around `10.8GB / 24GB`;
  - all four processes completed the intended 1h timeout window.
- Training-window reward and deploy-style eval disagree:
  - training best ranks diffusion latent highest;
  - 256-step eval ranks consistency latent highest, then flow matching.
- Action-chunk is currently not competitive in deploy eval despite writing multiple checkpoint variants.

### Remaining blocked/risky
- Numerical eval is still not a substitute for visual policy inspection. Need local viewer comparison to check thumb/index cooperation, slip, and stroke reset behavior.
- The eval used one seed and a short 256-step window; use it as a fast screen, not a final paper metric.

### Single recommended next step
- Locally visualize `ConsistencyLatentStudent` first, then `FlowMatchingLatentStudent`, then `DiffusionLatentStudent`; only inspect action-chunk if those three fail visually.
---
## v2-207 (2026-04-30) -- CoDrive BC + DAgger 1h Parallel Baseline Completed

### Target milestone/subgoal
- Use the stable CoDrive PPO teacher as the common baseline and run the newly integrated BC / DAgger student distillation baselines in parallel for 1 hour.

### What changed (files + behavior impact)
- Added CoDrive baseline scripts:
  - `scripts/dexh13_lightbulb_student_bc_codrive.sh`
  - `scripts/dexh13_lightbulb_student_dagger_codrive.sh`
  - `scripts/vis_dexh13_lightbulb_student_bc_codrive.sh`
  - `scripts/vis_dexh13_lightbulb_student_dagger_codrive.sh`
  - `outputs/codrive_bc_dagger_baseline_1h/run_codrive_bc_dagger_1h_parallel.sh`
- Both train scripts use:
  - task `Dexh13HoraLightbulbSim2RealTwoFingerCoDrive`
  - teacher `sim2real/codrive/best_reward_4159.37.pth`
  - CPU fp16 replay/demo buffers to keep VRAM low.

### What was verified (commands + key outcomes)
- Confirmed local CoDrive task/train YAMLs match `sim2real/codrive` copies by byte comparison.
- Verified teacher checkpoint and YAML hashes:
  - teacher pth `2c3ff7411c7da40401866569116d98de73feded6f10f5431beac64f13ecd18ec`
  - task yaml `019876eaf3de4c8bbc2180886fd90ec406b753a8cdbfce72a198c31b790c6e62`
  - train yaml `07f2903dd9aa511c44e358757a322cf361323b2876a5c2e15f507e627ce2a627`
- `bash -n` passed for the new train/vis/parallel scripts.
- `git diff --check` passed for the new scripts.
- Docker import check passed for `train`, `BCStudent`, and `DAggerStudent`.
- Parallel run:
  - command tag `codrive_bc_dagger_s42_1h_0430_local`
  - started `2026-04-30T17:13:34+08:00`
  - ended `2026-04-30T18:14:37+08:00`
  - `bc_exit_status=124`, `dagger_exit_status=124`, expected from the requested `timeout 3600`.
- GPU monitor during parallel run:
  - total VRAM around `6.4-6.6GB / 16GB`;
  - each student process stayed roughly `2.5-2.7GB`, comfortably below the requested 8GB per process.
- Post-run check:
  - no residual `python train.py` BC/DAgger process remained.

### Results
- BC student:
  - best pure-student eval: `1640.03`
  - checkpoint: `outputs/Dexh13HoraLightbulb_student_bc_codrive/codrive_bc_s42_1h/bc_nn/model_best.ckpt`
  - demo buffer saved normally at about `178MB`, indicating the buffer save clone fix worked.
  - later evals dropped, so use `model_best.ckpt`, not `model_last.ckpt`.
- DAgger student:
  - best pure-student eval: `1318.65`
  - checkpoint: `outputs/Dexh13HoraLightbulb_student_dagger_codrive/codrive_dagger_s42_1h/dagger_nn/model_best.ckpt`
  - `model_best_mixed.ckpt` exists but includes teacher-beta mixed rollout metric and should not be used as final pure-student comparison.

### Local conclusion
- BC / DAgger are now executable CoDrive student baselines in the active teacher-student chain.
- Numerically, 1h BC is stronger than 1h DAgger on pure-student eval, but BC is visibly unstable in scalar trajectory and needs viewer confirmation.
- Both are well below the CoDrive teacher and the stronger diffusion/PAdapt-style student results, so these currently look like baseline comparisons rather than best deployment candidates.

### Remaining blocked/risky
- No visual inspection yet for either BC or DAgger `model_best.ckpt`; behavior quality may not match scalar ranking.
- A nonfatal PhysX warning appeared once during DAgger eval (`PxScene::applyArticulationData...`), with no crash and checkpoints still written.
- Log grep is noisy because train outputs include the dirty `gitdiff.patch`; inspect actual runtime tail before treating old diff text as current errors.

### Single recommended next step
- Visualize BC and DAgger `model_best.ckpt` with the new CoDrive vis scripts, then compare against CoDrive diffusion / PAdapt / DOTPG candidates for the baseline summary.
---
## v2-207 (2026-04-30) -- CoDrive Diffusion Student Continue-2h From 1h Launched

### Target milestone/subgoal
- Continue the four diffusion-class CoDrive student algorithms from their completed 1h checkpoints for another 2h each, instead of restarting from the PPO teacher.

### What changed (files + behavior impact)
- Updated `dexscrew/algo/ppo/diffusion_latent_student.py`.
  - `restore_train()` now loads `sa_mean_std` when present in a diffusion-latent student checkpoint.
- Updated `dexscrew/algo/ppo/diffusion_action_chunk_student.py`.
  - Imported `cprint`.
  - `restore_train()` now loads the action-chunk `diffusion_model` from checkpoint.
  - Without this patch, action-chunk continuation would only load the backbone via `ProprioAdapt.restore_train()` and would not truly continue the action-chunk diffusion head.
- Added `scripts/cloud_codrive_diffusion4_student_continue2h.sh`.
  - Runs four parallel 2h continuation jobs.
  - Uses `WINDOW_SEC=7200`.
  - Uses the completed 1h run as `BASE_RUN=codrive_diffusion4_s42_1h_20260430_085750`.
  - Writes new outputs under `codrive_diffusion4_s42_continue2h_from_1h_...`, preserving the original 1h outputs.
- Updated `scripts/cloud_codrive_diffusion4_student_1h.sh`.
  - Made `PIPE_ROOT` configurable.
  - Made default `RUN_TAG` include `${WINDOW_SEC}s`.

### What was verified (commands + key outcomes)
- Local static checks:
  - Python compile via `compile(..., 'exec')` for patched student files.
    - Outcome: pass.
  - `bash -n scripts/cloud_codrive_diffusion4_student_continue2h.sh scripts/cloud_codrive_diffusion4_student_1h.sh`
    - Outcome: pass.
  - `git diff --check` on patched files/scripts.
    - Outcome: pass.
- Note:
  - Local `python -m py_compile` could not write into root-owned `dexscrew/algo/ppo/__pycache__`; this is a local permissions issue from prior root/Docker runs, not a syntax failure.
- Remote checks:
  - Synced patched files/scripts to SeeTaCloud in their correct paths.
  - Remote `bash -n` pass.
  - Remote `python -m py_compile` pass.
  - Remote GPU idle before launch: about `1 / 24564 MiB`.

### Current cloud run
- tmux session:
  - `codrive_diffusion4_continue2h_103524`
- Pipeline latest symlink:
  - `outputs/cloud_pipeline_codrive_diffusion4_continue2h/latest`
- Concrete run tag:
  - `codrive_diffusion4_s42_continue2h_from_1h_20260430_103525`
- Status checked shortly after launch:
  - phase: `running`
  - four `python train.py` processes active
  - GPU: about `10809 / 24564 MiB`, util about `93%`
- Resume verification lines:
  - diffusion latent:
    - resume ckpt: `outputs/Dexh13HoraLightbulb_student_diffusion_latent_codrive/codrive_diffusion4_s42_1h_20260430_085750/stage2_diffusion_nn/model_best.ckpt`
    - loaded `diffusion_model`
    - loaded `diffusion_optim`
  - consistency latent:
    - resume ckpt: `outputs/Dexh13HoraLightbulb_student_consistency_codrive/codrive_diffusion4_s42_1h_20260430_085750/stage2_consistency_nn/model_best.ckpt`
    - loaded `consistency_model`
    - loaded `consistency_optim`
    - loaded `agent_steps=653232`
  - flow matching:
    - resume ckpt: `outputs/Dexh13HoraLightbulb_student_flow_matching_codrive/codrive_diffusion4_s42_1h_20260430_085750/stage2_flow_nn/model_best.ckpt`
    - loaded `flow_model`
    - loaded `flow_optim`
  - diffusion action chunk:
    - resume ckpt: `outputs/Dexh13HoraLightbulb_student_diffusion_action_chunk_codrive/codrive_diffusion4_s42_1h_20260430_085750/stage2_diffusion_action_chunk_nn/model_best.ckpt`
    - loaded `action-chunk diffusion_model`

### Remaining blocked/risky
- Run is still in progress; final 2h continuation quality is unknown.
- `Current Best` in the new output starts from the new run's meter, not necessarily the previous 1h training best, so early lower values do not imply the checkpoint was not loaded.
- Logs still contain large printed git diffs; avoid broad grep false positives.

### Single recommended next step
- After `outputs/cloud_pipeline_codrive_diffusion4_continue2h/latest/summary.tsv` is written and statuses are `0` or `124`, sync the four continuation outputs locally and run the same 256-step headless eval used for the 1h comparison.
---
## v2-209 (2026-05-04) -- CoDrive Teacher TensorBoard Diagnostic Review

### Target milestone/subgoal
- Inspect the fixed CoDrive PPO teacher baseline and decide whether it still needs environment/reward micro-tuning before being frozen as the standard teacher for student distillation.

### What changed (files + behavior impact)
- No code/config changes.
- Parsed TensorBoard scalars from:
  - `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive/sim2real_twofinger_codrive_s42_2h/stage1_tb/events.out.tfevents.1777522009.di-20260428205452-zqhv8`
  - compared against old thumbstable teacher event:
    `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger/sim2real_twofinger_thumbstable/stage1_tb/events.out.tfevents.1777489633.wbz-ubuntu22-pc`
  - also spot-checked CoDrive PAdapt student event:
    `outputs/Dexh13HoraLightbulb_student_padapt_sim2real_twofinger_codrive/sim2real_twofinger_codrive_padapt_s42_2h/stage2_tb/events.out.tfevents.1777531369.di-20260428205452-zqhv8`

### What was verified (commands + key outcomes)
- Docker has TensorBoard parser available; local host Python does not.
- CoDrive PPO best:
  - `episode_rewards/step` max `4159.37` at step `191545344`
  - best elapsed time scalar about `116.27 min`
  - final reward stayed close: last `4136.52`, last-10 mean `4137.71`
- Stability:
  - `two_finger/gate` final `0.9933`, last-10 mean `0.9937`
  - `term/any_reset_frac` final `0.00049`, last-10 mean `0.00075`
  - `term/no_contact_frac = 0`, `term/screw_limit_frac = 0`
- CoDrive vs old thumbstable teacher, last-10 mean:
  - reward: `4137.71` vs `5954.8` (not directly comparable due reward redesign)
  - angular velocity: `1.2176` vs `0.9176`
  - index torque ratio: `0.4486` vs `0.8015`
  - thumb torque ratio: `0.5514` vs `0.1985`
  - thumb slip score: `0.0080` vs `0.0087`
  - thumb tip-speed p95: `0.5577` vs `0.4204`
- CoDrive PPO final drive decomposition:
  - index positive torque final `1.373`
  - thumb positive torque final `4.143`
  - index positive tangent velocity final `0.0199`
  - thumb positive tangent velocity final `0.0836`
  - active two-finger penalty final `0`
- A dead reward component was found:
  - `opposition_grip/reward_scaled = 0` for the whole CoDrive teacher run.
  - `opposition_grip/oppositeness = 0`, final radial dot about `0.259`.
  - `opposition_grip/pair_inward_w = 0`, inward forces are large negative with current sign.
- CoDrive PAdapt student event is consistent with teacher tendency:
  - `episode_rewards/step` max `3448.82`
  - last `env/thumb_slip/tip_speed_p95/frame = 0.6965`
  - last `env/opposition_grip/reward_scaled/frame = 0`

### Local conclusion
- CoDrive is a usable fixed teacher baseline: stable resets, strong gate, and better positive angular velocity than thumbstable.
- It is not perfectly clean as a final sim2real reward design:
  - the enabled `opposition_grip_reward` is effectively dead and should either be fixed or removed before calling the YAML final;
  - CoDrive achieved the intended two-finger torque sharing, but thumb still dominates tangent motion;
  - terminal thumb speed is higher than the old thumbstable teacher, which matches the visual end-of-stroke detach concern.

### Remaining blocked/risky
- Need visual judgment before deciding whether the current CoDrive teacher is "good enough" to freeze despite the dead opposition term.
- If tuning continues, avoid using raw reward as the only decision metric because CoDrive reward is intentionally redesigned.

### Single recommended next step
- If freezing now, document CoDrive as the standard student-distillation teacher and note the dead opposition term as a known limitation. If doing one more cleanup pass, run a bounded 2-variant PPO probe: one terminal-ease/torque-limit variant for thumb slip, and one opposition-grip target/sign diagnostic variant to make the clamp reward nonzero.
---
## v2-208 (2026-05-04) -- CoDrive Diffusion Continue-2h Synced and Evaled

### Target milestone/subgoal
- Recover status after the interrupted wait, sync the completed CoDrive diffusion continue-2h artifacts, and compare the 1h+2h continuation against the earlier 1h student results.

### What changed (files + behavior impact)
- No additional training code changes.
- Synced cloud artifacts locally:
  - `outputs/cloud_pipeline_codrive_diffusion4_continue2h/codrive_diffusion4_s42_continue2h_from_1h_20260430_103525/`
  - `outputs/Dexh13HoraLightbulb_student_diffusion_latent_codrive/codrive_diffusion4_s42_continue2h_from_1h_20260430_103525/`
  - `outputs/Dexh13HoraLightbulb_student_consistency_codrive/codrive_diffusion4_s42_continue2h_from_1h_20260430_103525/`
  - `outputs/Dexh13HoraLightbulb_student_flow_matching_codrive/codrive_diffusion4_s42_continue2h_from_1h_20260430_103525/`
  - `outputs/Dexh13HoraLightbulb_student_diffusion_action_chunk_codrive/codrive_diffusion4_s42_continue2h_from_1h_20260430_103525/`
- Ran and synced cloud 256-step headless eval under the same CoDrive task/eval overrides:
  - `outputs/cloud_pipeline_codrive_diffusion4_continue2h/codrive_diffusion4_s42_continue2h_from_1h_20260430_103525/eval_256/`

### What was verified (commands + key outcomes)
- Cloud status:
  - phase: `done`
  - GPU idle: about `1 / 24564 MiB`, `0%`
  - pipeline ended at `2026-04-30T12:36:25+00:00`
  - all four continuation statuses are `124`, expected from `timeout 7200`
- Continue-2h training-window best rewards:
  - `DiffusionLatentStudent`: `4246.32`
  - `FlowMatchingLatentStudent`: `4084.44`
  - `ConsistencyLatentStudent`: `4055.97`
  - `DiffusionActionChunkStudent`: `3620.43`
- Continue-2h 256-step eval:
  - `DiffusionLatentStudent`: `avg_reward=2.942062`, `avg_done_rate=0.001465`
  - `ConsistencyLatentStudent`: `avg_reward=4.979207`, `avg_done_rate=0.000000`
  - `FlowMatchingLatentStudent`: `avg_reward=4.364139`, `avg_done_rate=0.000570`
  - `DiffusionActionChunkStudent model_best`: `avg_reward=0.190792`, `avg_done_rate=0.015625`
  - `DiffusionActionChunkStudent model_best_student_reward`: `avg_reward=0.296496`, `avg_done_rate=0.007406`

### Local conclusion
- The apparent contradiction is metric-dependent:
  - training-window best reward ranks diffusion latent highest;
  - fixed 256-step deploy eval still ranks consistency latent highest.
- Continuation improved training-window rewards for latent/flow/consistency, but did not improve the short deploy eval ranking:
  - 1h eval consistency was `5.155176`; continue-2h consistency is `4.979207`.
  - 1h eval flow was `4.827848`; continue-2h flow is `4.364139`.
  - 1h eval latent was `2.855633`; continue-2h latent is `2.942062`.
- Current best numerical candidate for deploy-style evaluation remains `ConsistencyLatentStudent`, not diffusion latent.

### Remaining blocked/risky
- Visual inspection is still required. The metric may miss thumb/index motion quality, slip, stroke reset behavior, and visual smoothness.
- The eval is single-seed, 256-step, and should be treated as a fast screen.

### Single recommended next step
- Visualize the continue-2h consistency checkpoint first, then compare it against the original 1h consistency and continue-2h latent/flow.
---
## v2-210 (2026-05-04) -- CoDrive Thesis Return-Contact YAML Added

### Target milestone/subgoal
- Create a CoDrive thesis variant that reduces thumb end-of-stroke fling and discourages loaded backward dragging during the return/reset stroke, without strengthening always-on grasping.

### What changed (files + behavior impact)
- Updated `dexscrew/tasks/xhand_hora.py`.
  - Added optional `thumb_slip_penalty` return-stroke contact loss.
  - New config keys:
    - `return_contact_penalty_scale`
    - `return_tangent_vel_threshold`
    - `return_tangent_vel_span`
  - New TensorBoard/W&B extras:
    - `thumb_slip_penalty/return_contact_loss`
    - `thumb_slip_penalty/return_phase_w`
    - `thumb_slip_penalty/return_context_w`
    - `thumb_slip_penalty/thumb_tangent_vel`
  - Default scale is `0.0`, so existing CoDrive/thumbstable/student configs are behavior-preserving unless the new key is explicitly enabled.
- Added `configs/task/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveThesis.yaml`.
  - Forked from CoDrive.
  - `eval_cache_name: sim2real_twofinger_codrive_thesis`
  - `controller.torque_limit: 250.0`
  - Thumb terminal slowing made stronger:
    - `high_tip_speed: 0.15`
    - `high_tip_speed_span: 0.15`
    - `terminal_ease_penalty_scale: -0.5`
    - `terminal_ease_near_limit: 0.70`
    - `terminal_ease_vel_clip: 1.2`
  - Return-stroke pressure release enabled:
    - `return_contact_penalty_scale: -0.6`
    - `return_tangent_vel_threshold: 0.015`
    - `return_tangent_vel_span: 0.10`
- Added `configs/train/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveThesis.yaml`.
  - Same PPO train defaults as CoDrive so Hydra `train: ${task}` resolves cleanly.

### What was verified (commands + key outcomes)
- Static compile:
  - `PYTHONDONTWRITEBYTECODE=1 python - <<'PY' ... compile('dexscrew/tasks/xhand_hora.py') ... PY`
  - outcome: pass.
- Whitespace check:
  - `git diff --check -- dexscrew/tasks/xhand_hora.py configs/task/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveThesis.yaml configs/train/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveThesis.yaml`
  - outcome: pass.
- Docker smoke:
  - `./docker-run-isaacgym.sh timeout 180 scripts/run_with_cleanup.sh python train.py task=Dexh13HoraLightbulbSim2RealTwoFingerCoDriveThesis headless=True seed=42 train.algo=PPO wandb_activate=False task.env.numEnvs=4 train.ppo.minibatch_size=12 train.ppo.max_agent_steps=24 train.ppo.output_name=Dexh13HoraLightbulb_teacher/smoke_codrive_thesis_tmp task.env.termination.log=True`
  - outcome: Hydra resolved the new task/train YAML, environment built with `screw_contactviz`, 4 envs initialized, and the run ended with `max steps achieved`.

### Local conclusion
- The thesis variant is ready for a short PPO probe.
- This variant targets "release pressure during return" rather than "grip harder", matching the deployment observation that the thumb can stay too loaded while resetting to the initial stroke point.

### Remaining blocked/risky
- No medium/long PPO result yet. The new return-contact penalty could reduce rotation efficiency if too strong.
- Watch `thumb_slip_penalty/return_contact_loss`, `thumb_slip_penalty/thumb_tangent_vel`, `thumb_slip/tip_speed_p95`, `screw/angular_velocity`, and visual return-stroke behavior.

### Single recommended next step
- Run a 30-60 minute PPO probe with `task=Dexh13HoraLightbulbSim2RealTwoFingerCoDriveThesis`, then visualize whether thumb return pressure drops without losing the stable two-finger contact pattern.
---
## v2-211 (2026-05-04) -- Eval-Best Selection Mechanism Implemented

### Target milestone/subgoal
- Add teacher/student eval-best checkpoint selection so deployment candidates are selected by fixed rollout eval, not only by noisy training reward.

### What changed (files + behavior impact)
- Added `dexscrew/algo/eval_select.py`.
  - Shared in-training eval helper for PPO teacher and student distillers.
  - Computes `score = avg_reward - done_penalty * avg_done_rate`.
  - Logs `outputs/<run>/eval_select/train_eval_history.tsv`.
  - Saves eval-selected artifacts while preserving train-reward artifacts.
- Updated `dexscrew/algo/ppo/ppo.py`.
  - Optional teacher eval-select via `train.ppo.eval_select`.
  - Keeps old `best_reward_*.pth`.
  - Adds `stage1_nn/best_eval.pth` and `stage1_nn/best_deploy.pth` when enabled.
- Updated student distillers:
  - `dexscrew/algo/ppo/padapt.py`
  - `dexscrew/algo/ppo/diffusion_latent_student.py`
  - `dexscrew/algo/ppo/consistency_latent_student.py`
  - `dexscrew/algo/ppo/flow_matching_latent_student.py`
  - `dexscrew/algo/ppo/diffusion_action_chunk_student.py`
  - When eval-select is enabled, training-reward best is saved as `model_best_train.ckpt`; eval-selected best is saved as `model_best_eval.ckpt`, copied to `model_best.ckpt`, and also copied to `model_best_deploy.ckpt`.
- Added `scripts/eval_select_checkpoints.py`.
  - Runs fixed `train.py test=True +test_num_steps=...` evals over checkpoint candidates.
  - Supports default `train_like/clean/light/hard` condition grid or custom `--condition`.
  - Writes `summary.tsv` and `ranking.tsv`, then copies the best candidate to `best_deploy.pth` or `model_best_deploy.ckpt`.
- Added default-disabled `train.ppo.eval_select` blocks to current CoDrive train YAMLs:
  - `configs/train/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.yaml`
  - `configs/train/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveThesis.yaml`

### What was verified (commands + key outcomes)
- Static compile without writing pyc:
  - `python -c "import pathlib; files=[...]; [compile(pathlib.Path(f).read_text(), f, 'exec') for f in files]; print('compile ok')"`
  - outcome: pass.
- Whitespace check:
  - `git diff --check -- dexscrew/algo/eval_select.py ... scripts/eval_select_checkpoints.py ...`
  - outcome: pass.
- PPO teacher smoke:
  - `./docker-run-isaacgym.sh timeout 240 python train.py task=Dexh13HoraLightbulbSim2RealTwoFingerCoDrive headless=True seed=42 train.algo=PPO ... train.ppo.eval_select.enabled=True train.ppo.eval_select.interval_agent_steps=1 train.ppo.eval_select.num_steps=4 train.ppo.eval_select.final_eval=False`
  - outcome: pass; wrote `outputs/eval_select_smoke/ppo_teacher/stage1_nn/best_eval.pth`, `best_deploy.pth`, and `eval_select/train_eval_history.tsv`.
- PAdapt student smoke:
  - First attempt failed because `train.ppo.proprio_adapt=False` from the teacher YAML left no trainable `adapt_tconv`; rerun with `train.ppo.proprio_adapt=True`.
  - Manual stop after sufficient smoke output; wrote `model_best_train.ckpt`, `model_best_eval.ckpt`, `model_best.ckpt`, and `model_best_deploy.ckpt`.
- Deploy eval script smoke:
  - Dry-run initially exposed a bad all-failed selection edge case; fixed.
  - Real one-condition smoke succeeded and parsed `EvalSummary`:
    - `avg_reward=-1.050713`, `avg_done_rate=0.000000`
    - copied deploy best for the smoke checkpoint.

### Local conclusion
- The requested two-stage model selection is now wired:
  - in-training eval-best for teacher/student;
  - post-training candidate eval grid for final deploy-best.
- Existing train reward best behavior remains available for diagnostics, but `model_best.ckpt` becomes eval-selected when eval-select is enabled for student algorithms.

### Remaining blocked/risky
- In-training eval currently evaluates the active task/env only. Multi-condition `clean/light/hard` comparison is done by the post-training script, not inside the training process.
- Student smoke used very short 4-step eval and was manually stopped; it validates wiring, not policy quality.
- Action-chunk eval resets its internal windows after eval via the shared state reset, but a longer action-chunk smoke is still advisable before a large run.

### Single recommended next step
- For the next cloud PPO/student run, enable `train.ppo.eval_select.enabled=True` with a practical interval such as `20M` agent steps and run `scripts/eval_select_checkpoints.py` at the end over `best_train/best_eval/last` candidates to select `best_deploy`.

---
## v2-212 (2026-05-04) -- Eval-Select Mechanism Reviewed

### Target milestone/subgoal
- Review the new shared eval-best selector for the latest thesis/CoDrive training path and assess whether `eval_score = avg_reward - done_penalty * avg_done_rate` is a reasonable deploy checkpoint criterion.

### What changed (files + behavior impact)
- No code changed.
- Reviewed:
  - `dexscrew/algo/eval_select.py`
  - PPO integration in `dexscrew/algo/ppo/ppo.py`
  - PAdapt integration in `dexscrew/algo/ppo/padapt.py`
  - default-disabled eval-select blocks in current CoDrive train YAMLs.

### What was verified (commands + key outcomes)
- Source inspection confirmed:
  - Eval-select saves a fresh `best_eval` checkpoint via `self.save(...)` when score improves, then copies deploy aliases.
  - PPO preserves old `best_reward_*.pth` and adds `best_eval.pth` / `best_deploy.pth` only when enabled.
  - Student eval-select preserves train-best as `model_best_train.ckpt` and makes `model_best.ckpt` eval-selected when enabled.
  - Eval rollout resets the env and returns a fresh observation that the caller reconnects to training.
  - Config default is `enabled: False`, so old runs remain behavior-preserving.

### Local conclusion
- The mechanism is architecturally sound as a checkpoint selector: it decouples deployment selection from noisy training reward and directly penalizes instability/reset frequency.
- Main caution: `avg_reward` is fixed-rollout per-step mean reward, not episode return, so `done_penalty=2000` is intentionally strong and should be calibrated from actual `eval_select/train_eval_history.tsv`.
- For thesis/co-drive deployment behavior, reward+done is a good first selector, but thumb return/slip diagnostics should still be inspected or optionally added as score penalties later.

### Remaining blocked/risky
- Tiny eval-score improvements can overwrite the best checkpoint because `min_score_improvement=0.0`.
- `min_agent_steps=0` allows very early random-policy evals; harmless but noisy.
- In-training eval uses only the active train env. Robust `clean/light/hard` deploy selection still requires the post-training `scripts/eval_select_checkpoints.py` grid.

### Single recommended next step
- Enable eval-select for the next CoDriveThesis/cloud run with `min_agent_steps` set above zero and `min_score_improvement` above zero, then compare `best_reward`, `best_eval`, and `last` using the post-training deploy eval script before choosing a student teacher.

---
## v2-213 (2026-05-04) -- CoDriveThesis Eval-Select Enabled In Train YAML

### Target milestone/subgoal
- Apply the reviewed eval-best selection settings to the latest CoDriveThesis train YAML so future PPO/student runs can produce eval-selected deployment checkpoints by default.

### What changed (files + behavior impact)
- Updated `configs/train/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveThesis.yaml`.
  - `train.ppo.eval_select.enabled: True`
  - `interval_agent_steps: 20000000`
  - `min_agent_steps: 20000000`
  - `num_steps: 512`
  - `done_penalty: 2000.0`
  - `min_score_improvement: 0.02`
  - `final_eval: True`
  - `save_deploy_best: True`
- Behavior impact:
  - Future CoDriveThesis runs now preserve reward-best checkpoints and additionally emit eval-selected `best_eval` / `best_deploy` artifacts once enabled training reaches 20M agent steps.

### What was verified (commands + key outcomes)
- Confirmed YAML fragment via:
  - `sed -n '35,60p' configs/train/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveThesis.yaml`
- Whitespace check:
  - custom Python trailing-whitespace scan
  - outcome: `yaml_whitespace_ok`
- Note:
  - Host Python does not have `yaml` installed, so PyYAML parsing was not run locally.

### Remaining blocked/risky
- This train YAML is currently untracked in git status, matching the recent locally added CoDriveThesis files.
- No Docker Hydra compose/smoke was rerun in this small settings-only update.

### Single recommended next step
- Use this CoDriveThesis train YAML in the next 30-60 minute PPO probe and inspect `eval_select/train_eval_history.tsv` plus `best_reward`, `best_eval`, and `best_deploy` before choosing the student teacher checkpoint.

---
## v2-214 (2026-05-04) -- CoDriveThesis Local Env Count Raised To 10000

### Target milestone/subgoal
- Increase local PPO throughput for the latest CoDriveThesis config by raising the default parallel IsaacGym env count above 8192 while checking GPU memory headroom.

### What changed (files + behavior impact)
- Updated `configs/task/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveThesis.yaml`.
  - Default `task.env.numEnvs` changed from `8192` to `10000`.
- Updated `configs/train/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveThesis.yaml`.
  - `train.ppo.minibatch_size` changed from `16384` to `20000`.
- Behavior impact:
  - CoDriveThesis PPO now defaults to a 120k-sample rollout batch (`10000 envs * horizon 12`) and a proportionally larger minibatch.

### What was verified (commands + key outcomes)
- Whitespace check:
  - `git diff --check -- configs/task/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveThesis.yaml configs/train/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveThesis.yaml`
  - outcome: pass.
- Local GPU before probe:
  - RTX 4080 SUPER, total about `16376 MiB`, free about `14644 MiB`.
- Short Docker PPO probe:
  - `./docker-run-isaacgym.sh timeout 360 scripts/run_with_cleanup.sh python train.py task=Dexh13HoraLightbulbSim2RealTwoFingerCoDriveThesis headless=True seed=42 train.algo=PPO wandb_activate=False train.ppo.max_agent_steps=240000 train.ppo.output_name=Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_thesis/probe_env10000_mb20000 task.env.termination.log=True`
  - outcome: completed with `max steps achieved`, no CUDA OOM or segfault.
  - produced:
    - `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_thesis/probe_env10000_mb20000/stage1_nn/best_eval.pth`
    - `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_thesis/probe_env10000_mb20000/stage1_nn/best_deploy.pth`
    - `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_thesis/probe_env10000_mb20000/eval_select/train_eval_history.tsv`
- Observed local memory during probe:
  - During env build: total GPU used about `10850 MiB`, python about `9640 MiB`.
  - During PPO update: total GPU used peaked around `13168-13195 MiB`, python about `11954 MiB`, free about `2.7-2.8 GiB`.
  - After exit: GPU returned to desktop-only use.

### Local conclusion
- `10000 envs + minibatch_size 20000` is feasible on the local RTX 4080 SUPER and does not hit the memory limit.
- It is closer to the local sweet spot than 8192 if the goal is throughput, but it is not safe to call it unlimited headroom: only about `2.8 GiB` remained during PPO update.
- Further increases should be treated as probes (`11000/22000` at most) rather than a new default, because extra browser/viewer memory or heavier student models could push the local card into OOM.

### Remaining blocked/risky
- This was a short startup/one-epoch probe, not a full 30-60 minute throughput comparison against 8192.
- First-step eval metrics are not policy-quality evidence because the run was intentionally short.

### Single recommended next step
- Use `10000/20000` for local headless PPO probes; keep `8192/16384` or explicit lower overrides for visualization, concurrent workloads, or if any PhysX allocation failure reappears.

---
## v2-215 (2026-05-04) -- DOTPG Baseline Failure-Mode Review

### Target milestone/subgoal
- Analyze why the baseline DOTPG student has repeatedly shown weak reward compared with PAdapt/diffusion students, and decide whether the issue is slow warmup or an algorithm/config limitation.

### What changed (files + behavior impact)
- No code or config files changed.
- Reviewed DOTPG implementation and historical run records:
  - `dexscrew/dotpg/dotpg.py`
  - `dexscrew/dotpg/networks.py`
  - `dexscrew/dotpg/buffer.py`
  - `scripts/dexh13_lightbulb_student_dotpg_sim2real_twofinger.sh`
  - prior handoff entries `v2-201` through `v2-208`

### What was verified (commands + key outcomes)
- Confirmed there is no separate `BaselineDOTPG` class in the active pipeline.
  - `train.algo=DOTPG` resolves to `dexscrew.dotpg.DOTPGStudent`.
- Confirmed DOTPG uses an OT/dual reward objective:
  - environment reward is used for logging/checkpoint promotion,
  - policy/Q learning uses the learned dual reward, not the task reward directly.
- Parsed local DOTPG 2h logs after excluding `gitdiff.patch` text:
  - no-BC run: runtime `Current Best max=1.76`
  - BC-assisted run: runtime `Current Best max=1426.51`, last update reached by about `0001M` agent steps, then plateaued.
- Historical CoDrive DOTPG controls from handoff:
  - `baseline48_30m`: `max_best=146.67`
  - `gpu512_45m`: `max_best=494.10`
  - extended controls were queued specifically to separate teacher-state versus student-state/adaptation bottlenecks.

### Local conclusion
- The observed DOTPG weakness should not be treated as simply "training time too short" yet.
- Evidence points to a combination of:
  - DOTPG objective mismatch/instability for the contact-rich CoDrive task,
  - student-state adapter/proprio representation bottleneck,
  - BC acting as a stabilizer but not reliably converting into robust two-finger deployment behavior.
- Blindly running DOTPG much longer has low expected value unless a diagnostic variant first escapes the low-reward regime.

### Remaining blocked/risky
- The critical discriminator is still a teacher-state DOTPG control.
- DOTPG checkpoint promotion currently follows training mean episode reward, not the newer eval-select deploy score path.
- `reuse_expert_buffer=True` can be unsafe when switching teacher/YAML/output lineage unless the output path is clean or explicitly set to false.

### Single recommended next step
- Run a focused DOTPG diagnostic matrix before any long DOTPG run: teacher-state control first, then student-state with longer adapt/BC warmup, frozen adapter, `reuse_expert_buffer=False`, and deploy eval comparison.

---
## v2-216 (2026-05-04) -- DOTPG Draft PDF Converted To Markdown

### Target milestone/subgoal
- Prepare the newly added DOTPG draft reference for local reading before mapping its theory back to the active DOTPG implementation and diagnostics.

### What changed (files + behavior impact)
- Added `thesis_reference/DOTPG-draft.md`.
  - Extracted text from `thesis_reference/DOTPG-draft.pdf` with layout preservation.
  - Replaced PDF page-break controls with Markdown horizontal rules.
  - Added a short source/conversion header.
- No training, eval, export, or algorithm code changed.

### What was verified (commands + key outcomes)
- Bootstrap context read:
  - `docs/session_handoff_v2.md`
  - `docs/stage_acceptance_summary.md`
- PDF metadata:
  - `pdfinfo thesis_reference/DOTPG-draft.pdf`
  - outcome: LaTeX/pdfTeX PDF, unencrypted, 20 pages.
- Conversion:
  - `pdftotext -layout -enc UTF-8 -eol unix thesis_reference/DOTPG-draft.pdf thesis_reference/DOTPG-draft.md`
  - `perl -0pi -e 's/\x0c/\n\n---\n\n/g' thesis_reference/DOTPG-draft.md`
  - outcome: Markdown generated with layout-preserved text.
- Spot checks:
  - `wc -l thesis_reference/DOTPG-draft.md`
  - outcome: `1251` lines.
  - `LC_ALL=C grep -n $'\f' thesis_reference/DOTPG-draft.md || true`
  - outcome: no remaining form-feed page controls.
  - `rg -n "DOT-PG|Algorithm|Theorem|Wasserstein|Kantorovich|dual|policy gradient" thesis_reference/DOTPG-draft.md`
  - outcome: core DOTPG theory and algorithm sections are searchable.

### Local conclusion
- The DOTPG draft is now available as a readable/searchable Markdown reference.
- Layout extraction preserves most equations better than plain text mode, but it is still a text extraction from PDF rather than a semantic LaTeX/MathJax conversion.

### Remaining blocked/risky
- Figures, exact equation alignment, and some superscripts/subscripts may need manual cross-checking against the original PDF before making code-level conclusions.
- The markdown has not yet been analyzed against `dexscrew/dotpg/dotpg.py`; this session only prepared the reference.

### Single recommended next step
- Read `thesis_reference/DOTPG-draft.md` sections 4.1-4.3 and compare the paper's dual/Q/policy update equations with the current `dexscrew/dotpg/dotpg.py` losses before launching the teacher-state DOTPG diagnostic.

---
## v2-217 (2026-05-04) -- DOTPG Theory-Driven Iteration 1 Launched

### Target milestone/subgoal
- Start the new DOTPG optimization goal: use the CoDrive PPO teacher `sim2real/codrive/best_reward_4159.37.pth` as the fixed baseline, compare theory-driven DOTPG variants, and run each candidate under a 1.5h wall-clock timeout on the cloud GPU.

### What changed (files + behavior impact)
- Updated `dexscrew/dotpg/dotpg.py`.
  - Added optional `policy_arch=teacher_actor`, which wraps the PPO teacher `actor_mlp + mu` as the DOTPG policy and initializes it from the teacher checkpoint.
  - Added `policy_loss_mode=q|dual|q_dual`.
  - Added `dual_state_scale`, `dual_action_scale`, `critic_state_scale`, and `critic_action_scale` to test whether the implicit OT state-action metric is dominated by high-dimensional state.
  - Added `policy_q` and `policy_dual` train logging fields.
  - Fixed a real migration bug: copied teacher policy parameters inherited `requires_grad=False` from the frozen teacher model; the wrapper now re-enables gradients for the student policy.
- Updated `train.py`.
  - Added `DEXSCREW_SKIP_GIT_DIFF=1` support to skip printing huge dirty git diffs during cloud experiment logs.
- Added `outputs/cloud_pipeline_dotpg_theory_iter1/run_dotpg_theory_iter1.sh`.
  - Records command lines, start/end timestamps, status files, GPU usage logs, and a `summary.tsv`.
  - Uses `timeout 5400` for each candidate.
  - Runs candidates sequentially after IsaacGym multi-process parallel startup proved unreliable on this task.

### What was verified (commands + key outcomes)
- Local validation:
  - `compile()` checks passed for `train.py` and `dexscrew/dotpg/dotpg.py`.
  - `bash -n outputs/cloud_pipeline_dotpg_theory_iter1/run_dotpg_theory_iter1.sh` passed.
  - `git diff --check` passed for the touched files.
- Local Docker smoke:
  - A tiny `teacher_actor + dual` DOTPG run completed teacher load, expert data collection, BC pretrain, short train loop, and clean `max_agent_steps` exit.
- Cloud setup:
  - Synced `dexscrew/dotpg/`, `dexscrew/algo/student/`, `train.py`, CoDrive artifacts, and the new pipeline script to `/root/code/dexscrew-repro`.
  - Verified cloud GPU: `NVIDIA GeForce RTX 4090 D`, about `24564 MiB` total.
  - Fixed missing remote imports by syncing the full `dexscrew/algo/student/` directory.
- Cloud launch:
  - Active tmux session: `dotpg_theory_iter1`.
  - Active run id: `theory_iter1_20260504_045152`.
  - Current first candidate: `teacher_actor_q`.
  - It successfully completed environment build, teacher checkpoint load, teacher actor initialization, 500-step adapt warmup, 2000-step expert collection, and 3000-step BC pretrain.
  - Early training status after startup: `Current Best` had reached about `513.01`.

### Local conclusion
- DOTPG does need algorithm/code adaptation before judging the thesis idea.
- The draft theory supports the direct dual policy-gradient path: minimizing W1 implies maximizing the dual potential on policy actions; the existing baseline only updated the actor through a learned Q approximation.
- The baseline was also architecturally disadvantaged versus PAdapt/diffusion because it trained a fresh small MLP policy instead of reusing the PPO teacher actor backbone.
- Multi-process cloud parallelism is not currently worthwhile for this IsaacGym DOTPG setup: two concurrent 512/768-env processes stalled during environment startup. Single-process 1024 env starts reliably and uses the cloud GPU safely.

### Remaining blocked/risky
- The 1.5h `teacher_actor_q` candidate is still running; no final checkpoint quality or eval result is available yet.
- The full four-candidate script will take about 6h sequentially if all phases hit the 5400s timeout.
- `Current Best` is still a training reward screen. The selected DOTPG checkpoint should later be evaluated with the same deploy/eval path used for diffusion and PAdapt.

### Single recommended next step
- Monitor `outputs/cloud_pipeline_dotpg_theory_iter1/theory_iter1_20260504_045152/summary.tsv`; after `teacher_actor_q` exits with status `124`, compare its max reward to the prior DOTPG baseline and decide whether to let `teacher_actor_dual` continue or stop early for a tighter second iteration.

---
## v2-218 (2026-05-04) -- DOTPG Direct-Dual Candidate Shows Early Breakthrough

### Target milestone/subgoal
- Continue the first theory-driven DOTPG cloud matrix and compare the completed `teacher_actor_q` candidate against the direct-dual candidate that follows it.

### What changed (files + behavior impact)
- No new algorithm code changes in this step.
- Synced the first candidate's small cloud status artifacts locally:
  - `outputs/cloud_pipeline_dotpg_theory_iter1/theory_iter1_20260504_045152/summary.tsv`
  - `outputs/cloud_pipeline_dotpg_theory_iter1/theory_iter1_20260504_045152/teacher_actor_q_exit_status`

### What was verified (commands + key outcomes)
- Cloud pipeline status:
  - tmux session `dotpg_theory_iter1` is still active.
  - run id remains `theory_iter1_20260504_045152`.
  - current phase advanced to `teacher_actor_dual`.
- Completed candidate:
  - `teacher_actor_q` exited with status `124`, expected from `timeout 5400`.
  - summary row:
    - `max_best=869.14`
    - `last_best=869.14`
    - `median_last_fps=7711.1`
    - `max_mem_mib=6117`
    - `errors=0`
- Event scalar inspection for `teacher_actor_q`:
  - late `episode_rewards/step` dropped negative, around `-235`.
  - late `policy_q` was high positive, around `57`.
  - late `policy_dual` stayed negative, around `-5.8`.
  - `expert_action_mse` rose to about `1.04`.
  - Interpretation: Q-only actor optimization diverged from the dual objective and drifted away from expert action behavior despite high predicted Q.
- Active candidate:
  - `teacher_actor_dual` loaded the same CoDrive PPO teacher, completed expert collection and BC pretrain, and entered training.
  - early `Current Best` already reached about `1857.65`.
- Event scalar inspection for early `teacher_actor_dual`:
  - `episode_rewards/step` around `1896`.
  - `policy_dual` positive, around `3.8-4.0`.
  - `policy_q` positive, around `4.7-4.8`.
  - `expert_action_mse` low, around `0.067`.

### Local conclusion
- This is a meaningful positive signal for the thesis-driven DOTPG modification.
- Reusing the teacher actor alone is not sufficient: `teacher_actor_q` still plateaued/collapsed.
- Directly optimizing the dual potential, which is closer to the DOTPG theorem's action-gradient logic, is currently much stronger on this CoDrive task.
- The next question is whether `teacher_actor_dual` can keep improving through the full 1.5h and whether it survives deploy-style visualization/eval.

### Remaining blocked/risky
- `teacher_actor_dual` is still in progress; final timeout result is not known.
- A high training reward does not yet prove deploy-quality behavior. The best ckpt must be synced and evaluated/visualized after the candidate completes.
- `q_dual_metric` and `control_dotpg_q` have not run yet in the sequential script.

### Single recommended next step
- Let `teacher_actor_dual` complete its `timeout 5400` run, then sync its `model_best.ckpt` and run the same quick deploy/eval or visualization path used for the existing CoDrive students before deciding the second DOTPG iteration.

---
## v2-219 (2026-05-04) -- DOTPG Direct-Dual Run Still In Progress, Eval Path Added

### Target milestone/subgoal
- Continue the DOTPG optimization goal: validate whether the theory-aligned `teacher_actor + direct dual` student can close the gap to the current CoDrive diffusion/PAdapt students.

### What changed (files + behavior impact)
- Updated `dexscrew/dotpg/dotpg.py`.
  - DOTPG `test=True` now supports the repo-standard top-level `+test_num_steps`.
  - When `+test_num_steps` is set, DOTPG prints `EvalSummary steps=... avg_reward=... avg_done_rate=...`, so `scripts/eval_select_checkpoints.py` can rank DOTPG checkpoints with the same deploy-style clean/light/hard protocol used for other students.
  - This only changes test/eval behavior; training losses and update rules are unchanged.
- Added `scripts/vis_dexh13_lightbulb_student_dotpg_codrive.sh`.
  - Visualizes CoDrive DOTPG student checkpoints under `outputs/Dexh13HoraLightbulb_student_dotpg_codrive/<cache>/student_output/dotpg_nn/model_best.ckpt`.
  - Uses `task=Dexh13HoraLightbulbSim2RealTwoFingerCoDrive`, `train.algo=DOTPG`, `policy_arch=teacher_actor`, and deterministic/no-disturbance viewer overrides.
- Synced the DOTPG eval/test changes and the new visualization script to `/root/code/dexscrew-repro` on `cloud-training`.

### What was verified (commands + key outcomes)
- Local checks:
  - `python -m py_compile dexscrew/dotpg/dotpg.py`
  - `bash -n scripts/vis_dexh13_lightbulb_student_dotpg_codrive.sh`
  - `git diff --check -- dexscrew/dotpg/dotpg.py scripts/vis_dexh13_lightbulb_student_dotpg_codrive.sh`
- Remote checks:
  - `python -m py_compile dexscrew/dotpg/dotpg.py`
  - `bash -n scripts/vis_dexh13_lightbulb_student_dotpg_codrive.sh`
  - Verified accidental root-level rsync copies were removed.
- Theory/code comparison:
  - The DOTPG draft's Theorem 3 gives `grad_theta W = -E[grad_theta pi * grad_a f]`; descending W is equivalent to increasing the dual potential on policy actions.
  - Current `policy_loss_mode=dual` implements this direct direction with `loss = -mean(f(s, pi(s)))`.
  - The completed `teacher_actor_q` candidate therefore likely failed because Q approximation/off-policy distribution drift was poor, not because the OT dual direction is invalid.
- Cloud run status at last poll:
  - Active tmux session: `dotpg_theory_iter1`.
  - Run id: `theory_iter1_20260504_045152`.
  - Active phase: `teacher_actor_dual`.
  - `teacher_actor_q`: status `124`, `max_best=869.14`.
  - `teacher_actor_dual`: still running, current `max_best=2539.57` around agent step `0006M`.
  - Cloud GPU use: about `6117 / 24564 MiB`, util around `48%`.
  - `teacher_actor_dual` started at `2026-05-03T22:21:53+00:00`; expected timeout end is about `2026-05-03T23:51:53+00:00`.

### Local conclusion
- `teacher_actor + direct dual` remains the strongest DOTPG direction so far.
- The core algorithm logic is defensible under the thesis derivation, but the Q-only version is not reliable on this dexterous contact task.
- The next quality gate must be deploy-style eval/visualization, not training reward alone.

### Remaining blocked/risky
- `teacher_actor_dual` has not finished its 1.5h timeout yet, so no final checkpoint/eval conclusion is available.
- The later sequential candidates, `teacher_actor_qdual_metric` and `control_dotpg_q`, have not started yet.
- A high train `Current Best` can still fail visually if the policy exploits reset/reward details.

### Single recommended next step
- After `teacher_actor_dual` exits with status `124`, sync its `student_output/dotpg_nn/model_best.ckpt`, run DOTPG deploy eval via `scripts/eval_select_checkpoints.py`, then visualize with `scripts/vis_dexh13_lightbulb_student_dotpg_codrive.sh`.

---
## v2-220 (2026-05-04) -- DOTPG Iter1 Eval Complete, Iter2 Dual-Only Matrix Launched

### Target milestone/subgoal
- Continue the DOTPG optimization goal by evaluating the best first-round DOTPG candidate and launching a narrower second-round matrix based on the observed failure modes.

### What changed (files + behavior impact)
- Updated `dexscrew/dotpg/dotpg.py`.
  - Fixed DOTPG fixed-step eval: when top-level `+test_num_steps` is set, DOTPG now keeps rolling until that step budget is reached instead of exiting early after the default `train.dotpg.test_num_episodes=20`.
  - This makes DOTPG compatible with `scripts/eval_select_checkpoints.py` without requiring per-command `++train.dotpg.test_num_episodes=...`.
- Added `outputs/cloud_pipeline_dotpg_theory_iter2/run_dotpg_theory_iter2.sh`.
  - Launches a second 1.5h-per-candidate cloud matrix focused only on `policy_loss_mode=dual`.
  - Candidates:
    - `dual_metric_action2`: `dual_state_scale=0.5`, `dual_action_scale=2.0`, `bc_coef=2.5`.
    - `dual_bc5`: default metric, `bc_coef=5.0`, `bc_alpha_max=20.0`.
    - `dual_metric_bc5`: action metric plus stronger BC.
  - Uses the same CoDrive teacher `sim2real/codrive/best_reward_4159.37.pth`, `timeout 5400` per candidate, tmux execution, phase/status files, summary TSV, GPU log, and exact command logging.
- Synced the updated DOTPG eval code and iter2 script to `cloud-training`.

### What was verified (commands + key outcomes)
- Local/remote code checks:
  - `python -m py_compile dexscrew/dotpg/dotpg.py`
  - `bash -n scripts/vis_dexh13_lightbulb_student_dotpg_codrive.sh`
  - `bash -n outputs/cloud_pipeline_dotpg_theory_iter2/run_dotpg_theory_iter2.sh`
  - `git diff --check -- ...`
- DOTPG eval smoke:
  - Local 2-step DOTPG test on the existing smoke checkpoint printed:
    - `EvalSummary steps=2 avg_reward=-0.520926 avg_done_rate=0.000000`
- First eval attempt uncovered a bug:
  - Four DOTPG eval logs exited with status `0` but had `nan` summary because the old episode-based exit happened at step 149 before `EvalSummary`.
  - After the fixed-step eval patch, reran the deploy eval successfully.
- Iter1 cloud matrix:
  - `teacher_actor_q`: status `124`, `max_best=869.14`.
  - `teacher_actor_dual`: status `124`, `max_best=2622.18`.
  - `teacher_actor_qdual_metric`: status `124`, `max_best=1327.30`.
  - `control_dotpg_q` was manually stopped after qdual because q-only evidence was already low-value and cloud time was better spent on iter2.
  - Phase set to `stopped_after_qdual_for_iter2`; GPU was verified free before iter2 launch.
- Synced local artifacts:
  - `outputs/Dexh13HoraLightbulb_student_dotpg_codrive/theory_iter1_20260504_045152_teacher_actor_dual/student_output/dotpg_nn/model_best.ckpt`
  - `outputs/Dexh13HoraLightbulb_student_dotpg_codrive/theory_iter1_20260504_045152_teacher_actor_dual/student_output/dotpg_nn/model_best_deploy.ckpt`
  - `outputs/Dexh13HoraLightbulb_student_dotpg_codrive/theory_iter1_20260504_045152_teacher_actor_qdual_metric/student_output/dotpg_nn/model_best.ckpt`
  - `outputs/cloud_pipeline_dotpg_theory_iter1/theory_iter1_20260504_045152/summary.tsv`
- Deploy eval for `teacher_actor_dual`:
  - Output dir: `outputs/cloud_pipeline_dotpg_theory_iter1/theory_iter1_20260504_045152/eval_256_teacher_actor_dual_v2/`
  - `train_like`: `avg_reward=2.767024`, `avg_done_rate=0.000504`.
  - `clean`: `avg_reward=2.949418`, `avg_done_rate=0.000427`.
  - `light`: `avg_reward=2.338468`, `avg_done_rate=0.000504`.
  - `hard`: `avg_reward=1.675284`, `avg_done_rate=0.000870`.
  - Mean across four conditions: `mean_reward=2.4325485`, `mean_done_rate=0.00057625`.
- Iter2 launch:
  - tmux session: `dotpg_theory_iter2`.
  - run id: `theory_iter2_20260504_092416`.
  - active phase: `dual_metric_action2`.
  - early status after normal training start: `max_best=1149.01` at about `0005M`.

### Local conclusion
- The thesis-aligned direct dual actor update is the only strong DOTPG direction so far:
  - It strongly beats Q-only (`2622.18` vs `869.14`) and q+dual metric (`2622.18` vs `1327.30`).
  - The successful deploy eval confirms the improvement is not just a logging artifact.
- However, direct-dual is still below the current CoDrive diffusion/PAdapt family in scalar quality:
  - Earlier CoDrive diffusion eval examples include consistency around `avg_reward=5.155176` and flow around `4.827848` in the recorded 1h eval summary.
  - DOTPG direct-dual clean is `2.949418`.
- Q contamination appears harmful on this task; second-round optimization should keep the actor objective dual-only and tune metric/BC/learning dynamics around it.

### Remaining blocked/risky
- Iter2 is still running; `dual_metric_action2` is early and currently below the direct-dual baseline.
- No visual inspection has been done yet for DOTPG direct-dual; scalar eval is valid but not a complete deployment judgment.
- The local eval was run while another local PPO container was occupying the GPU. This should not change rollout reward semantics, but it can affect speed and is worth noting.

### Single recommended next step
- Let `dual_metric_action2` reach its 1.5h timeout unless it clearly fails, then compare its `max_best` to `teacher_actor_dual=2622.18`; if still far below, keep iter2 moving to `dual_bc5`, which is the more likely useful variant.

---
## v2-221 (2026-05-04) -- DOTPG BC5 Selected as Current CoDrive Baseline

### Target milestone/subgoal
- Close the DOTPG optimization loop by selecting the strongest CoDrive DOTPG candidate from the theory-driven and BC-scan runs, preserving it as a baseline artifact, and recording the algorithm/code conclusion.

### What changed (files + behavior impact)
- Added `outputs/cloud_pipeline_dotpg_theory_iter3/run_dotpg_theory_iter3_bc_scan.sh`.
  - Runs `dual_bc4`, `dual_bc6`, and `dual_bc8` as 1.5h timeout candidates.
  - Keeps the same CoDrive teacher, teacher-actor policy architecture, direct dual actor loss, default dual metric, and only varies BC strength.
- Added `sim2real/codrive/dotpg_bc5/`.
  - `model_best.ckpt`
  - `model_best_deploy.ckpt`
  - `best_reward_4159.37.pth`
  - `Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.task.yaml`
  - `Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.train.yaml`
  - `eval_256_summary.tsv`
  - `iter2_train_summary.tsv`
  - `README.md`

### What was verified (commands + key outcomes)
- Iter2 final status:
  - `dual_metric_action2`: status `124`, `max_best=1386.54`.
  - `dual_bc5`: status `124`, `max_best=2904.14`.
  - `dual_metric_bc5` was manually stopped early after metric-only failed and bc5 succeeded; cloud time was moved to BC scan.
- Iter2 `dual_bc5` deploy eval:
  - `train_like`: `avg_reward=4.420655`, `avg_done_rate=0.000092`.
  - `clean`: `avg_reward=4.496476`, `avg_done_rate=0.000122`.
  - `light`: `avg_reward=3.824169`, `avg_done_rate=0.000198`.
  - `hard`: `avg_reward=2.960864`, `avg_done_rate=0.000687`.
- Iter3 BC scan:
  - `dual_bc4`: status `124`, `max_best=2916.54`.
  - `dual_bc6`: status `124`, `max_best=2809.14`.
  - `dual_bc8`: status `124`, `max_best=3004.40`.
- Iter3 deploy eval checks:
  - `dual_bc4` looked strong by train reward but failed deploy eval:
    - clean `avg_reward=0.240152`, hard `avg_reward=0.287173`.
  - `dual_bc6` also failed deploy eval:
    - clean `avg_reward=2.293629`, hard `avg_reward=0.843302`.
  - `dual_bc8` had highest train reward but worse deploy eval than bc5:
    - clean `avg_reward=2.754137`, hard `avg_reward=1.348749`, higher done rates.
- Final cloud status:
  - `outputs/cloud_pipeline_dotpg_theory_iter3/theory_iter3_bcscan_20260504_123124/phase.txt`: `done`.
  - Cloud GPU idle: about `1 / 24564 MiB`, utilization `0%`.
- Packaging checks:
  - `find sim2real/codrive/dotpg_bc5 -maxdepth 1 -type f -printf '%f %s bytes\n'`
  - `git diff --check -- sim2real/codrive/dotpg_bc5/README.md ...`

### Local conclusion
- The current best DOTPG baseline is `dual_bc5`.
- The key algorithm conclusion is now concrete:
  - Direct dual actor loss is necessary; Q-only and q+dual are worse.
  - Reusing the PPO teacher actor architecture is necessary for a fair dexterous-hand baseline.
  - Stronger BC anchoring is necessary; it prevents the DOTPG actor from drifting off the teacher manifold while the dual objective provides improvement pressure.
  - Train reward alone is not reliable for DOTPG model selection: bc4 and bc8 had higher or similar train peaks but worse deploy eval.
- Current DOTPG deploy eval is now in the same general range as the CoDrive diffusion baselines:
  - DOTPG bc5 clean `4.496476`.
  - Recorded CoDrive flow/consistency 1h eval references are around `4.827848` and `5.155176`.

### Remaining blocked/risky
- No human visual inspection of `dual_bc5` has been performed in this session.
- The selected DOTPG baseline has not been exported/JIT-compiled; only checkpoint-level eval and packaging were completed.
- If future work needs a paper-grade ablation table, rerun deploy eval across multiple seeds, not just seed 42.

### Single recommended next step
- Use `sim2real/codrive/dotpg_bc5/model_best_deploy.ckpt` as the current DOTPG baseline for visualization/export; if continuing algorithm work, run multi-seed eval for `dual_bc5` before more hyperparameter search.

---
## v2-221 (2026-05-04) -- CoDriveThesis PPO 8h Baseline Run Complete

### Target milestone/subgoal
- Produce a stable final PPO teacher baseline for `Dexh13HoraLightbulbSim2RealTwoFingerCoDriveThesis`, using the new eval-select wiring and a more aggressive local env count.

### What changed (files + behavior impact)
- Updated `configs/task/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveThesis.yaml`.
  - Set `env.numEnvs=10000` for the main CoDriveThesis PPO baseline run.
- Updated `configs/train/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveThesis.yaml`.
  - Set `minibatch_size=20000`.
  - Enabled `eval_select` with 20M-step intervals, 512-step rollout, `done_penalty=2000.0`, and deploy-best saving.
- Added `outputs/local_pipeline_codrive_thesis_ppo8h/run_ppo8h.sh`.
  - Records status, command, timestamps, and training log under `outputs/local_pipeline_codrive_thesis_ppo8h/<cache>/`.
- Added `outputs/local_pipeline_codrive_thesis_ppo8h/codrive_thesis_env10000_eval_s42_8h_final/final_selection.md`.
  - Records the recommended final PPO baseline and why `best_eval.pth` should not be used as final.

### What was verified (commands + key outcomes)
- 8h PPO command:
  - `CACHE=codrive_thesis_env10000_eval_s42_8h_final outputs/local_pipeline_codrive_thesis_ppo8h/run_ppo8h.sh`
  - Start: `2026-05-04T04:18:07+08:00`.
  - End: `2026-05-04T12:18:11+08:00`.
  - Exit status: `124`, expected because `timeout 28800` reached the requested wall-clock limit.
- Resource health:
  - 10000 envs + minibatch 20000 allocated and ran stably.
  - Runtime GPU memory was about `12.6-12.8 GiB` used, no OOM/segfault observed.
  - After completion, GPU dropped to about `534 MiB` and no residual `python train.py` process remained.
- Checkpoints:
  - Highest reward checkpoint:
    - `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_thesis/codrive_thesis_env10000_eval_s42_8h_final/stage1_nn/best_reward_3655.17.pth`
  - `best_eval.pth` and `best_deploy.pth` were saved only at the first eval point and were not overwritten:
    - timestamp `2026-05-04 04:35:06`
    - corresponding train reward `1703.26`
- Eval-select history:
  - Best eval score stayed at the first eval row:
    - steps `20040000`, `avg_reward=4.1919`, `done_rate=0.000244`, `score=3.7040`.
  - Highest train reward row:
    - steps `501000000`, `train_reward=3655.17`, `avg_reward=5.3087`, `done_rate=0.001399`, `score=2.5099`.
    - `no_contact_frac=0`, `screw_limit_frac=0`, `thumb_slip/score=0.00677`, `screw/angular_velocity=1.14231`.
  - Highest eval reward row:
    - steps `561120000`, `train_reward=3645.84`, `avg_reward=5.3330`, `done_rate=0.001108`, `score=3.1166`.

### Local conclusion
- The 8h PPO run succeeded and produced a stronger teacher than earlier short runs.
- The recommended stable PPO baseline is:
  - `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_thesis/codrive_thesis_env10000_eval_s42_8h_final/stage1_nn/best_reward_3655.17.pth`
- The current eval-select implementation ran correctly, but its score is not a reasonable final selector for this task because `done_penalty=2000` penalizes all done events, including `term/max_eps_frac` / `time_outs`.
- Therefore `best_eval.pth` / `best_deploy.pth` from this run should not be treated as the final PPO baseline without a corrected or post-hoc selector.

### Remaining blocked/risky
- No visual inspection has been done yet for `best_reward_3655.17.pth`.
- The eval selector should be refined before relying on `best_deploy.pth`, ideally by excluding normal timeouts / max-episode completions from the done penalty or by penalizing only failure reset fractions.

### Single recommended next step
- Visualize `best_reward_3655.17.pth`; if behavior is acceptable, freeze it as the PPO teacher baseline for subsequent student distillation. Separately patch eval-select so deploy selection penalizes failure resets rather than normal timeout endings.

---
## v2-222 (2026-05-04) -- CoDriveThesis PPO Visualizer Added

### Target milestone/subgoal
- Make the new 8h CoDriveThesis PPO teacher baseline easy to visualize without hand-writing a long Hydra command.

### What changed (files + behavior impact)
- Added `scripts/vis_dexh13_lightbulb_teacher_codrive_thesis.sh`.
  - Uses `task=Dexh13HoraLightbulbSim2RealTwoFingerCoDriveThesis`.
  - Loads `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_thesis/<cache>/stage1_nn/best_reward_*.pth`.
  - Runs viewer mode with `headless=False`, `test=True`, `task.env.numEnvs=1`, and deterministic/no-disturbance visualization overrides.

### What was verified (commands + key outcomes)
- `bash -n scripts/vis_dexh13_lightbulb_teacher_codrive_thesis.sh`
  - Outcome: pass.
- Confirmed current 8h cache has one reward checkpoint:
  - `best_reward_3655.17.pth`

### Local conclusion
- The 8h PPO baseline can now be visualized with one script command.

### Remaining blocked/risky
- Visual behavior has not yet been inspected by the user.

### Single recommended next step
- Run the new visualizer on `codrive_thesis_env10000_eval_s42_8h_final` and inspect whether two-finger cooperation / thumb return behavior is acceptable.

---
## v2-223 (2026-05-04) -- CoDriveThesis Teacher Freeze And 6x3h Student Distillation Started

### Target milestone/subgoal
- Freeze the current CoDriveThesis PPO teacher as the thesis comparison baseline, then run six student distillation baselines for 3h each under the same task/checkpoint settings.

### What changed (files + behavior impact)
- Added `sim2real/codrive_thesis/`.
  - Contains the frozen CoDriveThesis task YAML, train YAML, teacher checkpoint `best_reward_3655.17.pth`, selection note, README, and SHA256 manifest.
  - This directory is the local sim2real teacher baseline package for final student comparisons.
- Added `outputs/local_pipeline_codrive_thesis_students_6x3h/run_students_6x3h.sh`.
  - Runs six sequential 3h student distillation phases: PAdapt, diffusion latent, consistency latent, flow matching latent, diffusion action chunk, and PureBC.
  - Uses the frozen teacher checkpoint `sim2real/codrive_thesis/best_reward_3655.17.pth`.
  - Records per-phase logs, status, command lines, GPU samples, and summary under `outputs/local_pipeline_codrive_thesis_students_6x3h/<run_tag>/`.

### What was verified (commands + key outcomes)
- `bash -n outputs/local_pipeline_codrive_thesis_students_6x3h/run_students_6x3h.sh`
  - Outcome: pass.
- `git diff --check -- outputs/local_pipeline_codrive_thesis_students_6x3h/run_students_6x3h.sh sim2real/codrive_thesis/README.md`
  - Outcome: pass.
- Launched:
  - `RUN_TAG=codrive_thesis_teacher3655_s42_3h_final_20260504 outputs/local_pipeline_codrive_thesis_students_6x3h/run_students_6x3h.sh`
- Initial `padapt` health:
  - Started at `2026-05-04T13:35:46+08:00`.
  - Loaded the frozen teacher checkpoint successfully.
  - GPU around `3.0 GiB / 16.0 GiB`, utilization around `70%+`.
  - `stage2_nn/model_best_train.ckpt` was written and the early current best rose above `1900`, so the phase is not stuck at startup.

### Local conclusion
- The frozen teacher package and the six-algorithm comparison pipeline are in place.
- The first student phase (`padapt`) is actively running and healthy.

### Remaining blocked/risky
- The full comparison is still in progress. Each phase should end with timeout status `124` if it consumes the requested 3h window.
- Need to verify checkpoint existence and logs after every phase before treating the comparison set as complete.

### Single recommended next step
- Continue supervising the 6x3h pipeline until all six student algorithms finish, then summarize checkpoints and visualize/evaluate the resulting students.

---
## v2-224 (2026-05-04) -- DOTPG CoDrive Goal Completion Audit

### Target milestone/subgoal
- Audit the DOTPG optimization goal against the requested requirements:
  thesis-theory review, code migration check, multiple 1.5h cloud DOTPG iterations from the CoDrive PPO teacher, deploy-oriented evaluation, and preservation of the strongest DOTPG baseline.

### What changed (files + behavior impact)
- No new source change in this audit step.
- Confirmed the implemented DOTPG changes remain present:
  - `dexscrew/dotpg/dotpg.py`
    - teacher-actor policy architecture option.
    - direct dual actor objective via `policy_loss_mode=dual`.
    - BC anchoring knobs including `bc_coef` / `bc_alpha_max`.
    - input metric scaling knobs for dual/critic.
    - fixed-step DOTPG eval with `EvalSummary`.
  - `train.py`
    - `DEXSCREW_SKIP_GIT_DIFF=1` to prevent cloud logs from being dominated by large dirty diffs.
  - `scripts/vis_dexh13_lightbulb_student_dotpg_codrive.sh`
    - viewer entrypoint for CoDrive DOTPG student checkpoints.
  - `sim2real/codrive/dotpg_bc5/`
    - packaged current best DOTPG baseline.

### What was verified (commands + key outcomes)
- Bootstrap docs reread:
  - `docs/session_handoff_v2.md`
  - `docs/stage_acceptance_summary.md`
- Syntax/static checks:
  - `python -m py_compile dexscrew/dotpg/dotpg.py train.py`
  - `bash -n scripts/vis_dexh13_lightbulb_student_dotpg_codrive.sh outputs/cloud_pipeline_dotpg_theory_iter1/run_dotpg_theory_iter1.sh outputs/cloud_pipeline_dotpg_theory_iter2/run_dotpg_theory_iter2.sh outputs/cloud_pipeline_dotpg_theory_iter3/run_dotpg_theory_iter3_bc_scan.sh`
  - `git diff --check -- dexscrew/dotpg/dotpg.py train.py scripts/vis_dexh13_lightbulb_student_dotpg_codrive.sh outputs/cloud_pipeline_dotpg_theory_iter1/run_dotpg_theory_iter1.sh outputs/cloud_pipeline_dotpg_theory_iter2/run_dotpg_theory_iter2.sh outputs/cloud_pipeline_dotpg_theory_iter3/run_dotpg_theory_iter3_bc_scan.sh sim2real/codrive/dotpg_bc5/README.md`
  - Outcome: pass.
- Thesis/code alignment checked:
  - `thesis_reference/DOTPG-draft.md` Theorem 3 gives the deterministic OT policy gradient through the Kantorovich dual potential.
  - The implemented `policy_loss_mode=dual` uses `loss=-mean(f(s, pi(s)))`, which is the direct practical form of descending the Wasserstein objective under that theorem.
  - The original Q-only route is still available, but experiments show it is not the right deploy route for this dexterous contact task.
- Cloud status:
  - `iter1_phase=stopped_after_qdual_for_iter2`
  - `iter2_phase=stopped_metric_bc5_for_bc_scan`
  - `iter3_phase=done`
  - GPU idle: `NVIDIA GeForce RTX 4090 D, 1 / 24564 MiB, 0% util`.
- Iter1 1.5h cloud results:
  - `teacher_actor_q`: status `124`, max best `869.14`.
  - `teacher_actor_dual`: status `124`, max best `2622.18`.
  - `teacher_actor_qdual_metric`: status `124`, max best `1327.30`.
- Iter2 1.5h cloud results:
  - `dual_metric_action2`: status `124`, max best `1386.54`.
  - `dual_bc5`: status `124`, max best `2904.14`.
- Iter3 1.5h cloud BC scan:
  - `dual_bc4`: status `124`, max best `2916.54`.
  - `dual_bc6`: status `124`, max best `2809.14`.
  - `dual_bc8`: status `124`, max best `3004.40`.
- Deploy eval for selected `dual_bc5`:
  - `train_like`: avg reward `4.420655`, done `0.000092`.
  - `clean`: avg reward `4.496476`, done `0.000122`.
  - `light`: avg reward `3.824169`, done `0.000198`.
  - `hard`: avg reward `2.960864`, done `0.000687`.
- Packaging verified:
  - `sim2real/codrive/dotpg_bc5/model_best.ckpt`
  - `sim2real/codrive/dotpg_bc5/model_best_deploy.ckpt`
  - `sim2real/codrive/dotpg_bc5/best_reward_4159.37.pth`
  - frozen task/train YAMLs, eval TSV, train summary TSV, and README.

### Local conclusion
- The code migration issue was real:
  - Reusing the PPO teacher actor architecture was required for a fair high-DOF dexterous-hand baseline.
  - The copied actor parameters must be trainable; the implementation now explicitly re-enables gradients in `TeacherActorPolicy`.
- The algorithmic issue is also clear:
  - The thesis dual-gradient logic is coherent for this task.
  - The Q-only DOTPG actor objective is not sufficiently reliable here; direct dual update is the correct primary objective.
  - BC anchoring is necessary to keep the student close to the expert manifold while the dual objective supplies improvement pressure.
  - Train reward alone is not a good DOTPG selector: `bc4` and `bc8` looked competitive by train reward but failed or weakened in deploy eval.
- Current selected DOTPG baseline:
  - `sim2real/codrive/dotpg_bc5/model_best_deploy.ckpt`
  - It is now in the same general deploy-eval range as the recorded CoDrive diffusion baselines, though not yet proven across seeds.

### Remaining blocked/risky
- No human visual inspection has been done for `dotpg_bc5`.
- No JIT/export package has been produced for deployment.
- Current deploy eval is seed-42 only; paper-grade claims need multi-seed eval.

### Single recommended next step
- Visualize `sim2real/codrive/dotpg_bc5/model_best_deploy.ckpt`; if behavior is acceptable, run a multi-seed eval pack before writing DOTPG as a serious baseline result.

---
## v2-225 (2026-05-05) -- DOTPG BC5 Synced And Viewer Started

### Target milestone/subgoal
- Sync the selected CoDrive DOTPG `dual_bc5` checkpoint locally and open a headed visualization for user inspection.

### What changed (files + behavior impact)
- Refreshed local DOTPG package checkpoint files from the cloud source run:
  - `sim2real/codrive/dotpg_bc5/model_best.ckpt`
  - `sim2real/codrive/dotpg_bc5/model_best_deploy.ckpt`
- Both files now point to the same synced `dual_bc5` checkpoint content.
- No source code/config behavior change.

### What was verified (commands + key outcomes)
- Bootstrap context read:
  - `docs/session_handoff_v2.md`
  - `docs/stage_acceptance_summary.md`
- Cloud source located:
  - `cloud-training:/root/code/dexscrew-repro/outputs/Dexh13HoraLightbulb_student_dotpg_codrive/theory_iter2_20260504_092416_dual_bc5/student_output/dotpg_nn/model_best.ckpt`
- Sync command completed:
  - `rsync -av -e ssh cloud-training:/root/code/dexscrew-repro/outputs/Dexh13HoraLightbulb_student_dotpg_codrive/theory_iter2_20260504_092416_dual_bc5/student_output/dotpg_nn/model_best.ckpt ...`
- Checkpoint hash after sync:
  - `03ca598e9f384887ad09edde882737b36d55381b44da766e453486bb5ffd4082`
  - hash matched for `model_best.ckpt` and `model_best_deploy.ckpt`.
- Local viewer command launched:
  - `./docker-run-isaacgym.sh python train.py task=Dexh13HoraLightbulbSim2RealTwoFingerCoDrive ... train.algo=DOTPG ... checkpoint=sim2real/codrive/dotpg_bc5/model_best_deploy.ckpt`
- Viewer reached rollout:
  - IsaacGym built the environment.
  - DOTPG checkpoint restored.
  - Test loop printed live `[DOTPG][TEST] step=... reward(mean)=... done=0` lines.

### Local conclusion
- The selected DOTPG BC5 checkpoint is synchronized locally and actively visualizing.
- A separate local CoDriveThesis PureBC student phase was still running during launch, using about 3.1GB of the 16GB local GPU; the DOTPG viewer still started successfully.

### Remaining blocked/risky
- Human behavior inspection is pending.
- If viewer is sluggish, stop the concurrent PureBC training or rerun visualization after it exits.

### Single recommended next step
- Inspect the open DOTPG BC5 viewer behavior and decide whether to run multi-seed eval or compare visually against flow/consistency students.

---
## v2-224 (2026-05-04) -- CoDriveThesis 6x3h Student Distillation Mid-Run Status

### Target milestone/subgoal
- Continue the final thesis student baseline comparison from the frozen CoDriveThesis PPO teacher.

### What changed (files + behavior impact)
- No additional code/config changes in this status update.
- Active pipeline remains:
  - `outputs/local_pipeline_codrive_thesis_students_6x3h/run_students_6x3h.sh`
  - run tag `codrive_thesis_teacher3655_s42_3h_final_20260504`
  - teacher `sim2real/codrive_thesis/best_reward_3655.17.pth`

### What was verified (commands + key outcomes)
- `padapt`
  - status `exit_124`, expected 3h timeout.
  - best checkpoint `outputs/Dexh13HoraLightbulb_student_padapt_codrive_thesis/codrive_thesis_teacher3655_s42_3h_final_20260504/stage2_nn/model_best_train.ckpt`
  - observed best reward about `3448.94`.
- `diffusion_latent`
  - status `exit_124`, expected 3h timeout.
  - best checkpoint `outputs/Dexh13HoraLightbulb_student_diffusion_latent_codrive_thesis/codrive_thesis_teacher3655_s42_3h_final_20260504/stage2_diffusion_nn/model_best_train.ckpt`
  - observed best reward about `4024.44`; high early peak, no later refresh.
- `consistency_latent`
  - status `exit_124`, expected 3h timeout.
  - best checkpoint `outputs/Dexh13HoraLightbulb_student_consistency_codrive_thesis/codrive_thesis_teacher3655_s42_3h_final_20260504/stage2_consistency_nn/model_best_train.ckpt`
  - observed best reward about `3890.29`.
- `flow_matching_latent`
  - currently running and healthy.
  - best checkpoint already exists at `outputs/Dexh13HoraLightbulb_student_flow_matching_codrive_thesis/codrive_thesis_teacher3655_s42_3h_final_20260504/stage2_flow_nn/model_best_train.ckpt`
  - observed best reward about `3871.48` at the last status check.
- Runtime health:
  - Each active student phase used about `3.0 GiB / 16 GiB` GPU memory.
  - No OOM, segfault, or lingering old viewer/train process observed.

### Local conclusion
- The six-algorithm comparison pipeline is executing correctly.
- The first three completed algorithms produced usable checkpoints.
- `flow_matching_latent` is currently healthy and on track.

### Remaining blocked/risky
- Three phases remain incomplete: `flow_matching_latent`, `diffusion_action_chunk`, and `purebc`.
- Reward alone is not enough for thesis conclusion; high-scoring diffusion/consistency/flow checkpoints still need visual/eval confirmation.

### Single recommended next step
- Continue supervising the pipeline until all six phases finish, then summarize final checkpoints and run visual/eval checks on the strongest/highest-risk students.

---
## v2-226 (2026-05-05) -- DOTPG CoDrive Theory Optimization Note Added

### Target milestone/subgoal
- Document why the CoDrive DOTPG baseline improved, with emphasis on algorithmic theory, previous failure causes, implementation changes, and the evidence behind the selected `dual_bc5` checkpoint.

### What changed (files + behavior impact)
- Added `docs/dotpg_codrive_optimization.md`.
  - Summarizes the optimal-transport derivation used by DOTPG.
  - Explains why `policy_loss_mode=dual` follows the deterministic OT policy-gradient theorem.
  - Records why earlier DOTPG was weak: actor parameterization mismatch, copied actor trainability bug, and noisy Q-only actor objective.
  - Records the current fixes: teacher actor policy architecture, teacher initialization, direct dual actor loss, BC anchoring, fixed-step eval, and cloud log control.
  - Records iter1/iter2/iter3 evidence and distillation-time interpretation.
- Updated `sim2real/codrive/dotpg_bc5/README.md`.
  - Added a short optimization summary and pointer to `docs/dotpg_codrive_optimization.md`.

### What was verified (commands + key outcomes)
- Read current bootstrap context:
  - `docs/session_handoff_v2.md`
  - `docs/stage_acceptance_summary.md`
- Checked written docs:
  - `sed -n '1,260p' docs/dotpg_codrive_optimization.md`
  - `tail -n 140 docs/dotpg_codrive_optimization.md`
  - `sed -n '1,180p' sim2real/codrive/dotpg_bc5/README.md`
- Formatting check:
  - `git diff --check -- docs/dotpg_codrive_optimization.md sim2real/codrive/dotpg_bc5/README.md`
  - Outcome: pass.

### Local conclusion
- The DOTPG optimization is now documented as a medium-size algorithm implementation correction, not a small hyperparameter tweak.
- The theoretical basis is explicitly tied to the Kantorovich dual potential and deterministic OT policy gradient.
- The selected result remains `sim2real/codrive/dotpg_bc5/model_best_deploy.ckpt`, with the caveat that multi-seed eval is still needed for paper-grade claims.

### Remaining blocked/risky
- The note is a technical summary, not yet a polished thesis subsection.
- It does not add new experiments; it records the current evidence.

### Single recommended next step
- If DOTPG is promoted from baseline to paper contribution, convert `docs/dotpg_codrive_optimization.md` into a thesis subsection and run multi-seed eval for `dual_bc5`.

---
## v2-227 (2026-05-05) -- CoDriveThesis 6x3h Student Distillation Completed

### Target milestone/subgoal
- Finish the final thesis student-baseline distillation comparison from the frozen CoDriveThesis PPO teacher.

### What changed (files + behavior impact)
- Created the frozen teacher package under `sim2real/codrive_thesis/`.
  - Includes task/train YAML snapshots, teacher checkpoint `best_reward_3655.17.pth`, teacher selection note, README, and SHA256 sums.
- Ran the six-student sequential pipeline:
  - `outputs/local_pipeline_codrive_thesis_students_6x3h/run_students_6x3h.sh`
  - run tag `codrive_thesis_teacher3655_s42_3h_final_20260504`
  - teacher `sim2real/codrive_thesis/best_reward_3655.17.pth`
- Added corrected strict result table:
  - `outputs/local_pipeline_codrive_thesis_students_6x3h/codrive_thesis_teacher3655_s42_3h_final_20260504/strict_summary.tsv`
  - This table should be used instead of the raw pipeline `summary.tsv`, because startup gitdiff text polluted the script's loose `best_from_log` parser.

### What was verified (commands + key outcomes)
- Phase completion:
  - `padapt`: `exit_124`, expected 3h timeout.
  - `diffusion_latent`: `exit_124`, expected 3h timeout.
  - `consistency_latent`: `exit_124`, expected 3h timeout.
  - `flow_matching_latent`: `exit_124`, expected 3h timeout.
  - `diffusion_action_chunk`: `exit_124`, expected 3h timeout.
  - `purebc`: `exit_124`, expected 3h timeout.
- Strict reward parsing from lines beginning with `Agent Steps:`:
  - `diffusion_latent`: `4024.44`
  - `flow_matching_latent`: `3905.74`
  - `consistency_latent`: `3890.47`
  - `purebc`: `3455.90`
  - `padapt`: `3451.98`
  - `diffusion_action_chunk`: `2849.68`
- Primary checkpoints:
  - `outputs/Dexh13HoraLightbulb_student_padapt_codrive_thesis/codrive_thesis_teacher3655_s42_3h_final_20260504/stage2_nn/model_best_train.ckpt`
  - `outputs/Dexh13HoraLightbulb_student_diffusion_latent_codrive_thesis/codrive_thesis_teacher3655_s42_3h_final_20260504/stage2_diffusion_nn/model_best_train.ckpt`
  - `outputs/Dexh13HoraLightbulb_student_consistency_codrive_thesis/codrive_thesis_teacher3655_s42_3h_final_20260504/stage2_consistency_nn/model_best_train.ckpt`
  - `outputs/Dexh13HoraLightbulb_student_flow_matching_codrive_thesis/codrive_thesis_teacher3655_s42_3h_final_20260504/stage2_flow_nn/model_best_train.ckpt`
  - `outputs/Dexh13HoraLightbulb_student_diffusion_action_chunk_codrive_thesis/codrive_thesis_teacher3655_s42_3h_final_20260504/stage2_diffusion_action_chunk_nn/model_best_train.ckpt`
  - `outputs/Dexh13HoraLightbulb_student_purebc_codrive_thesis/codrive_thesis_teacher3655_s42_3h_final_20260504/stage2_bc_nn/model_best.ckpt`
- Runtime health:
  - GPU returned to idle after completion, about `444 MiB / 16376 MiB`.
  - `pgrep -af 'python train.py' | grep -v grep` returned no residual training process.

### Local conclusion
- The six requested 3h student distillation runs completed successfully from the same frozen CoDriveThesis PPO teacher and task configuration.
- The strict training-reward ranking is:
  - diffusion latent > flow matching latent > consistency latent > purebc ~= PAdapt > diffusion action chunk.
- `purebc` was stronger than expected and nearly tied PAdapt after 3h, which makes it a useful low-complexity baseline for the thesis comparison.
- `diffusion_action_chunk` improved its student-side metrics but did not convert that into high closed-loop reward in 3h.

### Remaining blocked/risky
- Raw `summary.tsv` in the run directory contains polluted `5001.07` values and must not be used for reporting.
- Training reward alone is not enough for final thesis claims; the top candidates still need identical visual/eval checks.
- Multi-seed eval remains needed for paper-grade conclusions.

### Single recommended next step
- Visualize and fixed-rollout-evaluate the top candidates in this order: `diffusion_latent`, `flow_matching_latent`, `consistency_latent`, then compare against `purebc` and `padapt`.

---
## v2-228 (2026-05-05) -- CoDriveThesis Rising Students Continued 2h To Plateau

### Target milestone/subgoal
- Determine whether the six 3h student-distillation baselines had reached their reward limits, and continue only the algorithms with clear late-run upward trends.

### What changed (files + behavior impact)
- Added and ran:
  - `outputs/local_pipeline_codrive_thesis_students_continue2h/run_continue_rising_students_2h.sh`
- The continuation script resumes from the 3h student checkpoints, not from the PPO teacher:
  - PAdapt from `stage2_nn/model_best_train.ckpt`
  - FlowMatching latent from `stage2_flow_nn/model_best_train.ckpt`
  - PureBC from `stage2_bc_nn/model_best.ckpt`
- Added result note:
  - `outputs/local_pipeline_codrive_thesis_students_continue2h/codrive_thesis_continue2h_rising_s42_20260505/plateau_summary.md`

### What was verified (commands + key outcomes)
- Pre-run trend check from first 3h logs:
  - Continue: `padapt`, `flow_matching_latent`, `purebc`.
  - Do not continue: `diffusion_latent`, `consistency_latent`, `diffusion_action_chunk`.
- Continuation phase status:
  - `padapt`: `exit_124`, expected 2h timeout.
  - `flow_matching_latent`: `exit_124`, expected 2h timeout.
  - `purebc`: `exit_124`, expected 2h timeout.
- Strict continuation results:
  - `padapt`: `3451.98 -> 3607.14`, gain `+155.16`.
  - `flow_matching_latent`: `3905.74 -> 3986.55`, gain `+80.81`.
  - `purebc`: `3455.90 -> 3538.86`, gain `+82.96`.
- Plateau evidence:
  - `padapt`: best reached at 3.2% of continuation; final 90% gain `0.00`.
  - `flow_matching_latent`: best reached at 3.0% of continuation; final 90% gain `0.00`.
  - `purebc`: best reached at 62.4% of continuation; final 25% gain `0.00`.
- Runtime health:
  - GPU returned to idle, about `471 MiB / 16376 MiB`.
  - No residual `python train.py` process remained.

### Local conclusion
- The first 3h comparison was not fully saturated for `padapt`, `flow_matching_latent`, or `purebc`.
- After the 2h continuation, all three have a flat tail by strict `Agent Steps:` parsing.
- No further reward-only continuation is recommended before fixed-rollout eval and visual inspection.

### Remaining blocked/risky
- Reward-only ranking is still not enough for thesis claims.
- The new best continuation checkpoints need the same visual and fixed-rollout eval protocol as the original six 3h checkpoints.

### Single recommended next step
- Fixed-rollout evaluate and visualize these final candidates: `diffusion_latent` 3h, `flow_matching_latent` continued, `consistency_latent` 3h, `padapt` continued, and `purebc` continued.

---
## v2-228 (2026-05-05) -- CoDrive BC/DAgger Baseline Optimization With Cloud Parallel Eval-Select

### Target milestone/subgoal
- Optimize the weak BC and DAgger student baselines against the current CoDrive PPO teacher:
  `sim2real/codrive/best_reward_4159.37.pth`.
- Use the 24GB cloud GPU aggressively to test multiple BC/DAgger variants in parallel, then choose checkpoints by fixed-step deploy eval rather than episodic training reward alone.

### What changed (files + behavior impact)
- Updated BC/DAgger student implementations:
  - `dexscrew/algo/student/bc_student.py`
  - `dexscrew/algo/student/dagger_student.py`
  - Both now integrate `EvalSelectMixin`, add `_eval_select_action`, and can save eval-selected aliases such as `model_best_eval.ckpt`, `model_best.ckpt`, and `model_best_deploy.ckpt`.
  - When eval-select is enabled, periodic episodic eval saves `model_best_student_eval.ckpt` without overwriting deploy-selected `model_best`.
- Updated cloud-capable launch scripts:
  - `scripts/dexh13_lightbulb_student_bc_codrive.sh`
  - `scripts/dexh13_lightbulb_student_dagger_codrive.sh`
  - Added `EVAL_SELECT_*` env knobs and Hydra `++train.ppo.eval_select.*` overrides.
- Added cloud iteration scripts:
  - `outputs/cloud_pipeline_bc_dagger_opt_iter3/run_bc_dagger_opt_iter3_evalsel.sh`
  - `outputs/cloud_pipeline_bc_dagger_opt_iter3/eval_opt_iter3_fixed.sh`
- Behavior impact:
  - BC/DAgger can now run closed-loop fixed-step eval during/after training and keep deploy candidates separate from episodic-reward checkpoints.
  - The cloud pipeline runs four 512-env variants concurrently to use most of the 24GB GPU instead of serial low-utilization runs.

### What was verified (commands + key outcomes)
- Bootstrap context read:
  - `docs/session_handoff_v2.md`
  - `docs/stage_acceptance_summary.md`
- Static checks:
  - `python -m py_compile dexscrew/algo/student/bc_student.py dexscrew/algo/student/dagger_student.py`
  - `bash -n scripts/dexh13_lightbulb_student_bc_codrive.sh scripts/dexh13_lightbulb_student_dagger_codrive.sh`
  - `bash -n outputs/cloud_pipeline_bc_dagger_opt_iter3/run_bc_dagger_opt_iter3_evalsel.sh`
  - `bash -n outputs/cloud_pipeline_bc_dagger_opt_iter3/eval_opt_iter3_fixed.sh`
  - `git diff --check -- ...`
  - Outcome: pass.
- Remote sync and remote static checks:
  - Synced BC/DAgger code, `dexscrew/algo/eval_select.py`, launch scripts, and opt-iter3 pipeline to `cloud-training:/root/code/dexscrew-repro`.
  - Remote `py_compile` and `bash -n` passed.
- Cloud training run:
  - Pipeline:
    `outputs/cloud_pipeline_bc_dagger_opt_iter3/bc_dagger_opt_iter3_evalsel_s42_20260505_002232/`
  - Four concurrent variants:
    - `bc_latent_evalsel`
    - `dagger_blend_evalsel`
    - `dagger_pure_recent`
    - `dagger_pure_replay`
  - Each phase used its own `timeout 3600`.
  - Exit statuses:
    - `bc_latent_evalsel_exit_status=124`
    - `dagger_blend_evalsel_exit_status=124`
    - `dagger_pure_recent_exit_status=124`
    - `dagger_pure_replay_exit_status=124`
  - `124` is expected wall-clock timeout.
  - Runtime GPU use was about `21.3GB / 24GB`, with utilization near `100%`.
- Independent fixed-step eval:
  - Eval pipeline:
    `outputs/cloud_pipeline_bc_dagger_opt_iter3/bc_dagger_opt_iter3_evalsel_fixed_eval_s42_20260505_012555/`
  - Protocol:
    - task `Dexh13HoraLightbulbSim2RealTwoFingerCoDrive`
    - seed `42`
    - `num_envs=48`
    - `steps=256`
    - termination enabled
    - obs noise `t=0.01`, `e=0.02`
  - All eight eval jobs exited `0`.

| Variant | Checkpoint | avg_reward | avg_done_rate |
|---|---|---:|---:|
| `bc_latent_evalsel` | `model_best_deploy` | `3.449824` | `0.000244` |
| `bc_latent_evalsel` | `model_last` | `4.134591` | `0.000081` |
| `dagger_blend_evalsel` | `model_best_deploy` | `3.886585` | `0.000244` |
| `dagger_blend_evalsel` | `model_last` | `4.331795` | `0.000000` |
| `dagger_pure_recent` | `model_best_deploy` | `4.181801` | `0.000000` |
| `dagger_pure_recent` | `model_last` | `4.467756` | `0.000000` |
| `dagger_pure_replay` | `model_best_deploy` | `4.281959` | `0.000081` |
| `dagger_pure_replay` | `model_last` | `4.528048` | `0.000081` |

- Synced local artifacts:
  - `outputs/cloud_pipeline_bc_dagger_opt_iter3/bc_dagger_opt_iter3_evalsel_s42_20260505_002232/`
  - `outputs/cloud_pipeline_bc_dagger_opt_iter3/bc_dagger_opt_iter3_evalsel_fixed_eval_s42_20260505_012555/`
  - `outputs/Dexh13HoraLightbulb_student_bc_codrive/codrive_bc_opt_iter3_evalsel_latent_evalsel_s42/`
  - `outputs/Dexh13HoraLightbulb_student_dagger_codrive/codrive_dagger_opt_iter3_evalsel_blend_evalsel_s42/`
  - `outputs/Dexh13HoraLightbulb_student_dagger_codrive/codrive_dagger_opt_iter3_evalsel_pure_recent_s42/`
  - `outputs/Dexh13HoraLightbulb_student_dagger_codrive/codrive_dagger_opt_iter3_evalsel_pure_replay_s42/`
  - Large replay/demo `.pt` buffers were intentionally excluded from the final sync; ckpt/config/TensorBoard/eval history files were synced.
- Remote cleanup/status:
  - Cloud GPU returned to idle: about `1 MiB / 24564 MiB`, `0%` utilization.
  - No matching opt-iter3 training process remained.

### Local conclusion
- The main previous BC/DAgger problem was not only algorithm weakness; checkpoint selection by episodic reward was a poor proxy for deploy behavior.
- `model_last` is consistently stronger than `model_best_deploy` in this 1h opt-iter3 run, so the eval-select cadence/penalty still needs refinement before it can replace final fixed-step selection.
- Current best classical baseline candidate:
  - `outputs/Dexh13HoraLightbulb_student_dagger_codrive/codrive_dagger_opt_iter3_evalsel_pure_replay_s42/dagger_nn/model_last.ckpt`
  - fixed-step result: `avg_reward=4.528048`, `avg_done_rate=0.000081`
- DAgger is now a reasonable CoDrive baseline, close to the earlier PAdapt fixed-step reference (`avg_reward` about `4.72`) and clearly better than the old ~1000-level episodic impression.
- BC also improved materially, but still trails DAgger.

### Remaining blocked/risky
- The best checkpoint is `model_last`, so it depends on the chosen stopping time. For paper-grade reporting, run either:
  - longer fixed intervals with periodic fixed-step selection, or
  - multiseed fixed-step eval over the synced `model_last` and deploy candidates.
- Eval-select currently saved deploy aliases too early under this setup; interval/done-penalty should be tuned if it is used as the automatic selector.
- Behavior still needs visual confirmation before treating DAgger pure-replay as the final classical baseline.

### Single recommended next step
- Visualize:
  `outputs/Dexh13HoraLightbulb_student_dagger_codrive/codrive_dagger_opt_iter3_evalsel_pure_replay_s42/dagger_nn/model_last.ckpt`
  using `scripts/vis_dexh13_lightbulb_student_dagger_codrive.sh`, then decide whether to lock this as the optimized DAgger baseline or run a multiseed fixed-step comparison.

---

## v2-2026-05-06 -- Cloud Handoff Added And DOTPG/Diffusion Files Synced

### Target milestone/subgoal
- Prepare the cloud machine for later Windows-side unified CoDriveThesis paper eval development without launching new eval/training in this session.

### What changed (files + behavior impact)
- Updated `AGENTS.md`.
  - Added explicit cloud handoff expectations so Ubuntu-side and Windows-side Codex sessions can coordinate through the cloud repo.
  - Added a cloud preflight rule: before cloud execution or preparing cloud commands, read cloud-side documentation as needed, at minimum `/root/code/dexscrew-repro/docs/cloud_session_handoff.md` when reachable.
- Added/updated `docs/cloud_session_handoff.md`.
  - Records live cloud state, already-completed classic baseline/eval results, known eval pitfalls, and next cloud action.
- Synced local DOTPG/diffusion-relevant files to `cloud-training:/root/code/dexscrew-repro/`.
  - Source: `dexscrew/dotpg/`, diffusion-class student files, eval-select/student wrapper files, `train.py`.
  - Scripts/configs/docs: CoDriveThesis YAMLs, cloud diffusion scripts, DOTPG/diffusion docs, `thesis_reference/DOTPG-draft.md`.
  - Selected CoDriveThesis local ckpts: PAdapt, diffusion latent, consistency latent, flow matching, diffusion action chunk, PureBC.

### What was verified (commands + key outcomes)
- Remote handoff visibility:
  - `rsync -av AGENTS.md docs/cloud_session_handoff.md cloud-training:/root/code/dexscrew-repro/`
  - `ssh cloud-training 'cd /root/code/dexscrew-repro && sed -n "1,220p" docs/cloud_session_handoff.md'`
  - Outcome: cloud can read `docs/cloud_session_handoff.md`.
- Remote artifact spot checks:
  - Verified cloud now has DOTPG/diffusion source files, CoDriveThesis YAML/package files, teacher ckpt, and selected local CoDriveThesis student ckpts.
- Existing cloud paper eval inspection:
  - `outputs/paper_eval_codrive_thesis_cloud_s42_s43_s44_20260506_092231/validation_summary.txt`
  - Outcome: `all_ok=True` for existing teacher/LatentBC/DAgger/DOTPG `2048`-step eval table.

### Remaining blocked/risky
- No new unified eval was launched because the user explicitly deferred eval to a later Windows-side cloud-development session.
- The current clean paper eval covers only teacher/LatentBC/DAgger/DOTPG; PAdapt and diffusion-class/PureBC ckpts are now synced but still need the same fixed-step eval protocol.
- One rsync command returned code `23` because a listed local DOTPG config path did not exist; follow-up remote spot checks confirmed the important source/config/docs/ckpt files are present.

### Single recommended next step
- In the next Windows-side cloud session, read `docs/cloud_session_handoff.md`, then extend the existing paper eval pipeline to add PAdapt, diffusion latent, consistency latent, flow matching, diffusion action chunk, and PureBC under the same `Dexh13HoraLightbulbSim2RealTwoFingerCoDriveThesis`, seeds `42,43,44`, `2048`-step protocol.

---

## v2-2026-05-06 -- Windows Handoff Eval Checkpoint Verification

### Target milestone/subgoal
- Confirm that Windows-side Codex can continue cloud paper-eval development without relying on Ubuntu-local checkpoint files.

### What changed (files + behavior impact)
- Updated `docs/cloud_session_handoff.md` with a final Windows handoff checkpoint inventory.
- No eval/training was launched.

### What was verified (commands + key outcomes)
- Read cloud-side handoff first:
  - `ssh cloud-training 'cd /root/code/dexscrew-repro && sed -n "1,120p" docs/cloud_session_handoff.md'`
- Cloud status:
  - Host: `di-20260428205452-zqhv8`
  - GPU: RTX 4090 D, idle at about `1 MiB / 24564 MiB`, `0%`.
- Verified all 10 next-eval checkpoints are present on cloud:
  - PPO teacher
  - LatentBC
  - DAgger
  - DOTPG
  - PAdapt
  - diffusion latent
  - consistency latent
  - flow matching
  - diffusion action chunk
  - PureBC
- Existing clean paper eval remains valid:
  - `outputs/paper_eval_codrive_thesis_cloud_s42_s43_s44_20260506_092231/validation_summary.txt`
  - `raw_rows=12`, `expected_rows=12`, `all_ok=True`

### Remaining blocked/risky
- The clean existing eval covers only teacher/LatentBC/DAgger/DOTPG.
- PAdapt/diffusion-class/PureBC still need the same unified fixed-step eval in the next cloud session.

### Single recommended next step
- On Windows Codex, SSH/Remote-SSH into the cloud repo, read `docs/cloud_session_handoff.md`, and extend the existing paper eval pipeline for the six remaining synced models under the same `2048`-step `s42,s43,s44` protocol.

---

## v2-2026-05-11 -- Cloud Paper Eval Status Readback

### Target milestone/subgoal
- Read current cloud state after Windows-side Codex ran extended CoDriveThesis paper evals, without launching new eval/training.

### What changed (files + behavior impact)
- Synced the latest cloud `docs/cloud_session_handoff.md` back to local so Ubuntu-side docs reflect Windows-side cloud work.
- No training/eval process was started or stopped.

### What was verified (commands + key outcomes)
- Read cloud handoff first:
  - `ssh cloud-training 'cd /root/code/dexscrew-repro && sed -n "1,260p" docs/cloud_session_handoff.md'`
- Cloud runtime status:
  - Cloud host: `di-20260428205452-zqhv8`
  - GPU idle: RTX 4090 D, about `1 MiB / 24564 MiB`, `0%`
  - No active `python train.py` process.
  - One stale/empty tmux session remains:
    `paper_codrive_supervisor_20260506_210723`
- Main new cloud run:
  - `outputs/paper_codrive_thesis_full_20260506_210723/`
  - `status/phase.txt`: `done`
  - `validation_summary.txt`: no bad patterns and no missing deploy ckpts.
  - Active guard: `eval_json_count=291`, `bad_log_pattern_count=0`, `alert=ok`.
  - Pipeline phases completed:
    formal training, representation training, main eval, NFE/latency, representation eval, robustness, validation.
- Main table source:
  - `outputs/paper_codrive_thesis_full_20260506_210723/aggregate_csv/main_aggregate.csv`
  - `outputs/paper_codrive_thesis_full_20260506_210723/tables/main_table.tex`

### Key Cloud Results Observed
- Main fixed-step ranking:
  - `teacher_ppo`: reward `5.479`, return `3764.160`
  - `consistency_latent`: reward `5.228`, return `3538.866`
  - `flow_matching`: reward `5.102`, return `3428.253`
  - `padapt`: reward `4.880`, return `3275.302`
  - `purebc`: reward `4.817`, return `3192.494`
  - `dagger`: reward `4.768`, return `3283.343`
  - `diffusion_latent`: reward `4.569`, return `2934.284`
  - `bc_latentbc`: reward `4.086`, return `2684.759`
  - `dotpg`: reward `3.446`, return `2330.396`
- Robustness tables exist for:
  - nominal
  - obs2x
  - obs4x
  - friction wide
  - mass/COM wide
  - initpos noise
- NFE/latency tables exist for diffusion-class methods at NFE `1,2,4,8,10`.
- Representation ablation exists for `diffusion_action_chunk` vs `diffusion_action_chunk_len1`.

### Remaining blocked/risky
- The formal training jobs were timeout-bounded and did not reach the configured `eval_select.interval_agent_steps=20000000`; manifest records a checkpoint-selection hotfix:
  `model_best_deploy.ckpt -> model_best_train.ckpt`.
  This deviation should be disclosed in experiment notes/paper methods.
- `diffusion_action_chunk` remains behaviorally poor in the main/NFE tables despite valid execution.
- `matplotlib_available=False`, so artifact generation completed but plots may be absent; tables/CSVs are present.
- The stale tmux session can be cleaned later, but it is not consuming GPU and no train/eval process is active.

### Single recommended next step
- Treat `outputs/paper_codrive_thesis_full_20260506_210723/aggregate_csv/main_aggregate.csv` and related robustness/NFE CSVs as the current cloud-side paper data source, then decide how to present `diffusion_action_chunk` and the deploy-checkpoint symlink hotfix in the paper/appendix.

---

## v2-2026-05-11 -- DOTPG Checkpoints Synced From Cloud

### Target milestone/subgoal
- Bring DOTPG-related cloud checkpoints and eval artifacts back to the Ubuntu local workspace for local visualization/inspection.

### What changed (files + behavior impact)
- Synced DOTPG checkpoint/config/eval artifacts from cloud to local:
  - `outputs/Dexh13HoraLightbulb_student_dotpg_codrive/`
  - `outputs/Dexh13HoraLightbulb_student_dotpg_codrive_thesis/`
  - selected DOTPG cloud-pipeline logs/summaries
  - DOTPG paper-eval logs/raw JSON/main aggregate snippets
- No cloud training/eval was launched.

### What was verified (commands + key outcomes)
- Read cloud handoff first:
  - `ssh cloud-training 'cd /root/code/dexscrew-repro && sed -n "1,180p" docs/cloud_session_handoff.md'`
- Cloud status:
  - GPU idle, no active `python train.py`.
- Local main CoDriveThesis DOTPG checkpoint exists:
  - `outputs/Dexh13HoraLightbulb_student_dotpg_codrive_thesis/codrive_thesis_formal_dotpg_dual_bc5_s42/student_output/dotpg_nn/model_best.ckpt`
- Paper eval result for this DOTPG checkpoint:
  - fixed-step reward mean `3.4462656`
  - done-rate mean `0.0010376`
  - episode return mean `2330.3963`
  - screw progress mean `2.2405`

### Remaining blocked/risky
- The existing `scripts/vis_dexh13_lightbulb_student_dotpg_codrive.sh` targets the non-thesis CoDrive task/path by default; for the synced CoDriveThesis checkpoint, use a direct `train.py` command or add a thesis-specific visualizer wrapper.

### Single recommended next step
- Visualize the main local DOTPG CoDriveThesis checkpoint:
  `outputs/Dexh13HoraLightbulb_student_dotpg_codrive_thesis/codrive_thesis_formal_dotpg_dual_bc5_s42/student_output/dotpg_nn/model_best.ckpt`

---

## v2-2026-05-11 -- DOTPG Visualization Normal-Speed Support

### Target milestone/subgoal
- Make headed policy visualization run at human-inspection normal speed by default when requested, especially for DOTPG student visual checks.

### What changed (files + behavior impact)
- Updated `dexscrew/dotpg/dotpg.py`:
  - Added `train.dotpg.test_realtime`, `test_realtime_factor`, and `test_sleep_sec`.
  - DOTPG `test()` can now sync playback to `env.dt * env.control_freq_inv` without changing physics or controller settings.
- Updated `AGENTS.md`:
  - Future headed IsaacGym visualization should default to normal real-time playback unless the user asks for fast/headless eval.

### What was verified (commands + key outcomes)
- `python -m py_compile dexscrew/dotpg/dotpg.py`
  - Passed.
- The CoDriveThesis task uses `sim.dt=0.005` and `controlFrequencyInv=10`, so `++train.dotpg.test_realtime=True ++train.dotpg.test_realtime_factor=1.0` targets about `0.05s` per policy step, i.e. normal 1x policy playback.

### Remaining blocked/risky
- Other student visualizers may still need equivalent realtime/sleep flags if their test loops bypass IsaacGym frame sync.

### Single recommended next step
- Use DOTPG visualization with `++train.dotpg.test_realtime=True ++train.dotpg.test_realtime_factor=1.0` for normal-speed local inspection.

---

## v2-2026-05-11 -- DOTPG Deploy Packs Created

### Target milestone/subgoal
- Package the two latest DOTPG-BC5 students for deployment with their matching PPO teacher and frozen YAML files.

### What changed (files + behavior impact)
- Created clean deploy folders with exactly four files each:
  - `sim2real/deploy/dotpg_deployv1/`
    - CoDrive task.
    - PPO teacher `best_reward_4159.37.pth`.
    - DOTPG student `model_best.ckpt`.
    - Matching task/train YAMLs.
  - `sim2real/deploy/dotpg_deployv2/`
    - CoDriveThesis task.
    - PPO teacher `best_reward_3655.17.pth`.
    - DOTPG student `model_best.ckpt`.
    - Matching task/train YAMLs.
- Removed the temporary README files and renamed the initial `dotpg1`/`dotpg2` folders per user request.

### What was verified (commands + key outcomes)
- Verified each deploy folder contains exactly the intended four files by `find`.
- Verified checksums match source checkpoints/YAMLs:
  - `dotpg_deployv1/model_best.ckpt` matches `sim2real/codrive/dotpg_bc5/model_best.ckpt`.
  - `dotpg_deployv2/model_best.ckpt` matches `outputs/Dexh13HoraLightbulb_student_dotpg_codrive_thesis/codrive_thesis_formal_dotpg_dual_bc5_s42/student_output/dotpg_nn/model_best.ckpt`.

### Remaining blocked/risky
- These deploy folders are local; sync to cloud/other OS only if deployment will happen there.

### Single recommended next step
- Use `sim2real/deploy/dotpg_deployv1` for non-thesis CoDrive deployment and `sim2real/deploy/dotpg_deployv2` for CoDriveThesis deployment.

---

## v2-2026-05-11 -- CoDriveThesis Student Deploy Packs Created

### Target milestone/subgoal
- Sync the paper/eval-selected CoDriveThesis student checkpoints from cloud and package all relevant student algorithms for deployment under `sim2real/deploy`.

### What changed (files + behavior impact)
- Synced eval-selected student checkpoints from cloud to local `outputs/...` paths.
- Created/updated deploy folders, each with exactly four files:
  - `sim2real/deploy/bc_latentbc_deploy/`
  - `sim2real/deploy/dagger_deploy/`
  - `sim2real/deploy/dotpg_deployv2/`
  - `sim2real/deploy/padapt_deploy/`
  - `sim2real/deploy/diffusion_latent_deploy/`
  - `sim2real/deploy/consistency_latent_deploy/`
  - `sim2real/deploy/flow_matching_deploy/`
  - `sim2real/deploy/diffusion_action_chunk_deploy/`
  - `sim2real/deploy/purebc_deploy/`
- Each CoDriveThesis deploy folder contains:
  - `best_reward_3655.17.pth`
  - `Dexh13HoraLightbulbSim2RealTwoFingerCoDriveThesis.task.yaml`
  - `Dexh13HoraLightbulbSim2RealTwoFingerCoDriveThesis.train.yaml`
  - `model_best.ckpt`
- Existing `sim2real/deploy/dotpg_deployv1/` remains the non-thesis CoDrive DOTPG deploy package.

### What was verified (commands + key outcomes)
- Read cloud handoff first:
  - `ssh cloud-training 'cd /root/code/dexscrew-repro && sed -n "1,220p" docs/cloud_session_handoff.md'`
- Verified all cloud source checkpoints/YAMLs existed before sync.
- Verified every deploy folder under `sim2real/deploy` has exactly four files.
- Verified SHA256 equality between each deploy `model_best.ckpt` and its source eval checkpoint for:
  `bc_latentbc`, `dagger`, `dotpg`, `padapt`, `diffusion_latent`, `consistency_latent`, `flow_matching`, `diffusion_action_chunk`, and `purebc`.

### Remaining blocked/risky
- `diffusion_action_chunk_deploy` is included because it is a valid eval-run student, but paper eval showed it is behaviorally poor compared with the other deploy candidates.
- Deploy folders are local artifacts; sync them elsewhere only when deploying from that machine.

### Single recommended next step
- Use `sim2real/deploy/consistency_latent_deploy` or `sim2real/deploy/flow_matching_deploy` as the strongest student deployment candidates from the current CoDriveThesis paper eval, while keeping the other folders for controlled comparisons.

---

## v2-2026-05-11 -- CoDrive Working YAML Init Noise Restored

### Target milestone/subgoal
- Restore original-style small object/hand-root init-position randomization in the active CoDrive working task YAML for future PPO probes.

### What changed (files + behavior impact)
- Updated `configs/task/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.yaml` only:
  - `env.object.init_pos_noise: [0.005, 0.005, 0.0]`
  - `env.asset.handRootPosNoise: [0.001, 0.001, 0.001]`
- Frozen deploy/reference packages under `sim2real/codrive` and `sim2real/deploy` were not changed.

### What was verified (commands + key outcomes)
- `git diff --check -- configs/task/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.yaml`
  - Passed.
- Confirmed the non-thesis CoDrive PPO teacher referred to by the user is the `best_reward_4159.37.pth` run, i.e. the roughly 4000-reward PPO baseline.

### Remaining blocked/risky
- This YAML change makes future training/eval distribution different from the frozen `sim2real/codrive/best_reward_4159.37.pth` teacher package unless that package is intentionally regenerated.

### Single recommended next step
- If using this noised CoDrive working YAML as a new baseline, launch a fresh PPO probe instead of treating `best_reward_4159.37.pth` as trained under the modified init-noise distribution.

---

## v2-2026-05-15 -- CoDriveExper Saved Initpose PPO Visualization

### Target milestone/subgoal
- Test the latest keyboard-tuned CoDrive init pose by applying it to the experimental CoDrive task YAML and driving it with the frozen roughly 4000-reward CoDrive PPO teacher.

### What changed (files + behavior impact)
- Updated `configs/task/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveExper.yaml` only:
  - `env.asset.handRootPos: [0.106000, 0.028000, 0.265000]`
  - `env.asset.handRootRPY: [3.141500, 0.387266, 3.141500]`
  - `env.asset.handInitPose` matched the latest saved file at `outputs/initpose_tuning/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveExper_current.yaml`.
  - Main joint changes: `right_index_joint_0=0.3499999940`, `right_index_joint_3=0.3692995608`, `right_thumb_joint_0=-0.3499999940`.
- The frozen pure CoDrive reference YAML/checkpoint under `sim2real/codrive` was not changed.

### What was verified (commands + key outcomes)
- Verified the saved initpose file and the edited CoDriveExper asset block with `sed`.
- Launched headed local PPO visualization using the latest saved pose:
  - Task: `Dexh13HoraLightbulbSim2RealTwoFingerCoDriveExper`
  - Checkpoint: `sim2real/codrive/best_reward_4159.37.pth`
  - Randomization/noise disabled for visual alignment with the keyboard tuner.

### Remaining blocked/risky
- This visualization intentionally tests the frozen CoDrive PPO on a modified init pose; behavior may differ from the original training distribution.

### Single recommended next step
- Inspect the running headed viewer; if the contact geometry is good, decide whether to train a fresh CoDriveExper PPO or keep the pose only for deployment/init-pose probing.
