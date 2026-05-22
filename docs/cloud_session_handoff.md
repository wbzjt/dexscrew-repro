# Cloud Session Handoff

Scope: live cloud execution state for `/root/code/dexscrew-repro`.

Use this file as the first stop when switching between Ubuntu-side and Windows-side Codex sessions. It records what the cloud machine has already done and what should happen next.

---

## 2026-05-18 -- Middle1 PPO3799 Students Synced To Ubuntu Local

### Current Cloud State
- GPU remains idle.
- The completed Middle1 PPO3799 PAdapt/DOTPG cloud outputs have been synced to Ubuntu local.

### Synced Outputs
- PAdapt:
  `outputs/Dexh13HoraLightbulb_student_padapt_codrive_middle1/middle1_keyboardlatest_noinitnoise_students_s42_20260518_160530_padapt1h_from_ppo3799/stage2_nn/model_best.ckpt`
- DOTPG:
  `outputs/Dexh13HoraLightbulb_student_dotpg_codrive_middle1/middle1_keyboardlatest_noinitnoise_students_s42_20260518_160530_dotpg1h_from_ppo3799_dual_bc5/student_output/dotpg_nn/model_best.ckpt`
- Pipeline logs/status:
  `outputs/cloud_pipeline_codrive_middle1_students_after_noinitnoise1h/middle1_keyboardlatest_noinitnoise_students_s42_20260518_160530/`

### Recommended Next Action
- Headed-visualize Middle1 PPO/PAdapt/DOTPG locally and decide which policy is worth deploying or further distilling.

---

## 2026-05-18 -- Middle1 PPO3799 PAdapt/DOTPG 1h Parallel Complete

### Current Cloud State
- GPU is idle:
  `NVIDIA GeForce RTX 4090 D, 1 MiB / 24564 MiB, 0%`.
- The tmux session `middle1_students_ppo3799_20260518_160530` has exited.
- Pipeline:
  `outputs/cloud_pipeline_codrive_middle1_students_after_noinitnoise1h/middle1_keyboardlatest_noinitnoise_students_s42_20260518_160530/`
- `status/phase.txt=done`

### Teacher Checkpoint
- Both students used the latest Middle1 NoInitNoise cloud PPO teacher:
  `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_middle1/middle1_keyboardlatest_noinitnoise_s42_20260518_111917_ppo1h/stage1_nn/best_reward_3799.45.pth`

### Results
- PAdapt:
  - `padapt_exit_status=124`
  - `padapt_timeout_status=expected_1h_timeout`
  - max parsed `Current Best: 3236.33`
  - checkpoint:
    `outputs/Dexh13HoraLightbulb_student_padapt_codrive_middle1/middle1_keyboardlatest_noinitnoise_students_s42_20260518_160530_padapt1h_from_ppo3799/stage2_nn/model_best.ckpt`
- DOTPG:
  - `dotpg_exit_status=124`
  - `dotpg_timeout_status=expected_1h_timeout`
  - max parsed `Current Best: 2901.99`
  - checkpoint:
    `outputs/Dexh13HoraLightbulb_student_dotpg_codrive_middle1/middle1_keyboardlatest_noinitnoise_students_s42_20260518_160530_dotpg1h_from_ppo3799_dual_bc5/student_output/dotpg_nn/model_best.ckpt`

### Recommended Next Cloud Action
- Sync both student output dirs and this pipeline directory to Ubuntu local, then headed-visualize the two students on `Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle1`.

---

## 2026-05-18 -- Middle1 PPO3799 PAdapt/DOTPG 1h Parallel Active

### Current Cloud State
- Active tmux session:
  `middle1_students_ppo3799_20260518_160530`
- Pipeline:
  `outputs/cloud_pipeline_codrive_middle1_students_after_noinitnoise1h/middle1_keyboardlatest_noinitnoise_students_s42_20260518_160530/`
- `status/phase.txt`: `running_parallel`
- Cloud GPU startup check after launch:
  `NVIDIA GeForce RTX 4090 D, 9162 MiB / 24564 MiB, 95%`
- Active jobs:
  - PAdapt 1h student distillation, `task.env.numEnvs=512`, `train.ppo.minibatch_size=6144`
  - DOTPG 1h student distillation, `task.env.numEnvs=1024`, `train.ppo.minibatch_size=12288`

### Teacher Checkpoint
- Both students use the latest Middle1 NoInitNoise cloud PPO teacher:
  `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_middle1/middle1_keyboardlatest_noinitnoise_s42_20260518_111917_ppo1h/stage1_nn/best_reward_3799.45.pth`

### Output Targets
- PAdapt:
  `outputs/Dexh13HoraLightbulb_student_padapt_codrive_middle1/middle1_keyboardlatest_noinitnoise_students_s42_20260518_160530_padapt1h_from_ppo3799/`
- DOTPG:
  `outputs/Dexh13HoraLightbulb_student_dotpg_codrive_middle1/middle1_keyboardlatest_noinitnoise_students_s42_20260518_160530_dotpg1h_from_ppo3799_dual_bc5/`

### Logs And Status
- PAdapt log:
  `outputs/cloud_pipeline_codrive_middle1_students_after_noinitnoise1h/middle1_keyboardlatest_noinitnoise_students_s42_20260518_160530/logs/padapt.log`
- DOTPG log:
  `outputs/cloud_pipeline_codrive_middle1_students_after_noinitnoise1h/middle1_keyboardlatest_noinitnoise_students_s42_20260518_160530/logs/dotpg.log`
- GPU log:
  `outputs/cloud_pipeline_codrive_middle1_students_after_noinitnoise1h/middle1_keyboardlatest_noinitnoise_students_s42_20260518_160530/logs/gpu_usage.csv`

### Recommended Next Cloud Action
- Let both jobs reach their expected `timeout 3600` wall-clock completion (`exit_status=124`), then sync the two student output dirs and pipeline status back to Ubuntu local for headed visualization.

---

## 2026-05-18 -- Cloud Status Check: No Active Cloud Training

### Current Cloud State
- GPU is idle:
  `NVIDIA GeForce RTX 4090 D, 1 MiB / 24564 MiB, 0%`.
- No active `train.py` / `timeout 3600` cloud training processes were found.

### Middle1 Status
- `Middle1 KeyboardLatest NoInitNoise PPO1h` completed:
  - pipeline:
    `outputs/cloud_pipeline_codrive_middle1_ppo1h/middle1_keyboardlatest_noinitnoise_s42_20260518_111917/`
  - `status/phase.txt=done`
  - `ppo_exit_status=124`
  - `timeout_status=expected_1h_timeout`
  - final teacher:
    `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_middle1/middle1_keyboardlatest_noinitnoise_s42_20260518_111917_ppo1h/stage1_nn/best_reward_3799.45.pth`
- The planned follow-up script path exists locally in prior notes, but this cloud path did not contain a launched student-after-PPO status directory:
  `outputs/cloud_pipeline_codrive_middle1_students_after_noinitnoise1h/middle1_keyboardlatest_noinitnoise_students_s42_20260518_115617/`.
- Older Middle1 students from the earlier `middle1_ppo1h_s42_20260515_120420` teacher are complete:
  - DOTPG:
    `outputs/Dexh13HoraLightbulb_student_dotpg_codrive_middle1/middle1_dotpg_s42_20260515_131253_dual_bc5_from_ppo1h/student_output/dotpg_nn/model_best.ckpt`
  - PAdapt exists locally from the paired local run:
    `outputs/Dexh13HoraLightbulb_student_padapt_codrive_middle1/middle1_padapt_s42_20260515_131253_from_ppo1h/stage2_nn/model_best.ckpt`

### Small Status
- `CoDriveSmall` PPO and students were run locally on Ubuntu, not on cloud:
  - PPO final best:
    `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_small/codrive_small_keyboardlatest_s42_20260518_120558_ppo1h/stage1_nn/best_reward_3985.40.pth`
  - PAdapt 1h:
    `outputs/Dexh13HoraLightbulb_student_padapt_codrive_small/codrive_small_students_from_ppo3985_s42_20260518_131156_padapt1h/stage2_nn/model_best.ckpt`
  - DOTPG 1h:
    `outputs/Dexh13HoraLightbulb_student_dotpg_codrive_small/codrive_small_students_from_ppo3985_s42_20260518_131156_dotpg1h_dual_bc5/student_output/dotpg_nn/model_best.ckpt`

### Recommended Next Cloud Action
- If continuing Middle1, sync the final cloud PPO output back to local before visualization:
  `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_middle1/middle1_keyboardlatest_noinitnoise_s42_20260518_111917_ppo1h/`.
- If continuing Small, visualize the local PAdapt/DOTPG checkpoints first; cloud is currently free for a longer follow-up run.

---

## 2026-05-18 -- Middle1 KeyboardLatest NoInitNoise PPO1h Active

### Current Cloud State
- Active tmux session:
  `middle1_keyboardlatest_noinitnoise_ppo1h_20260518_111917`
- Pipeline:
  `outputs/cloud_pipeline_codrive_middle1_ppo1h/middle1_keyboardlatest_noinitnoise_s42_20260518_111917/`
- `status/phase.txt`: `ppo`
- Active command:
  PPO teacher on `Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle1`
- Output name:
  `Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_middle1/middle1_keyboardlatest_noinitnoise_s42_20260518_111917_ppo1h`
- Runtime budget:
  `timeout 3600` wall-clock seconds.
- Cloud resource settings:
  - `task.env.numEnvs=12288`
  - `train.ppo.num_actors=12288`
  - `train.ppo.minibatch_size=24576`
  - `num_threads=22`
- Confirmed cloud task YAML values:
  - `object.init_pos_noise: [0.0, 0.0, 0.0]`
  - `asset.handRootPosNoise: [0.0, 0.0, 0.0]`
  - `asset.handRootPosZScaleComp: 0.0`
  - `asset.handRootPos: [0.092000, 0.020000, 0.245000]`
  - `right_index_joint_3: 0.4492996037`
  - `right_thumb_joint_2: 0.2999999821`

### Synced Files
- `configs/task/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle1.yaml`
- `configs/train/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle1.yaml`
- `outputs/initpose_tuning/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle1_current.yaml`
- `outputs/cloud_pipeline_codrive_middle1_ppo1h/middle1_keyboardlatest_noinitnoise_s42_20260518_111917/run_middle1_keyboardlatest_noinitnoise_ppo1h_cloud.sh`

### Verification
- The previous `middle1_keyboardlatest_s42_20260518_111658` cloud run was stopped/restarted because init-pose noise was not zero.
- Cloud startup verified:
  - Python active under `timeout 3600`.
  - GPU around `14313 MiB / 24564 MiB`.
  - First best checkpoint appeared:
    `best_reward_61.34.pth` at about 2.6 minutes elapsed.
- Mid-run sync for local visualization:
  - Cloud was still active: `phase=ppo`.
  - GPU around `14313 MiB / 24564 MiB`.
  - Synced current output to Ubuntu local.
  - Latest synced checkpoint:
    `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_middle1/middle1_keyboardlatest_noinitnoise_s42_20260518_111917_ppo1h/stage1_nn/best_reward_2807.89.pth`.

### Recommended Next Cloud Action
- Let this no-noise Middle1 PPO run to expected 1h timeout (`ppo_exit_status=124`), then sync:
  `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_middle1/middle1_keyboardlatest_noinitnoise_s42_20260518_111917_ppo1h/`
- Visualize the final `stage1_nn/best_reward_*.pth`.

---

## 2026-05-18 -- Pure CoDrive Reprobe PPO1h Active

### Current Cloud State
- Active tmux session:
  `codrive_reprobe_ppo1h_20260518_105210`
- Pipeline:
  `outputs/cloud_pipeline_codrive_reprobe_ppo1h/codrive_reprobe_s42_20260518_105210/`
- `status/phase.txt`: `ppo`
- Active command:
  PPO teacher on `Dexh13HoraLightbulbSim2RealTwoFingerCoDrive`
- Output name:
  `Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive/codrive_reprobe_s42_20260518_105210_ppo1h`
- Runtime budget:
  `timeout 3600` wall-clock seconds.
- Cloud resource settings:
  - `task.env.numEnvs=12288`
  - `train.ppo.num_actors=12288`
  - `train.ppo.minibatch_size=24576`
  - `num_threads=22`

### Synced Files
- `configs/task/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.yaml`
- `configs/train/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.yaml`
- `outputs/cloud_pipeline_codrive_reprobe_ppo1h/codrive_reprobe_s42_20260518_105210/run_codrive_reprobe_ppo1h_cloud.sh`

### Verification
- Previous `Index112 NoInitNoise` cloud PPO was manually stopped by user request.
  - Stop time: `2026-05-18T02:51:59+00:00`.
  - Last observed best before stop:
    `best_reward_3360.56.pth`.
  - `status/phase.txt`: `stopped_manual`.
- Pure CoDrive YAML was compared against the deployment copy under local `configs/codrive/`.
  - Task YAML SHA256 matched exactly.
  - Train YAML SHA256 matched exactly.
- Pure CoDrive cloud startup verified:
  - Python active under `timeout 3600`.
  - GPU around `14329 MiB / 24564 MiB`.
  - First best checkpoint appeared:
    `best_reward_65.72.pth` at about 3 minutes elapsed.

### Recommended Next Cloud Action
- Let the pure CoDrive reprobe run to the expected 1h timeout (`ppo_exit_status=124`), then sync:
  `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive/codrive_reprobe_s42_20260518_105210_ppo1h/`
- Compare the final best against the historical deploy teacher:
  `configs/codrive/best_reward_4159.37.pth`.

---

## 2026-05-18 -- CoDriveMiddle1 Index112 NoInitNoise PPO1h Active

### Current Cloud State
- Active tmux session:
  `middle1_index112_noinitnoise_ppo1h_20260518_101206`
- Pipeline:
  `outputs/cloud_pipeline_codrive_middle1_index112_noinitnoise_ppo1h/middle1_index112_noinitnoise_s42_20260518_101206/`
- `status/phase.txt`: `ppo`
- Active command:
  PPO teacher on `Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle1Index112NoInitNoise`
- Output name:
  `Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_middle1_index112_noinitnoise/middle1_index112_noinitnoise_s42_20260518_101206_ppo1h`
- Runtime budget:
  `timeout 3600` wall-clock seconds.
- Cloud resource settings:
  - `task.env.numEnvs=12288`
  - `train.ppo.num_actors=12288`
  - `train.ppo.minibatch_size=24576`
  - `num_threads=22`
- Isolation variables relative to current Middle1:
  - `object.init_pos_noise: [0.0, 0.0, 0.0]`
  - `asset.handRootPosNoise: [0.0, 0.0, 0.0]`
  - `right_index_joint_3: 1.1200000000`
  - `asset.handRootPosZScaleComp` remains `0.0`

### Synced Files
- `configs/task/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle1Index112NoInitNoise.yaml`
- `configs/train/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle1Index112NoInitNoise.yaml`
- `outputs/cloud_pipeline_codrive_middle1_index112_noinitnoise_ppo1h/middle1_index112_noinitnoise_s42_20260518_101206/run_middle1_index112_noinitnoise_ppo1h_cloud.sh`

### Verification
- Cloud handoff was read before launch.
- Cloud was idle before launch: RTX 4090 D around `1 MiB / 24564 MiB`.
- Startup health verified after launch:
  - Python active under `timeout 3600`
  - GPU around `14339 MiB / 24564 MiB`
  - first best checkpoint appeared and continued updating:
    `best_reward_117.90.pth` at about 3.6 minutes elapsed.
- 30min visual-check sync:
  - `phase=ppo`, cloud training still active.
  - GPU around `14339 MiB / 24564 MiB`.
  - best at about 30.5 minutes:
    `best_reward_3237.63.pth`.
  - local sync pulled the slightly newer checkpoint:
    `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_middle1_index112_noinitnoise/middle1_index112_noinitnoise_s42_20260518_101206_ppo1h/stage1_nn/best_reward_3255.81.pth`.

### Recommended Next Cloud Action
- Local 30min checkpoint is ready for headed visualization:
  `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_middle1_index112_noinitnoise/middle1_index112_noinitnoise_s42_20260518_101206_ppo1h/`
- If waiting for full completion, expect `ppo_exit_status=124` as the correct 1h timeout result.

---

## 2026-05-15 -- CoDriveMiddle1 Latest Initpose PPO1h Completed And Synced

### Current Cloud State
- Completed tmux session:
  `middle1_latestinit_ppo1h_20260515_150457`
- Pipeline:
  `outputs/cloud_pipeline_codrive_middle1_ppo1h/middle1_latestinit_s42_20260515_150457/`
- `status/phase.txt`: `done`
- Completed command:
  PPO teacher on `Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle1`
- Output name:
  `Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_middle1/middle1_latestinit_s42_20260515_150457_ppo1h`
- Runtime budget:
  `timeout 3600` wall-clock seconds.
- Cloud resource settings:
  - `task.env.numEnvs=12288`
  - `train.ppo.minibatch_size=24576`
  - `train.ppo.num_actors=12288`
  - `num_threads=22`
- Latest init pose synced from Ubuntu:
  - `handRootPos: [0.080000, 0.020000, 0.239000]`
  - `handRootRPY: [3.141500, 0.439627, 3.141500]`
  - index joints: `[0.3499999940, 0.9852794409, 0.2628971040, 0.5692995787]`
  - thumb joints: `[0.1299999952, 1.5700000525, 0.0399999991, 0.5719662905]`
- Completion:
  - `ppo_exit_status=124`
  - `timeout_status=expected_1h_timeout`
  - final best checkpoint:
    `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_middle1/middle1_latestinit_s42_20260515_150457_ppo1h/stage1_nn/best_reward_3309.63.pth`

### Synced Files
- `configs/task/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle1.yaml`
- `configs/train/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle1.yaml`
- `outputs/initpose_tuning/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle1_current.yaml`
- `outputs/cloud_pipeline_codrive_middle1_ppo1h/middle1_latestinit_s42_20260515_150457/run_middle1_latestinit_ppo1h_cloud.sh`
- Cloud pipeline logs/status were synced back to Ubuntu local:
  `outputs/cloud_pipeline_codrive_middle1_ppo1h/middle1_latestinit_s42_20260515_150457/`
- PPO output/checkpoint/events were synced back to Ubuntu local:
  `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_middle1/middle1_latestinit_s42_20260515_150457_ppo1h/`

### Verification
- Cloud was idle before launch: RTX 4090 D around `1 MiB / 24564 MiB`.
- Launched in tmux and confirmed active Python process under `timeout 3600`.
- GPU after startup around `14273 MiB / 24564 MiB`.
- During training, best reached `3309.63`.
- Final status file confirms expected timeout `124`.
- Cloud after completion is idle: RTX 4090 D around `1 MiB / 24564 MiB`, `0%`.

### Recommended Next Cloud Action
- No active cloud action is required for this run.
- Next useful action is local headed visualization of `best_reward_3309.63.pth`.

---

## 2026-05-15 -- CoDriveMiddle1 DOTPG1h Completed And Synced

### Current Cloud State
- Completed tmux session:
  `middle1_dotpg1h_20260515_131253`
- Pipeline:
  `outputs/cloud_pipeline_codrive_middle1_dotpg1h/middle1_dotpg_s42_20260515_131253/`
- `status/phase.txt`: `done`
- Completed command:
  DOTPG student distillation on `Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle1`
- Teacher checkpoint:
  `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_middle1/middle1_ppo1h_s42_20260515_120420/stage1_nn/best_reward_3783.10.pth`
- Output name:
  `Dexh13HoraLightbulb_student_dotpg_codrive_middle1/middle1_dotpg_s42_20260515_131253_dual_bc5_from_ppo1h`
- Runtime budget:
  `timeout 3600` wall-clock seconds.
- Cloud resource settings:
  - `task.env.numEnvs=1024`
  - `train.ppo.minibatch_size=12288`
  - `train.ppo.num_actors=1024`
  - `num_threads=10`
- DOTPG variant:
  - `policy_arch=teacher_actor`
  - `policy_loss_mode=dual`
  - `bc_coef=5.0`
  - `bc_alpha_max=20.0`
- Best student checkpoint:
  `outputs/Dexh13HoraLightbulb_student_dotpg_codrive_middle1/middle1_dotpg_s42_20260515_131253_dual_bc5_from_ppo1h/student_output/dotpg_nn/model_best.ckpt`
- Completion:
  - `dotpg_exit_status=124`
  - `timeout_status=expected_1h_timeout`
  - observed final current best plateau around `2397.86`

### Synced Files
- `configs/task/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle1.yaml`
- `configs/train/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle1.yaml`
- Middle1 PPO teacher checkpoint `best_reward_3783.10.pth`
- `outputs/cloud_pipeline_codrive_middle1_dotpg1h/middle1_dotpg_s42_20260515_131253/run_middle1_dotpg1h_cloud.sh`
- Cloud DOTPG pipeline logs/status were synced back to Ubuntu local:
  `outputs/cloud_pipeline_codrive_middle1_dotpg1h/middle1_dotpg_s42_20260515_131253/`
- Cloud DOTPG output/checkpoint/events were synced back to Ubuntu local:
  `outputs/Dexh13HoraLightbulb_student_dotpg_codrive_middle1/middle1_dotpg_s42_20260515_131253_dual_bc5_from_ppo1h/`

### Verification
- Cloud was idle before launch.
- DOTPG process used GPU during training, around `6.1GB / 24GB`.
- Early training reached current best around `32.18` and saved `student_output/dotpg_nn/model_best.ckpt`.
- Final status file confirms expected timeout `124`.
- Cloud after completion is idle: RTX 4090 D around `1 MiB / 24564 MiB`, `0%`.

### Recommended Next Cloud Action
- No active cloud action is required for this DOTPG run.
- Next useful step is local visualization against the paired Middle1 PAdapt checkpoint.

---

## 2026-05-15 -- CoDriveMiddle2 PPO1h Completed And Synced

### Current Cloud State
- No active Middle2 training process.
- GPU idle after completion:
  RTX 4090 D around `1 MiB / 24564 MiB`, `0%`.
- Pipeline:
  `outputs/cloud_pipeline_codrive_middle2_ppo1h/middle2_s42_20260515_120420/`
- `status/phase.txt`: `done`
- Completed command:
  PPO teacher on `Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle2`
- Output name:
  `Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_middle2/middle2_s42_20260515_120420_ppo1h`
- Runtime budget:
  `timeout 3600` wall-clock seconds.
- Cloud resource settings:
  - `task.env.numEnvs=12288`
  - `train.ppo.minibatch_size=24576`
  - `train.ppo.num_actors=12288`
  - `num_threads=22`
- GPU during startup:
  RTX 4090 D around `14.3GB / 24GB`, active utilization.

### Synced Files
- `configs/task/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle2.yaml`
- `configs/train/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle2.yaml`
- `outputs/cloud_pipeline_codrive_middle2_ppo1h/middle2_s42_20260515_120420/run_middle2_ppo1h_cloud.sh`

### Verification
- Active retry uses the existing cloud conda IsaacGym environment (`dexscrew-ig`), matching earlier successful cloud pipelines.
- PPO progress confirmed after startup:
  - `Agent Steps: 0004M`
  - FPS about `29k`
  - current best reward `230.80`
- Completion confirmed:
  - `ppo_exit_status=124`
  - `timeout_status=expected_1h_timeout`
  - best checkpoint:
    `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_middle2/middle2_s42_20260515_120420_ppo1h/stage1_nn/best_reward_3503.45.pth`
- Results synced back to Ubuntu local:
  - cloud pipeline logs/status
  - PPO output directory and best checkpoint

### Notes
- First launch attempted Docker and failed immediately because this cloud machine does not have image `dexscrew:ig20-py38`.

### Recommended Next Cloud Action
- No immediate cloud action required for Middle2.
- Next useful step is local visualization/comparison against Middle1:
  - Middle1 local best: `best_reward_3783.10.pth`
  - Middle2 cloud-synced best: `best_reward_3503.45.pth`

---

## 2026-05-13 -- RealBulb PPO2h + PAdapt/DOTPG2h Pipeline Active

### Current Cloud State
- Active tmux session:
  `realbulb_ppo2h_students2h_20260513_031105`
- Pipeline:
  `outputs/cloud_pipeline_realbulb_ppo2h_padapt_dotpg2h/realbulb_s42_20260513_031105/`
- Convenience link:
  `outputs/cloud_pipeline_realbulb_ppo2h_padapt_dotpg2h/latest`
- `status/phase.txt`: `ppo`
- Active command:
  PPO teacher on `Dexh13HoraLightbulbSim2RealTwoFingerCoDriveRealBulb`
- GPU at startup confirmation:
  RTX 4090 D around `14359 MiB / 24564 MiB`, roughly `80%` utilization.

### Synced Files
- `assets/screw/realbulb/0000_lightbulb.urdf`
- `configs/task/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveRealBulb.yaml`
- `configs/train/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveRealBulb.yaml`
- `outputs/cloud_pipeline_realbulb_ppo2h_padapt_dotpg2h/run_realbulb_ppo2h_padapt_dotpg2h.sh`

### Pipeline Plan
- PPO teacher:
  - `timeout 7200`
  - `task.env.numEnvs=12288`
  - `train.ppo.minibatch_size=24576`
  - output:
    `outputs/Dexh13HoraLightbulb_teacher_realbulb/realbulb_s42_20260513_031105_ppo2h/`
- After PPO timeout, the script explicitly selects newest/best:
  `stage1_nn/best_reward_*.pth`
- Student phase starts automatically afterward:
  - PAdapt: `timeout 7200`, `task.env.numEnvs=512`, `train.ppo.minibatch_size=6144`
  - DOTPG: `timeout 7200`, `task.env.numEnvs=1024`, `train.ppo.minibatch_size=12288`
  - PAdapt and DOTPG run in parallel.

### Verification
- Cloud smoke passed before launch:
  - loaded `screw_realbulb`
  - generated initial poses at scales `0.975` and `1.025`
  - exited with `max steps achieved`
- PPO live progress confirmed:
  - `Agent Steps: 0005M ... Current Best: 57.18`

### Recommended Next Cloud Action
- Check:
  `cat outputs/cloud_pipeline_realbulb_ppo2h_padapt_dotpg2h/latest/status/phase.txt`
- While phase is `ppo`, monitor:
  `grep -a "Agent Steps" outputs/cloud_pipeline_realbulb_ppo2h_padapt_dotpg2h/latest/logs/ppo.log | tail`
- After PPO reaches expected timeout `124`, confirm:
  - `selected_teacher_ckpt.txt` exists
  - `status/phase.txt` becomes `students_parallel`
  - both PAdapt and DOTPG logs show training progress

---

## 2026-05-12 -- Real-Size CoDrive Bulb Added Local

### Current Cloud/Local State
- No cloud training/eval job was launched.
- Ubuntu local workspace added a real-size bulb asset and a matching CoDrive task variant.

### Local Files
- `assets/screw/realbulb/0000_lightbulb.urdf`
  - visual and collision both use `contact0.stl` / `contact1.stl`.
  - geometry scaled to about `140 mm` length and `60 mm` max diameter.
  - viewer display now matches the actual contact surface.
- `configs/task/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveRealBulb.yaml`
  - `object.type: screw_realbulb`
  - `baseObjScale: 1.00`
  - object scale randomization covers approximately `0.95-1.05`.
- `configs/train/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveRealBulb.yaml`
  - copied from the original CoDrive train YAML for Hydra compatibility.

### Verification
- XML/YAML parse checks passed.
- Local IsaacGym smoke loaded the new asset and exited with `max steps achieved`.

### Recommended Next Cloud Action
- No cloud action is required yet.
- Before any cloud training, sync the new asset/task/train files and visually confirm the inherited hand init pose still fits the taller real-size bulb.

---

## 2026-05-12 -- Original CoDrive Student Deploy Packages Completed Local

### Current Cloud/Local State
- No cloud training/eval job was launched.
- Ubuntu local deploy packages for original non-noise CoDrive were completed under `sim2real/deploy/codrive`.

### Local Deploy Packages
- Original CoDrive deploy packages now available:
  - `sim2real/deploy/codrive/padapt_deploy`
  - `sim2real/deploy/codrive/dotpg_deployv1`
  - `sim2real/deploy/codrive/diffusion_latent_deploy`
  - `sim2real/deploy/codrive/consistency_latent_deploy`
  - `sim2real/deploy/codrive/flow_matching_deploy`
  - `sim2real/deploy/codrive/diffusion_action_chunk_deploy`
  - `sim2real/deploy/codrive/bc_latentbc_deploy`
  - `sim2real/deploy/codrive/dagger_deploy`
- Each contains task YAML, train YAML, teacher `best_reward_4159.37.pth`, and one student `model_best.ckpt`.
- Teacher/YAML files in every package hash-match the frozen original CoDrive package under `sim2real/codrive`.

### Recommended Next Cloud Action
- No cloud action is required for this local packaging step.
- If evaluating deployment candidates on cloud later, use `sim2real/deploy/codrive` and remember these are the original no-init-noise CoDrive students, not CoDriveNoise.

---

## 2026-05-12 -- Local Deploy Packages Grouped By Task

### Current Cloud/Local State
- No cloud training/eval job was launched.
- Ubuntu local deploy packages were reorganized by task family.

### Local Deploy Layout
- `sim2real/deploy/codrive/`
  - non-thesis CoDrive deploy package, currently `dotpg_deployv1`
- `sim2real/deploy/codrive_thesis/`
  - CoDriveThesis deploy packages: BC/LatentBC, DAgger, DOTPG v2, PAdapt, diffusion latent, consistency, flow matching, action chunk, and PureBC
- `sim2real/deploy/codrive_noise/`
  - CoDriveNoise deploy packages: PAdapt, DOTPG, diffusion latent, consistency, flow matching, action chunk, PureBC, DAgger, and BC/LatentBC
- Verified every nested deploy package still has exactly four files.

### Recommended Next Cloud Action
- No cloud action is required for this local file organization change.
- If using a deploy package from another OS/session, use the task-family folder prefix.

---

## 2026-05-12 -- CoDriveNoise Student Deploy Packages Grouped Local

### Current Cloud/Local State
- Cloud training remains complete and idle; no new training job was launched.
- Ubuntu local workspace synced the cloud CoDriveNoise student outputs for:
  - diffusion latent
  - consistency latent
  - flow matching
  - diffusion action chunk
  - PureBC
  - DAgger
- Existing local/cloud-synced PAdapt, DOTPG, teacher pth, task YAML, and train YAML were verified.

### Local Deploy Packages
- The following local folders now exist under `sim2real/deploy/codrive_noise`, each with exactly four deploy files: task YAML, train YAML, PPO teacher pth, and student `model_best.ckpt`.
  - `codrive_noise_padapt_deploy`
  - `codrive_noise_dotpg_deploy`
  - `codrive_noise_diffusion_latent_deploy`
  - `codrive_noise_consistency_latent_deploy`
  - `codrive_noise_flow_matching_deploy`
  - `codrive_noise_diffusion_action_chunk_deploy`
  - `codrive_noise_purebc_deploy`
  - `codrive_noise_dagger_deploy`
  - `codrive_noise_bc_latentbc_deploy`
- Verified no `sim2real/deploy/codrive_noise_*_deploy` directories remain directly under `sim2real/deploy`.

### Recommended Next Cloud Action
- No cloud action is required for packaging.
- Next useful cloud action is unified fixed-step eval over these CoDriveNoise deploy candidates.

---

## 2026-05-12 -- CoDriveNoise Remaining Baselines 3h Completed

### Current Cloud State
- Completed run tag:
  `codrive_noise_remaining_s42_3h_20260512_013000`
- Main cloud pipeline:
  `outputs/cloud_pipeline_codrive_noise_remaining_baselines_3h/codrive_noise_remaining_s42_3h_20260512_013000/`
- `status/phase.txt`: `done`
- No active `python train.py` process.
- No active CoDriveNoise tmux sessions; only the old stale paper supervisor session remains.
- Cloud GPU after completion:
  RTX 4090 D idle, about `1 MiB / 24564 MiB`, `0%`.

### Time-Budget Allocation
- User allowed up to 10h total wall-clock budget.
- Chosen schedule keeps the formal remaining-baseline comparison at 3h per method, matching the existing PAdapt/DOTPG 3h CoDriveNoise runs.
- Remaining wall-clock budget is intentionally reserved for failure recovery, sync, visualization, and fixed-step eval rather than blindly extending all baselines and making them unfair against PAdapt/DOTPG.

### Cloud Jobs And Exit Status
- Cloud script:
  `outputs/cloud_pipeline_codrive_noise_remaining_baselines_3h/run_cloud_diffusion_purebc_3h.sh`
- Ran 5 jobs in parallel with `timeout 10800`, `task.env.numEnvs=192`, `train.ppo.minibatch_size=2304`:
  - `DiffusionLatentStudent`
  - `ConsistencyLatentStudent`
  - `FlowMatchingLatentStudent`
  - `DiffusionActionChunkStudent`
  - `PureBC`
- Separate cloud DAgger script:
  `outputs/cloud_pipeline_codrive_noise_remaining_baselines_3h/run_cloud_dagger_3h.sh`
- Ran DAgger with `timeout 10800`, `task.env.numEnvs=128`, `train.ppo.minibatch_size=1536`, CPU replay buffer.
- Exit statuses were all `124`, expected for requested 3h timeouts:
  - `diffusion_latent_exit_status = 124`
  - `consistency_latent_exit_status = 124`
  - `flow_matching_latent_exit_status = 124`
  - `diffusion_action_chunk_exit_status = 124`
  - `purebc_exit_status = 124`
  - `dagger_exit_status = 124`

### Teacher And Task
- Task:
  `Dexh13HoraLightbulbSim2RealTwoFingerCoDriveNoise`
- Teacher:
  `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_noise/codrive_noise_s42_20260511_094738_ppo3h/stage1_nn/best_reward_4076.32.pth`

### Parsed Cloud Rewards
- `DiffusionLatentStudent`: current-best max `4019.67`
- `ConsistencyLatentStudent`: current-best max `3840.09`
- `FlowMatchingLatentStudent`: current-best max `3850.96`
- `DiffusionActionChunkStudent`: current-best max `1341.73`
- `PureBC`: current-best max `3305.94`
- `DAgger`: no `Current Best` line; train return last `1655.33`, best student eval `979.06`
- Tail scans of real training logs found no `Traceback`, CUDA OOM, missing-key/size-mismatch, or segmentation-fault patterns.

### Cloud Checkpoints
- Diffusion latent:
  `outputs/Dexh13HoraLightbulb_student_diffusion_latent_codrive_noise/codrive_noise_remaining_s42_3h_20260512_013000/stage2_diffusion_nn/model_best.ckpt`
- Consistency:
  `outputs/Dexh13HoraLightbulb_student_consistency_codrive_noise/codrive_noise_remaining_s42_3h_20260512_013000/stage2_consistency_nn/model_best.ckpt`
- Flow matching:
  `outputs/Dexh13HoraLightbulb_student_flow_matching_codrive_noise/codrive_noise_remaining_s42_3h_20260512_013000/stage2_flow_nn/model_best.ckpt`
- Diffusion action chunk:
  `outputs/Dexh13HoraLightbulb_student_diffusion_action_chunk_codrive_noise/codrive_noise_remaining_s42_3h_20260512_013000/stage2_diffusion_action_chunk_nn/model_best.ckpt`
- PureBC:
  `outputs/Dexh13HoraLightbulb_student_purebc_codrive_noise/codrive_noise_remaining_s42_3h_20260512_013000/stage2_bc_nn/model_best.ckpt`
- DAgger:
  `outputs/Dexh13HoraLightbulb_student_dagger_codrive_noise/codrive_noise_remaining_s42_3h_20260512_013000_dagger_pure_replay/dagger_nn/model_best.ckpt`

### Local Companion Job
- Ubuntu local machine completed `BCStudent`/LatentBC for the same run tag and teacher.
- Local wrapper status file remained stale at `running`, but the Docker/train process is gone and checkpoint files exist.
- Local BC/LatentBC checkpoint:
  `outputs/Dexh13HoraLightbulb_student_bc_codrive_noise/codrive_noise_remaining_s42_3h_20260512_013000_latentbc/bc_nn/model_best.ckpt`
- Local BC eval-select history:
  - max `avg_reward = 3.47085`
  - max `score = 2.98257`

### Recommended Next Cloud Action
- Sync the six cloud baseline outputs locally.
- Run a unified fixed-step eval and visualization over CoDriveNoise: PPO teacher, PAdapt, DOTPG, diffusion latent, consistency, flow matching, action chunk, PureBC, DAgger, and BC/LatentBC.

---

## 2026-05-12 -- CoDriveNoise PPO/PAdapt/DOTPG Pipeline Completed

### Current Cloud State
- Pipeline completed:
  `outputs/cloud_pipeline_codrive_noise_ppo3h_padapt_dotpg3h/codrive_noise_s42_20260511_094738/`
- Convenience symlink:
  `outputs/cloud_pipeline_codrive_noise_ppo3h_padapt_dotpg3h/latest`
- `status/phase.txt`: `done`
- No active `python train.py` process.
- GPU after completion: RTX 4090 D idle, about `1 MiB / 24564 MiB`, `0%`.
- The launch tmux session has exited; only the old stale paper-eval tmux session remains.

### Exit Status
- PPO teacher: `status/ppo_exit_status = 124`
- PAdapt student: `status/padapt_exit_status = 124`
- DOTPG student: `status/dotpg_exit_status = 124`
- All three `124` statuses are expected timeout completions for the requested 3h phases.

### Result Checkpoints
- Selected PPO teacher:
  `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_noise/codrive_noise_s42_20260511_094738_ppo3h/stage1_nn/best_reward_4076.32.pth`
- PAdapt student:
  `outputs/Dexh13HoraLightbulb_student_padapt_codrive_noise/codrive_noise_s42_20260511_094738_padapt3h_from_ppo3h/stage2_nn/model_best.ckpt`
- DOTPG student:
  `outputs/Dexh13HoraLightbulb_student_dotpg_codrive_noise/codrive_noise_s42_20260511_094738_dotpg3h_from_ppo3h_dual_bc5/student_output/dotpg_nn/model_best.ckpt`

### Parsed Training Rewards
- PPO actual training lines: max/current best `4076.32`.
- PAdapt actual training lines: max/current best `3501.40`.
- DOTPG actual training lines: max/current best `2631.82`.
- Important log note: these logs include startup dirty-git-diff text. Broad greps may find unrelated historical `Current Best` or error words from embedded docs/diffs. Use only real progress lines matching `Agent Steps: ... Current Best`.
- Tail scans of the final 300 log lines for each phase found no `Traceback`, `RuntimeError`, CUDA OOM, missing-key/size-mismatch, or segmentation-fault patterns.

### Recommended Next Cloud Action
- Sync the three result checkpoints plus the two CoDriveNoise YAML files to local if the user wants visualization or deploy packaging.
- For final paper-style comparison, run the same fixed-step eval protocol as the existing CoDriveThesis table rather than relying only on training reward.

---

## 2026-05-11 -- CoDriveNoise PPO3h + PAdapt/DOTPG3h Pipeline Active

### Current Cloud State
- Active tmux session:
  `codrive_noise_ppo3h_students3h_20260511_174738`
- Active pipeline:
  `outputs/cloud_pipeline_codrive_noise_ppo3h_padapt_dotpg3h/codrive_noise_s42_20260511_094738/`
- Convenience symlink:
  `outputs/cloud_pipeline_codrive_noise_ppo3h_padapt_dotpg3h/latest`
- Current phase file:
  `outputs/cloud_pipeline_codrive_noise_ppo3h_padapt_dotpg3h/latest/status/phase.txt`
  currently showed `ppo` at launch validation.
- PPO process confirmed active after launch:
  `python train.py task=Dexh13HoraLightbulbSim2RealTwoFingerCoDriveNoise ...`
- GPU after PPO startup:
  RTX 4090 D, about `14327 MiB / 24564 MiB`, about `67%` utilization.
- Latest startup check observed PPO checkpoint:
  `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_noise/codrive_noise_s42_20260511_094738_ppo3h/stage1_nn/best_reward_70.57.pth`

### Config/Code State
- Original CoDrive task restored to no init-pose noise:
  - `configs/task/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.yaml`
  - `object.init_pos_noise: [0.0, 0.0, 0.0]`
  - `asset.handRootPosNoise: [0.0, 0.0, 0.0]`
- New noise variant added and synced:
  - `configs/task/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveNoise.yaml`
  - `configs/train/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveNoise.yaml`
  - `eval_cache_name: sim2real_twofinger_codrive_noise`
  - `object.init_pos_noise: [0.005, 0.005, 0.0]`
  - `asset.handRootPosNoise: [0.001, 0.001, 0.001]`
- Pipeline script:
  `outputs/cloud_pipeline_codrive_noise_ppo3h_padapt_dotpg3h/run_codrive_noise_ppo3h_padapt_dotpg3h.sh`

### Pipeline Plan
- PPO teacher:
  - timeout: `10800s` (3h)
  - task: `Dexh13HoraLightbulbSim2RealTwoFingerCoDriveNoise`
  - output: `Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_noise/codrive_noise_s42_20260511_094738_ppo3h`
  - aggressive 24GB GPU setting: `task.env.numEnvs=12288`, `train.ppo.minibatch_size=24576`
- After PPO exits, the script selects the max-reward `stage1_nn/best_reward_*.pth` and writes:
  `selected_teacher_ckpt.txt`
- PAdapt student:
  - timeout: `10800s` (3h)
  - output: `Dexh13HoraLightbulb_student_padapt_codrive_noise/codrive_noise_s42_20260511_094738_padapt3h_from_ppo3h`
- DOTPG student:
  - timeout: `10800s` (3h)
  - output: `Dexh13HoraLightbulb_student_dotpg_codrive_noise/codrive_noise_s42_20260511_094738_dotpg3h_from_ppo3h_dual_bc5`
  - DOTPG recipe: current dual actor loss + teacher actor init + BC anchor `bc_coef=5.0`.
- Expected timeout exit status for completed wall-clock phases: `124`.

### Logs And Status Files
- Pipeline log:
  `outputs/cloud_pipeline_codrive_noise_ppo3h_padapt_dotpg3h/latest/pipeline.log`
- PPO log:
  `outputs/cloud_pipeline_codrive_noise_ppo3h_padapt_dotpg3h/latest/logs/ppo.log`
- Student logs:
  `outputs/cloud_pipeline_codrive_noise_ppo3h_padapt_dotpg3h/latest/logs/padapt.log`
  `outputs/cloud_pipeline_codrive_noise_ppo3h_padapt_dotpg3h/latest/logs/dotpg.log`
- Exact commands:
  `commands_ppo.txt`, `commands_padapt.txt`, `commands_dotpg.txt`
- Exit statuses:
  `status/ppo_exit_status`, `status/padapt_exit_status`, `status/dotpg_exit_status`

### Recommended Next Cloud Action
- Monitor the active tmux/pipeline until PPO reaches the 3h timeout, then verify:
  - `status/ppo_exit_status` is `124` or clean `0`
  - `selected_teacher_ckpt.txt` exists and points to the intended best PPO checkpoint
  - PAdapt and DOTPG both start and complete their own 3h timeouts
- Do not modify the original CoDrive YAML for this ablation; use the `CoDriveNoise` task name for the init-noise experiment.

---

## 2026-05-11 -- DOTPG Normal-Speed Visualization Support Synced

### Current Cloud State
- No training or eval was launched for this update.
- Purpose: keep future Ubuntu/Windows/cloud visual checks aligned with the new local default that headed visualization should run at normal speed unless fast eval is explicitly requested.

### Synced/Expected Source Behavior
- `AGENTS.md` records that headed IsaacGym visualization intended for human inspection should default to real-time playback.
- `dexscrew/dotpg/dotpg.py` supports:
  - `++train.dotpg.test_realtime=True`
  - `++train.dotpg.test_realtime_factor=1.0`
  - optional fixed override `++train.dotpg.test_sleep_sec=<seconds>`
- For CoDriveThesis, normal speed is about one policy step per `sim.dt * controlFrequencyInv = 0.005 * 10 = 0.05s`.

### Recommended Next Cloud Action
- For DOTPG headed visualization, include:
  `++train.dotpg.test_realtime=True ++train.dotpg.test_realtime_factor=1.0`.
- Keep headless numeric eval fast unless the user asks for wall-clock playback.

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
