# Student Acceptance Summary

## Unified Metrics Table

| Algorithm | Max Current Best | Last EpReward | Last EpLen | Last DoneRate | Last TotalLoss | Median LastFPS | Error Hits | Best CKPT |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| padapt | 1496.08 | N/A | N/A | N/A | N/A | 848.10 | 0 | `outputs/XHandHoraScrewDriver_student_padapt/run_a/stage2_nn/model_best.ckpt` |
| purebc | 1495.38 | N/A | N/A | N/A | N/A | 873.90 | 0 | `outputs/XHandHoraScrewDriver_student_purebc/run_a_seed42_15min/stage2_bc_nn/model_best.ckpt` |
| diffusion_latent | 1786.12 | N/A | N/A | N/A | N/A | 794.00 | 0 | `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_recon05_seed42_15min/stage2_diffusion_nn/model_best.ckpt` |
| diffusion_action_chunk | 1203.64 | N/A | N/A | N/A | N/A | 828.70 | 0 | `outputs/XHandHoraScrewDriver_student_diffusion_action_chunk/run_a_action_chunk_fix_seed42_15min/stage2_diffusion_action_chunk_nn/model_best.ckpt` |
| diffusion_action_chunk_tune_v1 | 1280.19 | N/A | N/A | N/A | N/A | 834.50 | 0 | `outputs/XHandHoraScrewDriver_student_diffusion_action_chunk/run_a_action_chunk_tune_mix300k_fa2_cb02_seed42_15min/stage2_diffusion_action_chunk_nn/model_best.ckpt` |
| diffusion_action_chunk_tune_v2 | 1502.21 | N/A | N/A | N/A | N/A | 862.90 | 0 | `outputs/XHandHoraScrewDriver_student_diffusion_action_chunk/run_a_action_chunk_tune_mix500k_fa3_cb02_seed42_15min/stage2_diffusion_action_chunk_nn/model_best.ckpt` |
| diffusion_action_chunk_detfix_retrain | 1692.80 | N/A | N/A | N/A | N/A | 825.10 | 0 | `outputs/XHandHoraScrewDriver_student_diffusion_action_chunk/run_a_action_chunk_tune_fa5_cb02_mix500k_w0_detfix_seed42_15min/stage2_diffusion_action_chunk_nn/model_best.ckpt` |
| diffusion_action_chunk_alignsel_v1 | 1166.49 | N/A | N/A | N/A | N/A | 681.90 | 0 | `outputs/XHandHoraScrewDriver_student_diffusion_action_chunk/run_a_action_chunk_alignsel_fa5_cb02_mix120k_seed42_15min/stage2_diffusion_action_chunk_nn/model_best_student.ckpt` |
| padapt_trainrange_full | 32.31 | N/A | N/A | N/A | N/A | 800.90 | 0 | `outputs/XHandHoraScrewDriver_student_padapt_trainrange/run_a_trainrange_seed42_15min/stage2_nn/model_best.ckpt` |
| padapt_trainrange_adapt_mu | 69.11 | N/A | N/A | N/A | N/A | 793.50 | 0 | `outputs/XHandHoraScrewDriver_student_padapt_trainrange/run_a_trainrange_adapt_mu_seed42_15min/stage2_nn/model_best.ckpt` |
| padapt_trainrange_adapt_actor | 7.06 | N/A | N/A | N/A | N/A | 745.70 | 0 | `outputs/XHandHoraScrewDriver_student_padapt_trainrange/run_a_trainrange_adapt_actor_seed42_15min/stage2_nn/model_best.ckpt` |

## Run Artifacts

- `padapt`
  - run_dir: `outputs/XHandHoraScrewDriver_student_padapt/run_a`
  - log: `outputs/XHandHoraScrewDriver_student_padapt/run_a/train_15min.log`
  - event: `outputs/XHandHoraScrewDriver_student_padapt/run_a/stage2_tb/events.out.tfevents.1774003538.wbz-ubuntu22-pc`
  - best_ckpt: `outputs/XHandHoraScrewDriver_student_padapt/run_a/stage2_nn/model_best.ckpt`
- `purebc`
  - run_dir: `outputs/XHandHoraScrewDriver_student_purebc/run_a_seed42_15min`
  - log: `outputs/XHandHoraScrewDriver_student_purebc/run_a_seed42_15min/train_900s.log`
  - event: `outputs/XHandHoraScrewDriver_student_purebc/run_a_seed42_15min/stage2_bc_tb/events.out.tfevents.1774005528.wbz-ubuntu22-pc`
  - best_ckpt: `outputs/XHandHoraScrewDriver_student_purebc/run_a_seed42_15min/stage2_bc_nn/model_best.ckpt`
- `diffusion_latent`
  - run_dir: `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_recon05_seed42_15min`
  - log: `N/A` (direct command run without a persisted `train*.log`)
  - event: `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_recon05_seed42_15min/stage2_diffusion_tb/events.out.tfevents.1774284474.wbz-ubuntu22-pc`
  - best_ckpt: `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_recon05_seed42_15min/stage2_diffusion_nn/model_best.ckpt`
  - deploy_ckpt_note: deterministic multiseed512 on `seed=42,43,44` gives `nominal=2.272142±0.078250`, `light=1.907460±0.124687`, `hard=1.650344±0.149604`. Relative to the previous representative `run_a_latent_robust_light_seed42_15min`, the nominal and hard gains are clear (`2.083162±0.225799 -> 2.272142±0.078250`, `1.582382±0.210612 -> 1.650344±0.149604`). The earlier `light` comparison was too pessimistic because it mixed two different perturbation settings. After aligning `light` to the same stronger `v2` setting (`obs_noise=0.03/0.015`, `force=1.0`, `prob=0.2`), the old representative scores `1.935738±0.079995`, so the gap to `latent_recon05` is only `0.028278` rather than a large regression. Follow-up 15min sweeps `latent_recon03` and `latent_recon04` were both rejected, so `latent_recon05` remains the current best diffusion-latent representative; the next step should focus on seed-level `light_v2` diagnostics rather than more coefficient down-sweeps.
- `diffusion_action_chunk`
  - run_dir: `outputs/XHandHoraScrewDriver_student_diffusion_action_chunk/run_a_action_chunk_fix_seed42_15min`
  - log: `outputs/XHandHoraScrewDriver_student_diffusion_action_chunk/run_a_action_chunk_fix_seed42_15min/train_900s.log`
  - event: `outputs/XHandHoraScrewDriver_student_diffusion_action_chunk/run_a_action_chunk_fix_seed42_15min/stage2_diffusion_action_chunk_tb/events.out.tfevents.1774009950.wbz-ubuntu22-pc`
  - best_ckpt: `outputs/XHandHoraScrewDriver_student_diffusion_action_chunk/run_a_action_chunk_fix_seed42_15min/stage2_diffusion_action_chunk_nn/model_best.ckpt`
- `diffusion_action_chunk_tune_v1`
  - run_dir: `outputs/XHandHoraScrewDriver_student_diffusion_action_chunk/run_a_action_chunk_tune_mix300k_fa2_cb02_seed42_15min`
  - log: `outputs/XHandHoraScrewDriver_student_diffusion_action_chunk/run_a_action_chunk_tune_mix300k_fa2_cb02_seed42_15min/train_900s.log`
  - event: `outputs/XHandHoraScrewDriver_student_diffusion_action_chunk/run_a_action_chunk_tune_mix300k_fa2_cb02_seed42_15min/stage2_diffusion_action_chunk_tb/events.out.tfevents.1774026282.wbz-ubuntu22-pc`
  - best_ckpt: `outputs/XHandHoraScrewDriver_student_diffusion_action_chunk/run_a_action_chunk_tune_mix300k_fa2_cb02_seed42_15min/stage2_diffusion_action_chunk_nn/model_best.ckpt`
- `diffusion_action_chunk_tune_v2`
  - run_dir: `outputs/XHandHoraScrewDriver_student_diffusion_action_chunk/run_a_action_chunk_tune_mix500k_fa3_cb02_seed42_15min`
  - log: `outputs/XHandHoraScrewDriver_student_diffusion_action_chunk/run_a_action_chunk_tune_mix500k_fa3_cb02_seed42_15min/train_900s.log`
  - event: `outputs/XHandHoraScrewDriver_student_diffusion_action_chunk/run_a_action_chunk_tune_mix500k_fa3_cb02_seed42_15min/stage2_diffusion_action_chunk_tb/events.out.tfevents.1774027205.wbz-ubuntu22-pc`
  - best_ckpt: `outputs/XHandHoraScrewDriver_student_diffusion_action_chunk/run_a_action_chunk_tune_mix500k_fa3_cb02_seed42_15min/stage2_diffusion_action_chunk_nn/model_best.ckpt`
- `diffusion_action_chunk_detfix_retrain`
  - run_dir: `outputs/XHandHoraScrewDriver_student_diffusion_action_chunk/run_a_action_chunk_tune_fa5_cb02_mix500k_w0_detfix_seed42_15min`
  - event: `outputs/XHandHoraScrewDriver_student_diffusion_action_chunk/run_a_action_chunk_tune_fa5_cb02_mix500k_w0_detfix_seed42_15min/stage2_diffusion_action_chunk_tb/events.out.tfevents.1774171476.wbz-ubuntu22-pc`
  - best_ckpt: `outputs/XHandHoraScrewDriver_student_diffusion_action_chunk/run_a_action_chunk_tune_fa5_cb02_mix500k_w0_detfix_seed42_15min/stage2_diffusion_action_chunk_nn/model_best.ckpt`
  - deploy_ckpt_note: `model_last.ckpt` is the least-bad deploy candidate in this run, but multiseed512 nominal remains negative
- `diffusion_action_chunk_alignsel_v1`
  - run_dir: `outputs/XHandHoraScrewDriver_student_diffusion_action_chunk/run_a_action_chunk_alignsel_fa5_cb02_mix120k_seed42_15min`
  - event: `outputs/XHandHoraScrewDriver_student_diffusion_action_chunk/run_a_action_chunk_alignsel_fa5_cb02_mix120k_seed42_15min/stage2_diffusion_action_chunk_tb/events.out.tfevents.1774185437.wbz-ubuntu22-pc`
  - best_ckpt: `outputs/XHandHoraScrewDriver_student_diffusion_action_chunk/run_a_action_chunk_alignsel_fa5_cb02_mix120k_seed42_15min/stage2_diffusion_action_chunk_nn/model_best_student.ckpt`
  - deploy_ckpt_note: selector ranking under nominal seed42/512 is `model_best_student (-0.253256) > model_last (-0.388507) > model_best_student_reward (-0.899636)`; reward-aligned selector works mechanically but is not yet a better deploy proxy
- `diffusion_action_chunk_alignsel_len4_v1`
  - run_dir: `outputs/XHandHoraScrewDriver_student_diffusion_action_chunk/run_a_action_chunk_alignsel_len4_fa5_cb02_mix120k_seed42_15min`
  - log: `N/A` (direct command run without a persisted `train*.log`)
  - event: `outputs/XHandHoraScrewDriver_student_diffusion_action_chunk/run_a_action_chunk_alignsel_len4_fa5_cb02_mix120k_seed42_15min/stage2_diffusion_action_chunk_tb/events.out.tfevents.1774187181.wbz-ubuntu22-pc`
  - best_ckpt: `outputs/XHandHoraScrewDriver_student_diffusion_action_chunk/run_a_action_chunk_alignsel_len4_fa5_cb02_mix120k_seed42_15min/stage2_diffusion_action_chunk_nn/model_best_student_reward.ckpt`
  - deploy_ckpt_note: under nominal seed42/512 with `++train.ppo.action_chunk_len=4`, selector ranking becomes `model_best_student_reward (-0.328458) > model_last (-0.471724) > model_best_student (-0.893365)`; shorter chunk improves selector ordering but still does not beat `alignsel_v1` len8 best `model_best_student (-0.253256)`
- `diffusion_action_chunk_alignsel_probe_v1`
  - run_dir: `outputs/XHandHoraScrewDriver_student_diffusion_action_chunk/run_a_action_chunk_alignsel_probe_fa5_cb02_mix120k_seed42_15min`
  - log: `N/A` (direct command run without a persisted `train*.log`)
  - event: `outputs/XHandHoraScrewDriver_student_diffusion_action_chunk/run_a_action_chunk_alignsel_probe_fa5_cb02_mix120k_seed42_15min/stage2_diffusion_action_chunk_tb/events.out.tfevents.1774252709.wbz-ubuntu22-pc`
  - best_ckpt: `outputs/XHandHoraScrewDriver_student_diffusion_action_chunk/run_a_action_chunk_alignsel_probe_fa5_cb02_mix120k_seed42_15min/stage2_diffusion_action_chunk_nn/model_best_student.ckpt`
  - deploy_ckpt_note: adding `model_best_deploy_probe` did not beat the previous best selector; nominal seed42/512 ranking is `model_best_student (-0.253256) > model_last (-0.388507) > model_best_deploy_probe (-0.590261) > model_best_student_reward (-0.899636)`
- `padapt_trainrange_full`
  - run_dir: `outputs/XHandHoraScrewDriver_student_padapt_trainrange/run_a_trainrange_seed42_15min`
  - log: `outputs/XHandHoraScrewDriver_student_padapt_trainrange/run_a_trainrange_seed42_15min/train_900s.log`
  - event: `outputs/XHandHoraScrewDriver_student_padapt_trainrange/run_a_trainrange_seed42_15min/stage2_tb/events.out.tfevents.1774022154.wbz-ubuntu22-pc`
  - best_ckpt: `outputs/XHandHoraScrewDriver_student_padapt_trainrange/run_a_trainrange_seed42_15min/stage2_nn/model_best.ckpt`
- `padapt_trainrange_adapt_mu`
  - run_dir: `outputs/XHandHoraScrewDriver_student_padapt_trainrange/run_a_trainrange_adapt_mu_seed42_15min`
  - log: `outputs/XHandHoraScrewDriver_student_padapt_trainrange/run_a_trainrange_adapt_mu_seed42_15min/train_900s.log`
  - event: `outputs/XHandHoraScrewDriver_student_padapt_trainrange/run_a_trainrange_adapt_mu_seed42_15min/stage2_tb/events.out.tfevents.1774023374.wbz-ubuntu22-pc`
  - best_ckpt: `outputs/XHandHoraScrewDriver_student_padapt_trainrange/run_a_trainrange_adapt_mu_seed42_15min/stage2_nn/model_best.ckpt`
- `padapt_trainrange_adapt_actor`
  - run_dir: `outputs/XHandHoraScrewDriver_student_padapt_trainrange/run_a_trainrange_adapt_actor_seed42_15min`
  - log: `outputs/XHandHoraScrewDriver_student_padapt_trainrange/run_a_trainrange_adapt_actor_seed42_15min/train_900s.log`
  - event: `outputs/XHandHoraScrewDriver_student_padapt_trainrange/run_a_trainrange_adapt_actor_seed42_15min/stage2_tb/events.out.tfevents.1774024292.wbz-ubuntu22-pc`
  - best_ckpt: `outputs/XHandHoraScrewDriver_student_padapt_trainrange/run_a_trainrange_adapt_actor_seed42_15min/stage2_nn/model_best.ckpt`

## Notes

- `Current Best` comes from training stdout parsing, aligned with existing acceptance scripts.
- TensorBoard metrics are read from the latest event file under each run directory.
- If `done_rate/frame` or env-derived metrics are `N/A`, that run likely predates metric instrumentation.
