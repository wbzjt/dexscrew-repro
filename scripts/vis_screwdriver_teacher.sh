#!/bin/bash
GPUS=$1
SEED=$2
CACHE=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

python train.py task=XHandHoraScrewDriver headless=False seed=${SEED} \
sim_device=cuda:${GPUS} rl_device=cuda:${GPUS} graphics_device_id=7 \
task.env.numEnvs=6 test=True \
train.algo=PPO \
task.env.reset_dist_threshold=0.12 \
# Make visualization deterministic (avoid randomness hiding the true policy behavior)
task.env.randomization.randomizePDGains=False \
task.env.randomization.action_noise_e_scale=0.0 \
task.env.randomization.action_noise_t_scale=0.0 \
task.env.randomization.obs_noise_e_scale=0.0 \
task.env.randomization.obs_noise_t_scale=0.0 \
task.env.randomization.noisy_rpy_scale=0.0 \
task.env.randomization.noisy_pos_scale=0.0 \
# Also disable external random forces for clean policy inspection
task.env.forceScale=0.0 \
task.env.randomForceProbScalar=0.0 \
wandb_activate=False \
"checkpoint=outputs/XHandHoraScrewDriver_teacher/${CACHE}/stage1_nn/best_reward_*.pth" \
${EXTRA_ARGS}
