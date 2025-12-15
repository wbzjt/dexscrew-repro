#!/bin/bash
GPUS=$1
SEED=$2
CACHE=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}

CUDA_VISIBLE_DEVICES=${GPUS} \
python train.py task=XHandPasiniScrewDriver headless=True seed=${SEED} \
experiment=rl \
train.algo=PPO \
task.env.reset_dist_threshold=0.1 \
wandb_activate=False \
train.ppo.output_name=XHandPasiniScrewDriver_teacher/${CACHE} \
${EXTRA_ARGS}
