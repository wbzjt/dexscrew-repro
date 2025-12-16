#!/bin/bash
GPUS=$1
SEED=$2
CACHE=$3
HEADLESS=${4:-True}

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:4:$len}

NUM_ENVS_ARG=""
if [ "$HEADLESS" = "False" ] || [ "$HEADLESS" = "false" ]; then
    # Adjust minibatch size to match reduced batch size (16 envs * 12 horizon = 192)
    NUM_ENVS_ARG="task.env.numEnvs=1 train.ppo.minibatch_size=12"
fi

CUDA_VISIBLE_DEVICES=${GPUS} \
python train.py task=XHandPasiniScrewDriver headless=${HEADLESS} seed=${SEED} \
experiment=rl \
train.algo=PPO \
task.env.reset_dist_threshold=5 \
wandb_activate=False \
train.ppo.output_name=XHandPasiniScrewDriver_teacher/${CACHE} \
${NUM_ENVS_ARG} \
${EXTRA_ARGS}
