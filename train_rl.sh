#!/bin/bash
# GRPO RL fine-tuning of a pretrained SRM.
# Usage: bash train_rl.sh [experiment config name] [pretrained ckpt] [optional id] [hydra overrides]
# e.g.:  bash train_rl.sh ms1000_28 outputs/ms1000_28/paper/checkpoints/last.ckpt

config=$1
ckpt=$2
id=${3:-'null'}
params=${*:4}

if [ "$id" == 'null' ]; then
    id=$(date '+%Y-%m-%d_%H-%M-%S')
fi

DEBUG=false python -m src.main_rl +experiment=${config} \
    rl.pretrained_checkpoint=${ckpt} \
    hydra.run.dir=./outputs_rl/${config}/${id} \
    hydra.job.name=train_rl ${params}
