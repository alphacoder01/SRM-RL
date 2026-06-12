#!/bin/bash
# GRPO RL fine-tuning of a pretrained SRM.
# Usage: bash train_rl.sh [experiment config name] [pretrained ckpt] [optional id] [hydra overrides]
# e.g.:  bash train_rl.sh ms1000_28 outputs/ms1000_28/paper/checkpoints/last.ckpt
# Automatically runs distributed (torchrun) on all available GPUs.
# Note: rl.rollout.num_conditions is per GPU, so the effective batch scales
# with the number of GPUs.

config=$1
ckpt=$2
id=${3:-'null'}
params=${*:4}

if [ "$id" == 'null' ]; then
    id=$(date '+%Y-%m-%d_%H-%M-%S')
fi

NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
if [ "${NUM_GPUS:-0}" -gt 1 ]; then
    LAUNCHER="torchrun --standalone --nproc_per_node=${NUM_GPUS} -m src.main_rl"
else
    LAUNCHER="python -m src.main_rl"
fi

DEBUG=false ${LAUNCHER} +experiment=${config} \
    rl.pretrained_checkpoint=${ckpt} \
    hydra.run.dir=./outputs_rl/${config}/${id} \
    hydra.job.name=train_rl ${params}
