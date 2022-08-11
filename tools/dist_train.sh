#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=3
CONFIG=$1
GPUS=$2
PORT=${PORT:-28509}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} --deterministic --resume 'work_dirs/bevformer_tiny/epoch_29.pth' 
