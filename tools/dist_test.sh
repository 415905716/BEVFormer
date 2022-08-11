#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=2,3
CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-27666}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} --eval bbox 
