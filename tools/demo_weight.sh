#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=2,3

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

python tools/demo.py 
