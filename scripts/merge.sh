#!/bin/bash
MODEL_CKPT="SparrowVQE-3b-stage2"

python -m llava_vqe.eval.model_merge \
    --model-path ./checkpoints/$MODEL_CKPT \
    --model-base checkpoints/base/phi-2 
