#!/bin/bash
MODEL_CKPT="SparrowVQE-3b-stage2-lora"
# MODEL_CKPT="imp-v1-3b-lora" # eval your own checkpoint

python -m llava_vqe.eval.model_merge \
    --model-path ./checkpoints/$MODEL_CKPT \
    --model-base checkpoints/base/phi-2 
