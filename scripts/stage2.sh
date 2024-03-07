#!/bin/bash

# if not use all GPUs 
# deepspeed --include localhost:0,1,2,3 --master_port 29600

deepspeed llava_vqe/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path checkpoints/base/phi-2 \
    --version phi2 \
    --data_path datasets/stage2.json \
    --image_folder datasets/instruction_tuned_images \
    --vision_tower checkpoints/base/siglip-so400m-patch14-384 \
    --pretrain_mm_mlp_adapter ./checkpoints/SparrowVQE-3b-stage1/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio square \
    --group_by_modality_length True \
    --bf16 False \
    --fp16 True \
    --output_dir ./checkpoints/SparrowVQE-3b-stage-2 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 3072 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none
