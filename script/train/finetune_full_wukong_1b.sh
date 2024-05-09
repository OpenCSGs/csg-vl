#!/bin/bash

MODEL_TYPE=wukong
MODEL_NAME_OR_PATH=/path/to/base_llm_model
VISION_TOWER=/path/to/siglip-so400m-patch14-384
DATA_PATH=/path/to/data.json
IMAGE_FLODER=/path/to/images
PRETRAIN_DIR=csg-vl-$MODEL_TYPE-pretrain
OUTPUT_DIR=csg-vl-$MODEL_TYPE-finetune-full

mkdir -p ./checkpoints-$MODEL_TYPE/$OUTPUT_DIR

deepspeed --master_port 29600 --num_nodes=1 --num_gpus=8 \
    csg_vl/train/train.py \
    --deepspeed ./script/deepspeed/zero3.json \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --model_type $MODEL_TYPE \
    --version wukong \
    --data_path $DATA_PATH \
    --image_folder $IMAGE_FLODER \
    --vision_tower $VISION_TOWER \
    --pretrain_mm_mlp_adapter /path/to/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir ./checkpoints-$MODEL_TYPE/$OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none | tee 2>&1 ./checkpoints-$MODEL_TYPE/$OUTPUT_DIR/log.txt
