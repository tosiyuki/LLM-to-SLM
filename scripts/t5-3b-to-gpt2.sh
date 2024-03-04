#!/bin/bash

python train.py \
    --model_name_or_path openai-community/gpt2 \
    --llm_encoder google-t5/t5-3b \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --output_dir ./output/checkpoints/t5-3b-to-gpt2