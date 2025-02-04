#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Note that usually LoRA needs to use larger learning rate
OUTPUT_PATH=./output
mkdir -p $OUTPUT_PATH
rm -rf /tmp/data_files/
deepspeed main.py \
   --data_path local/jsonfile \
   --data_split 10,0,0 \
   --model_name_or_path EleutherAI/polyglot-ko-1.3b\
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --max_seq_len 1024 \
   --learning_rate 1e-3 \
   --weight_decay 0.1 \
   --num_train_epochs 2 \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --zero_stage 2 \
   --offload \
   --lora_dim 128 \
   --lora_module_name query_key_value \
   --only_optimize_lora \
   --enable_tensorboard \
   --tensorboard_path $OUTPUT_PATH \
   --deepspeed \
   --output_dir $OUTPUT_PATH \
