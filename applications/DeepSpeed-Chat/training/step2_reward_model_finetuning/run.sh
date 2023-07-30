#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=3
fi
mkdir -p $OUTPUT
rm -rf /tmp/data_files/
deepspeed main.py \
   --data_path local/jsonfile \
   --data_split 0,10,0 \
   --model_name_or_path EleutherAI/polyglot-ko-1.3b \
   --num_padding_at_beginning 0 \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --max_seq_len 1024 \
   --learning_rate 9e-6 \
   --weight_decay 0.1 \
   --num_train_epochs 1 \
   --disable_dropout \
   --gradient_accumulation_steps 16 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --lora_dim 128 \
   --lora_module_name query_key_value \
   --only_optimize_lora \
   --seed 1234 \
   --enable_tensorboard \
   --tensorboard_path $OUTPUT \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --output_dir $OUTPUT
