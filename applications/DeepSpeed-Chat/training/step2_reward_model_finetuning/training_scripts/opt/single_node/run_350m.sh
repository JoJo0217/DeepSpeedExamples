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

deepspeed main.py \
   --data_path local/jsonfile \
   --data_split 2,4,4 \
   --model_name_or_path EleutherAI/polyglot-ko-1.3b \
   --num_padding_at_beginning 1 \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --max_seq_len 512 \
   --learning_rate 5e-5 \
   --weight_decay 0.1 \
   --offload \
   --num_train_epochs 2 \
   --disable_dropout \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --enable_tensorboard \
   --tensorboard_path $OUTPUT \
   --output_dir $OUTPUT
