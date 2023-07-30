#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Add the path to the finetuned model
rm -rf /tmp/data_files/
deepspeed  test.py \
    --model_name_or_path /content/drive/MyDrive/rlhf/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step2_reward_model_finetuning/output_fullfine0.7 \
    --num_padding_at_beginning 0 \
    --per_device_eval_batch_size 2 \
    --max_seq_len 1024 \
    --data_path local/jsonfile \
    --data_split 0,10,0 \
