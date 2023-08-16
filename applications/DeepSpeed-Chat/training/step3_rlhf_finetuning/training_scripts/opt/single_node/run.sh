#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
ACTOR_MODEL_PATH=$1
CRITIC_MODEL_PATH=$2
ACTOR_ZERO_STAGE=$3
CRITIC_ZERO_STAGE=$4
OUTPUT=$5
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
if [ "$ACTOR_ZERO_STAGE" == "" ]; then
    ACTOR_ZERO_STAGE=3
fi
if [ "$CRITIC_ZERO_STAGE" == "" ]; then
    CRITIC_ZERO_STAGE=3
fi
mkdir -p $OUTPUT

Num_Padding_at_Beginning=1 # this is model related

Actor_Lr=5e-4
Critic_Lr=5e-6

deepspeed --master_port 12346 main.py \
   --data_path local/jsonfile \
   --data_split 2,4,4 \
   --actor_model_name_or_path $ACTOR_MODEL_PATH \
   --critic_model_name_or_path $CRITIC_MODEL_PATH \
   --num_padding_at_beginning 1 \
<<<<<<<< HEAD:applications/DeepSpeed-Chat/training/step3_rlhf_finetuning/training_scripts/opt/single_node/run.sh
   --per_device_train_batch_size 1 \
   --per_device_mini_train_batch_size 1 \
   --generation_batch_numbers 1 \
========
   --per_device_generation_batch_size 4 \
   --per_device_training_batch_size 4 \
   --generation_batches 1 \
>>>>>>>> upstream/master:applications/DeepSpeed-Chat/training/step3_rlhf_finetuning/training_scripts/opt/multi_node/run_66b.sh
   --ppo_epochs 1 \
   --max_answer_seq_len 256 \
   --max_prompt_seq_len 256 \
   --actor_learning_rate ${Actor_Lr} \
   --critic_learning_rate ${Critic_Lr} \
   --actor_weight_decay 0.1 \
   --critic_weight_decay 0.1 \
   --num_train_epochs 1 \
   --lr_scheduler_type cosine \
   --gradient_accumulation_steps 1 \
   --num_warmup_steps 100 \
   --deepspeed --seed 1234 \
   --enable_hybrid_engine \
   --inference_tp_size 8 \
   --tp_gather_partition_size 4 \
   --actor_zero_stage $ACTOR_ZERO_STAGE \
   --critic_zero_stage $CRITIC_ZERO_STAGE \
   --actor_gradient_checkpointing \
   --disable_actor_dropout \
   --actor_lora_dim 128 \
   --actor_lora_module_name decoder.layers. \
   --output_dir $OUTPUT \
    &> $OUTPUT/training.log
