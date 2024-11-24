#!/bin/bash

set -e
set -x

size=$1
dataset=$2
flair_batch_size=$3
SEED=$4
generations=5

directory="../datasets-pre/${dataset}"

attn_train="train_processed"
attn_dev="dev_processed"
template_name="tacred_template_relation"
export CUDA_VISIBLE_DEVICES="0"

run="5-${size}-${SEED}-${shouldLinearizeAllWords}-tokenfix"


python3 pretrain_dynamic_verbalize.py \
--directory $directory \
--train_file $attn_train \
--dev_file $attn_dev \
--epochs 10 \
--batch_size 16 \
--file_name $run \
--seed $SEED \
--template $template_name
