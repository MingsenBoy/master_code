#!/bin/bash

set -e
set -x


size=$1
dataset=$2
flair_batch_size=$3
SEED=$4
generation_normal=$5
generations_mixner=2

run="5-${size}-${SEED}-${shouldLinearizeAllWords}-tokenfix"

directory="../datasets-pre/${dataset}"
best_model="train_processed-${run}-final"
inference_file="train_processed"
template="tacred_template_relation"

output_file1="4_42_5_model_${generation_normal}_inference_ver"
output_file2="mix${generations_mixner}_model_inference_ver"
output_file3="model_train${generation_normal}mix${generations_mixner}_inference_ver"

python test-dynamic_inference.py \
--model $best_model \
--input_file $inference_file \
--triplet "verbalize" \
--directory $directory \
--topk 10 \
--num_of_sequences $generation_normal \
--max_length 200 \
--do_sample True \
--num_beams 5 \
--seed $SEED \
--template $template \
--output_file $output_file1

python test-dynamic_inference_mixner.py \
--model $best_model \
--input_file $inference_file \
--directory $directory \
--topk 10 \
--num_of_sequences $generations_mixner \
--max_length 200 \
--do_sample True \
--num_beams 5 \
--seed $SEED \
--template $template \
--output_file $output_file2 \

python mixner_fix.py \
--input_file $output_file2 \
--input_file2 $output_file1 \
--num_of_sequences $generations_mixner \
--num_of_sequences2 $generation_normal \
--directory $directory \
--output_file $output_file3 \

python mixner_filter.py \
--input_file $output_file3 \
--directory $directory \
--output_file ${output_file3}_filter
