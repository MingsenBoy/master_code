#!/bin/bash

set -e
set -x

size=$1
dataset=$2
flair_batch_size=$3
SEED=$4
generations=5
generation_normal=$5

run="5-${size}-${SEED}-${shouldLinearizeAllWords}-tokenfix"

directory="../datasets-pre/${dataset}"
best_model="train_processed-${run}-final"
inference_file="train_processed"

output_file1="4_42_5_model_${generation_normal}_inference_ver"


#　triplet　參數　deafult verbalize(or triple)
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
--output_file $output_file1

python generation_fix.py \
--input_file $output_file1 \
--num_of_sequences $generation_normal \
--directory $directory \
--output_file "model_inference_ver_${generation_normal}" \

python generation_filter.py \
--input_file "model_inference_ver_${generation_normal}" \
--directory $directory \
--output_file "model_inference_ver_${generation_normal}_filter"