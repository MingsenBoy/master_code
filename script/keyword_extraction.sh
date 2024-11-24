#!/bin/bash

set -e
set -x

sample_size=$1
dataset=$2
data_type=$3

file_path="../datasets/${dataset}"
output_path="../datasets-pre/${dataset}"

python3 keyword_extraction.py \
--sample_size $sample_size \
--data_type $data_type \
--file_path $file_path \
--output_path $output_path 