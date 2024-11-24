#!/bin/bash

set -e
set -x

dataset="TACRED"
directory="../data/${dataset}"
input_file="model_train5mix2_inference_ver_filter"

python self-training.py \
--dataset TACRED \
--directory $directory \
--input_file $input_file 