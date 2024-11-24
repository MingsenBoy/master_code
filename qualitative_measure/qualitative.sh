#!/bin/bash

set -e
set -x

dataset=$1
data_name=$2

directory="../datasets-pre/${dataset}"

input_file="${directory}/${data_name}"

python text_diversity.py \
--input_file $input_file \

python text_perplexity.py \
--input_file $input_file 