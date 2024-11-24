#!/bin/bash

directory="/home/mingsen547896/bioaug4re/SpanBERT-main/data"

tacred_dir="/home/mingsen547896/bioaug4re/SpanBERT-main/results"

export CUDA_VISIBLE_DEVICES="5"

python3 code/run_tacred.py \
  --do_train \
  --do_eval \
  --eval_test \
  --data_dir $directory \
  --model bert-base-cased \
  --train_batch_size 32 \
  --eval_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 10 \
  --max_seq_length 128 \
  --output_dir $tacred_dir