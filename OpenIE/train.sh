#!/bin/bash

# Python env interpreter for OIE task
python="miniconda3/envs/python3.6_cuda10.2/bin/python"

# Model config
bert_ckpt="pretrain/ckpt/ckpt_of_step_8000"  # raw pretrained bert checkpoint
# bert_weight_file=${bert_ckpt}_pytorch.bin  # converted weight for input
bert_model_dir="data/tourism-bert"  # bert model dir with bert_config.json / vocab.txt

# Dataset
data_dir="data/SAOKE"
# data_dir="data/tourism-OIE"

# Checkpoint dir
output_dir="logs/debug"
predicate_output_dir=$output_dir/predicate
entity_output_dir=$output_dir/entity

CUDA=2

## train
# echo "$python weight_name_transform.py $bert_ckpt"
# $python weight_name_transform.py $bert_ckpt

echo "$python train_predicate.py --bert_model $bert_model_dir --bert_weight_file $bert_ckpt --data_dir $data_dir --do_train --do_eval --eval_test --do_lower_case --max_seq_length 150 --seed 3 --num_train_epochs 2 --learning_rate 2e-5 --output_dir $predicate_output_dir"
$python train_predicate.py \
    --bert_model $bert_model_dir \
    --bert_weight_file $bert_ckpt \
    --data_dir $data_dir \
    --do_train \
    --do_eval \
    --eval_test \
    --do_lower_case \
    --max_seq_length 150 \
    --seed 3 \
    --num_train_epochs 2 \
    --learning_rate 2e-5 \
    --output_dir $predicate_output_dir

echo "$python train_entity.py --bert_model $bert_model_dir --bert_weight_file $bert_ckpt --data_dir $data_dir --do_train --do_eval --eval_test --do_lower_case --max_seq_length 150 --seed 3 --num_train_epochs 3 --learning_rate 3e-5 --output_dir $entity_output_dir"
$python train_entity.py \
    --bert_model $bert_model_dir \
    --bert_weight_file $bert_ckpt \
    --data_dir $data_dir \
    --do_train \
    --do_eval \
    --eval_test \
    --do_lower_case \
    --max_seq_length 150 \
    --seed 3 \
    --num_train_epochs 3 \
    --learning_rate 3e-5 \
    --output_dir $entity_output_dir

