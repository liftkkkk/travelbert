#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=4 TMPDIR=./cache python run_BERTRetrieval.py \
    --bert_config_file data/tourism-bert/bert_config.json \
    --vocab_file data/tourism-bert/vocab.txt \
    --init_checkpoint data/tourism-bert/ckpt_more_cae_pytorch.bin \
    --do_train \
    --train_batch_size 16 \
    --learning_rate 5e-5 \
    --cache \
    --cache_dir ./cache/travel \
    --datapath ../data/travel \
    --num_train_epochs 3.0 \
    --output_dir ./output/travel \
    --model_dir ./model/travel \
    --gradient_accumulation_steps 8

