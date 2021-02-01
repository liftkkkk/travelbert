#!/usr/bin/env bash

python="/data1/lzh/miniconda3/envs/kdconv/bin/python"

CUDA_VISIBLE_DEVICES=4 TMPDIR=./cache $python run_BERTRetrieval.py \
    --bert_config_file /data1/lzh/data/chinese_wwm_pytorch/bert_config.json \
    --vocab_file /data1/lzh/data/chinese_wwm_pytorch/vocab.txt \
    --init_checkpoint /data1/lzh/data/chinese_wwm_pytorch/pytorch_model.bin \
    --do_predict \
    --predict_batch_size 16 \
    --cache \
    --num_train_epochs 3.0 \
    --cache_dir ./cache/travel \
    --datapath ../data/travel \
    --output_dir ./output/travel \
    --model_dir ./model/travel
