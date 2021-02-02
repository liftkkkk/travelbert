#!/bin/bash

# Python env interpreter for OIE task
python="miniconda3/envs/python3.6_cuda10.2/bin/python"

# Predicate and entity model
data_dir="data/SAOKE"
# data_dir="data/tourism-OIE"
output_dir="logs/debug"
predicate_output_dir=$output_dir/predicate
entity_output_dir=$output_dir/entity


### test
CUDA_VISIBLE_DEVICES=7 $python train_predicate.py --bert_model $predicate_output_dir/best_model --bert_weight_file logs/cae_more/predicate/best_model/pytorch_model.bin --data_dir $data_dir --eval_file "test.json" --do_eval --do_lower_case --max_seq_length 150 --output_dir $output_dir/joint

CUDA_VISIBLE_DEVICES=7 $python train_entity.py --bert_model $entity_output_dir/best_model --bert_weight_file logs/cae_more/entity/best_model/pytorch_model.bin --data_dir $output_dir/joint --eval_file "detailed_relation_predictions.json" --do_eval --do_lower_case --max_seq_length 150 --output_dir $output_dir/joint

$python eval_joint.py --gold_file $data_dir/test.json --pred_file $output_dir/joint/detailed_entity_predictions.json --output_dir $output_dir/joint
