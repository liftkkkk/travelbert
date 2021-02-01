

CUDA_VISIBLE_DEVICES=6 python run_typing.py    --do_train   --do_lower_case   --data_dir /data1/lzh/data/ernie_data/FIGER   --bert_model /data1/lzh/data/bert/bert-base-uncased   --max_seq_length 256   --train_batch_size 2048   --learning_rate 2e-5   --num_train_epochs 3.0   --gradient_accumulation_steps 64 --threshold 0.3 --fp16 --loss_scale 128 --warmup_proportion 0.2  --output_dir logs/debug


CUDA_VISIBLE_DEVICES=6 python run_typing.py    --do_train   --do_lower_case   --data_dir /data/lzh/data/FIGER   --bert_model /data/lzh/data/bert-base-uncased   --max_seq_length 256   --train_batch_size 2048   --learning_rate 2e-5   --num_train_epochs 3.0   --gradient_accumulation_steps 64 --threshold 0.3 --loss_scale 128 --warmup_proportion 0.2  --output_dir logs/debug


# eval bert
CUDA_VISIBLE_DEVICES=6 python eval_figer.py    --do_eval   --do_lower_case   --data_dir /data/lzh/data/FIGER   --bert_model /data/lzh/data/bert-base-uncased   --max_seq_length 256   --train_batch_size 2048   --learning_rate 2e-5   --num_train_epochs 3.0   --output_dir logs/bert_2  --gradient_accumulation_steps 32 --threshold 0.3 --loss_scale 128 --warmup_proportion 0.2


# train bert on travel-FET
# TODO max_seq_length 256 -> 128
CUDA_VISIBLE_DEVICES=7 python run_typing.py    --do_train   --do_lower_case   --data_dir /data/lzh/data/FET   --bert_model /data/lzh/data/bert-base-uncased   --max_seq_length 256   --train_batch_size 256   --learning_rate 2e-5   --num_train_epochs 3.0   --gradient_accumulation_steps 8 --threshold 0.3 --loss_scale 128 --warmup_proportion 0.2  --output_dir logs/travel_2


# test bert on travel-FET
CUDA_VISIBLE_DEVICES=7 python eval_figer.py    --do_eval   --do_lower_case   --data_dir /data/lzh/data/FET   --bert_model /data/lzh/data/bert-base-uncased   --max_seq_length 512   --eval_batch_size 64   --learning_rate 2e-5   --num_train_epochs 3.0   --gradient_accumulation_steps 1 --threshold 0.3 --loss_scale 128 --warmup_proportion 0.2   --output_dir logs/travel_4

# train ckpt keg30
CUDA_VISIBLE_DEVICES=1 python run_typing.py    --do_train   --do_lower_case   --data_dir /data1/lzh/data/tourism-FET   --bert_model /data1/lzh/data/bert/bert-chinese   --bert_weight_file /data2/penghao/tourism-transformers/pretrain/ckpt/text-baike-gl/ckpt_of_step_200000  --max_seq_length 256   --train_batch_size 256   --learning_rate 2e-5   --num_train_epochs 3.0   --gradient_accumulation_steps 16 --threshold 0.3 --loss_scale 128 --warmup_proportion 0.2  --output_dir logs/baike_gl_2


# eval ckpt
CUDA_VISIBLE_DEVICES=0 python eval_figer.py    --do_eval   --do_lower_case   --data_dir /data1/lzh/data/tourism-FET   --bert_model /data1/lzh/data/bert/bert-chinese   --max_seq_length 256   --eval_batch_size 128   --learning_rate 2e-5   --num_train_epochs 3.0   --gradient_accumulation_steps 1 --threshold 0.3 --loss_scale 128 --warmup_proportion 0.2   --output_dir logs/baike_gl_2


