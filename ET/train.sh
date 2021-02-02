
# train ckpt keg30
CUDA_VISIBLE_DEVICES=1 python run_typing.py    --do_train   --do_lower_case   --data_dir data/tourism-FET   --bert_model data/bert/bert-chinese   --bert_weight_file pretrain/ckpt/text-baike-gl/ckpt_of_step_200000  --max_seq_length 256   --train_batch_size 256   --learning_rate 2e-5   --num_train_epochs 3.0   --gradient_accumulation_steps 16 --threshold 0.3 --loss_scale 128 --warmup_proportion 0.2  --output_dir logs/baike_gl_2


# eval ckpt
CUDA_VISIBLE_DEVICES=0 python eval_figer.py    --do_eval   --do_lower_case   --data_dir data/tourism-FET   --bert_model data/bert/bert-chinese   --max_seq_length 256   --eval_batch_size 128   --learning_rate 2e-5   --num_train_epochs 3.0   --gradient_accumulation_steps 1 --threshold 0.3 --loss_scale 128 --warmup_proportion 0.2   --output_dir logs/baike_gl_2


