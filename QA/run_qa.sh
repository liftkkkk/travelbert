export TASK_NAME=qnli
export SEED=3

CUDA_VISIBLE_DEVICES=5,6,4 python3 run_qa.py \
  --model_name_or_path bert-base-chinese \
  --task_name $TASK_NAME \
  --seed $SEED \
  --overwrite_cache \
  --do_train \
  --do_eval \
  --do_predict \
  --max_seq_length 200 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --overwrite_output_dir \
  --score_file base_score.pkl \
  --output_dir ./qnli-base/
