## The relevant files are currently on a shared Google
## drive at https://drive.google.com/drive/folders/1kC0I2UGl2ltrluI9NqDjaQJGw5iliw_J
## Monitor for changes and eventually migrate to nlp dataset
#curl -L 'https://drive.google.com/uc?export=download&id=1Jjhbal535VVz2ap4v4r_rN1UEHTdLK5P' \
#| grep -v "^#" | cut -f 2,3 | tr '\t' ' ' > train.txt.tmp
#curl -L 'https://drive.google.com/uc?export=download&id=1ZfRcQThdtAR5PPRjIDtrVP7BtXSCUBbm' \
#| grep -v "^#" | cut -f 2,3 | tr '\t' ' ' > dev.txt.tmp
#curl -L 'https://drive.google.com/uc?export=download&id=1u9mb7kNJHWQCWyweMDRMuTFoOHOfeBTH' \
#| grep -v "^#" | cut -f 2,3 | tr '\t' ' ' > test.txt.tmp

export MAX_LENGTH=128
export BERT_MODEL=bert-base-chinese
python3 scripts/preprocess.py /data1/anonymous/github/tner/bio.txtl.train $BERT_MODEL $MAX_LENGTH > /data1/anonymous/github/tner/train.txt
python3 scripts/preprocess.py /data1/anonymous/github/tner/bio.txtl.dev $BERT_MODEL $MAX_LENGTH > /data1/anonymous/github/tner/dev.txt
python3 scripts/preprocess.py /data1/anonymous/github/tner/bio.txtl.test $BERT_MODEL $MAX_LENGTH > /data1/anonymous/github/tner/test.txt
cat /data1/anonymous/github/tner/train.txt /data1/anonymous/github/tner/dev.txt /data1/anonymous/github/tner/test.txt | cut -d " " -f 2 | grep -v "^$"| sort | uniq > /data1/anonymous/github/tner/labels.txt
export OUTPUT_DIR=wklm-model
export BATCH_SIZE=64
export NUM_EPOCHS=40
export SAVE_STEPS=100
export SEED=3

CUDA_VISIBLE_DEVICES=1,2 python3 run_wklm.py \
--task_type NER \
--data_dir /data1/anonymous/github/tner \
--labels /data1/anonymous/github/tner/labels.txt \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--per_device_train_batch_size $BATCH_SIZE \
--save_steps $SAVE_STEPS \
--seed $SEED \
--learning_rate 5e-5 \
--save_total_limit 100 \
--overwrite_output_dir \
--do_train \
--do_eval \
--do_predict
