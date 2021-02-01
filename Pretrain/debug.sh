CUDA_VISIBLE_DEVICES=7 python  main.py \
	--cuda 7 \
	--lr 3e-5 --batch_size_per_gpu 4 --max_epoch 100 \
	--gradient_accumulation_steps 1 \
	--max_length 420 \
	--save_step 200000 \
	--record_step 100 \
	--p_neg 0.5 \
	--model KAST \
	--save_dir kast \
