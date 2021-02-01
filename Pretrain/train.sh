CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4  main.py \
	--cuda 0,1,2,3 \
	--lr 3e-5 --batch_size_per_gpu 32 --max_epoch 100 \
	--gradient_accumulation_steps 2 \
	--max_length 420 \
	--save_step 10000 \
	--record_step 20 \
	--p_neg 0.5 \
	--model KAST \
	--save_dir kast-nokast \
