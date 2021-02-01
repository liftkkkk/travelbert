python -m torch.distributed.launch --nproc_per_node 4  main.py \
	--cuda 4,5,6,7 \
	--lr 3e-5 --batch_size_per_gpu 32 --max_epoch 100 \
	--gradient_accumulation_steps 16 \
	--max_length 128 \
	--save_step 500 \
        --bag_size 16 \
	--model CLF \
	--save_dir CLF \
