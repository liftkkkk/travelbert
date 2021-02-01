### Pretrain code for tourism-transformers


#### Update on Oct 29
We pretrain the model using contrastive loss and mlm loss, and we use a fc layer before calculate similarity between text segment(inspired by SimCLR). And we **dynamically sample** training samples.
<img src="img/CP-loss_curve.png">

The pre-train log is:
```
Shell
Wed Oct 28 00:36:35 2020
Namespace(adam_epsilon=1e-08, alpha=0.3, bag_size=2, batch_size_per_gpu=32, cuda='4,5,6,7', device=device(type='cuda', index=0), gradient_accumulation_steps=16, hidden_size=768, local_rank=0, lr=3e-05, max_epoch=40, max_grad_norm=1, max_length=64, model='CP', n_gpu=4, save_dir='CP', save_step=500, seed=42, temperature=0.05, train_sample=False, warmup_steps=500, weight_decay=1e-05)
```



#### Update on Oct 20
We pretrain the model using contrastive loss and mlm loss. The loss in pre-training step is below.
<img src="img/loss_curve.png">

The pre-train log is:
```Shell
Tue Oct 20 00:48:10 2020
Namespace(adam_epsilon=1e-08, alpha=0.3, bag_size=2, batch_size_per_gpu=32, cuda='4,5,6,7', device=device(type='cuda', index=0), gradient_accumulation_steps=16, hidden_size=768, local_rank=0, lr=3e-05, max_epoch=40, max_grad_norm=1, max_length=64, model='', n_gpu=4, save_dir='baike', save_step=500, seed=42, temperature=0.05, train_sample=False, warmup_steps=500, weight_decay=1e-05)
```
