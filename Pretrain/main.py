import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
import sys
import argparse
import sklearn.metrics
import matplotlib
import pdb
import numpy as np 
import time
import random
import time
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from tqdm import tqdm
from tqdm import trange
from sklearn import metrics
from torch.utils import data
from collections import Counter
from transformers import AdamW, get_linear_schedule_with_warmup
from dataset import *
from model import *
from torch.cuda.amp import autocast, GradScaler


def log_loss(step_record, loss_record, name, model_name):
    if not os.path.exists("img"):
        os.mkdir("img")
    data1, data2 = loss_record[0], loss_record[1]
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('step')
    ax1.set_ylabel(name[0], color=color)
    ax1.plot(step_record, data1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel(name[1], color=color)  # we already handled the x-label with ax1
    ax2.plot(step_record, data2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(os.path.join("img", model_name + '-loss.png'))
    plt.close()


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

# def clip_padding(input, mask):
#     length = mask.sum(-1).max()
#     pdb.set_trace()
#     return input[:, :length], mask[:, :length]

def train(args, model, train_dataset):
    # total step
    step_tot = (len(train_dataset)  // args.gradient_accumulation_steps // args.batch_size_per_gpu // args.n_gpu) * args.max_epoch

    train_sampler = data.distributed.DistributedSampler(train_dataset) if args.local_rank != -1 else data.RandomSampler(train_dataset)
    params = {"batch_size": args.batch_size_per_gpu, "sampler": train_sampler}
    train_dataloader = data.DataLoader(train_dataset, **params)

    # optimizer
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=step_tot)

    # amp training
    scaler = GradScaler(enabled=True)

    # distributed training
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    print("Begin train...")
    print("We will train model in %d steps" % step_tot)
    global_step = 0
    loss_record = [[],[]]
    step_record = []
    for i in range(args.max_epoch):
        if args.local_rank != -1:
            train_sampler.set_epoch(i)
        for step, batch in enumerate(tqdm(train_dataloader)):
            if args.model == "KAST":
                inputs = {"input": batch[0].to(args.device), "mask": batch[1].to(args.device), "triple_mask":batch[2].to(args.device), "triple_label":batch[3].to(args.device)}
            elif args.model == "TEXT":
                inputs = {"input": batch[0].to(args.device), "mask": batch[1].to(args.device)}
            model.train()
            with autocast():
                m_loss, r_loss = model(**inputs)
                loss = m_loss + r_loss 
                loss = loss / args.gradient_accumulation_steps 
            scaler.scale(loss).backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if args.local_rank in [0, -1] and global_step % args.save_step == 0:
                    if not os.path.exists("ckpt"):
                        os.mkdir("ckpt")
                    if not os.path.exists("ckpt/"+args.save_dir):
                        os.mkdir("ckpt/"+args.save_dir)
                    ckpt = model.module.model.bert.state_dict()
                    torch.save(ckpt, os.path.join("ckpt/"+args.save_dir, "ckpt_of_step_"+str(global_step)))

                if args.local_rank in [0, -1] and global_step % args.record_step == 0:
                    step_record.append(global_step)
                    loss_record[0].append(m_loss)
                    loss_record[1].append(r_loss)
                
                if args.local_rank in [0, -1] and global_step % (args.record_step * 10) == 0:
                    log_loss(step_record, loss_record, ['mlm', 'bce'],  args.model)
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="baike")
    parser.add_argument("--cuda", dest="cuda", type=str, 
                        default="4", help="gpu id")

    parser.add_argument("--lr", dest="lr", type=float,
                        default=5e-5, help='learning rate')
    parser.add_argument("--batch_size_per_gpu", dest="batch_size_per_gpu", type=int, 
                        default=32, help="batch size per gpu")
    parser.add_argument("--gradient_accumulation_steps", dest="gradient_accumulation_steps", type=int,
                        default=1, help="gradient accumulation steps")
    parser.add_argument("--max_epoch", dest="max_epoch", type=int, 
                        default=3, help="max epoch number")
    
    parser.add_argument("--alpha", dest="alpha", type=float,
                        default=0.3, help="true entity(not `BLANK`) proportion")

    parser.add_argument("--model", dest="model", type=str,
                        default="", help="{MTB, CP}")
    parser.add_argument("--train_sample",action="store_true",
                        help="dynamic sample or not")
    parser.add_argument("--max_length", dest="max_length", type=int,
                        default=64, help="max sentence length")
    parser.add_argument("--bag_size", dest="bag_size", type=int,
                        default=2, help="bag size")
    parser.add_argument("--temperature", dest="temperature", type=float,
                        default=0.05, help="temperature for NTXent loss")
    parser.add_argument("--hidden_size", dest="hidden_size", type=int,
                        default=768, help="hidden size for mlp")
    
    parser.add_argument("--weight_decay", dest="weight_decay", type=float,
                        default=1e-5, help="weight decay")
    parser.add_argument("--adam_epsilon", dest="adam_epsilon", type=float,
                        default=1e-8, help="adam epsilon")
    parser.add_argument("--warmup_steps", dest="warmup_steps", type=int,
                        default=500, help="warmup steps")
    parser.add_argument("--max_grad_norm", dest="max_grad_norm", type=float,
                        default=1, help="max grad norm")
    
    parser.add_argument("--record_step", dest="record_step", type=int,
                        default=10, help="record loss step.")
    parser.add_argument("--save_step", dest="save_step", type=int, 
                        default=10000, help="step to save")
    parser.add_argument("--save_dir", dest="save_dir", type=str,
                        default="", help="ckpt dir to save")

    parser.add_argument("--p_neg", dest="p_neg", type=float,
                        default=0.5, help="Negative propality")
    parser.add_argument("--max_triples", dest="max_triples", type=int,
                        default=6, help="maximum number of triples")

    parser.add_argument("--seed", dest="seed", type=int,
                        default=42, help="seed for network")

    parser.add_argument("--local_rank", dest="local_rank", type=int,
                        default=-1, help="local rank")
    args = parser.parse_args()

    # print args
    print(args)
    # set cuda 
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda.split(",")[args.local_rank]
    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", rank=args.local_rank, init_method="tcp://127.0.0.1:6666", world_size=4)
    args.device = device
    args.n_gpu = len(args.cuda.split(","))
    set_seed(args)

    # log train
    if args.local_rank in [0, -1]:
        if not os.path.exists("log"):
            os.mkdir("log")
        with open("log/pretrain_log", 'a+') as f:
            f.write(str(time.ctime())+"\n")
            f.write(str(args)+"\n")
            f.write("----------------------------------------------------------------------------\n")

    # Model and dataset
    if args.model == "KAST":
        model = KAST(args)
        train_dataset = KASTDataset("data/KAST", args)
    elif args.model == "TEXT":
        model = TextPretrain(args)
        train_dataset = TextDataset("data/TEXT", args)
    else:
        raise Exception("No such mode.")

    # # load ckpt
    # path = "ckpt/text/ckpt_of_step_100000"
    # model.model.bert.load_state_dict(torch.load(path, map_location="cpu"))
    # print("-"*20, path, "-"*20)

    model.cuda(args.device)
    # Barrier to make sure all process train the model simultaneously.
    if args.local_rank != -1:
        torch.distributed.barrier()
    train(args, model, train_dataset)
