from __future__ import print_function

import argparse
import collections
import logging
import json
import math
import os
import random
import pickle
from tqdm import tqdm, trange
import re

import numpy as np
import torch
from torch import nn
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel, BertConfig
from pytorch_pretrained_bert.optimization import BertAdam
from myCoTK.dataloader import MyBERTRetrieval
from utils import try_cache
import jieba

from utils.MyMetrics import MyMetrics

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


# pylint: disable=W0221
class BERTRetrieval(BertPreTrainedModel):
    def __init__(self, num_choices, bert_config_file):
        self.num_choices = num_choices
        bert_config = BertConfig.from_json_file(bert_config_file)
        BertPreTrainedModel.__init__(self, bert_config)
        self.bert = BertModel(bert_config)
        self.dropout = nn.Dropout(bert_config.hidden_dropout_prob)
        self.classifier = nn.Linear(bert_config.hidden_size, 1)
        self.activation = nn.Sigmoid()
        self.apply(self.init_bert_weights)

    def forward(self, data, labels=None):
        input_ids, attention_mask, token_type_ids = data['input_ids'], data['input_mask'], data['segment_ids']
        hidden_states, output_cls = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pair_output = hidden_states[:, 0]
        if labels is not None:
            pair_output = self.dropout(pair_output)
        logits = self.classifier(pair_output)
        prob = self.activation(logits.view(-1))

        if labels is not None:
            loss = torch.mean(-labels * torch.log(prob) - (1 - labels) * torch.log(1 - prob))
            return loss
        else:
            prob = prob.view(-1, self.num_choices)
            pred = torch.argmax(prob.view(-1, self.num_choices), dim=1)
            return prob.cpu().numpy(), pred


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

def preprocess_batch(data, device=None):
    for key in ['input_ids', 'input_mask', 'segment_ids', 'labels']:
        if key != 'labels':
            data[key] = torch.LongTensor(data[key])
        else:
            data[key] = torch.FloatTensor(data[key])

        if device is not None:
            data[key] = data[key].to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_config_file", default="chinese_wwm_pytorch/bert_config.json",
                        type=str, help="The config json file corresponding to the pre-trained BERT model. "
                             "This specifies the model architecture.")
    parser.add_argument("--vocab_file", default="chinese_wwm_pytorch/vocab.txt",
                        type=str, help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--init_checkpoint", default="chinese_wwm_pytorch/pytorch_model.bin",
                        type=str, help="Initial checkpoint (usually from a pre-trained BERT model).")

    ## Required parameters
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--model_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--cache_dir", default=None, type=str, required=True, help="Whether to run training.")

    ## Other parameters
    parser.add_argument('--name', type=str, default='BERTRetrieval', help='name of model')

    parser.add_argument('--dataset', type=str, default='MyBERTRetrieval', help='Dataloader class. Default: OpenSubtitles')
    parser.add_argument('--datapath', type=str, default='resources://OpenSubtitles',
                        help='Directory for data set. Default: resources://OpenSubtitles')
    parser.add_argument("--num_choices", default=10, type=int,
                        help="the number of retrieval options")

    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_predict", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--cache", action='store_true', help="Whether to run training.")

    parser.add_argument("--train_batch_size", default=8, type=int, help="Total batch size for training.")
    parser.add_argument("--predict_batch_size", default=16, type=int, help="Total batch size for predictions.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% "
                             "of training.")

    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--do_lower_case",
                        default=True,
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")

    args = parser.parse_args()


    if not args.do_train and not args.do_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir, exist_ok=True)

    data_class = MyBERTRetrieval

    def load_dataset(file_id, bert_vocab_name, do_lower_case, num_choices):
        dm = data_class(file_id=file_id, bert_vocab_name=bert_vocab_name, do_lower_case=do_lower_case, num_choices=num_choices)
        return dm

    if args.cache:
        dataManager = try_cache(load_dataset, (args.datapath, args.vocab_file, args.do_lower_case, args.num_choices),
                                args.cache_dir,
                                data_class.__name__)
    else:
        dataManager = load_dataset(file_id=args.datapath, bert_vocab_name=args.vocab_file, do_lower_case=args.do_lower_case,
                                   num_choices=args.num_choices)

    if not os.path.exists(os.path.join(args.datapath, 'test_distractors.json')):
        test_distractors = dataManager.data['test']['resp_distractors']
        with open(os.path.join(args.datapath, 'test_distractors.json'), 'w') as f:
            json.dump(test_distractors, f, ensure_ascii=False, indent=4)

    if args.do_train:

        if not args.no_cuda:
            if not "CUDA_VISIBLE_DEVICES" in os.environ:
                os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'

        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
        logger.info("device: {} n_gpu: {}".format(device, n_gpu))

        if args.gradient_accumulation_steps < 1:
            raise ValueError(
                "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(args.gradient_accumulation_steps))

        args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

        # random.seed(args.seed)
        # np.random.seed(args.seed)
        # torch.manual_seed(args.seed)
        # if n_gpu > 0:
        #     torch.cuda.manual_seed_all(args.seed)

        num_train_steps = None
        logger.info("train examples {}".format(len(dataManager.data['train']['resp'])))
        num_train_steps = int(len(dataManager.data['train'][
                                      'resp']) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

        # Prepare model
        '''
        if os.path.exists(output_model_file):
            model_state_dict = torch.load(output_model_file)
            model = BERTRetrieval(num_choices=args.num_choices, bert_config_file=args.bert_config_file)
            model.load_state_dict(model_state_dict)
        '''
        model = BERTRetrieval(num_choices=args.num_choices, bert_config_file=args.bert_config_file)
        if args.init_checkpoint is not None:
            logger.info('load bert weight')
            state_dict = torch.load(args.init_checkpoint, map_location='cpu')
            missing_keys = []
            unexpected_keys = []
            error_msgs = []
            # copy state_dict so _load_from_state_dict can modify it
            metadata = getattr(state_dict, '_metadata', None)
            state_dict = state_dict.copy()
            if metadata is not None:
                state_dict._metadata = metadata

            def load(module, prefix=''):
                local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})

                module._load_from_state_dict(state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys,
                                             error_msgs)
                for name, child in module._modules.items():
                    # logger.info("name {} chile {}".format(name,child))
                    if child is not None:
                        load(child, prefix + name + '.')

            load(model, prefix='' if hasattr(model, 'bert') else 'bert.')
            logger.info("missing keys:{}".format(missing_keys))
            logger.info('unexpected keys:{}'.format(unexpected_keys))
            logger.info('error msgs:{}'.format(error_msgs))

        model.to(device)
        model = torch.nn.DataParallel(model)

        # Prepare optimizer
        param_optimizer = list(model.named_parameters())

        # hack to remove pooler, which is not used
        # thus it produce None grad that break apex
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

        t_total = num_train_steps
        optimizer = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate)
        global_step = 0

        logger.info("***** Running training *****")
        logger.info("  Num post-response pairs = %d", len(dataManager.data['train']['resp']))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        model.train()
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            model.zero_grad()
            dataManager.restart(key='train', batch_size=args.train_batch_size)
            data = dataManager.get_next_batch(key='train')
            step = 0
            loss_value = 0
            while data is not None:
                if n_gpu == 1:
                    preprocess_batch(data, device) # multi-gpu does scattering it-self
                else:
                    preprocess_batch(data)
                loss = model(data, data['labels'])
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss_value += loss.cpu().item() * args.gradient_accumulation_steps
                loss.backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = args.learning_rate * warmup_linear(global_step / t_total, args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                if (step + 1) % (args.gradient_accumulation_steps * 100) == 0:
                    logger.info("step: %d, loss: %f" % (step + 1, loss_value))
                    loss_value = 0

                step += 1
                data = dataManager.get_next_batch(key='train')

            output_model_file = os.path.join(args.model_dir, "pytorch_model.%d.%d.bin" % (int(args.num_train_epochs), epoch + 1))

            # Save a trained model
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            torch.save(model_to_save.state_dict(), output_model_file)

    # Load a trained model that you have fine-tuned

    if args.do_predict:

        total_epoch = int(args.num_train_epochs)
        # chosen_epoch = 5  # int(args.num_train_epochs)
        chosen_epoch = int(args.num_train_epochs)

        if not args.no_cuda:
            if not "CUDA_VISIBLE_DEVICES" in os.environ:
                os.environ["CUDA_VISIBLE_DEVICES"] = '3'

        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()

        random.seed(args.seed)

        output_model_file = os.path.join(args.model_dir, "pytorch_model.%d.%d.bin" % (total_epoch, chosen_epoch))

        model_state_dict = torch.load(output_model_file)
        model = BERTRetrieval(num_choices=args.num_choices, bert_config_file=args.bert_config_file)
        model.load_state_dict(model_state_dict)
        model.to(device)

        if n_gpu > 1:
            model = torch.nn.DataParallel(model)

        metric = MyMetrics()

        logger.info("***** Running testing *****")
        logger.info("  Num post-response pairs = %d", len(dataManager.data['test']['resp']))
        logger.info("  Batch size = %d", args.predict_batch_size)

        model.eval()
        logger.info("Start evaluating")
        dataManager.restart(key='test', batch_size=args.predict_batch_size, shuffle=False)
        data = dataManager.get_next_batch(key='test')

        gens = []
        gold = []
        choices = []

        hits = {1: [0, 0], 3:[0, 0]}
        while data is not None:
            preprocess_batch(data, device)
            truth_response, can_responses = data['resp'], data['can_resps']

            with torch.no_grad():
                prob, pred = model(data)

            assert len(pred) == len(truth_response)
            assert len(pred) == len(can_responses)
            assert len(can_responses[0]) == args.num_choices

            for truth, pd, cans, prb in zip(truth_response, pred, can_responses, prob):
                metric.forword(truth, cans[pd])

                gold.append(truth)
                gens.append(cans[pd])
                choices.append(cans)

                idx = cans.index(truth)
                p_sort = np.argsort(prb)
                for key, count in hits.items():
                    if idx in p_sort[-key:]:
                        count[0] += 1
                    count[1] += 1

            data = dataManager.get_next_batch(key='test')

        result = metric.close()
        result.update({'hits@%d' % key: value[0] / value[1] for key, value in hits.items()})

        output_prediction_file = args.output_dir + "/%s_%s.%d.%d.txt" % (args.name, "test", total_epoch, chosen_epoch)
        with open(output_prediction_file, "w") as f:
            print("Test Result:")
            res_print = list(result.items())
            res_print.sort(key=lambda x: x[0])
            for key, value in res_print:
                if isinstance(value, float):
                    print("\t%s:\t%f" % (key, value))
                    f.write("%s:\t%f\n" % (key, value))
            f.write('\n')

            for resp, gen, options in zip(gold, gens, choices):
                f.write("resp:\t%s\n" % resp)
                f.write("gen:\t%s\n\n" % gen)
                for i, option in enumerate(options):
                    f.write("candidate %d:\t%s\n" % (i, option))
                f.write("\n")


if __name__ == "__main__":
    main()
