import argparse
import collections
import json
import logging
import copy
import math
import os
import random
import sys
import timeit
from io import open
from collections import defaultdict, OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertConfig
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from pytorch_pretrained_bert.tokenization import BertTokenizer
from modeling import PredicateExtractionModel, GraphPredicateExtractionModel

from weight_name_transform import state_dict_normalize
from preprocess import DataProcessor
from feature_converter import convert_examples_to_features
from metric import predicate_evaluate

from squad_metrics import (
    compute_predictions_logits,
    squad_evaluate,
)

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

logger = logging.getLogger(__name__)


class SquadResult(object):
    """
    Constructs a SquadResult which can be used to evaluate a model's output on the SQuAD dataset.

    Args:
        unique_id: The unique identifier corresponding to that example.
        start_logits: The logits corresponding to the start of the answer
        end_logits: The logits corresponding to the end of the answer
    """

    def __init__(self, unique_id, start_logits, end_logits, start_top_index=None, end_top_index=None, cls_logits=None):
        self.start_logits = start_logits
        self.end_logits = end_logits
        self.unique_id = unique_id

        if start_top_index:
            self.start_top_index = start_top_index
            self.end_top_index = end_top_index
            self.cls_logits = cls_logits


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def get_linguistic_tensors(features):
    all_segment_token_span = torch.tensor([f.segment_token_span for f in features], dtype=torch.long)
    all_segment_mask = torch.tensor([f.segment_mask for f in features], dtype=torch.long)
    all_head = torch.tensor([f.head for f in features], dtype=torch.long)
    all_head_mask = torch.tensor([f.head_mask for f in features], dtype=torch.long)

    all_input_tensors = [all_segment_token_span, all_segment_mask, all_head, all_head_mask]
    return all_input_tensors


def train(args, train_examples, model, tokenizer, eval_examples):
    """ train model, note that no fp16 support during train """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    train_features = convert_examples_to_features(
        examples=train_examples,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        max_segment_length=args.max_segment_length,
        feature_type="relation")

    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_pred_start_positions = torch.tensor([f.pred_start_positions for f in train_features], dtype=torch.float32)
    all_pred_end_positions = torch.tensor([f.pred_end_positions for f in train_features], dtype=torch.float32)
    # all_linguistic_tensors = get_linguistic_tensors(train_features)
    all_linguistic_tensors = {}

    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                               all_pred_start_positions, all_pred_end_positions, *all_linguistic_tensors)
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    num_train_optimization_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())

    # hack to remove pooler, which is not used
    # thus it produce None grad that break apex
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_optimization_steps)

    # start training
    logger.info("***** Running training *****")
    logger.info("  Num orig examples = %d", len(train_examples))
    logger.info("  Num split examples = %d", len(train_features))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)

    global_step = 1
    best_results = None
    tr_loss, logging_loss = 0.0, 0.0

    optimizer.zero_grad()
    for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            model.train()
            if args.n_gpu == 1:
                batch = tuple(t.to(args.device) for t in batch)  # multi-gpu does scattering it-self
            input_ids, input_mask, segment_ids, pred_start_positions, pred_end_positions = batch
            loss = model(input_ids, segment_ids, input_mask, pred_start_positions, pred_end_positions)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    results = evaluate(args, eval_examples, model, tokenizer)[0]
                    logger.info("global step: %d", global_step)
                    for key, value in results.items():
                        logger.info("eval_{}: {}".format(key, value))
                    logger.info("training loss: %f", (tr_loss - logging_loss) / args.logging_steps)
                    logging_loss = tr_loss

                    # save best model
                    if best_results is None or best_results['f1'] < results['f1']:
                        # saving to best model dir for load convenience
                        output_dir = os.path.join(args.output_dir, "best_model")
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        best_results = results
                        with open(os.path.join(output_dir, "eval_results.json"), "w") as writer:
                            writer.write(json.dumps(best_results, indent=4))
                        logger.info("get best eval results -  f1: %f", best_results["f1"])

                        model_to_save = model.module if hasattr(model, "module") else model
                        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
                        output_config_file = os.path.join(output_dir, CONFIG_NAME)

                        torch.save(model_to_save.state_dict(), output_model_file)
                        model_to_save.config.to_json_file(output_config_file)
                        tokenizer.save_vocabulary(output_dir)
                        logger.info("Save best model to %s dir", output_dir)

    return global_step, tr_loss / global_step


def evaluate(args, examples, model, tokenizer, prefix="", save_predictions=True):
    features = convert_examples_to_features(examples, tokenizer, args.max_seq_length, args.max_segment_length, feature_type="relation")

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    # all_linguistic_tensors = get_linguistic_tensors(features)
    all_linguistic_tensors = {}

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index, *all_linguistic_tensors)

    if save_predictions and not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    all_results = []
    start_time = timeit.default_timer()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2]
            }
            example_indices = batch[3]

            outputs = model(**inputs)

        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)

            output = [to_list(output[i]) for output in outputs]

            pred_start_logits, pred_end_logits = output

            result = SquadResult(unique_id, pred_start_logits, pred_end_logits)
            all_results.append(result)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    all_predictions = defaultdict(list)
    threshold = 0.5
    for example_index, example in enumerate(examples):
        feature = features[example_index]
        result = unique_id_to_result[feature.unique_id]
        pred_start_logits, pred_end_logits = torch.tensor(result.start_logits), torch.tensor(result.end_logits)
        pred_start_probs, pred_end_probs = torch.sigmoid(pred_start_logits), torch.sigmoid(pred_end_logits)
        pred_start_probs, pred_end_probs = to_list(pred_start_probs), to_list(pred_end_probs)

        seq_len = len(feature.tokens)
        pred_start_probs, pred_end_probs = pred_start_probs[:seq_len], pred_end_probs[:seq_len]

        token_start_positions = [i for i, prob in enumerate(pred_start_probs) if prob > threshold]
        token_end_positions = [i for i, prob in enumerate(pred_end_probs) if prob > threshold]
        for token_start in token_start_positions:
            legal_token_end = [idx for idx in token_end_positions if idx >= token_start]
            if len(legal_token_end) == 0:
                continue
            token_end = min(legal_token_end)

            char_start = feature.token_to_orig_map[token_start]
            char_end = feature.token_to_orig_map[token_end + 1]
            all_predictions[example.unique_id].append((example.sentence[char_start:char_end],
                                                       char_start, char_end - 1))

    results = predicate_evaluate(examples, all_predictions)
    matched_gold = results.pop("matched_gold")

    # write predictions with their sentences (s, p, o)
    detailed_predictions = []
    for example in examples:
        unique_id = example.unique_id
        incomplete_spo_list = []
        incomplete_spo_spans = []
        for text, start, end in all_predictions[unique_id]:
            incomplete_spo_list.append(OrderedDict([
                ("subject", ""),
                ("predicate", text),
                ("object", "")
            ]))
            incomplete_spo_spans.append(OrderedDict([
                ("subject", [-1, -1]),
                ("predicate", [start, end]),
                ("object", [-1, -1])
            ]))
        example_with_prediction = collections.OrderedDict(
            [
                ("unique_id", unique_id),
                ("sentence", example.sentence),
                ("spo_list", incomplete_spo_list),
                ("spo_spans", incomplete_spo_spans),
                # additional information for debug purpose
                ("gold", example.all_predicates),
                ("predictions", all_predictions[unique_id]),
                ("matched_gold", matched_gold[unique_id])
            ]
        )
        detailed_predictions.append(example_with_prediction)

    if save_predictions:
        output_detailed_predictions_file = os.path.join(args.output_dir, "detailed_relation_predictions.json")
        with open(output_detailed_predictions_file, "w") as writer:
            writer.write(json.dumps(detailed_predictions, ensure_ascii=False, indent=4) + "\n")

    return results, detailed_predictions
    

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--bert_weight_file", default=None, type=str, help="Bert weight file.")  # non-required
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")

    ## Other parameters
    parser.add_argument("--data_dir", default=None, type=str)
    parser.add_argument("--train_file", default="train.json", type=str)
    parser.add_argument("--eval_file", default="dev.json", type=str)
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--max_segment_length", default=100, type=int)
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--eval_test", action="store_true", help="Whether to evaluate on final test set.")
    # parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
    # parser.add_argument("--eval_batch_size", default=8, type=int, help="Total batch size for predictions.")
    parser.add_argument("--per_gpu_train_batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=32, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% "
                             "of training.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json "
                             "output file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument("--logging_steps", type=int, default=200, help="Log every X updates steps.")

    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=None,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--version_2_with_negative',
                        action='store_true',
                        help='If true, the SQuAD examples contain some that do not have an answer.')
    parser.add_argument('--null_score_diff_threshold',
                        type=float, default=0.0,
                        help="If null_score - best_non_null is greater than the threshold predict null.")
    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        args.n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.n_gpu > 0:
            torch.cuda.manual_seed_all(args.seed)

    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
    #     raise ValueError("Output directory () already exists and is not empty.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    if args.do_train:
        train_fh = logging.FileHandler(os.path.join(args.output_dir, "train.log"), 'w')
        train_fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S'))
        logger.addHandler(train_fh)
    else:
        eval_fh = logging.FileHandler(os.path.join(args.output_dir, "eval.log"), 'w')
        eval_fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S'))
        logger.addHandler(eval_fh)

    logger.info(args)
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    # Prepare model
    args.bert_weight_file = args.bert_weight_file or os.path.join(args.bert_model, WEIGHTS_NAME)
    state_dict = torch.load(args.bert_weight_file, map_location="cpu")
    state_dict = state_dict_normalize(state_dict)
    model = PredicateExtractionModel.from_pretrained(args.bert_model, state_dict=state_dict,
                cache_dir=os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank)))

    if args.fp16:
        model.half()
    model.to(args.device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    # Either do_train or do_eval, eval_examples have to be loaded
    eval_processor = DataProcessor()
    eval_processor.read_examples_from_json(os.path.join(args.data_dir, args.eval_file))
    eval_examples = eval_processor.get_relation_examples()
    logger.info(" num examples in eval dataset: %d", len(eval_examples))

    if args.do_train:
        train_processor = DataProcessor()
        train_processor.read_examples_from_json(os.path.join(args.data_dir, args.train_file))
        train_examples = train_processor.get_relation_examples()
        logger.info(" num examples in train dataset: %d", len(train_examples))

        global_step, tr_loss = train(args, train_examples, model, tokenizer, eval_examples)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Save a trained model, configuration and tokenizer
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        tokenizer.save_vocabulary(args.output_dir)

        # save args
        output_args_file = os.path.join(args.output_dir, "training_args.bin")
        torch.save(args, output_args_file)
        logger.info("Saving training args to %s", output_args_file)

        # Load a trained model and vocabulary that you have fine-tuned
        # model = BertForQuestionAnswering.from_pretrained(args.output_dir)
        model = PredicateExtractionModel.from_pretrained(os.path.join(args.output_dir, 'best_model'))
        tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model.to(args.device)

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        if args.eval_test:
            eval_processor = DataProcessor()
            eval_processor.read_examples_from_json(os.path.join(args.data_dir, "test.json"))
            eval_examples = eval_processor.get_relation_examples()
            logger.info(" num examples in test dataset: %d", len(eval_examples))
        results = evaluate(args, eval_examples, model, tokenizer)[0]
        for key, value in results.items():
            logger.info("test_{}: {}".format(key, value))

if __name__ == "__main__":
    main()
