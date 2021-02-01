import os 
import re
import pdb
import ast 
import json
import random
import argparse
import numpy as np
import pandas as pd 
from tqdm import trange
from transformers import BertTokenizer
from collections import defaultdict, Counter

class EntityMarker():
    """Converts raw text to BERT-input ids.
    Attributes:
        tokenizer: Bert-base tokenizer.
        args: Args from command line. 
    """
    def __init__(self, args=None):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        self.args = args

    def get_sepid(self):
        return self.tokenizer.convert_tokens_to_ids(["[SEP]",])[0]

    def tokenize(self, raw_text, special_id=None):
        if special_id is not None:
            return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(raw_text)+["[unused%d]"%(special_id+10)])
        return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(raw_text))

    def tokenize_CLF(self, entity, text):
        entity = self.tokenizer.tokenize(entity)
        text = self.tokenizer.tokenize(text)
        return self.tokenizer.convert_tokens_to_ids(entity + ["[SEP]",] + text)


        

