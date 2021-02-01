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

    def tokenize(self, text):
        tokens = ['[CLS]',]
        tokens.extend(self.tokenizer.tokenize(text))
        tokens.append('[SEP]')
        return self.tokenizer.convert_tokens_to_ids(tokens)
    
    def tokenize_KAST_text(self, text):
        tokens = ['[CLS]',]
        tokens.extend(self.tokenizer.tokenize(text))
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def tokenize_KAST_triple(self, h, r, t):
        tokens = ['[SEP]',]
        tokens.extend(self.tokenizer.tokenize(h))
        tokens.extend(self.tokenizer.tokenize(r))
        tokens.extend(self.tokenizer.tokenize(t))
        return self.tokenizer.convert_tokens_to_ids(tokens)
    
    def tokenize_KAST_title(self, title):
        tokens = ['[SEP]',]
        tokens.extend(self.tokenizer.tokenize(title))
        return self.tokenizer.convert_tokens_to_ids(tokens)

    




        

