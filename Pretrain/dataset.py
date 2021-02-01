import os 
import re
import ast 
import sys 
sys.path.append("..")
import json 
import pdb
import random 
import torch 
import numpy as np 
from tqdm import tqdm 
from torch.utils import data
from utils import EntityMarker
from collections import defaultdict
import copy

class TextDataset(data.Dataset):
    """Data loader for Baike Dataset.
    """
    def __init__(self, path, args):
        # load raw data
        f = open(os.path.join(path, "data.txt"))
        data = f.readlines()
        f.close()
        
        # tokenize
        entityMarker = EntityMarker(args)
        tot_instance = len(data)

        # pre process data
        self.input_ids = np.zeros((tot_instance, args.max_length), dtype=int)
        self.mask = np.zeros((tot_instance, args.max_length), dtype=float) 

        for i, ins in enumerate(tqdm(data)):
            ids = entityMarker.tokenize(ins.strip())
            length = min(len(ids), args.max_length)
            self.input_ids[i][0:length] = ids[0:length]
            self.mask[i][0:length] = 1
            
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        input_ids = self.input_ids[index]
        mask = self.mask[index]

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(mask, dtype=torch.float32)

class KASTDataset(data.Dataset):
    def __init__(self, path, args):
        self.args = args 
        self.path = path 
        self.entityMarker = EntityMarker(args)
        data = json.load(open(os.path.join(path, "kastdata.json")))
        ent2titles = json.load(open(os.path.join(path, "ent2titles.json")))
        relation = json.load(open(os.path.join(path, "relations.json")))

        self.text_tokens = np.zeros((len(data), args.max_length), dtype=int)
        self.text_length = np.zeros((len(data)), dtype=int)

        for i, item in enumerate(tqdm(data, desc="Process text data")):
            text = item['text']
            ids = self.entityMarker.tokenize_KAST_text(text)
            length = min(len(ids), args.max_length)
            self.text_tokens[i][:length] = ids[:length]
            self.text_length[i] = length
        
        self.data = data  
        self.ent2titles = ent2titles
        self.relation = relation
    
    def __len__(self):
        return len(self.text_length)

    def sample_triple(self, index):
        triples = self.data[index]['triples']
        # if no triples
        if len(triples) == 0:
            return []
        # sample triple 
        triple_tensor = []
        for triple in triples:
            h, r, t = triple[0], triple[1], triple[2]
            if random.random() < self.args.p_neg: # negative sample
                r_neg = random.sample(self.relation, 1)[0]
                triple_tensor.append([self.entityMarker.tokenize_KAST_triple(h, r_neg, t), 0])
            else:
                triple_tensor.append([self.entityMarker.tokenize_KAST_triple(h, r, t), 1])
        return triple_tensor
    
    def sample_title(self, index):
        doc = self.data[index]
        # sample title
        titles = copy.deepcopy(self.ent2titles[doc['ent']])
        titles.remove(doc['title'])
        if random.random() < self.args.p_neg and len(titles) >= 1:
            return self.entityMarker.tokenize_KAST_title(random.sample(titles, 1)[0]), 0
        else:
            return self.entityMarker.tokenize_KAST_title(doc['title']), 1

    def __getitem__(self, index):
        max_length = 512
        triples = self.sample_triple(index)
        begin = self.text_length[index]

        input_ids = np.zeros((max_length), dtype=int)
        mask = np.zeros((max_length), dtype=float)
        triple_label = np.zeros((max_length), dtype=float)
        triple_mask = np.zeros((max_length), dtype=float)

        # set text data
        input_ids[:begin] = self.text_tokens[index][:begin]
        mask[:begin] = 1

        # set title
        # title_ids, title_label = self.sample_title(index)
        # end = min(len(title_ids) + begin, max_length)
        # input_ids[begin:end] = title_ids[:end-begin]
        # mask[begin:end] = 1
        # triple_label[begin] = title_label
        # triple_mask[begin] = 1
        # begin = end

        # set triples
        # for triple in triples:
        #     end = min(len(triple[0]) + begin, max_length)
        #     input_ids[begin:end] = triple[0][:end-begin]
        #     mask[begin:end] = 1 
        #     triple_label[begin] = triple[1]
        #     triple_mask[begin] = 1
        #     begin = end
        #     if begin >= max_length:
        #         break
    
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(mask, dtype=torch.float32), \
                torch.tensor(triple_mask, dtype=torch.float32), torch.tensor(triple_label, dtype=torch.float32)
        



