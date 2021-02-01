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
from torch.utils import data
from utils import EntityMarker


class BaikeDataset(data.Dataset):
    """Data loader for Baike Dataset.
    """
    def __init__(self, path, args):
        # load raw data
        data = json.load(open(os.path.join(path, "baikedata.json"))) 
        # attr2id = json.load(open(os.path.join(path, "rel2id.json")))
                   
        # tokenize
        entityMarker = EntityMarker(args)
        tot_instance = len(data)

        # pre process data
        self.input_ids = np.zeros((tot_instance, args.max_length), dtype=int)
        self.mask = np.zeros((tot_instance, args.max_length), dtype=int) 

        for i, ins in enumerate(data):
            ids = entityMarker.tokenize(ins)
            length = min(len(ids), args.max_length)
            self.input_ids[i][0:length] = ids[0:length]
            self.mask[i][0:length] = 1
            
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        input_ids = self.input_ids[index]
        mask = self.mask[index]

        return input_ids, mask


class CPDataset(data.Dataset):
    """Overwritten class Dataset for model CP.
    This class prepare data for training of CP.
    """
    def __init__(self, path, args):
        """Inits tokenized sentence and positive pair for CP.
        
        Args:
            path: path to your dataset.
            args: args from command line.
        
        Returns:
            No returns
        
        Raises:
            If the dataset in `path` is not the same format as described in 
            file 'prepare_data.py', there may raise:
                - `key nor found`
                - `integer can't be indexed`
                and so on.
        """
        self.path = path 
        self.args = args 
        data = json.load(open(os.path.join(path, "cpdata.json")))
        rel2scope = json.load(open(os.path.join(path, "rel2scope.json")))
        attr2id = json.load(open(os.path.join(path, "rel2id.json")))
        entityMarker = EntityMarker(args)

        self.tokens = np.zeros((len(data), args.max_length), dtype=int)
        self.mask = np.zeros((len(data), args.max_length), dtype=int)
        self.label = np.zeros((len(data)), dtype=int)

        # Distant supervised label for sentence.
        # Sentences whose label are the same in a batch 
        # is positive pair, otherwise negative pair.
        for i, rel in enumerate(rel2scope.keys()):
            scope = rel2scope[rel]
            for j in range(scope[0], scope[1]):
                self.label[j] = i

        for i, sentence in enumerate(data):
            attrid = attr2id[sentence['attr']]
            ids = entityMarker.tokenize(sentence["value"], attrid)
            length = min(len(ids), args.max_length)
            self.tokens[i][:length] = ids[:length]
            self.mask[i][:length] = 1
        self.__sample__()
    
    def __pos_pair__(self, scope):
        """Generate positive pair.
        Args:
            scope: A scope in which all sentences' label are the same.
                scope example: [0, 12]
        Returns:
            all_pos_pair: All positive pairs. 
            ! IMPORTTANT !
            Given that any sentence pair in scope is positive pair, there
            will be totoally (N-1)N/2 pairs, where N equals scope[1] - scope[0].
            The positive pair's number is proportional to N^2, which will cause 
            instance imbalance. And If we consider all pair, there will be a huge 
            number of positive pairs.
            So we sample positive pair which is proportional to N. And in different epoch,
            we resample sentence pair, i.e. dynamic sampling.
        """
        pos_scope = list(range(scope[0], scope[1]))
        
        # shuffle bag to get different pairs
        random.shuffle(pos_scope)   
        all_pos_pair = []
        bag = []
        for i, index in enumerate(pos_scope):
            bag.append(index)
            if (i+1) % 2 == 0:
                all_pos_pair.append(bag)
                bag = []
        return all_pos_pair
    
    def __sample__(self):
        """Samples positive pairs.
        After sampling, `self.pos_pair` is all pairs sampled.
        `self.pos_pair` example: 
                [
                    [0,2],
                    [1,6],
                    [12,25],
                    ...
                ]
        """
        rel2scope = json.load(open(os.path.join(self.path, "rel2scope.json")))
        self.pos_pair = []
        for rel in rel2scope.keys():
            scope = rel2scope[rel]
            pos_pair = self.__pos_pair__(scope)
            self.pos_pair.extend(pos_pair)

        print("Postive pair's number is %d" % len(self.pos_pair))

    def __len__(self):
        """Number of instances in an epoch.
        
        Overwitten function.
        """
        return len(self.pos_pair)

    def __getitem__(self, index):
        """Get training instance.
        Overwitten function.
        
        Args:
            index: Instance index.
        
        Return:
            input: Tokenized word id.
            mask: Attention mask for bert. 0 means masking, 1 means not masking.
            label: Label for sentence.
            h_pos: Position of head entity.
            t_pos: Position of tail entity.
        """
        bag = self.pos_pair[index]
        input = np.zeros((self.args.max_length * 2), dtype=int)
        mask = np.zeros((self.args.max_length * 2), dtype=int)
        label = np.zeros((2), dtype=int)

        for i, ind in enumerate(bag):
            input[i*self.args.max_length : (i+1)*self.args.max_length] = self.tokens[ind]
            mask[i*self.args.max_length : (i+1)*self.args.max_length] = self.mask[ind]
            label[i] = self.label[ind]

        return input, mask, label

class CLFDataset(data.Dataset):
    def __init__(self, path, args):
        self.path = path 
        self.args = args 
        data = json.load(open(os.path.join(path, "clfdata.json")))
        rel2id = json.load(open(os.path.join(path, "rel2id.json")))
        entityMarker = EntityMarker(args)

        self.tokens = np.zeros((len(data), args.max_length), dtype=int)
        self.mask = np.zeros((len(data), args.max_length), dtype=int)
        self.label = np.zeros((len(data)), dtype=int)

        for i, sentence in enumerate(data):
            relid = rel2id[sentence['attr']]
            ids = entityMarker.tokenize_CLF(sentence['ent'], sentence["value"]) # [ENTITY] + [attribute value]
            length = min(len(ids), args.max_length)
            self.tokens[i][:length] = ids[:length]
            self.mask[i][:length] = 1
            self.label[i] = relid
    
    def __len__(self):
        """Number of instances in an epoch.
        
        Overwitten function.
        """
        return len(self.tokens)

    def __getitem__(self, index):
        input = self.tokens[index]
        mask = self.mask[index]
        label = self.label[index]

        return input, mask, label

class CAEDataset(data.Dataset):
    def __init__(self, path, args):
        self.path = path
        self.args = args
        data = json.load(open(os.path.join(path, "caedata.json")))
        self.ins2scope = json.load(open(os.path.join(path, "ins2scope.json")))
        entityMarker = EntityMarker(args)

        self.txt_tokens = np.zeros((len(data), args.max_length), dtype=int)
        self.ent_tokens = np.zeros((len(data), args.max_length), dtype=int)
        self.sepid = entityMarker.get_sepid()
        self.ent_len = np.zeros((len(data),), dtype=int)
        self.txt_len = np.zeros((len(data),), dtype=int)

        for i, sentence in enumerate(data):
            ent_ids = entityMarker.tokenize(sentence['ent'])
            ent_len = min(len(ent_ids), args.max_length)
            self.ent_tokens[i][:ent_len] = ent_ids[:ent_len]
            self.ent_len[i] = ent_len
            
            txt_ids = entityMarker.tokenize(sentence['text'][:args.max_length])
            txt_len = min(len(txt_ids), args.max_length)
            self.txt_tokens[i][:txt_len] = txt_ids[:txt_len]
            self.txt_len[i] = txt_len
        
        self.__sample__()

    def __sample__(self):
        self.bag = [] # (pos, neg, ..., neg)
        for i in range(self.txt_tokens.shape[0]):
            scope = list(range(self.ins2scope[i][0], self.ins2scope[i][1]))
            scope.remove(i)
            
            sample_num = self.args.bag_size - 1
            extra_add = sample_num - len(scope)
            if extra_add > 0:
                print("This scope is too small!")
                scope += [0, self.ins2scope[i][0]]

            neg_list = random.sample(scope, sample_num)
            self.bag.append([i,]+neg_list)
    
    def __len__(self):
        return len(self.bag)
    
    def __getitem__(self, index):
        bag = self.bag[index]
        input = np.zeros((self.args.max_length * self.args.bag_size), dtype=int)
        mask = np.zeros((self.args.max_length * self.args.bag_size), dtype=int)
        label = np.zeros((self.args.bag_size), dtype=int)

        txtid = bag[0]
        for i, ind in enumerate(bag):
            ent_end = i*self.args.max_length+self.ent_len[ind]

            input[i*self.args.max_length : ent_end] = self.ent_tokens[ind][:self.ent_len[ind]] 
            input[ent_end : ent_end+1] = self.sepid

            length = min(self.txt_len[txtid], (i+1)*self.args.max_length-(ent_end+1))
            input[ent_end+1 : ent_end+1+length] = self.txt_tokens[txtid][:length]
            mask[i*self.args.max_length : ent_end+1+length] = 1

            label[i] = 0 if i == 0 else 1

        return input, mask, label


