# -*- coding: utf-8 -*-
import os 
import re 
import pdb
import sys  
import csv 
import json 
import jieba
import codecs
import time
import numpy as np
from tqdm import tqdm
from collections import defaultdict

# stop words
def get_stopwords_dict():
    f = open("KAST/cn_stopwords.txt")
    stopwords = {}
    for line in f.readlines():
        stopwords[line.strip()] = len(stopwords)
    stopwords['（'] = len(stopwords)
    stopwords['）'] = len(stopwords)
    return stopwords

STOPWORDS = get_stopwords_dict()

def filter_stopwords(tokens):
    result = []
    for t in tokens:
        if STOPWORDS.get(t, -1) != -1:
            continue
        if t.isdigit():
            continue
        result.append(t) 
    return result

def jaccard(s1, s2):
    # s1 = set(s1)
    # s2 = set(s2)
    # ret1 = s1.intersection(s2)
    # ret2 = s1.union(s2)
    # return len(ret1)/ len(ret2)
    s1 = set(s1)
    count = 0
    for t in s1:
        if t in s2:
            count += 1
    return count / (len(s1) + 1e-10)

def topk(a, k):
    return sorted(range(len(a)), key=lambda i: a[i])[-k:]

def select_triples(triples, text):
    scores = []
    for triple in triples:
        s1 = filter_stopwords(list(jieba.cut(''.join(triple))))
        s2 = text
        scores.append(jaccard(s1, s2))  

    result = []
    for i, score in enumerate(scores):
        if score > 0.3:
            result.append(triples[i])
    return result

def process_data_for_KAST(textfile, triplefile):
    ss_text = open(textfile)
    # process triples
    relations = set()
    tf = open(triplefile)
    ent2triples = defaultdict(list)
    for line in tf.readlines():
        items = line.strip().split('\t\t')
        ent = items[0]
        try:
            triples = json.loads(items[-1])
        except:
            continue
        for r, t in triples.items():
            if '中文名' in r:
                continue
            if [r, t] not in ent2triples[ent]:
                ent2triples[ent].append([r, t])
            relations.add(r)
    
    ent2titles = defaultdict(list)
    textdata = [] # list of {'text': 故宫..., 'triples': [], 'title': [], 'ent': 北京故宫}
    for line in tqdm(ss_text.readlines()):
        item = json.loads(line.strip())
        ent = item['header'].split('\t\t')[0]
        for title_text in item['text']:
            # set title
            title = title_text['title']
            ent2titles[ent].append(title)
            # loop for text
            for text in title_text['content']:
                instance = {
                    'text': text,
                    'triples': [],
                    'title': title,
                    'ent': ent
                }
                # if this entity don't have triple
                if ent2triples.get(ent, -1) == -1 or len(ent2triples.get(ent, -1)) == 0:
                    textdata.append(instance)
                    continue 

                # process triple
                rts = select_triples(ent2triples[ent], text)
                for rt in rts:
                    instance['triples'].append([ent, rt[0], rt[1]])
                textdata.append(instance)
    
    for ent in ent2titles.keys():
        ent2titles[ent] = list(set(ent2titles[ent]))

    json.dump(textdata, open("KAST/kastdata.json", 'w'))
    json.dump(ent2titles, open("KAST/ent2titles.json", 'w'))
    json.dump(list(relations), open("KAST/relations.json", 'w'))


if __name__ == "__main__":
    process_data_for_KAST("KAST/semi-structured.txt", "KAST/baidubd_infobox.txt.clean.txt")


