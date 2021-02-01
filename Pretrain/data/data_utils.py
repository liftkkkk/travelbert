import os 
import pdb
import sys  
import csv 
import json 
from collections import defaultdict


def convert_rawdata_to_list(path):
    rawdata = json.load(open(path))
    listdata = []
    MIN_LENGTH = 6
    print("We will filter %d documents" % len(rawdata))
    step = 0
    tot_length = 0
    for ins in rawdata:
        sens = ins['text'].strip().split("。")
        for sen in sens:
            sen = sen.strip()
            if len(sen) < MIN_LENGTH:
                continue
            listdata.append(sen)
            tot_length += len(sen)
        step += 1
        sys.stdout.write("%d processed\r" % step)
        sys.stdout.flush()
    print("Processed. We get %d sentences" % len(listdata))
    print("Avg length: %.3f" % (tot_length / len(listdata)))
    json.dump(listdata, open("baike_sentence.json", 'w'))

"""
Extract data from csv files.
"""
def parse_csv(filepath, type):
    f = open(filepath)
    csv_data = list(csv.reader(f))
    print("%d lines loaded." % len(csv_data))
    
    ent2name = {} 
    attrs = set()
    data = []
    for triple in csv_data[1:]:
        value = triple[2]
        if value[:4] == "http":
            continue
        ent = triple[0].split("/")[-1]
        attr = triple[1].split("/")[-1].split("#")[-1]
        if attr == "ChineseName" or attr == "Name":
            ent2name[ent] = value
        item = {'attr': attr, 'value': value, "ent": ent, "type": type}
        data.append(item)
        attrs.add(attr)
    noname = 0
    for item in data:
        if item['ent'] in ent2name:
            item['ent'] = ent2name[item['ent']]
        else:
            noname += 1

    print("%d triples has been generated, %d has no entity name" % (len(data), noname))
    return data, attrs

"""
"""
def process_data_for_CP(data):            
    washed_data = {}
    for key in data.keys():
        if len(data[key]) < 2:
            continue        
        washed_data[key] = data[key]

    ll = 0
    rel2scope = {}
    list_data = []
    for key in washed_data.keys():
        list_data.extend(washed_data[key])
        rel2scope[key] = [ll, len(list_data)]
        ll = len(list_data)
    
    if not os.path.exists("CP"):
        os.mkdir("CP")
    print("%d items saved" % len(list_data))
    json.dump(list_data, open("CP/cpdata.json","w"))
    json.dump(rel2scope, open("CP/rel2scope.json", 'w'))


"""
Process data for comparsion abstract and entity
"""
def process_data_for_CAE(data):
    data.sort(key=lambda x: x['type'])
    type2scope = {}
    last_type = ""
    for i, item in enumerate(data):
        if item['type'] != last_type:
            if i == 0:
                type2scope[item["type"]] = [i,]
            else:
                type2scope[last_type].append(i)
                type2scope[item["type"]] = [i, ]
            last_type = item['type']
    type2scope[last_type].append(len(data))

    ins2scope = []
    for i, item in enumerate(data):
        ins2scope.append(type2scope[item['type']])
    
    print(type2scope)
    json.dump(data,open("CAE/caedata.json", "w"))
    json.dump(ins2scope, open("CAE/ins2scope.json", 'w'))



if __name__ == "__main__":
    # all_attr, all_data = [], []
    # files = os.listdir("csvdata")
    # for i, file in enumerate(files):
    #     if file.endswith(".csv"):
    #         data, attr = parse_csv(os.path.join("csvdata", file), file.split("_")[0])    
    #         all_data.extend(data)
    #         all_attr.extend(list(attr))
    # all_attr = list(set(all_attr))
    # print("%d attributes has been saved" % len(all_attr))
    # print("%d triples has been saved" % len(all_data))
    # json.dump(all_attr, open("csvdata/attributes.json", 'w'))
    # json.dump(all_data, open("csvdata/triples.json", 'w'))
    
