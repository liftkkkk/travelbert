import os
import json
import requests
from collections import OrderedDict
import pandas as pd


test_file_path = "/data1/lzh/data/tourism-OIE/test.json"
with open(test_file_path, "r") as file:
    test_examples = json.load(file)

url = "http://10.1.1.30:1234/extract"
query = [{"sentence": example["sentence"]} for example in test_examples]

response = requests.post(url, json=query)
response.raise_for_status()
response_json = response.json()
output = response_json["output"]

# merge bert results with original examples
tagged_examples = []
for example, prediction in zip(test_examples, output):
    assert example["sentence"] == prediction["sentence"]
    spo_list = [{"subject": subj, "predicate": pred, "object": obj} for subj, pred, obj in prediction["triples"]]
    spo_spans = [{"subject": subj_span, "predicate": pred_span, "object": obj_span}
                 for subj_span, pred_span, obj_span in prediction["spans"]]
    tagged_examples.append(OrderedDict([
        ("unique_id", example["unique_id"]),
        ("sentence", example["sentence"]),
        ("spo_list", spo_list),
        ("spo_spans", spo_spans)
    ]))

# write to dir
test_file_dir = os.path.dirname(test_file_path)
with open(os.path.join(test_file_dir, "test_bert_tag.json"), "w") as file:
    json.dump(tagged_examples, file, ensure_ascii=False, indent=4)

# convert examples to table format for human-label convenience
data = []
for example in tagged_examples:
    data.append([example["sentence"], "", ""])
    data.extend([[triple["subject"], triple["predicate"], triple["object"]]
        for triple in example["spo_list"]
    ])
    data.append(["", "", ""])  # empty line to seperate examples

subj_col, pred_col, obj_col = zip(*data)
df = pd.DataFrame(data={"subject": subj_col, "predicate": pred_col, "object": obj_col})
df.to_excel(os.path.join(test_file_dir, "test_bert_tag.xlsx"), index=False)
