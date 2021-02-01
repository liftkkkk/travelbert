import os
import json
import pandas as pd
from collections import OrderedDict


def get_span(sentence, part):
    start = sentence.find(part)
    return [start, start + len(part) - 1]


def read_examples_from_excel(file_path):
    df = pd.read_excel(file_path, engine="openpyxl")

    examples = []
    n_rows = len(df.index)
    start, end = 0, 0
    while start < n_rows:
        # [start, end) range is an example
        start = end  # start from prev end
        while start < n_rows and pd.isna(df.iloc[start]["subject"]):
            start += 1
        if start == n_rows:
            break
        end = start
        while end < n_rows and not pd.isna(df.iloc[end]["subject"]):
            end += 1
        
        sentence = df.iloc[start]["subject"]
        spo_list, spo_spans = [], []
        for idx in range(start + 1, end):
            row = df.iloc[idx]
            subj, pred, obj = row["subject"], row["predicate"], row["object"]
            subj, pred, obj = subj.strip(), pred.strip(), obj.strip()
            assert all(part in sentence for part in (subj, pred, obj))

            pred_span = get_span(sentence, pred)
            subj_span, obj_span = get_span(sentence, subj), get_span(sentence, obj)
            spo_list.append({"subject": subj, "predicate": pred, "object": obj})
            # spo_spans maybe not accurate, but it does not affect test
            spo_spans.append({"subject": subj_span, "predicate": pred_span, "object": obj_span})
        
        examples.append(OrderedDict([
            ("sentence", sentence),
            ("spo_list", spo_list),
            ("spo_spans", spo_spans)
        ]))
    return examples


def main():
    data_dir = "/data1/lzh/data/tourism-OIE"
    # data_dir = "/Users/lvzhiheng/Desktop/data/travel_OIE"
    bert_tag_file_path = os.path.join(data_dir, "test_bert_tag.json")
    with open(bert_tag_file_path, "r") as file:
        bert_tag_examples = json.load(file)

    label_file_path = os.path.join(data_dir, "test_bert_tag_label.xlsx")
    labeled_examples = read_examples_from_excel(label_file_path)

    # sanity check
    for bert_tag_example, labeled_example in zip(bert_tag_examples, labeled_examples):
        assert bert_tag_example["sentence"] == labeled_example["sentence"], (bert_tag_example["sentence"], labeled_example["sentence"])
        bert_tag_example["spo_list"] = labeled_example["spo_list"]
        bert_tag_example["spo_spans"] = labeled_example["spo_spans"]

    # remove empty examples
    examples = [example for example in bert_tag_examples if len(example["spo_list"]) > 0]
    target_file = os.path.join(data_dir, "test.json")
    with open(target_file, "w") as file:
        json.dump(examples, file, ensure_ascii=False, indent=4)

    print(f"#all examples: {len(bert_tag_examples)}")
    print(f"#non empty examples: {len(examples)}")


if __name__ == "__main__":
    main()
