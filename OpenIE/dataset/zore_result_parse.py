import os
import json
import xml.dom.minidom
from collections import OrderedDict


def is_valid(pattern):
    """ only retain triples with obvious subject and object """
    if "nsubj" in pattern and "dobj" in pattern:
        return True
    elif "SBV" in pattern and ("VOB" in pattern or "POB" in pattern):
        return True
    else:
        return False


def parse_zore_result(file_path):
    """ parse zore xml result to (s, p, o) triples """
    n_tuples, n_filtered = 0, 0
    all_examples = []

    dom_tree = xml.dom.minidom.parse(file_path)
    doc = dom_tree.documentElement
    sentences = doc.getElementsByTagName("sentence")

    for sentence in sentences:
        origin_text = sentence.getElementsByTagName("origin_text")[0]
        sentence_text = origin_text.firstChild.data
        relations = sentence.getElementsByTagName("relation")
        n_tuples += len(relations)

        spo_list, spo_spans = [], []
        for relation in relations:
            pattern = relation.getAttribute("pattern")
            if not is_valid(pattern):
                continue

            relation_text = relation.getAttribute("pred")
            # TODO: special relation processing
            # relation_text = "的" if relation_text == "de" else relation_text

            subject_text, object_text = None, None
            arguments = relation.getElementsByTagName("argument")
            for argument in arguments:
                argument_text = argument.getAttribute("content")
                # ZORE output has additional (Z)
                argument_text = argument_text[:-3] if argument_text.endswith("(Z)") else argument_text
                argument_node_id = argument.getAttribute("node_id")
                sub_patterns = argument.getElementsByTagName("sub_pattern")

                pivot_idx = 0  # pivot_idx for judging argument types (subject, object)
                sem_patterns = []
                for idx, sub_pattern in enumerate(sub_patterns):
                    if argument_node_id == sub_pattern.getAttribute("node_id"):
                        pivot_idx = idx
                    sem_patterns.append(sub_pattern.getAttribute("sem_pattern"))

                if len(sem_patterns) == 1:
                    full_argument = argument_text
                elif len(sem_patterns) == 2:
                    # add additional adv. / prep. to argument
                    reordered_sem_patterns = sem_patterns
                    if (sem_patterns[1] + "_" + sem_patterns[0]) in pattern:
                        reordered_sem_patterns = [sem_patterns[1], sem_patterns[0]]
                    full_argument = [argument_text if "(" in sem_pattern and ")" in sem_pattern else
                                     "" if sem_pattern.split("-")[1] == "u" else sem_pattern.split("-")[1]
                                     for sem_pattern in reordered_sem_patterns]
                    full_argument = "".join(full_argument)
                else:
                    # when argument has more sem_patterns
                    full_argument = argument_text

                if len(sem_patterns) == 0:
                    continue
                if "nsubj" in sem_patterns[pivot_idx] or "SBV" in sem_patterns[pivot_idx]:
                    subject_text = full_argument
                elif any(label in sem_patterns[pivot_idx] for label in ["dobj", "VOB", "POB"]):
                    object_text = full_argument

            triple_text = [subject_text, relation_text, object_text]
            if any(text is None for text in triple_text):
                n_filtered += 1
                continue

            triple_start = [sentence_text.find(text) for text in triple_text]
            if any(start_pos == -1 for start_pos in triple_start):
                n_filtered += 1
                continue

            subject_start, relation_start, object_start = triple_start
            subject_end = subject_start + len(subject_text) - 1
            relation_end = relation_start + len(relation_text) - 1
            object_end = object_start + len(object_text) - 1

            assert subject_text == sentence_text[subject_start: subject_end + 1]
            assert relation_text == sentence_text[relation_start: relation_end + 1]
            assert object_text == sentence_text[object_start: object_end + 1]
            spo_list.append(OrderedDict([
                ("subject", subject_text),
                ("predicate", relation_text),
                ("object", object_text)
            ]))
            spo_spans.append(OrderedDict([
                ("subject", [subject_start, subject_end]),
                ("predicate", [relation_start, relation_end]),
                ("object", [object_start, object_end])
            ]))

        if len(spo_list) == 0:
            continue
        all_examples.append(OrderedDict([
            ("unique_id", len(all_examples)),
            ("sentence", sentence_text),
            ("spo_list", spo_list),
            ("spo_spans", spo_spans)
        ]))

    print(f"original zore #sentences: {len(sentences)}")
    print(f"original zore tuples: {n_tuples}")
    print(f"filtered illegal tuples: {n_filtered}")
    print(f"#sentences in processed dataset: {len(all_examples)}")
    print(f"#triples in processed dataset: {n_tuples - n_filtered}")
    return all_examples


def main():
    # file_path = "data/baike/relation_wiki.xml"
    file_path = "data/baike/relation_baike50000.xml"
    target_dir_path = "data/baike"

    all_examples = parse_zore_result(file_path)

    # split to train/dev/test examples
    train_file_path = os.path.join(target_dir_path, "train.json")
    dev_file_path = os.path.join(target_dir_path, "dev.json")
    test_file_path = os.path.join(target_dir_path, "test.json")

    n_examples = len(all_examples)
    n_train, n_dev = int(n_examples * 0.8), int(n_examples * 0.1)

    train_examples = all_examples[:n_train]
    dev_examples = all_examples[n_train:n_train + n_dev]
    test_examples = all_examples[n_train + n_dev:]
    print(f"n_train / n_dev / n_test: {n_train} / {n_dev} / {len(test_examples)}")

    for part_examples, part_file_path in zip([train_examples, dev_examples, test_examples],
                                             [train_file_path, dev_file_path, test_file_path]):
        with open(part_file_path, "w") as writer:
            writer.write(json.dumps(part_examples, ensure_ascii=False, indent=4) + "\n")


if __name__ == "__main__":
    main()
