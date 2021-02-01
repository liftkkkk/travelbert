import os
import re
import json
from json import JSONDecodeError
from tqdm import tqdm
from collections import OrderedDict


class RelationInputExample(object):
    """ A single sentence with all annotated predicate spans. """

    def __init__(self, unique_id, sentence, pred_spans, segments, dependency):
        self.unique_id = unique_id
        self.sentence = sentence
        self.pred_spans = pred_spans
        self.segments = segments
        self.dependency = dependency

    def __getattr__(self, name):
        if name == "all_predicates":
            all_predicates = [self.sentence[pred_start:pred_end + 1]
                for pred_start, pred_end in self.pred_spans]
            return all_predicates


class EntityInputExample(object):
    # TODO: pickle.dump copy.deepcopy(entity_example) issue
    """A single training/test example for subj && obj span prediction."""

    def __init__(self, unique_id, orig_id, sentence, subj_span, pred_span, obj_span,
                 answer_type="subject", segments=None, dependency=None):
        self.unique_id = unique_id
        # one sentence may have multiple EntityExample(s)
        # orig_id are original example uniqud_id for merging predictions
        self.orig_id = orig_id
        self.sentence = sentence
        self.subj_span = subj_span
        self.pred_span = pred_span
        self.obj_span = obj_span
        # whether subject or object as answer in SQuAD metric
        self.answer_type = answer_type
        self.segments = segments
        self.dependency = dependency

    def __getattr__(self, name):
        """ Wrapper for using SQuAD metric. """
        if name == "qas_id":
            return self.unique_id
        elif name == "answers":
            answers = []
            if self.answer_type == "subject":
                answers.append({"text": self.subject})
            else:
                answers.append({"text": self.object})
            return answers
        elif name in ["subject", "predicate", "object"]:
            prefix = name[:3] if name == "object" else name[:4]
            start, end = getattr(self, prefix + "_span")
            return self.sentence[start:end+1] if (start, end) != (-1, -1) else ""

    def __deepcopy__(self, memo):
        """ Avoid TypeError when using copy.deepcopy() with __getattr__ defined. """
        if id(self) in memo:
            return memo[id(self)]
        else:
            copied = type(self)(self.unique_id, self.orig_id, self.sentence,
                                self.subj_span, self.pred_span, self.obj_span,
                                self.answer_type, self.segments, self.dependency)
            memo[id(self)] = copied
            return copied

    def to_dict(self):
        example_dict = OrderedDict(
            [
                ("unique_id", self.unique_id),
                ("sentence", self.sentence),
                ("subject", self.subject),
                ("object", self.object),
                ("predicate", self.predicate)
            ]
        )
        return example_dict


class InputExample(object):
    """A single sentence with all annotated (s,p,o)."""

    def __init__(self, unique_id, sentence, spo_list, spo_spans,
                 segments=None, pos=None, dependency=None):
        self.unique_id = unique_id
        self.sentence = sentence
        self.spo_list = spo_list
        self.spo_spans = spo_spans
        # segments, pos, dependency for chinese
        self.segments = segments
        self.pos = pos  # part of speech, not position
        self.dependency = dependency

    def to_entity_examples(self, start_unique_id):
        entity_examples = []
        for i, spo_span in enumerate(self.spo_spans):
            pred_span = spo_span["predicate"]
            subj_span, obj_span = spo_span["subject"], spo_span["object"]
            entity_examples.append(EntityInputExample(
                                    unique_id=start_unique_id + i,
                                    orig_id=self.unique_id,
                                    sentence=self.sentence,
                                    subj_span=subj_span,
                                    pred_span=pred_span,
                                    obj_span=obj_span,
                                    segments=self.segments,
                                    dependency=self.dependency
                                    ))
        return entity_examples

    def to_relation_examples(self, unique_id):
        relation_examples = [RelationInputExample(
            unique_id=unique_id,
            sentence=self.sentence,
            pred_spans=[spo_span["predicate"] for spo_span in self.spo_spans],
            segments=self.segments,
            dependency=self.dependency
        )]
        return relation_examples


def get_spans(sentence, pattern):
    """
    spans: [[start, end], ...]
    """
    spans = [match.span(1) for match in re.finditer(f"(?=({re.escape(pattern)}))", sentence)]
    for match in re.finditer(f"(?=({re.escape(pattern)}))", sentence):
        assert match.group(1) == pattern, ("raw match", match.group(1), pattern)
    if len(spans) == 0 and ("[" in pattern or "]" in pattern or "|" in pattern):
        pattern = pattern.replace("[", "").replace("]", "")
        pattern = pattern.replace("|", ".*?")  # non-greedy match
        spans = [match.span(1) for match in re.finditer(f"(?=({pattern}))", sentence)]
        # print([match.group(1) for match in re.finditer(f"(?=({pattern}))", sentence)], "\n")
    if len(spans) == 0:
        # TODO: non-exact matching
        pass

    spans = [(start, end - 1) for start, end in spans]
    return spans


def span_print(sentence, spo_spans):
    """ Print sentences with s,p,o start-end markers.  """
    for spo_span in spo_spans:
        sentence_with_marker = ""
        subj_span, pred_span, obj_span = spo_span
        spo_span = {"subject": subj_span, "predicate": pred_span, "object": obj_span}
        for i, char in enumerate(sentence):
            for name in ["subject", "predicate", "object"]:
                if i == spo_span[name][0]:
                    sentence_with_marker += f"[{name}_start]"
            sentence_with_marker += char
            for name in ["subject", "predicate", "object"]:
                if i == spo_span[name][1]:
                    sentence_with_marker += f"[{name}_end]"
        print(sentence_with_marker)


def select_spans(subj_spans, pred_spans, obj_spans, nested=False):
    """ Select optimal spo_span for multiple span candidates.
    nested: whether predicate should contain object
    """
    optimal_span = None
    if len(subj_spans) == 0 or len(pred_spans) == 0 or len(obj_spans) == 0:
        return optimal_span

    def _is_overlap(span1, span2):
        """ Whether two spans has overlap. """
        # span: [start, end]
        return not (span1[1] < span2[0] or span2[1] < span1[0])

    # all spans with/without overlap
    overlap_spans, non_overlap_spans = [], []
    for subj_span in subj_spans:
        for pred_span in pred_spans:
            for obj_span in obj_spans:
                if _is_overlap(subj_span, pred_span) or _is_overlap(subj_span, obj_span) or \
                    _is_overlap(pred_span, obj_span):
                    overlap_spans.append((subj_span, pred_span, obj_span))
                else:
                    non_overlap_spans.append((subj_span, pred_span, obj_span))

    candidate_spans = []
    if nested:
        fallback_candidates = []
        for spo_span in overlap_spans:
            subj_span, pred_span, obj_span = spo_span
            # predicate should contain object
            if pred_span[0] <= obj_span[0] and obj_span[1] <= pred_span[1]:
                # prior choice requires subject not in predicate
                if not _is_overlap(subj_span, pred_span):
                    candidate_spans.append(spo_span)
                else:
                    fallback_candidates.append(spo_span)
        candidate_spans = candidate_spans if len(candidate_spans) > 0 else fallback_candidates
    else:
        candidate_spans = non_overlap_spans if len(non_overlap_spans) > 0 else overlap_spans

    if len(candidate_spans) == 1:
        return candidate_spans[0]

    # choose spo_spans with shortest length, cause regex may have longer match
    shortest_length = None
    shortest_candidates = []
    for spo_span in candidate_spans:
        length = sum(span[1] - span[0] + 1 for span in spo_span)
        if shortest_length is None or length == shortest_length:
            shortest_length = length
            shortest_candidates.append(spo_span)
        elif length < shortest_length:
            shortest_length = length
            shortest_candidates = [spo_span]

    # choose spo_span with smallest predicate-object distance
    optim_po_distance = None
    optim_po_candidates = []
    for spo_span in shortest_candidates:
        _, pred_span, obj_span = spo_span
        po_distance = max(obj_span[0] - pred_span[1], pred_span[0] - obj_span[1])
        if optim_po_distance is None or po_distance == optim_po_distance:
            optim_po_distance = po_distance
            optim_po_candidates.append(spo_span)
        elif po_distance < optim_po_distance:
            optim_po_distance = po_distance
            optim_po_candidates = [spo_span]

    # choose spo_spans with closest subject-predicate-object distance
    closest_distance = None
    for spo_span in optim_po_candidates:
        spo_span_sort = sorted(spo_span)
        distance = max(0, spo_span_sort[1][0] - spo_span_sort[0][1])  # max(0, .) for nested case
        distance += max(0, spo_span_sort[2][0] - spo_span_sort[1][1])
        if closest_distance is None or distance < closest_distance:
            closest_distance = distance
            optimal_span = spo_span

    return optimal_span


class DataProcessor(object):

    def __init__(self):
        self.examples = None

    def read_examples(self, data_file):
        """
        read, process, filter (s, p, o) from data_file
        """
        data = []
        with open(data_file, 'r') as file:
            try:  # load already normalized data
                data = json.load(file)
            except JSONDecodeError:
                file.seek(0)
                for line in file:
                    example = json.loads(line)
                    example["natural"] = example["natural"].replace("\u2002", " ")  # " " = 20H = "\u0020"
                    data.append(example)
        # with open(os.path.join(os.path.dirname(data_file), 'SAOKE_NORM.json'), 'w') as writer:
        #     json.dump(data, writer, ensure_ascii=False)

        examples = []
        n_sentences, n_facts = 0, 0

        def is_empty(arg):
            return arg in ["", "_"]

        for example in tqdm(data, desc="Origin Example"):
            sentence = example["natural"]
            spo_list = []
            spo_spans = []
            # filter facts that satisfy:
            # non-empty subject, predicate, object (['_'])
            # predicate is not special: BIRTH, DEATH, IN, DESC, ISA
            # not multiple subjects and objects, only one for each
            for fact in example["logic"]:
                subj, pred, objs = fact["subject"], fact["predicate"], fact["object"]
                subj, pred, objs = subj.strip(), pred.strip(), [obj.strip() for obj in objs]
                assert len(objs) >= 1
                if is_empty(subj) or is_empty(pred) or len(objs) > 1 or is_empty(objs[0]):
                    continue
                if pred in ["BIRTH", "DEATH", "IN", "DESC", "ISA"]:
                    continue
                # get subject, predicate, object spans
                obj = objs[0]
                nested = "X" in pred
                subj_spans = get_spans(sentence, subj)
                obj_spans = get_spans(sentence, obj)
                # replace X with located obj span
                if nested and len(obj_spans) > 0:
                    # expand content of the shortest span
                    obj_start, obj_end = min(obj_spans, key=lambda x: x[1] - x[0] + 1)
                    # | means pred could have extra characters around object
                    pred = pred.replace("X", "|{}|".format(sentence[obj_start:obj_end + 1]))
                pred_spans = get_spans(sentence, pred)
                spo_span = select_spans(subj_spans, pred_spans, obj_spans, nested=nested)

                if spo_span is None:
                    continue
                spo_list.append({"subject": subj, "predicate": pred, "object": obj})
                spo_spans.append({"subject": spo_span[0], "predicate": spo_span[1], "object": spo_span[2]})
                n_facts += 1
            examples.append(InputExample(
                                unique_id=len(examples),
                                sentence=sentence,
                                spo_list=spo_list,
                                spo_spans=spo_spans))
            if len(examples) > 0 and examples[-1].sentence == sentence:
                n_sentences += 1
        print("#sentences:", n_sentences)
        print("#facts(spo triples):", n_facts)
        self.examples = examples

    def filter_empty_examples(self):
        """ Filter examples with no spo_list. """
        if self.examples is None:
            raise ValueError("No examples to filter.")

        self.examples = [example for example in self.examples if len(example.spo_list) > 0]
        print("#examples after filtering:", len(self.examples))

    def get_entity_examples(self, start_unique_id=0):
        """ Get examples for entity span prediction. """
        entity_examples = []
        for example in self.examples:
            entity_examples += example.to_entity_examples(start_unique_id)
            if len(entity_examples) > 0:
                start_unique_id = entity_examples[-1].unique_id + 1

        return entity_examples

    def get_relation_examples(self):
        """ Get examples for relation span prediction. """
        relation_examples = []
        for example in self.examples:
            # one example corresponds to one relation_example, so the same id
            unique_id = example.unique_id
            relation_examples += example.to_relation_examples(unique_id)
        return relation_examples

    def read_examples_from_json(self, data_file_or_data):
        data = data_file_or_data
        if isinstance(data_file_or_data, str):
            with open(data_file_or_data, "r") as file:
                data = json.load(file)

        n_facts = 0
        examples = []
        for unique_id, example in enumerate(data):
            # if example already has a unique_id field
            unique_id = example.get("unique_id", unique_id)
            sentence = example["sentence"]
            spo_list = example.get("spo_list", [])
            spo_spans = example.get("spo_spans", [])
            segments = example.get("ltp_segments", None)
            pos = example.get("ltp_pos", None)
            dependency = example.get("ltp_dependency", None)

            examples.append(InputExample(
                unique_id=unique_id,
                sentence=sentence,
                spo_list=spo_list,
                spo_spans=spo_spans,
                segments=segments,
                pos=pos,
                dependency=dependency
            ))
            n_facts += len(spo_spans)
        n_sentences = len(examples)
        print("#sentences:", n_sentences)
        print("#facts(spo triples):", n_facts)
        self.examples = examples
