import unicodedata
import numpy as np
from tqdm import tqdm
from pytorch_pretrained_bert.tokenization import _is_control, _is_whitespace, _is_punctuation


class InputFeatures(object):
    def __init__(self, unique_id, input_ids, input_mask, segment_ids, tokens, token_to_orig_map, char_to_token_index=None):
        self.unique_id = unique_id
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.char_to_token_index = char_to_token_index


class RelationInputFeatures(InputFeatures):
    """ Input features for subj && obj span prediction model. """

    def __init__(self, unique_id, input_ids, input_mask, segment_ids, tokens, token_to_orig_map,
                 pred_start_positions, pred_end_positions, segment_token_span=None,
                 segment_mask=None, head=None, head_mask=None, adj=None):
        super(RelationInputFeatures, self).__init__(unique_id, input_ids, input_mask, segment_ids, tokens, token_to_orig_map)
        self.pred_start_positions = pred_start_positions
        self.pred_end_positions = pred_end_positions
        self.segment_token_span = segment_token_span
        self.segment_mask = segment_mask
        self.head = head
        self.head_mask = head_mask
        self.adj = adj


class EntityInputFeatures(InputFeatures):
    """ Input features for subj && obj span prediction model. """

    def __init__(self, unique_id, input_ids, input_mask, segment_ids, tokens, token_to_orig_map, char_to_token_index,
                 subj_start_position, subj_end_position, obj_start_position, obj_end_position,
                 segment_token_span=None, segment_mask=None, head=None, head_mask=None, adj=None):
        super(EntityInputFeatures, self).__init__(unique_id, input_ids, input_mask, segment_ids, tokens,
                                                  token_to_orig_map, char_to_token_index)
        self.subj_start_position = subj_start_position
        self.subj_end_position = subj_end_position
        self.obj_start_position = obj_start_position
        self.obj_end_position = obj_end_position
        self.segment_token_span = segment_token_span
        self.segment_mask = segment_mask
        self.head = head
        self.head_mask = head_mask
        self.adj = adj


def _is_chinese_char(char):
    """Checks whether CP is the codepoint of a CJK character."""
    cp = ord(char)
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
        return True

    return False


def _is_non_segment(char):
    """ If a char is not included in segmentation result. """
    cp = ord(char)
    return cp == 0 or cp == 0xfffd or _is_control(char) or \
            _is_whitespace(char)

def get_linguistic_features(example, segment_char_span, char_to_token_index, n_tokens, max_seq_length, max_segment_length):
    segment_token_span = []
    for char_start, char_end in segment_char_span:
        token_start = min(char_to_token_index[char_start], max_seq_length - 1)
        token_end = min(char_to_token_index[char_end], max_seq_length - 1)
        segment_token_span.append([token_start, token_end])

    # make segments cover all tokens
    segment_token_span[0][0] = 0
    segment_token_span[-1][-1] = n_tokens - 1
    for idx in range(1, len(segment_token_span)):
        if segment_token_span[idx - 1][-1] + 1 < segment_token_span[idx][0]:
            segment_token_span[idx][0] = segment_token_span[idx - 1][-1] + 1

    # truncated node points to ROOT
    dependency_head = [head_index if head_index <= max_segment_length else 0
                       for word_index, head_index, dep_rel in example.dependency]

    assert len(dependency_head) == len(segment_token_span)
    if len(segment_token_span) > max_segment_length:
        segment_token_span = segment_token_span[:max_segment_length]
        dependency_head = dependency_head[:max_segment_length]
    segment_mask = [1] * len(segment_token_span)
    head_mask = [1] * len(dependency_head)

    padding_length = max_segment_length - len(segment_token_span)
    segment_token_span += [(-1, -1)] * padding_length
    segment_mask += [0] * padding_length
    dependency_head += [0] * padding_length
    head_mask += [0] * padding_length

    linguistic_features = {
        "segment_token_span": segment_token_span,
        "segment_mask": segment_mask,
        "head": dependency_head,
        "head_mask": head_mask
    }

    adj = np.zeros((max_seq_length + max_segment_length, max_seq_length + max_segment_length), dtype=np.float32)
    # segment (word) - token edge
    for segment_offset, (token_start, token_end) in enumerate(segment_token_span):
        if (token_start, token_end) == (-1, -1):
            break
        segment_idx = max_seq_length + segment_offset
        adj[token_start : token_end + 1, segment_idx] = 1.0

    # segment - segment edge
    for offset, head_offset in enumerate(dependency_head):
        if head_offset == 0:
            continue
        segment_idx = max_seq_length + offset
        head_idx = max_seq_length + head_offset - 1
        adj[segment_idx, head_idx] = 1.0
    # token - token edge
    for segment_offset, (token_start, token_end) in enumerate(segment_token_span):
        if (token_start, token_end) == (-1, -1):
            break
        head_offset = dependency_head[segment_offset]
        if head_offset == 0:
            continue
        head_token_start, head_token_end = segment_token_span[head_offset - 1]
        adj[token_start : token_end + 1, head_token_start : head_token_end + 1] = 1.0
    # undirected edge
    adj = adj + adj.T
    # segment_span and its head_span may be overlapping
    adj = np.clip(adj, 0, 1)
    assert np.min(adj) == 0 and np.max(adj) == 1
    # self-loop
    # adj = adj + np.eye(max_seq_length + max_segment_length, dtype=np.float32)
    linguistic_features.update({
        "adj": adj
    })

    return linguistic_features


def get_segment_char_span(example):
    sentence = example.sentence
    segment_char_span = []
    char_start = 0
    while char_start < len(sentence):
        segment = example.segments[len(segment_char_span)]
        # remove additional whitespaces from segment result
        segment = "".join(char for char in segment if not _is_whitespace(char))
        matched = False
        for char_end in range(char_start, len(sentence)):
            # remove whitespaces and special chars from sentence
            sentence_span = "".join(char for char in sentence[char_start:char_end + 1]
                                    if not _is_non_segment(char))
            if sentence_span == segment:
                # assign whitespaces and special chars to previous segments
                while char_end + 1 < len(sentence) and _is_non_segment(sentence[char_end + 1]):
                    char_end += 1
                matched = True
                segment_char_span.append([char_start, char_end])
                char_start = char_end + 1
                break
        assert matched, (sentence, segment)

    # make segments cover all chars
    segment_char_span[0][0] = 0
    segment_char_span[-1][-1] = len(sentence) - 1
    for idx in range(1, len(segment_char_span)):
        if segment_char_span[idx - 1][-1] + 1 < segment_char_span[idx][0]:
            segment_char_span[idx][0] = segment_char_span[idx - 1][-1] + 1
    return segment_char_span


def convert_examples_to_features(examples, tokenizer, max_seq_length, max_segment_length, feature_type="entity"):
    """
    convert examples to input features for NN model
    """
    features = []
    for example_index, example in enumerate(tqdm(examples)):
        # if example_index > 50:
        #     break
        sentence = example.sentence

        # some chinese chars are not in bert vocab, convert them to english counterparts
        # unicode ord: [8216, 8217, 8220, 8221, 8212, 8212, 8213, 8213] -> [39, 39, 34, 34, 45, 45, 45, 45]
        table = str.maketrans('‘’“”——――', '\'\'""----')
        sentence = sentence.translate(table)

        words = []
        char_to_word_index = []
        word_to_char_index = []

        tokens = []
        char_to_token_index = []
        token_to_char_index = []

        # whitespace split to words
        prev_is_whitespace = True
        for idx, char in enumerate(sentence):
            if _is_whitespace(char):
                prev_is_whitespace = True
            else:
                if _is_chinese_char(char) or _is_control(char) or _is_punctuation(char):
                    words.append(char)
                    word_to_char_index.append(idx)
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        words.append(char)
                        word_to_char_index.append(idx)
                    else:
                        words[-1] += char
                    prev_is_whitespace = False
            char_to_word_index.append(len(words) - 1)

        # words to word-piece tokens
        tokens.append("[CLS]")
        token_to_char_index.append(0)
        for idx, word in enumerate(words):
            char_start = word_to_char_index[idx]
            char_end = word_to_char_index[idx]
            while char_end < len(sentence) and char_to_word_index[char_end] == idx:
                char_to_token_index.append(len(tokens))
                char_end += 1

            word_tokens = []
            assert word == sentence[char_start:char_end].strip(), (word, sentence[char_start:char_end].strip())
            for sub_token in tokenizer.tokenize(word):
                tokens.append(sub_token)
                token_to_char_index.append(char_start)
                word_tokens.append(sub_token)

            # rectify char_to_token_index for empty word_tokens
            if len(word_tokens) == 0:
                for char_index in range(char_start, char_end):
                    char_to_token_index[char_index] = max(0, len(tokens) - 1)

            # improve char -- token alignment
            assert hasattr(tokenizer, "basic_tokenizer")
            no_accent_word = tokenizer.basic_tokenizer._run_strip_accents(word)
            word_tokens = [sub_token.replace("##", "") for sub_token in word_tokens]
            if len(word_tokens) > 1 and ("[UNK]" not in word_tokens) and ("".join(word_tokens).lower() == no_accent_word.lower()):
                for i, sub_token in enumerate(word_tokens[:-1]):
                    cur_token = len(tokens) - len(word_tokens) + i
                    # print(cur_token, len(tokens), len(token_to_char_index))
                    token_to_char_index[cur_token + 1] = token_to_char_index[cur_token] + len(word_tokens[i])
                    for j in range(token_to_char_index[cur_token], token_to_char_index[cur_token + 1]):
                        char_to_token_index[j] = cur_token
                for j in range(token_to_char_index[-1], char_end):
                    char_to_token_index[j] = len(tokens) - 1

        tokens.append("[SEP]")
        token_to_char_index.append(len(sentence))

        if feature_type == "relation":
            # Note that no negative examples

            if len(tokens) > max_seq_length:
                tokens = tokens[:max_seq_length]
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)

            # padding
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            pred_start_positions = [0] * max_seq_length
            pred_end_positions = [0] * max_seq_length
            for pred_start, pred_end in example.pred_spans:
                token_start = char_to_token_index[pred_start]
                token_end = char_to_token_index[pred_end]
                # print(tokens[token_start:token_end+1], example.sentence[pred_start:pred_end+1])
                if token_start < max_seq_length:
                    pred_start_positions[token_start] = 1
                if token_end < max_seq_length:
                    pred_end_positions[token_end] = 1
            # print(tokens)
            # print(pred_start_positions)
            # print(pred_end_positions)

            linguistic_features = {}
            if example.segments is not None:
                segment_char_span = get_segment_char_span(example)
                linguistic_features = get_linguistic_features(example, segment_char_span, char_to_token_index,
                                                              len(tokens), max_seq_length, max_segment_length)

            features.append(RelationInputFeatures(
                            unique_id=example_index,
                            input_ids=input_ids,
                            input_mask=input_mask,
                            segment_ids=segment_ids,
                            tokens=tokens,
                            token_to_orig_map=token_to_char_index,
                            pred_start_positions=pred_start_positions,
                            pred_end_positions=pred_end_positions,
                            **linguistic_features
                            ))
            continue

        # add predicate marker to tokens sequence
        entity_marker = {
            "PRED_START": "[unused1]",
            "PRED_END": "[unused2]"
        }

        PRED_START, PRED_END = entity_marker["PRED_START"], entity_marker["PRED_END"]

        tokens_with_marker = []
        subj_start, subj_end = example.subj_span
        pred_start, pred_end = example.pred_span
        obj_start, obj_end = example.obj_span

        # rebind tokens with original char after adding markers
        token_to_orig_map = []
        # get predicate token position [start_position, end_position]
        subj_start_position, subj_end_position = 0, 0
        obj_start_position, obj_end_position = 0, 0
        pred_start_marker_position, pred_end_marker_position = 0, 0
        for idx, token in enumerate(tokens):
            char_index = token_to_char_index[idx]
            if idx == char_to_token_index[pred_start]:
                pred_start_marker_position = len(tokens_with_marker)
                tokens_with_marker.append(PRED_START)
                token_to_orig_map.append(char_index)

            # Note: if an answer span is adjacent with marker, marker is not included in the span
            if idx == char_to_token_index[subj_start]:
                subj_start_position = len(tokens_with_marker)
            if idx == char_to_token_index[obj_start]:
                obj_start_position = len(tokens_with_marker)

            tokens_with_marker.append(token)
            token_to_orig_map.append(char_index)

            if idx == char_to_token_index[subj_end]:
                subj_end_position = len(tokens_with_marker) - 1
            if idx == char_to_token_index[obj_end]:
                obj_end_position = len(tokens_with_marker) - 1

            if idx == char_to_token_index[pred_end]:
                pred_end_marker_position = len(tokens_with_marker)
                tokens_with_marker.append(PRED_END)
                token_to_orig_map.append(char_index)

        # print(sentence)
        # subj_start, subj_end = example.subj_span
        # print(subj_start, subj_end, f"x{sentence[subj_start:subj_end+1]}x")
        # print(char_to_word_index)

        # print(words)
        # print(tokens_with_marker)
        # print([x for x in sentence[subj_start:subj_end+1].strip()])
        # print([words[x] for x in char_to_word_index[subj_start:subj_end+1]])
        # print(tokens_with_marker[subj_start_position])
        #
        # orig_tokens = "".join(sentence[subj_start:subj_end+1].lower().split())
        # mapped_tokens = tokens_with_marker[subj_start_position:subj_end_position+1]
        # if "[unused1]" not in mapped_tokens and "[unused2]" not in mapped_tokens and \
        #     "[UNK]" not in mapped_tokens and not orig_tokens == "".join(mapped_tokens).replace("##", "").lower():
        #     span_print(sentence, [(example.subj_span, example.pred_span, example.obj_span)])
        #     print(example.unique_id, orig_tokens,
        #           tokens_with_marker[subj_start_position:subj_end_position+1], "\n")

        if len(tokens_with_marker) > max_seq_length:
            tokens_with_marker = tokens_with_marker[:max_seq_length]
            if subj_end_position >= max_seq_length:
                subj_end_position = max_seq_length - 1
            if subj_start_position >= max_seq_length:
                subj_start_position, subj_end_position = 0, 0

            if obj_end_position >= max_seq_length:
                obj_end_position = max_seq_length - 1
            if obj_start_position >= max_seq_length:
                obj_start_position, obj_end_position = 0, 0
        input_ids = tokenizer.convert_tokens_to_ids(tokens_with_marker)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)

        # padding
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        updated_char_to_token_index = [index if index < pred_start_marker_position else
                                       index + 1 if index < pred_end_marker_position - 1 else
                                       index + 2 for index in char_to_token_index]
        linguistic_features = {}
        if example.segments is not None:
            segment_char_span = get_segment_char_span(example)
            assert pred_start_marker_position != 0 and pred_end_marker_position != 0
            linguistic_features = get_linguistic_features(example, segment_char_span, updated_char_to_token_index,
                                                          len(tokens_with_marker), max_seq_length, max_segment_length)

        features.append(EntityInputFeatures(
                        unique_id=example_index,
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        tokens=tokens_with_marker,
                        token_to_orig_map=token_to_orig_map,
                        char_to_token_index=updated_char_to_token_index,
                        subj_start_position=subj_start_position,
                        subj_end_position=subj_end_position,
                        obj_start_position=obj_start_position,
                        obj_end_position=obj_end_position,
                        **linguistic_features
                        ))
        # if example_index > -1:
        #     break
        # print("input_ids:", input_ids)
        # print("input_mask:", input_mask)
        # print("segment_ids:", segment_ids)
        # print("subj_start_position, subj_end_position:", subj_start_position, subj_end_position)
        # print("obj_start_position, obj_end_position:", obj_start_position, obj_end_position)
        # print("predicate tokens:", tokens_with_marker[start_position:end_position+1])
        # print("tokens:", tokens_with_marker)
        # print("subject tokens:", tokens_with_marker[subj_start_position:subj_end_position + 1])
        # print("object tokens:", tokens_with_marker[obj_start_position:obj_end_position + 1])
        # print("sentence:", sentence)
        # print("token_to_orig_map", token_to_orig_map)
        # print()
    return features


# import numpy as np
# from preprocess import DataProcessor
# from pytorch_pretrained_bert.tokenization import BertTokenizer
#
# sent_len = []
# tokenizer = BertTokenizer.from_pretrained("data/bert/bert-chinese", do_lower_case=True)
# processor = DataProcessor()
# processor.read_examples_from_json("data/SAOKE/dataset_json_s1/train.json")
# # # examples = processor.get_relation_examples()
# examples = processor.get_entity_examples()
# print(len(processor.examples))
# example = examples[13824]
# print(example.sentence, example.unique_id, example.orig_id)
#
# # features = convert_examples_to_features(examples, tokenizer, 150, 100, feature_type="relation")
# features = convert_examples_to_features(examples, tokenizer, 150, 128, feature_type="entity")
# print("total features:", len(features))

#
# segment_len = np.array([len(example.segments) for example in tqdm(examples)])
# for i in range(90, 101):
#     print(i, np.percentile(segment_len, i))


# (precentile, segment_length) relation examples
# 95 61.0
# 96 64.0
# 97 70.0
# 98 77.0
# 99 89.0
# 100 183.0

# (precentile, segment_length) entity examples
# 95 77.0
# 96 81.0
# 97 87.0
# 98 96.0
# 99 112.0
# 100 183.0

# examples = processor.get_relation_examples("data/SAOKE/SAOKE_DATA.json")
# print("total examples:", len(examples))
# for example in tqdm(examples):
#     tokens = tokenizer.tokenize(example.sentence)
#     sent_len.append(len(tokens) + 2)  # [CLS] [SEP] 4-entity-marker
# sent_len = np.array(sent_len)
# for i in range(90, 101):
#     print(i, np.percentile(sent_len, i))

# (percentile, length) of relation examples
# 97 113.14999999999782
# 98 126.09999999999854
# 99 147.0

# (percentile, length) of entity examples
# 96 135.0
# 97 145.0
# 98 159.0
# 99 185.3199999999997
