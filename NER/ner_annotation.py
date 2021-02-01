import glob,re,jieba,re,json,random
import pandas as pd
random.seed(0)
types = ["菜品","组织机构","建筑","文物","人物","门店","品牌","景点"]

def manual_add(vocab, clazz, entities):
	vocab[clazz] += entities
	vocab[clazz] = sorted(vocab[clazz])

def build_vocab(files):

	vocab = {}
	for f in files:
		df = pd.read_csv(f) 
		df.fillna("", inplace = True)
		for t in types:
			if t in f:
				v = []
				for index, row in df.iterrows():
					if t in ["景点","菜品","门店"] and row[1] == "http://travel.org/Name":
						v.append(row[2].strip())
					if t in ["品牌"] and row[1] == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type":
						v.append(row[2].strip())
					if t in ["人物","文物","建筑","组织机构"] and row[1] == "http://travel.org/ChineseName":
					 	v.append(row[2].strip())
				vocab[t] = sorted(set(v))
	# print(vocab.keys())
	manual_add(vocab,"景点",["天安门","故宫"])
	return vocab

def _add_item_offset(token, sentence):
	"""Get the start and end offset of a token in a sentence
		a chunk may occur multiple times, the return can be a list of chunks
	"""
	s_pattern = re.compile(re.escape(token))
	# print('pattern',s_pattern)
	token_offset_list = []
	for m in s_pattern.finditer(sentence):
		# print(m)
		token_offset_list.append((m.group(), m.start(), m.end()))
	return token_offset_list


def _cal_item_pos(target_offset, idx_list):
	"""Get the index list where the token is located
		the input can be a list of chunks
	"""
	target_idx = []
	for target in target_offset:
		start, end = target[1], target[2]
		cur_idx = []
		for i, idx in enumerate(idx_list):
			
			if idx >= start and idx < end:
				# print('find',start,idx,end)
				cur_idx.append(i)
		if len(cur_idx) > 0:
			target_idx.append(cur_idx)
	return target_idx


def _get_token_idx(sentence_term_list, sentence, space_number=0):
	"""Get the start offset of every token
		the input must include all the tokens, or the index is wrong
		space_number is for english
	"""
	token_idx_list = []
	start_idx = 0
	for sent_term in sentence_term_list:
		if start_idx >= len(sentence):
			break
		token_idx_list.append(start_idx)
		start_idx += len(sent_term)
		# if there is space in english, chinese don't have space
		start_idx += space_number
	return token_idx_list


def annotate_entity(entity, text):
	token_offset_list = _add_item_offset(entity, text)
	return token_offset_list

def get_token_id(fn):
	def func(text):
		return _get_token_idx(fn(text), text)
	return func

@get_token_id
def tokenize_text(text):
	return list(text.strip())

@get_token_id
def tokenize_text2(text):
	return text.split()

# http://c.biancheng.net/view/2270.html
# def get_token_id(text):
# 	return _get_token_idx(tokenize_text(text), text)

def annotate_text(entities, text):
	find_entities = []
	for entity in entities:
		# token_idx_list = get_token_id(text)
		token_idx_list = tokenize_text(text)
		# print(token_idx_list)
		token_offset_list = annotate_entity(entity, text)
		if len(token_offset_list) > 0:
			# print(token_offset_list)
			find_entities += token_offset_list
	all_entities = _cal_item_pos(find_entities, token_idx_list)
	return all_entities

class Solution:
	def merge(self, intervals):

		intervals.sort(key=lambda x: x[0])

		merged = []
		for interval in intervals:
			# if the list of merged intervals is empty or if the current
			# interval does not overlap with the previous, simply append it.
			if not merged or merged[-1][1] < interval[0]:
				merged.append(interval)
			else:
			# otherwise, there is overlap, so we merge the current and previous
			# intervals.
				merged[-1][1] = max(merged[-1][1], interval[1])
		return merged

def merge_entities(entities):
	intervals = [[interval[0], interval[-1]+1] for interval in res]
	return mi_tool.merge(intervals)


def split_sent(txt):
	sentences = re.split('(。|！|\!|\.|？|\?)',txt)         # 保留分割符
	# print(len(sentences))
	new_sents = []
	for i in range(int(len(sentences)/2)):
		sent = sentences[2*i] + sentences[2*i+1]
		new_sents.append(sent)
		# print(sentences)
	return new_sents


if __name__ == "__main__":

	# ===== load corpus =====
	with open("/Users/zhuhongyin/Downloads/data.json","r") as f:
		data = json.load(f)
	sents = []
	for txt in data:
		# print(txt["text"])
		cuts = split_sent(txt["text"])
		random.shuffle(cuts)
		sents += cuts[:10]
	print(len(sents))

	# ===== load vocab =========
	print("load vocab")
	files = glob.glob("./*.csv", recursive=True)
	files.sort()
	vocab = build_vocab(files)
	mi_tool = Solution()

	js_all = []

	with open("result.json","a+") as fp:
		# fp.write("[")
		for i,sent in enumerate(sents):
			if i <= 106021:
				continue
			js = {"uuid":i}
			js["sentence"]=list(sent.strip())
			js["entity"] = []
			if len(list(sent.strip())) < 4:
				continue
			for t in types:
				res = annotate_text(vocab[t], sent)
				if len(res) > 0:
					# print(t, res)
					mi_res = merge_entities(res)
					# print(t, mi_res)
					for s,e in mi_res:
						# print(list(txt.strip())[s:e])
						js["entity"].append({"{}_{}_{}".format(s,e,t):list(sent.strip())[s:e]})
				# js_all.append(js)
			print(js)
			json.dump(js,fp,ensure_ascii=False, indent = 4)
			fp.write(",\n")
		fp.write("]")
