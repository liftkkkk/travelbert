from urllib.request import Request, urlopen
from urllib.parse import quote
import re
from nltk.util import ngrams
import string
import json
import jieba

punct = list("！？。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～《》｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.")+list(string.punctuation)
# print(punct)

def tokenize_method(s):
	return jieba.lcut(s)

def ngram_method(s):
	res = []
	# res += generate_ngram(s, 1)
	res += generate_ngram(s, 2)
	res += generate_ngram(s, 3)
	res += generate_ngram(s, 4)
	res += generate_ngram(s, 5)
	return list(set([e for e in res if e not in punct]))

def generate_ngram(s, n = 5):
	tokens = list(s)
	output = list(ngrams(tokens, n))
	res = []
	for p in output:
		phrase = "".join(p)
		res.append(phrase)
	return res

def entity_linking(phrase):
	if len(phrase) <2:
		return None
	request = Request('http://api.xlore.org/query?instances='+quote(phrase))
	response_body = urlopen(request).read()
	entity_info = json.loads(response_body.decode())
	return entity_info

def entity_link_sentence(input_sentence, ngram_method = False):
	"""
	input_sentence is a sentence not whole document.
	"""
	phrases = []
	if ngram_method:
		phrases = ngram_method(input_sentence)
	else:
		phrases = tokenize_method(input_sentence)

	entities = []
	for t in phrases:
		# print("linking phrase:", t)
		entity = entity_linking(t)
		if entity is not None:
			entities.append(entity)
	return entities

if __name__ == "__main__":
	# 这里自己使用 python 的 BeautifulSoup 提取出 网页中的句子,假设存储在下面的 input_sentences
	input_sentences = ["我爱北京天安门。","天安门上太阳升。"]
	for s in input_sentences:
		entities = entity_link_sentence(s)
		# print("============这句话包含的实体=============",s)
		for e in entities:
			# print(e)
			for w in e['Instances']:
				print(w["Label"], w["Types"], w["Uri"])

"""
import requests
import json
import sys


# url  = "http://166.111.68.66:9019/XLink/linkingSubmit.action"
# url="https://xlink.xlore.org/"
url="http://10.1.1.68:8081/EntityLinkingWeb/linkingSubmit.action"
text = "中华人民共和国成立于 1949.10.1"
lang = "zh"
data = {"text": text, "lang": lang}

request_result = requests.post(url, data)
print(request_result)
sys.exit()
link_result = json.loads(request_result.text)
print(link_result)
sys.exit()

url = "https://api.xlore.org/sigInfo?uri=http://xlore.org/instance/"
entity_id = "eni1"
entity_info_res = requests.get(url + entity_id)
entity_info = json.loads(entity_info_res.text)
"""