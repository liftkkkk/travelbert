import json
from collections import Counter

def get_entity_type(entity):
	res = []
	for e in entity:
		t = list(e.keys())[0].split("_")[-1]
		res.append(t)
	return res


with open("sample.json","r") as f:
	data = json.load(f)


res = []
for i, e in enumerate(data):		
	ent = get_entity_type(e["entity"])
	res += ent

c = Counter(res)
print(c)