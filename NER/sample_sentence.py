import json,random
random.seed(0)

def has_entity_type(t, entity):
	for e in entity:
		return t in list(e.keys())[0]


with open("result.json","r") as f:
	data = json.load(f)

# data = data[:80000]
# :80000 80000:100000 100000:120000
data = data[100000:140000]

random.shuffle(data)
res = []

types = ["菜品","组织机构","建筑","文物","人物","门店","品牌","景点"]
num = 68
# num = 500

r0, r1, r2, r3, r4, r5, r6, r7 = [], [], [], [], [], [], [], []

c = 0

for i, e in enumerate(data):
	if len(e["sentence"]) > 6:
		e["sentence"] = ["_" if w.strip() =="" else w for w in e["sentence"]]
		# if "\n" in e["sentence"]:
		# 	continue

		if len(e["sentence"]) > 150:
			continue
			
		if has_entity_type(types[0], e["entity"]):
			if len(r0) < num:
				r0.append(e)
				continue
		if has_entity_type(types[1], e["entity"]):
			if len(r1) < num:
				r1.append(e)
				continue
		if has_entity_type(types[2], e["entity"]):
			if len(r2) < num:
				r2.append(e)
				continue
		if has_entity_type(types[3], e["entity"]):
			if len(r3) < num:
				r3.append(e)
				continue
		if has_entity_type(types[4], e["entity"]):
			if len(r4) < num:
				r4.append(e)
				continue
		if has_entity_type(types[5], e["entity"]):
			if len(r5) < num:
				r5.append(e)
				continue
		if has_entity_type(types[6], e["entity"]):
			if len(r6) < num:
				r6.append(e)
				continue
		if has_entity_type(types[7], e["entity"]):
			if len(r7) < num:
				r7.append(e)
				continue
	c += 1

print("c is", c)

res = res + r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7
print("result is", len(res))

with open("sample.json","w") as f:
	json.dump(res, f, ensure_ascii=False, indent = 4)

# 30%, 30%, 30%, 10%




