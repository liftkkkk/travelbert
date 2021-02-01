import json,random
random.seed(0)
with open("result.json","r") as f:
	data = json.load(f)

random.shuffle(data)
res = []

r0, r1, r2, r3 = [], [], [], []
for e in data:
	if len(e["sentence"]) > 6:
		e["sentence"] = ["_" if w.strip() =="" else w for w in e["sentence"]]
		# if "\n" in e["sentence"]:
		# 	continue

		if len(e["sentence"]) > 150:
			continue
			
		if len(e["entity"]) == 3:
			if len(r3) >= 900:
				continue
			r3.append(e)

		if len(e["entity"]) == 2:
			if len(r2) >= 900:
				continue
			r2.append(e)
		if len(e["entity"]) == 1:
			if len(r1) >= 900:
				continue
			r1.append(e)
		if len(e["entity"]) == 0:
			if len(r0) >= 300:
				continue
			r0.append(e)

assert(len(r3) == 900)
assert(len(r2) == 900)
assert(len(r1) == 900)
assert(len(r0) == 300)

res = res + r3 + r2 + r1 + r0


with open("sample.json","w") as f:
	json.dump(res, f, ensure_ascii=False, indent = 4)

# 30%, 30%, 30%, 10%




