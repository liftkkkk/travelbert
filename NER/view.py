import json
with open("baike.json") as f:
	for l in f:
		data = json.loads(l)
print("done")