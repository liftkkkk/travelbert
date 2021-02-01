import json,random
random.seed(0)

with open("sample.json","r") as f:
	data = json.load(f)

random.shuffle(data)

all_res = []
for e in data:
	# filter blank token
	# tokens = [" " if w.strip() == "" else w for w in e["sentence"]]
	# tokens = e["sentence"]
	labels = ["O"]*len(e["sentence"])
	# only select positive samples
	# if len(e["entity"]) < 2:
	# 	continue

	for entity in e["entity"]:
		# print(entity.keys())
		entity_index = list(entity.keys())[0]
		s, e, _type = entity_index.split("_")
		
		s_, e_ = int(s), int(e)
		# deepcopy
		tmp_labels = list(labels)
		# judge if the span is O
		F = True
		for i in range(s_,e_):
			if labels[i] != "O":
				F = False
		if F:
			labels[s_] =  "B-{}".format(_type)
			for i in range(s_+1,e_):
				labels[i] = "I-{}".format(_type)
			
	assert(len(tokens) == len(labels))

	sent = []
	for k,v in zip(tokens,labels):
		line = "{} {}".format(k.strip(),v)
		sent.append(line)
	sent_column = "\n".join(sent)
	all_res.append(sent_column)

output = "\n\n".join(all_res)
with open("bio.txtl","w") as f:
	f.write(output)

