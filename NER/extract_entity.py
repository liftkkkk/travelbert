import json


# def BIO2IOB(self, BIO):
def BIO2entity(BIO):

	all_sentece = []
	# What should I do to make it easier to convert BIO to IOBES not IOB to IOBES
	# First extract each entity fragment and then convert it to IOBES
	for seq in BIO:
		# all entities
		entity_table = []
		# single entity
		entity = []
		for i,t in enumerate(seq):

			if t == "O":
				if len(entity) > 0:
					# if the pre is "O"
					if entity[-1] == "O":
						entity.append(t)
					# if the pre is PER
					else:
						entity_table.append(entity)
						entity = [t]

				# if there is nothing in entity
				else:
					entity = [t]

			else:
				if t[:2] == "B-":
					# If there is something in the entity
					if len(entity) > 0:
						entity_table.append(entity)
						entity = [t[2:]]

					else:
						entity = [t[2:]]

				if t[:2] == "I-":
					entity.append(t[2:])

		if len(entity) > 0:
			entity_table.append(entity)

		sequence_length = sum(len(chunk) for chunk in entity_table)
		iobes_length = len(seq)
		# print(sequence_length , iobes_length)
		assert(sequence_length == iobes_length)

		all_sentece.append(entity_table)
	return all_sentece


res_token = []
res_label = []
sentence_label = []
sentence_token = []
with open("bio.txt","r") as f:
	for l in f:
		if l.strip() == "":
			if len(sentence_label) > 0:
				res_token.append(sentence_token)
				res_label.append(sentence_label)
			sentence_label, sentence_token = [], []
			continue
		# print(l)
		arr = l.strip().split(" ")
		if len(arr) < 2:
			continue
		token, label = arr
		sentence_label.append(label)
		sentence_token.append(token)
	if len(sentence_label) > 0:
		res_token.append(sentence_token)
		res_label.append(sentence_label)

# for seq in res:
# 	print(seq)

entity = BIO2entity(res_label)

with open("BIO2entity.json","w") as fp:
	fp.write("[")
	for i in range(len(entity)):
		js = {
		"token":res_token[i],
		"label":entity[i],
		"entity":[]}
		print("=====")
		index = 0
		for chunk in entity[i]:
			if chunk[0] != "O":
				entity_chunk = res_token[i][index:index+len(chunk)]
				print(entity_chunk, chunk[0])
				js["entity"].append((entity_chunk,chunk[0]))
			index += len(chunk)

		json.dump(js,fp,ensure_ascii=False, indent = 4)
		fp.write(",\n")
	fp.write("]")








