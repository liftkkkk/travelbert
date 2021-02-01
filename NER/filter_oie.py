import json, argparse

def valid(t):
	
	if len(t["subject"]) == 0 or len(t["object"]) == 0 or len(t["predicate"]) == 0:
		return False
	if "," in t["predicate"] or "," in t["subject"] or "," in t["object"]:
		return False
	if len(t["subject"].strip()) > 20 or len(t["object"].strip()) > 10 or len(t["predicate"].strip()) > 10:
		return False
	if len(t["subject"].strip()) == 1 or len(t["object"].strip()) == 1:
		return False
	return True
	
if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	## Required parameters
	parser.add_argument("--input", default="https://you.ctrip.com/travels/beijing1/t3-p3.html", type=str, required=True, help=" ")
	parser.add_argument("--output", default=None, type=str, required=True, help=" ")
	
	args = parser.parse_args()

	with open(args.input,"r") as f:
		data = json.load(f)
	
	for d in data:
		spo = []
		spans = []
		for t, span in zip(d["spo_list"],d["spo_spans"]):
			if valid(t):
				spo.append(t)
				spans.append(span)

		d["spo_list"] = spo
		d["spo_spans"] = spans

	with open(args.output,"w") as f:
		json.dump(data,f,ensure_ascii=False, indent = 4)
