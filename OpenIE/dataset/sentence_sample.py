import re
import json
import numpy as np


np.random.seed(1234)

file_path = "data/baike/baike_tourism.json"
# sample_file_path = "data/baike/baike_samples10000.txt"
sample_file_path = "data/baike/baike_samples50000.txt"
with open(file_path, "r") as file:
    data = json.load(file)

candidates = []
for entry in data:
    name, content = entry["_id"], entry["text"]
    content = re.sub(r"([。？！?!])", r"\1\n", content)
    for sentence in content.split():
        if len(sentence) < 10:
            continue
        candidates.append(sentence)

n_candidates = len(candidates)
n_samples = 50000
sample_indices = np.random.choice(n_candidates, n_samples, replace=False)
print(f"sample {n_samples} sentences from {n_candidates}")

sample_sentences = []
for index in sample_indices:
    sentence = candidates[index]
    sample_sentences.append(sentence)

with open(sample_file_path, "w") as writer:
    for sentence in sample_sentences:
        writer.write(sentence + "\n")
