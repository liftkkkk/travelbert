import os 
import sys
import pdb 
import torch
import torch.nn as nn 
import numpy as np
import copy
from utils import EntityMarker
from transformers import BertForMaskedLM, BertTokenizer, BertForPreTraining

class KnowledgeProbing(nn.Module):
    def __init__(self):
        super(KnowledgeProbing, self).__init__()
        self.model = BertForMaskedLM.from_pretrained('bert-base-chinese')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    
    def tokenize(self, text):
        f = open("tokenized_text.txt", 'w')
        tokenized_text = ['[CLS]', ]
        tokenized_text += self.tokenizer.tokenize(text)
        f.write(" ".join(tokenized_text) + '\n')
        f.close()
    
    def convert_tokens_to_ids(self):
        f = open("tokenized_text.txt")
        tokens = f.readline().strip()
        return self.tokenizer.convert_tokens_to_ids(tokens.split())
    
    def prepare_input(self):
        input_ids = np.zeros((1, 512), dtype=np.int)
        mask = np.zeros((1, 512), dtype=np.int)
        ids = self.convert_tokens_to_ids()
        length = min(512, len(ids))
        input_ids[0][:length] = ids[:length]
        mask[0][:length] = 1

        labels = copy.deepcopy(input_ids)
        labels[0][:] = -100
        indices = input_ids[0] == 103
        labels[0][indices] = input_ids[0][indices]


        return torch.tensor(input_ids, dtype=torch.long).cuda(), torch.tensor(mask, dtype=torch.float32).cuda(), torch.tensor(labels, dtype=torch.long).cuda(),
    
    def forward(self):
        input_ids, mask, labels = self.prepare_input()
        outputs = self.model(input_ids=input_ids, labels=labels, attention_mask=mask)
        s_mask = (labels == 103)[0]
        logits = outputs[1][0] * s_mask.unsqueeze(1)
        pred = torch.argmax(logits, 1)
        
        for i, s in enumerate(s_mask.detach().cpu().numpy().tolist()):
            if mask[0][i] == 0:
                break
            if s == 1:
                print(self.tokenizer.convert_ids_to_tokens([pred[i]])[0], end=" ")
            else:
                print(self.tokenizer.convert_ids_to_tokens([input_ids[0][i]])[0], end=" ")
        print("")
        return None

if __name__ == "__main__":
    model = KnowledgeProbing()
    if sys.argv[1] == 'tokenize':
        text = sys.argv[2]
        model.tokenize(text)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "7"
        model.eval()
        model.cuda()
        model()

        



    

