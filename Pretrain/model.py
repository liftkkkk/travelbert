import os 
import pdb 
import torch
import torch.nn as nn 
from pytorch_metric_learning.losses import NTXentLoss
from transformers import BertForMaskedLM, BertTokenizer, BertForPreTraining

def mask_tokens(inputs, tokenizer, not_mask_pos=None):
    """Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    
    Args:
        inputs: Inputs to mask. (batch_size, max_length) 
        tokenizer: Tokenizer.
        not_mask_pos: Using to forbid masking entity mentions. 1 for not mask.
    
    Returns:
        inputs: Masked inputs.
        labels: Masked language model labels.
    """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, 0.15)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    if not_mask_pos is None:
        masked_indices = torch.bernoulli(probability_matrix).bool()
    else:
        masked_indices = torch.bernoulli(probability_matrix).bool() & (~(not_mask_pos.bool())) # ** can't mask entity marker **
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs.cuda(), labels.cuda()

class BaikePretrain(nn.Module):
    """Pre-training model.
    """
    def __init__(self, args):
        super(BaikePretrain, self).__init__()
        self.model = BertForMaskedLM.from_pretrained('bert-base-chinese')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.args = args 
    
    def forward(self, input, mask):
        # masked language model loss
        input = input.view(-1, self.args.max_length)
        mask = mask.view(-1, self.args.max_length)

        m_input, m_labels = mask_tokens(input.cpu(), self.tokenizer)
        m_outputs = self.model(input_ids=m_input, masked_lm_labels=m_labels, attention_mask=mask)
        m_loss = m_outputs[0]

        return m_loss, 0


class CP(nn.Module):
    """Contrastive Pre-training model.
    This class implements `CP` model based on model `BertForMaskedLM`. And we 
    use NTXentLoss as contrastive loss function.
    Attributes:
        model: Model to train.
        tokenizer: Tokenizer.
        ntxloss: Contrastive loss function.
        args: Args from command line. 
    """
    def __init__(self, args):
        super(CP, self).__init__()
        self.model = BertForMaskedLM.from_pretrained('bert-base-chinese')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.ntxloss = NTXentLoss(temperature=args.temperature)
        self.fc = nn.Linear(768, 768)
        self.relu = nn.ReLU()
        self.args = args 
    
    def forward(self, input, mask, label):
        # masked language model loss
        input = input.view(-1, self.args.max_length)
        mask = mask.view(-1, self.args.max_length)
        label = label.view(-1) # (batch_size * 2)

        # Ensure that `mask_tokens` function doesn't mask entity mention.
        #not_mask_pos = torch.zeros((input.size()[0], input.size()[1]), dtype=int)
        #not_mask_pos[:,1] = 1
        #not_mask_pos[:,1] = 1

        m_input, m_labels = mask_tokens(input.cpu(), self.tokenizer)
        m_outputs = self.model(input_ids=m_input, masked_lm_labels=m_labels, attention_mask=mask)
        m_loss = m_outputs[0]

        outputs = m_outputs

        # entity marker starter
        state = outputs[2][:, 0, :] # (batch, max_length, hidden_size)
        state = self.relu(self.fc(state))
        r_loss = self.ntxloss(state, label)

        return m_loss, r_loss

class CLF(nn.Module):
    def __init__(self, args):
        super(CLF, self).__init__()
        self.model = BertForMaskedLM.from_pretrained('bert-base-chinese')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.loss = nn.CrossEntropyLoss()
        self.fc = nn.Linear(768, 58) ############## !!!!!!!!!!!!!! 
        self.args = args 
    
    def forward(self, input, mask, label):
        m_input, m_labels = mask_tokens(input.cpu(), self.tokenizer)
        m_outputs = self.model(input_ids=m_input, masked_lm_labels=m_labels, attention_mask=mask)
        m_loss = m_outputs[0]

        outputs = m_outputs

        # entity marker starter
        state = outputs[2][:, 0, :] # (batch, max_length, hidden_size)
        logits = self.fc(state)     # (batch, rel_num)
        r_loss = self.loss(logits, label)

        return m_loss, r_loss

class CAE(nn.Module):
    def __init__(self, args):
        super(CAE, self).__init__()
        self.model = BertForPreTraining.from_pretrained('bert-base-chinese')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.args = args 
    
    def forward(self, input, mask, label):
        input = input.view(-1, self.args.max_length)
        mask = mask.view(-1, self.args.max_length)
        label = label.view(-1) # (batch_size * bag_size)

        m_input, m_labels = mask_tokens(input.cpu(), self.tokenizer)
        m_outputs = self.model(input_ids=m_input, labels=m_labels, attention_mask=mask, next_sentence_label=label)
        m_loss = m_outputs[0]

        return m_loss, 0




