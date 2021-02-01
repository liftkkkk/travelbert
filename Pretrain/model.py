import os 
import pdb 
import torch
import torch.nn as nn 
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

class TextPretrain(nn.Module):
    """Pre-training model.
    """
    def __init__(self, args):
        super(TextPretrain, self).__init__()
        self.model = BertForMaskedLM.from_pretrained('bert-base-chinese')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.args = args 
    
    def forward(self, input, mask):
        # masked language model loss
        input = input.view(-1, self.args.max_length)
        mask = mask.view(-1, self.args.max_length)

        m_input, m_labels = mask_tokens(input.cpu(), self.tokenizer)
        m_outputs = self.model(input_ids=m_input, labels=m_labels, attention_mask=mask)
        m_loss = m_outputs[0]

        return m_loss, 0


class KAST(nn.Module):
    def __init__(self, args):
        super(KAST, self).__init__()
        self.model = BertForMaskedLM.from_pretrained('bert-base-chinese')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.loss = nn.BCEWithLogitsLoss(reduction="none")
        self.args = args
        self.fc = nn.Linear(768, 1)
    
    def forward(self, input, mask, triple_mask, triple_label):
        """
            input: (batch, max_length)
            mask: (batch, max_length)
            triple_mask: (batch, max_length)
            triple_label: (batch, max_length)
        """
        # masked language model loss 
        m_input, m_labels = mask_tokens(input.cpu(), self.tokenizer, triple_mask.cpu())
        m_outputs = self.model(input_ids=m_input, labels=m_labels, attention_mask=mask)
        m_loss = m_outputs[0]

        # pdb.set_trace()

        outputs = m_outputs
        state = outputs[2]  # (batch, max_length, hidden_size)
        logits = self.fc(state).squeeze(2)  # (batch, max_length)

        # loss = self.loss(logits, triple_label)  # (batch, max_length)
        # loss = loss * triple_mask
        # loss = loss.sum(-1) / (triple_mask.sum(-1) + 1e-6)
        # loss = loss.mean()

        return m_loss, 0