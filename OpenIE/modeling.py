import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
import numpy as np
import torch.nn.functional as F
# from tree import head_to_tree,tree_to_adj
# from gat import GAT
# from aggcn import AGGCN, GraphConvLayer, MultiGraphConvLayer



class BertForQuestionAnswering(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForQuestionAnswering, self).__init__(config)
        self.bert = BertModel(config)
        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.subject_outputs = nn.Linear(config.hidden_size, 2)
        self.object_outputs = nn.Linear(config.hidden_size, 2)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, subj_start_positions=None,
                subj_end_positions=None, obj_start_positions=None, obj_end_positions=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        sub_logits = self.subject_outputs(sequence_output)
        ob_logits = self.object_outputs(sequence_output)
        # print(sub_logits.shape)
        # print(ob_logits.shape)
        sub_start_logits, sub_end_logits = sub_logits.split(1, dim=-1)
        ob_start_logits, ob_end_logits = ob_logits.split(1,dim=-1)

        #print("START_LOGITS",sub_start_logits.shape)
        sub_start_logits = sub_start_logits.squeeze(-1)
        sub_end_logits = sub_end_logits.squeeze(-1)
        ob_start_logits = ob_start_logits.squeeze(-1)
        ob_end_logits = ob_end_logits.squeeze(-1)
        #print("END_LOGITS", end_logits.shape)

        sub_start_pos, sub_end_pos, ob_start_pos, ob_end_pos = subj_start_positions, subj_end_positions, obj_start_positions, obj_end_positions

        if subj_start_positions is not None and subj_end_positions is not None and obj_end_positions is not None and obj_start_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(sub_start_pos.size()) > 1:
                sub_start_pos = sub_start_pos.squeeze(-1)
            if len(sub_end_pos.size()) > 1:
                sub_end_pos = sub_end_pos.squeeze(-1)
            if len(ob_start_pos.size()) > 1:
                ob_start_pos = ob_start_pos.squeeze(-1)
            if len(ob_end_pos.size()) > 1:
                ob_end_pos = ob_end_pos.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = sub_start_logits.size(1)
            sub_start_pos.clamp_(0, ignored_index)
            sub_end_pos.clamp_(0, ignored_index)
            ob_start_pos.clamp_(0, ignored_index)
            ob_end_pos.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            sub_start_loss = loss_fct(sub_start_logits, sub_start_pos)
            sub_end_loss = loss_fct(sub_end_logits, sub_end_pos)
            ob_start_loss = loss_fct(ob_start_logits, ob_start_pos)
            ob_end_loss = loss_fct(ob_end_logits, ob_end_pos)
            total_loss = (sub_start_loss + sub_end_loss + ob_start_loss + ob_end_loss) / 4
            return total_loss
        else:
            return sub_start_logits, sub_end_logits, ob_start_logits, ob_end_logits


class PredicateExtractionModel(BertPreTrainedModel):
    def __init__(self, config):
        super(PredicateExtractionModel, self).__init__(config)
        self.bert = BertModel(config)
        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.predicate_outputs = nn.Linear(config.hidden_size, 2)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, pred_start_positions=None, pred_end_positions=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pred_logits = self.predicate_outputs(sequence_output)
        pred_start_logits, pred_end_logits = pred_logits.split(1, dim=-1)
        pred_start_logits = pred_start_logits.squeeze(-1)
        pred_end_logits = pred_end_logits.squeeze(-1)

        if pred_start_positions is not None and pred_end_positions is not None:
            criterion = nn.BCEWithLogitsLoss(reduction="none")
            start_loss = criterion(pred_start_logits, pred_start_positions)
            end_loss = criterion(pred_end_logits, pred_end_positions)
            attention_mask = attention_mask.to(dtype=torch.float32)
            start_loss = torch.sum(start_loss * attention_mask) / torch.sum(attention_mask)
            end_loss = torch.sum(end_loss * attention_mask) / torch.sum(attention_mask)
            total_loss = start_loss + end_loss
            return total_loss
        else:
            return pred_start_logits, pred_end_logits


class GraphPredicateExtractionModel(BertPreTrainedModel):
    def __init__(self, config):
        super(GraphPredicateExtractionModel, self).__init__(config)
        self.bert = BertModel(config)
        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.predicate_outputs = nn.Linear(config.hidden_size, 2)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, pred_start_positions=None, pred_end_positions=None,
                segment_token_span=None, segment_mask=None, head=None, head_mask=None):
        # if pred_start_positions is not None:
        #     total_loss = torch.tensor(0.0, requires_grad=True)
        #     return total_loss
        # else:
        #     return torch.randn(input_ids.size()), torch.randn(input_ids.size())
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pred_logits = self.predicate_outputs(sequence_output)
        pred_start_logits, pred_end_logits = pred_logits.split(1, dim=-1)
        pred_start_logits = pred_start_logits.squeeze(-1)
        pred_end_logits = pred_end_logits.squeeze(-1)

        if pred_start_positions is not None and pred_end_positions is not None:
            criterion = nn.BCEWithLogitsLoss(reduction="none")
            start_loss = criterion(pred_start_logits, pred_start_positions)
            end_loss = criterion(pred_end_logits, pred_end_positions)
            attention_mask = attention_mask.to(dtype=torch.float32)
            start_loss = torch.sum(start_loss * attention_mask) / torch.sum(attention_mask)
            end_loss = torch.sum(end_loss * attention_mask) / torch.sum(attention_mask)
            total_loss = start_loss + end_loss
            return total_loss
        else:
            return pred_start_logits, pred_end_logits


class GraphEntityExtractionModel(BertPreTrainedModel):
    def __init__(self, config):
        super(GraphEntityExtractionModel, self).__init__(config)
        self.bert = BertModel(config)
        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        opt = {'hidden_dim':768, 'num_layers':2, 'mlp_layers':2, 'cuda':True, 'input_dropout':0.5, 'gcn_dropout':0.5,'pooling':'avg'}
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.subject_outputs = nn.Linear(config.hidden_size, 2)
        self.object_outputs = nn.Linear(config.hidden_size, 2)
        # self.gcn_model = GCNRelationModel(opt)
        # self.lstm = nn.LSTM(input_size=config.hidden_size,
        #                     hidden_size=config.hidden_size // 2,
        #                     num_layers=1,
        #                     batch_first=True,
        #                     dropout=config.hidden_dropout_prob,
        #                     bidirectional=True)
        self.apply(self.init_bert_weights)

        # self.self_loop = False
        # # TODO: need to delete self-loop in adj
        # # self.gcn = GCN(opt)
        # # TODO: need to set self-loop of adj
        # self.self_loop = True
        # n_heads = 6
        # self.gcn = GAT(opt['hidden_dim'], opt['hidden_dim'] // 6, opt['hidden_dim'],
        #                0.3, 0.1, n_heads, 2)

        # self.gcn = GraphConvLayer()
        # self.gcn_hidden = 300
        # self.gcn = AGGCN(config.hidden_size, self.gcn_hidden, 2, 3, 4, 3, 0.5)
        # self.concat_outputs = nn.Linear(config.hidden_size + self.gcn_hidden, config.hidden_size)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, subj_start_positions=None,
                subj_end_positions=None, obj_start_positions=None, obj_end_positions=None, adj=None):
        hidden_states, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        sequence_output = hidden_states
        # hidden_states = self.lstm(hidden_states)[0]
        # # sequence_output=self.gcn_model((hidden_states, input_ids, attention_mask, segment_token_span,segment_mask, head, head_mask))
        #
        # if self.self_loop:
        #     diag = torch.eye(adj.size(1)).unsqueeze(0)
        #     adj = adj + diag.to(adj)
        #
        # max_seq_length = input_ids.size(1)
        # max_segment_length = adj.size(1) - max_seq_length
        #
        # word_token_adj = adj[:, max_seq_length:, :max_seq_length]
        # # print(word_token_adj.size(), hidden_states.size())
        # word_hidden = (word_token_adj @ hidden_states) / (torch.sum(word_token_adj, dim=-1, keepdim=True) + 1e-8)
        # # [batch_size, max_segment_length, hidden_size]
        # # print("word_hidden", word_hidden.size())
        # unified_features = torch.cat((hidden_states, word_hidden), dim=1)
        # sequence_output = self.gcn(unified_features, adj)
        # if sequence_output.size(-1) != 768:
        #     sequence_output = torch.cat((unified_features, sequence_output), dim=-1)
        #     sequence_output = self.concat_outputs(sequence_output)
        # sequence_output = sequence_output[:, :max_seq_length, :]

        sub_logits = self.subject_outputs(sequence_output)
        ob_logits = self.object_outputs(sequence_output)
        sub_start_logits, sub_end_logits = sub_logits.split(1, dim=-1)
        ob_start_logits, ob_end_logits = ob_logits.split(1, dim=-1)

        # print("START_LOGITS",sub_start_logits.shape)
        # .masked_fill(attention_mask == 0, -1e9)
        sub_start_logits = sub_start_logits.squeeze(-1).masked_fill(attention_mask == 0, -1e9)
        sub_end_logits = sub_end_logits.squeeze(-1).masked_fill(attention_mask == 0, -1e9)
        ob_start_logits = ob_start_logits.squeeze(-1).masked_fill(attention_mask == 0, -1e9)
        ob_end_logits = ob_end_logits.squeeze(-1).masked_fill(attention_mask == 0, -1e9)
        # print("END_LOGITS", end_logits.shape)

        sub_start_pos, sub_end_pos, ob_start_pos, ob_end_pos = subj_start_positions, subj_end_positions, obj_start_positions, obj_end_positions

        if subj_start_positions is not None and subj_end_positions is not None and obj_end_positions is not None and obj_start_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(sub_start_pos.size()) > 1:
                sub_start_pos = sub_start_pos.squeeze(-1)
            if len(sub_end_pos.size()) > 1:
                sub_end_pos = sub_end_pos.squeeze(-1)
            if len(ob_start_pos.size()) > 1:
                ob_start_pos = ob_start_pos.squeeze(-1)
            if len(ob_end_pos.size()) > 1:
                ob_end_pos = ob_end_pos.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = sub_start_logits.size(1)
            sub_start_pos.clamp_(0, ignored_index)
            sub_end_pos.clamp_(0, ignored_index)
            ob_start_pos.clamp_(0, ignored_index)
            ob_end_pos.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            sub_start_loss = loss_fct(sub_start_logits, sub_start_pos)
            sub_end_loss = loss_fct(sub_end_logits, sub_end_pos)
            ob_start_loss = loss_fct(ob_start_logits, ob_start_pos)
            ob_end_loss = loss_fct(ob_end_logits, ob_end_pos)
            total_loss = (sub_start_loss + sub_end_loss + ob_start_loss + ob_end_loss) / 4
            # total_loss = (sub_start_loss + sub_end_loss) / 2
            # total_loss = (ob_start_loss + ob_end_loss) / 2
            return total_loss
        else:
            return sub_start_logits, sub_end_logits, ob_start_logits, ob_end_logits
