# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel

import collections


Output = collections.namedtuple(
    "Output", 
    (
        'loss', 
        'prediction', 
        'label',
    )
)


def focal_loss(input, target, gamma=2, eps=1e-7, ignore_index=-100, reduction='mean'):
    """
    A function version of focal loss, meant to be easily swappable with F.cross_entropy. The equation implemented here
    is L_{focal} = - \sum (1 - p_{pred})^\gamma p_{target} \log p_{pred}
    If with_logits is true, then input is expected to be a tensor of raw logits, and a softmax is applied
    If with_logits is false, then input is expected to be a tensor of probabiltiies (softmax previously applied)
    target is expected to be a batch of integer targets (i.e. sparse, not one-hot). This is the same behavior as
    nn.CrossEntropyLoss.
    Loss is ignored at indices where the target is equal to ignore_index
    batch behaviour: reduction = 'none', 'mean', 'sum'
    """
    y = F.one_hot(target, input.size(-1))
    pt = F.log_softmax(input, dim=-1)
    loss = -y * pt  # cross entropy
    loss *= (1 - pt) ** gamma  # focal loss factor
    loss = torch.sum(loss, dim=-1)

    # mask the logits so that values at indices which are equal to ignore_index are ignored
    loss = loss[target != ignore_index]

    # batch reduction
    if reduction == 'mean':
        return torch.mean(loss, dim=-1)
    elif reduction == 'sum':
        return torch.sum(loss, dim=-1)
    else:  # 'none'
        return loss


class CLSTuningWFocalLoss(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        if config.activation == "relu":
            self.cls = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(config.hidden_size, config.num_labels),
            )
        else:
            self.cls = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.Tanh(),
                nn.Dropout(0.1),
                nn.Linear(config.hidden_size, config.num_labels),
            )
        self.init_weights()

    def forward(self, inputs):
        text_indices, text_mask, text_segments, label = inputs

        hidden_states = self.bert(text_indices, attention_mask=text_mask, token_type_ids=text_segments)[0]
        hidden_states = hidden_states[:, 0]

        logit = self.cls(hidden_states)

        loss = focal_loss(logit, label, gamma=1.0, reduction="none")

        return Output(
            loss=loss, 
            prediction=logit.argmax(-1), 
            label=label,
        )


        
