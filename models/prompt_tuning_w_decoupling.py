# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertOnlyMLMHead

import collections


Output = collections.namedtuple(
    "Output", 
    (
        'loss', 
        'prediction', 
        'label',
    )
)


class PromptTuningWDecoupling(BertPreTrainedModel):
    def __init__(self, config):
        setattr(config, "tie_word_embeddings", False)
        super().__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)
        self.init_weights()

    # def get_output_embeddings(self):
    #     return self.cls.predictions.decoder

    # def set_output_embeddings(self, new_embeddings):
    #     self.cls.predictions.decoder = new_embeddings

    def forward(self, inputs):
        text_indices, text_mask, text_segments, mask_position, verbalizer_indices, verbalizer_mask, label = inputs

        hidden_states = self.bert(text_indices, attention_mask=text_mask, token_type_ids=text_segments)[0]
        hidden_states = torch.gather(hidden_states, 1, mask_position.unsqueeze(2).expand(-1, -1, hidden_states.shape[2])).squeeze(1)

        logit = self.cls(hidden_states)
        logit = torch.gather(logit.unsqueeze(1).expand(-1, verbalizer_indices.shape[1], -1), 2, verbalizer_indices)
        logit = torch.sum(logit * verbalizer_mask.float(), 2) / verbalizer_mask.float().sum(2)

        loss = F.cross_entropy(logit, label, reduction='none')

        return Output(
            loss=loss, 
            prediction=logit.argmax(-1), 
            label=label,
        )


        
