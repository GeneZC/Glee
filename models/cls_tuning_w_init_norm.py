# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel


Output = collections.namedtuple(
    "Output", 
    (
        'loss', 
        'prediction', 
        'label',
    )
)


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense_ = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = nn.ReLU()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense_(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder_ = nn.Linear(config.hidden_size, config.num_labels, bias=False)

        self.bias_ = nn.Parameter(torch.zeros(config.num_labels))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder_.bias = self.bias_

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder_(hidden_states)
        return hidden_states

class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

class CLSTuningWInitNorm(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)
        self.init_weights()

    def forward(self, inputs):
        text_indices, text_mask, text_segments, label = inputs

        hidden_states = self.bert(text_indices, attention_mask=text_mask, token_type_ids=text_segments)[0]
        hidden_states = hidden_states[:, 0]

        logit = self.cls(hidden_states)
        
        if logit.shape[-1] == 1:
            loss = F.mse_loss(logit.squeeze(-1), label.float(), reduction='none')
            prediction = logit.squeeze(-1)
            label = label.float()
        else:
            loss = F.cross_entropy(logit, label, reduction='none')
            prediction = logit.argmax(-1)
        return Output(
            loss=loss, 
            prediction=prediction, 
            label=label,
        )


        
