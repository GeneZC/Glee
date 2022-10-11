# -*- coding: utf-8 -*-

import collections

from transformers import (
    BertTokenizer,
    BertConfig,
)

from models.cls_tuning import CLSTuning
from models.cls_tuning_w_layer_norm import CLSTuningWLayerNorm
from models.cls_tuning_w_focal_loss import CLSTuningWFocalLoss
from models.cls_tuning_w_eta_norm import CLSTuningWEtaNorm
from models.cls_tuning_w_init_norm import CLSTuningWInitNorm
from models.cls_tuning_w_prompt import CLSTuningWPrompt
from models.prompt_tuning_w_decoupling import PromptTuningWDecoupling
from models.prompt_tuning import PromptTuning
from models.prompt_tuning_w_focal_loss import PromptTuningWFocalLoss


def get_model_class(model_type):
    if model_type == "cls_tuning":
        tokenizer_class = BertTokenizer
        config_class = BertConfig
        model_class = CLSTuning
    elif model_type == "cls_tuning_w_layer_norm":
        tokenizer_class = BertTokenizer
        config_class = BertConfig
        model_class = CLSTuningWLayerNorm
    elif model_type == "cls_tuning_w_focal_loss":
        tokenizer_class = BertTokenizer
        config_class = BertConfig
        model_class = CLSTuningWFocalLoss
    elif model_type == "cls_tuning_w_eta_norm":
        tokenizer_class = BertTokenizer
        config_class = BertConfig
        model_class = CLSTuningWEtaNorm
    elif model_type == "cls_tuning_w_init_norm":
        tokenizer_class = BertTokenizer
        config_class = BertConfig
        model_class = CLSTuningWInitNorm
    elif model_type == "cls_tuning_w_prompt":
        tokenizer_class = BertTokenizer
        config_class = BertConfig
        model_class = CLSTuningWPrompt
    elif model_type == "prompt_tuning":
        tokenizer_class = BertTokenizer
        config_class = BertConfig
        model_class = PromptTuning
    elif model_type == "prompt_tuning_w_decoupling":
        tokenizer_class = BertTokenizer
        config_class = BertConfig
        model_class = PromptTuningWDecoupling
    elif model_type == "prompt_tuning_w_focal_loss":
        tokenizer_class = BertTokenizer
        config_class = BertConfig
        model_class = PromptTuningWFocalLoss
    else:
        raise KeyError(f"Unknown model type {model_type}.")

    return tokenizer_class, config_class, model_class
