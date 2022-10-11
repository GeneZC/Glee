# -*- coding: utf-8 -*-

import torch
import collections


class DataCollator:
    def __init__(self, tokenizer, max_length=None):
        self.tokenizer = tokenizer
        self.max_length = max_length

    @staticmethod
    def _pad(indices, max_length, pad_idx):
        """Pad a sequence to the maximum length."""
        pad_length = max_length - len(indices)
        return indices + [pad_idx] * pad_length

    def __call__(self, batch):
        raise NotImplementedError()

CombinedBatch = collections.namedtuple(
    "CombinedBatch", 
    (
        "text_indices", 
        "text_mask",
        "text_segments", 
        "label",
    )
)

class CombinedCollator(DataCollator):
    def __init__(self, tokenizer, max_length=None):
        super().__init__(tokenizer, max_length)

    def __call__(self, batch):
        if self.max_length is None:
            max_length = max([inst.text_length for inst in batch])
        else:
            max_length = self.max_length
        
        batch_text_indices = []
        batch_text_mask = []
        batch_text_segments = []
        batch_label = []
        for inst in batch:
            text_indices = self._pad(inst.text_indices, max_length, self.tokenizer.pad_token_id)
            batch_text_indices.append(text_indices)
            text_mask = self._pad(inst.text_mask, max_length, 0)
            batch_text_mask.append(text_mask)
            text_segments = self._pad(inst.text_segments, max_length, 0)
            batch_text_segments.append(text_segments)
            batch_label.append(inst.label)
        return CombinedBatch(
            text_indices=torch.tensor(batch_text_indices, dtype=torch.long),
            text_mask=torch.tensor(batch_text_mask, dtype=torch.bool),
            text_segments=torch.tensor(batch_text_segments, dtype=torch.long),
            label=torch.tensor(batch_label, dtype=torch.long),
        )


PromptedBatch = collections.namedtuple(
    "PromptededBatch", 
    (
        "text_indices", 
        "text_mask",
        "text_segments", 
        "mask_position",
        "verbalizer_indices",
        "verbalizer_mask",
        "label",
    )
)

class PromptedCollator(DataCollator):
    def __init__(self, tokenizer, max_length=None):
        super().__init__(tokenizer, max_length)

    def __call__(self, batch):
        if self.max_length is None:
            max_length = max([inst.text_length for inst in batch])
        else:
            max_length = self.max_length
        
        batch_text_indices = []
        batch_text_mask = []
        batch_text_segments = []
        batch_mask_position = []
        batch_verbalizer_indices = []
        batch_verbalizer_mask = []
        batch_label = []
        for inst in batch:
            text_indices = self._pad(inst.text_indices, max_length, self.tokenizer.pad_token_id)
            batch_text_indices.append(text_indices)
            text_mask = self._pad(inst.text_mask, max_length, 0)
            batch_text_mask.append(text_mask)
            text_segments = self._pad(inst.text_segments, max_length, 0)
            batch_text_segments.append(text_segments)
            batch_mask_position.append(inst.mask_position)
            batch_verbalizer_indices.append(inst.verbalizer_indices)
            batch_verbalizer_mask.append(inst.verbalizer_mask)
            batch_label.append(inst.label)
        return PromptedBatch(
            text_indices=torch.tensor(batch_text_indices, dtype=torch.long),
            text_mask=torch.tensor(batch_text_mask, dtype=torch.bool),
            text_segments=torch.tensor(batch_text_segments, dtype=torch.long),
            mask_position=torch.tensor(batch_mask_position, dtype=torch.long),
            verbalizer_indices=torch.tensor(batch_verbalizer_indices, dtype=torch.long),
            verbalizer_mask=torch.tensor(batch_verbalizer_mask, dtype=torch.bool),
            label=torch.tensor(batch_label, dtype=torch.long),
        )