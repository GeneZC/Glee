# -*- coding: utf-8 -*-

import os
import json
import collections

from utils import Logger


logger = Logger()


class DataBuilder:
    def __init__(self, tokenizer, label_map, max_length=None):
        self.tokenizer = tokenizer
        self.label_map = label_map
        if max_length is None:
            self.max_length = tokenizer.model_max_length
        else:
            self.max_length = max_length

    @staticmethod
    def _truncate_pair(text_a_tokens, text_b_tokens, max_length):
        """Truncates a pair input in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(text_a_tokens) + len(text_b_tokens)
            if total_length <= max_length:
                break
            if len(text_a_tokens) > len(text_b_tokens):
                text_a_tokens.pop()
            else:
                text_b_tokens.pop()

    def build(self, examples, **kwargs):
        raise NotImplementedError()

CombinedInstance = collections.namedtuple(
    "CombinedInstance", 
    (
        "text_indices", 
        "text_mask", 
        "text_segments", 
        "text_length", 
        "label",
    )
)

class CombinedBuilder(DataBuilder):
    def __init__(self, tokenizer, label_map, max_length=None):
        super().__init__(tokenizer, label_map, max_length)

    def build(self, examples, **kwargs):
        instances = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                logger.info("Converting example %d of %d" % (ex_index, len(examples)))

            label = self.label_map(example.label)
            text_a_tokens = self.tokenizer.tokenize(example.text_a)
            text_b_tokens = None
            if example.text_b:
                text_b_tokens = self.tokenizer.tokenize(example.text_b)
                # Account for [CLS], [SEP], [SEP] with "- 3" for combined input.
                self._truncate_pair(text_a_tokens, text_b_tokens, self.max_length - 3)
                text_tokens = [self.tokenizer.cls_token] + text_a_tokens + [self.tokenizer.sep_token]
                text_segments = [0] * (len(text_a_tokens) + 2)
                text_tokens += text_b_tokens + [self.tokenizer.sep_token]
                text_segments += [1] * (len(text_b_tokens) + 1)
                text_length = len(text_tokens)
                text_mask = [1] * text_length
                text_indices = self.tokenizer.convert_tokens_to_ids(text_tokens)

                assert text_length <= self.max_length

                if ex_index < 5:
                    logger.info("*** Example ***")
                    logger.info("uid: %s" % (example.uid))
                    logger.info("text_tokens: %s" % " ".join([str(x) for x in text_tokens]))
                    logger.info("text_indices: %s" % " ".join([str(x) for x in text_indices]))
                    logger.info("text_mask: %s" % " ".join([str(x) for x in text_mask]))
                    logger.info("text_segments: %s" % " ".join([str(x) for x in text_segments]))
                    logger.info("text_length: %d" % text_length)
                    logger.info("label: %s (id = %d)" % (example.label, label))

                instances.append(
                    CombinedInstance(
                        text_indices=text_indices,
                        text_mask=text_mask,
                        text_segments=text_segments,
                        text_length=text_length,
                        label=label,
                    )
                )
            else:
                # Account for [CLS] and [SEP] with "- 2"
                if len(text_a_tokens) > self.max_length - 2:
                    text_a_tokens = text_a_tokens[:(self.max_length - 2)]
                text_tokens = [self.tokenizer.cls_token] + text_a_tokens + [self.tokenizer.sep_token]
                text_segments = [0] * len(text_tokens)
                text_length = len(text_tokens)
                text_mask = [1] * text_length
                text_indices = self.tokenizer.convert_tokens_to_ids(text_tokens)

                if ex_index < 5:
                    logger.info("*** Example ***")
                    logger.info("uid: %s" % (example.uid))
                    logger.info("text_tokens: %s" % " ".join([str(x) for x in text_tokens]))
                    logger.info("text_indices: %s" % " ".join([str(x) for x in text_indices]))
                    logger.info("text_mask: %s" % " ".join([str(x) for x in text_mask]))
                    logger.info("text_segments: %s" % " ".join([str(x) for x in text_segments]))
                    logger.info("text_length: %d" % text_length)
                    logger.info("label: %s (id = %d)" % (example.label, label))

                instances.append(
                    CombinedInstance(
                        text_indices=text_indices,
                        text_mask=text_mask,
                        text_segments=text_segments,
                        text_length=text_length,
                        label=label,
                    )
                )
        return instances


PromptedInstance = collections.namedtuple(
    "PromptedInstance", 
    (
        "text_indices", 
        "text_mask", 
        "text_segments", 
        "text_length", 
        "mask_position",
        "verbalizer_indices",
        "verbalizer_mask",
        "label",
    )
)

class PromptedBuilder(DataBuilder):
    def __init__(self, tokenizer, label_map, max_length=None):
        super().__init__(tokenizer, label_map, max_length)

    @staticmethod
    def parse_template(template, tokenizer):
        """
            {cls}{text_a}这里的{text_b}看起来{mask}好。{sep}
            => [cls_token, text_a, 这, 里, 的, text_b, 看, 起, 来, mask_token, 好, 。, sep_token]
            {cls}{p0}{text_a}{p1}{p2}{p3]{text_b}{p4}{p5}{p6}{mask}{p7}{sep}
            => [cls_token, p0_token, text_a, p1_token, p2_token, p3_token, text_b, p4_token, p5_token, p6_token, mask_token, p7_token, sep_token]
        """
        template_tokens = []
        insert_positions = []
        is_container = False
        pattern = ""
        for c in template:
            if c == "{":
                if pattern:
                    template_tokens.extend(tokenizer.tokenize(pattern))
                pattern = ""
                is_container = True
            elif c == "}":
                if pattern == "cls":
                    template_tokens.append(tokenizer.cls_token)
                elif pattern == "sep":
                    template_tokens.append(tokenizer.sep_token)
                elif pattern == "mask":
                    template_tokens.append(tokenizer.mask_token)
                elif pattern == "text_a":
                    insert_positions.append(len(template_tokens))
                    template_tokens.append("text_a")
                elif pattern == "text_b":
                    insert_positions.append(len(template_tokens))
                    template_tokens.append("text_b")
                elif pattern.startswith("p"): # pseudo token
                    template_tokens.append(f"[{pattern.upper()}]")
                else:
                    raise ValueError(f"Unkonwn recognized pattern {temp}.")
                pattern = ""
                is_container = False
            else:
                pattern += c
        return template_tokens, insert_positions

    @staticmethod
    def parse_verbalizer(verbalizer, tokenizer):
        """
            {"-1": "不", "0": "较", "1": "很"}
            => {"-1": ["不"], "0": ["较"], "1": ["很"]}
            {"-1": "不好", "0": "还可以", "1": "不错"}
            => {"-1": ["不", "好", "[PAD]"], "0": ["还", "可", "以"], "1": ["不", "错", "[PAD]"]}
            or a path to json-like file
        """
        if os.path.exists(verbalizer):
            verbalizer = json.load(open(verbalizer, "r", encoding="utf-8"))
        else:
            verbalizer = eval(verbalizer)
        verbalizer = {k: tokenizer.tokenize(verbalizer[k]) for k in verbalizer}
        max_verbalizer_length = max([len(verbalizer[k]) for k in verbalizer])
        verbalizer_tokens = [verbalizer[k] + [tokenizer.pad_token] * (max_verbalizer_length - len(verbalizer[k])) for k in verbalizer] 
        verbalizer_indices = [tokenizer.convert_tokens_to_ids(vt) for vt in verbalizer_tokens] 
        verbailzer_mask = [[1] * len(verbalizer[k]) + [0] * (max_verbalizer_length - len(verbalizer[k])) for k in verbalizer]
        return verbalizer_tokens, verbalizer_indices, verbailzer_mask

    def build(self, examples, **kwargs):
        template = kwargs.get("template", "")
        verbalizer = kwargs.get("verbalizer", "")
        if not template or not verbalizer:
            ValueError("Either template or verbalizer is not offered for prompting.")
        template_tokens, insert_positions = self.parse_template(template, self.tokenizer)
        template_length = len(template_tokens)
        verbalizer_tokens, verbalizer_indices, verbalizer_mask = self.parse_verbalizer(verbalizer, self.tokenizer)
        instances = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                logger.info("Converting example %d of %d" % (ex_index, len(examples)))
            
            label = self.label_map(example.label)
            text_a_tokens = self.tokenizer.tokenize(example.text_a)
            if example.text_b:
                assert len(insert_positions) == 2, "Exmaple.text_b is given but not in the template."
                text_b_tokens = self.tokenizer.tokenize(example.text_b)
                self._truncate_pair(text_a_tokens, text_b_tokens, self.max_length - template_length + 2)
                text_tokens = template_tokens[:insert_positions[0]] + text_a_tokens \
                    + template_tokens[insert_positions[0] + 1: insert_positions[1]] \
                    + text_b_tokens + template_tokens[insert_positions[1] + 1:]
            else:
                assert len(insert_positions) == 1, "Exmaple.text_b is not given but in the template."
                if len(text_a_tokens) > self.max_length - template_length + 1:
                    text_a_tokens = text_a_tokens[:(self.max_length - template_length + 1)]
                text_tokens = template_tokens[:insert_positions[0]] + text_a_tokens \
                    + template_tokens[insert_positions[0] + 1:]
            text_segments = [0] * len(text_tokens)
            text_length = len(text_tokens)
            text_mask = [1] * text_length
            mask_position = [text_tokens.index(self.tokenizer.mask_token)]
            assert mask_position[0] < self.max_length, "It seems the truncatenation does not work."
            text_indices = self.tokenizer.convert_tokens_to_ids(text_tokens)

            if ex_index < 5:
                logger.info("*** Example ***")
                logger.info("uid: %s" % (example.uid))
                logger.info("text_tokens: %s" % " ".join([str(x) for x in text_tokens]))
                logger.info("text_indices: %s" % " ".join([str(x) for x in text_indices]))
                logger.info("text_mask: %s" % " ".join([str(x) for x in text_mask]))
                logger.info("text_segments: %s" % " ".join([str(x) for x in text_segments]))
                logger.info("text_length: %d" % text_length)
                logger.info("mask_position: %s" % " ".join([str(x) for x in mask_position]))
                logger.info("verbalizer_tokens: %s" % " ".join([str(x) for x in verbalizer_tokens]))
                logger.info("verbalizer_indices: %s" % " ".join([str(x) for x in verbalizer_indices]))
                logger.info("verbalizer_mask: %s" % " ".join([str(x) for x in verbalizer_mask]))
                logger.info("label: %s (id = %d)" % (example.label, label))

            instances.append(
                PromptedInstance(
                    text_indices=text_indices,
                    text_mask=text_mask,
                    text_segments=text_segments,
                    text_length=text_length,
                    mask_position=mask_position,
                    verbalizer_indices=verbalizer_indices,
                    verbalizer_mask=verbalizer_mask,
                    label=label,
                )
            )
        return instances