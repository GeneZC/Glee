# -*- coding: utf-8 -*-

"""
Data Pipeline
    DataReader -> reading RawData from tsv, json, etc., in the form of Examples. [task-specific]
    DataBuilder -> converting Examples to Instances. [model-specific]
    DataCollator -> collating Instances as Batches. [model-specific]
"""


from data.readers import (
    IflytekReader,
    CMIDReader,
    MSRAReader,
    CTCReader,
    EComReader,
    RTEReader,
    BoolQReader,
    R52Reader,
)
from data.builders import (
    CombinedBuilder,
    PromptedBuilder,
)
from data.collators import (
    CombinedCollator,
    PromptedCollator,
)


READER_CLASS = {
    "iflytek": IflytekReader,
    "cmid": CMIDReader,
    "msra": MSRAReader,
    "ctc": CTCReader,
    "ecom": EComReader,
    "rte": RTEReader,
    "boolq": BoolQReader,
    "r52": R52Reader,
}


def get_reader_class(task_name):
    return READER_CLASS[task_name]


BUILDER_CLASS = {
    "combined": CombinedBuilder,
    "prompted": PromptedBuilder,
}

def get_builder_class(builder_name):
    return BUILDER_CLASS[builder_name]


COLLATOR_CLASS = {
    "combined": CombinedCollator,
    "prompted": PromptedCollator,
}

def get_collator_class(builder_name):
    return COLLATOR_CLASS[builder_name]
