# -*- coding: utf-8 -*-

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score
from functools import partial

import numpy as np

"""
Metric Facotry:
    Get metric function. [task-specific]
"""


def split_f1(preds, labels, ratio_threshold=0.65):
    preds = np.asarray(preds)
    labels = np.asarray(labels)

    class_ratios = []
    for c in np.unique(labels):
        class_ratios.append((c, len(labels[labels == c]) / labels.shape[0]))

    accum_ratio = 0.
    head_classes, tail_classes = [], []
    for class_ratio in sorted(class_ratios, key=lambda x:x[1], reverse=True):
        accum_ratio += class_ratio[1]
        if accum_ratio < ratio_threshold:
            head_classes.append(class_ratio[0])
        else:
            tail_classes.append(class_ratio[0])

    f1_lst = f1_score(y_true=labels, y_pred=preds, labels=np.arange(np.max(labels)+1), average=None)
    #print(f1_lst)
    head_f1_lst = []
    for c in head_classes:
        head_f1_lst.append(f1_lst[c])
    head_f1 = np.mean(head_f1_lst)

    tail_f1_lst = []
    for c in tail_classes:
        #try:
        tail_f1_lst.append(f1_lst[c])
        #except:
        #    tail_f1_lst.append(1.0)
    tail_f1 = np.mean(tail_f1_lst)
         
    return head_f1, tail_f1


def acc_and_f1(preds, labels, ratio_threshold=0.65, average="macro"):
    acc = accuracy_score(y_true=labels, y_pred=preds)
    head_f1, tail_f1 = split_f1(preds, labels, ratio_threshold=ratio_threshold)
    f1 = f1_score(y_true=labels, y_pred=preds, average=average)
    return {"acc": acc, "head_f1": head_f1, "tail_f1": tail_f1, "f1": f1}


def acc(preds, labels):
    acc = accuracy_score(y_true=labels, y_pred=preds)
    return {"acc": acc}


def f1(preds, labels, average="macro"):
    f1 = f1_score(y_true=labels, y_pred=preds, average=average)
    return {"f1": f1}


def matthews(preds, labels):
    matthews_corr = matthews_corrcoef(y_true=labels, y_pred=preds)
    return {"matthews_corr": matthews_corr}


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {"pearson_corr": pearson_corr, "spearman_corr": spearman_corr}


METRIC_FN = {
    "iflytek": partial(acc_and_f1, average="macro"),
    "cmid": partial(acc_and_f1, average="macro"),
    "msra": partial(acc_and_f1, ratio_threshold=0.55, average="macro"),
    "ctc": partial(acc_and_f1, ratio_threshold=0.8, average="macro"),
    "rte": acc,
    "boolq": acc,
    "ecom": acc,
    "r52": partial(acc_and_f1, ratio_threshold=0.55, average="macro"),
}


def get_metric_fn(task_name):
    return METRIC_FN[task_name]
