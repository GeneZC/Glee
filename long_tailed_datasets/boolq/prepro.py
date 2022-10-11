
import json
import csv
import random
random.seed(1234)

import numpy as np

def process():
    train_examples = []
    dev_examples = []
    test_examples = []
    label2idx = {}
    with open("train.jsonl", "r") as fi:
        for i, line in enumerate(fi):
            d = json.loads(line.strip())
            text_a = d["question"]
            text_b = d["passage"]
            label = d["label"]
            if label not in label2idx:
                label2idx[label] = len(label2idx)
            train_examples.append((text_a, text_b, label))
    with open("dev32.jsonl", "r") as fi:
        for i, line in enumerate(fi):
            d = json.loads(line.strip())
            text_a = d["question"]
            text_b = d["passage"]
            label = d["label"]
            if label not in label2idx:
                label2idx[label] = len(label2idx)
            dev_examples.append((text_a, text_b, label))
    with open("val.jsonl", "r") as fi:
        for i, line in enumerate(fi):
            d = json.loads(line.strip())
            text_a = d["question"]
            text_b = d["passage"]
            label = d["label"]
            if label not in label2idx:
                label2idx[label] = len(label2idx)
            test_examples.append((text_a, text_b, label))
    with open("label2idx.json", "w") as fo:
        json.dump(label2idx, fo, indent=4, ensure_ascii=False)
    
    lengths = []
    with open("test.tsv", "w") as fo:
        writer = csv.writer(fo, delimiter="\t")
        writer.writerow(("text_a", "text_b", "label"))
        for example in test_examples:
            writer.writerow(example)
            lengths.append(len(example[0].split())+len(example[1].split()))

    with open("dev.tsv", "w") as fo:
        writer = csv.writer(fo, delimiter="\t")
        writer.writerow(("text_a", "text_b", "label"))
        for example in dev_examples:
            writer.writerow(example)
            lengths.append(len(example[0].split())+len(example[1].split()))
    label_dist = {}
    with open("train.tsv", "w") as fo:
        writer = csv.writer(fo, delimiter="\t")
        writer.writerow(("text_a", "text_b", "label"))
        for example in train_examples:
            writer.writerow(example)
            lengths.append(len(example[0].split())+len(example[1].split()))
    print(f"avg length {np.mean(lengths)}")

process()

"""
avg length 105.26574685062987
"""