
import json
import csv
import random
random.seed(1234)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc("font", family="DengXian")

import torch
import numpy as np

def process():
    examples = []
    labels = []
    label2idx = {}
    with open("R52_corpus.txt", "r", encoding="latin") as fi:
        for i, line in enumerate(fi):
            line = line.strip()
            examples.append(line)
    with open("R52_labels.txt", "r", encoding="latin") as fi:
        for i, line in enumerate(fi):
            _, set_, label = line.strip().split("\t")
            if set_ == "train":
                if random.random() < 0.1:
                    set_ = "dev"
            if label not in label2idx:
                label2idx[label] = len(label2idx)
            labels.append((set_, label))
    #print(label2idx)
    with open("label2idx.json", "w") as fo:
        json.dump(label2idx, fo, indent=4, ensure_ascii=False)
    
    lengths = []
    with open("test.tsv", "w") as fo:
        writer = csv.writer(fo, delimiter="\t")
        writer.writerow(("text", "label"))
        for example, label in zip(examples, labels):
            if label[0] == "test":
                writer.writerow((example, label[1]))
                lengths.append(len(example.split()))

    with open("dev.tsv", "w") as fo:
        writer = csv.writer(fo, delimiter="\t")
        writer.writerow(("text", "label"))
        for example, label in zip(examples, labels):
            if label[0] == "dev":
                writer.writerow((example, label[1]))
                lengths.append(len(example.split()))

    label_dist = {}
    with open("train.tsv", "w") as fo:
        writer = csv.writer(fo, delimiter="\t")
        writer.writerow(("text", "label"))
        for example, label in zip(examples, labels):
            if label[0] == "train":
                writer.writerow((example, label[1]))
                lengths.append(len(example.split()))
                if label[1] not in label_dist:
                    label_dist[label[1]] = 0
                label_dist[label[1]] += 1
    print(len(label_dist))
    print(label_dist)
    print(f"avg length {np.mean(lengths)}")
    label_dist = sorted(label_dist.items(), key=lambda x: x[1])
    print(sum([v for k, v in label_dist[-int(len(label_dist) * 0.2):]]) / sum([v for k, v in label_dist]))
    print([k for k, v in label_dist[-int(len(label_dist) * 0.2):]])
    x = np.arange(len(label_dist))
    fig, ax1 = plt.subplots()
    ax1.bar(x, [i[1] for i in label_dist], label="dist")
    ax1.set_xticks(x)
    #ax1.set_xticklabels(list(label2idx.keys()))
    ax1.tick_params(rotation=90)
    plt.show()

process()

"""
avg length 114.80646221248631
0.8789704271631983
"""
