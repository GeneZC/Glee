
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
    train_examples = []
    dev_examples = []
    label2idx = {}
    with open("train_data.txt", "r") as fi:
        reader = csv.reader(fi, delimiter="\t")
        for i, line in enumerate(reader):
            _, label, text = line
            if label not in label2idx:
                label2idx[label] = len(label2idx)
            train_examples.append((text, label))
    with open("validation_data.txt", "r") as fi:
        reader = csv.reader(fi, delimiter="\t")
        for i, line in enumerate(reader):
            _, label, text = line
            if label not in label2idx:
                label2idx[label] = len(label2idx)
            dev_examples.append((text, label))
    #print(label2idx)
    with open("label2idx.json", "w") as fo:
        json.dump(label2idx, fo, indent=4, ensure_ascii=False)
    
    lengths = []
    with open("test.tsv", "w") as fo:
        writer = csv.writer(fo, delimiter="\t")
        writer.writerow(("text", "label"))
        for example in dev_examples:
            writer.writerow(example)
            lengths.append(len(example[0]))
    num_train_examples = len(train_examples)
    random.shuffle(train_examples)
    with open("dev.tsv", "w") as fo:
        writer = csv.writer(fo, delimiter="\t")
        writer.writerow(("text", "label"))
        for example in train_examples[:int(0.1 * num_train_examples)]:
            writer.writerow(example)
            lengths.append(len(example[0]))
    label_dist = {}
    with open("train.tsv", "w") as fo:
        writer = csv.writer(fo, delimiter="\t")
        writer.writerow(("text", "label"))
        for example in train_examples[int(0.1 * num_train_examples):]:
            writer.writerow(example)
            lengths.append(len(example[0]))
            if example[-1] not in label_dist:
                label_dist[example[-1]] = 0
            label_dist[example[-1]] += 1
    print(f"avg length {np.mean(lengths)}")
    label_dist = sorted(label_dist.items(), key=lambda x: x[1])
    print(sum([v for k, v in label_dist[-int(len(label_dist) * 0.2):]]) / sum([v for k, v in label_dist]))
    print([k for k, v in label_dist[-int(len(label_dist) * 0.2):]])
    x = np.arange(len(label_dist))
    fig, ax1 = plt.subplots()
    ax1.bar(x, [i[1] for i in label_dist], label="dist")
    ax1.set_xticks(x)
    ax1.set_xticklabels(list(label2idx.keys()))
    ax1.tick_params(rotation=90)
    plt.show()

process()

"""
avg length 27.16482835139016
0.7322171682957515
['Age', 'Pregnancy-related Activity', 'Laboratory Examinations', 'Diagnostic', 'Consent', 'Therapy or Surgery', 'Multiple', 'Disease']
"""