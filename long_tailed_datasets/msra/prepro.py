
import xml.etree.cElementTree as ET
import csv
import json
import random
random.seed(1234)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc("font", family="DengXian")

import torch
import numpy as np
from transformers import BertTokenizer, BertModel
tok = BertTokenizer.from_pretrained("../../plms/bert-base-chinese", never_split=["[unused1]"])
model = BertModel.from_pretrained("../../plms/bert-base-chinese")

def process():
    train_examples = []
    test_examples = []
    label2idx = {}
    with open("msra_bakeoff3_training.xml", "r", encoding="gb18030") as fi:
        root = ET.fromstring(fi.read())
        #root = tree.getroot()
        for s in root.iter("SENTENCE"):
            text = []
            entities = []
            for w in s.iter("w"):
                if list(w):
                    w = list(w)[0]
                    entities.append((w.text, w.get("TYPE")))
                if w.text:
                    text.append(w.text)
            #print(text)
            #exit(0)
            text = "".join(text)
            for entity in entities:
                train_examples.append((text,) + entity)
                if entity[-1] not in label2idx:
                    label2idx[entity[-1]] = len(label2idx)
    with open("msra_bakeoff3_test.xml", "r", encoding="gb18030") as fi:
        root = ET.fromstring(fi.read())
        #root = tree.getroot()
        for s in root.iter("SENTENCE"):
            text = []
            entities = []
            for w in s.iter("w"):
                if list(w):
                    w = list(w)[0]
                    entities.append((w.text, w.get("TYPE")))
                if w.text:
                    text.append(w.text)
            text = "".join(text)
            for entity in entities:
                test_examples.append((text,) + entity)
                if entity[-1] not in label2idx:
                    label2idx[entity[-1]] = len(label2idx)
    #print(label2idx)
    with open("label2idx.json", "w") as fo:
        json.dump(label2idx, fo, indent=4, ensure_ascii=False)
    
    with open("test.tsv", "w") as fo:
        writer = csv.writer(fo, delimiter="\t")
        writer.writerow(("text", "entity", "label"))
        for example in test_examples:
            writer.writerow(example)

    num_train_examples = len(train_examples)
    random.shuffle(train_examples)
    with open("dev.tsv", "w") as fo:
        writer = csv.writer(fo, delimiter="\t")
        writer.writerow(("text", "entity", "label"))
        for example in train_examples[:int(0.1 * num_train_examples)]:
            writer.writerow(example)
    label_dist = {}
    with open("train.tsv", "w") as fo:
        writer = csv.writer(fo, delimiter="\t")
        writer.writerow(("text", "entity", "label"))
        for example in train_examples[int(0.1 * num_train_examples):]:
            writer.writerow(example)
            if example[-1] not in label_dist:
                label_dist[example[-1]] = 0
            label_dist[example[-1]] += 1
    print(label_dist)
    label_norm = {}
    for k in label_dist:
        indices = torch.tensor(tok.convert_tokens_to_ids(tok.tokenize(k)))
        label_norm[k] = torch.sqrt(torch.sum(model.embeddings.word_embeddings(indices).mean(0) ** 2)).detach().numpy()
    label_dist = sorted(label_dist.items(), key=lambda x: x[1])
    print(sum([v for k, v in label_dist[-int(len(label_dist) * 0.2):]]) / sum([v for k, v in label_dist]))
    x = np.arange(len(label_dist))
    width = 0.5

    fig, ax1 = plt.subplots()
    ax1.bar(x - width / 2, [i[1] for i in label_dist], width=width, label="dist")
    ax2 = ax1.twinx()
    ax2.bar(x + width / 2, [label_norm[i[0]] for i in label_dist], width=width, color="orange", label="norm")
    ax1.set_xticks(x)
    #ax1.set_xticklabels(list(label2idx.keys()))
    ax1.tick_params(rotation=90)
    #plt.legend()
    plt.show()

process()