
import json
import csv
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

label_map = {"药物功效":"药物作用", "药物药物禁忌":"药物禁忌", "病症病症禁忌":"病症禁忌", \
    "药物设备用法":"其他设备用法", "病症诱因":"病症病因", "治疗方案手术时间":"治疗方案治疗时间"}

def process():
    examples = []
    label2idx = {}
    with open("CMID.json", "r") as fi:
        lines  = json.load(fi)
        for i, d in enumerate(lines):
            #d = json.loads(line.strip())
            text = d["originalText"]
            label = d["label_4class"][0].strip("'")+d["label_36class"][0].strip("'")
            if label in label_map:
                label = label_map[label]
            if label not in label2idx:
                label2idx[label] = len(label2idx)
            examples.append((text, label))
    #print(label2idx)
    with open("label2idx.json", "w") as fo:
        json.dump(label2idx, fo, indent=4, ensure_ascii=False)
    
    num_examples = len(examples)
    random.shuffle(examples)
    with open("test.tsv", "w") as fo:
        writer = csv.writer(fo, delimiter="\t")
        writer.writerow(("text", "label"))
        for example in examples[:int(0.2 * num_examples)]:
            writer.writerow(example)
    with open("dev.tsv", "w") as fo:
        writer = csv.writer(fo, delimiter="\t")
        writer.writerow(("text", "label"))
        for example in examples[int(0.2 * num_examples): int(0.3 * num_examples)]:
            writer.writerow(example)
    label_dist = {}
    with open("train.tsv", "w") as fo:
        writer = csv.writer(fo, delimiter="\t")
        writer.writerow(("text", "label"))
        for example in examples[int(0.3 * num_examples):]:
            writer.writerow(example)
            if example[-1] not in label_dist:
                label_dist[example[-1]] = 0
            label_dist[example[-1]] += 1
    #print(label_dist)
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