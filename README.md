## Glee <img src="./assets/glee.png" width="22" height="22" alt="glee" align=center/>

This repository contains code for EMNLP 2022 paper titled [Making Pretrained Language Models Good Long-tailed Learners](https://arxiv.org/abs/2205.05461).

**************************** **Updates** ****************************

<!-- Thanks for your interest in our repo! -->

* 10/11/22: We released our paper, code, and data. Check it out!

## Quick Links

  - [Overview](#overview)
  - [Getting Started](#getting-started)
    - [Requirements](#requirements)
    - [Training & Evaluation](#training&evaluation)
    - [Adapting to a New Task](#adapting-to-a-new-task) 
  - [Bugs or Questions?](#bugs-or-questions)
  - [Citation](#citation)

## Overview

Prompt-tuning has shown appealing performance in few-shot classification by virtue of its capability in effectively exploiting pretrained knowledge. This motivates us to check the hypothesis that prompt-tuning is also a promising choice for long-tailed classification, since the tail classes are intuitively few-shot ones. To achieve this aim, we conduct empirical studies to examine the hypothesis. The results demonstrate that prompt-tuning makes pretrained language models at least good longtailed learners. For intuitions on why prompttuning can achieve good performance in longtailed classification, we carry out in-depth analyses by progressively bridging the gap between prompt-tuning and commonly used finetuning. The summary is that the classifier structure and parameterization form the key to making good long-tailed learners, in comparison with the less important input structure.

## Getting Started

### Requirements

- PyTorch
- Numpy
- Transformers

### Training & Evaluation

The training and evaluation are achieved in a single script. We provide example scripts for both CLS-tuning and Prompt-tuning, along with their variants.

**CLS-tuning scripts**

For example, in `scripts/run_cls_tuning_r52.sh`, we provide an example for CLS-tuning on R52. We explain some important arguments in following:
* `--model_type`: Variant to use, can be chosen from `cls_tuning`, `cls_tuning_w_focal_loss`, `cls_tuning_w_eta_norm`, `cls_tuning_w_layer_norm`, `cls_tuning_w_init_norm`, and `cls_tuning_w_prompt`.
* `--model_name_or_path`: Pretrained language models to start with.
* `--task_name`: Task to use, can be chosen from `cmid`, `iflytek`, `ctc`, `msra`, `r52`, `ecom`, `rte`, and `boolq`.
* `--data_type`: Input format to use, `combined` for CLS-tuning.
* `--activation`: Activation to use in the classifier, can be chosen from `relu` and `tanh`.
* `--model_suffix`: Additional information to add so that experiments can be better organized.

**Prompt-tuning scripts**

For example, in `scripts/run_prompt_tuning_r52.sh`, we provide an example for Prompt-tuning on R52. We explain some important arguments in following:
* `--model_type`: Variant to use, can be chosen from `prompt_tuning`, `prompt_tuning_w_focal_loss`, and `prompt_tuning_w_decoupling`.
* `--model_name_or_path`: Pretrained language models to start with.
* `--task_name`: Task to use, can be chosen from `cmid`, `iflytek`, `ctc`, `msra`, `r52`, `ecom`, `rte`, and `boolq`.
* `--data_type`: Input format to use, `prompted` for Prompt-tuning.
* `--template`: Template to use, should be formulated properly, e.g., `{cls}{text_a} This is {mask} news . {sep}`.
* `--verbalizer`: Verbalizer to use, should be loaded from a json file, e.g., `{"Copper": "copper", "Livestock": "livestock"}`.
* `--model_suffix`: Additional information to add so that experiments can be better organized.

**Logs**

For results in the paper, we use Nvidia V100 GPUs with CUDA 11. Using different types of devices or different versions of CUDA/other softwares may lead to slightly different performance. The experimental logs can be found in `logs` for sanity checks.

### Adapting to a New Task

**Data**

The dataset of the new task should be converted to a format similar to the format as those placed in `long_tailed_datasets`. And a new reader should be abstracted to read the dataset by mimicking those placed in `data/readers.py`. 

**Template and verbalizer**

The template should at contain the input `text_a` and (optionally) `text_b`, the special tokens `{cls}`, `{sep}`, `{mask}`, and necessary connection tokens. The verbalizer is a key-value json that maps labels to multiword expressions.

**Script**

A new script should be prepared as you like, e.g., carefully tuning the hyperparameters.

## Bugs or Questions?

If you have any questions related to the code or the paper, feel free to email Chen (`czhang@bit.edu.cn`). If you encounter any problems when using the code, or want to report a bug, you can open an issue. Please try to specify the problem with details so we can help you better and quicker!

## Citation

Please cite our paper if you use the code in your work:

```bibtex
@inproceedings{zhang2022glee,
   title={Making Pretrained Language Models Good Long-tailed Learners},
   author={Zhang, Chen and Ren, Lei and Wang, Jingang and Wu, Wei and Song, Dawei},
   booktitle={EMNLP},
   year={2022}
}
```
