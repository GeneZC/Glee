# -*- coding: utf-8 -*-

import os
import re
import time
import math
import argparse

import torch
import torch.distributed as dist
import torch.cuda.amp as amp
from torch.utils.data import RandomSampler, SequentialSampler, DistributedSampler, DataLoader
from torch.nn.parallel import DistributedDataParallel

import transformers
from transformers import AdamW, get_scheduler

from tqdm.auto import tqdm

from data import get_reader_class, get_builder_class, get_collator_class
from metrics import get_metric_fn
from models import get_model_class
from utils import set_seed, add_kwargs_to_config, keep_recent_ckpt, Logger, AverageMeter

from torch.utils.tensorboard import SummaryWriter

logger = Logger()


def gather(tensor, num_instances):
    output_tensors = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    output = concat[:num_instances] # Truncate dummy elements added by DistributedSampler.
    return output


"""
GLUE parameter setting
max_length 128
train_batch_size 32
learning_rate {1e-5, 2e-5, 3e-5, 5e-5}
num_train_epoch {3, 5, 10} CoLA 25
warmup_proportion 0.1 
weight_decay 0.01
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a classification task.")
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        help="Type of pretrained model, for indexing model class.",   
    )
    parser.add_argument( # We'd better download the model for ease of use.
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",    
    )
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        help="The task to train on, for indexing data reader.",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        required=True,
        help="Type of formatted data, for indexing data builder and collator.",
    )
    parser.add_argument( # {cls}{text_a}这里的{text_b}看起来{mask}好。{sep}
        "--template",
        type=str,
        default="",
        help="Template for constructing the prompt.",
    )
    parser.add_argument( # {"-1": "不", "0": "较", "1": "很"}
        "--verbalizer",
        type=str,
        default="",
        help="Verbalizer for constructing the prompt.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="long_tailed_datasets",
        help="Where to load a glue dataset.",
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="outputs", 
        help="Where to store the final model.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded."
        ),
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=32,
        help="Batch size (per device) for the training loader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=32,
        help="Batch size (per device) for the evaluation loader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--log_interval", type=int, default=1000, help="Interval of logging and possible saving.")
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Total number of training epochs to perform.")
    parser.add_argument("--num_patience_epochs", type=int, default=2, help="Total number of patience epochs for early stop.")
    parser.add_argument(
        "--num_grad_accum_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--warmup_proportion", type=float, default=0.1, help="Proportion of the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--max_grad_norm", type=float, default=1.0, help="Maximum norm of gradients."
    )
    parser.add_argument(
        "--selection_metric", type=str, default="acc_and_f1", help="Metric for selection criterion."
    )
    parser.add_argument("--seed", type=int, default=776, help="A seed for reproducible training.")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training.")
    parser.add_argument("--use_fp16", action="store_true", help="Use FP16 or not.")
    parser.add_argument("--use_cpu", action="store_true", help="Use CPU or not.")
    parser.add_argument("--do_train", action="store_true", help="Do train or not.")
    parser.add_argument("--do_test", action="store_true", help="Do test or not.")
    parser.add_argument("--activation", type=str, default="tanh", help="Activation function for CLS head.")
    parser.add_argument("--model_suffix", type=str, default="none", help="Suffix for outputs.")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    args.output_dir = os.path.join(args.output_dir, f"{args.model_type}_{args.model_suffix}_{args.task_name}_{args.seed}")
    os.makedirs(args.output_dir, exist_ok=True)
    args.data_dir = os.path.join(args.data_dir, args.task_name)

    is_dist = (args.local_rank != -1)
    is_main = (args.local_rank == -1 or args.local_rank == 0)
    is_fp16 = is_dist and args.use_fp16
    device = torch.device("cpu") if args.use_cpu else torch.device("cuda")

    if is_dist:
        # Initialize DDP
        dist.init_process_group(backend='nccl')
        # Pin GPU to be used to process local rank (one GPU per process)
        torch.cuda.set_device(args.local_rank)

    # Setup logging, we only want one process per machine to log things on the screen.
    logger.add_stream_handler()
    logger.add_file_handler(args.output_dir)
    if is_main:
        logger.set_verbosity_info() 
        #summary = SummaryWriter(args.output_dir)
    else:
        logger.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Load metric functin and data reader.
    metric_fn = get_metric_fn(args.task_name)
    data_reader = get_reader_class(args.task_name)(args.data_dir)
    label_map, num_labels = data_reader.get_label_map()
    
    # Train is conducted in certain accelaration.
    if args.do_train:
        # Find tokens to add from the template.
        tokens_to_add = re.findall(r"{p\d+}", args.template)
        tokens_to_add = [t.strip("{").strip("}") for t in tokens_to_add]
        tokens_to_add = [f"[{t.upper()}]" for t in tokens_to_add]

        # Load pretrained tokenizer with necessary resizing.
        tokenizer_class, config_class, model_class = get_model_class(args.model_type)
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
        # It is safe to add an empty list of tokens.
        tokenizer.add_tokens(tokens_to_add)
        
        # Data pipeline.
        data_builder = get_builder_class(args.data_type)(tokenizer, label_map, args.max_length)
        data_collator = get_collator_class(args.data_type)(tokenizer, args.max_length)

        config = config_class.from_pretrained(args.model_name_or_path)
        add_kwargs_to_config(config, activation=args.activation, num_labels=num_labels, num_added_tokens=len(tokens_to_add), orig_vocab_size=config.vocab_size)
        model = model_class.from_pretrained(
            args.model_name_or_path,
            config=config,
        )
        model.resize_token_embeddings(len(tokenizer)) 
        # NOTE: ``config.vocab_size'' has also been modified secretly while resizing the embeddings,
        # so that subsequent initializations with the config could perfectly fit any fine-tuned checkpoints.
        model = model.to(device)
        if is_dist:
            model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

        train_examples = data_reader.get_train_examples()
        train_instances = data_builder.build(train_examples, template=args.template, verbalizer=args.verbalizer)

        dev_examples = data_reader.get_dev_examples()
        dev_instances = data_builder.build(dev_examples, template=args.template, verbalizer=args.verbalizer)

        if is_dist:
            train_sampler = DistributedSampler(train_instances, shuffle=True)
        else:
            train_sampler = RandomSampler(train_instances)
        train_loader = DataLoader(train_instances, batch_size=args.per_device_train_batch_size, sampler=train_sampler, collate_fn=data_collator)
        
        if is_dist:
            dev_sampler = DistributedSampler(dev_instances, shuffle=False)
        else:
            dev_sampler = SequentialSampler(dev_instances)
        dev_loader = DataLoader(dev_instances, batch_size=args.per_device_eval_batch_size, sampler=dev_sampler, collate_fn=data_collator)

        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "LayerNorm.weight"]
        grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(grouped_parameters, lr=args.learning_rate)

        # Note -> the training loader needs to be prepared before we grab his length below (cause its length will be
        # shorter in multiprocess)

        # Scheduler and math around the number of training steps.
        num_update_steps_per_epoch = math.ceil(len(train_loader) / args.num_grad_accum_steps)
        num_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        num_patience_steps = args.num_patience_epochs * num_update_steps_per_epoch
        num_warmup_steps = int(num_train_steps * args.warmup_proportion)

        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps,
        )

        # Train!
        total_batch_size = args.per_device_train_batch_size * args.num_grad_accum_steps
        if is_dist:
            total_batch_size = total_batch_size * dist.get_world_size()

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_examples)}")
        logger.info(f"  Num epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. accumulation, parallel & distributed) = {total_batch_size}")
        logger.info(f"  Gradient accumulation steps = {args.num_grad_accum_steps}")
        logger.info(f"  Total optimization steps = {num_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(num_train_steps), disable=not (is_main or not is_dist))
        num_completed_steps = 0
        train_losses = AverageMeter()
        best_dev_step = 0
        best_dev_path = ""
        best_dev_metric = {}

        if is_fp16:
            scaler = amp.GradScaler()

        for epoch in range(args.num_train_epochs):
            # Set the epoch as the seed if dist; otherwise each epoch is shuffled with the same seed.
            if is_dist:
                train_loader.sampler.set_epoch(epoch) 
            for step, batch in enumerate(train_loader):
                model.train()
                batch = [v.to(device) for k, v in batch._asdict().items()]
                if is_fp16:
                    with amp.autocast():
                        output = model(batch)
                else:
                    output = model(batch)
                loss = output.loss.mean()
                train_losses.update(loss.item())
                loss = loss / args.num_grad_accum_steps
                if is_fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                if step % args.num_grad_accum_steps == 0 or step == len(train_loader) - 1:
                    if is_fp16:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        scaler.step(optimizer) # Will check whether the gradients are unscaled or not before stepping.
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    num_completed_steps += 1

                
                if num_completed_steps % args.log_interval == 0:
                    logger.info("***** Running evaluation *****")
                    logger.info(f"  Num completed epochs = {epoch}")
                    logger.info(f"  Num completed steps = {num_completed_steps}")
                    model.eval()
                    with torch.no_grad():
                        losses, preds, labels = [], [], []
                        for batch in dev_loader:
                            batch = [v.to(device) for k, v in batch._asdict().items()]
                            output = model(batch)
                            loss, pred, label = output.loss, output.prediction, output.label
                            if is_dist:
                                losses.extend(gather(loss).cpu().numpy().tolist())
                                preds.extend(gather(pred).cpu().numpy().tolist())
                                labels.extend(gather(label).cpu().numpy().tolist())
                            else:
                                losses.extend(loss.cpu().numpy().tolist())
                                preds.extend(pred.cpu().numpy().tolist())
                                labels.extend(label.cpu().numpy().tolist())

                    dev_metric = metric_fn(preds, labels)
                    logger.info(f"  Train loss = {train_losses.avg}")
                    logger.info(f"  Dev metric = {dev_metric}")

                    if not best_dev_metric or dev_metric[args.selection_metric] > best_dev_metric[args.selection_metric]:
                        best_dev_step = num_completed_steps
                        best_dev_metric.update(**dev_metric)
                        if is_main:
                            time_stamp = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()) 
                            best_dev_path = os.path.join(args.output_dir, \
                                f"ckpt-{num_completed_steps}-{time_stamp}")
                            tokenizer.save_pretrained(best_dev_path)
                            config.save_pretrained(best_dev_path)
                            if is_dist:
                                model_to_save = model.module
                            else:
                                model_to_save = model
                            model_to_save.save_pretrained(best_dev_path)
                            keep_recent_ckpt(args.output_dir, 1)

                if num_completed_steps - best_dev_step >= num_patience_steps:
                    logger.info("***** Early stopping *****")
                    break
            # If early stop, then break the outer loop.
            else:
                continue
            break          

        logger.info("***** Finalizing training *****")
        logger.info(f"  Best dev step = {best_dev_step}")
        logger.info(f"  Best dev metric = {best_dev_metric}")

    # Test is only conducted in the main process.
    if args.do_test and is_main:
        try:
            model_path = best_dev_path
        except:
            model_path = args.model_name_or_path

        # Load pretrained tokenizer with necessary resizing.
        tokenizer_class, config_class, model_class = get_model_class(args.model_type)
        tokenizer = tokenizer_class.from_pretrained(model_path, never_split=[f"[unused{x}]" for x in range(100)])
        
        # Data pipeline.
        data_builder = get_builder_class(args.data_type)(tokenizer, label_map, args.max_length)
        data_collator = get_collator_class(args.data_type)(tokenizer, args.max_length)
        
        config = config_class.from_pretrained(model_path)
        model = model_class.from_pretrained(
            model_path,
            config=config,
        )
        model = model.to(device)

        test_examples = data_reader.get_test_examples()
        test_instances = data_builder.build(test_examples, template=args.template, verbalizer=args.verbalizer)
        
        test_sampler = SequentialSampler(test_instances)
        test_loader = DataLoader(test_instances, batch_size=args.per_device_eval_batch_size, sampler=test_sampler, collate_fn=data_collator)

        # Test!
        logger.info("***** Running testing *****")
        model.eval()
        with torch.no_grad():
            losses, preds, labels = [], [], []
            for batch in test_loader:
                batch = [v.to(device) for k, v in batch._asdict().items()]
                output = model(batch)
                loss, pred, label = output.loss, output.prediction, output.label
                losses.extend(loss.cpu().numpy().tolist())
                preds.extend(pred.cpu().numpy().tolist())
                labels.extend(label.cpu().numpy().tolist())

        test_metric = metric_fn(preds, labels)
        logger.info("***** Finalizing testing *****") 
        logger.info(f"  Test metric = {test_metric}")
    

if __name__ == "__main__":
    """
    1. Single-Node multi-process distributed training

    ::

        >>> python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE
                YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3 and all other
                arguments of your training script)

    2. Multi-Node multi-process distributed training: (e.g. two nodes)


    Node 1: *(IP: 192.168.1.1, and has a free port: 1234)*

    ::

        >>> python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE
                --nnodes=2 --node_rank=0 --master_addr="192.168.1.1"
                --master_port=1234 YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3
                and all other arguments of your training script)

    Node 2:

    ::

        >>> python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE
                --nnodes=2 --node_rank=1 --master_addr="192.168.1.1"
                --master_port=1234 YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3
                and all other arguments of your training script)
    """
    main()
