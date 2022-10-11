# !/bin/sh

for sd in 775 776 777 778 779
do
    python run_cls.py \
        --model_type cls_tuning_w_focal_loss \
        --model_name_or_path ../plms/bert-base-chinese \
        --task_name msra \
        --data_type combined \
        --template "" \
        --verbalizer "" \
        --max_length 128 \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 32 \
        --learning_rate 1e-5 \
        --weight_decay 0.0 \
        --log_interval 1000 \
        --num_train_epochs 10 \
        --num_patience_epochs 2 \
        --warmup_proportion 0.1 \
        --max_grad_norm 1.0 \
        --seed ${sd} \
        --selection_metric f1 \
        --do_train \
        --do_test \
        --activation ${1} \
        --model_suffix ${2}
done
