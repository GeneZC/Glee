# !/bin/sh

for sd in 775 776 777 778 779
do
    python run_cls.py \
        --model_type prompt_tuning \
        --model_name_or_path ../plms/bert-base-chinese \
        --task_name ecom \
        --data_type prompted \
        --template "{cls}{text_a}。这个评论中的物品{mask}好。{sep}" \
        --verbalizer "verbalizers/ecom.verbalizer" \
        --max_length 64 \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 32 \
        --learning_rate 1e-5 \
        --weight_decay 0.0 \
        --log_interval 1 \
        --num_train_epochs 5 \
        --num_patience_epochs 5 \
        --warmup_proportion 0.1 \
        --max_grad_norm 1.0 \
        --seed ${sd} \
        --selection_metric acc \
        --do_train \
        --do_test \
        --model_suffix ${1}
done
