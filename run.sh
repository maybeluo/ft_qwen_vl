#!/bin/bash

CUDA_VISIBLE_DEVICES="0" /usr/bin/python3 ft_qwen_vl/train.py \
    --model_name_or_path "Qwen/Qwen2.5-VL-3B-Instruct" \
    --data_path "./data/demo.json" \
    --output_dir "./ckpts" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 3 \
    --logging_steps 10 \
    --num_train_epochs 3 \
    --batch_size 2


# torchrun --standalone --nnodes=1 --nproc-per-node=2 ft_qwen_vl/train.py \
#     --model_name_or_path "Qwen/Qwen2.5-VL-3B-Instruct" \
#     --data_path "./data/demo.json" \
#     --output_dir "./ckpts" \
#     --save_strategy "steps" \
#     --save_steps 500 \
#     --save_total_limit 3 \
#     --logging_steps 10 \
#     --num_train_epochs 3 \
#     --batch_size 2
