#!/bin/bash

# 训练任务1，使用端口29501
CUDA_VISIBLE_DEVICES=0,1,2,3 /home/yu/anaconda3/envs/MIA/bin/torchrun --standalone --nproc_per_node=4 --master_port=29501 /home/yu/workspace/mia/train/train.py \
  --outdir=/home/yu/workspace/mia/training-runs/ \
  --data=/home/yu/celeba/img_align_celeba_112_train \
  --cond=1 --arch=ddpmpp --precond=edm

# 训练任务2，使用端口29502
CUDA_VISIBLE_DEVICES=0,1 /home/yu/anaconda3/envs/MIA/bin/torchrun --standalone --nproc_per_node=2 --master_port=29502 /home/yu/workspace/mia/train/train_2.py \
  --outdir=/home/yu/workspace/mia/training-runs/ \
  --data=/home/yu/celeba/img_align_celeba_112 \
  --cond=1 --arch=ddpmpp --precond=edm \
  --desc=model2_run


CUDA_VISIBLE_DEVICES=2,3 /home/yu/anaconda3/envs/MIA/bin/torchrun --standalone --nproc_per_node=2 --master_port=29503 /home/yu/workspace/mia/train/train_3.py \
  --outdir=/home/yu/workspace/mia/training-runs/ \
  --data=/home/yu/celeba/img_align_celeba_112 \
  --cond=1 --arch=ddpmpp --precond=edm \
  --desc=model3_run

CUDA_VISIBLE_DEVICES=2,3 /home/yu/anaconda3/envs/MIA/bin/torchrun --standalone --nproc_per_node=2 --master_port=29504 /home/yu/workspace/mia/train/train_4.py \
--outdir=/home/yu/workspace/mia/training-runs/ \
--data=/home/yu/celeba/img_align_celeba_112 \
--cond=1 --arch=ddpmpp --precond=edm \
--desc=model4_run
echo "两个训练任务已分别在端口29501和29502启动，日志分别输出到train_model1.log和train_model2.log"