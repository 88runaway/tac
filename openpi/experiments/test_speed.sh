#!/bin/bash
# Pi0.5 E2E speed benchmark
# Usage: bash experiments/test_speed.sh

# Pi0.5 checkpoint 目录（含 model.safetensors，LeRobot 格式）
CKPT_PATH=${CKPT_PATH:-/data1/zjb/ckpt/lerobot/pi05_base}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4} python experiments/speed_e2e.py \
    --checkpoint "${CKPT_PATH}" \
    --warmup 5 --repeats 30 \
    --image_size 256 \
    --n_images 4 \
    --num_steps 10 \
    --state_dim 14
