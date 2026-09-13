#!/bin/bash
# 单变量实验 #1（以 74.83% 为真实基线）
# 唯一改动：model.prompt_fusion_weight 0.2 -> -1.0（sigmoid(-1.0)≈0.27）
# 其余配置与 reproduce_default_baseline 完全一致
set -e
cd /mnt/d/zhuomian/Clip_LoRA_ELR_Noise_Class

EXP_NAME="sv1_fusion_neg1"
mkdir -p outputs/${EXP_NAME}

source /home/jierui/anaconda3/etc/profile.d/conda.sh
conda activate clora

python -u main.py \
    --dataset webfg400 \
    --data_root ./data \
    --output_dir ./outputs \
    --exp_name ${EXP_NAME} \
    --seed 42 \
    --overrides model.prompt_fusion_weight=-1.0 \
    2>&1 | tee outputs/${EXP_NAME}/train.log