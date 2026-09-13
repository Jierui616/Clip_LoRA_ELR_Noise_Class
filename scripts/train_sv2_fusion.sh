#!/bin/bash
# 单变量实验 #2（对照实验 #1: fusion=-1.0 → 75.81%）
# 唯一改动：model.prompt_fusion_weight -1.0 -> -3.0（sigmoid(-3.0)≈0.047，几乎关闭文本分支）
# 其余配置与 sv1 完全一致
set -e
cd /mnt/d/zhuomian/Clip_LoRA_ELR_Noise_Class

EXP_NAME="sv2_fusion_neg3"
mkdir -p outputs/${EXP_NAME}

source /home/jierui/anaconda3/etc/profile.d/conda.sh
conda activate clora

python -u main.py \
    --dataset webfg400 \
    --data_root ./data \
    --output_dir ./outputs \
    --exp_name ${EXP_NAME} \
    --seed 42 \
    --overrides model.prompt_fusion_weight=-3.0 \
    2>&1 | tee outputs/${EXP_NAME}/train.log