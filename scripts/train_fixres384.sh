#!/bin/bash
# FixRes 实验：image_size 224 -> 384（唯一变量）
# 对比 sv1_fusion_neg1（224, fusion=-1.0, 单crop 75.81%）
# 其余配置与 sv1 完全一致（batch=64, accum=2, fusion=-1.0, seed=42）
set -e
cd /mnt/d/zhuomian/Clip_LoRA_ELR_Noise_Class

EXP_NAME="fixres384"
mkdir -p outputs/${EXP_NAME}

source /home/jierui/anaconda3/etc/profile.d/conda.sh
conda activate clora

python -u main.py \
    --dataset webfg400 \
    --data_root ./data \
    --output_dir ./outputs \
    --exp_name ${EXP_NAME} \
    --seed 42 \
    --overrides model.image_size=384 data.image_size=384 model.prompt_fusion_weight=-1.0 \
    2>&1 | tee outputs/${EXP_NAME}/train.log