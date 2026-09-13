#!/bin/bash
# 复现实验：用完全默认配置（不加载任何 override）重跑，验证当前代码能否复现 baseline 84.80%
# 目的：钉死"当前代码 + 默认配置"的真实基线，判断 85% 的终点跳变是否稳定可复现
set -e
cd /mnt/d/zhuomian/Clip_LoRA_ELR_Noise_Class

EXP_NAME="reproduce_default_baseline"
mkdir -p outputs/${EXP_NAME}

# 完全使用 config/default.py 默认值 + get_config("webfg400") 的 dataset 覆盖
# 不传任何 --overrides，确保与 optimized_v1 基线配置一致
# (cosine head + hd1024 + prompt learner + RandAug + MixUp/CutMix + onecycle + lr5e-4 + 40 epoch)
source /home/jierui/anaconda3/etc/profile.d/conda.sh
conda activate base

python -u main.py \
    --dataset webfg400 \
    --data_root ./data \
    --output_dir ./outputs \
    --exp_name ${EXP_NAME} \
    --seed 42 \
    2>&1 | tee outputs/${EXP_NAME}/train.log