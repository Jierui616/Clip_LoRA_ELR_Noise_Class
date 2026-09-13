#!/bin/bash
# 单变量实验 #2 恢复（resume sv2_fusion_neg3，从 epoch 4 best_model 续训）
# 唯一改动：model.prompt_fusion_weight -1.0 -> -3.0（sigmoid(-3.0)≈0.047，几乎关闭文本分支）
# 已完成 4 个 epoch（best 50.71%），从 epoch 5 继续到 epoch 40
set -e
cd /mnt/d/zhuomian/Clip_LoRA_ELR_Noise_Class

EXP_NAME="sv2_fusion_neg3"
mkdir -p outputs/${EXP_NAME}

source /home/jierui/anaconda3/etc/profile.d/conda.sh
conda activate clora

# 备份原 train.log（保留前 4 个 epoch 的历史）
if [ -f outputs/${EXP_NAME}/train.log ]; then
    cp outputs/${EXP_NAME}/train.log outputs/${EXP_NAME}/train.log.part1
fi

python -u main.py \
    --dataset webfg400 \
    --data_root ./data \
    --output_dir ./outputs \
    --exp_name ${EXP_NAME} \
    --seed 42 \
    --resume ./outputs/${EXP_NAME}/best_model.pth \
    --overrides model.prompt_fusion_weight=-3.0 \
    2>&1 | tee -a outputs/${EXP_NAME}/train.log
