#!/bin/bash
# P0 纠错实验：对齐社区同赛题结论（Luqhhh 仓库审计结果）
# 1) Linear head (fc, hidden=0) 替代 cosine head（社区 +6pp）
# 2) 关闭全部强增强，仅 CLIP 标准预处理（社区实测增强负收益）
# 3) 提升学习率 + cosine 调度（head lr=5e-3, lora lr=1e-3）
# 4) 关闭 prompt learner（数字类名无语义，社区已证伪）
# 5) 关闭 label refinement（社区：删除 > 重标）
# 其余保留项目框架：ELR+GCE+EMA teacher+GMM+SupCon
set -e
cd /mnt/d/zhuomian/Clip_LoRA_ELR_Noise_Class

EXP_NAME="p0_linhead_noaug_lr1e-3_cosine"
mkdir -p outputs/${EXP_NAME}

python -u main.py \
    --dataset webfg400 \
    --data_root ./data \
    --output_dir ./outputs \
    --exp_name ${EXP_NAME} \
    --seed 42 \
    --overrides \
        model.classifier_type='"fc"' \
        model.classifier_hidden_dim=0 \
        model.use_prompt_learner=False \
        model.zero_shot_init=False \
        data.use_randaug=False \
        data.use_mixup=False \
        data.use_cutmix=False \
        train.lr=1e-3 \
        train.lr_head_multiplier=5 \
        train.lr_scheduler='"cosine"' \
        train.epochs=50 \
        train.warmup_epochs=5 \
        train.early_stopping_patience=15 \
        loss.use_label_refinement=False