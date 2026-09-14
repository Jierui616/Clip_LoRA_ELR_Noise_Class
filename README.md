# CLIP + LoRA + ELR: 面向噪声标签数据的细粒度图像识别鲁棒微调

<div align="center">

[![GitHub Stars](https://img.shields.io/github/stars/Jierui616/Clip_LoRA_ELR_Noise_Class?style=flat-square&logo=github&color=ffd700)](https://github.com/Jierui616/Clip_LoRA_ELR_Noise_Class/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/Jierui616/Clip_LoRA_ELR_Noise_Class?style=flat-square&logo=github&color=63666E)](https://github.com/Jierui616/Clip_LoRA_ELR_Noise_Class/network/members)
[![Last Commit](https://img.shields.io/github/last-commit/Jierui616/Clip_LoRA_ELR_Noise_Class?style=flat-square&logo=github&color=blue)](https://github.com/Jierui616/Clip_LoRA_ELR_Noise_Class/commits/main)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CLIP ViT-B/32](https://img.shields.io/badge/Backbone-CLIP%20ViT--B%2F32-8A2BE2?style=flat-square)](https://github.com/mlfoundations/open_clip)
[![Top-1 Acc](https://img.shields.io/badge/Top--1%20Acc-78.64%25-brightgreen?style=flat-square)]()
[![License](https://img.shields.io/badge/License-Academic%20Use-blue?style=flat-square)]()

</div>

```bash
# 在 Ubuntu 终端（WSL）中激活 conda 环境
conda activate clora
```

> **赛题来源**: [全球校园人工智能算法精英大赛 - 赛题四：网络监督细粒度图像识别](https://www.aicomp.cn/tracks/3633.html)

## 📋 项目概述

本项目针对**网络爬取带噪声标签数据**的细粒度图像识别任务，采用 **CLIP ViT-B/32** 作为骨干网络，结合 **LoRA (Low-Rank Adaptation)** 高效微调、**ELR (Early-Learning Regularization)** 噪声标签鲁棒训练，实现在不依赖大规模人工标注条件下的高精度细粒度分类。
 

### 核心挑战

- **标签噪声**: 网络爬取数据未经人工核验，存在大量错误标签
- **数据偏差**: 网络图像分布与真实分布存在偏差
- **长尾分布**: 类别间样本数极度不均衡
- **细粒度差异**: 子类别间差异微小（如不同型号的飞机、汽车）

### 技术路线 (v2.0 — 2026-06-30 前沿改进)

```
CLIP ViT-B/32 (frozen backbone)
    ↓
LoRA Adapters (q_proj, v_proj, k_proj, out_proj — 仅训练 ~1% 参数)
    ↓
Visual Features (512-dim, L2-normalized)
    ↓                                ↓
Cosine Classifier              Prompt Learner (CoOp)
(余弦原型,噪声鲁棒)            (可学习文本提示,融合CLIP文本知识)
    ↓                                ↓
    └────────── Learnable Fusion ─────┘
                    ↓
           Combined Logits
                    ↓
    ┌───────────────┴───────────────┐
    ↓                               ↓
ELR Loss (EMA目标, 防止噪声记忆)   GCE Loss (噪声鲁棒)
    ↓                               ↓
    └──→ GMM Sample Selection ──→──┘
    (DivideMix式动态分离干净/噪声样本)
                    ↓
    ┌───────────────┴───────────────┐
    ↓                               ↓
EMA Teacher Distillation     SupCon Contrastive
(参数滑动平均教师)           (干净样本特征拉近同类)
                    ↓
            Final Loss → Backward
```

## 📊 数据集

比赛使用南京理工大学公开数据集（脱敏处理后提供）：

| 数据集 | 类别数 | 训练图片 | 测试图片 | 内容 |
|--------|--------|----------|----------|------|
| **WebFG-400** | 400 | 43,087 | 13,882 | 飞机(100类)、汽车(196类)、鸟类(200类) |
| **WebiNat-5000** | 5,000 | 580,865 | 100,000 | 植物、昆虫、鸟类、爬行动物、真菌等 |

数据划分为初赛 A 榜和复赛 B 榜，训练数据按类别文件夹组织，测试集为均匀分布的人工精标数据。

### 数据准备

数据集已解压至 `data/` 目录，结构如下：
data/
├── train/                      # 训练数据（按类别分文件夹）
│   ├── class_001/
│   │   ├── img1.jpg
│   │   └── ...
│   ├── class_002/
│   └── ...
└── test/                       # 测试数据（平铺图片，无标签）
    ├── test_00001.jpg
    ├── test_00002.jpg
    └── ...
```

> **提示**: WebFG-400 和 WebiNat-5000 各有一套 train.zip + test.zip，分别解压到不同目录或用 `--data_root` 指定路径即可。

## 🆕 v2.0 前沿改进 (2026-06-30)

| 改进 | 方法 | 预期收益 |
|------|------|----------|
| **余弦原型分类器** | Cosine Classifier with learnable prototypes | 比 FC 更鲁棒，适合细粒度任务 |
| **可学习文本提示** | CoOp-style Prompt Learner | 利用 CLIP 文本编码器的零样本知识，与视觉分支互补 |
| **GMM 动态样本筛选** | DivideMix-style 2-component GMM | 在线分离干净/噪声样本，差异化训练 |
| **EMA 教师蒸馏** | Mean Teacher with momentum α=0.999 | 参数平滑的教师模型提供稳定软标签 |
| **监督对比学习** | Noise-Robust SupCon (Khosla et al.) | 拉近同类特征，提升细粒度判别力 |
| **置信度标签修正** | Label Co-Refinement | 逐步修正潜在噪声标签 |

## 🚀 快速开始

### 环境安装 (WSL Ubuntu)

```bash
# 激活预先配置的 conda 环境（依赖已安装完毕）
conda activate clora
```

### 快速验证 (测试数据，5 分钟跑通)

```bash
# 一键测试：创建合成数据 → 训练 → 评估
bash scripts/quick_test.sh
```

### 训练

```bash
# WebFG-400 训练
bash scripts/train_webfg400.sh

# WebiNat-5000 训练
bash scripts/train_webinat5000.sh

# 或自定义参数
python main.py \
    --dataset webfg400 \
    --data_root ./data \
    --output_dir ./outputs \
    --exp_name my_experiment \
    --overrides train.lr=1e-4 train.batch_size=64 train.epochs=100
```

### 评估

```bash
# 评估最佳模型
bash scripts/eval.sh ./outputs/webfg400_r16_lambda3/best_model.pth webfg400

# 仅评估（不训练）
python main.py --eval_only --resume ./outputs/best_model.pth --dataset webfg400
```

### 噪声检测

```bash
# 训练后自动检测噪声标签
python main.py --detect_noise --dataset webfg400 --data_root ./data/WebFG-400
```

## ⚙️ 核心配置

| 超参数 | 默认值 | 说明 |
|--------|--------|------|
| `model.lora_r` | 16 | LoRA 秩，越大容量越大 |
| `model.lora_alpha` | 32 | LoRA 缩放因子 |
| `model.classifier_type` | fc | fc 或 cosine（余弦分类器更鲁棒） |
| `model.use_prompt_learner` | False | CoOp 可学习提示，利用 CLIP 文本编码器 |
| `model.prompt_fusion_weight` | 0.3 | 文本分支融合权重 |
| `loss.elr_lambda` | 3.0 (FG) / 5.0 (iNat) | ELR 正则化强度 |
| `loss.elr_beta` | 0.7 | EMA 动量 |
| `loss.gce_q` | 0.7 | GCE q 参数 |
| `loss.use_gmm_selection` | False | GMM 动态干净/噪声分离 |
| `loss.gmm_warmup_epochs` | 10 | GMM 预热轮数 |
| `loss.use_label_refinement` | False | 标签修正 |
| `loss.use_supcon` | False | 监督对比学习 |
| `loss.supcon_weight` | 0.1 | 对比损失权重 |
| `train.lr` | 1e-4 | 基础学习率 |
| `train.epochs` | 100 / 80 | 训练轮数 |
| `train.ema_teacher_alpha` | 0.999 | EMA 教师动量 |
| `train.use_self_distillation` | True | 自蒸馏 |

完整配置见 `config/default.py`。

## 📁 项目结构

```
Clip_LoRA_ELR_Noise_Class/
├── README.md                   # 项目说明
├── CLAUDE.md                   # AI 辅助开发文档
├── requirements.txt            # Python 依赖
├── main.py                     # 主入口
├── config/
│   ├── __init__.py
│   └── default.py              # 默认配置（所有超参数）
├── data/
│   ├── __init__.py
│   ├── dataset.py              # 索引化数据集（支持 ELR 的样本级目标）
│   ├── transforms.py           # 数据增强（RandAugment, MixUp, CutMix）
│   └── sampler.py              # 平衡采样器（缓解长尾分布）
├── models/
│   ├── __init__.py
│   ├── clip_lora.py            # CLIP + LoRA 完整模型
│   ├── lora.py                 # LoRA 低秩适配器实现
│   └── classifier.py           # 分类头
├── losses/
│   ├── __init__.py
│   ├── elr.py                  # ELR 损失（防止噪声标签记忆）
│   ├── gce.py                  # 广义交叉熵损失（噪声鲁棒）
│   └── combined.py             # 组合损失策略
├── trainers/
│   ├── __init__.py
│   ├── trainer.py              # 训练循环（AMP/梯度累积/蒸馏）
│   ├── evaluator.py            # 评估器（TTA/多尺度投票）
│   └── lr_scheduler.py         # 学习率调度（Cosine+Warmup）
├── utils/
│   ├── __init__.py
│   ├── logger.py               # 日志工具
│   ├── metrics.py              # 指标计算（Top1/Top5/Per-Class）
│   ├── checkpoint.py           # 检查点保存/加载
│   ├── seed.py                 # 随机种子设置
│   └── noise_detector.py       # 噪声标签检测与分析
└── scripts/
    ├── train_webfg400.sh       # WebFG-400 训练脚本
    ├── train_webinat5000.sh    # WebiNat-5000 训练脚本
    └── eval.sh                 # 评估脚本
```

## 🔬 关键技术详解

### 1. ELR (Early-Learning Regularization)

**论文**: [Early-Learning Regularization Prevents Memorization of Noisy Labels (NeurIPS 2020)](https://arxiv.org/abs/2007.00151)

**核心思想**: DNN 在噪声标签上训练时，会先学习"干净模式"（早期学习阶段），然后才逐渐"记忆"错误标签。ELR 为每个样本维护一个预测分布的指数移动平均（EMA），当模型后期试图记忆错误标签时，EMA 目标会阻止它。

$$
\mathcal{L}_{\text{ELR}} = \mathcal{L}_{\text{CE}}(f(x), y) + \lambda \cdot \log(1 - \langle t_{\text{ema}}, f(x) \rangle)
$$

### 2. LoRA (Low-Rank Adaptation)

**核心思想**: 冻结预训练权重 W，添加低秩可训练矩阵 A, B：

$$
W' = W + \frac{\alpha}{r} \cdot B A
$$

其中 $A \in \mathbb{R}^{r \times d_{\text{in}}}$, $B \in \mathbb{R}^{d_{\text{out}} \times r}$，$r \ll \min(d_{\text{in}}, d_{\text{out}})$。仅训练约 1% 的参数，大幅节省显存并保持预训练知识。

### 3. GCE (Generalized Cross-Entropy)

**论文**: [Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels (NeurIPS 2018)](https://arxiv.org/abs/1805.07836)

在 CE（对噪声敏感但收敛快）和 MAE（对噪声鲁棒但收敛慢）之间平滑插值：

$$
\mathcal{L}_q = \frac{1 - f_y(x)^q}{q}, \quad q \in (0, 1]
$$

### 4. 自蒸馏 (Self-Distillation)

使用模型自身的指数移动平均版本作为教师，产生软标签指导学生训练，提供额外的正则化信号。

## 📈 评价指标

- **Top-1 准确率**: `正确预测数 / 总样本数`
- **最终得分**: WebFG-400 和 WebiNat-5000 准确率的均值
- **决赛**: 70% 客观评分 + 30% 主观评分（答辩 + 技术方案 + 代码文档）

## 🚫 比赛约束

- ✅ 骨干网络: CLIP ViT-B/32
- ✅ 单模型（禁止模型集成）
- ✅ 禁止使用外部数据
- ✅ 允许使用 ImageNet-1k 预训练权重
- ✅ 需要代码复现验证

## 📝 引用

```bibtex
@inproceedings{liu2020early,
  title={Early-Learning Regularization Prevents Memorization of Noisy Labels},
  author={Liu, Sheng and Niles-Weed, Jonathan and Razavian, Narges and Fernandez-Granda, Carlos},
  booktitle={NeurIPS},
  year={2020}
}

@inproceedings{zhang2018generalized,
  title={Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels},
  author={Zhang, Zhilu and Sabuncu, Mert R},
  booktitle={NeurIPS},
  year={2018}
}

@article{radford2021learning,
  title={Learning Transferable Visual Models From Natural Language Supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and others},
  journal={ICML},
  year={2021}
}

@article{hu2021lora,
  title={LoRA: Low-Rank Adaptation of Large Language Models},
  author={Hu, Edward J and Shen, Yelong and Wallis, Phillip and others},
  journal={ICLR},
  year={2022}
}
```

## 📄 License

本项目仅用于学术研究和竞赛目的。
