
# CLAUDE.md — AI 辅助开发指南

## 项目概述

本项目是**全球校园人工智能算法精英大赛赛题四**（面向噪声标签数据的细粒度图像识别鲁棒微调）的参赛方案。

- **赛题链接**: https://www.aicomp.cn/tracks/3633.html
- **核心任务**: 利用网络爬取的带噪声标签数据，微调 CLIP ViT-B/32 实现细粒度图像识别
- **数据集**: WebFG-400 (400类, 43K训练图) + WebiNat-5000 (5000类, 580K训练图)
- **技术栈**: CLIP + LoRA + ELR + GCE + Self-Distillation
- **评价指标**: Top-1 Accuracy（两个数据集取均值）
- **硬约束**: 单模型，禁止集成，禁止外部数据，必须 CLI P ViT-B/32 骨干

## 开发环境

**所有开发/训练/推理都在 WSL (Ubuntu-22.04) 的 `clora` conda 环境中进行。**

```bash
# 在 WSL 终端中激活环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate clora

# 验证环境
python -c "import torch; print(f'torch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
# → torch: 2.5.1+cu124, CUDA: True

# 项目在 Windows 的路径，WSL 中对应的路径：
# Windows: D:\zhuomian\Clip_LoRA_ELR_Noise_Class
# WSL:     /mnt/d/zhuomian/Clip_LoRA_ELR_Noise_Class
cd /mnt/d/zhuomian/Clip_LoRA_ELR_Noise_Class

# 所有 Python 命令使用 WSL + clora 环境：
# wsl bash -c "source ~/anaconda3/etc/profile.d/conda.sh && conda activate clora && cd /mnt/d/zhuomian/Clip_LoRA_ELR_Noise_Class && python main.py ..."
```

### 核心依赖

- torch 2.5.1+cu124 (CUDA 12.4)
- open-clip-torch 3.3.0
- peft 0.19.1 (可选，我们使用自定义 LoRA)
- sklearn (GMM 噪声过滤需要)

### 预训练权重

CLIP ViT-B-32 权重已下载到本地：`data/ViT-B-32_openai.bin` (577 MB)。
HuggingFace 在国内可能被墙，代码会自动检测并使用本地权重文件。

## 项目架构

```
main.py                          # 唯一入口，支持训练/评估/噪声检测
config/default.py                # 所有可配置超参数（dataclass 结构）
data/dataset.py                  # IndexedImageDataset — 每个样本返回 (image, label, index)
data/transforms.py               # RandAugment, MixUp, CutMix, CLIP 归一化
models/clip_lora.py              # CLIPLoRAModel — 冻结 ViT + LoRA 适配器 + 可训练分类头
models/lora.py                   # LoRALinear + apply_lora_to_clip()
losses/elr.py                    # ELRLoss — 维持 (N, C) EMA 目标，防止噪声记忆
losses/gce.py                    # GCELoss — CE 与 MAE 的平滑插值
losses/combined.py               # CombinedLoss — 灵活组合 CE/ELR/GCE/Distill
trainers/trainer.py              # Trainer — AMP/梯度累积/自蒸馏/早停/日志
trainers/evaluator.py            # Evaluator — 标准评估 + TTA 多尺度投票
utils/noise_detector.py          # 训练后噪声标签检测与分析
scripts/                         # Shell 脚本（train_webfg400.sh, train_webinat5000.sh, eval.sh）
```

## 关键设计决策

### 1. 为什么需要样本索引 (index)？
ELR 损失为**每个训练样本**维护一个 EMA 目标向量 `(C,)`。由于 DataLoader 会 shuffle，每次 batch 的样本顺序不同，必须通过索引在内存中找到对应样本的 EMA 目标。因此 `IndexedImageDataset.__getitem__` 返回 `(image, label, index)`。

### 2. 为什么 MixUp/CutMix 默认关闭？
MixUp/CutMix 生成软标签（概率分布），但 ELR 的交叉熵组件需要硬标签（整数）。这是一个已知的兼容性问题。数据增强通过 RandAugment 在图像空间完成，不修改标签。

### 3. LoRA 应用在哪些层？
默认应用在 `q_proj`, `v_proj`, `k_proj`, `out_proj`（ViT 的 QKV 投影和输出投影）。这些是注意力机制的核心参数，经验证明在这些层应用 LoRA 效果最好。

### 4. WebFG-400 vs WebiNat-5000 配置差异
- WebiNat-5000 类别数多 12.5 倍，噪声更严重，因此：
  - `elr_lambda` 从 3.0 → 5.0（更强的抗噪声正则化）
  - `elr_beta` 从 0.7 → 0.9（更慢的 EMA 适应，更稳定的目标）
  - `batch_size` 从 64 → 32（受显存限制）
  - `classifier_hidden_dim` 从 512 → 1024（更多类别需要更大容量）
  - `lr_backbone_multiplier` 从 0.1 → 0.05（更多类别需要更小的骨干学习率）

### 5. 自蒸馏如何工作？
不使用单独的教师模型。训练过程中维护模型参数的 EMA 版本，用 EMA 模型的预测作为软标签蒸馏当前模型。这提供了额外的正则化信号，对噪声标签训练特别有效。

## 常见开发任务

### 添加新的损失函数
1. 在 `losses/` 下创建新文件
2. 实现 `forward(self, outputs, labels)` 或 `forward(self, indices, outputs, labels)` 接口
3. 在 `losses/__init__.py` 中导出
4. 在 `losses/combined.py` 的 `CombinedLoss.__init__` 中添加分支
5. 在 `config/default.py` 的 `LossConfig` 中添加对应超参数

### 添加新的数据增强
1. 在 `data/transforms.py` 的 `get_train_transforms()` 中添加
2. 如果是标签级增强（MixUp 类），修改 `MixUpCutMixCollator`
3. 在 `DataConfig` 中添加开关参数

### 调整 LoRA 配置
修改 `config/default.py` 中 `ModelConfig` 的以下字段：
- `lora_r`: 秩（越大容量越大，但参数更多）
- `lora_alpha`: 缩放因子
- `lora_target_modules`: 目标层名称列表

### 超参数搜索
```bash
# 通过命令行覆盖任意配置
python main.py --dataset webfg400 \
    --overrides \
        train.lr=5e-5 \
        loss.elr_lambda=4.0 \
        model.lora_r=8 \
        model.classifier_dropout=0.5
```

### 调试模式
```bash
# 小数据快速验证流程
python main.py --dataset webfg400 \
    --overrides \
        train.epochs=5 \
        train.warmup_epochs=1 \
        data.use_randaug=False \
        train.use_self_distillation=False \
        train.batch_size=16
```

## 噪声检测使用

训练完成后，ELR 的 EMA 目标可用于检测噪声标签：

```python
from utils.noise_detector import NoiseDetector

# 获取 ELR 目标矩阵 (N, C)
targets = criterion.get_elr_targets()
labels = torch.tensor([dataset.samples[i][1] for i in range(len(dataset))])

detector = NoiseDetector(targets, labels)

# 方法1: argmax 不一致
is_noisy = detector.detect_by_argmax()

# 方法2: 给定标签置信度过低
is_noisy = detector.detect_by_confidence(threshold=0.3)

# 方法3: 预测熵过高
is_noisy = detector.detect_by_entropy(percentile=90)

# 分析方法
analysis = detector.analyze_noise()
print(f"Estimated noise rate: {analysis['estimated_noise_rate']:.2%}")

# 获取干净/噪声子集
clean_idx, noisy_idx = detector.get_clean_noisy_split(method="argmax")
```

## 复现注意事项

1. **随机种子**: 使用 `--seed 42` 确保可复现
2. **数据路径**: 数据下载后放在 `data/` 目录下，见 README 中的数据准备说明
3. **CUDA 版本**: 推荐 CUDA 11.8+，PyTorch 2.0+
4. **显存需求**: WebFG-400 batch_size=64 约需 12GB，WebiNat-5000 batch_size=32 约需 16GB
5. **训练时间**: WebFG-400 约 8-12 小时 (RTX 3090)，WebiNat-5000 约 48-72 小时

## 文件修改日志

当修改项目文件时，在此记录：

| 日期 | 文件 | 修改内容 |
|------|------|----------|
| 2026-06-29 | 全部 | 初始项目框架创建 |
|        |        |          |

## 已知问题和待办事项

- [ ] 验证 open-clip-torch 版本兼容性（需 2.24+）
- [ ] 添加 WebFG-400 和 WebiNat-5000 的数据集类（当前使用通用 IndexedImageDataset）
- [ ] 实现类别名称索引到 CLIP 文本编码的映射（用于 prompt learning）
- [ ] 添加 Co-Training（协同训练）策略以进一步利用噪声样本
- [ ] 分布式训练支持（DDP/FSDP）
- [ ] 在数据到位后进行完整的训练流程测试
- [ ] 添加 Sweep 配置用于自动化超参数搜索
