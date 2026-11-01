"""
Double-Softmax Loss (DSPT: Intrinsic Gradient Suppression for Label-Noise).

Reference:
    Li et al., "Intrinsic Gradient Suppression for Label-Noise Prompt Tuning
    in Vision-Language Models", ICML 2026. arXiv:2605.00591

Key insight:
    对 logits 做两次顺序 softmax 归一化（p = softmax(z), q = softmax(p)），
    再对 q 计算交叉熵。这个"饱和区"会自适应地把高误差噪声样本的梯度压到
    接近 0（因为它们与 CLIP 强先验冲突、产生极端梯度），而对有信息量的
    样本保留有效梯度。无超参、有界、drop-in，特别适合有强先验的 VLM。

    L = -log( softmax(softmax(z))_y )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleSoftmaxLoss(nn.Module):
    """
    Double-softmax cross-entropy loss for noisy-label fine-tuning.

    两次顺序 softmax 归一化：
        p = softmax(outputs)      # 第一次：logits -> 概率
        q = softmax(p)            # 第二次：对概率再归一化（饱和区）
        loss = -log(q[y])         # 交叉熵

    性质（来自原论文）：
        - 有界：log(1 + (C-1)/e) <= L <= log(e + C - 1)
        - 对高误差噪声样本，梯度趋零（抑制记忆噪声标签）
        - 对干净/不确定但有信息量的样本，梯度保留

    Args:
        num_classes: 类别数（仅用于信息记录）
    """

    def __init__(self, num_classes: int = 400):
        super().__init__()
        self.num_classes = num_classes

    def forward(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            outputs: (B, C) 模型 logits
            labels: (B,) 硬标签，或 (B, C) 软标签

        Returns:
            double-softmax 交叉熵损失（标量）
        """
        # 第一次 softmax -> 概率 p
        p = F.softmax(outputs, dim=1)

        # 第二次 softmax（顺序概率归一化）-> q
        q = F.softmax(p, dim=1)
        q = torch.clamp(q, 1e-7, 1.0)
        log_q = torch.log(q)

        soft = labels.dim() > 1 and labels.dtype in (
            torch.float16, torch.float32, torch.float64,
        )
        if soft:
            # 软标签（MixUp/CutMix）：CE 形式
            return -(labels * log_q).sum(dim=1).mean()
        else:
            # 硬标签
            return -log_q.gather(1, labels.unsqueeze(1)).squeeze(1).mean()