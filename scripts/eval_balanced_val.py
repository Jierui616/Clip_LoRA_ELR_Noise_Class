#!/usr/bin/env python3
"""
均衡验证集复评：按每类固定 N 个样本建类均衡验证集，评测模型。

背景：官方测试集是「类别均衡 + 人工精确标注」，而训练时的 val 是
random_split 10%（长尾 + 含噪标签），两者分布不一致，导致历史结论
（平衡采样/logit adjustment 有害等）可能是在错误的尺子上测得。

本脚本：每类抽 N 个样本，得到与官方测试集同分布（均衡）的验证集，
评测 single-crop 与 TTA 的 top1 / mean-per-class accuracy，并可选
logit adjustment（长尾训练先验 -> 均匀测试先验的贝叶斯校正）。

Usage:
    python scripts/eval_balanced_val.py \
        --resume outputs/fixres384_hd512_dsp_deft/swa_model.pth \
        --data_root ./data --per_class 10 --mode both --logit_adjust
"""

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data.dataset import IndexedImageDataset, stratified_split
from data.transforms import get_test_transforms
from models import CLIPLoRAModel


def build_model(checkpoint, pretrained_path, num_classes, device, image_size, class_names):
    config = checkpoint.get("config", None)
    if config is not None:
        cm = config.model
        return CLIPLoRAModel(
            num_classes=num_classes,
            image_size=image_size,
            pretrained=pretrained_path,
            lora_r=cm.lora_r,
            lora_alpha=cm.lora_alpha,
            lora_dropout=cm.lora_dropout,
            classifier_type=cm.classifier_type,
            classifier_hidden_dim=cm.classifier_hidden_dim,
            classifier_dropout=cm.classifier_dropout,
            zero_shot_init=cm.zero_shot_init,
            class_names=class_names,
            use_prompt_learner=cm.use_prompt_learner,
            prompt_n_ctx=cm.prompt_n_ctx,
            prompt_fusion_weight=cm.prompt_fusion_weight,
        ).to(device)
    return CLIPLoRAModel(
        num_classes=num_classes, image_size=image_size, pretrained=pretrained_path,
        class_names=class_names,
    ).to(device)


def balanced_indices(samples, candidate_idx, num_classes, per_class, seed):
    """在 candidate_idx（分层 held-out 索引）内每类抽 per_class 个样本。"""
    by_class = defaultdict(list)
    for i in candidate_idx:
        by_class[samples[i][1]].append(i)

    rng = random.Random(seed)
    sel = []
    counts = {}
    for c in range(num_classes):
        idxs = by_class.get(c, [])
        k = min(per_class, len(idxs))
        if k > 0:
            sel.extend(rng.sample(idxs, k))
        counts[c] = k
    return sel, counts


@torch.no_grad()
def run_eval(model, loader, device, num_classes, is_tta, adj=None):
    """自定义评测循环，支持 single/TTA + logit adjustment。"""
    correct = 0
    total = 0
    class_correct = torch.zeros(num_classes, device=device)
    class_total = torch.zeros(num_classes, device=device)

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        images, labels, _ = batch
        labels = labels.to(device)
        if is_tta:
            bs, nc, c, h, w = images.shape
            images = images.view(-1, c, h, w).to(device)
            chunks = []
            for i in range(0, images.size(0), 512):
                chunk = images[i:i + 512]
                with torch.amp.autocast("cuda", enabled=True):
                    chunks.append(model(chunk))
            logits = torch.cat(chunks, dim=0).view(bs, nc, -1).mean(dim=1)
        else:
            images = images.to(device)
            with torch.amp.autocast("cuda", enabled=True):
                logits = model(images)

        if adj is not None:
            logits = logits - adj.unsqueeze(0)

        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        for c in range(num_classes):
            m = (labels == c)
            class_total[c] += m.sum()
            if m.any():
                class_correct[c] += (preds[m] == c).sum().item()

    acc = correct / total if total > 0 else 0.0
    non_empty = class_total > 0
    per_class = class_correct[non_empty] / class_total[non_empty]
    return {
        "top1_acc": acc,
        "num_samples": total,
        "mean_per_class_acc": per_class.mean().item(),
        "min_per_class_acc": per_class.min().item(),
        "max_per_class_acc": per_class.max().item(),
        "zero_correct_classes": int((class_total > 0).sum().item() - (per_class > 0).sum().item()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", required=True)
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--num_classes", type=int, default=750)
    parser.add_argument("--per_class", type=int, default=10)
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--mode", choices=["single", "tta", "both"], default="both")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logit_adjust", action="store_true",
                        help="减去训练集类频率先验 log(n_y)，校正到均匀测试分布")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="分层 held-out 比例（须与训练时一致）")
    parser.add_argument("--split_seed", type=int, default=42,
                        help="分层 split 随机种子（须与训练时一致）")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    pretrained_path = f"{args.data_root}/ViT-B-32_openai.bin"
    if not Path(pretrained_path).exists():
        pretrained_path = "openai"

    train_root = Path(args.data_root) / "train"
    class_names = sorted([d.name for d in train_root.iterdir() if d.is_dir()])

    print(f"Loading checkpoint: {args.resume}")
    checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
    cfg = checkpoint.get("config", None)
    image_size = int(getattr(cfg.model, "image_size", args.image_size)) if cfg else args.image_size
    print(f"image_size={image_size}, checkpoint val_acc={checkpoint.get('val_acc')}")

    model = build_model(checkpoint, pretrained_path, args.num_classes, device, image_size, class_names)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 复现训练时的分层 held-out split（与 get_dataloaders 同一函数/seed）
    full_ref = IndexedImageDataset(str(train_root), transform=None, is_train=True)
    train_idx, val_idx = stratified_split(
        full_ref.samples, val_split=args.val_split, seed=args.split_seed
    )
    print(f"分层 held-out: train={len(train_idx)} val={len(val_idx)} "
          f"(每类至少 1 个 held-out, val_split={args.val_split})")

    # 统计全量训练集每类样本数（用于 logit adjustment）
    adj = None
    if args.logit_adjust:
        cnt = torch.zeros(args.num_classes)
        for _, lbl in full_ref.samples:
            cnt[lbl] += 1
        adj = torch.log(cnt + 1e-6).to(device)
        print(f"logit_adjust: class count min={cnt.min().item():.0f} "
              f"max={cnt.max().item():.0f} mean={cnt.mean().item():.1f}")

    modes = (["single", "tta"] if args.mode == "both" else [args.mode])
    for mode in modes:
        is_tta = (mode == "tta")
        transform = get_test_transforms(
            image_size=image_size, crop_size=image_size, is_tta=is_tta
        )
        dataset = IndexedImageDataset(str(train_root), transform=transform, is_train=True)
        sel, counts = balanced_indices(dataset.samples, val_idx, args.num_classes, args.per_class, args.seed)
        n_empty = sum(1 for c, k in counts.items() if k == 0)
        print(f"\n[mode={mode}] 均衡验证集: {len(sel)} 样本, "
              f"每类目标 {args.per_class}, 空类 {n_empty}/{args.num_classes}")

        loader = DataLoader(
            Subset(dataset, sel),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )

        res = run_eval(model, loader, device, args.num_classes, is_tta, adj=adj)
        tag = "TTA" if is_tta else "Single"
        adj_tag = " + logit_adjust" if adj is not None else ""
        print(f"{tag} 均衡验证 Top-1{adj_tag}: {res['top1_acc']*100:.2f}%")
        print(f"  mean_per_class={res['mean_per_class_acc']*100:.2f}%  "
              f"min={res['min_per_class_acc']*100:.2f}%  max={res['max_per_class_acc']*100:.2f}%  "
              f"全错类个数={res['zero_correct_classes']}  (n={res['num_samples']})")


if __name__ == "__main__":
    main()