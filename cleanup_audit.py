#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L0/L1 地基：量化 24 个崩溃类训练图的「贴错/学错」程度。

对每个崩溃类 C，统计其训练文件夹内所有图片的模型预测分布：
  - 预测==C 的占比（模型认不认回来）
  - 预测去向 top-3（被吸收成哪个类）
  - 硬标签==C 但模型高置信判成别的 的数量（候选贴错样本）

用法（服务器端）:
    python cleanup_audit.py --resume outputs/fixres384_hd512_dsp_deft_strat/swa_model.pth \
        --data_root ./data
"""
import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import IndexedImageDataset
from data.transforms import get_test_transforms
from models import CLIPLoRAModel


CRASHED = [19, 39, 46, 179, 183, 264, 270, 305, 318, 324, 407, 413,
           535, 555, 561, 589, 603, 640, 671, 682, 723, 728, 730, 737]


def build_model(checkpoint, pretrained_path, num_classes, device, image_size, class_names):
    cm = checkpoint["config"].model
    return CLIPLoRAModel(
        num_classes=num_classes, image_size=image_size, pretrained=pretrained_path,
        lora_r=cm.lora_r, lora_alpha=cm.lora_alpha, lora_dropout=cm.lora_dropout,
        classifier_type=cm.classifier_type, classifier_hidden_dim=cm.classifier_hidden_dim,
        classifier_dropout=cm.classifier_dropout, zero_shot_init=cm.zero_shot_init,
        class_names=class_names, use_prompt_learner=cm.use_prompt_learner,
        prompt_n_ctx=cm.prompt_n_ctx, prompt_fusion_weight=cm.prompt_fusion_weight,
    ).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", required=True)
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--num_classes", type=int, default=750)
    ap.add_argument("--image_size", type=int, default=384)
    ap.add_argument("--batch_size", type=int, default=128)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_path = f"{args.data_root}/ViT-B-32_openai.bin"
    if not Path(pretrained_path).exists():
        pretrained_path = "openai"

    train_root = Path(args.data_root) / "train"
    class_names = sorted([d.name for d in train_root.iterdir() if d.is_dir()])

    ckpt = torch.load(args.resume, map_location=device, weights_only=False)
    cfg = ckpt.get("config", None)
    image_size = int(getattr(cfg.model, "image_size", args.image_size)) if cfg else args.image_size
    model = build_model(ckpt, pretrained_path, args.num_classes, device, image_size, class_names)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 只为 24 个崩溃类建一个子集（用其全部训练图）
    target = set(CRASHED)
    samples = IndexedImageDataset(str(train_root), transform=None, is_train=True).samples
    sel = [i for i, (p, lbl) in enumerate(samples) if lbl in target]
    print(f"崩溃类训练图总数: {len(sel)}")

    transform = get_test_transforms(image_size=image_size, crop_size=image_size, is_tta=False)
    ds = IndexedImageDataset(str(train_root), transform=transform, is_train=True)
    from torch.utils.data import Subset
    loader = DataLoader(Subset(ds, sel), batch_size=args.batch_size, shuffle=False,
                        num_workers=8, pin_memory=True)

    # 按崩溃类聚合预测
    by_label = defaultdict(Counter)     # label C -> Counter(pred)
    by_conf = defaultdict(int)          # label C -> 高置信且判错(≠C)的数量
    by_total = defaultdict(int)
    with torch.no_grad():
        for images, labels, _ in tqdm(loader, desc="audit", leave=False):
            with torch.amp.autocast("cuda", enabled=True):
                logits = model(images.to(device))
            probs = logits.softmax(1)
            conf, preds = probs.max(1)
            for lbl, p, c in zip(labels.tolist(), preds.tolist(), conf.tolist()):
                by_total[lbl] += 1
                by_label[lbl][p] += 1
                if p != lbl and c > 0.5:
                    by_conf[lbl] += 1

    print(f"\n{'类':>5} {'总图':>5} {'预测==自己':>10} {'高置信判错(>0.5)':>14}   top3去向")
    for c_idx in CRASHED:
        tot = by_total[c_idx]
        self_n = by_label[c_idx].get(c_idx, 0)
        to3 = by_label[c_idx].most_common(3)
        s = ", ".join(f"{p}:{n}" for p, n in to3 if p != c_idx)
        print(f"{c_idx:5d} {tot:5d} {self_n:5d}({self_n/tot*100:4.1f}%) {by_conf[c_idx]:10d}   {s}")


if __name__ == "__main__":
    main()