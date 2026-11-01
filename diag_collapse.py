#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断：哪些类在均衡 held-out 上全错，及其训练图片数 + 混淆去向。

用法（服务器端）:
    python diag_collapse.py --resume outputs/fixres384_hd512_dsp_deft_strat/swa_model.pth \
        --data_root ./data --per_class 10
"""
import argparse
import random
import sys
from collections import defaultdict, Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

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
            num_classes=num_classes, image_size=image_size, pretrained=pretrained_path,
            lora_r=cm.lora_r, lora_alpha=cm.lora_alpha, lora_dropout=cm.lora_dropout,
            classifier_type=cm.classifier_type, classifier_hidden_dim=cm.classifier_hidden_dim,
            classifier_dropout=cm.classifier_dropout, zero_shot_init=cm.zero_shot_init,
            class_names=class_names, use_prompt_learner=cm.use_prompt_learner,
            prompt_n_ctx=cm.prompt_n_ctx, prompt_fusion_weight=cm.prompt_fusion_weight,
        ).to(device)
    return CLIPLoRAModel(
        num_classes=num_classes, image_size=image_size, pretrained=pretrained_path,
        class_names=class_names).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", required=True)
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--num_classes", type=int, default=750)
    ap.add_argument("--per_class", type=int, default=10)
    ap.add_argument("--image_size", type=int, default=384)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_split", type=float, default=0.1)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_path = f"{args.data_root}/ViT-B-32_openai.bin"
    if not Path(pretrained_path).exists():
        pretrained_path = "openai"

    train_root = Path(args.data_root) / "train"
    class_names = sorted([d.name for d in train_root.iterdir() if d.is_dir()])
    print(f"class_names={len(class_names)}")

    ckpt = torch.load(args.resume, map_location=device, weights_only=False)
    cfg = ckpt.get("config", None)
    image_size = int(getattr(cfg.model, "image_size", args.image_size)) if cfg else args.image_size
    model = build_model(ckpt, pretrained_path, args.num_classes, device, image_size, class_names)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    full_ref = IndexedImageDataset(str(train_root), transform=None, is_train=True)
    train_idx, val_idx = stratified_split(full_ref.samples, val_split=args.val_split, seed=args.seed)

    by_class = defaultdict(list)
    for i in val_idx:
        by_class[full_ref.samples[i][1]].append(i)
    rng = random.Random(args.seed)
    sel = []
    for c in range(args.num_classes):
        idxs = by_class.get(c, [])
        k = min(args.per_class, len(idxs))
        if k > 0:
            sel.extend(rng.sample(idxs, k))

    transform = get_test_transforms(image_size=image_size, crop_size=image_size, is_tta=False)
    ds = IndexedImageDataset(str(train_root), transform=transform, is_train=True)
    loader = DataLoader(Subset(ds, sel), batch_size=args.batch_size, shuffle=False,
                        num_workers=8, pin_memory=True)

    all_labels, all_preds = [], []
    with torch.no_grad():
        for images, labels, _ in tqdm(loader, desc="eval", leave=False):
            labels = labels.to(device)
            with torch.amp.autocast("cuda", enabled=True):
                logits = model(images.to(device))
            preds = logits.argmax(1)
            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())
    labels = torch.cat(all_labels)
    preds = torch.cat(all_preds)

    # 每类训练图数量
    img_cnt = Counter(lbl for _, lbl in full_ref.samples)

    class_correct = torch.zeros(args.num_classes)
    class_total = torch.zeros(args.num_classes)
    for c in range(args.num_classes):
        m = labels == c
        class_total[c] = m.sum()
        if m.any():
            class_correct[c] = (preds[m] == c).sum()

    acc = torch.where(class_total > 0, class_correct / class_total,
                      torch.zeros_like(class_total))
    rows = []
    for c in range(args.num_classes):
        if class_total[c] > 0:
            rows.append((c, acc[c].item(), img_cnt[c], int(class_total[c]),
                         int(class_correct[c])))
    rows.sort(key=lambda r: r[1])

    print(f"\n=== 最差 40 类（idx | acc% | 训练图数 | heldout数 | 正确数）===")
    for c, a, cnt, tot, cor in rows[:40]:
        flag = " <<< 全错" if cor == 0 else ""
        print(f"  {c:3d}  {a*100:5.1f}%  imgs={cnt:3d}  h={tot:2d}  ok={cor:2d}{flag}")

    # 全错类的混淆去向
    zero_cls = [c for c, a, *_ in rows if a == 0.0]
    print(f"\n=== 全错类 {len(zero_cls)} 个的混淆去向 ===")
    for c in zero_cls:
        m = labels == c
        conf = Counter(preds[m].tolist())
        top3 = conf.most_common(3)
        s = ", ".join(f"{cc}:{n}" for cc, n in top3)
        print(f"  类{c:3d}(imgs={img_cnt[c]:3d}) 被预测为 -> {s}")


if __name__ == "__main__":
    main()