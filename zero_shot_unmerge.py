#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L1：零样本二元解合并 — 对每个吸收对 (C,A) 用 CLIP 零样本重新二分。

对每对 (C,A)：
  1. 取 data/train/<C> 与 <A> 两文件夹全部图片
  2. 用 CLIP 文本编码 "a photo of a <name>." + 图像编码做二元余弦
  3. 若某图对另一个类的相似度高出 margin，则判定该图被贴错 -> 记入重标注 map
输出：relabel_map（rel_path -> new_class_idx），并打印每对统计。

用法（服务器端）:
    python zero_shot_unmerge.py --data_root ./data --out relabel_map.json
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn.functional as F
from tqdm import tqdm
import open_clip
from PIL import Image

# 24 个崩溃类 -> 主吸收邻类（来自 cleanup_audit 的 top-1 去向）
PAIRS = [
    (19, 387), (39, 446), (46, 211), (179, 247), (183, 464), (264, 203),
    (270, 612), (305, 17), (318, 421), (324, 571), (407, 448), (413, 458),
    (535, 202), (555, 246), (561, 694), (589, 224), (603, 601), (640, 502),
    (671, 745), (682, 334), (723, 191), (728, 105), (730, 184), (737, 420),
]
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--out", default="relabel_map.json")
    ap.add_argument("--margin", type=float, default=0.05,
                    help="目标类相似度须高出原类至少 margin 才重标注")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载 CLIP（零样本，无 LoRA）
    pretrained = "openai"
    local = f"{args.data_root}/ViT-B-32_openai.bin"
    if Path(local).exists():
        pretrained = local
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained=pretrained)
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    train_root = Path(args.data_root) / "train"
    class_names = sorted([d.name for d in train_root.iterdir() if d.is_dir()])
    name2idx = {n: i for i, n in enumerate(class_names)}

    # 文本特征（750 路）
    prompts = [f"a photo of a {n}." for n in class_names]
    text_feats = []
    with torch.no_grad():
        for i in range(0, len(prompts), 256):
            t = tokenizer(prompts[i:i+256]).to(device)
            f = F.normalize(model.encode_text(t), dim=-1)
            text_feats.append(f.cpu())
    text_feats = torch.cat(text_feats, dim=0)  # (750, D)

    @torch.no_grad()
    def encode_images(paths):
        feats = []
        for i in range(0, len(paths), args.batch_size):
            batch = []
            for p in paths[i:i+args.batch_size]:
                try:
                    img = preprocess(Image.open(p).convert("RGB"))
                except Exception:
                    img = torch.zeros(3, 224, 224)
                batch.append(img)
            x = torch.stack(batch).to(device)
            f = F.normalize(model.encode_image(x), dim=-1)
            feats.append(f.cpu())
        return torch.cat(feats, dim=0)

    relabel_map = {}
    print(f"margin={args.margin:.3f}")
    print(f"{'对(C->A)':>14} {'C图数':>6} {'A图数':>6} | {'C中被判A':>8} {'A中被判C':>8} | 判定")
    for C, A in PAIRS:
        cp = train_root / class_names[C]
        ap_ = train_root / class_names[A]
        c_files = [f for f in cp.iterdir() if f.suffix.lower() in IMG_EXT]
        a_files = [f for f in ap_.iterdir() if f.suffix.lower() in IMG_EXT]
        all_files = c_files + a_files
        if not all_files:
            continue
        feats = encode_images([str(f) for f in all_files])
        sim = feats @ text_feats.T  # (N, 750)
        simC = sim[:, C]
        simA = sim[:, A]
        # 每图二元判定
        n_c2a = 0
        n_a2c = 0
        for j, f in enumerate(all_files):
            orig_label = C if j < len(c_files) else A
            other = A if j < len(c_files) else C
            if sim[j, other] > sim[j, orig_label] + args.margin:
                rel = str(Path(f).relative_to(train_root))
                relabel_map[rel] = other
                if orig_label == C:
                    n_c2a += 1
                else:
                    n_a2c += 1
        # 零样本对 C 文件夹的"纯净度"：多少 C 图被判为 C
        n_c_self = len(c_files) - n_c2a
        c_purity = n_c_self / len(c_files) * 100 if c_files else 0
        verdict = ("救回" if c_purity > 50 else "确认误标/难分")
        print(f"{class_names[C]}->{class_names[A]} {len(c_files):6d} {len(a_files):6d} | "
              f"{n_c2a:8d} {n_a2c:8d} | C自认{c_purity:4.1f}% {verdict}")

    out = args.out
    with open(out, "w") as f:
        json.dump(relabel_map, f, ensure_ascii=False, indent=0)
    print(f"\n共重标注 {len(relabel_map)} 张，已写入 {out}")


if __name__ == "__main__":
    main()