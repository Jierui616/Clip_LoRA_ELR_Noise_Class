#!/usr/bin/env python3
"""
Generate test set predictions and pred_results.csv for competition submission.

Usage:
    python scripts/predict_test.py \
        --resume outputs/fast30/best_model.pth \
        --data_root ./data \
        --output pred_results.csv
"""

import argparse
import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from config import get_config
from models import CLIPLoRAModel
from data.dataset import IndexedImageDataset
from data.transforms import get_test_transforms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", required=True, help="Path to best model checkpoint")
    parser.add_argument("--data_root", default="./data", help="Data root directory")
    parser.add_argument("--output", default="pred_results.csv", help="Output CSV file")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--use_tta", action="store_true", help="Use TTA for better accuracy")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint
    checkpoint = torch.load(args.resume, map_location=device)
    config = checkpoint.get("config", None)

    num_classes = 500
    # FixRes 分辨率提升实验（224->384）的模型需按 checkpoint 中的 image_size 重建，
    # 否则位置嵌入尺寸不匹配会导致 load_state_dict 失败。
    image_size = int(getattr(config.model, "image_size", 224)) if config is not None else 224
    print(f"image_size: {image_size}")

    pretrained_path = f"{args.data_root}/ViT-B-32_openai.bin"
    if not Path(pretrained_path).exists():
        pretrained_path = "openai"

    # Build model (use config from checkpoint if available)
    print("Loading model...")
    if config is not None:
        cfg_model = config.model
        model = CLIPLoRAModel(
            num_classes=num_classes,
            image_size=image_size,
            pretrained=pretrained_path,
            lora_r=cfg_model.lora_r,
            lora_alpha=cfg_model.lora_alpha,
            lora_dropout=cfg_model.lora_dropout,
            classifier_type=cfg_model.classifier_type,
            classifier_hidden_dim=cfg_model.classifier_hidden_dim,
            classifier_dropout=cfg_model.classifier_dropout,
            zero_shot_init=cfg_model.zero_shot_init,
            use_prompt_learner=cfg_model.use_prompt_learner,
            prompt_n_ctx=cfg_model.prompt_n_ctx,
            prompt_fusion_weight=cfg_model.prompt_fusion_weight,
        )
    else:
        model = CLIPLoRAModel(
            num_classes=num_classes,
            image_size=image_size,
            pretrained=pretrained_path,
            lora_r=64, lora_alpha=64,
            classifier_type="cosine",
            classifier_hidden_dim=1024,
            zero_shot_init=True,
            use_prompt_learner=True,
            prompt_n_ctx=16,
            prompt_fusion_weight=0.2,
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"Model loaded from {args.resume}")

    # Test dataset
    test_transform = get_test_transforms(
        image_size=image_size,
        crop_size=image_size,
        is_tta=args.use_tta,
    )
    test_dataset = IndexedImageDataset(
        root=f"{args.data_root}/test",
        transform=test_transform,
        is_train=False,
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=2, pin_memory=True)

    # Generate predictions
    print(f"Predicting {len(test_dataset)} test images...")
    all_filenames = []
    all_preds = []

    use_amp = device.type == "cuda"
    tta_chunk = 512  # 每次前向的最大图像数，避免 20-view 展开导致的显存峰值

    with torch.no_grad():
        for images, _, _ in tqdm(test_loader, desc="Predicting"):
            if args.use_tta:
                bs, nc, c, h, w = images.shape
                images = images.view(-1, c, h, w).to(device)
                # 分块前向（fp16），求和后对 views 取均值
                logits_chunks = []
                for i in range(0, images.size(0), tta_chunk):
                    chunk = images[i:i + tta_chunk]
                    if use_amp:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            logits_chunks.append(model(chunk))
                    else:
                        logits_chunks.append(model(chunk))
                logits = torch.cat(logits_chunks, dim=0).view(bs, nc, -1).mean(dim=1)
            else:
                images = images.to(device)
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        logits = model(images)
                else:
                    logits = model(images)

            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu())

    all_preds = torch.cat(all_preds)

    # Write pred_results.csv
    print(f"Writing {len(test_dataset)} predictions to {args.output}...")
    with open(args.output, "w") as f:
        for i, (path, _) in enumerate(test_dataset.samples):
            filename = Path(path).name
            f.write(f"{filename},{all_preds[i].item()}\n")

    print(f"Done! Result saved to {args.output}")
    print(f"First 5 predictions:")
    with open(args.output) as f:
        for _ in range(5):
            print(f"  {f.readline().strip()}")


if __name__ == "__main__":
    main()
