#!/usr/bin/env python3
"""
Generate test set predictions and result.csv for competition submission.

Usage:
    python scripts/predict_test.py \
        --resume outputs/fast30/best_model.pth \
        --data_root ./data \
        --output result.csv
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
    parser.add_argument("--output", default="result.csv", help="Output CSV file")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--use_tta", action="store_true", help="Use TTA for better accuracy")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint
    checkpoint = torch.load(args.resume, map_location=device)
    config = checkpoint.get("config", None)

    num_classes = 500
    pretrained_path = f"{args.data_root}/ViT-B-32_openai.bin"
    if not Path(pretrained_path).exists():
        pretrained_path = "openai"

    # Build model
    print("Loading model...")
    model = CLIPLoRAModel(
        num_classes=num_classes,
        pretrained=pretrained_path,
        lora_r=32, lora_alpha=64,
        classifier_type="cosine",
        classifier_hidden_dim=512,
        zero_shot_init=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"Model loaded from {args.resume}")

    # Test dataset
    test_transform = get_test_transforms(image_size=224, is_tta=args.use_tta)
    test_dataset = IndexedImageDataset(
        root=f"{args.data_root}/test",
        transform=test_transform,
        is_train=False,
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)

    # Generate predictions
    print(f"Predicting {len(test_dataset)} test images...")
    all_filenames = []
    all_preds = []

    with torch.no_grad():
        for images, _, _ in tqdm(test_loader, desc="Predicting"):
            if args.use_tta:
                bs, nc, c, h, w = images.shape
                images = images.view(-1, c, h, w)
                logits = model(images.to(device))
                logits = logits.view(bs, nc, -1).mean(dim=1)
            else:
                logits = model(images.to(device))

            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu())

    all_preds = torch.cat(all_preds)

    # Write result.csv
    print(f"Writing {len(test_dataset)} predictions to {args.output}...")
    with open(args.output, "w") as f:
        f.write("filename,label\n")
        for i, (path, _) in enumerate(test_dataset.samples):
            filename = Path(path).name
            f.write(f"{filename},{all_preds[i].item()}\n")

    print(f"Done! Result saved to {args.output}")
    print(f"First 5 predictions:")
    with open(args.output) as f:
        for _ in range(6):
            print(f"  {f.readline().strip()}")


if __name__ == "__main__":
    main()
