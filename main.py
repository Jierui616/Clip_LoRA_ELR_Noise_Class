#!/usr/bin/env python3
"""
Main entry point for CLIP + LoRA + ELR training on noisy-label fine-grained datasets.

Competition: 全球校园人工智能算法精英大赛 - 赛题四
Task: 面向噪声标签数据的细粒度图像识别鲁棒微调 (网络监督细粒度图像识别)

Usage:
    # Train on WebFG-400 (解压 train.zip + test.zip 到 data/ 下)
    python main.py --dataset webfg400 --data_root ./data \
        --output_dir ./outputs --exp_name webfg400_elr

    # Train on WebiNat-5000
    python main.py --dataset webinat5000 --data_root ./data \
        --output_dir ./outputs --exp_name webinat5000_elr \
        --train.batch_size 32 --train.epochs 80

    # Resume training
    python main.py --resume ./outputs/webfg400_elr/checkpoint_epoch_50.pth

    # Evaluate only
    python main.py --eval_only --resume ./outputs/webfg400_elr/best_model.pth
"""

import argparse
import sys

import torch

from config import get_config
from data import get_dataloaders, get_train_transforms, get_test_transforms
from data.transforms import MixUpCutMixCollator
from models import CLIPLoRAModel
from losses import CombinedLoss
from trainers import Trainer
from trainers.evaluator import Evaluator
from utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="CLIP + LoRA + ELR for Noisy-Label Fine-Grained Recognition"
    )

    # Main settings
    parser.add_argument("--dataset", type=str, default="webfg400",
                        choices=["webfg400", "webinat5000"],
                        help="Dataset to train on")
    parser.add_argument("--data_root", type=str, default="./data",
                        help="Root directory containing train/ and test/ folders")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Directory for model checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs",
                        help="Directory for TensorBoard logs")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Experiment name. Auto-generated if not provided.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # Mode
    parser.add_argument("--eval_only", action="store_true",
                        help="Only run evaluation (no training)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--detect_noise", action="store_true",
                        help="Run noise detection analysis after training")

    # Config overrides (dot-separated: model.lora_r=8 train.lr=1e-4 ...)
    parser.add_argument("--overrides", nargs="*", default=[],
                        help="Config overrides in key=value format (e.g., train.lr=1e-4)")

    return parser.parse_args()


def build_overrides_dict(overrides_list):
    """Parse dot-separated overrides into a nested dict."""
    overrides = {}
    for item in overrides_list:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        # Try to convert value to appropriate type
        try:
            value = eval(value)
        except (NameError, SyntaxError):
            pass
        overrides[key] = value
    return overrides


def main():
    args = parse_args()

    # Build config
    overrides = build_overrides_dict(args.overrides)
    overrides["seed"] = args.seed
    overrides["output_dir"] = args.output_dir
    overrides["log_dir"] = args.log_dir

    if args.exp_name:
        overrides["exp_name"] = args.exp_name

    cfg = get_config(args.dataset, **overrides)

    # Auto-generate experiment name
    if not args.exp_name:
        cfg.exp_name = (
            f"{args.dataset}"
            f"_r{cfg.model.lora_r}"
            f"_lambda{cfg.loss.elr_lambda}"
            f"_beta{cfg.loss.elr_beta}"
            f"_bs{cfg.train.batch_size}"
            f"_lr{cfg.train.lr}"
            f"_seed{args.seed}"
        )

    # Set random seed
    set_seed(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data ---
    print("\n[1/5] Loading data...")
    train_transform = get_train_transforms(
        image_size=cfg.data.image_size,
        use_randaug=cfg.data.use_randaug,
        randaug_n=cfg.data.randaug_n,
        randaug_m=cfg.data.randaug_m,
    )
    test_transform = get_test_transforms(
        image_size=cfg.data.image_size,
        crop_size=cfg.data.image_size,
        is_tta=cfg.eval.use_tta if args.eval_only else False,
        tta_crops=cfg.eval.tta_crops,
    )

    train_root = f"{args.data_root}/{cfg.data.train_dir}"
    test_root = f"{args.data_root}/{cfg.data.test_dir}"

    train_loader, test_loader, train_dataset, val_loader = get_dataloaders(
        train_root=train_root,
        test_root=test_root,
        train_transform=train_transform,
        test_transform=test_transform,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        prefetch_factor=cfg.data.prefetch_factor,
        val_split=getattr(cfg.data, 'val_split', 0.1),  # 10% for validation
    )

    # Replace default collate with MixUp/CutMix collator for stronger augmentation
    # keep_hard_labels=True ensures ELR compatibility (image mixing only, no label mixing)
    mixup_collator = MixUpCutMixCollator(
        num_classes=cfg.model.num_classes,
        mixup_alpha=cfg.data.mixup_alpha,
        cutmix_alpha=cfg.data.cutmix_alpha,
        label_smoothing=cfg.data.label_smoothing,
        use_mixup=cfg.data.use_mixup,
        use_cutmix=cfg.data.use_cutmix,
        keep_hard_labels=True,
        mix_prob=0.5,
    )
    train_loader.collate_fn = mixup_collator
    print("  MixUp/CutMix collator enabled (hard-label mode, prob=0.5)")
    num_train = len(train_dataset)
    print(f"  Train samples: {num_train}")
    print(f"  Classes: {cfg.model.num_classes}")
    print(f"  Class distribution (first 10): {dict(list(train_dataset.get_class_distribution().items())[:10])}")

    # --- Model ---
    print("\n[2/5] Building model...")
    # Extract class folder names for zero-shot initialization
    class_names = train_dataset.classes if cfg.model.zero_shot_init else None

    model = CLIPLoRAModel(
        num_classes=cfg.model.num_classes,
        image_size=cfg.model.image_size,
        clip_model=cfg.model.clip_model,
        pretrained=cfg.model.pretrained,
        lora_r=cfg.model.lora_r,
        lora_alpha=cfg.model.lora_alpha,
        lora_dropout=cfg.model.lora_dropout,
        classifier_hidden_dim=cfg.model.classifier_hidden_dim,
        classifier_dropout=cfg.model.classifier_dropout,
        classifier_type=getattr(cfg.model, 'classifier_type', 'fc'),
        freeze_backbone=cfg.model.freeze_backbone,
        use_prompt_learner=cfg.model.use_prompt_learner,
        prompt_n_ctx=cfg.model.prompt_n_ctx,
        prompt_fusion_weight=cfg.model.prompt_fusion_weight,
        zero_shot_init=cfg.model.zero_shot_init,
        class_names=class_names,
    )

    # --- Loss ---
    print("\n[3/5] Setting up loss function...")
    criterion = CombinedLoss(
        num_examples=num_train,
        num_classes=cfg.model.num_classes,
        loss_type=cfg.loss.loss_type,
        elr_lambda=cfg.loss.elr_lambda,
        elr_beta=cfg.loss.elr_beta,
        elr_weight=cfg.loss.elr_weight,
        gce_q=cfg.loss.gce_q,
        gce_weight=cfg.loss.gce_weight,
        use_distill=cfg.train.use_self_distillation,
        distill_temperature=cfg.train.distill_temperature,
        distill_weight=cfg.train.distill_weight,
        label_smoothing=cfg.data.label_smoothing,
    )
    print(f"  Loss type: {cfg.loss.loss_type}")
    if "elr" in cfg.loss.loss_type:
        print(f"  ELR: λ={cfg.loss.elr_lambda}, β={cfg.loss.elr_beta}")
    if "gce" in cfg.loss.loss_type:
        print(f"  GCE: q={cfg.loss.gce_q}")
    if cfg.loss.use_supcon:
        print(f"  SupCon: weight={cfg.loss.supcon_weight}, "
              f"temp={cfg.loss.supcon_temperature}")
    if cfg.loss.use_gmm_selection:
        print(f"  GMM Selection: warmup={cfg.loss.gmm_warmup_epochs} epochs")
    if cfg.loss.use_label_refinement:
        print(f"  Label Refinement: alpha={cfg.loss.refine_alpha}")
    if cfg.model.use_prompt_learner:
        print(f"  Prompt Learner: n_ctx={cfg.model.prompt_n_ctx}")
    if getattr(cfg.model, 'classifier_type', 'fc') == 'cosine':
        print(f"  Classifier: cosine (noise-robust)")

    # --- Evaluation only ---
    if args.eval_only:
        print("\n[Eval] Running evaluation only...")
        if args.resume:
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"  Loaded checkpoint from {args.resume}")

        evaluator = Evaluator(model, device, use_amp=cfg.train.use_amp)

        if cfg.eval.use_tta:
            results = evaluator.evaluate_tta(
                test_loader,
                num_classes=cfg.model.num_classes,
                tta_crops=cfg.eval.tta_crops,
            )
        else:
            results = evaluator.evaluate(
                test_loader,
                num_classes=cfg.model.num_classes,
            )

        print(f"\nEvaluation Results:")
        print(f"  Top-1 Accuracy: {results['top1_acc'] * 100:.2f}%")
        if 'top5_acc' in results:
            print(f"  Top-5 Accuracy: {results['top5_acc'] * 100:.2f}%")
        if 'mean_per_class_acc' in results:
            print(f"  Mean Per-Class Acc: {results['mean_per_class_acc'] * 100:.2f}%")
        return

    # --- Trainer ---
    print("\n[4/5] Setting up trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,  # Validation set from train split
        test_loader=test_loader,  # Competition test set (no labels)
        criterion=criterion,
        config=cfg,
        device=device,
        num_train_full=num_train,  # Full dataset size for GMM/ELR indices
        full_train_dataset=train_dataset,  # Original full dataset for GMM
    )

    # --- Train ---
    print("\n[5/5] Starting training...")
    results = trainer.train(resume_from=args.resume)

    # --- Post-training analysis ---
    if args.detect_noise:
        print("\n[Analysis] Running noise detection...")
        # Collect all training labels
        all_labels = torch.tensor(
            [train_dataset.samples[i][1] for i in range(len(train_dataset))]
        )
        noisy_mask = trainer.detect_noisy_labels(all_labels)
        if noisy_mask is not None:
            num_noisy = noisy_mask.sum().item()
            print(f"  Estimated noisy samples: {num_noisy}/{len(all_labels)} "
                  f"({100 * num_noisy / len(all_labels):.2f}%)")

    print(f"\nDone! Best accuracy: {results['best_acc'] * 100:.2f}%")
    print(f"Output saved to: {trainer.output_dir}")


if __name__ == "__main__":
    main()
