"""
Main training loop for CLIP + LoRA + ELR fine-tuning.

Handles:
- Mixed precision training (AMP)
- Gradient accumulation for large effective batch sizes
- EMA Teacher for self-distillation (Mean Teacher style)
- GMM-based clean/noisy sample selection (DivideMix-style)
- Supervised contrastive loss (SupCon) for better features
- Label co-refinement for noisy samples
- Logging (TensorBoard, W&B)
- Checkpointing with best-model tracking
- Early stopping
"""

import os
import time
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW, SGD
from torch.cuda.amp import GradScaler

from tqdm import tqdm

from .evaluator import Evaluator
from .lr_scheduler import build_scheduler
from losses.ema_teacher import EMATeacher
from losses.noise_filter import GMMSampleSelector, CoRefinementModule
from losses.supcon import NoiseRobustSupConLoss


class Trainer:
    """
    Trainer for CLIP + LoRA + ELR on noisy-label fine-grained datasets.

    Features:
    - AMP training for memory efficiency
    - EMA teacher model for stable self-distillation
    - GMM-based dynamic clean/noisy sample separation
    - Supervised contrastive loss on clean samples
    - Label co-refinement for potential noisy samples
    - Gradient accumulation for large effective batch sizes
    - Early stopping on validation metric
    - Automatic checkpointing of best model
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        config,
        device: Optional[torch.device] = None,
        test_loader: DataLoader = None,
        num_train_full: int = None,
        full_train_dataset = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.config = config
        self.full_train_dataset = full_train_dataset

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = self.model.to(self.device)

        # Optimizer
        self.optimizer = self._build_optimizer()

        # Scheduler
        steps_per_epoch = len(train_loader) // config.train.gradient_accumulation_steps
        self.scheduler = build_scheduler(
            self.optimizer,
            scheduler_type=config.train.lr_scheduler,
            warmup_epochs=config.train.warmup_epochs,
            total_epochs=config.train.epochs,
            warmup_lr=config.train.warmup_lr,
            min_lr=config.train.min_lr,
            steps_per_epoch=steps_per_epoch,
        )
        self.scheduler_is_per_batch = (config.train.lr_scheduler == "onecycle")

        # AMP
        self.use_amp = config.train.use_amp
        self.scaler = GradScaler(enabled=self.use_amp)

        # Evaluator
        self.evaluator = Evaluator(self.model, self.device, use_amp=self.use_amp)

        # --- EMA Teacher (for self-distillation) ---
        self.use_ema_teacher = config.train.use_self_distillation
        if self.use_ema_teacher:
            self.ema_teacher = EMATeacher(
                self.model,
                alpha=config.train.ema_teacher_alpha,
                warmup_steps=config.train.ema_teacher_warmup,
                use_dynamic_alpha=True,
            )
            self.ema_teacher.to(self.device)
            print(f"[Trainer] EMA Teacher enabled (alpha={config.train.ema_teacher_alpha})")
        else:
            self.ema_teacher = None

        # --- GMM Sample Selection ---
        num_gmm = num_train_full if num_train_full is not None else len(train_loader.dataset)
        self.use_gmm = config.loss.use_gmm_selection
        if self.use_gmm:
            self.sample_selector = GMMSampleSelector(
                num_samples=num_gmm,
                momentum=config.loss.gmm_momentum,
                warmup_epochs=config.loss.gmm_warmup_epochs,
                p_threshold=config.loss.gmm_p_threshold,
                use_confidence=True,
            )
            print(f"[Trainer] GMM Sample Selection enabled "
                  f"(warmup={config.loss.gmm_warmup_epochs} epochs)")
        else:
            self.sample_selector = None

        # --- Label Co-Refinement ---
        self.use_label_refine = config.loss.use_label_refinement
        if self.use_label_refine:
            self.refiner = CoRefinementModule(
                num_classes=config.model.num_classes,
                refine_alpha=config.loss.refine_alpha,
                warmup_epochs=config.loss.gmm_warmup_epochs,
                max_alpha=config.loss.refine_max_alpha,
            )
            print(f"[Trainer] Label Co-Refinement enabled")
        else:
            self.refiner = None

        # --- SupCon Loss ---
        self.use_supcon = config.loss.use_supcon
        if self.use_supcon:
            self.supcon_loss = NoiseRobustSupConLoss(
                temperature=config.loss.supcon_temperature,
                num_classes=config.model.num_classes,
            )
            print(f"[Trainer] SupCon loss enabled "
                  f"(temp={config.loss.supcon_temperature}, "
                  f"weight={config.loss.supcon_weight})")
        else:
            self.supcon_loss = None

        # Output paths
        self.output_dir = Path(config.output_dir) / config.exp_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(config.log_dir) / config.exp_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.current_epoch = 0
        self.best_acc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        self.global_step = 0

        # Metrics history
        self.train_losses = []
        self.val_accs = []

        # Logger
        self.writer = None
        if config.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=config.wandb_project,
                    name=config.exp_name,
                    config=vars(config) if hasattr(config, "__dict__") else {},
                )
                self.use_wandb = True
            except ImportError:
                self.use_wandb = False
        else:
            self.use_wandb = False

        # TensorBoard
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb_writer = SummaryWriter(self.log_dir)
        except ImportError:
            self.tb_writer = None

    def _build_optimizer(self):
        """Build optimizer with per-parameter learning rates."""
        cfg = self.config.train

        # Group parameters by LR multiplier
        backbone_params = []
        lora_params = []
        head_params = []
        prompt_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "lora_A" in name or "lora_B" in name:
                lora_params.append(param)
            elif "classifier" in name:
                head_params.append(param)
            elif "prompt_learner" in name:
                prompt_params.append(param)
            elif "visual" in name:
                backbone_params.append(param)
            else:
                other_params.append(param)

        param_groups = []

        if backbone_params:
            param_groups.append({
                "params": backbone_params,
                "lr": cfg.lr * cfg.lr_backbone_multiplier,
                "name": "backbone",
            })
        if lora_params:
            param_groups.append({
                "params": lora_params,
                "lr": cfg.lr,
                "name": "lora",
            })
        if head_params:
            param_groups.append({
                "params": head_params,
                "lr": cfg.lr * cfg.lr_head_multiplier,
                "name": "head",
            })
        if prompt_params:
            param_groups.append({
                "params": prompt_params,
                "lr": cfg.lr * 0.5,  # Prompt learner uses half LR for stability
                "name": "prompt",
            })
        if other_params:
            param_groups.append({
                "params": other_params,
                "lr": cfg.lr,
                "name": "other",
            })

        if cfg.optimizer == "adamw":
            return AdamW(
                param_groups,
                lr=cfg.lr,
                betas=cfg.betas,
                weight_decay=cfg.weight_decay,
            )
        elif cfg.optimizer == "sgd":
            return SGD(
                param_groups,
                lr=cfg.lr,
                momentum=0.9,
                weight_decay=cfg.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

    def train(self, resume_from: Optional[str] = None) -> Dict[str, Any]:
        """Main training loop."""
        if resume_from:
            self._load_checkpoint(resume_from)

        cfg = self.config.train
        start_epoch = self.current_epoch

        print(f"\n{'='*60}")
        print(f"Training {self.config.exp_name}")
        print(f"Device: {self.device}")
        print(f"Epochs: {start_epoch + 1} -> {cfg.epochs}")
        print(f"Batch size: {cfg.batch_size} "
              f"(×{cfg.gradient_accumulation_steps} accumulation)")
        print(f"Optimizer: {cfg.optimizer}, LR: {cfg.lr}")
        print(f"AMP: {self.use_amp}")
        print(f"EMA Teacher: {self.use_ema_teacher}")
        print(f"GMM Selection: {self.use_gmm}")
        print(f"SupCon: {self.use_supcon}")
        print(f"Output: {self.output_dir}")
        print(f"{'='*60}\n")

        for epoch in range(start_epoch, cfg.epochs):
            self.current_epoch = epoch

            # Train one epoch
            train_loss = self._train_epoch(epoch)
            self.train_losses.append(train_loss)

            # Full GMM update at epoch level (after worst samples stabilized)
            if self.use_gmm and epoch >= self.config.loss.gmm_warmup_epochs:
                if hasattr(self.criterion, 'elr') and self.criterion.elr is not None:
                    all_targets = self.criterion.elr.get_all_targets()
                    # Use full_train_dataset (not subset) to get all labels
                    ds = self.full_train_dataset if self.full_train_dataset is not None else self.train_loader.dataset
                    all_labels = torch.tensor(
                        [ds.samples[i][1] for i in range(len(ds))]
                    )
                    self.sample_selector.full_update(all_targets, all_labels, epoch)
                    stats = self.sample_selector.get_probs()
                    print(f"  GMM: clean_ratio={stats['clean_ratio']:.2%}, "
                          f"means=[{stats['gmm_mean_clean']:.4f}, "
                          f"{stats['gmm_mean_noisy']:.4f}]")

            # Evaluate
            if (epoch + 1) % self.config.eval_interval == 0:
                val_results = self.evaluator.evaluate(
                    self.val_loader,
                    num_classes=self.config.model.num_classes,
                    desc=f"Val Epoch {epoch + 1}",
                )
                val_acc = val_results["top1_acc"]
                self.val_accs.append(val_acc)

                # Logging
                log_str = (
                    f"Epoch {epoch + 1:3d}/{cfg.epochs} | "
                    f"Loss: {train_loss:.4f} | "
                    f"Val Acc: {val_acc * 100:.2f}%"
                )
                if "top5_acc" in val_results:
                    log_str += f" | Top-5: {val_results['top5_acc'] * 100:.2f}%"
                if "mean_per_class_acc" in val_results:
                    log_str += (
                        f" | Mean-PC: {val_results['mean_per_class_acc'] * 100:.2f}%"
                    )
                print(log_str)

                # TensorBoard
                if self.tb_writer:
                    self.tb_writer.add_scalar("train/loss", train_loss, epoch)
                    self.tb_writer.add_scalar("val/top1_acc", val_acc, epoch)
                    self.tb_writer.add_scalar(
                        "lr", self.optimizer.param_groups[0]["lr"], epoch
                    )

                # W&B
                if self.use_wandb:
                    import wandb
                    wandb.log({
                        "train/loss": train_loss,
                        "val/top1_acc": val_acc,
                        "epoch": epoch,
                        "lr": self.optimizer.param_groups[0]["lr"],
                    })
                    if self.use_gmm and self.sample_selector:
                        stats = self.sample_selector.get_probs()
                        wandb.log({
                            "gmm/clean_ratio": stats["clean_ratio"],
                            "epoch": epoch,
                        })

                # Plateau scheduler (OneCycleLR already stepped per batch)
                if cfg.lr_scheduler == "plateau":
                    self.scheduler.step(val_acc)
                elif cfg.lr_scheduler != "onecycle":
                    self.scheduler.step()

                # Best model tracking
                if val_acc > self.best_acc:
                    self.best_acc = val_acc
                    self.best_epoch = epoch
                    self.patience_counter = 0
                    self._save_checkpoint("best_model.pth", epoch, val_acc)
                    print(f"  >> New best model! Acc: {val_acc * 100:.2f}%")
                else:
                    self.patience_counter += 1

                # Early stopping
                if self.patience_counter >= cfg.early_stopping_patience:
                    print(f"\nEarly stopping at epoch {epoch + 1} "
                          f"(best: {self.best_acc * 100:.2f}% at epoch "
                          f"{self.best_epoch + 1})")
                    break

            # Regular checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                self._save_checkpoint(
                    f"checkpoint_epoch_{epoch + 1}.pth", epoch, 0.0
                )

        # Training complete
        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"Best Val Acc: {self.best_acc * 100:.2f}% "
              f"(epoch {self.best_epoch + 1})")
        print(f"{'='*60}")

        if self.use_wandb:
            import wandb
            wandb.finish()

        if self.tb_writer:
            self.tb_writer.close()

        return {
            "best_acc": self.best_acc,
            "best_epoch": self.best_epoch,
            "train_losses": self.train_losses,
            "val_accs": self.val_accs,
        }

    def _train_epoch(self, epoch: int) -> float:
        """Train for one epoch. Returns average loss."""
        self.model.train()
        if self.ema_teacher:
            self.ema_teacher.train()
        cfg = self.config.train

        total_loss = 0.0
        num_batches = 0

        self.optimizer.zero_grad()

        pbar = tqdm(
            self.train_loader,
            desc=f"Train Epoch {epoch + 1}/{cfg.epochs}",
            leave=False,
        )

        for batch_idx, batch in enumerate(pbar):
            images, labels, indices = batch
            images = images.to(self.device)
            indices = indices.to(self.device)

            # Handle label format (hard or soft)
            is_soft = (
                labels.dtype in (torch.float16, torch.float32, torch.float64)
                and labels.dim() > 1
            )
            if is_soft:
                labels = labels.to(self.device)
            else:
                labels = labels.to(self.device)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                # Forward pass
                features, logits = self.model(images, return_features=True)

                # --- Per-sample loss for GMM ---
                if is_soft:
                    per_sample_loss = -(labels * F.log_softmax(logits, dim=1)
                                       ).sum(dim=1)
                else:
                    per_sample_loss = F.cross_entropy(
                        logits, labels, reduction='none'
                    )

                # --- GMM sample selection (per-batch update) ---
                clean_mask = None
                if self.use_gmm and self.sample_selector:
                    elr_batch = None
                    if hasattr(self.criterion, 'elr') and self.criterion.elr is not None:
                        elr_batch = self.criterion.elr.get_targets(indices)
                    self.sample_selector.update(
                        indices, per_sample_loss, elr_batch, labels, epoch
                    )
                    clean_mask = self.sample_selector.clean_mask[indices.cpu()].to(self.device)

                # --- Main loss (ELR / GCE / Combined) ---
                loss = self._compute_main_loss(indices, logits, labels, features)

                # --- Label refinement ---
                if self.use_label_refine and self.refiner and not is_soft:
                    if clean_mask is not None:
                        noisy_mask = ~clean_mask
                        if noisy_mask.any():
                            with torch.no_grad():
                                pred_probs = F.softmax(logits.detach(), dim=1)
                            refined_labels = self.refiner.refine_labels(
                                labels, pred_probs, noisy_mask, epoch
                            )
                            # Compute refinement loss with soft labels
                            log_probs = F.log_softmax(logits, dim=1)
                            refine_loss = -(
                                refined_labels * log_probs
                            ).sum(dim=1).mean()
                            loss = loss + 0.3 * refine_loss

                # --- SupCon loss (on clean samples only) ---
                if self.use_supcon and self.supcon_loss and clean_mask is not None:
                    if clean_mask.any() and self.ema_teacher:
                        teacher_feats = self.ema_teacher.generate_features(
                            images[clean_mask]
                        )
                        supcon_l = self.supcon_loss(
                            student_features=features[clean_mask],
                            teacher_features=teacher_feats,
                            labels=labels[clean_mask]
                            if not is_soft
                            else labels[clean_mask].argmax(dim=1),
                            clean_mask=torch.ones(
                                clean_mask.sum(), device=self.device
                            ).bool(),
                        )
                        loss = loss + self.config.loss.supcon_weight * supcon_l

                # --- Self-distillation from EMA teacher ---
                if self.use_ema_teacher and self.ema_teacher is not None:
                    if self.global_step >= cfg.ema_teacher_warmup:
                        with torch.no_grad():
                            teacher_logits = self.ema_teacher.generate_logits(images)

                        distill_loss = self._distillation_loss(
                            logits, teacher_logits, cfg.distill_temperature
                        )
                        loss = loss + cfg.distill_weight * distill_loss

                # Scale for gradient accumulation
                loss = loss / cfg.gradient_accumulation_steps

            # Backward with AMP
            self.scaler.scale(loss).backward()

            # Gradient step
            if (batch_idx + 1) % cfg.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), cfg.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                # OneCycleLR steps per optimizer step
                if self.scheduler_is_per_batch:
                    self.scheduler.step()

                # Update EMA teacher after each optimizer step
                if self.use_ema_teacher and self.ema_teacher:
                    self.ema_teacher.step(self.global_step)

            total_loss += loss.item() * cfg.gradient_accumulation_steps
            num_batches += 1
            self.global_step += 1

            # Progress bar
            avg_loss = total_loss / num_batches
            status = {"loss": f"{avg_loss:.4f}"}
            if self.use_gmm and clean_mask is not None:
                status["clean%"] = f"{clean_mask.float().mean():.2f}"
            pbar.set_postfix(status)

            # Interval logging
            if self.global_step % self.config.log_interval == 0:
                if self.tb_writer:
                    self.tb_writer.add_scalar(
                        "train/step_loss",
                        loss.item() * cfg.gradient_accumulation_steps,
                        self.global_step,
                    )

        return total_loss / max(num_batches, 1)

    def _compute_main_loss(
        self,
        indices: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the main supervised loss (ELR/GCE/CE)."""
        # Try ELR-style (with indices)
        try:
            loss = self.criterion(indices, logits, labels)
        except TypeError:
            try:
                loss = self.criterion(logits, labels)
            except TypeError:
                loss = F.cross_entropy(logits, labels)
        return loss

    def _distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        temperature: float = 4.0,
    ) -> torch.Tensor:
        """KL divergence distillation loss."""
        student_soft = F.log_softmax(student_logits / temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / temperature, dim=1)
        return F.kl_div(
            student_soft, teacher_soft, reduction="batchmean"
        ) * (temperature ** 2)

    def _save_checkpoint(self, filename: str, epoch: int, val_acc: float):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_acc": self.best_acc,
            "best_epoch": self.best_epoch,
            "val_acc": val_acc,
            "config": self.config,
            "train_losses": self.train_losses,
            "val_accs": self.val_accs,
        }

        # Save EMA teacher
        if self.ema_teacher:
            checkpoint["ema_teacher"] = self.ema_teacher.state_dict()

        # Save ELR targets
        if hasattr(self.criterion, "elr") and self.criterion.elr is not None:
            checkpoint["elr_targets"] = self.criterion.elr.get_all_targets()

        # Save GMM state
        if self.sample_selector:
            checkpoint["loss_history"] = self.sample_selector.loss_history
            checkpoint["loss_count"] = self.sample_selector.loss_count

        path = self.output_dir / filename
        torch.save(checkpoint, path)
        print(f"  Checkpoint saved: {path}")

    def _load_checkpoint(self, path: str):
        """Load model checkpoint and resume training."""
        print(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.current_epoch = checkpoint["epoch"] + 1
        self.best_acc = checkpoint.get("best_acc", 0.0)
        self.best_epoch = checkpoint.get("best_epoch", 0)
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_accs = checkpoint.get("val_accs", [])

        # Restore EMA teacher
        if "ema_teacher" in checkpoint and self.ema_teacher:
            self.ema_teacher.load_state_dict(checkpoint["ema_teacher"])

        # Restore ELR targets
        if "elr_targets" in checkpoint and hasattr(self.criterion, "elr"):
            if self.criterion.elr is not None:
                self.criterion.elr.target.copy_(checkpoint["elr_targets"])

        # Restore GMM state
        if "loss_history" in checkpoint and self.sample_selector:
            self.sample_selector.loss_history = checkpoint["loss_history"]
            self.sample_selector.loss_count = checkpoint["loss_count"]

        print(f"  Resumed from epoch {self.current_epoch}")

    def detect_noisy_labels(self, given_labels: torch.Tensor):
        """Run noise detection after training (ELR + GMM)."""
        results = {}

        if hasattr(self.criterion, "detect_noisy_labels"):
            noisy_mask = self.criterion.detect_noisy_labels(given_labels)
            num_noisy = noisy_mask.sum().item()
            print(f"\n=== Noise Detection (ELR argmax) ===")
            print(f"  Noisy: {num_noisy}/{len(given_labels)} "
                  f"({100 * num_noisy / len(given_labels):.2f}%)")
            results["elr_noisy_mask"] = noisy_mask

        if self.sample_selector:
            gmm_noisy = self.sample_selector.noisy_mask
            num_gmm_noisy = gmm_noisy.sum().item()
            print(f"\n=== Noise Detection (GMM) ===")
            print(f"  Noisy: {num_gmm_noisy}/{len(given_labels)} "
                  f"({100 * num_gmm_noisy / len(given_labels):.2f}%)")
            print(f"  GMM means: {self.sample_selector.gmm_means}")
            results["gmm_noisy_mask"] = gmm_noisy

        return results.get("elr_noisy_mask", None)
