"""
Default configuration for CLIP + LoRA + ELR training on noisy-label fine-grained datasets.

All hyperparameters are documented with their effects.
Modify via command-line overrides or by creating dataset-specific configs.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List


@dataclass
class ModelConfig:
    """CLIP ViT-B/32 backbone with LoRA adapters."""

    # --- CLIP backbone ---
    clip_model: str = "ViT-B-32"
    pretrained: str = "openai"  # openai | laion2b_s34b_b79k
    freeze_backbone: bool = True  # Freeze CLIP backbone weights
    image_size: int = 224

    # --- LoRA ---
    lora_r: int = 16  # LoRA rank
    lora_alpha: int = 32  # LoRA scaling factor
    lora_dropout: float = 0.1
    lora_target_modules: Tuple[str, ...] = (
        "q_proj", "v_proj", "k_proj", "out_proj",
    )  # Which attention projections to apply LoRA to

    # --- Classification head ---
    num_classes: int = 400  # Override per dataset
    classifier_dropout: float = 0.3
    classifier_hidden_dim: int = 512  # Hidden dim before final FC
    classifier_type: str = "fc"  # fc | cosine — cosine is more noise-robust
    zero_shot_init: bool = False  # Init classifier with CLIP text encoder (class names)

    # --- Prompt Learning (CoOp-style) ---
    use_prompt_learner: bool = False  # Learnable text prompts for CLIP text encoder
    prompt_n_ctx: int = 8  # Number of learnable context tokens
    prompt_init: str = "a photo of a"  # Initial context words (set to "" for random)
    prompt_fusion_weight: float = 0.3  # Weight for text branch logits (0-1)


@dataclass
class DataConfig:
    """Data loading and augmentation."""

    # --- Paths ---
    data_root: str = "./data"
    train_dir: str = "train"
    test_dir: str = "test"

    # --- Input ---
    image_size: int = 224
    mean: Tuple[float, ...] = (0.48145466, 0.4578275, 0.40821073)  # CLIP stats
    std: Tuple[float, ...] = (0.26862954, 0.26130258, 0.27577711)

    # --- Augmentation ---
    use_randaug: bool = True
    randaug_n: int = 2
    randaug_m: int = 10
    use_mixup: bool = True
    mixup_alpha: float = 0.4
    use_cutmix: bool = True
    cutmix_alpha: float = 1.0
    label_smoothing: float = 0.1

    # --- Loading ---
    num_workers: int = 8
    pin_memory: bool = True
    prefetch_factor: int = 2
    val_split: float = 0.1  # Fraction of training data for validation


@dataclass
class LossConfig:
    """Loss function configuration."""

    # --- ELR (Early-Learning Regularization) ---
    elr_lambda: float = 3.0  # Regularization strength (main hyperparameter)
    elr_beta: float = 0.7  # EMA momentum for target estimation

    # --- GCE (Generalized Cross-Entropy) ---
    gce_q: float = 0.7  # q < 1.0 for noise robustness; q=1.0 is standard CE

    # --- Combined loss ---
    loss_type: str = "elr"  # Options: "ce", "elr", "gce", "elr+gce"
    elr_weight: float = 1.0  # Weight for ELR term when combined
    gce_weight: float = 0.5  # Weight for GCE term when combined

    # --- SupCon (Supervised Contrastive Learning) ---
    use_supcon: bool = False  # Enable supervised contrastive loss
    supcon_weight: float = 0.1  # Weight for contrastive loss
    supcon_temperature: float = 0.07  # Contrastive temperature
    supcon_clean_only: bool = True  # Only apply SupCon to GMM-clean samples

    # --- GMM Sample Selection (DivideMix-style) ---
    use_gmm_selection: bool = False  # Enable dynamic clean/noisy separation
    gmm_warmup_epochs: int = 10  # Epochs before starting GMM selection
    gmm_momentum: float = 0.9  # EMA momentum for loss history per sample
    gmm_p_threshold: float = 0.5  # Posterior threshold for clean classification

    # --- Label Co-Refinement ---
    use_label_refinement: bool = False  # Correct noisy labels with model predictions
    refine_alpha: float = 0.5  # Initial trust in model predictions
    refine_max_alpha: float = 0.9  # Maximum trust in model predictions


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    # --- Optimization ---
    optimizer: str = "adamw"  # adamw | sgd
    lr: float = 1e-4  # Base learning rate
    lr_backbone_multiplier: float = 0.1  # Backbone LR = lr * this
    lr_head_multiplier: float = 1.0  # Classifier head LR = lr * this
    weight_decay: float = 0.05
    betas: Tuple[float, float] = (0.9, 0.999)

    # --- Scheduler ---
    lr_scheduler: str = "cosine"  # cosine | step | plateau | onecycle
    warmup_epochs: int = 5
    warmup_lr: float = 1e-6
    min_lr: float = 1e-7

    # --- Training loop ---
    epochs: int = 100
    batch_size: int = 64
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # --- Mixed precision ---
    use_amp: bool = True  # Automatic Mixed Precision (float16)
    amp_dtype: str = "float16"

    # --- Knowledge distillation (teacher-free / self-distillation) ---
    use_self_distillation: bool = True
    distill_temperature: float = 4.0
    distill_weight: float = 0.5
    ema_teacher_alpha: float = 0.999  # EMA momentum for teacher model
    ema_teacher_warmup: int = 100  # Steps before starting EMA teacher updates

    # --- Early stopping ---
    early_stopping_patience: int = 20
    early_stopping_metric: str = "val_acc"


@dataclass
class EvalConfig:
    """Evaluation configuration."""

    batch_size: int = 128
    num_workers: int = 8
    use_tta: bool = True  # Test-time augmentation (multi-crop voting)
    tta_crops: int = 10  # Number of crops for TTA


@dataclass
class DefaultConfig:
    """Main configuration aggregating all sub-configs."""

    # --- Experiment ---
    exp_name: str = "clip_lora_elr"
    dataset: str = "webfg400"  # webfg400 | webinat5000
    seed: int = 42
    output_dir: str = "./outputs"
    log_dir: str = "./logs"
    resume_from: Optional[str] = None  # Path to checkpoint to resume

    # --- Sub-configs ---
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # --- Logging ---
    log_interval: int = 50  # Log every N steps
    eval_interval: int = 1  # Evaluate every N epochs
    save_interval: int = 10  # Save checkpoint every N epochs
    use_wandb: bool = True
    wandb_project: str = "clip-lora-elr-noise"


def get_config(dataset: str = "webfg400", **overrides) -> DefaultConfig:
    """
    Get configuration for a specific dataset with optional overrides.

    Args:
        dataset: "webfg400" or "webinat5000"
        **overrides: Any config field to override (key=value)

    Returns:
        DefaultConfig with dataset-specific settings applied
    """
    cfg = DefaultConfig()

    if dataset == "webfg400":
        cfg.dataset = "webfg400"
        cfg.model.num_classes = 400
        # WebFG-400: 400 classes, ~43k train images
        cfg.train.epochs = 100
        cfg.train.batch_size = 64
        cfg.loss.elr_lambda = 3.0
    elif dataset == "webinat5000":
        cfg.dataset = "webinat5000"
        cfg.model.num_classes = 5000
        # WebiNat-5000: 5000 classes, ~580k train images
        cfg.train.epochs = 80
        cfg.train.batch_size = 32  # More classes -> smaller batch
        cfg.loss.elr_lambda = 5.0  # Stronger regularization for more noise
        cfg.train.gradient_accumulation_steps = 2
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Apply overrides
    for key, value in overrides.items():
        parts = key.split(".")
        obj = cfg
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)

    return cfg
