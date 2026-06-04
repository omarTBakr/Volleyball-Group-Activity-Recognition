"""
Model, transform, and scheduler builders driven by Hydra config.

These factories read from an OmegaConf ``DictConfig`` and return
fully-initialized PyTorch objects ready for training.
"""

from __future__ import annotations

from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
from torchvision import models, transforms

# ── Model ────────────────────────────────────────────────────────────────────


def build_model(cfg: DictConfig) -> nn.Module:
    """
    Build a pretrained torchvision model and replace the classification head.

    Reads ``cfg.model.name``, ``cfg.model.pretrained``, and
    ``cfg.model.num_classes`` (with fallback to ``len(cfg.class_names)``).

    Returns
    -------
    nn.Module
        The fine-tunable model with the correct output dimension.

    """
    model_name = cfg.model.name
    pretrained = cfg.model.get("pretrained", True)

    # Load the requested architecture
    if model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
    else:
        model_fn = getattr(models, model_name)
        model = model_fn(weights="DEFAULT" if pretrained else None)

    # Determine class count
    try:
        num_classes = cfg.model.num_classes
    except Exception:
        num_classes = len(cfg.class_names)

    # Replace the final fully-connected layer
    if hasattr(model, "fc"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    return model


# ── Transforms ───────────────────────────────────────────────────────────────


def build_transforms(cfg: DictConfig) -> dict[str, transforms.Compose]:
    """
    Build train / validation / test transform pipelines via Hydra instantiate.

    Returns
    -------
    dict[str, transforms.Compose]
        Keyed by ``"train"``, ``"validation"``, and ``"test"``.

    """
    train_list = instantiate(cfg.transforms.train)
    val_list = instantiate(cfg.transforms.validation)
    test_list = instantiate(cfg.transforms.test)

    return {
        "train": transforms.Compose(train_list),
        "validation": transforms.Compose(val_list),
        "test": transforms.Compose(test_list),
    }


# ── Scheduler ────────────────────────────────────────────────────────────────


def build_scheduler(optimizer: Optimizer, cfg: DictConfig) -> LRScheduler | None:
    """
    Build a learning-rate scheduler from the config.

    Currently supports ``CosineAnnealingLR``.

    Raises
    ------
    ValueError
        If the scheduler name is not recognized.

    """
    sched_cfg = cfg.get("lr_scheduler", None)
    
    if sched_cfg is None:
        return None

    if sched_cfg.name == "CosineAnnealingLR":
        return CosineAnnealingLR(
            optimizer,
            T_max=sched_cfg.T_max,
            eta_min=sched_cfg.eta_min,
        )
    
    return None
