from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models, transforms


def build_model(cfg: DictConfig):
    """
    Build a model from the config and return it.
    Fine-tunes a ResNet by replacing the final classification layer.
    """
    model_name = cfg.model.name
    pretrained = cfg.model.get("pretrained", True)

    # 1. Load the requested model
    if model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
    else:
        # Generic fallback for other models
        model_fn = getattr(models, model_name)
        model = model_fn(weights="DEFAULT" if pretrained else None)

    # 2. Determine the number of classes from your config
    # Fallback to len(cfg.class_names) in case ${len(class_names)} interpolation fails
    try:
        num_classes = cfg.model.num_classes
    except Exception:
        num_classes = len(cfg.class_names)

    # 3. Replace the final fully connected layer
    if hasattr(model, "fc"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)

    return model

def build_transforms(cfg: DictConfig):
    """
    Build train, validation, and test transforms using Hydra's instantiate.
    Returns a dictionary of composed transform pipelines.
    """
    # 1. hydra.utils.instantiate converts the YAML list into a Python list
    #    of instantiated PyTorch transform objects (e.g., [Resize(...), ToTensor(), ...])
    train_list = instantiate(cfg.transforms.train)
    val_list = instantiate(cfg.transforms.validation)
    test_list = instantiate(cfg.transforms.test)

    # 2. Wrap the lists in transforms.Compose
    transform_pipelines = {
        "train": transforms.Compose(train_list),
        "validation": transforms.Compose(val_list),
        "test": transforms.Compose(test_list),
    }

    return transform_pipelines

def build_scheduler(optimizer, cfg: DictConfig):
    """
    Build scheduler from the config and return it.
    """
    # Access the scheduler configuration block
    sched_cfg = cfg.lr_scheduler

    if sched_cfg.name == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=sched_cfg.T_max,   # Interpolated automatically by Hydra to 100
            eta_min=sched_cfg.eta_min,
        )
        return scheduler
    raise ValueError(f"Unsupported scheduler name: {sched_cfg.name}")

