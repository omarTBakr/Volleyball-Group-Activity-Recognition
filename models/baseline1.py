# baseline1.py
import hydra
from omegaconf import DictConfig
from pathlib import Path
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import models, transforms

# --- Add project root to sys.path ---
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from utils.utility import save_model, train_one_epoch, validate_one_epoch, test_one_epoch, load_model
from src.data.data_loader import VolleyballDataset, collate_fn

# =================================================================
# === 1. EXTENDED MODEL ===
# =================================================================


class Model(nn.Module):
    def __init__(self, num_classes, num_players=12):
        super(Model, self).__init__()

   
        # 2. Backbone: Standard ResNet50
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # 3. Classifier: Change final layer to 8 classes
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        # x shape comes in as: [Batch, 3, Height, Width]

        # Pass through the rest of ResNet
        return self.backbone(x)


# =================================================================
# === 2. SETUP FUNCTIONS ===
# =================================================================


def build_model(cfg: DictConfig, num_classes: int):
    print("Building Extended ResNet-50...")
    # Instantiate our Custom Class
    model = Model(num_classes=num_classes, num_players=12)
    return model


def build_transforms(cfg: DictConfig):
    train_augs = [hydra.utils.instantiate(aug) for aug in cfg.transforms.train]
    val_augs = [hydra.utils.instantiate(aug) for aug in cfg.transforms.validation]
    test_augs = [hydra.utils.instantiate(aug) for aug in cfg.transforms.test]
    return transforms.Compose(train_augs), transforms.Compose(val_augs), transforms.Compose(test_augs)


def build_scheduler(optimizer, cfg):
    """
    Constructs the Learning Rate Scheduler based on the config.
    """
    # Check if lr_scheduler is defined in config and has a name
    if hasattr(cfg, "lr_scheduler") and cfg.lr_scheduler and cfg.lr_scheduler.get("name") == "CosineAnnealingLR":
        t_max = cfg.lr_scheduler.get("T_max", cfg.num_epochs)
        eta_min = cfg.lr_scheduler.get("eta_min", 0)
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
    return None


# =================================================================
# === 3. MAIN TRAINING LOOP ===
# =================================================================


# CHANGED: config_name="baseline1"
@hydra.main(config_path="../configs", config_name="baseline1", version_base=None)
def train_test(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter()

    train_transforms, val_transforms, test_transform = build_transforms(cfg)

    # Note: crop=True is required for this model
    train_dataset = VolleyballDataset(mode="train", transform=train_transforms, middle=  True,crop=False)
    val_dataset = VolleyballDataset(mode="validation", transform=val_transforms, middle=True , crop=False)
    test_dataset = VolleyballDataset(mode="test", transform=test_transform, middle= True,crop=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=collate_fn,
    )

    # Robust Label Mapping
    class_to_idx = {}
    for i, name in enumerate(cfg.class_names):
        class_to_idx[name] = i
        class_to_idx[name.replace("_", "-")] = i
        class_to_idx[name.replace("-", "_")] = i

    num_classes = len(cfg.class_names)

    # Build Model
    model = build_model(cfg, num_classes=num_classes).to(device)

    # --- VERIFICATION CHECK ---
    print(f"Model Type: {type(model)}")
    if not isinstance(model, Model):
        raise RuntimeError("Model is NOT ExtendedResNet! Check build_model function.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    
    # --- BUILD SCHEDULER ---
    scheduler = build_scheduler(optimizer, cfg)

    best_f1 = 0.0

    for epoch in range(cfg.num_epochs):
        print(f"\n--- Epoch {epoch+1}/{cfg.num_epochs} ---")

        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device, class_to_idx
        )

        val_loss, val_acc, val_f1 = validate_one_epoch(model, val_loader, criterion, device, class_to_idx)

        # --- STEP SCHEDULER ---
        if scheduler:
            scheduler.step()
            # Log the current LR to TensorBoard
            writer.add_scalar("Learning_Rate", scheduler.get_last_lr()[0], epoch)

        # --- UPDATED LOGGING ---
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)  # Added this
        writer.add_scalar("F1_Score/train", train_f1, epoch)
        writer.add_scalar("F1_Score/val", val_f1, epoch)

        print(f"Train -> Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"Val   -> Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            # CHANGED: "baseline1"
            save_model("baseline1", epoch, model, optimizer, val_loss, class_to_idx)
            print(f"New best model saved with F1 score: {best_f1:.4f}")

    writer.close()

    print("\n--- Testing Best Model ---")
    best_model = build_model(cfg, num_classes=num_classes)
    # CHANGED: "baseline1"
    best_model, _, _, _, loaded_idx = load_model("baseline1", best_model)
    best_model.to(device)

    final_map = loaded_idx if loaded_idx else class_to_idx
    test_loss, test_acc, test_f1 = test_one_epoch(best_model, test_loader, criterion, device, final_map)
    print(f"Final Test Results -> Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, F1: {test_f1:.4f}")


if __name__ == "__main__":
    train_test()