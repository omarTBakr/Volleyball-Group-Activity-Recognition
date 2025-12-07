"""
    in this model we will fine-tune a ResNet-50 backbone wrapped in a SequenceResNet class
    to handle video sequences. 
"""

# baseline3.py
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

from utils.utility import save_model, train_one_epoch, validate_one_epoch, test_one_epoch, load_model  # noqa: E402
from src.data.data_loader import VolleyballDataset, collate_fn  # noqa: E402

# =================================================================
# === 1. MODEL CLASS (Moved to Global Scope) ===
# =================================================================


class SequenceResNet(nn.Module):
    """
    Wraps a ResNet so it can accept either 4D inputs [B,C,H,W]
    or 5D inputs [B,T,C,H,W].
    """

    def __init__(self, backbone: nn.Module, frame_chunk_size=None):
        super().__init__()
        self.backbone = backbone
        self.frame_chunk_size = frame_chunk_size
    
    def forward(self, x):
         # Frame-wise Processing with Late Fusion
         
        # If input is 5D: [B, players, C, H, W] -> process frames individually
        if x.dim() == 5:
            B, T, C, H, W = x.shape

            # If no chunking requested or chunk >= T, process all frames at once
            if (self.frame_chunk_size is None) or (self.frame_chunk_size >= T):
                x = x.reshape(B * T, C, H, W)
                logits = self.backbone(x)  # (B*T, num_classes)
                logits = logits.reshape(B, T, -1)  # (B, T, num_classes)
                logits = logits.mean(dim=1)  # average over T -> (B, num_classes)
                return logits

            # Process frames in chunks to limit peak memory
            chunks = []
            for start in range(0, T, self.frame_chunk_size):
                end = min(T, start + self.frame_chunk_size)
                x_chunk = x[:, start:end]
                b, t_chunk, c, h, w = x_chunk.shape
                x_chunk = x_chunk.reshape(b * t_chunk, c, h, w)

                logits_chunk = self.backbone(x_chunk)
                logits_chunk = logits_chunk.reshape(b, t_chunk, -1)
                chunks.append(logits_chunk)

            # Concatenate along time dimension and average
            logits = torch.cat(chunks, dim=1)  # (B, T, num_classes)
            logits = logits.mean(dim=1)  # (B, num_classes)
            return logits

        # Otherwise assume standard 4D input [B, C, H, W]
        return self.backbone(x)


# =================================================================
# === 2. BUILDERS ===
# =================================================================


def build_transforms(cfg: DictConfig):
    train_augs = [hydra.utils.instantiate(aug) for aug in cfg.transforms.train]
    val_augs = [hydra.utils.instantiate(aug) for aug in cfg.transforms.validation]
    test_augs = [hydra.utils.instantiate(aug) for aug in cfg.transforms.test]
    return transforms.Compose(train_augs), transforms.Compose(val_augs), transforms.Compose(test_augs)


def build_model(cfg: DictConfig, num_classes: int):
    print("Building ResNet-50 model (Sequence wrapper)...")

    def _cfg_get(cfg_obj, key, default=None):
        if cfg_obj is None:
            return default
        try:
            return cfg_obj.get(key, default)
        except Exception:
            return getattr(cfg_obj, key, default)

    pretrained = _cfg_get(cfg, "pretrained", True)
    frame_chunk_size = _cfg_get(cfg, "frame_chunk_size", None)

    if pretrained:
        base_resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    else:
        base_resnet = models.resnet50(weights=None)

    # Replace the final FC layer
    num_ftrs = base_resnet.fc.in_features
    base_resnet.fc = nn.Linear(num_ftrs, num_classes)

    # Wrap in SequenceResNet
    model = SequenceResNet(base_resnet, frame_chunk_size=frame_chunk_size)
    return model


def build_scheduler(optimizer, cfg):
    if cfg.lr_scheduler.name == "CosineAnnealingLR":
        t_max = cfg.lr_scheduler.get("T_max", cfg.num_epochs)
        eta_min = cfg.lr_scheduler.get("eta_min", 0)
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
    return None


# =================================================================
# === 3. MAIN TRAINING ===
# =================================================================


@hydra.main(config_path="../configs", config_name="baseline3", version_base=None)
def train_test(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter()

    # --- Data ---
    train_transforms, val_transforms, test_transform = build_transforms(cfg)

    # crop=True ensures [Batch, 12, 3, H, W]
    train_dataset = VolleyballDataset(mode="train", transform=train_transforms, crop=True)
    val_dataset = VolleyballDataset(mode="validation", transform=val_transforms, crop=True)
    test_dataset = VolleyballDataset(mode="test", transform=test_transform, crop=True)

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

    # --- Class Mapping ---
    class_to_idx = {}
    for i, name in enumerate(cfg.class_names):
        class_to_idx[name] = i
        class_to_idx[name.replace("_", "-")] = i
        class_to_idx[name.replace("-", "_")] = i

    num_classes = len(cfg.class_names)

    # --- Model ---
    model = build_model(cfg, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = build_scheduler(optimizer, cfg)

    best_f1 = 0.0

    # --- Loop ---
    for epoch in range(cfg.num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{cfg.num_epochs} ---")

        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device, class_to_idx
        )

        val_loss, val_acc, val_f1 = validate_one_epoch(model, val_loader, criterion, device, class_to_idx)

        if scheduler:
            scheduler.step()
            writer.add_scalar("Learning_Rate", scheduler.get_last_lr()[0], epoch)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("F1_Score/train", train_f1, epoch)
        writer.add_scalar("F1_Score/val", val_f1, epoch)

        print(f"Train -> Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"Val   -> Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            save_model("baseline3", epoch, model, optimizer, val_loss, class_to_idx)
            print(f"New best model saved with F1 score: {best_f1:.4f}")

    writer.close()

    print("\n--- Testing Best Model ---")
    best_model = build_model(cfg, num_classes=num_classes)
    best_model, _, _, _, loaded_idx = load_model("baseline3", best_model)
    best_model.to(device)

    final_map = loaded_idx if loaded_idx else class_to_idx

    test_loss, test_acc, test_f1 = test_one_epoch(best_model, test_loader, criterion, device, final_map)

    print(f"Final Test Results -> Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, F1: {test_f1:.4f}")


if __name__ == "__main__":
    train_test()
