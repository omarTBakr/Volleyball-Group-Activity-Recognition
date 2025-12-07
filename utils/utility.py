from sklearn.metrics import f1_score
import torch
from tqdm import tqdm
import os
import sys
from pathlib import Path

# --- Add project root to sys.path ---
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from configs.path_config import MODEL_SAVE_DIR


def save_model(model_name, epoch, model, optimizer, loss, class2index):
    """
    Saves the model state, optimizer state, and other metadata.
    """
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)

    save_path = os.path.join(MODEL_SAVE_DIR, f"{model_name}.pth")

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "class_to_idx": class2index,
        },
        save_path,
    )

    print(f"Model saved to {save_path}")


def load_model(model_name, model, optimizer=None):
    """
    Loads the model weights. Optimizes backward compatibility.
    """
    load_path = os.path.join(MODEL_SAVE_DIR, f"{model_name}.pth")

    if not os.path.exists(load_path):
        raise FileNotFoundError(f"No checkpoint found at {load_path}")

    checkpoint = torch.load(load_path)

    # Load model weights
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load optimizer state if provided
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss", 0.0)
    class_to_idx = checkpoint.get("class_to_idx", None)

    print(f"Loaded model '{model_name}' from epoch {epoch} with loss {loss:.4f}")

    return model, optimizer, epoch, loss, class_to_idx


def get_images_from_batch(batch, device):
    """
    Helper function to handle backward compatibility for image keys.
    Checks for 'image' (old baseline) or 'images' (new baseline/crop).
    """
    if "image" in batch:
        return batch["image"].to(device)
    elif "images" in batch:
        return batch["images"].to(device)
    else:
        raise KeyError(f"Batch does not contain 'image' or 'images'. Found keys: {batch.keys()}")


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    class_to_idx,
    use_amp: bool = False,
    accumulation_steps: int = 1,
):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    # Map class names to indices for labels
    # Assumes labels in batch are strings. If they are already ints, this step might change.

    pbar = tqdm(dataloader, desc="Training", leave=False)

    all_preds = []
    all_labels = []

    # Use a GradScaler when using AMP on CUDA devices
    scaler = torch.cuda.amp.GradScaler() if (use_amp and device.type == "cuda") else None

    for step, batch in enumerate(pbar):
        # --- BACKWARD COMPATIBILITY FIX ---
        images = get_images_from_batch(batch, device)
        # ----------------------------------

        # Handle labels: convert string labels to tensor indices
        labels_raw = batch["group_activity"]
        labels = torch.tensor([class_to_idx[l] for l in labels_raw], device=device)

        # Forward / backward with optional mixed precision and gradient accumulation
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            loss = loss / accumulation_steps
            scaler.scale(loss).backward()
            if (step + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss = loss / accumulation_steps
            loss.backward()
            if (step + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        # For reporting multiply by accumulation_steps back when using accumulation
        running_loss += (loss.item() * accumulation_steps) * images.size(0)

        # Metrics
        _, preds = torch.max(outputs, 1)
        correct_predictions += torch.sum(preds == labels.data)
        total_samples += images.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({"loss": loss.item()})

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions.double() / total_samples

    # F1 Score calculation (macro average usually preferred for multiclass)
    epoch_f1 = f1_score(all_labels, all_preds, average="macro")

    return epoch_loss, epoch_acc.item(), epoch_f1


def validate_one_epoch(model, dataloader, criterion, device, class_to_idx, use_amp: bool = False):
    """
    Validates the model.
    """
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            # --- BACKWARD COMPATIBILITY FIX ---
            images = get_images_from_batch(batch, device)
            # ----------------------------------

            labels_raw = batch["group_activity"]
            labels = torch.tensor([class_to_idx[l] for l in labels_raw], device=device)

            if use_amp and device.type == "cuda":
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)
            total_samples += images.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions.double() / total_samples

    epoch_f1 = f1_score(all_labels, all_preds, average="macro")

    return epoch_loss, epoch_acc.item(), epoch_f1


def test_one_epoch(model, dataloader, criterion, device, class_to_idx, use_amp: bool = False):
    """
    Tests the model.
    """
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing", leave=False):
            # --- BACKWARD COMPATIBILITY FIX ---
            images = get_images_from_batch(batch, device)
            # ----------------------------------

            labels_raw = batch["group_activity"]
            labels = torch.tensor([class_to_idx[l] for l in labels_raw], device=device)

            if use_amp and device.type == "cuda":
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)
            total_samples += images.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions.double() / total_samples

    epoch_f1 = f1_score(all_labels, all_preds, average="macro")

    return epoch_loss, epoch_acc.item(), epoch_f1
