
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from tqdm import tqdm

from configs.path_config import MODEL_SAVE_DIR, PLOTS_DIR


def save_model(model_name, model, optimizer, loss):

    check_point= {
		   "model_state_dict":  model.state_dict (),
	   "optimizer_state_dict":  optimizer.state_dict() ,
						"loss": loss,
    }

    torch.save(check_point , MODEL_SAVE_DIR / model_name)

def load_model(model_name , model, optimizer=None):

    check_point= torch.load(MODEL_SAVE_DIR / model_name)  # Load the checkpoint

    model.load_state_dict(check_point["model_state_dict"])  # Load the model state dict

    if optimizer is not None:
        optimizer.load_state_dict(check_point["optimizer_state_dict"])  # Load the optimizer state dict

    return model, optimizer

def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,

)->tuple[float,float , float , np.ndarray]:
    """
    Train the model for only a single epoch 
        
    Args:
        model: the model to be trained 
        dataloader: the dataloader to be used for training 
        criterion: the loss function to be used for training 
        optimizer: the optimizer to be used for training 
        device: the device to be used for training 
        
    Returns:
        loss_epoch: float -> the loss for the epoch 
        acc_epoch: float -> the accuracy for the epoch 
        f1_epoch: float -> the f1 score for the epoch 
        confusion_matrix: np.ndarray -> the confusion matrix for the epoch 
            
    """
    y_true = []
    y_pred = []
    running_loss = 0.0

    model.train()

    pbar = tqdm(dataloader, desc="Training", unit="batch", dynamic_ncols=True, leave=True)

    for batch_idx, (data, target) in enumerate(pbar):
        data , target = data.to(device) , target.to(device)

        optimizer.zero_grad()  # reset gradients

        output = model(data)   # forward pass

        loss = criterion(output, target)  # calculate loss
        running_loss += loss.item()

        loss.backward()  # backward pass

        optimizer.step()  # update weights

        # Append true and predicted labels for this batch
        y_true.extend(target.cpu().numpy())
        y_pred.extend(output.argmax(dim=1).cpu().numpy())

    loss_epoch = running_loss / len(dataloader)  # avg loss per batch
    acc_epoch = accuracy_score(y_true, y_pred)
    f1_epoch = f1_score(y_true, y_pred, average="macro")  # choose your preferred average
    conf_mat = confusion_matrix(y_true, y_pred)

    return loss_epoch, acc_epoch, f1_epoch, conf_mat




def validate_one_epoch(model, dataloader, criterion, device ):
    """
    Validate the model for a single epoch 
    
    Args:
        model: the model to be validated 
        dataloader: the dataloader to be used for validation 
        criterion: the loss function to be used for validation 
        device: the device to be used for validation 
    
    Returns:
        loss_epoch: float -> the loss for the epoch 
        acc_epoch: float -> the accuracy for the epoch 
        f1_epoch: float -> the f1 score for the epoch 
        confusion_matrix: np.ndarray -> the confusion matrix for the epoch 
        
    """
    y_true = []
    y_pred = []
    running_loss = 0.0

    model.eval()

    pbar = tqdm(dataloader, desc="Validation", unit="batch", dynamic_ncols=True, leave=True)

    for batch_idx, (data, target) in enumerate(pbar):
        data , target = data.to(device) , target.to(device)

        with torch.no_grad():
            output = model(data)   # forward pass

            loss = criterion(output, target)  # calculate loss
            running_loss += loss.item()

        # Append true and predicted labels for this batch
        y_true.extend(target.cpu().numpy())
        y_pred.extend(output.argmax(dim=1).cpu().numpy())

    loss_epoch = running_loss / len(dataloader)  # avg loss per batch
    acc_epoch = accuracy_score(y_true, y_pred)
    f1_epoch = f1_score(y_true, y_pred, average="macro")  # choose your preferred average
    conf_mat = confusion_matrix(y_true, y_pred)

    return loss_epoch, acc_epoch, f1_epoch, conf_mat

def test_one_epoch(model, dataloader, criterion, device ):
   """
   Test the model for a single epoch 
    
   Args:
        model: the model to be tested 
        dataloader: the dataloader to be used for testing 
        criterion: the loss function to be used for testing 
        device: the device to be used for testing 
    
   Returns:
        loss_epoch: float -> the loss for the epoch 
        acc_epoch: float -> the accuracy for the epoch 
        f1_epoch: float -> the f1 score for the epoch 
        confusion_matrix: np.ndarray -> the confusion matrix for the epoch 
        
   """
   y_true = []
   y_pred = []
   running_loss = 0.0

   model.eval()

   pbar = tqdm(dataloader, desc="Testing", unit="batch", dynamic_ncols=True, leave=True)

   for batch_idx, (data, target) in enumerate(pbar):
       data , target = data.to(device) , target.to(device)

       with torch.no_grad():
           output = model(data)   # forward pass

           loss = criterion(output, target)  # calculate loss
           running_loss += loss.item()

       # Append true and predicted labels for this batch
       y_true.extend(target.cpu().numpy())
       y_pred.extend(output.argmax(dim=1).cpu().numpy())

   loss_epoch = running_loss / len(dataloader)  # avg loss per batch
   acc_epoch = accuracy_score(y_true, y_pred)
   f1_epoch = f1_score(y_true, y_pred, average="macro")  # choose your preferred average
   conf_mat = confusion_matrix(y_true, y_pred)

   return loss_epoch, acc_epoch, f1_epoch, conf_mat


def plot_confusion_matrix(cm:np.ndarray , class_names:list , save_path:Path = PLOTS_DIR , title:str="Confusion Matrix"):



    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 10))
    plt.imshow(cm_normalized, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Loop over data dimensions and create text annotations.
    fmt = ".2f" # format specifier for values in the array
    thresh = cm_normalized.max() / 2.    # only set white text if values are light
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm_normalized[i, j], fmt),   # the displayed value
                     ha="center", va="center",
                     color="white" if cm_normalized[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(save_path / f"{title}.png", bbox_inches="tight", dpi=300)
    plt.close()


def plot_loss(train_loss, val_loss, save_path, title="Training & Validation Loss"):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_loss) + 1), train_loss, label="Training Loss", marker="o")
    plt.plot(range(1, len(val_loss) + 1), val_loss, label="Validation Loss", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path / f"{title}.png", bbox_inches="tight", dpi=300)
    plt.close()


def plot_accuracy(train_acc, val_acc, save_path, title="Training & Validation Accuracy"):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_acc) + 1), train_acc, label="Training Accuracy", marker="o")
    plt.plot(range(1, len(val_acc) + 1), val_acc, label="Validation Accuracy", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path / f"{title}.png", bbox_inches="tight", dpi=300)
    plt.close()


def plot_f1_score(train_f1, val_f1, save_path, title="Training & Validation F1 Score"):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_f1) + 1), train_f1, label="Training F1 Score", marker="o")
    plt.plot(range(1, len(val_f1) + 1), val_f1, label="Validation F1 Score", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path / f"{title}.png", bbox_inches="tight", dpi=300)
    plt.close()
