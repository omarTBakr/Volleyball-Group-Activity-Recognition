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
from utils.utility import save_model, train_one_epoch, validate_one_epoch, test_one_epoch, load_model
from src.data.data_loader import VolleyballDataset

# =================================================================
# === 1. EXTENDED MODEL ===
# =================================================================


class Model(nn.Module):
    def __init__(self, num_classes,):
 

    def forward(self, x):
 

# =================================================================
# === 2. SETUP FUNCTIONS ===
# =================================================================


def build_model(cfg: DictConfig, num_classes: int):
 

def build_transforms(cfg: DictConfig):
 

def build_scheduler(optimizer, cfg):
 


# =================================================================
# === 3. MAIN TRAINING LOOP ===
# =================================================================


# CHANGED: config_name="baseline1"
@hydra.main(config_path="../configs", config_name="baseline1", version_base=None)
def train_test(cfg: DictConfig) -> None:
 

if __name__ == "__main__":
    train_test()