from sklearn.metrics import f1_score
import torch
from tqdm import tqdm
import os
import sys
from pathlib import Path

from configs.path_config import MODEL_SAVE_DIR


def save_model(model_name, epoch, model, optimizer, loss, class2index):
 


def load_model(model_name, model, optimizer=None):
 
    pass 


def get_images_from_batch(batch, device):
 

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
    pass 


def validate_one_epoch(model, dataloader, criterion, device, class_to_idx, use_amp: bool = False):
    pass 

def test_one_epoch(model, dataloader, criterion, device, class_to_idx, use_amp: bool = False):
   pass 