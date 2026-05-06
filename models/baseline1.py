# baseline1.py

from argparse import ArgumentParser

import hydra
from omegaconf import DictConfig
from torch import nn
from torchvision import models

from configs.labels import (
    NUM_PERSON_ACTIONS,
)

# =================================================================
# === 1. EXTENDED MODEL ===
# =================================================================


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_classes = NUM_PERSON_ACTIONS
        self.backbone = models.resnet50(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, self.num_classes)


    def forward(self, x):
        return self.backbone(x)




# =================================================================
# === 3. MAIN TRAINING LOOP ===
# =================================================================


# CHANGED: config_name="baseline1"
@hydra.main(config_path="../configs", config_name="baseline1", version_base=None)
def train(cfg: DictConfig) -> None:

    pass

def test(cfg: DictConfig) -> None:
    pass

def val(cfg: DictConfig) -> None:
    pass

def validate_terminal_args(args):
    # if not os.path.isfile(args.cfg):
    #     raise FileNotFoundError(f"Configuration file not found: {args.cfg}")
    if not (args.train or args.test or args.val):
        raise ValueError("Please specify at least one of --train, --test, or --val")

if __name__ == "__main__":
    parser = ArgumentParser()
    # parser.add_argument("--cfg", "--config", type=str, action="store_true", required=True, help="path to the config file")
    parser.add_argument("--train", action="store_true", help="train the model")
    parser.add_argument("--test", action="store_true", help="test the model")
    parser.add_argument("--val", action="store_true", help="validate the model")
    args = parser.parse_args()

    validate_terminal_args(args)

    if args.train:
        train(args.cfg)
    if args.test:
        test(args.cfg)
    if args.val:
        val(args.cfg)
