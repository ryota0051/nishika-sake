import timm
import torch.nn as nn
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from torch.optim import AdamW
from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from src.layer.gem import GeM


class ArcFaceModelEfficientNet(nn.Module):
    def __init__(self, model_name, out_dim, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.n_features = self.model.classifier.in_features
        self.model.classifier = nn.Identity()  # custom head にするため
        self.fc = nn.Linear(self.n_features, out_dim)

    def forward(self, images, labels=None):
        x = self.model(images)
        x = self.fc(x)
        return x


class ArcFaceModelEfficientNetWithGem(nn.Module):
    def __init__(self, config, out_dim, pretrained=False):
        super().__init__()
        self.config = config
        self.model = timm.create_model(self.config.model_name, pretrained=pretrained)
        self.n_features = self.model.classifier.in_features
        self.model.global_pool = GeM()
        self.model.classifier = nn.Identity()  # custom head にするため
        self.fc = nn.Linear(self.n_features, out_dim)

    def forward(self, images, labels=None):
        bs = images.size(0)
        x = self.model(images).view(bs, -1)
        x = self.fc(x)
        return x


def get_optimizer(optimizer_config, params):
    if optimizer_config["optimizer_name"] == "AdamW":
        optimizer = AdamW(
            params,
            lr=optimizer_config["lr"],
            betas=optimizer_config["beta"],
            eps=optimizer_config["eps"],
        )
        return optimizer
    else:
        raise NotImplementedError


def get_scheduler(scheduler_config, optimizer, num_train_steps):
    if scheduler_config["scheduler_name"] == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(
                scheduler_config["num_warmup_steps_rate"] * num_train_steps
            ),
            num_training_steps=num_train_steps,
        )
        return scheduler

    elif scheduler_config["scheduler_name"] == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(
                scheduler_config["num_warmup_steps_rate"] * num_train_steps
            ),
            num_training_steps=num_train_steps,
            num_cycles=scheduler_config["num_cycles"],
        )
        return scheduler

    elif scheduler_config["scheduler_name"] == "cosine_restarts":
        """
        example:
            first_cycle_steps_ratio = 0.25,
            cycle_mult = 1.0,
            max_lr = 2e-5,
            min_lr = 1e-7,
            warmup_steps=100,
            gamma=0.8)
        """
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=int(
                num_train_steps * scheduler_config["first_cycle_steps_ratio"]
            ),
            cycle_mult=scheduler_config["cycle_mult"],
            max_lr=scheduler_config["max_lr"],
            min_lr=scheduler_config["min_lr"],
            warmup_steps=scheduler_config["warmup_steps"],
            gamma=scheduler_config["gamma"],
        )
        return scheduler

    else:
        raise NotImplementedError
