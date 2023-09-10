#!/usr/bin/env python
# coding: utf-8

# ## 概要
#
# - efficientnetv1b0(gloval_average_pooling => gemに変更) + arcface を用いて meigara の予測 タスクを実行
# - 画像サイズを1024 x 1024
# - 学習済みモデルを用いて test, cite の画像の embedding を取得し、それらの距離から類似画像を選択
# - 回転とfocal loss使用

# ## Library

# In[1]:


import datetime
import gc
import itertools
import json
import logging
import math
import os
import shutil
from pathlib import Path

import albumentations as A
import cv2
import faiss
import japanize_matplotlib
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations import ImageOnlyTransform
from albumentations.pytorch import ToTensorV2
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from PIL import Image
from pytorch_metric_learning import losses
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    MultiStepLR,
    ReduceLROnPlateau,
)
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from src.const import CITE_IMAGES_PATH, DATA_ROOT, OUTPUT_ROOT, QUERY_IMAGES_PATH
from src.layer import GeM
from src.layer.arcface import ArcMarginProduct
from src.loss.focal_loss import FocalLoss
from src.utils import seed_torch


class Config:
    competition = "sake"
    name = "effnetb0_1024_apply_rotate"

    debug = False
    use_amp = True

    training = True
    evaluation = True
    embedding = True
    embedding_dim = 512

    base_model_root = (
        OUTPUT_ROOT
        / "train_arcface_with_label_smoothing"
        / "outputs"
        / "effnetb0_apply_rotate_and_focal_loss"
        / "models"
    )

    seed = 8823
    n_fold = 5
    trn_fold = [0, 1, 2, 3, 4]
    scale = 30

    target_columns = ["meigara_label"]
    size = 1024

    model_name = "tf_efficientnet_b0.ns_jft_in1k"
    max_epochs = 5
    train_batch_size = 6
    valid_batch_size = 15
    num_workers = 4
    gradient_accumulation_steps = 1
    clip_grad_norm = 1000
    label_smoothing = 0.1

    optimizer = dict(
        optimizer_name="AdamW",
        lr=3.75e-5,
        # lr=1e-4,
        weight_decay=1e-2,
        eps=1e-6,
        beta=(0.9, 0.999),
        encoder_lr=1e-4,
        decoder_lr=1e-4,
    )

    scheduler = dict(
        scheduler_name="cosine",
        num_warmup_steps_rate=0,
        num_cycles=0.5,
    )
    batch_scheduler = True


if Config.debug:
    Config.max_epochs = 2
    Config.n_fold = 2
    Config.trn_fold = [0, 1]
    Config.name = Config.name + "_debug"


# constants
HOME = OUTPUT_ROOT / "train_arcface_with_label_smoothing"
EXP_NAME = Config.name
INPUTS = DATA_ROOT  # input data
OUTPUTS = HOME / "outputs"
INTERMIDIATES = HOME / "intermidiates"  # intermidiate outputs
SUBMISSIONS = HOME / "submissions"
OUTPUTS_EXP = OUTPUTS / EXP_NAME
EXP_MODELS = OUTPUTS_EXP / "models"
EXP_REPORTS = OUTPUTS_EXP / "reports"
EXP_PREDS = OUTPUTS_EXP / "predictions"

CITE_IMAGES = CITE_IMAGES_PATH
QUERY_IMAGES = QUERY_IMAGES_PATH


def setup(Config):
    for d in [
        HOME,
        INPUTS,
        SUBMISSIONS,
        EXP_MODELS,
        EXP_REPORTS,
        EXP_PREDS,
        INTERMIDIATES,
    ]:
        d.mkdir(parents=True, exist_ok=True)


def check_file_exists(folder_path, file_name):
    folder = Path(folder_path)
    for file_path in folder.glob("**/*"):
        if file_path.is_file() and file_path.stem == file_name:
            return True
    return False


class Logger:
    def __init__(self, path):
        self.general_logger = logging.getLogger(path)
        stream_handler = logging.StreamHandler()
        file_general_handler = logging.FileHandler(os.path.join(path, "Experiment.log"))
        if len(self.general_logger.handlers) == 0:
            self.general_logger.addHandler(stream_handler)
            self.general_logger.addHandler(file_general_handler)
            self.general_logger.setLevel(logging.INFO)

    def info(self, message):
        # display time
        self.general_logger.info("[{}] - {}".format(self.now_string(), message))

    @staticmethod
    def now_string():
        return str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def copy_file(source_file, destination_file):
    try:
        shutil.copy(source_file, destination_file)
        print("copy complete")
    except IOError as e:
        print(f"copy error: {e}")


setup(Config)
LOGGER = Logger(OUTPUTS_EXP.as_posix())
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_train_label(train_df) -> pd.DataFrame:
    le = LabelEncoder()
    train_df["brand_id_label"] = le.fit_transform(train_df["brand_id"])
    train_df["meigara_label"] = le.fit_transform(train_df["meigara"])

    return train_df


def make_filepath(input_df):
    def join_path(dirpath, filename):
        return (dirpath / filename).as_posix()

    output_df = input_df.assign(
        filepath=input_df["filename"].apply(
            lambda x: join_path(QUERY_IMAGES, x)
            if str(x)[0] == "2"
            else join_path(CITE_IMAGES, x)
        )
    )
    return output_df


def make_label_dict(train_df):
    MLABEL2MEIGARA = (
        train_df[["meigara", "meigara_label"]]
        .set_index("meigara_label")
        .to_dict()["meigara"]
    )
    BLABEL2BLAND = (
        train_df[["brand_id", "brand_id_label"]]
        .set_index("brand_id_label")
        .to_dict()["brand_id"]
    )
    return MLABEL2MEIGARA, BLABEL2BLAND


train_df = pd.read_csv(INPUTS / "train.csv")
test_df = pd.read_csv(INPUTS / "test.csv")
cite_df = pd.read_csv(INPUTS / "cite.csv").rename(
    columns={"cite_gid": "gid", "cite_filename": "filename"}
)
sample_submission_df = pd.read_csv(INPUTS / "sample_submission.csv")

if Config.debug:
    train_df = train_df.sample(1000, random_state=Config.seed).reset_index(drop=True)
    cite_df = cite_df.sample(100, random_state=Config.seed).reset_index(drop=True)

# make label
train_df = make_train_label(train_df)
MLABEL2MEIGARA, BLABEL2BLAND = make_label_dict(train_df)
Config.num_classes = len(MLABEL2MEIGARA)

# make filepath
train_df = make_filepath(train_df)
test_df = make_filepath(test_df)
cite_df = make_filepath(cite_df)

# for submission
IDX2CITE_GID = cite_df.to_dict()["gid"]

print(train_df.shape)


def add_fold_idx(config, train_df):
    fold = StratifiedKFold(
        n_splits=config.n_fold, shuffle=True, random_state=config.seed
    )
    train_df["fold"] = -1
    for i_fold, (train_index, val_index) in enumerate(
        fold.split(train_df, train_df[config.target_columns])
    ):
        train_df.iloc[val_index, train_df.columns.get_loc("fold")] = int(i_fold)
    train_df["fold"] = train_df["fold"].astype(int)
    return train_df


train_df = add_fold_idx(config=Config, train_df=train_df)


class TrainDataset(Dataset):
    def __init__(self, df, target_columns=Config.target_columns, transform_fn=None):
        self.df = df
        self.file_names = df["filepath"].to_numpy()
        self.targets = df[target_columns].to_numpy()
        if len(target_columns) == 1:
            self.targets = np.ravel(self.targets)

        self.transform_fn = transform_fn

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        filepath = self.file_names[idx]
        image = cv2.imread(filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform_fn:
            image = self.transform_fn(image=image)["image"]
        target = torch.tensor(self.targets[idx])
        return {"images": image, "targets": target}


class TestDataset(Dataset):
    def __init__(self, df, transform_fn=None):
        self.df = df
        self.file_names = df["filepath"].to_numpy()
        self.transform_fn = transform_fn

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        filepath = self.file_names[idx]
        image = cv2.imread(filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform_fn:
            image = self.transform_fn(image=image)["image"]
        return {"images": image}


def set_transform(dataset, transform):
    if isinstance(dataset, torch.utils.data.Subset):
        set_transform(dataset.dataset, transform)
    else:
        dataset.transform_fn = transform


def get_transforms(size, data="train", p=0.5):
    if data == "train":
        return A.Compose(
            [
                # 正方形切り出し
                A.RandomResizedCrop(size, size, scale=(0.85, 1.0)),
                A.OneOf(
                    transforms=[
                        A.augmentations.geometric.rotate.SafeRotate(
                            limit=(-45, -90),
                        ),
                        A.augmentations.geometric.rotate.SafeRotate(
                            limit=(45, 90),
                        ),
                    ],
                    p=p,
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )

    elif data == "valid":
        return A.Compose(
            [
                A.Resize(size, size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )


class CustomModel(nn.Module):
    def __init__(self, config, pretrained=False, ls_eps=0.1):
        super().__init__()
        self.config = config
        self.model = timm.create_model(self.config.model_name, pretrained=pretrained)
        self.n_features = self.model.classifier.in_features
        self.model.global_pool = GeM()
        self.model.classifier = nn.Identity()  # custom head にするため
        self.fc = nn.Linear(self.n_features, config.embedding_dim)
        self.bn = nn.BatchNorm1d(config.embedding_dim)
        self.final = ArcMarginProduct(
            config.embedding_dim, config.num_classes, ls_eps=ls_eps
        )

    def forward(self, x, label):
        x = self.extract_features(x)
        x = self.final(x, label)
        return x

    def extract_features(self, images, labels=None):
        bs = images.size(0)
        x = self.model(images).view(bs, -1)
        x = self.fc(x)
        x = self.bn(x)
        return x


def train_fn(
    config,
    model,
    dataloader,
    criterion,
    optimizer,
    scheduler,
    device,
    _custom_step,
):
    model.train()
    scaler = torch.cuda.amp.GradScaler(init_scale=512, enabled=config.use_amp)
    losses = []

    tbar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, batch in tbar:
        for k, v in batch.items():
            batch[k] = v.to(device)
        targets = batch["targets"]
        batch_size = targets.size(0)

        with torch.cuda.amp.autocast(enabled=config.use_amp):
            batch_outputs = model(batch["images"], label=batch["targets"])
            loss = criterion(batch_outputs, targets)

        if config.gradient_accumulation_steps > 1:
            loss = loss / config.gradient_accumulation_steps

        scaler.scale(loss).backward()
        if config.clip_grad_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
            torch.nn.utils.clip_grad_norm_(
                criterion.parameters(), config.clip_grad_norm
            )

        if (step + 1) % config.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            _custom_step += 1
            if config.batch_scheduler:
                scheduler.step()

        losses.append(float(loss) * config.gradient_accumulation_steps)
        tbar.set_description(
            f"loss: {np.mean(losses) :.4f} lr: {scheduler.get_lr()[0]:.6f}"
        )

    loss = np.mean(losses)
    return loss, _custom_step


def valid_fn(
    config,
    model,
    dataloader,
    criterion,
    device,
    _custom_step,
):
    model.eval()
    outputs, targets = [], []
    losses = []

    tbar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, batch in tbar:
        targets.append(batch["targets"])

        for k, v in batch.items():
            batch[k] = v.to(device)

        batch_size = batch["targets"].size(0)
        with torch.no_grad():
            batch_outputs = model(batch["images"], label=batch["targets"])
            loss = criterion(batch_outputs, batch["targets"])

        # if config.gradient_accumulation_steps > 1:
        #     loss = loss / config.gradient_accumulation_steps

        batch_outputs = batch_outputs.to("cpu").numpy()
        outputs.append(batch_outputs)

        _custom_step += 1
        losses.append(float(loss))

        tbar.set_description(f"loss: {np.mean(losses):.4f}")

    outputs = np.concatenate(outputs)
    targets = np.concatenate(targets)

    loss = np.mean(losses)
    return (loss, outputs, targets, _custom_step)


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


def get_probability_by_epoch(epoch: int):
    return 2 ** (-epoch)


def train_loop(config, name, train_df, valid_df, device, summary_writer, criterion):
    LOGGER.info(f"========== {name} training ==========")

    # dataset, dataloader
    train_dataset = TrainDataset(
        df=train_df,
        target_columns=config.target_columns,
        transform_fn=get_transforms(data="train", size=config.size),
    )
    valid_dataset = TrainDataset(
        df=valid_df,
        target_columns=config.target_columns,
        transform_fn=get_transforms(data="valid", size=config.size),
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=config.valid_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # set model & optimizer
    model = CustomModel(config, pretrained=True)
    model.to(device)
    model.load_state_dict(torch.load(config.base_model_root / f"{name}.pth"))
    criterion.to(device)
    params = [{"params": model.parameters()}, {"params": criterion.parameters()}]
    optimizer = get_optimizer(optimizer_config=config.optimizer, params=params)

    # set scheduler
    num_train_steps = int(
        len(train_dataloader) * config.max_epochs // config.gradient_accumulation_steps
    )
    scheduler = get_scheduler(
        scheduler_config=config.scheduler,
        optimizer=optimizer,
        num_train_steps=num_train_steps,
    )

    # loop
    best_loss = np.inf
    tr_step, val_step = 0, 0
    for epoch in range(Config.max_epochs):
        train_dataset = TrainDataset(
            df=train_df,
            target_columns=config.target_columns,
            transform_fn=get_transforms(
                data="train", size=config.size, p=get_probability_by_epoch(epoch + 1)
            ),
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.train_batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        LOGGER.info(train_dataloader.dataset.transform_fn)
        # training
        loss, tr_step = train_fn(
            config=config,
            model=model,
            dataloader=train_dataloader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            _custom_step=tr_step,
        )
        torch.cuda.empty_cache()
        gc.collect()

        # validation
        val_loss, val_outputs, val_targets, val_step = valid_fn(
            config=config,
            model=model,
            dataloader=valid_dataloader,
            criterion=criterion,
            device=device,
            _custom_step=val_step,
        )
        summary_writer.add_scalar("train/loss", loss, epoch)
        summary_writer.add_scalar("val/loss", val_loss, epoch)

        logs = {
            "Epoch": epoch,
            "train_loss_epoch": loss.item(),
            "valid_loss_epoch": val_loss.item(),
        }
        LOGGER.info(logs)

        if best_loss > val_loss.item():
            best_loss = val_loss.item()
            LOGGER.info(f"epoch {epoch} - best loss: {best_loss:.4f} model")

            torch.save(model.state_dict(), f"{name}.pth")  # save model weight
            joblib.dump(val_outputs, f"{name}.pkl")  # save outputs

        if not config.batch_scheduler:
            scheduler.step()

    torch.cuda.empty_cache()
    gc.collect()

    # to escape drive storage error
    copy_file(f"{name}.pth", EXP_MODELS / f"{name}.pth")
    copy_file(f"{name}.pkl", EXP_PREDS / f"{name}.pkl")

    # save best predictions with id
    best_val_outputs = joblib.load(f"{name}.pkl")
    outputs = {
        "gid": valid_df["gid"].tolist(),
        "predictions": np.array(best_val_outputs, dtype=np.float16),
        "targets": val_targets,  # type: ignore
    }
    joblib.dump(outputs, EXP_PREDS / f"{name}_best.pkl")


seed_torch()
if Config.training:
    for i_fold in range(Config.n_fold):
        if i_fold not in Config.trn_fold:
            continue

        train_df_fold = train_df[train_df["fold"] != i_fold]
        valid_df_fold = train_df[train_df["fold"] == i_fold]
        criterion = FocalLoss()

        summary_writer = SummaryWriter(log_dir=OUTPUTS_EXP / f"fold_{i_fold}")
        train_loop(
            config=Config,
            name=f"fold_{i_fold}",
            train_df=train_df_fold,
            valid_df=valid_df_fold,
            device=DEVICE,
            summary_writer=summary_writer,
            criterion=criterion,
        )


def inference_fn(test_dataloader, model, device, features=False):
    preds, targets_masks = [], []
    model.eval()
    model.to(device)

    tbar = tqdm(test_dataloader, total=len(test_dataloader))
    for batch in tbar:
        for k, v in batch.items():
            batch[k] = v.to(device)

        with torch.no_grad():
            outputs = model.extract_features(batch["images"])

        outputs = outputs.cpu().detach().numpy()
        preds.append(outputs)

    return np.concatenate(preds)


def get_features(config, test_df, model_path, device):
    test_dataset = TestDataset(
        df=test_df, transform_fn=get_transforms(data="valid", size=config.size)
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.valid_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # get model
    model = CustomModel(config, pretrained=False)
    state = torch.load(model_path)
    model.load_state_dict(state)
    features = inference_fn(test_dataloader, model, device, features=True)

    del model, state, test_dataloader
    gc.collect()
    torch.cuda.empty_cache()

    outputs = {
        "gid": test_df["gid"].tolist(),
        "features": np.array(features, dtype=np.float16),
    }
    return outputs


if Config.embedding:
    # oof
    oof_features_filepath = EXP_PREDS / "oof_embeddings.pkl"
    if not oof_features_filepath.is_file():
        oof_features, oof_gids = [], []
        for i_fold in range(Config.n_fold):
            if i_fold not in Config.trn_fold:
                continue

            gids = joblib.load(EXP_PREDS / f"fold_{i_fold}_best.pkl")["gid"]
            df = train_df[train_df["gid"].isin(gids)].reset_index(drop=True)

            outputs = get_features(
                config=Config,
                test_df=df,
                model_path=EXP_MODELS / f"fold_{i_fold}.pth",
                device=DEVICE,
            )
            oof_features.append(outputs["features"])
            oof_gids.extend(outputs["gid"])
        oof_features = np.concatenate(oof_features, axis=0)
        joblib.dump(
            {"gid": oof_gids, "embeddings": oof_features}, oof_features_filepath
        )

    # query images
    query_features_filepath = EXP_PREDS / "test_embeddings.pkl"
    if not query_features_filepath.is_file():
        query_features = []
        for i_fold in range(Config.n_fold):
            if i_fold not in Config.trn_fold:
                continue

            outputs = get_features(
                config=Config,
                test_df=test_df,
                model_path=EXP_MODELS / f"fold_{i_fold}.pth",
                device=DEVICE,
            )
            query_features.append(outputs["features"])
        joblib.dump(
            {"gid": outputs["gid"], "embeddings_list": query_features},
            query_features_filepath,
        )

    # cite images
    cite_features_filepath = EXP_PREDS / "cite_embeddings.pkl"
    if not cite_features_filepath.is_file():
        cite_features = []
        for i_fold in range(Config.n_fold):
            if i_fold not in Config.trn_fold:
                continue

            outputs = get_features(
                config=Config,
                test_df=cite_df,
                model_path=EXP_MODELS / f"fold_{i_fold}.pth",
                device=DEVICE,
            )
            cite_features.append(outputs["features"])
        joblib.dump(
            {"gid": outputs["gid"], "embeddings_list": cite_features},
            cite_features_filepath,
        )


class SimilaritySearcher:
    def __init__(self, embeddings):
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.dimension)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)  # type:ignore

    def search(self, queries, k=10):
        assert (
            queries.shape[1] == self.dimension
        ), "Query dimensions should match embeddings dimension."
        faiss.normalize_L2(queries)
        D, I = self.index.search(queries, k)  # type:ignore
        return D, I


cite_features = np.array(
    joblib.load(EXP_PREDS / "cite_embeddings.pkl")["embeddings_list"]
)
query_features = np.array(
    joblib.load(EXP_PREDS / "test_embeddings.pkl")["embeddings_list"]
)

ave_cite_feature = np.mean(cite_features, axis=0)
ave_query_feature = np.mean(query_features, axis=0)

searcher = SimilaritySearcher(ave_cite_feature.astype(np.float32))
D, I = searcher.search(ave_query_feature.astype(np.float32), k=20)


def make_submission(indices):
    vfunc = np.vectorize(lambda x: IDX2CITE_GID[x])
    gid_array = vfunc(I)
    submission_df = test_df[["gid"]].assign(
        cite_gid=[" ".join(list(x)) for x in gid_array.astype(str)]
    )
    return submission_df


submission_df = make_submission(indices=I)
submission_df.to_csv(SUBMISSIONS / f"{Config.name}.csv", index=False)


submission_df = pd.read_csv(SUBMISSIONS / f"{Config.name}.csv")
