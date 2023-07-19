import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, df, target_columns, transform_fn=None):
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
