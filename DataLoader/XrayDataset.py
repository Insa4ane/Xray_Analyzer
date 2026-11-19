import numpy as np
from config.config import IMG_SIZE
from torch.utils.data import Dataset
import torch as tc
import logging

class XrayDataset(Dataset):
    def __init__(self, df, labels, image_loader):
        self.df = df.reset_index(drop=True)
        self.labels = labels
        self.image_loader = image_loader
        self.size = IMG_SIZE
        pass


    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):
        row=self.df.iloc[idx]
        img=self.image_loader(row['path'])
        try:
            X = np.array(img).reshape(1, self.size, self.size) / 255.0
            y = row[self.labels].to_numpy(dtype=np.float32)
            X = tc.tensor(X, dtype=tc.float32)
            y = tc.tensor(y, dtype=tc.float32)
            return X, y
        except Exception as e:
            raise ValueError(f"Cannot build sample for {row['path']}: {e}") from e

