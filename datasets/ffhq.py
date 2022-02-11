from io import BytesIO

import lmdb
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from pytorch_lightning import LightningDataModule
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, Subset, DataLoader

import clip

class FFHQ(Dataset):
    def __init__(self, path, transform, resolution=8):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)
        target = 0

        return img, target

class FFHQ_Datamodule(LightningDataModule):
    def __init__(self, path, config):
        super().__init__()
        self.path = path
        self.config = config

        _, self.transform = clip.load(self.config.clip.model)

    def setup(self, stage):
        dataset = FFHQ(path=self.path, transform=self.transform, resolution=256)

        num_items = len(dataset)
        indices = list(range(num_items))
        random_state = np.random.get_state()
        np.random.seed(2022)
        np.random.shuffle(indices)
        np.random.set_state(random_state)
        self.train_indices, self.val_indices = indices[:int(num_items * 0.95)], indices[int(num_items * 0.95):]

        self.train_dataset = Subset(dataset, self.train_indices)
        self.val_dataset = Subset(dataset, self.val_indices)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                num_workers=self.config.data.num_workers, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config.training.batch_size, shuffle=False,
                                 num_workers=self.config.data.num_workers, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config.training.batch_size, shuffle=False,
                                 num_workers=self.config.data.num_workers, drop_last=True)

    def predict_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config.training.batch_size, shuffle=False,
                                 num_workers=self.config.data.num_workers, drop_last=True)
        
