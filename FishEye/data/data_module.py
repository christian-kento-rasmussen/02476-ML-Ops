import os

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split


class FishDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "data/processed", batch_size: int = 32):
        """Inits FishDataModule.

        Args:
            data_dir (str, optional): location of processed data. Defaults to "data/processed".
            batch_size (int, optional): batch size. Defaults to 32.
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.prepare_data_per_node = True

    def setup(self, stage: str):
        """Loads data from data_dir and splits into train, val, test sets.

        Args:
            stage (str): stage of training (fit, test, predict)
        """
        images = torch.load(os.path.join(self.data_dir, "images.pt"))
        labels = torch.load(os.path.join(self.data_dir, "labels.pt"))
        dataset = TensorDataset(images, labels)
        self.train, self.test, self.val = random_split(
            dataset,
            [int(len(dataset) * 0.8), int(len(dataset) * 0.1), int(len(dataset) * 0.1)],
            generator=torch.Generator().manual_seed(69),
        )

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        pass
