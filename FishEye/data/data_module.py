import os

import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class FishEyeDataset(Dataset):
    def __init__(self, images, labels, transforms=None):
        assert torch.is_tensor(images), "images must be a tensor"
        assert torch.is_tensor(labels), "labels must be a tensor"
        
        self.images = images
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transforms:
            image = self.transforms(image)

        return image, label

# Rest of your code
class FishDataModule(pl.LightningDataModule):
    def __init__(
        self, data_dir: str = "data/processed", batch_size: int = 32, augment: bool = True, num_workers: int = 4
    ):
        """Inits FishDataModule.

        Args:
            data_dir (str, optional): location of processed data. Defaults to "data/processed".
            batch_size (int, optional): batch size. Defaults to 32.
            augment (bool, optional): whether to augment the data. Defaults to True.
            num_workers (int, optional): number of workers for the dataloader. Defaults to 4.
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.prepare_data_per_node = True
        self.augment = augment
        self.num_workers = num_workers

        self.aug_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomAffine(
                    degrees=(-20, 20), scale=(0.95, 1.5)
                ),  # Scales the image by a factor between 0.95 and 1.05
            ]
        ) 

    def setup(self, stage: str):
        """Loads data from data_dir and splits into train, val, test sets.

        Args:
            stage (str): stage of training (fit, test, predict)
        """
        images = torch.load(os.path.join(self.data_dir, "images.pt"))
        labels = torch.load(os.path.join(self.data_dir, "labels.pt"))

        # Split the data into train and remaining data
        train_images, remaining_images, train_labels, remaining_labels = train_test_split(
            images, labels, test_size=0.2, stratify=labels, random_state=69
        )

        # Split the remaining data into validation and test sets
        val_images, test_images, val_labels, test_labels = train_test_split(
            remaining_images, remaining_labels, test_size=0.5, stratify=remaining_labels, random_state=69
        )

        self.train_dataset = FishEyeDataset(train_images, train_labels, transforms=self.aug_transforms)
        self.val_dataset = FishEyeDataset(val_images, val_labels, transforms=None)
        self.test_dataset = FishEyeDataset(test_images, test_labels, transforms=None)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
