from typing import Any

import pytorch_lightning as L
import torch
from omegaconf import DictConfig
from torch import nn, optim
from torchmetrics.classification import Accuracy


class FishNN(L.LightningModule):
    def __init__(self, cfg: DictConfig, *args: Any, **kwargs: Any) -> None:
        """Initializes the FishNN model

        Args:
            cfg (DictConfig): hydra config file
        """
        super().__init__(*args, **kwargs)
        self.accuracy = Accuracy(task="multiclass", num_classes=9)

        self.classifier = nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, stride=2), # dim (h,w) = (445-3)/2+1=222,  (590-3)/2+1=294
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=1), # dim (h,w)
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(2055680, 256), # dim: 32*292*220 = 2037760
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 9), # dim: 32*292*220 = 2037760
            torch.nn.Dropout(0.2),
        )

        self.lr = cfg.trainer_hyperparameters.learning_rate
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output logits
        """
        return self.classifier(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Training step for the model

        Args:
            batch (torch.Tensor): batch of shape (batch_size, 3, 590, 445)
            batch_idx (int): the index of the batch

        Returns:
            torch.Tensor: loss tensor
        """
        data, target = batch
        preds = self(data)
        loss = self.criterion(preds, target)
        train_acc = self.accuracy(preds, target)
        self.log("train_acc", train_acc, prog_bar=True, logger=True)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """validation step for the model

        Args:
            batch (torch.Tensor): batch of shape (batch_size, 3, 590, 445)
            batch_idx (int): the index of the batch

        Returns:
            torch.Tensor: loss tensor
        """
        data, target = batch
        preds = self(data)
        loss = self.criterion(preds, target)
        val_acc = self.accuracy(preds, target)
        self.log("val_acc", val_acc, prog_bar=True, logger=True)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Test step for the model

        Args:
            batch (torch.Tensor): batch of shape (batch_size, 3, 590, 445)
            batch_idx (int): the index of the batch

        Returns:
            torch.Tensor: loss tensor
        """
        data, target = batch
        preds = self(data)
        loss = self.criterion(preds, target)
        test_acc = self.accuracy(preds, target)

        self.log("test_acc", test_acc, prog_bar=True, logger=True)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configures the optimizer for the model

        Returns:
            torch.optim.Optimizer: optimizer
        """
        return optim.Adam(self.parameters(), lr=self.lr)
