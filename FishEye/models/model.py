from typing import Any

import pytorch_lightning as L
import torch
from omegaconf import DictConfig
from torch import nn, optim


class FishNN(L.LightningModule):
    def __init__(self, cfg: DictConfig, *args: Any, **kwargs: Any) -> None:
        """Initializes the FishNN model

        Args:
            cfg (DictConfig): hydra config file
        """
        super().__init__(*args, **kwargs)
        self.classifier = nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, stride=2),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(16 * 294 * 222, 10),
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

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
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
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configures the optimizer for the model

        Returns:
            torch.optim.Optimizer: optimizer
        """
        return optim.Adam(self.parameters(), lr=self.lr)
