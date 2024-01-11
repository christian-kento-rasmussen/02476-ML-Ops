from typing import Any

import pytorch_lightning as L
import torch
from torch import nn, optim


class FishNN(L.LightningModule):
    def __init__(self, cfg, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.classifier = nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, stride=2),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(16 * 294 * 222, 10),
        )

        self.lr = cfg.trainer_hyperparameters.learning_rate
        self.criterium = torch.nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        data, target = batch
        preds = self(data)
        loss = self.criterium(preds, target)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        data, target = batch
        preds = self(data)
        loss = self.criterium(preds, target)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> Any:
        data, target = batch
        preds = self(data)
        loss = self.criterium(preds, target)
        return loss

    def configure_optimizers(self) -> Any:
        return optim.Adam(self.parameters(), lr=self.lr)


if __name__ == "__main__":
    pass
