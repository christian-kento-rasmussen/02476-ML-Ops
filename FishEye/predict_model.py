import numpy as np
import torch
from FishEye.models.model import FishNN
from FishEye.data.data_module import FishDataModule

def predict(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader) -> None:
    """this function predicts the output of the model

    Args:
        model (torch.nn.Module):    the model to predict
        dataloader (torch.utils.data.DataLoader): _description_

    Returns:
        _type_: _description_
    """

    return torch.cat([model(batch) for batch in dataloader], 0)

if __name__ == "__main__":
    model = FishNN.load_from_checkpoint("models/epoch=10-step=121.ckpt")
    fishDataModule = FishDataModule(batch_size=32)
    predict(model, )