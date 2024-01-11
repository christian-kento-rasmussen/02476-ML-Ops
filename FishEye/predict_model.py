import torch


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
    pass
