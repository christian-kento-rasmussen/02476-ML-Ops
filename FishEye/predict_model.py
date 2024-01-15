import torch


def predict(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader) -> None:
    """This function predicts the output of the model

    Args:
        model (torch.nn.Module): the model to predict
        dataloader (torch.utils.data.DataLoader): dataloader

    Returns:
        (torch.Tensor): the predictions
    """

    return torch.cat([model(batch) for batch in dataloader], 0)
