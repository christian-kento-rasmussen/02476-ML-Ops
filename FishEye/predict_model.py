import json
from typing import Dict, List
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.nn.functional import interpolate

from FishEye.models.model import FishNN


def predict(model: torch.nn.Module, images: torch.Tensor, label_mapping: Dict) -> None:
    """This function predicts the output of the model

    Args:
        model (torch.nn.Module): A trained model used to predict
        dataloader (torch.utils.data.DataLoader): dataloader

    Returns:
        (torch.Tensor): the predictions
    """

    # Get model predictions
    predictions = model(images)

    predictions = torch.argmax(predictions, dim=1)

    # Convert the string integer labels to the corresponding fish names using the label_mapping
    predictions = [label_mapping[str(prediction.item())] for prediction in predictions]

    return predictions


def preprocess_images(images: List[torch.Tensor]) -> torch.Tensor:
    # For each image in images resize it to 590x445 and convert it to RGB format
    for i in range(len(images)):
        image = images[i]
        image = interpolate(image.unsqueeze(0), size=(445, 590), mode="bilinear", align_corners=False)
        images[i] = image.squeeze(0)

    return torch.stack(images)


if __name__ == "__main__":
    # INstanciate a FishNN model
    # model = FishNN(cfg=OmegaConf.load("config/config.yaml"))

    # Load the best model weights into the model
    import torchvision.transforms as transforms

    model = FishNN.load_from_checkpoint("models/epoch=409-step=4510.ckpt", cfg=OmegaConf.load("config/config.yaml"))
    model.to("cpu")
    model.eval()

    # Raw data to predict
    image_filenames = [
        "data/raw/NA_Fish_Dataset/Black Sea Sprat/00001.png",
        "data/raw/NA_Fish_Dataset/Black Sea Sprat/00002.png",
        "data/raw/NA_Fish_Dataset/Black Sea Sprat/00003.png",
        "data/raw/NA_Fish_Dataset/Black Sea Sprat/00004.png",
        "data/raw/NA_Fish_Dataset/Black Sea Sprat/00005.png",
        "data/raw/NA_Fish_Dataset/Black Sea Sprat/00006.png",
        "data/raw/NA_Fish_Dataset/Black Sea Sprat/00007.png",
        "data/raw/NA_Fish_Dataset/Black Sea Sprat/00008.png",
        "data/raw/NA_Fish_Dataset/Black Sea Sprat/00009.png",
        "data/raw/NA_Fish_Dataset/Black Sea Sprat/00010.png",
        "data/raw/NA_Fish_Dataset/Black Sea Sprat/00011.png",
        "data/raw/NA_Fish_Dataset/Black Sea Sprat/00012.png",
        "data/raw/NA_Fish_Dataset/Black Sea Sprat/00013.png",
        "data/raw/NA_Fish_Dataset/Black Sea Sprat/00014.png",
        "data/raw/NA_Fish_Dataset/Black Sea Sprat/00015.png",
        "data/raw/NA_Fish_Dataset/Black Sea Sprat/00016.png",
        "data/raw/NA_Fish_Dataset/Black Sea Sprat/00017.png",
        "data/raw/NA_Fish_Dataset/Black Sea Sprat/00018.png",
        "data/raw/NA_Fish_Dataset/Black Sea Sprat/00019.png",
        "data/raw/NA_Fish_Dataset/Black Sea Sprat/00020.png",
    ]

    # Load the images from the given paths as tensors
    images = [transforms.ToTensor()(Image.open(filename)) for filename in image_filenames]

    # Run the preprocessing on the images
    images = preprocess_images(images)

    # Specify the label mapping
    with open("data/processed/label_map.json", "r") as fp:
        label_mapping = json.load(fp)

    # Run the prediction
    predictions = predict(model, images, label_mapping)

    print(predictions)
