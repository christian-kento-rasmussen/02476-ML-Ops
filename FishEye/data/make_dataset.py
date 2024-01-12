import json
import os
from glob import glob

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def process_data(raw_data_path: str = "data/raw/NA_Fish_Dataset", processed_data_path: str = "data/processed") -> None:
    """Processes the raw data and saves it to the processed_data_path

    Args:
        raw_data_path (str, optional): Location of raw data folder . Defaults to "data/raw/NA_Fish_Dataset".
        processed_data_path (str, optional): Location of processed data folder. Defaults to "data/processed".
    """
    folder_paths = glob(os.path.join(raw_data_path, "*"))

    # maps label index to label name
    label_map = {}

    images = []
    labels = []

    # iterates through each folder in raw_data_path and adds to images and labels lists
    for label_i, folder in tqdm(enumerate(folder_paths), total=len(folder_paths), desc="Processing data"):
        label_map[label_i] = folder.split("/")[-1]
        for image_path in glob(os.path.join(folder, "*")):
            image = Image.open(image_path)
            image = image.resize((590, 445))  # converts all images to the same size (smallest image is 590x445)
            image = image.convert("RGB")  # Convert image to RGB format
            image = torch.tensor(image)  # Convert image to torch tensor

            images.append(image)
            labels.append(torch.tensor(label_i))

    stacked_images = torch.stack(images)
    stacked_images = stacked_images.permute(0, 3, 1, 2)  # fixes the shape of the images

    stacked_labels = torch.stack(labels)

    # saves data to processed_data_path
    torch.save(stacked_images, os.path.join(processed_data_path, "images.pt"))
    torch.save(stacked_labels, os.path.join(processed_data_path, "labels.pt"))

    # saves the mapping of labels to their names
    with open(os.path.join(processed_data_path, "label_map.json"), "w") as fp:
        json.dump(label_map, fp)


if __name__ == "__main__":
    process_data()
