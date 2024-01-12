import json
import os
from glob import glob

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def process_data(raw_data_path:str ="data/raw/NA_Fish_Dataset", processed_data_path:str="data/processed") -> None:
    """_summary_

    Args:
        raw_data_path (str, optional): _description_. Defaults to "data/raw/NA_Fish_Dataset".
        processed_data_path (str, optional): _description_. Defaults to "data/processed".
    """


    folder_paths = glob(os.path.join(raw_data_path, "*"))

    label_map = {}

    images = []
    labels = []

    for label_i, folder in tqdm(enumerate(folder_paths), total=len(folder_paths), desc="Processing data"):
        label_map[label_i] = folder.split("/")[-1]
        for image_path in glob(os.path.join(folder, "*")):
            image = Image.open(image_path)
            image = image.resize((590, 445))
            image = image.convert("RGB")  # Convert image to RGB format
            image = torch.Tensor(np.array(image))  # Convert image to torch tensor
            images.append(image)
            labels.append(torch.tensor(label_i))

    stacked_images = torch.stack(images)
    stacked_images = stacked_images.permute(0, 3, 1, 2)
    torch.save(stacked_images, os.path.join(processed_data_path, "images.pt"))
    stacked_labels = torch.stack(labels)
    torch.save(stacked_labels, os.path.join(processed_data_path, "labels.pt"))
    with open(os.path.join(processed_data_path, "label_map.json"), "w") as fp:
        json.dump(label_map, fp)



if __name__ == "__main__":
    # Get the data and process it

    process_data()
