import os
import os.path
import pdb

import pytest
import torch

from FishEye.data.data_module import FishDataModule
from FishEye.data.make_dataset import process_data


# @pytest.mark.skipif(not os.path.exists("data/processed/images.pt"), reason="Data files not found")
def test_makedata():
    # Run the script to make the data
    process_data()

    # Load the processed data
    images = torch.load("data/processed/images.pt")
    labels = torch.load("data/processed/labels.pt")

    # We should have 430 images and labels
    assert len(images) == 430, "Dataset did not have the correct number of images, 430 expected"
    assert len(labels) == 430, "Dataset did not have the correct number of labels, 430 expected"

    # The shape of each image should be [3, 445, 590]
    assert images[0].shape == torch.Size([3, 445, 590]), "Image shape was not as expected. Need to be 3x445x590"

    # Assert that all 9 class labels are represented
    assert (
        len(set(labels.numpy())) == 9
    ), "Not all class labels are represented in the dataset. Should contain labels from 0 through 8"


def test_datamodule():
    # Instantiate the data module
    data_module = FishDataModule()
    data_module.setup(stage="fit")

    # Check that the data module has the correct number of images in
    # each of the train, val, and test data loaders
    assert len(data_module.train_dataloader().dataset) == int(
        430 * 0.8
    ), "Train data loader does not have the correct number of images"
    assert len(data_module.val_dataloader().dataset) == int(
        430 * 0.1
    ), "Validation data loader does not have the correct number of images"
    assert len(data_module.test_dataloader().dataset) == int(
        430 * 0.1
    ), "Test data loader does not have the correct number of images"

    # Check that all images in each dataloader have correct shape 3x445x590 and type tensor
    for batch in data_module.train_dataloader():
        for image in batch[0]:
            assert image.shape == torch.Size([3, 445, 590]), "Train data loader did not have the correct image shape"
            assert type(image) == torch.Tensor, "Train data loader did not have the correct image type"
        assert type(batch[1]) == torch.Tensor, "Train data loader did not have the correct label type"
