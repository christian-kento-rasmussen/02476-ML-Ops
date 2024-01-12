import pytest
import torch
import os
from tests import _PATH_DATA, _TEST_ROOT, _PROJECT_ROOT
import os.path

@pytest.mark.skipif(not os.path.exists("data/processed/images.pt"), reason="Data files not found")
def test_data():
    
    # Load data
    images = torch.load('data/processed/images.pt')
    labels = torch.load('data/processed/labels.pt')
    
    # We should have 430 images and labels
    assert len(images) == 430, "Dataset did not have the correct number of images, 430 expected"
    assert len(labels) == 430, "Dataset did not have the correct number of labels, 430 expected"
    
    # The shape of each image should be [3, 445, 590]
    assert images[0].shape == torch.Size([3, 445, 590]), "Image shape was not as expected. Need to be 3x445x590"
    
    # Assert that all 9 class labels are represented
    assert len(set(labels.numpy())) == 9, "Not all class labels are represented in the dataset. Should contain labels from 0 through 8"
    