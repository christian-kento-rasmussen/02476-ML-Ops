import hydra
import pytest
import torch
from omegaconf import OmegaConf

from FishEye.models.model import FishNN


def test_output_shape_of_model(cfg=OmegaConf.load("config/config.yaml")):
    # Create an instance of your model
    model = FishNN(cfg)

    # Create an example input tensor
    input_tensor = torch.randn(1, 3, 445, 590)

    # Pass the input tensor through the model
    output_tensor = model(input_tensor)

    # Check if the output tensor has the correct shape
    expected_shape = (1, 9)
    assert output_tensor.shape == expected_shape
