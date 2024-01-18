import pytest
from omegaconf import OmegaConf
import os 
from FishEye.train_model import train


def test_train_loop(cfg=OmegaConf.load("config/config.yaml")):
    
    # Overwrite the config file for testing purposes
    cfg.trainer_hyperparameters.max_epochs = 1
    cfg.wandb_settings.mode = 'offline'
    cfg.trainer_hyperparameters.save_by = "last"
    
    # Overwrite the save_model_path location
    cfg.paths.save_model_path = "tests"
    cfg.paths.processed = "data/processed"
    
    # Test that we can run the train loop without errors for 1 epoch
    train(cfg)
    