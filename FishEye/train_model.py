import pytorch_lightning as pl
from pytorch_lightning import Trainer

from FishEye.data.data_module import FishDataModule
from FishEye.models.model import FishNN

import wandb
from pytorch_lightning.loggers import WandbLogger
from omegaconf import OmegaConf


def train(cfg):
    # Load all hyperparameters from the config file
    BATCH_SIZE = cfg.trainer_hyperparameters.batch_size
    MAX_EPOCHS = cfg.trainer_hyperparameters.max_epochs
    PATIENCE = cfg.trainer_hyperparameters.patience
    CHECK_VAL_EVERY_N_EPOCH = cfg.trainer_hyperparameters.check_val_every_n_epoch
    MODE = cfg.trainer_hyperparameters.mode
    MONITOR = cfg.trainer_hyperparameters.monitor

    model = FishNN(cfg)  # this is our LightningModule
    fishDataModule = FishDataModule(batch_size=BATCH_SIZE)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath="./models", monitor=MONITOR, mode=MODE)
    early_stopping_callback = pl.callbacks.EarlyStopping(monitor=MONITOR, patience=PATIENCE)

    # Initialize a W&B logger
    wandb.init(
        project=cfg.wandb_settings.project,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        entity=cfg.wandb_settings.entity,
        mode=cfg.wandb_settings.mode,
    )
    wandb_logger = WandbLogger(experiment=wandb.run)

    trainer = Trainer(
        max_epochs=MAX_EPOCHS,
        check_val_every_n_epoch=CHECK_VAL_EVERY_N_EPOCH,
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=wandb_logger,
    )

    trainer.fit(model, fishDataModule)

    # Test the model with the lowest validation loss
    trainer.test(datamodule=fishDataModule, ckpt_path="best")
