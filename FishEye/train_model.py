import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

import wandb
from FishEye.data.data_module import FishDataModule
from FishEye.models.model import FishNN


@hydra.main(version_base=None, config_path="../config", config_name="config")
def train(cfg: DictConfig):
    """This function trains the model

    Args:
        cfg (DictConfig): Hydra config
    """

    # Load all hyperparameters from the config file
    BATCH_SIZE = cfg.trainer_hyperparameters.batch_size
    MAX_EPOCHS = cfg.trainer_hyperparameters.max_epochs
    PATIENCE = cfg.trainer_hyperparameters.patience
    CHECK_VAL_EVERY_N_EPOCH = cfg.trainer_hyperparameters.check_val_every_n_epoch
    MODE = cfg.trainer_hyperparameters.mode
    MONITOR = cfg.trainer_hyperparameters.monitor
    SAVE_BY = cfg.trainer_hyperparameters.save_by
    AUGMENT_TRAIN = cfg.trainer_hyperparameters.augment_train

    # Extracting path
    DATAPATH = cfg.paths.processed
    SAVE_MODEL_PATH = cfg.paths.save_model_path

    # Initialize the model and the data module
    model = FishNN(cfg)
    fishDataModule = FishDataModule(data_dir=DATAPATH, batch_size=BATCH_SIZE, augment=AUGMENT_TRAIN)

    # Initialize the callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=SAVE_MODEL_PATH, monitor=MONITOR, mode=MODE)
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

    # log a few train images to wandb
    fishDataModule.setup("train")
    train_loader = fishDataModule.train_dataloader()
    train_batch = next(iter(train_loader))
    train_images, train_labels = train_batch
    wandb.log({"train_images": [wandb.Image(image, caption=label) for image, label in zip(train_images, train_labels)]})

    # compiles model
    # model.compile()

    # train image classifier
    trainer.fit(model, fishDataModule)

    # Test the model with the lowest validation loss
    trainer.test(datamodule=fishDataModule, ckpt_path=SAVE_BY)


if __name__ == "__main__":
    train()
