import pytorch_lightning as pl
from pytorch_lightning import Trainer

from FishEye.data.data_module import FishDataModule
from FishEye.models.model import FishNN

import hydra

@hydra.main(config_path="conf", config_name="config.yaml")
def train(cfg):

    # Load all hyperparameters from the config file
    BATCH_SIZE = cfg.trainer_hyperparameters.batch_size
    LEARNING_RATE = cfg.trainer_hyperparameters.learning_rate
    SEED = cfg.trainer_hyperparameters.seed # TODO: Fix seed
    MAX_EPOCHS = cfg.trainer_hyperparameters.max_epochs
    PATIENCE = cfg.trainer_hyperparameters.patience
    CHECK_VAL_EVERY_N_EPOCH = cfg.trainer_hyperparameters.check_val_every_n_epoch
    MODE = cfg.trainer_hyperparameters.mode
    MONITOR =  cfg.trainer_hyperparameters.monitor

    model = FishNN(cfg)  # this is our LightningModule
    fishDataModule = FishDataModule(batch_size=BATCH_SIZE)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath="./models", monitor=MONITOR, mode=MODE)
    early_stopping_callback = pl.callbacks.EarlyStopping(monitor=MONITOR, patience=PATIENCE)

    trainer = Trainer(
        max_epochs=MAX_EPOCHS,
        check_val_every_n_epoch=CHECK_VAL_EVERY_N_EPOCH,
        callbacks=[checkpoint_callback, early_stopping_callback]
    )

    trainer.fit(model, fishDataModule)
    
    # Load the best model from the checkpoint
    best_model_path = checkpoint_callback.best_model_path
    best_model = FishNN.load_from_checkpoint(best_model_path)

    # Test the best model
    trainer.test(best_model, fishDataModule)

if __name__ == "__main__":
    train()
