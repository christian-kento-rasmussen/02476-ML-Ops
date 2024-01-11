import pytorch_lightning as pl
from pytorch_lightning import Trainer

from FishEye.data.data_module import FishDataModule
from FishEye.models.model import FishNN

import hydra

@hydra.main(config_path="conf", config_name="config")
def train():
    model = FishNN()  # this is our LightningModule
    fishDataModule = FishDataModule(batch_size=32)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath="./models", monitor="val_loss", mode="min")
    early_stopping_callback = pl.callbacks.EarlyStopping(monitor="val_loss", patience=10)
    trainer = Trainer(
        max_epochs=100,
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_callback, early_stopping_callback],
    )

    trainer.fit(model, fishDataModule)
    
    # Load the best model from the checkpoint
    best_model_path = checkpoint_callback.best_model_path
    best_model = FishNN.load_from_checkpoint(best_model_path)

    # Test the best model
    trainer.test(best_model, fishDataModule)

if __name__ == "__main__":
    train()
