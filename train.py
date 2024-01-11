from FishEye.train_model import train
import hydra
import wandb
from omegaconf import OmegaConf


@hydra.main(config_path="config", config_name="config")
def main(cfg):
    

    # Print the config
    if cfg.print_cfg:
        print(OmegaConf.to_yaml(cfg))

    train(cfg)


if __name__ == "__main__":
    main()
