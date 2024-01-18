import hydra
from omegaconf import DictConfig, OmegaConf

from FishEye.train_model import train


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    """Main entry point for training.

    Args:
        cfg (DictConfig): hydra config file
    """

    if cfg.print_cfg:
        print(OmegaConf.to_yaml(cfg))

    train(cfg)


if __name__ == "__main__":
    main()
