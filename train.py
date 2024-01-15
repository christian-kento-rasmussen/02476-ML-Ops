import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
import torch.autograd.profiler as profiler

from FishEye.train_model import train


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    """Main entry point for training.

    Args:
        cfg (DictConfig): hydra config file
    """

    if cfg.print_cfg:
        print(OmegaConf.to_yaml(cfg))

    if cfg.profile:
        with profiler.profile(record_shapes=True) as prof:
            train(cfg)
        print(prof.key_averages().table(sort_by="self_cpu_time_total"))

    else:
        train(cfg)


if __name__ == "__main__":
    main()
