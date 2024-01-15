from FishEye.train_model import train
import hydra
from omegaconf import OmegaConf
import torch.autograd.profiler as profiler


@hydra.main(config_path="config", config_name="config")
def main(cfg):
    # Print the config
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
