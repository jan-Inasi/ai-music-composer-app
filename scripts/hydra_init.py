import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../config", config_name="init_config.yaml")
def main(cfg: DictConfig):
    # print("hello")
    print(cfg)


if __name__ == "__main__":
    main()
