"""Hydra-based training script for flexible experiments."""
from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from src.trainers import run_training


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    run_training(cfg)


if __name__ == "__main__":
    main()
