"""Simple Mixtral 8x7B preset that reuses the Hydra training pipeline."""
from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from src.trainers import run_training

DEFAULT_OUTPUT_DIR = "./models/mixtral-8x7b-sft"
DEFAULT_WANDB_NAME = "mixtral-8x7b-sft-test-1"


def _load_base_config() -> DictConfig:
    config_dir = Path(__file__).resolve().parent.parent / "configs"
    with initialize_config_dir(config_dir=str(config_dir), job_name="mixtral_script"):
        cfg = compose(config_name="config")
    OmegaConf.set_readonly(cfg, False)
    return cfg


def main() -> None:
    cfg = _load_base_config()
    cfg.training.output_dir = DEFAULT_OUTPUT_DIR
    cfg.wandb.name = DEFAULT_WANDB_NAME
    run_training(cfg)


if __name__ == "__main__":
    main()
