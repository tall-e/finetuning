"""Validate Hydra config without full dependencies"""
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def validate(cfg: DictConfig):
    print("=" * 60)
    print("CONFIG VALIDATION")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)
    print("✓ Config loaded successfully!")
    print(f"✓ Model: {cfg.model.name}")
    print(f"✓ Data mixture: {len(cfg.data_mixture.sources)} source(s)")
    for i, source in enumerate(cfg.data_mixture.sources):
        print(f"  - Source {i+1}: {source.name} (weight: {source.weight})")
    print(f"✓ Mixing strategy: {cfg.data_mixture.mixing_strategy}")
    print("=" * 60)


if __name__ == "__main__":
    validate()
