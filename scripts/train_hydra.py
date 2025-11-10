"""Hydra-based training script for flexible experiments."""
from dataclasses import dataclass
from string import Formatter
from typing import List, Optional

import torch
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer,AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from trl import SFTTrainer
from datasets import load_dataset, concatenate_datasets, interleave_datasets
from peft import LoraConfig
import wandb
import random


@dataclass
class DataSourceConfig:
    """Normalized view of a dataset source from the config."""

    name: Optional[str] = None
    path: Optional[str] = None
    split: str = "train"
    weight: float = 1.0
    max_samples: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.name and not self.path:
            raise ValueError("Each data source must define either 'name' or 'path'.")
        if self.name and self.path:
            raise ValueError("Specify only one of 'name' or 'path' per data source.")
        if self.weight <= 0:
            raise ValueError("Data source weights must be positive.")
        if self.max_samples is not None and self.max_samples <= 0:
            raise ValueError("max_samples must be positive when provided.")


def _extract_template_fields(template: str) -> List[str]:
    fields = {field_name for _, field_name, _, _ in Formatter().parse(template) if field_name}
    return sorted(fields)


def load_and_mix_datasets(cfg: DictConfig):
    """Load multiple datasets and mix them according to config."""
    random.seed(cfg.data_mixture.seed)

    if not cfg.data_mixture.sources:
        raise ValueError("cfg.data_mixture.sources must include at least one dataset source.")

    datasets = []
    source_metadata: List[DataSourceConfig] = []

    for source_cfg in cfg.data_mixture.sources:
        source_dict = OmegaConf.to_container(source_cfg, resolve=True)
        normalized = DataSourceConfig(**source_dict)

        if normalized.path:
            data_path = to_absolute_path(normalized.path)
            data_files = f"{data_path}/**/*.parquet"
            ds = load_dataset("parquet", data_files=data_files, split=normalized.split)
        else:
            ds = load_dataset(normalized.name, split=normalized.split)

        if normalized.max_samples is not None:
            ds = ds.select(range(min(normalized.max_samples, len(ds))))

        datasets.append(ds)
        source_metadata.append(normalized)
        identifier = normalized.name or normalized.path
        print(f"Loaded {identifier}: {len(ds)} samples (weight: {normalized.weight})")

    weights = [source.weight for source in source_metadata]
    total_weight = sum(weights)
    if total_weight <= 0:
        raise ValueError("Sum of data source weights must be positive.")
    probabilities = [weight / total_weight for weight in weights]

    strategy = cfg.data_mixture.mixing_strategy
    if strategy == "concatenate":
        mixed = concatenate_datasets(datasets)
    elif strategy == "interleave":
        mixed = interleave_datasets(datasets, probabilities=probabilities, seed=cfg.data_mixture.seed)
    elif strategy == "sample_proportional":
        mixed = interleave_datasets(
            datasets,
            probabilities=probabilities,
            seed=cfg.data_mixture.seed,
            stopping_strategy="all_exhausted",
        )
    else:
        raise ValueError(f"Unknown mixing strategy: {strategy}")

    return mixed


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    
    wandb.init(
        project=cfg.wandb.project,
        name=cfg.wandb.name,
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    
    print(f"Loading model: {cfg.model.name}")
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=cfg.model.quantization.load_in_4bit,
        bnb_4bit_compute_dtype=getattr(torch, cfg.model.quantization.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=cfg.model.quantization.bnb_4bit_use_double_quant,
        bnb_4bit_quant_type=cfg.model.quantization.bnb_4bit_quant_type
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    if torch.cuda.is_available():
        print(f"Model loaded on CUDA device(s): {model.device}")
        torch.cuda.reset_peak_memory_stats()
    else:
        print("Model loaded on CPU")
    
    peft_config = LoraConfig(
        r=cfg.model.lora.r,
        lora_alpha=cfg.model.lora.lora_alpha,
        lora_dropout=cfg.model.lora.lora_dropout,
        bias=cfg.model.lora.bias,
        task_type=cfg.model.lora.task_type,
        target_modules=cfg.model.lora.target_modules,
    )

    dataset = load_and_mix_datasets(cfg)
    print(f"Final mixed dataset size: {len(dataset)}")

    prompt_template: str = cfg.preprocessing.prompt_template
    template_fields = _extract_template_fields(prompt_template)
    if not template_fields:
        raise ValueError("preprocessing.prompt_template must reference at least one column.")

    def formatting_func(example):
        missing = [field for field in template_fields if field not in example]
        if missing:
            raise KeyError(
                f"Example missing required field(s) {missing}. "
                "Update preprocessing.prompt_template or ensure the dataset provides the column."
            )
        values = {field: example[field] for field in template_fields}
        return prompt_template.format(**values)
    
    training_config = OmegaConf.to_container(cfg.training, resolve=True)
    training_args = TrainingArguments(**training_config)
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        formatting_func=formatting_func,
    )
    
    print("Starting training...")
    trainer.train()
    
    if torch.cuda.is_available():
        peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"Peak GPU memory: {peak_mem_gb:.2f} GB")
    
    print(f"Saving model to {cfg.training.output_dir}")
    trainer.save_model(cfg.training.output_dir)
    
    print("Training complete!")
    wandb.finish()


if __name__ == "__main__":
    main()
