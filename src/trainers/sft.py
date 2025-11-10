"""Shared runtime for supervised fine-tuning experiments."""
from __future__ import annotations

import random
from typing import Callable, List

import torch
import wandb
from datasets import concatenate_datasets, interleave_datasets, load_dataset
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizerBase,
    TrainingArguments,
)
from trl import SFTTrainer

from src.utils.config import DataSourceConfig, extract_template_fields


def _build_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def _build_quantization_config(cfg: DictConfig) -> BitsAndBytesConfig:
    dtype = getattr(torch, cfg.bnb_4bit_compute_dtype)
    return BitsAndBytesConfig(
        load_in_4bit=cfg.load_in_4bit,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
    )


def _build_peft_config(cfg: DictConfig) -> LoraConfig:
    return LoraConfig(
        r=cfg.r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias=cfg.bias,
        task_type=cfg.task_type,
        target_modules=cfg.target_modules,
    )


def _load_and_mix_datasets(cfg: DictConfig):
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
            limit = min(normalized.max_samples, len(ds))
            ds = ds.select(range(limit))

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

    print(f"Final mixed dataset size: {len(mixed)}")
    return mixed


def _build_prompt_formatter(template: str) -> Callable[[dict], str]:
    template_fields = extract_template_fields(template)
    if not template_fields:
        raise ValueError("preprocessing.prompt_template must reference at least one column.")

    def formatting_func(example: dict) -> str:
        missing = [field for field in template_fields if field not in example]
        if missing:
            raise KeyError(
                f"Example missing required field(s) {missing}. "
                "Update preprocessing.prompt_template or ensure the dataset provides the column."
            )
        values = {field: example[field] for field in template_fields}
        return template.format(**values)

    return formatting_func


def run_training(cfg: DictConfig) -> None:
    """Execute an SFT run based on a composed Hydra config."""

    wandb_run = wandb.init(
        project=cfg.wandb.project,
        name=cfg.wandb.name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    try:
        print(f"Loading model: {cfg.model.name}")
        tokenizer = _build_tokenizer(cfg.model.name)
        quantization_config = _build_quantization_config(cfg.model.quantization)
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

        peft_config = _build_peft_config(cfg.model.lora)
        dataset = _load_and_mix_datasets(cfg)
        formatting_func = _build_prompt_formatter(cfg.preprocessing.prompt_template)

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
    finally:
        if wandb_run is not None:
            wandb.finish()


__all__ = ["run_training"]
