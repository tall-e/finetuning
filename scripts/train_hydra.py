"""
Hydra-based training script for flexible experiments
"""
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer,AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from trl import SFTTrainer
from datasets import load_dataset, concatenate_datasets, interleave_datasets, Dataset
from peft import LoraConfig
import wandb
import random
from pathlib import Path


def load_and_mix_datasets(cfg: DictConfig):
    """Load multiple datasets and mix them according to config"""
    random.seed(cfg.data_mixture.seed)

    datasets = []
    for source in cfg.data_mixture.sources:
        # Load dataset from HuggingFace or local path
        if "path" in source:
            ds = load_dataset("parquet", data_files=f"{source.path}/**/*.parquet", split=source.split)
        else:
            ds = load_dataset(source.name, split=source.split)

        # Limit samples if specified
        if source.max_samples is not None:
            ds = ds.select(range(min(source.max_samples, len(ds))))

        datasets.append(ds)
        print(f"Loaded {source.name}: {len(ds)} samples (weight: {source.weight})")

    # Mix datasets according to strategy
    strategy = cfg.data_mixture.mixing_strategy
    if strategy == "concatenate":
        mixed = concatenate_datasets(datasets)
    elif strategy == "interleave":
        weights = [s.weight for s in cfg.data_mixture.sources]
        mixed = interleave_datasets(datasets, probabilities=weights, seed=cfg.data_mixture.seed)
    elif strategy == "sample_proportional":
        weights = [s.weight for s in cfg.data_mixture.sources]
        mixed = interleave_datasets(datasets, probabilities=weights, seed=cfg.data_mixture.seed, stopping_strategy="all_exhausted")
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
    
    print(f"Model memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    
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
    
    def formatting_func(example):
        return example["text"]
    
    training_args = TrainingArguments(
        output_dir=cfg.training.output_dir,
        num_train_epochs=cfg.training.num_train_epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        fp16=cfg.training.fp16,
        bf16=cfg.training.bf16,
        logging_steps=cfg.training.logging_steps,
        save_strategy=cfg.training.save_strategy,
        save_steps=cfg.training.save_steps,
        warmup_steps=cfg.training.warmup_steps,
        optim=cfg.training.optim,
        report_to=cfg.training.report_to,
        max_steps=cfg.training.max_steps,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
    )
    
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
    
    print(f"Saving model to {cfg.training.output_dir}")
    trainer.save_model(cfg.training.output_dir)
    
    print("Training complete!")
    wandb.finish()


if __name__ == "__main__":
    main()