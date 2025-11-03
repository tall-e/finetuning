"""
Hydra-based training script for flexible experiments
"""
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer,AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from trl import SFTTrainer
from datasets import load_dataset
from peft import LoraConfig
import wandb


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
    
    dataset = load_dataset(cfg.dataset.name, split=cfg.dataset.split)
    print(f"Dataset size: {len(dataset)}")
    
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