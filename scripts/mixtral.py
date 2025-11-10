"""
Simple SFT training script for Mixtral 8x7B
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from trl import SFTTrainer
from datasets import load_dataset
from peft import LoraConfig

import wandb
wandb.init(project="finetuning", name="mixtral-8x7b-sft-test-1")

model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
output_dir = "./models/mixtral-8x7b-sft"

print(f"Loading model: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=quantization_config,
    trust_remote_code=True,
)

print(f"Model loaded on: {model.device}")
print(f"Model memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

dataset = load_dataset("timdettmers/openassistant-guanaco", split="train[:1000]")
print(f"Dataset size: {len(dataset)}")

def formatting_func(example):
    return example["text"]

# Training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    fp16=False,
    bf16=True,
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
    warmup_steps=50,
    optim="paged_adamw_8bit",
    report_to="wandb",
    max_steps=100,
    gradient_checkpointing=True,
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
print(f"Saving model to {output_dir}")
trainer.save_model(output_dir)
print("Training complete!")
wandb.finish()