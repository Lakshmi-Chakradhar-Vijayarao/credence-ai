import os
import torch
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

def train_dpo():
    model_name = "microsoft/phi-2"
    dataset_path = "data/epistemic_compression_training.json"
    output_dir = "./credence-dpo-final"
    
    # Load the 5,000-triple dataset
    dataset = load_dataset("json", data_files=dataset_path, field="examples")
    
    def transform_to_dpo(example):
        return {
            "prompt": f"User: Summarize faithfully: {example['input_conversation']}\\nAssistant:", 
            "chosen": example["faithful_summary"], 
            "rejected": example["unfaithful_summary"]
        }
    
    dpo_dataset = dataset["train"].map(transform_to_dpo)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config.pad_token_id = tokenizer.eos_token_id
    
    print(f"Loading {model_name} in STEEL-PLATED Mode (Full Precision)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        config=config, 
        torch_dtype=torch.float32, 
        device_map="auto", 
        trust_remote_code=True
    )
    
    peft_config = LoraConfig(
        r=4,
        lora_alpha=8, 
        target_modules=["q_proj", "v_proj"], 
        lora_dropout=0.05, 
        bias="none", 
        task_type="CAUSAL_LM"
    )
    
    training_args = DPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=1e-7, 
        num_train_epochs=1,
        beta=0.01, 
        fp16=False,
        max_grad_norm=0.3, 
        report_to="none",
        remove_unused_columns=False,
        logging_steps=5
    )
    
    trainer = DPOTrainer(
        model, 
        ref_model=None, 
        args=training_args, 
        train_dataset=dpo_dataset, 
        processing_class=tokenizer, 
        peft_config=peft_config
    )
    
    print(f"Starting Steel-Plated Run (fp32)...")
    trainer.train()
    trainer.save_model(output_dir)

if __name__ == '__main__':
    train_dpo()
