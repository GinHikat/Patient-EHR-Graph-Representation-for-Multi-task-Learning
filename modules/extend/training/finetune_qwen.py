import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# --- Configuration ---
# Since you are using an AMD ROCm GPU with massive VRAM (like an MI210 with 64GB), 
# we DO NOT need to use buggy 4-bit quantization (bitsandbytes). 
# We can load the model in native bfloat16, which is incredibly fast and stable on AMD.

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
TRAIN_DATA = "doc_class_train.jsonl"
EVAL_DATA = "doc_class_test.jsonl"
OUTPUT_DIR = "./qwen_finetuned_adapter"

def format_chat_template(example, tokenizer):
    """
    Converts the {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    format into the raw string format using Qwen's official chat template.
    """
    # Assuming your JSONL has a "messages" key. If your JSONL is formatted as a flat list of dicts,
    # adjust the data loading below. Based on your previous snippet:
    # {"messages": [{"role": "user", "content": "text"}, {"role": "assistant", "content": "[]"}]}
    
    # Apply the chat template
    prompt = tokenizer.apply_chat_template(example["messages"], tokenize=False)
    return {"text": prompt}

def main():
    print(f"Loading Tokenizer from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading Datasets...")
    # Load the JSONL files we created in Step 2
    dataset = load_dataset("json", data_files={"train": TRAIN_DATA, "test": EVAL_DATA})
    
    # Apply chat template
    print("Applying Chat Template to dataset...")
    dataset = dataset.map(lambda x: format_chat_template(x, tokenizer), num_proc=4)

    print(f"Loading Model {MODEL_NAME} in bfloat16 (Optimized for AMD ROCm)...")
    # Load directly in bf16 to easily fit in 64GB VRAM and maximize throughput
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Gradient Checkpointing saves VRAM at the cost of slight compute overhead
    model.gradient_checkpointing_enable()

    print("Setting up LoRA (Low-Rank Adaptation)...")
    # Target the linear layers of the Qwen architecture
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Training Arguments specifically tuned for AMD ROCm and fast convergence
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,      # Effective batch size = 16
        optim="adamw_torch",                # Standard optimizer
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        num_train_epochs=3,           
        logging_steps=10,
        eval_strategy="epoch",              # Evaluate at the end of every epoch
        save_strategy="epoch",
        bf16=True,                          # for AMD ROCm
        max_grad_norm=1.0,
        warmup_ratio=0.1,
        report_to="none"                    # Set to "wandb" if use Weights & Biases
    )

    response_template = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer)

    print("Initializing SFT Trainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=512,                 # Clinical sentences are short, 512 is plenty
        tokenizer=tokenizer,
        args=training_args,
        data_collator=collator,
    )

    print("🚀 Starting Fine-Tuning!")
    trainer.train()

    print(f"Training Complete! Saving adapter to {OUTPUT_DIR}...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("Done! You can now merge this adapter or load it via PEFT for inference.")

if __name__ == "__main__":
    main()
