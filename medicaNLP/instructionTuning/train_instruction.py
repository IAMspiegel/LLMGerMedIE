from typing import Optional, List, Tuple

from datasets import load_dataset, Dataset

import torch
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, PeftModel
from trl import SFTTrainer


def format_instruction(sample):
    return f"""### Instruction:\n{sample['instruction']}\n\n### Input:\n{sample['input']}\n\n### Response:\n{sample['response']}"""


def mistral_instruct_format(sample):
    return f"""<s>[INST] {sample['instruction']}\n{sample['input']}[/INST]</s> \n{sample['response']}"""


def germed_format(sample):
    return f"{sample['text']}"

def instruction_finetuning(
    model_id: str,
    output_dir: str,
    train_log_dir: str,
    train_data_path: str,
    tokenizer_path: Optional[str] = None
):
    """Method to instruction tune LLM with SFTTrainer"""
    train_data = load_train_data(train_data_path)

    if 'mistral' in model_id:
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    else:
        target_modules = None

    model, peft_cfg = load_peft_model(model_id, target_modules)

    # load tokenizer
    if tokenizer_path:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    use_flash_attention = False

    args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=train_log_dir,
        num_train_epochs=8,
        per_device_train_batch_size=6 if use_flash_attention else 4,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        logging_steps=10,
        save_strategy="epoch",
        learning_rate=2e-4,
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        warmup_ratio=0.01,
        lr_scheduler_type="constant",
        disable_tqdm=True  # disable tqdm since with packing values are in correct
    )

    max_seq_length = 2048  # max sequence length for model and packing of the dataset

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data['train'],
        peft_config=peft_cfg,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        packing=True,
        formatting_func=format_instruction,
        args=args,
    )

    # train
    trainer.train()

    trainer.save_model()

    del model
    torch.cuda.empty_cache()


def load_peft_model(model_id: str, target_modules: Optional[List[str]] = None, use_flash = False) -> Tuple[PeftModel, LoraConfig]:

    # config for quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    #use_flash_attention = False

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        use_cache=False,
        use_flash_attention_2=use_flash,
        device_map="auto"
    )

    model.config.pretraining_tp = 1

    # LoRA config based on QLoRA paper
    peft_cfg = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']
    )

    # prepare model for training
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_cfg)

    return model, peft_cfg


def load_train_data(data_path: str,) -> Dataset:
    return load_dataset('json', data_files={'train': data_path})


if __name__ == "__main__":
    instruction_finetuning(
        model_id="path/to/base/Model",
        train_log_dir='path/to/log',
        output_dir='/path/to/output/',
        train_data_path="/path/to/instruction_train.json"
    )
