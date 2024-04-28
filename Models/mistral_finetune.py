import pandas as pd
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, 
    DataCollatorForLanguageModeling, BitsAndBytesConfig
)
from torch.utils.data import Dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import os

class CSVTextDataset(Dataset):
    def __init__(self, tokenizer, file_path, block_size, start_index, num_examples):
        self.tokenizer = tokenizer
        self.block_size = block_size
        data = pd.read_csv(file_path)
        if start_index + num_examples > len(data):
            num_examples = len(data) - start_index
        self.data = data[start_index:start_index+num_examples]
        combined_texts = "<PROMPT> " + self.data['Prompt'] + " <PREDICTION> " + self.data['Prediction']
        if not combined_texts.empty:
            self.examples = self.tokenizer(combined_texts.tolist(), add_special_tokens=True, 
                                           truncation=True, padding="max_length", 
                                           max_length=block_size, return_tensors="pt").input_ids
        else:
            self.examples = torch.tensor([], dtype=torch.long)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def main():
    base_model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    train_file = "final_reduced.csv"
    total_rows = 9539
    test_start_index = 0
    test_num_examples = 1000
    remaining_start_index = test_num_examples
    remaining_num_examples = total_rows - test_num_examples
    train_num_examples = int(0.9 * remaining_num_examples)
    eval_num_examples = remaining_num_examples - train_num_examples
    train_start_index = remaining_start_index
    eval_start_index = train_start_index + train_num_examples

    # Set tokenizer parallelism environment variable
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["WANDB_DISABLED"] = "true"


    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True, 
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config)
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8, lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],
        bias="none", lora_dropout=0.05, task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    train_dataset = CSVTextDataset(tokenizer, train_file, 128, train_start_index, train_num_examples)
    eval_dataset = CSVTextDataset(tokenizer, train_file, 128, eval_start_index, eval_num_examples)
    test_dataset = CSVTextDataset(tokenizer, train_file, 128, test_start_index, test_num_examples)

    training_args = TrainingArguments(
        output_dir="./output",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        do_eval=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to=None  # Remove W&B integration
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    
    trainer.train()

    model.eval()
    outputs = []
    for idx, input_ids in enumerate(test_dataset):
        input_ids = input_ids.unsqueeze(0).to(trainer.args.device)
        generated_ids = model.generate(input_ids, max_length=128)
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        outputs.append({'Prompt': test_dataset.data['Prompt'].iloc[idx], 'Generated Text': generated_text})

    pd.DataFrame(outputs).to_csv('test_outputs.csv', index=False)
    metrics = trainer.state.log_history
    pd.DataFrame(metrics).to_csv('training_stats.csv', index=False)

    print("Test outputs and training statistics have been saved.")

if __name__ == "__main__":
    main()
