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
            tokenized_data = self.tokenizer(combined_texts.tolist(), add_special_tokens=True, 
                                            truncation=True, padding="max_length", 
                                            max_length=block_size, return_tensors="pt", 
                                            return_attention_mask=True)
            self.examples = tokenized_data.input_ids
            self.attention_masks = tokenized_data.attention_mask
        else:
            self.examples = torch.tensor([], dtype=torch.long)
            self.attention_masks = torch.tensor([], dtype=torch.long)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {
            "input_ids": self.examples[idx],
            "attention_mask": self.attention_masks[idx]
        }

def main():
    base_model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    checkpoint_path = "output/checkpoint-5766"
    train_file = "final_reduced.csv"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["WANDB_DISABLED"] = "true"

    # Load tokenizer and ensure pad_token is set correctly
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            # Add a new pad token and set it if eos_token is not suitable or missing
            pad_token = '[PAD]'
            tokenizer.add_special_tokens({'pad_token': pad_token})
            tokenizer.pad_token = pad_token

    # Load model and ensure pad_token_id is set correctly
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],
        bias="none", lora_dropout=0.05, task_type="CAUSAL_LM"
    ))

    # Create dataset instances
    test_dataset = CSVTextDataset(tokenizer, train_file, 128, 0, 1000)

    # Prepare for evaluation or further steps
    model.eval()
    outputs = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for sample in test_dataset:
        input_ids = sample["input_ids"].unsqueeze(0).to(device)
        attention_mask = sample["attention_mask"].unsqueeze(0).to(device)
        generated_ids = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=150)
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        outputs.append(generated_text)

    pd.DataFrame(outputs, columns=['Generated Text']).to_csv('test_outputs.csv', index=False)
    print("Generated texts have been saved.")

if __name__ == "__main__":
    main()
