import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import Dataset
import os

class CSVTextDataset(Dataset):
    def __init__(self, tokenizer, file_path, block_size=128, limit=None):
        self.tokenizer = tokenizer
        self.block_size = block_size

        # Load the dataset into a pandas dataframe
        self.data = pd.read_csv(file_path)
        
        # If a limit is specified, select only the last 'limit' examples
        if limit:
            self.data = self.data.tail(limit)

        # Combine the prompts and predictions into one text string per row
        combined_texts = self.data['Prompt'] + " " + self.data['Prediction']
        # Tokenize the text
        self.examples = self.tokenizer(combined_texts.tolist(), add_special_tokens=True, truncation=True, max_length=block_size, return_tensors="pt").input_ids

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def fine_tune_gpt2(model_name, train_file, output_dir, num_examples=1000):
    print("Begin fine-tuning")
    # Load GPT-2 model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_name, torch_dtype=torch.float16)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Load training dataset, last 'num_examples' entries
    train_dataset = CSVTextDataset(tokenizer=tokenizer, file_path=train_file, block_size=128, limit=num_examples)
    print("Loaded dataset")

    # Create data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Set training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        save_steps=500,  # Save more frequently
        save_total_limit=2,
    )

    # Train the model
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    trainer.train()
    print("Done training")

    # Explicitly save the model and tokenizer
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model and tokenizer are saved in {output_dir}")

# Fine-tune the model
fine_tune_gpt2("gpt2", "final.csv", "output")
