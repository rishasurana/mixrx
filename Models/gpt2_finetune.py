import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset
from transformers import DataCollatorForLanguageModeling
from accelerate import DataLoaderConfiguration
import os

# Check if CUDA is available and print CUDA version
print(torch.cuda.is_available())
print(torch.version.cuda)

torch.cuda.empty_cache()  # Clearing CUDA cache to free up unused memory

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

dataloader_config = DataLoaderConfiguration(
    dispatch_batches=None, 
    split_batches=False, 
    even_batches=True, 
    use_seedable_sampler=True
)

import torch
from torch.utils.data import Dataset

class CSVTextDataset(Dataset):
    def __init__(self, tokenizer, file_path, block_size=128):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.file_path = file_path  # Store the file path for reference in __getitem__

        # Load the dataset into a pandas dataframe and assign it to self.data
        self.data = pd.read_csv(file_path)

        # Print the columns to verify them
        print("Columns in the CSV:", self.data.columns.tolist())

        # Check if expected columns exist
        if 'Prompt' not in self.data.columns or 'Prediction' not in self.data.columns:
            raise ValueError("CSV file must include 'Prompt' and 'Prediction' columns.")

        # Combine the prompts and predictions into one text string per row
        text = self.data['Prompt'] + " " + self.data['Prediction']
        # Tokenize the text
        self.examples = [self.tokenizer(text.tolist(), add_special_tokens=True, truncation=True, max_length=self.block_size)["input_ids"]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Accessing the row directly from self.data without loading CSV again
        text = self.data.iloc[idx]['Prompt'] + " " + self.data.iloc[idx]['Prediction']
        tokenized_text = self.tokenizer(text, add_special_tokens=True, truncation=True, max_length=self.block_size, return_tensors="pt")
        return tokenized_text['input_ids'].squeeze(0)


def fine_tune_gpt2(model_name, train_file, output_dir):
    print("begin fine-tuning")
    # Load GPT-2 model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_name, torch_dtype=torch.float16)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Load training dataset
    train_dataset = CSVTextDataset(tokenizer=tokenizer, file_path=train_file, block_size=128)
    print("loaded dataset")

    # Create data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Set training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8, 
        save_steps=10_000,
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
    print("done training")

    # Save the fine-tuned model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

# Fine-tune the model
fine_tune_gpt2("gpt2", "final.csv", "output")
