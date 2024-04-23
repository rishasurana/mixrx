# -*- coding: utf-8 -*-
"""baseline_v1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1zXKn33f5PmO3tWu3CIqgDRlq0IRd268d
"""

# !pip install datasets
# !pip install transformers[torch]

from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel

import torch
import pandas as pd
import json
import datasets

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained('gpt2')

text = "Replace me by any text you'd like.", "Hello"
encoded_input = tokenizer(text, return_tensors='pt', padding=True, padding_side='left')

# output = model(**encoded_input)

input_ids = encoded_input['input_ids']

# Generate text response
max_length = 50  # Maximum length of the generated response
temperature = 0.7  # Temperature parameter for sampling
top_k = 50  # Top-k sampling parameter
top_p = 0.9  # Top-p sampling parameter

# Generate text using the model
output = model.generate(
    input_ids,
    max_length=max_length,
    temperature=temperature,
    top_k=top_k,
    top_p=top_p,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    do_sample=True,
    num_return_sequences=1
)

# Decode the generated token IDs into text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print("Generated text:", generated_text)

from datasets import load_dataset, DatasetDict, load_from_disk, Dataset

# Create prompt tensors from prompts_contexts.csv file.

# Read the CSV file into a pandas DataFrame
prompts_contexts = 'final.csv'
# prompts_contexts = '/content/drive/MyDrive/CSCI499 Natural Language Processing/preprocessing/updated_prompts_contexts.csv'
df = pd.read_csv(prompts_contexts)
dataset = Dataset.from_pandas(df)

def create_mult_seq(example):
    sentences = []
    # Concatenate sentences from all fields
    for column, value in example.items():
        sentences.extend(value.split('.'))  # Split sentences based on '. ' delimiter

    example['Sequences'] = sentences
    tokens = tokenizer(example['Sequences'])
    example['input_ids'] = tokens['input_ids']
    # examples['mask???'] = tokens['mask???']
    return example

def tokenize(example):
  example = tokenizer(example['Prompt'], example['Context'], example['Updated Context'], truncation=True, padding=True, return_tensors="pt")
  return example

tokenized_dataset = dataset.map(tokenize, batched=True)

tokenized_dataset = tokenized_dataset.remove_columns(['Prompt', 'Context', 'Updated Context', 'labels'])
print(tokenized_dataset)

# Split the dataset into train/test/valid (80/10/10)
train_test_ds = tokenized_dataset.train_test_split(test_size=.2)
test_valid_ds = train_test_ds['test'].train_test_split(test_size=.5)

ds = DatasetDict({
    'train': train_test_ds['train'],
    'test': test_valid_ds['test'],
    'valid': test_valid_ds['train']})

from transformers import DataCollatorWithPadding
from transformers import TrainingArguments
from torch.utils.data import DataLoader

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

train_dataloader = DataLoader(
    ds["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    ds["valid"], batch_size=8, collate_fn=data_collator
)


# Generate text response
max_length = 600  # Maximum length of the generated response
temperature = 0.7  # Temperature parameter for sampling
top_k = 50  # Top-k sampling parameter
top_p = 0.9  # Top-p sampling parameter

num_epochs = 1
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v for k, v in batch.items()}

        # Generate text using the model
        output = model.generate(
            # input_ids,
            **batch,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            num_return_sequences=1
        )

        # Decode the generated token IDs into text
        generated_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        [print("Generated text:", text) for text in generated_text]
    break