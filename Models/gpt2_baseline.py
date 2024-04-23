# !pip install datasets
# !pip install transformers[torch]

# Set up tokenizer and pretrained model.
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel

import torch
import pandas as pd
import json

# Dataset sample consists of:
# 1. prompt -- drug combo
# 2. context -- for ea pariwise drug, relavent information needed for inference.
# 3. Ground truth
#  a. prediction -- synergy response (one word)
#  b. reasoning -- models explanation for prediction

# Expected model output:
# 1. prediction
# 2. reasoning


tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'
# model = GPT2Model.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

print(model.config)

text = "Replace me by any text you'd like.", "Hello"
encoded_input = tokenizer(text, return_tensors='pt', padding=True)
# tokens = tokenizer.tokenize(text)
# print(encoded_input)
# print(tokenizer.decode(encoded_input["input_ids"][0]))
# ids = tokenizer.convert_tokens_to_ids(tokens)
# print(tokenizer.decode(ids))

# The model accepts embeddings -- i.e. input_ids -- as an input for a single sample.
input_ids = encoded_input['input_ids']

# Generate text response
max_length = 50  # Maximum length of the generated response
temperature = 0.7  # Temperature parameter for sampling
top_k = 50  # Top-k sampling parameter
top_p = 0.9  # Top-p sampling parameter

# Quick model check: Generate text using the model
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
# dataset = load_dataset('glue', 'mrpc', split='train')
# print(dataset[0])

# Create prompt text tensors from prompts_contexts.csv file,
#   and colate all training, dev, and test dataset into a map.

# Model input pipeline:
#     csv -> dataframe -> dataset -> dataloader (batched) -> model (inputs are batches of samples)

# Read the CSV file into a pandas DataFrame
# prompts_contexts = '../preprocessing/updated_prompts_contexts.csv'

#prompts_contexts = '/content/drive/MyDrive/CSCI499 Natural Language Processing/preprocessing/final.csv'

prompts_contexts = 'preprocessing/final.csv'
df = pd.read_csv(prompts_contexts)
df = df.drop(labels=['Drugs', 'Prediction', 'Reasoning', 'Context'], axis=1)   # The context column is, I believe a legacy input included in the csv for posterity.

# Create dataset.
dataset = Dataset.from_pandas(df)
# dataset = load_dataset('csv', data_files=prompts_contexts)   # Alt method of loading dataset.

# print(dataset)

# create_mult_seq() is not used.
def create_mult_seq(example):
    sentences = []
    # Concatenate sentences from all fields
    for column, value in example.items():
        sentences.extend(value.split('.'))  # Split sentences based on '. ' delimiter
    # print(sentences)
    example['Sequences'] = sentences
    tokens = tokenizer(example['Sequences'])
    example['input_ids'] = tokens['input_ids']
    return example

# tokenize() is applied to each sample in the dataset using dataset.map() function.
def tokenize(example):
  # example = tokenizer(example['Prompt'], example['Context'], example['Updated Context'], truncation=True, padding=True, return_tensors="pt")
  # example = tokenizer(example['Prompt'], example['Updated Context'], truncation=True, padding=True, return_tensors="pt")
  example = tokenizer(example['Prompt'], truncation=True, padding=True, return_tensors="pt")   # Used with `final.csv`
  # print(example)
  # assert False
  return example

# create_mult_seq(dataset['train'][0])
# tokenized_dataset = dataset.map(tokenize)
# print(tokenized_dataset)
# print(dataset['train']['input_ids'][0])
# print(tokenizer.convert_ids_to_tokens(dataset['train']["input_ids"][0]))
# assert False
# tokenized_dataset = dataset.map(create_mult_seq)

# Tokenize the dataset.
#   Dataset map method is a way to apply a function to all samples in a dataset -- tokenize in this case.
tokenized_dataset = dataset.map(tokenize, batched=True)
# print(tokenized_dataset)

# Drop columns that are not needed in our context, but returned by the GPT2 tokenizer.
# tokenized_dataset = tokenized_dataset.remove_columns(['Prompt', 'Updated Context'])   # For some reason, 'labels' are not generated when 'Context' column is removed from pandas Dataframe.
tokenized_dataset = tokenized_dataset.remove_columns(['Prompt'])   # Now using final.csv
# tokenized_dataset = tokenized_dataset.remove_columns(['Prompt', 'Context', 'Updated Context', 'labels'])
print(tokenized_dataset)  # Quick inspection of the tokenizer.
# print(seq_dataset)
# print(seq_dataset['train'][0]['Sequences'])
# tokens = seq_dataset.map(tokenize)
# tokens = seq_dataset.map(tokenize)
# print(tokens)
# print(len(tokens['train'][0]['input_ids'][0]))
# print(tokenizer.convert_ids_to_tokens(tokens['train']['input_ids'][0][0]))

# Split the dataset into train/test/valid (80/10/10)
train_test_ds = tokenized_dataset.train_test_split(test_size=.2)
test_valid_ds = train_test_ds['test'].train_test_split(test_size=.5)
# print(train_test_ds)

# Colate the datasets into a single dataset dictionary.
ds = DatasetDict({
    'train': train_test_ds['train'],
    'test': test_valid_ds['test'],
    'valid': test_valid_ds['train']})
# print(ds)

# Generate responses from a pretrained model.
from transformers import DataCollatorWithPadding, AdamW
from transformers import TrainingArguments
from torch.utils.data import DataLoader

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
# training_args = TrainingArguments("test-trainer")

# Create dataloaders for the training and test sets.
batch_size = 1
train_dataloader = DataLoader(
    ds["train"], shuffle=True, batch_size=batch_size, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    ds["valid"], batch_size=batch_size, collate_fn=data_collator
)

# Generate a single text response from the model.
#    Next: For each sample, generate text response and append to a csv file.
max_length = 800  # Maximum length of the generated response
temperature = 0.7  # Temperature parameter for sampling
top_k = 50  # Top-k sampling parameter
top_p = 0.9  # Top-p sampling parameter
num_epochs = 1

# print(f'Generation configuration: {model.generation_config}')

# for epoch in range(num_epochs):
#     for batch in train_dataloader:
#         batch = {k: v for k, v in batch.items()}

#         # Generate text using the model
#         output = model.generate(
#             # input_ids,  # Used with non-batched, single input.
#             **batch,   # Unpacks the input_ids for all inputs in the batch.
#             max_length=max_length,
#             temperature=temperature,
#             top_k=top_k,
#             top_p=top_p,
#             pad_token_id=tokenizer.eos_token_id,
#             eos_token_id=tokenizer.eos_token_id,
#             do_sample=True,
#             num_return_sequences=1
#         )

#         # Decode the generated token IDs into text
#         generated_text = tokenizer.batch_decode(output, skip_special_tokens=True)

#         # [print("Generated text:", text) for text in generated_text]
#         print(generated_text[0])
#         assert False
#     break

# Freeze all layers except for last 4.
for param in model.parameters():
    param.requires_grad = False

for layer in model.transformer.h[-4:]:
    for param in layer.parameters():
        param.requires_grad = True

# Set up AdamW optimizer
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)

print(len(train_dataloader))
print(len(eval_dataloader))

# Training
model.train()
num_epochs = 1
for epoch in range(num_epochs):
    total_loss = 0
    for batch in train_dataloader:
        optimizer.zero_grad()
        inputs = {k: v.to(model.device) for k, v in batch.items() if k != 'labels'}
        outputs = model(**inputs, labels=batch['input_ids'].to(model.device))
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch} Average Loss: {total_loss / len(train_dataloader)}")

# Evaluation
model.eval()
total_eval_loss = 0
for batch in eval_dataloader:
    inputs = {k: v.to(model.device) for k, v in batch.items() if k != 'labels'}
    with torch.no_grad():
        outputs = model(**inputs, labels=batch['input_ids'].to(model.device))
        loss = outputs.loss
    total_eval_loss += loss.item()

print(f"Validation Loss: {total_eval_loss / len(eval_dataloader)}")