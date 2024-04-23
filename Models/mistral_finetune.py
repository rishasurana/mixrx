from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import Dataset
import pandas as pd
import torch
import os

print(torch.cuda.is_available())
print(torch.version.cuda)

torch.cuda.empty_cache()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

class CSVTextDataset(Dataset):
    def __init__(self, tokenizer, file_path, block_size=128):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.file_path = file_path

    def __len__(self):
        return sum(1 for line in open(self.file_path)) - 1

    def __getitem__(self, idx):
        line = pd.read_csv(self.file_path, skiprows=idx + 1, nrows=1)
        text = line['Prompt'].item() + " " + line['Prediction'].item()
        tokenized_text = self.tokenizer(text, add_special_tokens=True, truncation=True, max_length=self.block_size, return_tensors="pt")
        return tokenized_text['input_ids'].squeeze(0)

def fine_tune_mistral(model_name, train_file, output_dir):
    print("begin fine-tuning")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Freeze all the parameters except for the lm_head
    for name, param in model.named_parameters():
        if 'lm_head' not in name:
            param.requires_grad = False

    train_dataset = CSVTextDataset(tokenizer=tokenizer, file_path=train_file, block_size=128)
    print("loaded dataset")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        save_steps=10_000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    trainer.train()
    print("done training")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

fine_tune_mistral("mistralai/Mistral-7B-v0.1", "final.csv", "output")
