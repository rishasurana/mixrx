import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# Check if CUDA is available and print CUDA version
print(torch.cuda.is_available())
print(torch.version.cuda)

torch.cuda.empty_cache()  # Clearing CUDA cache to free up unused memory

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

print("instruct mistral")

model_name = "mistralai/Mistral-7B-Instruct-v0.2"

# Setting up the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Load the model in half-precision (float16)
try:
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16)
    model.to(device)
except RuntimeError as e:
    print(f"Running on CPU due to CUDA memory error: {e}")
    device = torch.device("cpu")  # Explicitly setting device to CPU
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)

print('Model added to device: ', device)  # Indicate which device is being used

# Set tokenizer padding token and side
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Load your data
df_full = pd.read_csv("final_reduced.csv")
print('Data loaded')

df = df_full.head(1000)

def generate_text(prompt):
    print(f"Using device: {device}")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded_output

# Prepare to collect results
results = []

# Generate text for each row and column
for index, row in df.iterrows():
    prompt = row['Prompt']
    output = generate_text(prompt)
    results.append(output)

# Save results to CSV
print('Saving results')
results_df = pd.DataFrame(results)
results_df.to_csv("outputs_mistral_pretrained.csv", index=False)
print('Results saved')
