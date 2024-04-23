import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# Load the Mistral Instruct model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

def generate_response(prompt):
    # Tokenize the prompt with automatic padding and truncation
    inputs = tokenizer(prompt, return_tensors='pt', max_length=1024, truncation=True, padding="max_length")
    
    # Generate response, ensuring not to exceed the model's maximum length
    outputs = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        pad_token_id=tokenizer.pad_token_id,
        max_length=1024,  # Ensuring it doesn't exceed the model's maximum length
        num_return_sequences=1
    )
    # Decode the output tokens to a string
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    # Read the CSV file
    data = pd.read_csv('final.csv')
    
    # Process only the first 1000 prompts
    limited_data = data.head(1000)

    # Initialize progress bar and generate responses for each prompt
    tqdm.pandas(desc="Generating responses")
    limited_data['Model Response'] = limited_data['Prompt'].progress_apply(generate_response)
    
    # Save the responses to a new CSV file
    limited_data[['Prompt', 'Model Response']].to_csv('outputs.csv', index=False)
    print("Model responses have been saved to 'outputs.csv'.")

if __name__ == "__main__":
    main()
