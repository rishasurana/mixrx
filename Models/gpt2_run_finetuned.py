import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm

# Path to fine-tuned model's directory
MODEL_DIR = "output/"

# Load the model and tokenizer from the fine-tuned model directory
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)

# Set the tokenizer padding side to left and pad token to eos_token
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

def generate_response(prompt):
    # Ensure prompt does not exceed model's max input length - 50 to allow some room for additional tokens
    max_length_for_prompt = 1024 - 50
    inputs = tokenizer(prompt, return_tensors='pt', max_length=max_length_for_prompt, truncation=True, padding="max_length")

    # Generate response, only allowing up to 50 new tokens
    outputs = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        pad_token_id=tokenizer.pad_token_id,
        max_length=1024,  # Not exceeding the model's maximum length
        num_return_sequences=1,
        no_repeat_ngram_size=2,  # prevent repeating short phrases
        temperature=0.7,  # Lower value (<1.0) makes results more deterministic; higher (>1.0) more diverse
        top_k=50  # Limits the likely next words to the top k candidates

    )

    # Decode only the generated part, excluding the input prompt length
    input_len = inputs['input_ids'].shape[1]  # Length of the input including padding
    response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    return response

def main():
    # Read the CSV file
    data = pd.read_csv('final_reduced.csv')

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
