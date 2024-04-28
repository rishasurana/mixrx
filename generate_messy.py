import pandas as pd
import random

def switch_letters(word):
    # Switch two middle letters in the word
    if len(word) > 1:
        mid_index = len(word) // 2
        if len(word) % 2 == 0:
            return word[:mid_index-1] + word[mid_index] + word[mid_index-1] + word[mid_index+1:]
        else:
            return word[:mid_index] + word[mid_index+1] + word[mid_index] + word[mid_index+2:]
    return word

def remove_last_letter(word):
    # Remove the last letter of the word
    return word[:-1] if word else word

def modify_prompt(prompt):
    if "The drug combination to analyze is:" in prompt:
        modification_function = random.choice([switch_letters, remove_last_letter])
        parts = prompt.split("The drug combination to analyze is:")
        before = parts[0]
        after = parts[1]
        drugs_and_rest = after.split(".")
        drugs = drugs_and_rest[0].split(',')
        modified_drugs = [modification_function(drug.strip()) for drug in drugs]
        new_drugs = ', '.join(modified_drugs)
        new_prompt = before + "The drug combination to analyze is: " + new_drugs + "." + '.'.join(drugs_and_rest[1:])
        return new_prompt
    return prompt

def process_csv(file_path):
    df = pd.read_csv(file_path)
    
    # Apply random modification to each row
    df['Prompt'] = df['Prompt'].apply(modify_prompt)
    
    # Save modified DataFrame
    messy_path = file_path.replace('.csv', '_messy.csv')
    df.to_csv(messy_path, index=False)
    print(f"File saved: {messy_path}")

# Example usage
file_path = 'final_reduced.csv'  # Modify this path as needed
process_csv(file_path)
