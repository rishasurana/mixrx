import pandas as pd
import os
from collections import defaultdict
import json
import random
import csv

# Combine all the drug datasets into file_path = 'preprocessing/drug_data.csv'
directory = 'preprocessing/db_data/'

# Initialize an empty DataFrame to hold the combined data
combined_df = pd.DataFrame()

# Loop through each file in the directory
for filename in os.listdir(directory):
    if filename.endswith('.csv'):  # Check if the file is a CSV
        # Construct the full file path
        file_path = os.path.join(directory, filename)
        # Read the CSV file and append it to the combined DataFrame
        df = pd.read_csv(file_path)
        combined_df = pd.concat([combined_df, df], ignore_index=True)

# Save the combined DataFrame to a new CSV file
combined_df.to_csv('preprocessing/drug_data.csv', index=False)

print("All CSV files have been combined into preprocessing/drug_data.csv")

filename = 'preprocessing/drug_data.csv'

# Initialize a defaultdict with set to collect unique combinations
drug_combinations = defaultdict(set)

# Read the CSV file
with open(filename, mode='r') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        idDrugA = row['idDrugA']
        idDrugB = row['idDrugB']
        
        # Add each drug to the other's list of combinations
        drug_combinations[idDrugA].add(idDrugB)
        drug_combinations[idDrugB].add(idDrugA)

# Convert sets to lists for the final output
drug_combinations = {drug_id: list(combinations) for drug_id, combinations in drug_combinations.items()}

# Specify the file path
file_path = 'preprocessing/drug_combo_dict.json'

# Save the dictionary to a JSON file
with open(file_path, 'w') as json_file:
    json.dump(drug_combinations, json_file, indent=4)

print(f"Dictionary saved to {file_path}")

# Function to load drug combinations from a JSON file
def load_drug_combinations(json_file_path):
    with open(json_file_path, 'r') as file:
        drug_combinations = json.load(file)
    return drug_combinations

# Helper function to check if a new drug can form valid combinations with all drugs in a group
def is_valid_combination(group, new_drug, drug_combinations_dict):
    for drug in group:
        if new_drug not in drug_combinations_dict[drug] or drug not in drug_combinations_dict[new_drug]:
            return False
    return True

# Function to generate valid random drug combinations based on the JSON data
def generate_valid_drug_combinations(drug_combinations_dict, num_combinations=50000, combo_range=(2, 5)):
    valid_combinations = []
    drug_ids = list(drug_combinations_dict.keys())
    
    for _ in range(num_combinations):
        combo_length = random.randint(*combo_range)
        combination = []
        
        while len(combination) < combo_length:
            potential_drugs = [drug for drug in drug_ids if drug not in combination and is_valid_combination(combination, drug, drug_combinations_dict)]
            
            if potential_drugs:  # If there are valid drugs that can be added
                combination.append(random.choice(potential_drugs))
            else:  # If no valid drug can be added, break the loop
                break
        
        if len(combination) == combo_length:  # Ensure the combination meets the desired length
            valid_combinations.append(combination)
    
    return valid_combinations

# Load the drug combinations from the JSON file
json_path = "preprocessing/drug_combo_dict.json"
drug_combinations_dict = load_drug_combinations(json_path)

# Generate valid random combinations
valid_combinations = generate_valid_drug_combinations(drug_combinations_dict)

# Save the valid combinations to a CSV file
csv_file_path = "preprocessing/valid_drug_combinations.csv"
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Combination Number", "Drug Combination"])
    for i, combo in enumerate(valid_combinations, 1):
        writer.writerow([i, ', '.join(combo)])

print(f"CSV file of valid drug combinations created at: {csv_file_path}")



