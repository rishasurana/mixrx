import requests
import csv
from collections import defaultdict
import json
import random


def fetch_drug_data(n):
    """
    Fetches a specified number of drugs (name, idDrug, and idPubChem) from the SynergX database.

    Parameters:
    - n (int): The number of drugs to fetch.

    Returns:
    - list of dicts: A list where each item is a dict with 'name', 'idDrug', and 'idPubChem' keys.
    """
    BASE_URL = "https://www.synergxdb.ca/api/combos?"
    drugs = []
    page = 1
    per_page = 500
    dataset = 4

    while len(drugs) < n:
        prompt = f"{BASE_URL}dataset={dataset}&page={page}&perPage={per_page}"
        print(prompt)

        response = requests.get(prompt)

        if response.status_code == 200:
            data = response.json()
            for drug in data:
                drugs.append({
                    'drugNameA': drug['drugNameA'],
                    'drugNameB': drug['drugNameB'],
                    'idDrugA': drug['idDrugA'],
                    'idDrugB': drug['idDrugB'],
                    'comboId': drug['comboId'],
                    'bliss': drug['bliss'],
                    'loewe': drug['loewe'],
                    'hsa': drug['hsa'],
                    'zip': drug['zip']
                })
                if len(drugs) == n:
                    break
            page += 1
        else:
            print(f"Failed to fetch data: {response.status_code}")
            break

    return drugs[:n]

def save_drugs_to_csv(drugs, filename="drug_data.csv"):
    """
    Saves the fetched drugs to a CSV file.

    Parameters:
    - drugs (list of dicts): The drugs to save, where each drug is a dict with 'name', 'idDrug', and 'idPubChem'.
    - filename (str): The name of the CSV file to save.
    """
    with open(filename, mode='w', newline='') as file:
        fieldnames = ['drugNameA', 'drugNameB', 'idDrugA', 'idDrugB', 'comboId', 'bliss', 'loewe', 'hsa', 'zip']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(drugs)

# Example usage
filename = 'drug_data.csv'
n_drugs = 5778  # Specify the number of drugs to fetch (MIT Dataset)
drug_data = fetch_drug_data(n_drugs)
save_drugs_to_csv(drug_data)
print(f"CSV file with drug data created: '{filename}'")


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
file_path = 'drug_combo_dict.json'

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
def generate_valid_drug_combinations(drug_combinations_dict, num_combinations=10000, combo_range=(2, 5)):
    valid_combinations = []
    drug_ids = list(drug_combinations_dict.keys())
    
    for _ in range(num_combinations):
        combo_length = random.randint(*combo_range)
        combination = []
        
        while len(combination) < combo_length:
            if not combination:  # If combination is empty, start with a random drug
                combination.append(random.choice(drug_ids))
            else:
                # Attempt to add a new drug that forms valid combinations with all in the group
                potential_drugs = [drug for drug in drug_ids if is_valid_combination(combination, drug, drug_combinations_dict)]
                if not potential_drugs:  # If no valid drug can be added, break the loop
                    break
                combination.append(random.choice(potential_drugs))
        
        if len(combination) == combo_length:  # Ensure the combination meets the desired length
            valid_combinations.append(combination)
    
    return valid_combinations

# Load the drug combinations from the JSON file
json_path = "drug_combo_dict.json"
drug_combinations_dict = load_drug_combinations(json_path)

# Generate valid random combinations
valid_combinations = generate_valid_drug_combinations(drug_combinations_dict)

# Save the valid combinations to a CSV file
csv_file_path = "valid_drug_combinations.csv"
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Combination Number", "Drug Combination"])
    for i, combo in enumerate(valid_combinations, 1):
        writer.writerow([i, ', '.join(combo)])

print(f"CSV file of valid drug combinations created at: {csv_file_path}")
