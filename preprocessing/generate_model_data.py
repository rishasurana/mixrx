import csv

# Step 1: Read drug data and map IDs to names and scores
drug_scores = {}
with open('drug_data.csv', mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        pair = tuple(sorted((int(row['idDrugA']), int(row['idDrugB']))))
        drug_scores[pair] = {
            'nameA': row['drugNameA'], 'nameB': row['drugNameB'],
            'loewe': row['loewe'], 'hsa': row['hsa'], 'zip': row['zip']
        }

# Step 2: Read drug combinations
combinations = []
with open('valid_drug_combinations.csv', mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        combinations.append([int(id.strip()) for id in row['Drug Combination'].split(',')])

# Step 3: Generate context for each combination including scores
contexts = []
for combo in combinations:
    drug_names_list = [drug_scores[tuple(sorted((combo[i], combo[j])))] for i in range(len(combo)) for j in range(i + 1, len(combo)) if tuple(sorted((combo[i], combo[j]))) in drug_scores]
    prompt = "Given the following set of drugs, decide if the synergy of the drug combination is synergistic, antagonistic, or additive in one word in square brackets. Then provide reasoning by analyzing the pairwise interactions of each. The drugs are: " + ", ".join(set([info['nameA'] for info in drug_names_list] + [info['nameB'] for info in drug_names_list]))

    context = " ".join([f"{info['nameA']} and {info['nameB']} have a Loewe score of: {info['loewe']}, HSA score of: {info['hsa']}, and ZIP score of: {info['zip']}." for info in drug_names_list])
    
    contexts.append((prompt, context))

# Step 4: Write the prompts and contexts to a new CSV
with open('output_prompts_contexts.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Prompt', 'Context'])
    for prompt, context in contexts:
        writer.writerow([prompt, context])
