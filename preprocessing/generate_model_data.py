import csv
import pandas as pd

# Step 1: Read drug data and map IDs to names and scores
drug_scores = {}
with open('preprocessing/drug_data.csv', mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        pair = tuple(sorted((int(row['idDrugA']), int(row['idDrugB']))))
        drug_scores[pair] = {
            'nameA': row['drugNameA'], 'nameB': row['drugNameB'],
            'loewe': row['loewe'], 'hsa': row['hsa'], 'zip': row['zip']
        }

# Step 2: Read drug combinations
combinations = []
with open('preprocessing/valid_drug_combinations.csv', mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        combinations.append([int(id.strip()) for id in row['Drug Combination'].split(',')])

# Step 3: Generate context for each combination including scores
contexts = []
for combo in combinations:
    # Extract drug interactions and scores for the given combination
    drug_names_list = [
        drug_scores[tuple(sorted((combo[i], combo[j])))]
        for i in range(len(combo))
        for j in range(i + 1, len(combo))
        if tuple(sorted((combo[i], combo[j]))) in drug_scores
    ]
    
    # Prepare the prompt and example text for response formatting
    prompt = 'According to the rule, if Loewe > 0.1, the outcome is Antagonistic; if Loewe < -0.1, it is Synergistic; otherwise, it is Additive. Decide if the combination is synergistic, antagonistic, or additive.'# Then provide reasoning by analyzing each pairwise interaction.'
    #example = 'Format your response as a JSON object with the following keys (for example): [{"Prediction": "Synergistic", "Reasoning": "DrugA and DrugB are highly synergistic, DrugB and DrugC are additive, DrugC and DrugA are synergistic"}]'
    
    # List the drugs involved in the current combination
    drugs = 'The drug combination to analyze is: ' + ', '.join(set([info['nameA'] for info in drug_names_list] + [info['nameB'] for info in drug_names_list]))
    
    # Generate a context string detailing interactions and scores
    context = 'Context: ' +  ' '.join([
        f'{info['nameA']} and {info['nameB']} have a Loewe score of: {info['loewe']}.'
        for info in drug_names_list
    ])
    
    # Append both the detailed prompt and the context to the list
    # contexts.append([prompt + " " + example + " " + drugs + " " + context, context])
    contexts.append([context + " " + " " + drugs + " " + prompt, context])

# Step 4: Write the prompts and contexts to a new CSV
with open('preprocessing/prompts.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Prompt', 'Context'])
    for prompt, context in contexts:
        writer.writerow([prompt, context])


# Step 5: Combine prompts with model_inputs
model_inputs = pd.read_csv('preprocessing/model_inputs.csv') 
df2 = pd.read_csv('preprocessing/prompts.csv')

model_inputs['Prompt'] = df2['Prompt']
model_inputs['Context'] = df2['Context']

# Save the updated DataFrame to a new CSV file
model_inputs.to_csv('final_reduced.csv', index=False)
