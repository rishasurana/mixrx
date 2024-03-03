import pandas as pd
import json

# Load data
drug_data_path = '../preprocessing/drug_data.csv' 
drug_data_df = pd.read_csv(drug_data_path)

validation_data_path = '../preprocessing/valid_drug_combinations.csv' 
validation_data_df = pd.read_csv(validation_data_path)

synergy_results_path = '../preprocessing/synergy_test_results.csv'
synergy_results_df = pd.read_csv(synergy_results_path)

# Mapping drug names to IDs
drug_name_to_id = {row['drugNameA']: row['idDrugA'] for _, row in drug_data_df.iterrows()}
drug_name_to_id.update({row['drugNameB']: row['idDrugB'] for _, row in drug_data_df.iterrows()})

def extract_drug_names(prompt):
    start_phrase = "Given the following set of drugs, decide if the synergy of the drug combination is synergistic or antagonistic: "
    end_phrase = " have a Loewe score of:"
    start_index = prompt.find(start_phrase) + len(start_phrase)
    end_index = prompt.find(end_phrase)
    
    drug_names_str = prompt[start_index:end_index].strip()
    drug_names_list = [name.strip() for name in drug_names_str.split(",")]
    
    return drug_names_list

def extract_drug_ids_from_prompt(prompt):
    drug_names = extract_drug_names(prompt)
    return [drug_name_to_id[name] for name in drug_names if name in drug_name_to_id]

def find_combination_response(drug_ids, synergy_results_df):
    drug_ids_str = ", ".join(map(str, sorted(drug_ids)))
    row = synergy_results_df[synergy_results_df['Drug Combination'] == drug_ids_str]
    if not row.empty:
        return row.iloc[0]['Overall Synergy Response']
    return "Unknown"

model_predictions_path = './model_responses.json'
model_predictions = json.load(open(model_predictions_path))

true_positives = 0
false_positives = 0
false_negatives = 0

for prediction in model_predictions:
    drug_ids = extract_drug_ids_from_prompt(prediction["prompt"])
    actual_response = find_combination_response(drug_ids, synergy_results_df)
    
    response_str = prediction["response"].strip('`')
    if response_str.startswith('json'):
        response_str = response_str[5:]
    
    try:
        model_response = json.loads(response_str)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        continue

    model_prediction = model_response[0]["Prediction"]

    if actual_response.lower() == "unknown": 
        print(f"Skipping unknown combination: {drug_ids}")
        continue

    if actual_response.lower() == model_prediction.lower():
        true_positives += 1
    else:
        if actual_response.lower() == "synergistic":
            false_negatives += 1
        if model_prediction.lower() == "synergistic":
            false_positives += 1

precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1_score:.2f}")
