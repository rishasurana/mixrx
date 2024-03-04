import requests
import csv

'''
1. MERCK, 583
2. NCI-ALMANAC, 5354
3. DNE
4. YALE-PDAC, 861
5. YALE_TNBC, 768
6. CLOUD, 1327
7. MIT, 5778
8. Stanford, 1818
9. DECREASE, 36

Total: 16525

'''
n_drugs = 36  # Specify the number of drugs to fetch (MIT Dataset)
dataset = 9

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
filename = f'preprocessing/drug_data_{dataset}.csv'
drug_data = fetch_drug_data(n_drugs)
save_drugs_to_csv(drug_data, filename)
print(f"CSV file with drug data created: '{filename}'")
