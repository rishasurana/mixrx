# mixrx
MixRx uses a Large Language Model (LLM) to analyze drug interactions and recommend an Additive, Synergistic, or Antagonistic drug, given a multi-drug patient history.

# Preprocessing
1. Run generate_validation_data.py
- drug_combo_dict.json contains all the existing pairs of drug synergy calculations that exist
- drug_data contains the raw data pulled from synergxdb
- valid_drug_combinations.csv contains random possible drug combinations where every pair of drugs have synergy data
2. Run generate_model_data.py (for the model's prompts)
- output_prompts_contexts turns the input data for the model into a string based prompt
3. Run synergy.ipynb
- validate the model's output via the automated synergy calculator