import pandas as pd
import json
from rouge_score import rouge_scorer
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
)
from bert_score import score
import numpy as np
import matplotlib.pyplot as plt

# final_data_path = "../preprocessing/final.csv"
final_data_path = "final_reduced.csv"
output_data_path = "evaluate/outputs-gpt2-finetuned.csv"
final_df = pd.read_csv(final_data_path)
model_responses = pd.read_csv(output_data_path)  # columns: ['Prompt', 'Model Response']

# model_responses = [
#     {
#         "prompt": final_df.head(1)["Prompt"].values[0],
#         "response": '```json\n[\n  {\n    "Prediction": "Antagonistic",\n    "Reasoning": "BML-190 and Temozolomide have negative scores across all interaction metrics (Loewe, HSA, ZIP), indicating antagonism."\n  }\n]\n```',
#     },
#     {
#         "prompt": final_df.head(2)["Prompt"].values[0],
#         "response": '```json\n[\n  {\n    "Prediction": "Antagonistic",\n    "Reasoning": "Carbamazepine and Temozolomide have negative scores across all interaction metrics (Loewe, HSA, ZIP), indicating antagonism."\n  }\n]\n```',
#     },
# ]

def find_prediction(output_text):
    pred1 = 'synergistic'
    pred2 = 'antagonistic'
    pred3 = 'additive'
    no_pred = 'no_pred'

    # Make string all lowercase letter.
    output_text = output_text.lower()

    # Check for synergistic.
    if output_text.find(pred1) != -1: return pred1
    # Check for antagonistic.
    if output_text.find(pred2) != -1: return pred2
    # Check for additive.
    if output_text.find(pred3) != -1: return pred3
    
    return no_pred

def normalize_prompt(prompt):
    return "".join([i for i in prompt if not i.isdigit()]).lower().strip()


def compare_predictions(model_data, ground_truth_df):
    correct_predictions = 0
    total_predictions = len(model_data)

    # for model_entry in model_data:
    for idx, model_entry in model_data.iterrows():
        # model_prompt = normalize_prompt(model_entry["prompt"])
        model_prompt = normalize_prompt(model_entry["Prompt"])
        ground_truth_row = ground_truth_df[
            ground_truth_df["Prompt"].apply(normalize_prompt) == model_prompt
        ]

        if not ground_truth_row.empty:
            # model_response = json.loads(
            #     model_entry["response"].strip("`").strip("json").strip("\n")
            # )
            # model_prediction = model_response[0]["Prediction"]
            model_prediction = find_prediction(model_entry['Model Response'])

            if (
                model_prediction.lower()
                == ground_truth_row["Prediction"].iloc[0].lower()
            ):
                correct_predictions += 1

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    return accuracy


def calculate_rouge_scores(model_data, ground_truth_df):
    """
    ROUGE-N: Measures the overlap of n-grams between the system output and reference texts
    ROUGE-L: Measures the longest matching sequence of words using longest common subsequence (LCS) statistics
    """

    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    scores = []

    for model_entry in model_data:
        model_prompt = normalize_prompt(model_entry["prompt"])
        ground_truth_row = ground_truth_df[
            ground_truth_df["Prompt"].apply(normalize_prompt) == model_prompt
        ]

        if not ground_truth_row.empty:
            model_response = json.loads(model_entry["response"].strip("`json\n"))
            model_reasoning = model_response[0]["Reasoning"]
            ground_truth_reasoning = ground_truth_row["Reasoning"].iloc[0]

            scores.append(scorer.score(ground_truth_reasoning, model_reasoning))

    return scores


def calculate_precision_recall_f1(model_data, ground_truth_df):
    """
    Helps us understand the accuracy of positive predictions (precision),
    the ability of the model to find all positive samples (recall),
    and a balance between precision and recall (F1-score).
    """

    y_true = []
    y_pred = []

    for model_entry in model_data:
        model_prompt = normalize_prompt(model_entry["prompt"])
        ground_truth_row = ground_truth_df[
            ground_truth_df["Prompt"].apply(normalize_prompt) == model_prompt
        ]

        if not ground_truth_row.empty:
            model_response = json.loads(
                model_entry["response"].strip("`").strip("json").strip("\n")
            )
            model_prediction = model_response[0]["Prediction"]
            y_true.append(ground_truth_row["Prediction"].iloc[0].lower())
            y_pred.append(model_prediction.lower())

    precision, recall, fscore, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted"
    )
    return precision, recall, fscore


def plot_confusion_matrix(model_data, ground_truth_df):
    """
    Confusion matrix helps visualize the performance of the model across different categories.
    """
    y_true = []
    y_pred = []

    for model_entry in model_data:
        model_prompt = normalize_prompt(model_entry["prompt"])
        ground_truth_row = ground_truth_df[
            ground_truth_df["Prompt"].apply(normalize_prompt) == model_prompt
        ]

        if not ground_truth_row.empty:
            model_response = json.loads(
                model_entry["response"].strip("`").strip("json").strip("\n")
            )
            model_prediction = model_response[0]["Prediction"]
            y_true.append(ground_truth_row["Prediction"].iloc[0].lower())
            y_pred.append(model_prediction.lower())

    labels = list(set(y_true + y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots()
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    fig.colorbar(cax)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


def calculate_bert_scores(model_data, ground_truth_df):
    """
    BERTScore leverages the pre-trained contextual embeddings from BERT and matches words in candidate
    and reference texts based on cosine similarity.
    """
    cands = []
    refs = []

    for model_entry in model_data:
        model_prompt = normalize_prompt(model_entry["prompt"])
        ground_truth_row = ground_truth_df[
            ground_truth_df["Prompt"].apply(normalize_prompt) == model_prompt
        ]

        if not ground_truth_row.empty:
            model_response = json.loads(
                model_entry["response"].strip("`").strip("json").strip("\n")
            )
            model_reasoning = model_response[0]["Reasoning"]
            ground_truth_reasoning = ground_truth_row["Reasoning"].iloc[0]

            cands.append(model_reasoning)
            refs.append(ground_truth_reasoning)

    # Calculating BERTScore
    P, R, F1 = score(cands, refs, lang="en", rescale_with_baseline=True)

    # Convert to python floats and calculate average
    P = P.tolist()
    R = R.tolist()
    F1 = F1.tolist()
    avg_P = sum(P) / len(P)
    avg_R = sum(R) / len(R)
    avg_F1 = sum(F1) / len(F1)

    return avg_P, avg_R, avg_F1

accuracy = compare_predictions(model_responses, final_df)
rouge_results = calculate_rouge_scores(model_responses, final_df)
precision, recall, fscore = calculate_precision_recall_f1(model_responses, final_df)
average_rouge1 = sum(score["rouge1"].fmeasure for score in rouge_results) / len(
    rouge_results
)
average_rougeL = sum(score["rougeL"].fmeasure for score in rouge_results) / len(
    rouge_results
)
bert_P, bert_R, bert_F1 = calculate_bert_scores(model_responses, final_df)

print(f"Accuracy: {accuracy:.2f}")

# for score in rouge_results:
#     print(
#         f"ROUGE-1: {score['rouge1'].fmeasure:.4f}, ROUGE-L: {score['rougeL'].fmeasure:.4f}"
#     )

print(f"Average ROUGE-1: {average_rouge1:.4f}, Average ROUGE-L: {average_rougeL:.4f}")
print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {fscore:.2f}")
print(
    f"BERTScore Precision: {bert_P:.4f}, Recall: {bert_R:.4f}, F1-Score: {bert_F1:.4f}"
)

plot_confusion_matrix(model_responses, final_df)