import pandas as pd
from rouge_score import rouge_scorer
from sklearn.metrics import precision_recall_fscore_support
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

# Load data
final_data_path = "../final_reduced.csv"
output_data_path = "./GPT/outputs-gpt2-finetuned.csv"
final_df = pd.read_csv(final_data_path)
model_responses = pd.read_csv(output_data_path)

min_length = min(len(final_df), len(model_responses))
final_df = final_df.head(min_length)
model_responses = model_responses.head(min_length)


def find_prediction(text):
    """Extracts predictions from the given text, searching after '<PREDICTION>' token."""
    text = text.lower()
    if "antagonistic" in text:
        return "antagonistic"
    elif "synergistic" in text:
        return "synergistic"
    elif "additive" in text:
        return "additive"
    return "no_pred"


def compare_predictions(model_data, ground_truth_df):
    correct_predictions = 0
    y_true = []
    y_pred = []

    for idx, (model_row, gt_row) in enumerate(
        zip(model_data.itertuples(), ground_truth_df.itertuples())
    ):
        prediction = find_prediction(model_row._2)
        actual = find_prediction(gt_row.Prediction)
        y_true.append(actual)
        y_pred.append(prediction)

        if prediction == actual:
            correct_predictions += 1

    accuracy = correct_predictions / len(model_data) if len(model_data) > 0 else 0
    precision, recall, fscore, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    tp = sum(
        1
        for i in range(len(y_true))
        if y_pred[i] == y_true[i] and y_true[i] != "no_pred"
    )
    tn = sum(
        1
        for i in range(len(y_true))
        if y_pred[i] == y_true[i] and y_true[i] == "no_pred"
    )
    fp = sum(
        1
        for i in range(len(y_true))
        if y_pred[i] != y_true[i] and y_pred[i] != "no_pred"
    )
    fn = sum(
        1
        for i in range(len(y_true))
        if y_pred[i] != y_true[i] and y_pred[i] == "no_pred"
    )

    return accuracy, precision, recall, fscore, tp, tn, fp, fn


# Calculate accuracy, precision, recall, and F1-score
accuracy, precision, recall, fscore, tp, tn, fp, fn = compare_predictions(
    model_responses, final_df
)
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {fscore:.2f}")
print(
    f"True positives: {tp}, True negatives: {tn}, False positives: {fp}, False negatives: {fn}"
)

# Calculate ROUGE scores
scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
scores = []
for idx, (model_row, gt_row) in enumerate(
    zip(model_responses.itertuples(), final_df.itertuples())
):
    model_response = find_prediction(model_row._2)
    scores.append(scorer.score(gt_row.Prompt, model_response))
average_rouge1 = sum(score["rouge1"].fmeasure for score in scores) / len(scores)
average_rougeL = sum(score["rougeL"].fmeasure for score in scores) / len(scores)
print(f"Average ROUGE-1: {average_rouge1:.4f}, Average ROUGE-L: {average_rougeL:.4f}")

# Calculate BLEU scores
reference_texts = [[row.Prompt.split()] for row in final_df.itertuples()] 
candidate_texts = [row._2.split() for row in model_responses.itertuples()]
bleu_score = corpus_bleu(reference_texts, candidate_texts, smoothing_function=SmoothingFunction().method1)
print(f"Average BLEU score: {bleu_score:.4f}")