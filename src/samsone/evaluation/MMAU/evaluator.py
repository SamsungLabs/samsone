import json
from pathlib import Path

from clearml import Logger
import pandas as pd
import polars as pl
import re

from samsone.evaluation.base import ALMEvaluator

# evaluation code taken from: https://github.com/Sakshi113/MMAU/blob/main/evaluation.py


def string_match(answer, prediction, choices):
    # Function to normalize and tokenize text
    def tokenize(text):
        # Convert to lowercase and find all word tokens
        return set(re.findall(r"\b\w+\b", text.lower()))

    # Tokenize prediction and answer
    prediction_tokens = tokenize(prediction)
    answer_tokens = tokenize(answer)

    if not prediction_tokens:
        return False

    # Tokenize incorrect choices and exclude tokens present in the answer
    incorrect_tokens = set()
    for choice in choices:
        choice_tokens = tokenize(choice)
        if choice_tokens != answer_tokens:
            incorrect_tokens.update(choice_tokens - answer_tokens)

    # Condition 1: All tokens of the answer are in the prediction
    cond1 = answer_tokens.issubset(prediction_tokens)

    # Condition 2: Prediction does not contain any tokens from incorrect choices (excluding shared words)
    cond2 = prediction_tokens.isdisjoint(incorrect_tokens)

    return cond1 and cond2


def evaluate_mmau(df, model_output_column_name: str, meta_file_path: str):
    """
    Evaluate MMAU dataset predictions.

    Args:
        df: DataFrame containing the data and model outputs
        model_output_column_name: Name of the column containing model outputs
        meta_file_path: Path to the meta file (for reference)

    Returns:
        Dictionary containing evaluation results
    """
    corr, total = 0, 0
    no_pred_count = 0

    # Track metrics for different categories:
    task_metrics = {"sound": [0, 0], "music": [0, 0], "speech": [0, 0]}
    diff_metrics = {"easy": [0, 0], "hard": [0, 0], "medium": [0, 0]}

    # Here is the new dict for sub-category metrics
    subcat_metrics = {}

    output_key = model_output_column_name
    matched_outputs = []
    new_data = []

    for idx, sample in df.iterrows():
        # If there's no model output key, skip
        if output_key not in sample or pd.isna(sample[output_key]):
            _prediction = ""
            no_pred_count += 1
        else:
            _prediction = str(sample[output_key])

        _answer = str(sample["answer"])
        task = sample["task"]
        difficulty = sample["difficulty"]

        # Get the sub-category
        subcat = sample.get("sub-category", None)
        if subcat is not None:
            # If we haven't seen this sub-category before, initialize
            if subcat not in subcat_metrics:
                subcat_metrics[subcat] = [0, 0]

        # match_result = string_match(_answer, _prediction, choices) # original method
        match_result = (
            True
            if _prediction.split(")")[0].lower() == _answer.split(")")[0].lower()
            else False
        )

        if match_result:
            task_metrics[task][0] += 1
            diff_metrics[difficulty][0] += 1
            if subcat is not None:
                subcat_metrics[subcat][0] += 1
            matched_outputs.append([_answer, _prediction])
            corr += 1
            sample["match"] = 1
        else:
            sample["match"] = 0

        total += 1
        new_data.append(sample)
        task_metrics[task][1] += 1
        diff_metrics[difficulty][1] += 1
        if subcat is not None:
            subcat_metrics[subcat][1] += 1

    # Prepare results summary
    results = {
        "total_accuracy": (corr / total) * 100 if total > 0 else 0,
        "total_samples": total,
        "correct_predictions": corr,
        "no_prediction_count": no_pred_count,
    }

    # Task-wise accuracy
    for task in task_metrics:
        n_correct, n_total = task_metrics[task]
        acc = (n_correct / n_total) * 100 if n_total > 0 else 0
        results[f"{task}_accuracy"] = acc
        results[f"{task}_n_correct"] = n_correct
        results[f"{task}_n_total"] = n_total

    # Difficulty-wise accuracy
    for diff in diff_metrics:
        n_correct, n_total = diff_metrics[diff]
        acc = (n_correct / n_total) * 100 if n_total > 0 else 0
        results[f"{diff}_accuracy"] = acc
        results[f"{diff}_n_correct"] = n_correct
        results[f"{diff}_n_total"] = n_total

    # Sub-category-wise accuracy
    for subcat in subcat_metrics:
        n_correct, n_total = subcat_metrics[subcat]
        acc = (n_correct / n_total) * 100 if n_total > 0 else 0
        results[f"{subcat}_accuracy"] = acc
        results[f"{subcat}_n_correct"] = n_correct
        results[f"{subcat}_n_total"] = n_total

    # Print results (matching original script format)
    print("*" * 30)
    print("Task-wise Accuracy:")
    for task in task_metrics:
        n_correct, n_total = task_metrics[task]
        acc = (n_correct / n_total) * 100 if n_total > 0 else 0
        print(f"{task} : {acc:.2f}% over {n_total} samples")

    print("*" * 30)
    print("Difficulty-wise Accuracy:")
    for diff in diff_metrics:
        n_correct, n_total = diff_metrics[diff]
        acc = (n_correct / n_total) * 100 if n_total > 0 else 0
        print(f"{diff} : {acc:.2f}% over {n_total} samples")

    print("*" * 30)
    print("Sub-category-wise Accuracy:")
    for subcat in subcat_metrics:
        n_correct, n_total = subcat_metrics[subcat]
        acc = (n_correct / n_total) * 100 if n_total > 0 else 0
        print(f"{subcat} : {acc:.2f}% over {n_total} samples")

    print("*" * 30)
    print(f"Total Accuracy: {(corr / total) * 100:.2f}% over {total} samples")
    print("*" * 30)
    print(f"No prediction count: {no_pred_count}")

    return results


class MMAUEvaluator(ALMEvaluator):
    model_output_column_name = "model_output"

    def __init__(
        self,
        meta_file_path: str,
        id_column_name: str = "id",
        test_dataset_name: str = "MMAU",
    ):
        super().__init__(test_dataset_name=test_dataset_name)
        self.meta_file_path = meta_file_path
        self.df = pd.read_parquet(self.meta_file_path)
        self.id_column_name = id_column_name

        self.results_filename = f"{self.test_dataset_name}_results.json"

    def evaluate(self, generated_answers: dict):
        self.df[MMAUEvaluator.model_output_column_name] = self.df[
            self.id_column_name
        ].map(generated_answers)

        self.results_summary = evaluate_mmau(
            self.df, self.model_output_column_name, self.meta_file_path
        )


class MMAUFullEvaluator(ALMEvaluator):
    """
    Evaluator for MMAUFull dataset that adds model predictions to JSON file.

    This evaluator is designed for the closed MMAUFull dataset where predictions
    need to be submitted to an online evaluator. It reads a JSON file, adds
    model predictions matched by IDs, and saves the modified JSON file.
    """

    id2int = {
        "a": 0,
        "b": 1,
        "c": 2,
        "d": 3,
        "e": 4,
        "f": 5,
        "g": 6,
        "h": 7,
        "i": 8,
        "j": 9,
        "k": 10,
        "l": 11,
        "m": 12,
        "n": 13,
        "o": 14,
        "p": 15,
        "q": 16,
    }

    def __init__(
        self,
        json_file_path: str,
        id_column_name: str = "id",
        test_dataset_name: str = "MMAUFull",
    ):
        """
        Initialize the MMAUFull evaluator.

        Args:
            json_file_path: Path to the JSON file containing test data
            id_column_name: Name of the ID column (default: "id")
            test_dataset_name: Name of the test dataset (default: "MMAUFull")
        """
        super().__init__(test_dataset_name=test_dataset_name)
        self.json_file_path = json_file_path
        self.id_column_name = id_column_name
        self.results_filename = f"{self.test_dataset_name}_results.json"

        # Load the JSON file
        with open(self.json_file_path, "r") as f:
            self.data = json.load(f)

        # Create a mapping from ID to index for efficient lookup
        self.id_to_index = {
            entry[self.id_column_name]: idx for idx, entry in enumerate(self.data)
        }

    def evaluate(self, generated_answers: dict):
        """
        Add model predictions to the JSON data.

        Args:
            generated_answers: Dictionary mapping IDs to model predictions
        """
        # Track statistics
        matched_count = 0
        unmatched_count = 0
        properly_mapped_count = 0
        improperly_mapped_count = 0

        # Add model_prediction field to each entry
        for entry in self.data:
            entry_id = entry[self.id_column_name]

            if entry_id in generated_answers:
                try:
                    response = generated_answers[entry_id]
                    actual_answer = entry["choices"][self.id2int[response[0]]]
                    entry["model_prediction"] = actual_answer
                    properly_mapped_count += 1
                except:
                    entry["model_prediction"] = response
                    improperly_mapped_count += 1
                matched_count += 1
            else:
                entry["model_prediction"] = ""
                unmatched_count += 1

        # Prepare results summary
        self.results_summary = {
            "total_samples": len(self.data),
            "matched_predictions": matched_count,
            "unmatched_predictions": unmatched_count,
            "properly_mapped": properly_mapped_count,
            "improperly_mapped": improperly_mapped_count,
            "match_rate": (matched_count / len(self.data) * 100)
            if len(self.data) > 0
            else 0,
        }

        print(f"***** {self.test_dataset_name} *****")
        print(f"Total samples: {self.results_summary['total_samples']}")
        print(f"Matched predictions: {self.results_summary['matched_predictions']}")
        print(f"Unmatched predictions: {self.results_summary['unmatched_predictions']}")
        print(f"Properly mapped: {self.results_summary['properly_mapped']}")
        print(f"Improperly mapped: {self.results_summary['improperly_mapped']}")
        print(f"Match rate: {self.results_summary['match_rate']:.2f}%")

    def finalize(self, results_dir: str):
        table = pl.from_dict(self.results_summary)
        clearml_logger = Logger.current_logger()
        if clearml_logger is not None:
            clearml_logger.report_table(
                title="Results",
                series=self.test_dataset_name,
                iteration=None,
                table_plot=table.to_pandas(),
            )

        output_filename = Path(results_dir) / self.results_filename

        with open(output_filename, "w") as f:
            json.dump(self.results_summary, f, indent=2, default=float)

        answers_filename = (
            Path(results_dir) / f"{self.test_dataset_name}_eval_file.json"
        )

        with open(answers_filename, "w") as f:
            json.dump(self.data, f, indent=2, default=float)
