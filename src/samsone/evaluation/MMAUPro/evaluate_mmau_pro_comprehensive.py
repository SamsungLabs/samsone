#!/usr/bin/env python3
"""
Comprehensive MMAU-Pro Evaluation Script (source: https://github.com/sonalkum/MMAUPro/blob/main/evaluate_mmau_pro_comprehensive.py)

This script evaluates different types of questions from a parquet file using appropriate evaluation methods:
- Open-ended questions: Qwen 2.5 LLM judge evaluation
- Instruction following questions: Audio Instruction Following (AIF) format evaluation
- Closed-ended questions: NVEmbed similarity-based evaluation
"""

import re
import warnings

from transformers import AutoModel
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
from pandas import DataFrame
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")

nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("punkt_tab")

# ================================
# Audio Instruction Following (AIF) Evaluation Functions
# ================================


def count_words(text):
    return len(text.split())


def count_sentences(text):
    sentences = sent_tokenize(text)
    return len(sentences)


def count_paragraphs(text):
    # paragprah1 *** paragraph2
    paragraphs = text.split("***")
    return len([p for p in paragraphs if p.strip()])


def count_bullet_points(text):
    # Match bullets at the start of the string OR start of a line
    bullets = re.findall(r"(?:^|\n)\s*\*\s+", text)
    return len(bullets)


def count_highlighted_sections(text):
    # * highlight *
    highlights = re.findall(r"\*([^*]+)\*", text)
    return len(highlights)


def count_placeholders(text):
    # [placeholder]
    placeholders = re.findall(r"\[[^\]]+\]", text)
    return len(placeholders)


def count_capital_words(text):
    words = text.split()
    capital_words = []
    for word in words:
        if word.isupper():
            capital_words.append(word)
    return len(capital_words)


def count_keyword_frequency(text, keyword):
    pattern = r"\b" + re.escape(keyword.lower()) + r"\b"
    matches = re.findall(pattern, text.lower())
    return len(matches)


def has_title(text):
    # <<title>>
    return bool(re.search(r"<<[^>]+>>", text))


def has_postscript(text, marker):
    text_alpha = re.sub(r"[^a-zA-Z]", "", text).lower()
    marker_alpha = re.sub(r"[^a-zA-Z]", "", marker).lower()
    return marker_alpha in text_alpha


def starts_with_phrase(text, phrase):
    text_alpha = re.sub(r"[^a-zA-Z ]", "", text).lower()
    phrase_alpha = re.sub(r"[^a-zA-Z ]", "", phrase).lower()
    return text_alpha.startswith(phrase_alpha)


def ends_with_phrase(text, phrase):
    text_alpha = re.sub(r"[^a-zA-Z ]", "", text).lower()
    phrase_alpha = re.sub(r"[^a-zA-Z ]", "", phrase).lower()
    return text_alpha.endswith(phrase_alpha)


def is_wrapped_in_quotes(text):
    stripped = text.strip()
    return stripped.startswith('"') and stripped.endswith('"')


def has_no_commas(text):
    return "," not in text


def check_sections(text, num_sections, splitter):
    # Escape the splitter in case it contains special regex characters
    escaped_splitter = re.escape(splitter)

    # Split on the exact delimiter (not inside words)
    sections = re.split(rf"\s*{escaped_splitter}\s*", text.strip())

    # Remove empty/whitespace-only sections
    actual_sections = [s for s in sections if s.strip()]

    return len(actual_sections) == num_sections


def evaluate_aif_sample(response, sample_data):
    """Evaluate Audio Instruction Following sample"""
    task_identifier = sample_data.get("task_identifier", "")
    kwargs = sample_data.get("kwargs", {}) or {}

    success = False

    if task_identifier == "Include Keywords":
        keywords = kwargs.get("keywords", "").split(", ")
        success = all(keyword.lower() in response.lower() for keyword in keywords)

    elif task_identifier == "Keyword Frequency":
        keyword = kwargs.get("keyword", "")
        target = kwargs.get("N", 0)
        actual = count_keyword_frequency(response, keyword)
        success = actual == target

    elif task_identifier == "Forbidden Words":
        forbidden_words = kwargs.get("forbidden_words", "").split(", ")
        success = not any(word.lower() in response.lower() for word in forbidden_words)

    elif task_identifier == "Number Paragraphs":
        target = kwargs.get("N", 0)
        actual = count_paragraphs(response)
        success = actual == target

    elif task_identifier == "Number Words (at least)":
        target = kwargs.get("N", 0)
        actual = count_words(response)
        success = actual >= target

    elif task_identifier == "Number Words (at most)":
        target = kwargs.get("N", 0)
        actual = count_words(response)
        success = actual <= target

    elif task_identifier == "Number Words (range)":
        N1 = kwargs.get("N1", 0)
        N2 = kwargs.get("N2", 999)
        actual = count_words(response)
        success = N1 <= actual <= N2

    elif task_identifier == "Number Sentences (at least)":
        target = kwargs.get("N", 0)
        actual = count_sentences(response)
        success = actual >= target

    elif task_identifier == "Number Sentences (at most)":
        target = kwargs.get("N", 0)
        actual = count_sentences(response)
        success = actual <= target

    elif task_identifier == "Number Sentences (range)":
        N1 = kwargs.get("N1", 0)
        N2 = kwargs.get("N2", 999)
        actual = count_sentences(response)
        success = N1 <= actual <= N2

    elif task_identifier == "Postscript":
        marker = kwargs.get("postscript_marker", "")
        success = has_postscript(response, marker)

    elif task_identifier == "Number Placeholder":
        target = kwargs.get("N", 0)
        actual = count_placeholders(response)
        success = actual >= target

    elif task_identifier == "Number Bullets":
        target = kwargs.get("N", 0)
        actual = count_bullet_points(response)
        success = actual == target

    elif task_identifier == "Title":
        success = has_title(response)

    elif task_identifier == "Minimum Number Highlighted Section":
        target = kwargs.get("N", 0)
        actual = count_highlighted_sections(response)
        success = actual >= target

    elif task_identifier == "Multiple Sections":
        target = kwargs.get("N", 0)
        splitter = kwargs.get("section_splitter", "")
        success = check_sections(response, target, splitter)

    elif task_identifier == "Repeat Prompt":
        original_prompt = sample_data.get("prompt_transcription", "")
        success = response.strip().lower().startswith(original_prompt.strip().lower())

    elif task_identifier == "Two Responses":
        separator = "******"
        parts = response.split(separator)
        success = (
            len(parts) == 2 and parts[0].lower().strip() != parts[1].lower().strip()
        )

    elif task_identifier == "All Uppercase":
        success = response.isupper()

    elif task_identifier == "All Lowercase":
        success = response.islower()

    elif task_identifier == "All-capital Words (at least)":
        target = kwargs.get("N", 0)
        actual = count_capital_words(response)
        success = actual >= target

    elif task_identifier == "All-capital Words (at most)":
        target = kwargs.get("N", 0)
        actual = count_capital_words(response)
        success = actual <= target

    elif task_identifier == "All-capital Words (range)":
        N1 = kwargs.get("N1", 0)
        N2 = kwargs.get("N2", 999)
        actual = count_capital_words(response)
        success = N1 <= actual <= N2

    elif task_identifier == "Start Checker":
        phrase = kwargs.get("start_phrase", "")
        success = starts_with_phrase(response, phrase)

    elif task_identifier == "End Checker":
        phrase = kwargs.get("end_phrase", "")
        success = ends_with_phrase(response, phrase)

    elif task_identifier == "Quotation":
        success = is_wrapped_in_quotes(response)

    elif task_identifier == "No Commas":
        success = has_no_commas(response)

    return success


# ================================
# Open-ended Evaluation Functions (Qwen 2.5)
# ================================


def load_qwen_model(use_reduced_models: bool):
    """Load the Qwen 2.5 model"""
    if not use_reduced_models:
        print("Loading Qwen 2.5-7B model...")
        model_name = "Qwen/Qwen2.5-7B-Instruct" # this is the original model
    else:
        print("Loading Qwen 2.5-0.5B model...")
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"  # for speed purposes

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    print("Qwen 2.5 model loaded successfully!")
    return model, tokenizer


def create_evaluation_prompt(question, reference_answer, model_response, task_type):
    """Create a structured prompt for the LLM judge to evaluate responses"""

    task_context = {
        "sound": "audio content analysis and sound identification",
        "speech": "speech recognition and conversation understanding",
        "music": "music analysis and musical element identification",
        "open": "general open-ended question answering",
    }

    context = task_context.get(task_type, "general question answering")

    prompt = f"""You are an expert evaluator for {context} tasks. Please evaluate the quality of a model's response to a question.

Question: {question}

Reference Answer: {reference_answer}

Model Response: {model_response}

Please evaluate the model response on the following criteria and provide scores from 1-5 (where 5 is best):

1. **Correctness**: How factually accurate is the response compared to the reference?
2. **Relevance**: How well does the response address the specific question asked?
3. **Completeness**: Does the response cover all important aspects mentioned in the reference?
4. **Clarity**: How clear and well-structured is the response?

For each criterion, provide:
- A score from 1-5
- A brief justification (1-2 sentences)

Format your response as:

CORRECTNESS: [score] - [justification]
RELEVANCE: [score] - [justification]
COMPLETENESS: [score] - [justification]
CLARITY: [score] - [justification]
OVERALL: [average score] - [overall assessment]"""

    return prompt


def extract_scores_from_evaluation(evaluation_text):
    """Extract numerical scores from the LLM judge evaluation"""
    scores = {}

    # Define patterns to extract scores
    patterns = {
        "correctness": r"CORRECTNESS:\s*(\d+)",
        "relevance": r"RELEVANCE:\s*(\d+)",
        "completeness": r"COMPLETENESS:\s*(\d+)",
        "clarity": r"CLARITY:\s*(\d+)",
        "overall": r"OVERALL:\s*(\d+(?:\.\d+)?)",
    }

    for criterion, pattern in patterns.items():
        match = re.search(pattern, evaluation_text, re.IGNORECASE)
        if match:
            scores[criterion] = float(match.group(1))
        else:
            # Fallback: assign neutral score if not found
            scores[criterion] = 3.0

    # Calculate overall if not found
    if "overall" not in scores or scores["overall"] == 3.0:
        criteria_scores = [
            scores.get(k, 3.0)
            for k in ["correctness", "relevance", "completeness", "clarity"]
        ]
        scores["overall"] = np.mean(criteria_scores)

    return scores


def evaluate_openended_with_qwen(
    model, tokenizer, questions, reference_answers, model_responses, task_types
):
    """Evaluate open-ended responses using Qwen 2.5 as a judge"""
    all_scores = []
    detailed_evaluations = []

    print("Performing Qwen 2.5 LLM judge evaluation...")

    for i, (question, ref_answer, model_response, task_type) in tqdm(
        enumerate(zip(questions, reference_answers, model_responses, task_types))
    ):
        # Create evaluation prompt
        eval_prompt = create_evaluation_prompt(
            question, ref_answer, model_response, task_type
        )

        # Tokenize and generate evaluation
        messages = [
            {"role": "system", "content": "You are a helpful and objective evaluator."},
            {"role": "user", "content": eval_prompt},
        ]

        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Tokenize
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # Generate evaluation
        with torch.no_grad():
            generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.1,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode response
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        evaluation_text = tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        # Extract scores
        scores = extract_scores_from_evaluation(evaluation_text)
        all_scores.append(scores)

        detailed_evaluations.append(
            {
                "question": question,
                "reference_answer": ref_answer,
                "model_response": model_response,
                "evaluation": evaluation_text,
                "scores": scores,
                "task_type": task_type,
            }
        )

    return all_scores, detailed_evaluations


# ================================
# Closed-ended Evaluation Functions (NVEmbed)
# ================================


def load_embed_model(use_reduced_models: bool):
    # Original model used in the evaluation pipeline
    if not use_reduced_models:
        print("Loading nvidia/NV-Embed-V2 model...")
        model = AutoModel.from_pretrained(
            "nvidia/NV-Embed-V2", trust_remote_code=True, local_files_only=False
        )
    else:
        print("Loading Qwen/Qwen3-Embedding-0.6B model...")
        model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

    model.to("cuda" if torch.cuda.is_available() else "cpu")

    print("Embed model loaded successfully!")
    return model


def evaluate_closedended_with_embed(
    model, questions, choices_list, ground_truth_answers, predicted_answers, task_types, use_reduced_models,
):
    """Embedding-based evaluation: match predictions to choices using embedding similarity"""
    predictions = []
    confidence_scores = []

    print("Performing Embed evaluation...")

    for i, (question, choices, gt_answer, model_prediction, task_type) in tqdm(
        enumerate(
            zip(
                questions,
                choices_list,
                ground_truth_answers,
                predicted_answers,
                task_types,
            )
        )
    ):
        # Encode the prediction (what the model said)
        if use_reduced_models:
            prediction_embedding = model.encode([model_prediction], convert_to_tensor=True)
        else:
            prediction_embedding = model.encode([model_prediction], instruction="", max_length=4096)

        prediction_embedding = F.normalize(prediction_embedding, p=2, dim=1)

        # Encode each choice option
        if use_reduced_models:
            choice_embeddings = model.encode(choices, convert_to_tensor=True)
        else:
            choice_embeddings = model.encode(choices, instruction="", max_length=4096)

        choice_embeddings = F.normalize(choice_embeddings, p=2, dim=1)

        # Calculate similarity between prediction and each choice
        scores = (prediction_embedding @ choice_embeddings.T) * 100
        scores = scores.squeeze()

        # Find the choice most similar to the prediction
        best_choice_idx = torch.argmax(scores).item()
        matched_choice = choices[best_choice_idx]
        confidence = torch.max(scores).item()

        predictions.append(matched_choice)
        confidence_scores.append(confidence)

    return predictions, confidence_scores


# ================================
# Utility Functions
# ================================


def calculate_metrics(ground_truth, predictions):
    """Calculate evaluation metrics"""
    if len(ground_truth) == 0 or len(predictions) == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0}

    accuracy = accuracy_score(ground_truth, predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        ground_truth, predictions, average="weighted", zero_division=0
    )

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

    return metrics


def calculate_openended_metrics(all_scores):
    """Calculate evaluation metrics from LLM judge scores"""
    if not all_scores:
        return {}

    # Calculate averages for each criterion
    criteria = ["correctness", "relevance", "completeness", "clarity", "overall"]
    metrics = {}

    for criterion in criteria:
        scores = [score_dict.get(criterion, 3.0) for score_dict in all_scores]
        metrics[f"avg_{criterion}"] = np.mean(scores)
        metrics[f"std_{criterion}"] = np.std(scores)

    # Calculate percentage of good responses (score >= 4.0)
    good_responses = sum(
        1 for scores in all_scores if scores.get("overall", 3.0) >= 4.0
    )
    metrics["good_response_rate"] = good_responses / len(all_scores)

    # Calculate percentage of poor responses (score <= 2.0)
    poor_responses = sum(
        1 for scores in all_scores if scores.get("overall", 3.0) <= 2.0
    )
    metrics["poor_response_rate"] = poor_responses / len(all_scores)

    return metrics


def analyze_by_category(categories, metrics_list):
    """Analyze performance by different categories"""
    analysis = {}

    # Group by category
    category_groups = {}
    for i, category in enumerate(categories):
        if category not in category_groups:
            category_groups[category] = []
        if i < len(metrics_list):
            category_groups[category].append(metrics_list[i])

    for category, category_metrics in category_groups.items():
        if category_metrics:
            if (
                isinstance(category_metrics[0], dict)
                and "overall" in category_metrics[0]
            ):
                # Open-ended metrics
                overall_metrics = calculate_openended_metrics(category_metrics)
                analysis[category] = {
                    "count": len(category_metrics),
                    "type": "openended",
                    "avg_overall": overall_metrics.get("avg_overall", 0.0),
                    "avg_correctness": overall_metrics.get("avg_correctness", 0.0),
                    "good_response_rate": overall_metrics.get(
                        "good_response_rate", 0.0
                    ),
                }
            elif isinstance(category_metrics[0], (bool, int, float)):
                # AIF metrics (boolean success)
                success_rate = np.mean([float(m) for m in category_metrics])
                analysis[category] = {
                    "count": len(category_metrics),
                    "type": "aif",
                    "success_rate": success_rate,
                }
            else:
                # Default case
                analysis[category] = {"count": len(category_metrics), "type": "unknown"}

    return analysis


def calculate_weighted_performance(category_results):
    """Calculate overall weighted performance using preferred metrics for each type"""
    total_weighted_score = 0.0
    total_samples = 0
    category_scores = {}

    for category, result in category_results.items():
        count = result["count"]

        # Use preferred performance metric for each type
        if result["type"] == "openended":
            # Use avg_overall score (1-5 scale), normalize to 0-1 for weighting
            score = result["metrics"].get("avg_overall", 3.0) / 5.0
        elif result["type"] == "aif":
            # Use success rate directly (0-1 scale)
            score = result["success_rate"]
        elif result["type"] == "closed":
            # Use accuracy as preferred metric (0-1 scale)
            score = result["metrics"].get("accuracy", 0.0)
        else:
            # Unknown type, skip
            continue

        category_scores[category] = score
        total_weighted_score += score * count
        total_samples += count

    overall_weighted_performance = (
        total_weighted_score / total_samples if total_samples > 0 else 0.0
    )

    return overall_weighted_performance, category_scores


# ================================
# Main Evaluation Function
# ================================


def main(df: DataFrame, model_output_column_name: str, parquet_file_path: str, use_reduced_models: bool):
    model_output_column = model_output_column_name

    print(f"Loaded {len(df)} samples")
    print(f"Categories: {df['category'].value_counts()}")

    # Check if model output column exists
    if model_output_column not in df.columns:
        print(f"Warning: Model output column '{model_output_column}' not found.")
        print("Available columns:", list(df.columns))
        print("Creating dummy model outputs for demonstration...")
        df[model_output_column] = df["answer"]  # Use ground truth as placeholder

    # Initialize results storage
    category_results = {}

    # ================================
    # Process Open-ended Questions
    # ================================
    print("\n" + "=" * 60)
    print("EVALUATING OPEN-ENDED QUESTIONS")
    print("=" * 60)

    open_df = df[df["category"] == "open"].copy()
    if len(open_df) > 0:
        print(f"Found {len(open_df)} open-ended questions")

        # Load Qwen model for open-ended evaluation
        qwen_model, qwen_tokenizer = load_qwen_model(use_reduced_models)

        questions = open_df["question"].tolist()
        reference_answers = open_df["answer"].tolist()
        model_responses = open_df[model_output_column].fillna("").tolist()
        task_types = ["open"] * len(open_df)

        # Evaluate open-ended questions
        openended_scores, openended_detailed = evaluate_openended_with_qwen(
            qwen_model,
            qwen_tokenizer,
            questions,
            reference_answers,
            model_responses,
            task_types,
        )

        # Calculate metrics
        openended_metrics = calculate_openended_metrics(openended_scores)
        category_results["open"] = {
            "type": "openended",
            "count": len(open_df),
            "metrics": openended_metrics,
            "scores": openended_scores,
        }

        print(f"Open-ended evaluation completed: {len(openended_scores)} samples")
        print(
            f"Average Overall Score: {openended_metrics.get('avg_overall', 0.0):.3f}/5.0"
        )

        # Clean up memory
        del qwen_model
        del qwen_tokenizer
        torch.cuda.empty_cache()
    else:
        print("No open-ended questions found")
        openended_scores = []

    # ================================
    # Process Instruction Following Questions
    # ================================
    print("\n" + "=" * 60)
    print("EVALUATING INSTRUCTION FOLLOWING QUESTIONS")
    print("=" * 60)

    aif_df = df[df["category"] == "instruction following"].copy()
    if len(aif_df) > 0:
        print(f"Found {len(aif_df)} instruction following questions")

        aif_results = []
        for idx, row in tqdm(aif_df.iterrows(), total=len(aif_df)):
            model_response = str(row.get(model_output_column, ""))

            # Create sample data for AIF evaluation
            sample_data = {
                "task_identifier": row.get("task_identifier"),
                "kwargs": row.get("kwargs"),
                "prompt_transcription": row.get("question", ""),
            }

            # Evaluate AIF sample
            success = evaluate_aif_sample(model_response, sample_data)
            aif_results.append(success)

        # Calculate AIF metrics
        success_rate = np.mean([float(r) for r in aif_results])
        category_results["instruction following"] = {
            "type": "aif",
            "count": len(aif_df),
            "success_rate": success_rate,
            "results": aif_results,
        }

        print(f"Instruction following evaluation completed: {len(aif_results)} samples")
        print(f"Success Rate: {success_rate:.3f}")
    else:
        print("No instruction following questions found")
        aif_results = []

    # ================================
    # Process Closed-ended Questions
    # ================================
    print("\n" + "=" * 60)
    print("EVALUATING CLOSED-ENDED QUESTIONS")
    print("=" * 60)

    closed_categories = [
        cat
        for cat in df["category"].unique()
        if cat not in ["open", "instruction following"]
    ]
    closed_df = df[df["category"].isin(closed_categories)].copy()

    if len(closed_df) > 0:
        print(
            f"Found {len(closed_df)} closed-ended questions across categories: {closed_categories}"
        )

        # Filter out samples without proper choices
        closed_df = closed_df[closed_df["choices"].notna()].copy()
        closed_df = closed_df[
            closed_df["choices"].apply(
                lambda x: len(x) > 1 if hasattr(x, "__len__") else False
            )
        ].copy()

        print(f"After filtering: {len(closed_df)} samples with valid choices")

        if len(closed_df) > 0:
            # Load Embeddings model for closed-ended evaluation
            embed_model = load_embed_model(use_reduced_models)

            questions = closed_df["question"].tolist()
            ground_truth_answers = closed_df["answer"].tolist()
            choices_list = [
                list(choices) if hasattr(choices, "__iter__") else [str(choices)]
                for choices in closed_df["choices"].tolist()
            ]
            model_predictions = closed_df[model_output_column].fillna("").tolist()
            task_types = closed_df["category"].tolist()

            # Evaluate closed-ended questions
            predictions, confidence_scores = evaluate_closedended_with_embed(
                embed_model,
                questions,
                choices_list,
                ground_truth_answers,
                model_predictions,
                task_types,
                use_reduced_models=use_reduced_models
            )

            # Calculate metrics overall and by category
            overall_closed_metrics = calculate_metrics(
                ground_truth_answers, predictions
            )

            # Store results by category
            for category in closed_categories:
                cat_mask = closed_df["category"] == category
                if cat_mask.sum() > 0:
                    cat_gt = [
                        ground_truth_answers[i]
                        for i, mask in enumerate(cat_mask)
                        if mask
                    ]
                    cat_pred = [
                        predictions[i] for i, mask in enumerate(cat_mask) if mask
                    ]
                    cat_metrics = calculate_metrics(cat_gt, cat_pred)

                    category_results[category] = {
                        "type": "closed",
                        "count": cat_mask.sum(),
                        "metrics": cat_metrics,
                    }

            print(f"Closed-ended evaluation completed: {len(predictions)} samples")
            print(f"Overall Accuracy: {overall_closed_metrics['accuracy']:.4f}")

            # Clean up memory
            del embed_model
            torch.cuda.empty_cache()
        else:
            print("No valid closed-ended questions with choices found")
            predictions = []
            overall_closed_metrics = {}
    else:
        print("No closed-ended questions found")
        predictions = []
        overall_closed_metrics = {}

    # ================================
    # Calculate Weighted Performance
    # ================================
    overall_weighted_performance, category_scores = calculate_weighted_performance(
        category_results
    )

    # ================================
    # Generate Comprehensive Report
    # ================================
    print("\n" + "=" * 80)
    print("COMPREHENSIVE EVALUATION RESULTS")
    print("=" * 80)

    total_samples = len(df)
    evaluated_samples = sum([result["count"] for result in category_results.values()])

    print(f"Total samples: {total_samples}")
    print(f"Successfully evaluated: {evaluated_samples}")
    print(f"Overall Weighted Performance: {overall_weighted_performance:.4f}")

    print("\nBREAKDOWN BY CATEGORY:")
    print("-" * 60)

    for category, result in category_results.items():
        print(f"\n{category.upper()}:")
        print(f"  Type: {result['type']}")
        print(f"  Count: {result['count']}")
        print(f"  Performance Score: {category_scores.get(category, 0.0):.4f}")

        if result["type"] == "openended":
            metrics = result["metrics"]
            print(f"  Avg Overall Score: {metrics.get('avg_overall', 0.0):.3f}/5.0")
            print(f"  Avg Correctness: {metrics.get('avg_correctness', 0.0):.3f}/5.0")

        elif result["type"] == "aif":
            print(f"  Success Rate: {result['success_rate']:.3f}")

        elif result["type"] == "closed":
            metrics = result["metrics"]
            print(f"  Accuracy: {metrics.get('accuracy', 0.0):.4f}")
            print(f"  F1 Score: {metrics.get('f1_score', 0.0):.4f}")

    # ================================
    # Save Results
    # ================================
    results_summary = {
        "total_samples": total_samples,
        "evaluated_samples": evaluated_samples,
        "parquet_file": parquet_file_path,
        "model_output_column": model_output_column,
        "overall_weighted_performance": overall_weighted_performance,
    }

    # Format results for JSON serialization
    for category, result in category_results.items():
        json_result = {
            "type": result["type"],
            "count": result["count"],
            "performance_score": category_scores.get(category, 0.0),
        }

        if result["type"] == "openended":
            json_result["metrics"] = result["metrics"]

        elif result["type"] == "aif":
            json_result["success_rate"] = result["success_rate"]

        elif result["type"] == "closed":
            json_result["metrics"] = result["metrics"]

        for k, v in json_result.items():
            results_summary[f"{category}_{k}"] = v

    print("\nEvaluation completed successfully!")
    print("\n🎯 SUMMARY:")
    print(f"   Overall Weighted Performance: {overall_weighted_performance:.4f}")
    print(f"   Total Samples Evaluated: {evaluated_samples}/{total_samples}")
    print("\nNOTE: This comprehensive evaluation requires:")
    print("- Qwen 2.5 LLM judge for open-ended questions (no fallback)")
    print("- Format constraint checking for instruction following")
    print("- NVEmbed similarity matching for closed-ended questions (no fallback)")
    print("\nPerformance metric per category:")
    print("- Open-ended: LLM overall score / 5.0 (continuous quality assessment)")
    print("- Instruction following: Success rate (binary constraint satisfaction)")
    print("- Closed-ended: Classification accuracy (exact match rate)")
    print("- Overall metric is weighted by sample count per category")

    return results_summary
