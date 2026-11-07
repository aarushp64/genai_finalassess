"""
NeuroTutor AI - Hallucination Checker & Grounding Evaluation Module
Author: Aarush Parate (Roll No: 1)
Unit V: Responsible AI and Ethics
Evaluates generated answers against retrieved academic notes to detect hallucinations and verify claim grounding.
"""

import re
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, List, Tuple


class HallucinationChecker:
    """Verifies claim entailment against retrieved source passages and computes grounding metrics."""

    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        """Splits answer text into individual candidate claim sentences."""
        if not text:
            return []
        # Split on sentence terminals while avoiding abbreviation splits
        raw_sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        sentences = [
            s.strip()
            for s in raw_sentences
            if len(s.strip()) > 15 and not s.strip().startswith("#")
        ]
        return sentences

    @classmethod
    def check_grounding(
        cls,
        answer: str,
        retrieved_passages: List[Dict[str, Any]],
        supported_threshold: float = 0.60,
        partial_threshold: float = 0.30,
    ) -> Dict[str, Any]:
        """
        Calculates lexical and semantic overlap between claims in the answer
        and evidence in the retrieved passages.
        Returns:
            Dictionary with grounding score, status, verified claims, and unsupported claims.
        """
        if not retrieved_passages:
            return {
                "grounding_score": 0.0,
                "status": "Not Supported",
                "verified_claims_count": 0,
                "total_claims_count": 1,
                "unsupported_claims": [answer],
                "explanation": "No reference passages retrieved. Zero grounding evidence.",
                "disclaimer": "Grounding check is heuristic-based and probabilistic; it does not constitute absolute proof.",
            }

        # Combine all retrieved text
        corpus_text = " ".join([p.get("text", "") for p in retrieved_passages]).lower()
        corpus_words = set(re.findall(r"\b[a-z]{3,}\b", corpus_text))

        sentences = cls.split_into_sentences(answer)
        if not sentences:
            sentences = [answer.strip()]

        supported_claims = []
        unsupported_claims = []
        claim_scores = []

        for sent in sentences:
            sent_words = set(re.findall(r"\b[a-z]{3,}\b", sent.lower()))
            if not sent_words:
                continue

            # Calculate content word overlap (excluding common stop words)
            stop_words = {"the", "and", "that", "this", "with", "from", "for", "are", "was", "were", "been"}
            content_words = sent_words - stop_words
            if not content_words:
                continue

            overlap = content_words.intersection(corpus_words)
            overlap_ratio = len(overlap) / len(content_words)
            claim_scores.append(overlap_ratio)

            if overlap_ratio >= 0.35:
                supported_claims.append(sent)
            else:
                unsupported_claims.append(sent)

        # Average grounding score
        overall_score = float(np.mean(claim_scores)) if claim_scores else 0.0
        overall_score = round(min(1.0, max(0.0, overall_score)), 3)

        # Classify status
        if overall_score >= supported_threshold:
            status = "Supported"
        elif overall_score >= partial_threshold:
            status = "Partially Supported"
        else:
            status = "Not Supported"

        return {
            "grounding_score": overall_score,
            "status": status,
            "total_claims_count": len(sentences),
            "verified_claims_count": len(supported_claims),
            "unsupported_claims_count": len(unsupported_claims),
            "supported_claims": supported_claims,
            "unsupported_claims": unsupported_claims,
            "disclaimer": "Probabilistic heuristic grounding analysis. Does not guarantee absolute factual infallibility.",
        }

    @staticmethod
    def evaluate_test_dataset(test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Runs batch evaluation on a curated ground-truth labelled test dataset.
        Reports Accuracy, Precision, Recall, and Confusion Matrix.
        """
        classes = ["Supported", "Partially Supported", "Not Supported"]
        matrix = {true_c: {pred_c: 0 for pred_c in classes} for true_c in classes}

        correct_count = 0
        results_breakdown = []

        for case in test_cases:
            true_label = case["ground_truth_label"]
            answer = case["answer"]
            passages = [{"text": p} for p in case["passages"]]

            res = HallucinationChecker.check_grounding(answer, passages)
            pred_label = res["status"]

            if true_label in matrix and pred_label in matrix[true_label]:
                matrix[true_label][pred_label] += 1

            is_correct = (true_label == pred_label)
            if is_correct:
                correct_count += 1

            results_breakdown.append({
                "question": case["question"],
                "true_label": true_label,
                "predicted_label": pred_label,
                "grounding_score": res["grounding_score"],
                "is_correct": is_correct,
            })

        total = len(test_cases)
        accuracy = round((correct_count / total) * 100, 2) if total > 0 else 0.0

        return {
            "total_cases": total,
            "correct_cases": correct_count,
            "incorrect_cases": total - correct_count,
            "accuracy_percent": accuracy,
            "confusion_matrix": matrix,
            "results_breakdown": results_breakdown,
        }

    @staticmethod
    def plot_confusion_matrix(matrix: Dict[str, Dict[str, int]]) -> go.Figure:
        """Visualizes evaluation confusion matrix using Plotly Heatmap."""
        labels = ["Supported", "Partially Supported", "Not Supported"]
        z_values = [[matrix[true_l][pred_l] for pred_l in labels] for true_l in labels]

        fig = go.Figure(
            data=go.Heatmap(
                z=z_values,
                x=labels,
                y=labels,
                colorscale="Blues",
                text=z_values,
                texttemplate="%{text}",
                textfont={"size": 16, "color": "white"},
                hoverongaps=False,
            )
        )

        fig.update_layout(
            title=dict(
                text="Grounding & Hallucination Evaluation Confusion Matrix",
                font=dict(color="#F2F2F2", size=14),
            ),
            xaxis=dict(title="Predicted Label", color="#AEB7BF"),
            yaxis=dict(title="Ground Truth Label", color="#AEB7BF", autorange="reversed"),
            paper_bgcolor="#2A3140",
            plot_bgcolor="#343F4E",
            margin=dict(l=60, r=40, t=50, b=50),
            height=350,
        )
        return fig
