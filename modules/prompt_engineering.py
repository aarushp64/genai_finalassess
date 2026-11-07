"""
NeuroTutor AI - Prompt Engineering Lab Module
Author: Aarush Parate (Roll No: 1)
Unit I: Foundations of Generative AI
Features: Zero-Shot, Few-Shot, Chain-of-Thought, Temperature Softmax Dynamics, and Latent Space Mathematics.
"""

import numpy as np
import time
from typing import Dict, Any, List
from modules.llm_engine import LLMEngine


class PromptEngineeringLab:
    """Implements Zero-Shot, Few-Shot, and Chain-of-Thought prompt strategies and temperature probability math."""

    def __init__(self, llm_engine: LLMEngine):
        self.llm = llm_engine

    def build_zero_shot_prompt(self, user_question: str, level: str) -> str:
        """Constructs a direct Zero-Shot instruction prompt."""
        return (
            f"You are NeuroTutor AI, an educational tutor.\n"
            f"Target Audience Level: {level}.\n\n"
            f"Instruction: Explain the following concept directly, accurately, and clearly for the specified audience level without providing unrelated examples.\n\n"
            f"Concept/Question: {user_question}\n\n"
            f"Answer:"
        )

    def build_few_shot_prompt(self, user_question: str, level: str) -> str:
        """Constructs a Few-Shot prompt with structured exemplary input-output demonstrations."""
        return (
            f"You are NeuroTutor AI, an educational tutor. Follow the demonstration pattern below to answer the user's question at the '{level}' level.\n\n"
            f"--- Example 1 ---\n"
            f"Concept: Supervised Learning\n"
            f"Explanation: A machine learning paradigm where the model trains on input data paired with correct ground-truth labels (like a student studying with an answer key).\n"
            f"Core Mechanism: Minimizes difference between prediction f(x) and ground truth y via loss function L(f(x), y).\n"
            f"Key Takeaway: Used when historical labeled data is readily available for classification or regression.\n\n"
            f"--- Example 2 ---\n"
            f"Concept: Overfitting\n"
            f"Explanation: When a model memorizes noise and specific peculiarities of the training dataset instead of generalizing the underlying statistical distribution.\n"
            f"Core Mechanism: High variance, low bias; training loss continues decreasing while validation loss increases.\n"
            f"Key Takeaway: Prevented via L1/L2 regularization, dropout, data augmentation, and early stopping.\n\n"
            f"--- Now Answer for This Concept ---\n"
            f"Concept: {user_question}\n"
            f"Explanation:"
        )

    def build_chain_of_thought_prompt(self, user_question: str, level: str) -> str:
        """Constructs a Chain-of-Thought (CoT) prompt encouraging explicit structured reasoning."""
        return (
            f"You are NeuroTutor AI. Solve and explain the following concept by breaking down your cognitive reasoning step by step.\n"
            f"Target Audience Level: {level}.\n\n"
            f"Concept/Question: {user_question}\n\n"
            f"Structure your response strictly into these 4 logical reasoning steps:\n"
            f"1. Intuitive Definition & Goal (Why does this concept exist?)\n"
            f"2. Step-by-Step Mechanism / Breakdown (How does it function internally?)\n"
            f"3. Concrete Educational Example or Mathematical Formulation\n"
            f"4. Summary & Practical Takeaway\n\n"
            f"Reasoning:"
        )

    def compare_prompt_techniques(
        self, user_question: str, level: str = "Undergraduate Level", temperature: float = 0.7
    ) -> Dict[str, Dict[str, Any]]:
        """
        Executes all three prompting strategies (Zero-Shot, Few-Shot, Chain-of-Thought)
        for identical user question and records response text, latency, token count, and prompt template.
        """
        zero_shot_prompt = self.build_zero_shot_prompt(user_question, level)
        few_shot_prompt = self.build_few_shot_prompt(user_question, level)
        cot_prompt = self.build_chain_of_thought_prompt(user_question, level)

        # Run Zero-Shot
        res_zero = self.llm.generate(
            prompt=zero_shot_prompt,
            temperature=temperature,
            level=level
        )

        # Run Few-Shot
        res_few = self.llm.generate(
            prompt=few_shot_prompt,
            temperature=temperature,
            level=level
        )

        # Run Chain-of-Thought
        res_cot = self.llm.generate(
            prompt=cot_prompt,
            temperature=temperature,
            level=level
        )

        return {
            "Zero-Shot": {
                "prompt": zero_shot_prompt,
                "response": res_zero["text"],
                "latency": res_zero["latency"],
                "tokens": res_zero["tokens_approx"],
                "provider": res_zero["provider"],
            },
            "Few-Shot": {
                "prompt": few_shot_prompt,
                "response": res_few["text"],
                "latency": res_few["latency"],
                "tokens": res_few["tokens_approx"],
                "provider": res_few["provider"],
            },
            "Chain-of-Thought": {
                "prompt": cot_prompt,
                "response": res_cot["text"],
                "latency": res_cot["latency"],
                "tokens": res_cot["tokens_approx"],
                "provider": res_cot["provider"],
            },
        }

    @staticmethod
    def calculate_token_probabilities(
        logits: List[float], temperature: float
    ) -> Dict[str, Any]:
        """
        Applies temperature scaling to raw logits and computes resulting softmax probability distribution.
        p_i = exp(z_i / T) / sum(exp(z_j / T))
        Also computes Shannon entropy H = -sum(p_i * log2(p_i)) to quantify randomness.
        """
        logits_arr = np.array(logits, dtype=np.float64)

        if temperature <= 0.001:
            # Greedy argmax
            probs = np.zeros_like(logits_arr)
            probs[np.argmax(logits_arr)] = 1.0
            entropy = 0.0
        else:
            scaled_logits = logits_arr / temperature
            # Numerical stability shift
            shifted_logits = scaled_logits - np.max(scaled_logits)
            exp_logits = np.exp(shifted_logits)
            probs = exp_logits / np.sum(exp_logits)
            
            # Shannon entropy calculation
            non_zero_probs = probs[probs > 1e-12]
            entropy = -np.sum(non_zero_probs * np.log2(non_zero_probs))

        return {
            "probabilities": probs.tolist(),
            "entropy": round(float(entropy), 4),
            "temperature": temperature,
        }
