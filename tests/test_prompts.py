"""
Unit Tests for Prompt Engineering Lab
Author: Aarush Parate (Roll No: 1)
"""

import unittest
from modules.llm_engine import LLMEngine
from modules.prompt_engineering import PromptEngineeringLab


class TestPromptEngineering(unittest.TestCase):
    def setUp(self):
        self.llm = LLMEngine()
        self.lab = PromptEngineeringLab(self.llm)

    def test_prompt_builders(self):
        question = "What is Self-Attention?"
        level = "Undergraduate Level"

        zero_shot = self.lab.build_zero_shot_prompt(question, level)
        few_shot = self.lab.build_few_shot_prompt(question, level)
        cot = self.lab.build_chain_of_thought_prompt(question, level)

        self.assertIn("Instruction:", zero_shot)
        self.assertIn("Example 1", few_shot)
        self.assertIn("logical reasoning steps", cot)

    def test_temperature_softmax_math(self):
        logits = [2.0, 1.0, 0.1]
        
        # Test low temperature (approaching argmax)
        res_low = PromptEngineeringLab.calculate_token_probabilities(logits, temperature=0.1)
        probs_low = res_low["probabilities"]
        self.assertGreater(probs_low[0], 0.9)  # Dominates highest logit

        # Test high temperature (flattening)
        res_high = PromptEngineeringLab.calculate_token_probabilities(logits, temperature=1.5)
        probs_high = res_high["probabilities"]
        self.assertGreater(res_high["entropy"], res_low["entropy"])


if __name__ == "__main__":
    unittest.main()
