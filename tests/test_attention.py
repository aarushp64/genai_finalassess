"""
Unit Tests for Attention Visualizer
Author: Aarush Parate (Roll No: 1)
"""

import unittest
from modules.attention_visualizer import AttentionVisualizer


class TestAttentionVisualizer(unittest.TestCase):
    def setUp(self):
        self.visualizer = AttentionVisualizer()

    def test_attention_extraction(self):
        text = "Neural networks learn representations."
        result = self.visualizer.extract_attentions(text)
        self.assertIsNotNone(result)
        self.assertIn("tokens", result)
        self.assertIn("attentions_by_layer", result)
        self.assertEqual(result["num_layers"], 6)
        self.assertEqual(result["num_heads"], 12)

        # Check shape of layer 0 attention tensor: (12 heads, seq_len, seq_len)
        l0_matrix = result["attentions_by_layer"][0]
        self.assertEqual(len(l0_matrix.shape), 3)
        self.assertEqual(l0_matrix.shape[0], 12)
        seq_len = len(result["tokens"])
        self.assertEqual(l0_matrix.shape[1], seq_len)
        self.assertEqual(l0_matrix.shape[2], seq_len)


if __name__ == "__main__":
    unittest.main()
