import unittest

import pandas as pd

from gedicorrect.scorer import CorrectionScorer


class ScorerTests(unittest.TestCase):
    def test_all_criteria_uses_one_correlation_method(self):
        scorer = CorrectionScorer(criteria="all")
        self.assertIn("wave_pearson", scorer.criteria)
        self.assertNotIn("wave_spearman", scorer.criteria)

    def test_constant_correlations_normalize_to_valid_scores(self):
        scorer = CorrectionScorer(criteria="wave_pearson")
        scores = scorer._normalize_correl(pd.Series([0.5, 0.5]))
        self.assertEqual(scores.tolist(), [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
