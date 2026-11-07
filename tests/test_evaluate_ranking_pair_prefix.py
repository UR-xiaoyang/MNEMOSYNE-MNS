"""Test pair_id prefix validation in evaluate_ranking.py."""

import unittest

from scripts.evaluate_ranking import paired_bootstrap_comparison


class TestPairIdPrefixValidation(unittest.TestCase):
    """Test that pair_id with :: must have prefix matching query_id."""

    def test_pair_id_with_double_colon_must_match_query_id_in_predictions_a(self):
        """pair_id with :: must have prefix == query_id in predictions_a."""
        predictions_a = [
            {"query_id": "q1", "pair_id": "q999::c1", "label": 1, "score_a": 0.9},
            {"query_id": "q1", "pair_id": "q1::c2", "label": 0, "score_a": 0.3},
        ]
        predictions_b = [
            {"query_id": "q1", "pair_id": "q999::c1", "label": 1, "score_b": 0.8},
            {"query_id": "q1", "pair_id": "q1::c2", "label": 0, "score_b": 0.4},
        ]

        with self.assertRaises(ValueError) as ctx:
            paired_bootstrap_comparison(
                predictions_a, predictions_b, "score_a", "score_b", n_resamples=10
            )

        self.assertIn("pair_id", str(ctx.exception).lower())
        self.assertIn("q999::c1", str(ctx.exception))
        self.assertIn("q1", str(ctx.exception))

    def test_pair_id_with_double_colon_must_match_query_id_in_predictions_b(self):
        """pair_id with :: must have prefix == query_id in predictions_b."""
        predictions_a = [
            {"query_id": "q2", "pair_id": "q2::c1", "label": 1, "score_a": 0.9},
            {"query_id": "q2", "pair_id": "q2::c2", "label": 0, "score_a": 0.3},
        ]
        predictions_b = [
            {"query_id": "q2", "pair_id": "q2::c1", "label": 1, "score_b": 0.8},
            {"query_id": "q2", "pair_id": "qXXX::c2", "label": 0, "score_b": 0.4},
        ]

        with self.assertRaises(ValueError) as ctx:
            paired_bootstrap_comparison(
                predictions_a, predictions_b, "score_a", "score_b", n_resamples=10
            )

        self.assertIn("pair_id", str(ctx.exception).lower())
        self.assertIn("qXXX::c2", str(ctx.exception))

    def test_pair_id_without_double_colon_is_allowed(self):
        """Opaque pair_id without :: should pass validation."""
        predictions_a = [
            {"query_id": "q1", "pair_id": "opaque_id_123", "label": 1, "score_a": 0.9},
            {"query_id": "q1", "pair_id": "another_opaque", "label": 0, "score_a": 0.3},
        ]
        predictions_b = [
            {"query_id": "q1", "pair_id": "opaque_id_123", "label": 1, "score_b": 0.8},
            {"query_id": "q1", "pair_id": "another_opaque", "label": 0, "score_b": 0.4},
        ]

        # Should not raise
        result = paired_bootstrap_comparison(
            predictions_a, predictions_b, "score_a", "score_b", n_resamples=10
        )
        self.assertIsInstance(result, dict)

    def test_mixed_pair_id_formats_with_mismatch_raises(self):
        """Mix of :: and opaque IDs, with one :: mismatch, should raise."""
        predictions_a = [
            {"query_id": "q1", "pair_id": "q1::c1", "label": 1, "score_a": 0.9},
            {"query_id": "q1", "pair_id": "opaque_ok", "label": 0, "score_a": 0.5},
            {"query_id": "q1", "pair_id": "q999::bad", "label": 0, "score_a": 0.2},
        ]
        predictions_b = [
            {"query_id": "q1", "pair_id": "q1::c1", "label": 1, "score_b": 0.8},
            {"query_id": "q1", "pair_id": "opaque_ok", "label": 0, "score_b": 0.6},
            {"query_id": "q1", "pair_id": "q999::bad", "label": 0, "score_b": 0.3},
        ]

        with self.assertRaises(ValueError) as ctx:
            paired_bootstrap_comparison(
                predictions_a, predictions_b, "score_a", "score_b", n_resamples=10
            )

        self.assertIn("q999::bad", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
