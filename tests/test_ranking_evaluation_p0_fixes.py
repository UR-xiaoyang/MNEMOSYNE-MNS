"""
P0 Fix Tests: Strict paired bootstrap validation.

These tests document the missing validation bugs that allow invalid comparisons.
They MUST fail first, then pass after fixes are applied.
"""

import unittest


class TestStrictPairedValidation(unittest.TestCase):
    """Test strict validation requirements for paired bootstrap."""

    def test_rejects_missing_query_in_one_prediction_set(self):
        """
        CRITICAL: If A has queries [q1, q2] but B has only [q1],
        paired bootstrap MUST reject the comparison entirely.
        
        Current bug: Takes intersection and silently drops q2.
        Expected: Raise ValueError stating query sets don't match.
        """
        from scripts.evaluate_ranking import paired_bootstrap_comparison

        # Model A has q1 and q2
        predictions_a = [
            {"pair_id": "q1::m1", "query_id": "q1", "label": 1, "score": 0.9},
            {"pair_id": "q1::m2", "query_id": "q1", "label": 0, "score": 0.5},
            {"pair_id": "q2::m1", "query_id": "q2", "label": 1, "score": 0.8},
            {"pair_id": "q2::m2", "query_id": "q2", "label": 0, "score": 0.6},
        ]

        # Model B has only q1 (missing q2)
        predictions_b = [
            {"pair_id": "q1::m1", "query_id": "q1", "label": 1, "score": 0.95},
            {"pair_id": "q1::m2", "query_id": "q1", "label": 0, "score": 0.55},
        ]

        with self.assertRaises(ValueError) as context:
            paired_bootstrap_comparison(
                predictions_a, predictions_b, "score", "score", n_resamples=100, seed=42
            )

        # Error should clearly state query sets don't match
        error_msg = str(context.exception).lower()
        self.assertIn("query", error_msg)
        # Should indicate sets don't match (either "mismatch" or "must match")
        self.assertTrue(
            "mismatch" in error_msg or "must match" in error_msg,
            msg=f"Should state query sets don't match. Got: {error_msg}"
        )

    def test_rejects_duplicate_pair_id_within_query(self):
        """
        CRITICAL: If B has duplicate pair_id within a query,
        paired bootstrap MUST reject it.
        
        Current bug: set(pair_ids) hides duplicates, passes validation.
        Expected: Raise ValueError about duplicate pair_id.
        """
        from scripts.evaluate_ranking import paired_bootstrap_comparison

        predictions_a = [
            {"pair_id": "q1::m1", "query_id": "q1", "label": 1, "score": 0.9},
            {"pair_id": "q1::m2", "query_id": "q1", "label": 0, "score": 0.5},
        ]

        # B has q1::m1 duplicated within q1
        predictions_b = [
            {"pair_id": "q1::m1", "query_id": "q1", "label": 1, "score": 0.95},
            {"pair_id": "q1::m1", "query_id": "q1", "label": 1, "score": 0.94},  # DUPLICATE!
            {"pair_id": "q1::m2", "query_id": "q1", "label": 0, "score": 0.55},
        ]

        with self.assertRaises(ValueError) as context:
            paired_bootstrap_comparison(
                predictions_a, predictions_b, "score", "score", n_resamples=100, seed=42
            )

        error_msg = str(context.exception).lower()
        self.assertIn("duplicate", error_msg)
        self.assertIn("pair_id", error_msg)

    def test_rejects_pair_id_appearing_in_multiple_queries(self):
        """
        CRITICAL: If the same pair_id appears in multiple queries,
        this violates the (query_id, pair_id) uniqueness constraint.
        
        Expected: Raise ValueError about pair_id cross-query violation.
        
        NOTE: The new pair_id prefix validation (::) will catch this even earlier,
        since 'shared::m1' has prefix 'shared' that doesn't match query_id.
        Both validation paths are correct; prefix check is just stricter.
        """
        from scripts.evaluate_ranking import paired_bootstrap_comparison

        # A has pair "shared::m1" in both q1 and q2
        predictions_a = [
            {"pair_id": "shared::m1", "query_id": "q1", "label": 1, "score": 0.9},
            {"pair_id": "q1::m2", "query_id": "q1", "label": 0, "score": 0.5},
            {"pair_id": "shared::m1", "query_id": "q2", "label": 1, "score": 0.8},  # Same pair_id!
            {"pair_id": "q2::m2", "query_id": "q2", "label": 0, "score": 0.6},
        ]

        predictions_b = [
            {"pair_id": "shared::m1", "query_id": "q1", "label": 1, "score": 0.95},
            {"pair_id": "q1::m2", "query_id": "q1", "label": 0, "score": 0.55},
            {"pair_id": "shared::m1", "query_id": "q2", "label": 1, "score": 0.85},
            {"pair_id": "q2::m2", "query_id": "q2", "label": 0, "score": 0.65},
        ]

        with self.assertRaises(ValueError) as context:
            paired_bootstrap_comparison(
                predictions_a, predictions_b, "score", "score", n_resamples=100, seed=42
            )

        error_msg = str(context.exception).lower()
        self.assertIn("pair_id", error_msg)
        # Accept either cross-query violation OR prefix mismatch (both are correct)
        # The prefix check now catches this case first
        self.assertTrue(
            "multiple" in error_msg or "cross" in error_msg or "queries" in error_msg 
            or "prefix" in error_msg or "does not match" in error_msg,
            msg="Should indicate pair_id/query_id constraint violation"
        )

    def test_rejects_missing_required_fields(self):
        """
        Test that missing basic fields (query_id, pair_id, label) are caught upfront.
        """
        from scripts.evaluate_ranking import paired_bootstrap_comparison

        # Missing query_id
        predictions_missing_query = [
            {"pair_id": "q1::m1", "label": 1, "score": 0.9},  # No query_id
        ]
        predictions_valid = [
            {"pair_id": "q1::m1", "query_id": "q1", "label": 1, "score": 0.95},
        ]

        with self.assertRaises((KeyError, ValueError)):
            paired_bootstrap_comparison(
                predictions_missing_query, predictions_valid, "score", "score"
            )

        # Missing pair_id
        predictions_missing_pair = [
            {"query_id": "q1", "label": 1, "score": 0.9},  # No pair_id
        ]

        with self.assertRaises((KeyError, ValueError)):
            paired_bootstrap_comparison(
                predictions_missing_pair, predictions_valid, "score", "score"
            )

        # Missing label
        predictions_missing_label = [
            {"pair_id": "q1::m1", "query_id": "q1", "score": 0.9},  # No label
        ]

        with self.assertRaises((KeyError, ValueError)):
            paired_bootstrap_comparison(
                predictions_missing_label, predictions_valid, "score", "score"
            )


if __name__ == "__main__":
    unittest.main()
