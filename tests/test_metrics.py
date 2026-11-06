
import unittest

from src.metrics import (
    compute_metrics,
    compute_success,
    compute_degradation,
    aggregate_seeds,
)


class TestEvaluationMetrics(unittest.TestCase):

    def test_average_return_and_success_rate(self):
        result = compute_metrics(
            episode_returns=[10, 20, 30],
            success_flags=[True, False, True],
        )

        self.assertAlmostEqual(result["average_return"], 20.0)
        self.assertAlmostEqual(result["success_rate"], 2 / 3)

    def test_success_threshold(self):
        self.assertTrue(compute_success(0.03))
        self.assertFalse(compute_success(0.05))
        self.assertFalse(compute_success(0.08))

    def test_performance_degradation(self):
        result = compute_degradation(-10, -15)

        self.assertAlmostEqual(result["absolute_drop"], 5.0)
        self.assertAlmostEqual(result["relative_drop"], 0.5)

    def test_seed_aggregation(self):
        result = aggregate_seeds(
            seed_mean_returns=[-10, -12, -14],
            seed_success_rates=[0.8, 0.7, 0.6],
        )

        self.assertAlmostEqual(result["mean_return"], -12.0)
        self.assertAlmostEqual(result["mean_success_rate"], 0.7)

    def test_empty_inputs(self):
        with self.assertRaises(ValueError):
            compute_metrics([], [])

    def test_mismatched_inputs(self):
        with self.assertRaises(ValueError):
            compute_metrics([10, 20], [True])

    def test_zero_clean_return(self):
        result = compute_degradation(0, -5)

        self.assertEqual(result["absolute_drop"], 5.0)
        self.assertIsNone(result["relative_drop"])


if __name__ == "__main__":
    unittest.main()
