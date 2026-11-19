"""Unit tests for LIFE completeness detection statistics."""

import unittest

import astropy.units as u
import numpy as np

from EXOSIMS.Completeness.LIFECompleteness import LIFECompleteness
from EXOSIMS.Prototypes.Completeness import Completeness


class TestLIFECompleteness(unittest.TestCase):
    """Validate LIFE Monte Carlo detection statistics."""

    def setUp(self):
        """Construct four simulated planetary systems."""
        self.times = (
            np.array(
                [
                    [1.0, 2.0, np.inf],
                    [np.inf, np.inf, np.inf],
                    [0.5, 2.5, 4.0],
                    [np.inf, np.inf, np.inf],
                ]
            )
            * u.h
        )

    def test_exosims_inheritance(self):
        """Use the standard EXOSIMS Completeness module type."""
        self.assertTrue(issubclass(LIFECompleteness, Completeness))
        self.assertEqual(LIFECompleteness._modtype, "Completeness")

    def test_detection_statistics(self):
        """Distinguish expected yield and at-least-one probability."""
        result = LIFECompleteness.detection_statistics(self.times, 2 * u.h)
        self.assertAlmostEqual(result["expected_detections"], 0.75)
        self.assertAlmostEqual(result["probability_at_least_one"], 0.5)

    def test_inclusive_threshold(self):
        """Count planets detectable exactly at the time budget."""
        result = LIFECompleteness.detection_statistics(self.times, 4 * u.h)
        self.assertAlmostEqual(result["expected_detections"], 1.25)
        self.assertAlmostEqual(result["probability_at_least_one"], 0.5)

    def test_unit_conversion(self):
        """Verify consistent results for hours and seconds."""
        hours = LIFECompleteness.detection_statistics(self.times, 2 * u.h)
        seconds = LIFECompleteness.detection_statistics(self.times.to(u.s), 7200 * u.s)
        self.assertEqual(hours, seconds)

    def test_zero_observation_time(self):
        """Require positive observing time for a detection."""
        result = LIFECompleteness.detection_statistics(self.times, 0 * u.h)
        self.assertEqual(result["expected_detections"], 0.0)
        self.assertEqual(result["probability_at_least_one"], 0.0)

    def test_empty_planet_population(self):
        """Handle universes containing no planets."""
        times = np.empty((3, 0)) * u.h
        result = LIFECompleteness.detection_statistics(times, 10 * u.h)
        self.assertEqual(result["expected_detections"], 0.0)
        self.assertEqual(result["probability_at_least_one"], 0.0)

    def test_invalid_detection_times(self):
        """Reject nonphysical detection-time arrays."""
        invalid = [
            np.array([1.0, 2.0]) * u.h,
            np.empty((0, 2)) * u.h,
            np.array([[np.nan]]) * u.h,
            np.array([[-np.inf]]) * u.h,
            np.array([[-1.0]]) * u.h,
            np.array([[0.0]]) * u.h,
        ]

        for times in invalid:
            with self.subTest(shape=times.shape):
                with self.assertRaises(ValueError):
                    LIFECompleteness.detection_statistics(times, 2 * u.h)

    def test_invalid_budget(self):
        """Reject negative, infinite, and vector observation times."""
        invalid = [
            -1 * u.h,
            np.inf * u.h,
            np.nan * u.h,
            np.array([1.0, 2.0]) * u.h,
        ]

        for budget in invalid:
            with self.subTest(budget=budget):
                with self.assertRaises(ValueError):
                    LIFECompleteness.detection_statistics(self.times, budget)

    def test_units_required(self):
        """Reject detection times and budgets without physical units."""
        with self.assertRaises(TypeError):
            LIFECompleteness.detection_statistics(self.times.value, 2 * u.h)

        with self.assertRaises(TypeError):
            LIFECompleteness.detection_statistics(self.times, 2)

    def test_unimplemented_interfaces_are_blocked(self):
        """Prevent fallback to artificial prototype completeness."""
        comp = object.__new__(LIFECompleteness)

        calls = [
            ("target_completeness", (None,)),
            ("comp_per_intTime", (None,) * 7),
            ("completeness_update", (None,) * 4),
            ("comp_calc", (None,) * 3),
            ("dcomp_dt", (None,) * 7),
        ]

        for method, args in calls:
            with self.subTest(method=method):
                with self.assertRaises(NotImplementedError):
                    getattr(comp, method)(*args)


if __name__ == "__main__":
    unittest.main()
