"""Unit tests for LIFE completeness detection statistics."""

import unittest

import astropy.units as u
import numpy as np
from unittest.mock import Mock

from tests.SurveySimulation.test_LIFESurveySimulation import _detection_case
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

    def test_detection_times_from_spectral_rates(self):
        """Check integration times and downstream detection statistics."""
        signal = (
            np.array(
                [
                    [[5.0, 0.0], [0.0, 0.0], [1.0, -1.0]],
                    [[0.0, 0.0], [2.0, 0.0], [0.0, 0.0]],
                ]
            )
            / u.s
        )
        variance = np.ones((2, 3, 2)) / u.s

        times = LIFECompleteness.detection_times_from_rates(
            signal, variance, snr_threshold=5.0
        )

        self.assertEqual(times.shape, (2, 3))
        self.assertTrue(times.unit.is_equivalent(u.s))

        expected = np.array([[1.0, np.inf, 12.5], [np.inf, 6.25, np.inf]])
        np.testing.assert_allclose(times.to_value(u.s), expected)

        result = LIFECompleteness.detection_statistics(times, 2 * u.s)
        self.assertAlmostEqual(result["expected_detections"], 0.5)
        self.assertAlmostEqual(result["probability_at_least_one"], 0.5)

    def test_signed_spectral_signals(self):
        """Verify that positive and negative chopped signals do not cancel."""
        signal = np.array([[[3.0, -3.0]]]) / u.s
        variance = np.ones((1, 1, 2)) / u.s

        times = LIFECompleteness.detection_times_from_rates(
            signal, variance, snr_threshold=3.0
        )

        self.assertAlmostEqual(times[0, 0].to_value(u.s), 0.5)

    def test_unreachable_systematic_floor(self):
        """Represent an unreachable detection threshold with infinity."""
        signal = np.ones((1, 1, 2)) / u.s
        variance = np.ones((1, 1, 2)) / u.s
        systematic = np.ones((1, 1, 2)) / u.s

        times = LIFECompleteness.detection_times_from_rates(
            signal,
            variance,
            snr_threshold=5.0,
            systematic_rates=systematic,
        )

        self.assertTrue(np.isposinf(times[0, 0].value))

    def test_empty_planet_slots_from_rates(self):
        """Allow simulated universes containing no planetary slots."""
        signal = np.empty((3, 0, 2)) / u.s
        variance = np.empty((3, 0, 2)) / u.s

        times = LIFECompleteness.detection_times_from_rates(
            signal, variance, snr_threshold=5.0
        )

        self.assertEqual(times.shape, (3, 0))
        result = LIFECompleteness.detection_statistics(times, 10 * u.h)
        self.assertEqual(result["expected_detections"], 0.0)

    def test_invalid_spectral_rate_inputs(self):
        """Reject incorrect dimensions, units and noise-rate shapes."""
        good_signal = np.ones((1, 1, 2)) / u.s
        good_variance = np.ones((1, 1, 2)) / u.s

        with self.assertRaises(TypeError):
            LIFECompleteness.detection_times_from_rates(
                good_signal.value, good_variance, 5.0
            )

        with self.assertRaises(ValueError):
            LIFECompleteness.detection_times_from_rates(
                np.ones((1, 2)) / u.s,
                np.ones((1, 2)) / u.s,
                5.0,
            )

        with self.assertRaises(ValueError):
            LIFECompleteness.detection_times_from_rates(
                good_signal,
                np.ones((1, 1, 1)) / u.s,
                5.0,
            )

        with self.assertRaises(ValueError):
            LIFECompleteness.detection_times_from_rates(good_signal, good_variance, 0.0)

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

    @staticmethod
    def make_planet_universe(n_planets):
        """Create correctly shaped synthetic planet inputs."""
        return {
            "planet_sky_positions": np.zeros((n_planets, 2)) * u.mas,
            "planet_spectral_fluxes": (
                np.ones((n_planets, 2)) * u.photon / (u.s * u.m**2 * u.um)
            ),
        }

    def test_survey_detection_times_and_statistics(self):
        """Combine variable planet counts across independent universes."""
        survey = Mock()
        survey.get_life_detection_metrics.side_effect = [
            {"required_integration_time": np.array([1.0, np.nan]) * u.h},
            {"required_integration_time": np.array([2.0]) * u.h},
        ]

        observing = {
            "sInd": 0,
            "wavelengths": np.array([10.0, 12.0]) * u.um,
            "nEZ": 1.0,
        }
        universes = [
            self.make_planet_universe(2),
            self.make_planet_universe(1),
            self.make_planet_universe(0),
        ]

        times = LIFECompleteness.detection_times_from_survey(
            survey, observing, universes, 2 * u.h
        )

        self.assertEqual(times.shape, (3, 2))
        np.testing.assert_allclose(
            times.to_value(u.h),
            [[1.0, np.inf], [2.0, np.inf], [np.inf, np.inf]],
        )
        self.assertEqual(survey.get_life_detection_metrics.call_count, 2)

        for call in survey.get_life_detection_metrics.call_args_list:
            self.assertEqual(call.kwargs["integration_time"].to_value(u.h), 2.0)

        result = LIFECompleteness.detection_statistics(times, 2 * u.h)
        self.assertAlmostEqual(result["expected_detections"], 2 / 3)
        self.assertAlmostEqual(result["probability_at_least_one"], 2 / 3)

    def test_all_universes_without_planets(self):
        """Return an empty planet axis without running instrument physics."""
        survey = Mock()
        observing = {"wavelengths": np.array([10.0, 12.0]) * u.um}
        universes = [
            self.make_planet_universe(0),
            self.make_planet_universe(0),
        ]

        times = LIFECompleteness.detection_times_from_survey(
            survey, observing, universes, 1 * u.h
        )

        self.assertEqual(times.shape, (2, 0))
        survey.get_life_detection_metrics.assert_not_called()

    def test_invalid_survey_population(self):
        """Reject absent universes and inconsistent spectral dimensions."""
        survey = Mock()
        observing = {"wavelengths": np.array([10.0, 12.0]) * u.um}

        with self.assertRaises(ValueError):
            LIFECompleteness.detection_times_from_survey(survey, observing, [], 1 * u.h)

        invalid = self.make_planet_universe(1)
        invalid["planet_spectral_fluxes"] = (
            np.ones((1, 3)) * u.photon / (u.s * u.m**2 * u.um)
        )

        with self.assertRaisesRegex(ValueError, "Planet spectra"):
            LIFECompleteness.detection_times_from_survey(
                survey, observing, [invalid], 1 * u.h
            )

    def test_invalid_survey_detection_time(self):
        """Reject invalid detection times returned by SurveySimulation."""
        survey = Mock()
        survey.get_life_detection_metrics.return_value = {
            "required_integration_time": np.array([-1.0]) * u.s
        }

        observing = {"wavelengths": np.array([10.0, 12.0]) * u.um}

        with self.assertRaisesRegex(ValueError, "invalid detection times"):
            LIFECompleteness.detection_times_from_survey(
                survey,
                observing,
                [self.make_planet_universe(1)],
                1 * u.h,
            )

    def test_real_survey_detection_integration(self):
        """Reuse the existing physical LIFE SurveySimulation test case."""
        survey, inputs = _detection_case()

        observation_time = inputs["integration_time"]
        observing = {
            key: value
            for key, value in inputs.items()
            if key
            not in (
                "integration_time",
                "planet_sky_positions",
                "planet_spectral_fluxes",
            )
        }
        universe = {
            "planet_sky_positions": inputs["planet_sky_positions"],
            "planet_spectral_fluxes": inputs["planet_spectral_fluxes"],
        }

        actual = LIFECompleteness.detection_times_from_survey(
            survey, observing, [universe], observation_time
        )

        expected = survey.get_life_detection_metrics(**inputs)[
            "required_integration_time"
        ]

        self.assertEqual(actual.shape, (1, 2))
        np.testing.assert_allclose(
            actual[0].to_value(u.s),
            np.where(
                np.isnan(expected.to_value(u.s)),
                np.inf,
                expected.to_value(u.s),
            ),
        )


if __name__ == "__main__":
    unittest.main()
