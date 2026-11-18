"""Geometry and EXOSIMS integration tests for the LIFE observatory."""

import tempfile
import unittest

import astropy.units as u
import numpy as np

from EXOSIMS.Observatory.LIFEObservatory import LIFEObservatory
from EXOSIMS.Observatory.ObservatoryL2Halo import ObservatoryL2Halo
from EXOSIMS.OpticalSystem.LIFEOpticalSystem import LIFEOpticalSystem
from EXOSIMS.Prototypes.Observatory import Observatory


class TestLIFEObservatory(unittest.TestCase):
    """Validate LIFE observatory geometry and integration with EXOSIMS."""

    @classmethod
    def setUpClass(cls):
        """Construct shared observatories using static ephemerides."""
        super().setUpClass()
        cls._cache_dir = tempfile.TemporaryDirectory(prefix="life_observatory_")
        try:
            cls.observatory = LIFEObservatory(
                cachedir=cls._cache_dir.name,
                forceStaticEphem=True,
                verbose=False,
            )
            cls.bounded_observatory = LIFEObservatory(
                nulling_baseline=14.5 * u.m,
                baseline_ratio=6.0,
                nulling_baseline_range=(10.0 * u.m, 30.0 * u.m),
                imaging_baseline_range=(60.0 * u.m, 180.0 * u.m),
                cachedir=cls.observatory.cachedir,
                forceStaticEphem=True,
                verbose=False,
            )
        except BaseException:
            cls._cache_dir.cleanup()
            raise

    @classmethod
    def tearDownClass(cls):
        """Release the temporary EXOSIMS cache."""
        cls._cache_dir.cleanup()
        super().tearDownClass()

    def test_reference_geometry_and_collector_order(self):
        """Check all four reference coordinates without permitting reordering."""
        observatory = self.observatory
        positions = observatory.get_collector_positions(0.0 * u.deg)
        expected = np.array(
            [
                [-7.25, -43.5],
                [-7.25, 43.5],
                [7.25, -43.5],
                [7.25, 43.5],
            ]
        )
        self.assertIsInstance(positions, u.Quantity)
        self.assertTrue(positions.unit == u.m)
        self.assertTrue(positions.shape == (1, 4, 2))
        np.testing.assert_allclose(positions[0].to_value(u.m), expected, atol=1e-14)
        np.testing.assert_allclose(positions.sum(axis=1).to_value(u.m), 0, atol=1e-14)

    def test_reference_baselines_and_outspec(self):
        """Verify the nulling and imaging baselines and serializable settings."""
        observatory = self.observatory
        coords = observatory.get_collector_positions()[0]
        np.testing.assert_allclose(
            np.linalg.norm((coords[0] - coords[2]).to_value(u.m)), 14.5
        )
        np.testing.assert_allclose(
            np.linalg.norm((coords[0] - coords[1]).to_value(u.m)), 87.0
        )
        self.assertTrue(observatory.nulling_baseline == 14.5 * u.m)
        self.assertTrue(observatory.imaging_baseline == 87.0 * u.m)
        self.assertAlmostEqual(observatory.baseline_ratio, 6.0)
        self.assertAlmostEqual(observatory._outspec["nulling_baseline"], 14.5)
        self.assertAlmostEqual(observatory._outspec["baseline_ratio"], 6.0)

    def test_configurable_baselines_and_overheads(self):
        """Check unit conversion, ratio configuration, and EXOSIMS settling time."""
        observatory = self.observatory
        custom = LIFEObservatory(
            nulling_baseline=2000 * u.cm,
            baseline_ratio=3.0,
            settlingTime=0.125,
            cachedir=observatory.cachedir,
            forceStaticEphem=True,
            verbose=False,
        )
        expected = np.array([[-10, -30], [-10, 30], [10, -30], [10, 30]])
        np.testing.assert_allclose(
            custom.get_collector_positions()[0].to_value(u.m), expected
        )
        self.assertTrue(custom.nulling_baseline == 20 * u.m)
        self.assertTrue(custom.imaging_baseline == 60 * u.m)
        self.assertTrue(custom.settlingTime == 0.125 * u.d)
        self.assertTrue(observatory.settlingTime == 0 * u.d)

    def test_rotation_zero_ninety_and_full_turn(self):
        """A positive 90-degree roll maps (x, y) to (-y, x)."""
        observatory = self.observatory
        positions = observatory.get_collector_positions(
            np.array([0.0, 90.0, 360.0]) * u.deg
        ).to_value(u.m)
        self.assertTrue(positions.shape == (3, 4, 2))
        ninety_expected = np.column_stack((-positions[0, :, 1], positions[0, :, 0]))
        np.testing.assert_allclose(positions[1], ninety_expected, atol=1e-12)
        np.testing.assert_allclose(positions[2], positions[0], atol=1e-12)

    def test_scalar_and_vector_rotations(self):
        """Single and batched rotations use the same three-dimensional shape."""
        observatory = self.observatory
        scalar = observatory.get_collector_positions(30.0 * u.deg)
        vector = observatory.get_collector_positions(np.array([30.0]) * u.deg)
        self.assertTrue(scalar.shape == vector.shape == (1, 4, 2))
        np.testing.assert_allclose(scalar.to_value(u.m), vector.to_value(u.m))

    def test_rotation_preserves_all_pairwise_distances(self):
        """Rigid planar rotations must preserve all six collector separations."""
        observatory = self.observatory
        positions = observatory.get_collector_positions(
            np.array([0.0, 23.0, 90.0, 180.0, 359.0]) * u.deg
        ).to_value(u.m)
        differences = positions[:, :, None, :] - positions[:, None, :, :]
        separations = np.linalg.norm(differences, axis=-1)
        np.testing.assert_allclose(
            separations, np.broadcast_to(separations[0], separations.shape), atol=1e-12
        )

    def test_invalid_nulling_baselines(self):
        """Reject invalid lengths before running the halo-orbit constructor."""
        cases = [
            (0.0, ValueError),
            (-14.5, ValueError),
            (np.nan, ValueError),
            (np.inf, ValueError),
            (2.0 * u.s, ValueError),
            (np.array([14.5, 20.0]) * u.m, ValueError),
            ("14.5", TypeError),
            (True, TypeError),
        ]
        for baseline, error in cases:
            with self.subTest(baseline=baseline, error=error):
                with self.assertRaises(error):
                    LIFEObservatory(nulling_baseline=baseline)

    def test_invalid_baseline_ratios(self):
        """Reject invalid ratios before running the halo-orbit constructor."""
        cases = [
            (0.0, ValueError),
            (-6.0, ValueError),
            (np.nan, ValueError),
            (np.inf, ValueError),
            ("6", TypeError),
            (6.0 * u.m, TypeError),
            (False, TypeError),
        ]
        for ratio, error in cases:
            with self.subTest(ratio=ratio, error=error):
                with self.assertRaises(error):
                    LIFEObservatory(baseline_ratio=ratio)

    def test_invalid_rotation_inputs(self):
        """Reject ambiguous units, malformed shapes, and nonfinite angles."""
        observatory = self.observatory
        cases = [
            (90.0, TypeError),
            (90.0 * u.m, ValueError),
            (np.array([]) * u.deg, ValueError),
            (np.array([0.0, np.nan]) * u.deg, ValueError),
            (np.array([0.0, np.inf]) * u.deg, ValueError),
            (np.zeros((2, 2)) * u.deg, ValueError),
        ]
        for angles, error in cases:
            with self.subTest(angles=angles, error=error):
                with self.assertRaises(error):
                    observatory.get_collector_positions(angles)

    def test_inherited_orbit_and_keepout_interfaces(self):
        """Verify that original observatory behavior remains inherited."""
        observatory = self.observatory
        self.assertIsInstance(observatory, ObservatoryL2Halo)
        self.assertIsInstance(observatory, Observatory)
        self.assertIs(LIFEObservatory.orbit, ObservatoryL2Halo.orbit)
        self.assertIs(LIFEObservatory.keepout, Observatory.keepout)
        self.assertIs(
            LIFEObservatory.calculate_observableTimes,
            Observatory.calculate_observableTimes,
        )

    def test_geometry_with_life_transmission(self):
        """Verify geometry shape, null, and nonzero chopped off-axis response."""
        observatory = self.observatory
        optics = LIFEOpticalSystem.__new__(LIFEOpticalSystem)
        positions = observatory.get_collector_positions(
            np.array([0.0, 60.0, 120.0]) * u.deg
        )
        sky = np.array([[0.0, 0.0], [4.848e-7, 1.0e-7]]) * u.rad
        wavelengths = np.array([10.0, 12.0]) * u.um

        left, right, chopped = optics.get_transmission(positions, wavelengths, sky)
        for output in (left, right, chopped):
            self.assertTrue(output.shape == (3, 2, 2))
            self.assertTrue(output.unit == u.dimensionless_unscaled)
            np.testing.assert_allclose(output[:, 0, :].value, 0, atol=1e-14)
        np.testing.assert_allclose(chopped.value, (left - right).value, atol=1e-14)
        self.assertTrue(np.any(np.abs(chopped[:, 1, :].value) > 1e-05))

    def test_per_observation_baseline_override_without_limits(self):
        """Use a different geometry without changing the Observatory defaults."""
        observatory = self.observatory
        positions = observatory.get_collector_positions(
            np.array([0.0, 90.0]) * u.deg,
            nulling_baseline=25.0 * u.m,
            baseline_ratio=5.0,
        )
        expected = np.array(
            [
                [-12.5, -62.5],
                [-12.5, 62.5],
                [12.5, -62.5],
                [12.5, 62.5],
            ]
        )
        self.assertTrue(positions.shape == (2, 4, 2))
        np.testing.assert_allclose(positions[0].to_value(u.m), expected)
        np.testing.assert_allclose(
            positions[1].to_value(u.m),
            np.column_stack((-expected[:, 1], expected[:, 0])),
            atol=1e-12,
        )
        self.assertTrue(observatory.nulling_baseline == 14.5 * u.m)
        self.assertTrue(observatory.imaging_baseline == 87.0 * u.m)
        self.assertTrue(observatory.baseline_ratio == 6.0)

    def test_baseline_range_configuration_and_endpoints(self):
        """Accept inclusive range boundaries and record them in the outspec."""
        bounded_observatory = self.bounded_observatory
        obs = bounded_observatory
        self.assertTrue(obs.nulling_baseline_range == (10.0 * u.m, 30.0 * u.m))
        self.assertTrue(obs.imaging_baseline_range == (60.0 * u.m, 180.0 * u.m))
        self.assertTrue(obs._outspec["nulling_baseline_range"] == [10.0, 30.0])
        self.assertTrue(obs._outspec["imaging_baseline_range"] == [60.0, 180.0])
        for value in (10.0, 30.0):
            positions = obs.get_collector_positions(nulling_baseline=value * u.m)
            self.assertTrue(positions.shape == (1, 4, 2))
            np.testing.assert_allclose(
                np.abs(positions[0, 0, 0].to_value(u.m)), value / 2.0
            )

    def test_dynamic_ratio_checks_imaging_limit(self):
        """The imaging constraint applies to the product of ratio and baseline."""
        bounded_observatory = self.bounded_observatory
        obs = bounded_observatory
        positions = obs.get_collector_positions(
            nulling_baseline=25 * u.m, baseline_ratio=5.0
        )
        np.testing.assert_allclose(
            positions[0, :, 1].to_value(u.m),
            [-62.5, 62.5, -62.5, 62.5],
        )
        with self.assertRaisesRegex(ValueError, "imaging baseline"):
            obs.get_collector_positions(nulling_baseline=25 * u.m, baseline_ratio=8.0)

    def test_reject_out_of_range_nulling_baselines(self):
        """Reject unphysical or unavailable nulling baselines at call time."""
        bounded_observatory = self.bounded_observatory
        cases = [
            9.0 * u.m,
            31.0 * u.m,
            0.0 * u.m,
            np.nan * u.m,
        ]
        for override in cases:
            with self.subTest(override=override):
                with self.assertRaises(ValueError):
                    bounded_observatory.get_collector_positions(
                        nulling_baseline=override
                    )

    def test_reject_invalid_per_observation_configuration(self):
        """Validate optional parameters on every geometry request."""
        observatory = self.observatory
        cases = [
            {"nulling_baseline": "20"},
            {"nulling_baseline": 20 * u.s},
            {"baseline_ratio": -2.0},
            {"baseline_ratio": float("inf")},
            {"baseline_ratio": "6"},
        ]
        for overrides in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaises((ValueError, TypeError)):
                    observatory.get_collector_positions(**overrides)

    def test_reject_invalid_baseline_ranges(self):
        """Reject malformed ranges or defaults outside configured bounds."""
        cases = [
            {"nulling_baseline_range": (30.0, 10.0)},
            {"nulling_baseline_range": (0.0, 20.0)},
            {"nulling_baseline_range": (10.0, float("inf"))},
            {"nulling_baseline_range": (10.0,)},
            {"nulling_baseline_range": (10.0 * u.s, 20.0 * u.m)},
            {"nulling_baseline_range": (20.0, 30.0)},
            {"imaging_baseline_range": (100.0, 200.0)},
        ]
        for settings in cases:
            with self.subTest(settings=settings):
                with self.assertRaises((ValueError, TypeError)):
                    LIFEObservatory(**settings)

    def test_baseline_override_changes_optical_response(self):
        """Changing the requested geometry changes off-axis transmission."""
        bounded_observatory = self.bounded_observatory
        obs = bounded_observatory
        optical_system = LIFEOpticalSystem.__new__(LIFEOpticalSystem)
        kwargs = {
            "wavelengths": 10.0 * u.um,
            "sky_positions": np.array([100.0, 20.0]) * u.mas,
        }
        small = obs.get_collector_positions(nulling_baseline=10.0 * u.m)
        large = obs.get_collector_positions(nulling_baseline=30.0 * u.m)
        left_small, _, _ = optical_system.get_transmission(small, **kwargs)
        left_large, _, _ = optical_system.get_transmission(large, **kwargs)
        self.assertTrue(left_small.shape == left_large.shape == (1, 1, 1))
        self.assertFalse(np.allclose(left_small.value, left_large.value, atol=1e-08))

    def test_default_baseline_operating_range(self):
        """Verify the default 10–100 m nulling-baseline operating range."""
        observatory = self.observatory
        obs = observatory

        self.assertTrue(obs.nulling_baseline == 14.5 * u.m)
        self.assertAlmostEqual(obs.baseline_ratio, 6.0)
        self.assertTrue(obs.imaging_baseline == 87.0 * u.m)
        self.assertTrue(obs.nulling_baseline_range == (10.0 * u.m, 100.0 * u.m))
        self.assertIsNone(obs.imaging_baseline_range)
        self.assertTrue(obs._outspec["nulling_baseline_range"] == [10.0, 100.0])

        # Both endpoints must be permitted.
        for baseline in (10.0, 100.0):
            positions = obs.get_collector_positions(nulling_baseline=baseline * u.m)
            self.assertTrue(positions.shape == (1, 4, 2))
            np.testing.assert_allclose(
                abs(positions[0, 0, 0].to_value(u.m)),
                baseline / 2.0,
            )

    def test_per_observation_baseline_override(self):
        """Change the array configuration without changing stored defaults."""
        observatory = self.observatory
        positions = observatory.get_collector_positions(
            rotation_angles=np.array([0.0, 90.0]) * u.deg,
            nulling_baseline=25.0 * u.m,
            baseline_ratio=5.0,
        )

        expected = np.array(
            [
                [-12.5, -62.5],
                [-12.5, 62.5],
                [12.5, -62.5],
                [12.5, 62.5],
            ]
        )

        self.assertTrue(positions.shape == (2, 4, 2))
        np.testing.assert_allclose(positions[0].to_value(u.m), expected)
        np.testing.assert_allclose(
            positions[1].to_value(u.m),
            np.column_stack((-expected[:, 1], expected[:, 0])),
            atol=1e-12,
        )

        # The stored reference configuration must remain unchanged.
        self.assertTrue(observatory.nulling_baseline == 14.5 * u.m)
        self.assertTrue(observatory.imaging_baseline == 87.0 * u.m)
        self.assertTrue(observatory.baseline_ratio == 6.0)

    def test_explicitly_disable_default_baseline_range(self):
        """Setting None must disable the default nulling-baseline limit."""
        observatory = self.observatory
        obs = LIFEObservatory(
            nulling_baseline_range=None,
            cachedir=observatory.cachedir,
            forceStaticEphem=True,
            verbose=False,
        )

        self.assertIsNone(obs.nulling_baseline_range)
        self.assertIsNone(obs._outspec["nulling_baseline_range"])

        positions = obs.get_collector_positions(nulling_baseline=120.0 * u.m)
        self.assertTrue(positions.shape == (1, 4, 2))
        np.testing.assert_allclose(abs(positions[0, 0, 0].to_value(u.m)), 60.0)

    def test_custom_baseline_ranges_and_endpoints(self):
        """Accept both inclusive boundaries of configured baseline intervals."""
        bounded_observatory = self.bounded_observatory
        obs = bounded_observatory

        self.assertTrue(obs.nulling_baseline_range == (10.0 * u.m, 30.0 * u.m))
        self.assertTrue(obs.imaging_baseline_range == (60.0 * u.m, 180.0 * u.m))
        self.assertTrue(obs._outspec["nulling_baseline_range"] == [10.0, 30.0])
        self.assertTrue(obs._outspec["imaging_baseline_range"] == [60.0, 180.0])

        for baseline in (10.0, 30.0):
            positions = obs.get_collector_positions(nulling_baseline=baseline * u.m)

            self.assertTrue(positions.shape == (1, 4, 2))
            np.testing.assert_allclose(
                abs(positions[0, 0, 0].to_value(u.m)),
                baseline / 2.0,
            )
            np.testing.assert_allclose(
                abs(positions[0, 0, 1].to_value(u.m)),
                baseline * 6.0 / 2.0,
            )

    def test_dynamic_ratio_respects_imaging_limits(self):
        """Check the imaging baseline after applying the requested ratio."""
        bounded_observatory = self.bounded_observatory
        obs = bounded_observatory

        positions = obs.get_collector_positions(
            nulling_baseline=25.0 * u.m,
            baseline_ratio=5.0,
        )

        np.testing.assert_allclose(
            positions[0, :, 1].to_value(u.m),
            [-62.5, 62.5, -62.5, 62.5],
        )

        # 25 m * 8 = 200 m, exceeding the 180 m imaging limit.
        with self.assertRaisesRegex(ValueError, "imaging baseline"):
            obs.get_collector_positions(
                nulling_baseline=25.0 * u.m,
                baseline_ratio=8.0,
            )

    def test_reject_nulling_baselines_outside_custom_range(self):
        """Reject baselines outside the limits or physically invalid values."""
        bounded_observatory = self.bounded_observatory
        cases = [
            9.0 * u.m,
            31.0 * u.m,
            0.0 * u.m,
            np.nan * u.m,
        ]
        for baseline in cases:
            with self.subTest(baseline=baseline):
                with self.assertRaises(ValueError):
                    bounded_observatory.get_collector_positions(
                        nulling_baseline=baseline
                    )

    def test_reject_invalid_dynamic_baseline_inputs(self):
        """Reject unsupported types, units, and ratios during observations."""
        observatory = self.observatory
        cases = [
            {"nulling_baseline": "20"},
            {"nulling_baseline": 20.0 * u.s},
            {"baseline_ratio": -2.0},
            {"baseline_ratio": float("inf")},
            {"baseline_ratio": "6"},
        ]
        for overrides in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaises((ValueError, TypeError)):
                    observatory.get_collector_positions(**overrides)

    def test_reject_invalid_baseline_range_configuration(self):
        """Reject malformed ranges and defaults outside permitted limits."""
        cases = [
            {"nulling_baseline_range": (30.0, 10.0)},
            {"nulling_baseline_range": (0.0, 20.0)},
            {"nulling_baseline_range": (10.0, float("inf"))},
            {"nulling_baseline_range": (10.0,)},
            {"nulling_baseline_range": (10.0 * u.s, 20.0 * u.m)},
            {"nulling_baseline_range": (20.0, 30.0)},
            {"imaging_baseline_range": (100.0, 200.0)},
        ]
        for settings in cases:
            with self.subTest(settings=settings):
                with self.assertRaises((ValueError, TypeError)):
                    LIFEObservatory(**settings)

    def test_baseline_override_changes_optical_transmission(self):
        """Check that different collector separations alter transmission."""
        bounded_observatory = self.bounded_observatory
        obs = bounded_observatory
        optics = LIFEOpticalSystem.__new__(LIFEOpticalSystem)

        kwargs = {
            "wavelengths": 10.0 * u.um,
            "sky_positions": np.array([100.0, 20.0]) * u.mas,
        }

        small = obs.get_collector_positions(nulling_baseline=10.0 * u.m)
        large = obs.get_collector_positions(nulling_baseline=30.0 * u.m)

        left_small, _, _ = optics.get_transmission(small, **kwargs)
        left_large, _, _ = optics.get_transmission(large, **kwargs)

        self.assertTrue(left_small.shape == (1, 1, 1))
        self.assertTrue(left_large.shape == (1, 1, 1))

        self.assertFalse(np.allclose(left_small.value, left_large.value, atol=1e-08))
