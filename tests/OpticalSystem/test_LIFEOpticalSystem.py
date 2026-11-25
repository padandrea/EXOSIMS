"""Tests for the LIFE nulling interferometer optical system."""

import json
import unittest

import numpy as np
import astropy.units as u

from EXOSIMS.OpticalSystem.LIFEOpticalSystem import LIFEOpticalSystem
from tests.TestSupport.Info import resource_path
from EXOSIMS.ZodiacalLight.Mennesson import Mennesson


def reference_collector_positions(rotation_angles):
    """Generate reference X-array coordinates for transmission tests."""
    angles = np.atleast_1d(rotation_angles.to_value(u.rad))

    # Reference geometry: 14.5 m nulling baseline, ratio 6.
    positions = np.array(
        [
            [-7.25, -43.5],
            [-7.25, 43.5],
            [7.25, -43.5],
            [7.25, 43.5],
        ]
    )

    x = positions[:, 0]
    y = positions[:, 1]

    cos_a = np.cos(angles)[:, None]
    sin_a = np.sin(angles)[:, None]

    rotated = np.stack(
        [
            x * cos_a - y * sin_a,
            x * sin_a + y * cos_a,
        ],
        axis=-1,
    )

    return rotated * u.m


def make_fov_test_optical_system():
    """Construct a simple LIFE instrument for field-of-view tests."""
    return LIFEOpticalSystem(
        pupilDiam=2.0,
        obscurFac=0.0,
        scienceInstruments=[
            {
                "name": "LIFE-spectrometer",
                "QE": 0.5,
                "optics": 0.8,
            }
        ],
        starlightSuppressionSystems=[
            {
                "name": "nuller",
                "lam": 10000,
                "optics": 0.75,
            }
        ],
    )


class TestLIFEOpticalSystem(unittest.TestCase):
    """LIFE optical-system physics and EXOSIMS integration tests."""

    def test_total_collecting_area(self):
        """Verify the combined area of four equal collectors."""
        life = LIFEOpticalSystem(
            pupilDiam=2.0,
            obscurFac=0.0,
        )

        # Each 2 m collector has an area of pi m².
        # Four collectors therefore have an area of 4*pi m².
        np.testing.assert_allclose(
            life.total_collecting_area.to_value(u.m**2),
            4.0 * np.pi,
            rtol=1e-6,
            atol=1e-12,
        )

    def test_mid_ir_observing_mode(self):
        """Verify that LIFE supports a 15 um observing mode."""
        path = resource_path("test-scripts/template_minimal.json")

        with open(path, encoding="utf-8") as file:
            specs = json.load(file)

        specs["starlightSuppressionSystems"][0]["lam"] = 15000

        optical_system = LIFEOpticalSystem(**specs)

        np.testing.assert_allclose(
            optical_system.observingModes[0]["lam"].to_value(u.um),
            15.0,
            rtol=1e-6,
            atol=1e-12,
        )

    def test_get_spectral_bins_reference(self):
        """Verify geometric spectral binning at constant resolving power."""
        life = LIFEOpticalSystem.__new__(LIFEOpticalSystem)

        centers, widths = life.get_spectral_bins(
            4.0 * u.um,
            18.5 * u.um,
            33.3,
        )

        self.assertEqual(len(centers), 51)
        self.assertEqual(len(widths), 51)

        # The first bin starts at 4 um.
        lower_edges = centers - widths / 2
        upper_edges = centers + widths / 2

        np.testing.assert_allclose(
            lower_edges[0].to_value(u.um),
            4.0,
            atol=1e-12,
        )

        # The final bin ends exactly at the upper wavelength limit.
        self.assertLess(lower_edges[-1], 18.5 * u.um)
        np.testing.assert_allclose(
            upper_edges[-1].to_value(u.um),
            18.5,
            atol=1e-12,
        )

        # Check the geometric spacing.
        ratios = lower_edges[1:] / lower_edges[:-1]

        np.testing.assert_allclose(
            ratios.to_value(u.dimensionless_unscaled),
            1.0 + 1.0 / (33.3 - 0.5),
            rtol=1e-12,
        )

        # Every complete bin must have exactly the requested
        # resolving power at its arithmetic center.
        # The final truncated bin is excluded.
        actual_resolving_power = centers[:-1] / widths[:-1]

        np.testing.assert_allclose(
            actual_resolving_power.to_value(u.dimensionless_unscaled),
            33.3,
            rtol=1e-12,
        )

        self.assertTrue(np.all(widths > 0 * u.m))

    def test_get_spectral_bins_from_mode(self):
        """Verify that spectral bins use observing-mode parameters."""
        life = LIFEOpticalSystem.__new__(LIFEOpticalSystem)

        mode = {
            "lam": 11.25 * u.um,
            "deltaLam": 14.5 * u.um,
            "inst": {
                "name": "LIFE-spectrometer",
                "Rs": 33.3,
            },
        }

        centers, widths = life.get_spectral_bins_from_mode(mode)

        expected_centers, expected_widths = life.get_spectral_bins(
            4.0 * u.um,
            18.5 * u.um,
            33.3,
        )

        np.testing.assert_allclose(
            centers.to_value(u.um),
            expected_centers.to_value(u.um),
        )

        np.testing.assert_allclose(
            widths.to_value(u.um),
            expected_widths.to_value(u.um),
        )

    def test_life_explicit_wavelength_limits(self):
        """Verify LIFE wavelength limits from JSON configuration."""
        path = resource_path("test-scripts/template_minimal.json")

        with open(path, encoding="utf-8") as file:
            specs = json.load(file)

        specs["scienceInstruments"][0]["name"] = "LIFE-spectrometer"
        specs["scienceInstruments"][0]["Rs"] = 33.3

        specs["starlightSuppressionSystems"][0]["name"] = "nuller"

        specs["observingModes"] = [
            {
                "instName": "LIFE-spectrometer",
                "systName": "nuller",
                "LIFE_lambda_min": 4000,
                "LIFE_lambda_max": 18500,
                "bandpass_model": "box",
                "SNR": 5,
            }
        ]

        life = LIFEOpticalSystem(**specs)
        mode = life.observingModes[0]

        np.testing.assert_allclose(
            mode["lam"].to_value(u.um),
            11.25,
            rtol=1e-6,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            mode["deltaLam"].to_value(u.um),
            14.5,
            rtol=1e-6,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            mode["inst"]["Rs"],
            33.3,
            rtol=1e-6,
            atol=1e-12,
        )

        centers, widths = life.get_spectral_bins_from_mode(mode)

        lower_edges = centers - widths / 2
        upper_edges = centers + widths / 2

        np.testing.assert_allclose(
            lower_edges[0].to_value(u.um),
            4.0,
            rtol=1e-6,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            upper_edges[-1].to_value(u.um),
            18.5,
            rtol=1e-6,
            atol=1e-12,
        )

    def test_life_wavelength_limits_require_both_keys(self):
        """Reject a partially specified LIFE wavelength range."""
        path = resource_path("test-scripts/template_minimal.json")

        with open(path, encoding="utf-8") as file:
            specs = json.load(file)

        specs["observingModes"] = [
            {
                "instName": "imager",
                "systName": "coronagraph",
                "LIFE_lambda_min": 4000,
            }
        ]

        with self.assertRaisesRegex(ValueError, "must be specified together"):
            LIFEOpticalSystem(**specs)

    def test_transmission_shape_and_on_axis_null(self):
        """Both nulled outputs vanish for an on-axis point source."""
        life = LIFEOpticalSystem.__new__(LIFEOpticalSystem)

        # Temporary geometry input until LIFEObservatory exists.
        positions = reference_collector_positions(np.array([0.0, 90.0]) * u.deg)

        left, right, chopped = life.get_transmission(
            positions,
            np.array([10.0, 12.0]) * u.um,
            np.array([0.0, 0.0]) * u.mas,
        )

        for transmission in (left, right, chopped):
            self.assertEqual(transmission.shape, (2, 1, 2))
            self.assertEqual(transmission.unit, u.dimensionless_unscaled)
            np.testing.assert_allclose(transmission.value, 0.0, atol=1e-14)

    def test_transmission_matches_fourier_pair_sum(self):
        """Verify agreement between field summation and the Fourier pair sum."""
        life = LIFEOpticalSystem.__new__(LIFEOpticalSystem)

        rotations = np.array([0.0, 0.3, 1.1]) * u.rad
        sky = np.array([4.848e-7, 1.0e-7]) * u.rad
        wavelength = 10.0 * u.um
        positions = reference_collector_positions(rotations)

        left, right, chopped = life.get_transmission(positions, wavelength, sky)

        # Calculate pairwise baseline vectors from the supplied positions.
        positions_m = positions.to_value(u.m)
        baselines = positions_m[:, :, None, :] - positions_m[:, None, :, :]
        visibility = np.exp(
            1j
            * (2.0 * np.pi / wavelength.to_value(u.m))
            * np.einsum("rjkc,c->rjk", baselines, sky.to_value(u.rad))
        )
        phases = (
            np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2]),
            np.array([0.0, 3 * np.pi / 2, np.pi, np.pi / 2]),
        )

        for actual, phase in zip((left, right), phases, strict=True):
            phase_difference = phase[:, None] - phase[None, :]
            expected = (
                np.sum(np.real(np.exp(1j * phase_difference) * visibility), axis=(1, 2))
                / 16.0
            )
            np.testing.assert_allclose(actual[:, 0, 0].value, expected, atol=1e-14)

        np.testing.assert_allclose(chopped.value, (left - right).value, atol=1e-14)

    def test_transmission_mirror_symmetry_and_odd_chop(self):
        """The right output mirrors the left; the chopped response is odd."""
        life = LIFEOpticalSystem.__new__(LIFEOpticalSystem)

        sky = np.array([[4.848e-7, 0.0], [-4.848e-7, 0.0]]) * u.rad
        rotations = np.linspace(0.0, 2.0 * np.pi, 13, endpoint=False) * u.rad
        positions = reference_collector_positions(rotations)

        left, right, chopped = life.get_transmission(positions, 10.0 * u.um, sky)

        np.testing.assert_allclose(
            left[:, 0, :].value, right[:, 1, :].value, atol=1e-14
        )
        np.testing.assert_allclose(
            chopped[:, 0, :].value, -chopped[:, 1, :].value, atol=1e-14
        )
        self.assertTrue(np.all(left.value >= 0.0))
        self.assertTrue(np.all(right.value >= 0.0))
        self.assertTrue(np.all(left.value <= 1.0 + 1e-14))
        self.assertTrue(np.all(right.value <= 1.0 + 1e-14))

    def test_transmission_changes_with_collector_geometry(self):
        """Verify that different array geometries change transmission."""
        life = LIFEOpticalSystem.__new__(LIFEOpticalSystem)

        sky = np.array([4.848e-7, 1.0e-7]) * u.rad

        # X-array with baseline ratio 6.
        positions_6 = (
            np.array(
                [
                    [-7.25, -43.5],
                    [-7.25, 43.5],
                    [7.25, -43.5],
                    [7.25, 43.5],
                ]
            )
            * u.m
        )

        # X-array with baseline ratio 8.
        positions_8 = (
            np.array(
                [
                    [-7.25, -58.0],
                    [-7.25, 58.0],
                    [7.25, -58.0],
                    [7.25, 58.0],
                ]
            )
            * u.m
        )

        left_6, _, _ = life.get_transmission(positions_6, 10.0 * u.um, sky)

        left_8, _, _ = life.get_transmission(positions_8, 10.0 * u.um, sky)

        self.assertFalse(np.isclose(left_6.value[0, 0, 0], left_8.value[0, 0, 0]))

    def test_transmission_rejects_invalid_inputs(self):
        """Require valid collector positions, wavelengths, and sky coordinates."""
        life = LIFEOpticalSystem.__new__(LIFEOpticalSystem)

        positions = (
            np.array(
                [
                    [-7.25, -43.5],
                    [-7.25, 43.5],
                    [7.25, -43.5],
                    [7.25, 43.5],
                ]
            )
            * u.m
        )

        wavelength = 10.0 * u.um
        sky = np.array([0.0, 0.0]) * u.rad

        # Collector positions must have valid units and shape.
        with self.assertRaisesRegex(TypeError, "collector_positions"):
            life.get_transmission(positions.value, wavelength, sky)

        with self.assertRaisesRegex(ValueError, "length units"):
            life.get_transmission(positions.value * u.rad, wavelength, sky)

        with self.assertRaisesRegex(ValueError, "collector_positions"):
            life.get_transmission(np.zeros((3, 2)) * u.m, wavelength, sky)

        with self.assertRaisesRegex(ValueError, "collector_positions"):
            life.get_transmission(np.full((4, 2), np.nan) * u.m, wavelength, sky)

        # Wavelengths must be positive physical quantities.
        with self.assertRaisesRegex(TypeError, "wavelengths"):
            life.get_transmission(positions, 10.0, sky)

        with self.assertRaisesRegex(ValueError, "wavelengths"):
            life.get_transmission(positions, -10.0 * u.um, sky)

        # Sky coordinates must have angular units and the correct shape.
        with self.assertRaisesRegex(ValueError, "sky_positions"):
            life.get_transmission(positions, wavelength, np.array([0, 0, 0]) * u.rad)

        with self.assertRaisesRegex(ValueError, "angular units"):
            life.get_transmission(positions, wavelength, np.array([0, 0]) * u.m)

    def test_planet_photon_rates(self):
        """Verify absolute photon rates in both chopped outputs."""
        life = LIFEOpticalSystem(
            pupilDiam=2.0,
            obscurFac=0.0,
            scienceInstruments=[
                {
                    "name": "LIFE-spectrometer",
                    "QE": 0.5,
                    "optics": 0.8,
                }
            ],
            starlightSuppressionSystems=[
                {
                    "name": "nuller",
                    "lam": 10000,
                    "optics": 0.75,
                }
            ],
        )

        positions = reference_collector_positions(np.array([0.0]) * u.deg)
        wavelengths = np.array([10.0, 12.0]) * u.um
        widths = np.array([0.3, 0.36]) * u.um
        sky = np.array([5.0, 2.0]) * u.mas

        flux = np.array([100.0, 200.0]) * u.photon / (u.s * u.m**2 * u.um)

        mode = life.observingModes[0]

        rate_left, rate_right = life.get_planet_photon_rates(
            positions, wavelengths, widths, sky, flux, mode
        )

        transmission_left, transmission_right, _ = life.get_transmission(
            positions, wavelengths, sky
        )

        # Four unobscured 2 m collectors: area = 4*pi m2.
        # Instrument and nuller throughput: 0.8 * 0.75.
        # Detector quantum efficiency: 0.5.
        expected_rate = (
            flux * widths * (4.0 * np.pi * u.m**2) * 0.8 * 0.75 * 0.5 / u.photon
        ).to(1 / u.s)

        self.assertEqual(rate_left.shape, (1, 1, 2))
        self.assertEqual(rate_right.shape, (1, 1, 2))

        np.testing.assert_allclose(
            rate_left.to_value(1 / u.s),
            (transmission_left * expected_rate).to_value(1 / u.s),
        )

        np.testing.assert_allclose(
            rate_right.to_value(1 / u.s),
            (transmission_right * expected_rate).to_value(1 / u.s),
        )

    def test_stellar_leakage_rates(self):
        """Compare analytic stellar leakage with numerical disk averaging."""
        life = LIFEOpticalSystem(
            pupilDiam=2.0,
            obscurFac=0.0,
            scienceInstruments=[
                {
                    "name": "LIFE-spectrometer",
                    "QE": 0.5,
                    "optics": 0.8,
                }
            ],
            starlightSuppressionSystems=[
                {
                    "name": "nuller",
                    "lam": 10000,
                    "optics": 0.75,
                }
            ],
        )

        positions = reference_collector_positions(np.array([0.0, 60.0]) * u.deg)
        wavelengths = np.array([10.0, 12.0]) * u.um
        widths = np.array([0.3, 0.36]) * u.um
        radius = 1.0 * u.mas

        flux = np.array([1.0e7, 1.2e7]) * u.photon / (u.s * u.m**2 * u.um)
        mode = life.observingModes[0]

        left_rate, right_rate = life.get_stellar_leakage_rates(
            positions, wavelengths, widths, radius, flux, mode
        )

        # Independently average the point-source transmission
        # over a uniform stellar disk.
        n_radial = 32
        n_azimuth = 48

        radial = np.sqrt((np.arange(n_radial) + 0.5) / n_radial)
        angles = 2.0 * np.pi * np.arange(n_azimuth) / n_azimuth

        x = radial[:, None] * np.cos(angles[None, :])
        y = radial[:, None] * np.sin(angles[None, :])

        sky_points = np.column_stack((x.ravel(), y.ravel())) * radius

        left_map, right_map, _ = life.get_transmission(
            positions, wavelengths, sky_points
        )

        # The numerical samples represent equal-area disk elements.
        average_left = left_map.mean(axis=1)
        average_right = right_map.mean(axis=1)

        incident_rate = (
            flux
            * widths
            * life.total_collecting_area
            * mode["attenuation"]
            * mode["inst"]["QE"](wavelengths)
        ).to(1 / u.s)

        expected_left = average_left * incident_rate
        expected_right = average_right * incident_rate

        self.assertEqual(left_rate.shape, (2, 2))
        self.assertEqual(right_rate.shape, (2, 2))

        np.testing.assert_allclose(
            left_rate.to_value(1 / u.s),
            expected_left.to_value(1 / u.s),
            rtol=1e-3,
        )
        np.testing.assert_allclose(
            right_rate.to_value(1 / u.s),
            expected_right.to_value(1 / u.s),
            rtol=1e-3,
        )

        # A centered, symmetric stellar disk has identical
        # mean leakage in both chopped outputs.
        np.testing.assert_allclose(
            left_rate.to_value(1 / u.s),
            right_rate.to_value(1 / u.s),
            rtol=1e-12,
        )

        self.assertTrue(np.all(left_rate > 0 * (1 / u.s)))

    def test_extended_background_rates(self):
        """Verify integration of background radiance over sky pixels."""
        life = LIFEOpticalSystem(
            pupilDiam=2.0,
            obscurFac=0.0,
            scienceInstruments=[
                {
                    "name": "LIFE-spectrometer",
                    "QE": 0.5,
                    "optics": 0.8,
                }
            ],
            starlightSuppressionSystems=[
                {
                    "name": "nuller",
                    "lam": 10000,
                    "optics": 0.75,
                }
            ],
        )

        positions = reference_collector_positions(np.array([0.0, 60.0]) * u.deg)
        wavelengths = np.array([10.0, 12.0]) * u.um
        widths = np.array([0.3, 0.36]) * u.um

        sky = (
            np.array(
                [
                    [0.0, 0.0],
                    [10.0, 5.0],
                    [-6.0, 8.0],
                ]
            )
            * u.mas
        )

        radiance = (
            np.array(
                [
                    [100.0, 120.0],
                    [200.0, 220.0],
                    [150.0, 180.0],
                ]
            )
            * u.photon
            / (u.s * u.m**2 * u.um * u.sr)
        )

        solid_angles = np.array([1.0e-14, 2.0e-14, 1.5e-14]) * u.sr

        mode = life.observingModes[0]

        rate_left, rate_right = life.get_extended_background_rates(
            positions, wavelengths, widths, sky, radiance, solid_angles, mode
        )

        left, right, _ = life.get_transmission(positions, wavelengths, sky)

        # Independent reference: add the contribution of each pixel.
        patch_rates = (
            radiance
            * solid_angles[:, None]
            * widths[None, :]
            * (4.0 * np.pi * u.m**2)
            * 0.8
            * 0.75
            * 0.5
            / u.photon
        ).to(1 / u.s)

        expected_left = (left * patch_rates[None, :, :]).sum(axis=1)

        expected_right = (right * patch_rates[None, :, :]).sum(axis=1)

        self.assertEqual(rate_left.shape, (2, 2))
        self.assertEqual(rate_right.shape, (2, 2))

        np.testing.assert_allclose(
            rate_left.to_value(1 / u.s),
            expected_left.to_value(1 / u.s),
        )
        np.testing.assert_allclose(
            rate_right.to_value(1 / u.s),
            expected_right.to_value(1 / u.s),
        )

        self.assertTrue(np.all(rate_left >= 0 / u.s))
        self.assertTrue(np.all(rate_right >= 0 / u.s))

    def test_detector_noise_rates(self):
        """Verify dark current and read-noise variance rates."""
        life = LIFEOpticalSystem(
            pupilDiam=2.0,
            obscurFac=0.0,
            scienceInstruments=[
                {
                    "name": "LIFE-spectrometer",
                    "QE": 0.5,
                    "optics": 0.8,
                    "idark": 0.02,
                    "sread": 4.0,
                    "texp": 40.0,
                }
            ],
            starlightSuppressionSystems=[
                {
                    "name": "nuller",
                    "lam": 10000,
                    "optics": 0.75,
                }
            ],
        )

        wavelengths = np.array([10.0, 12.0]) * u.um
        pixels = np.array([3.0, 5.0])

        dark, read = life.get_detector_noise_rates(
            wavelengths, pixels, life.observingModes[0]
        )

        np.testing.assert_allclose(dark.to_value(1 / u.s), [0.06, 0.10])
        np.testing.assert_allclose(read.to_value(1 / u.s), [0.30, 0.50])

        self.assertEqual(dark.shape, (2,))
        self.assertEqual(read.shape, (2,))

        scalar_dark, scalar_read = life.get_detector_noise_rates(
            wavelengths, 3.0, life.observingModes[0]
        )

        np.testing.assert_allclose(scalar_dark.to_value(1 / u.s), [0.06, 0.06])
        np.testing.assert_allclose(scalar_read.to_value(1 / u.s), [0.30, 0.30])

    def test_detector_noise_rejects_invalid_inputs(self):
        """Reject invalid pixel counts and detector parameters."""
        life = LIFEOpticalSystem.__new__(LIFEOpticalSystem)

        wavelengths = np.array([10.0, 12.0]) * u.um

        mode = {
            "inst": {
                "idark": 0.02 / u.s,
                "sread": 4.0,
                "texp": 40.0 * u.s,
            }
        }

        with self.assertRaisesRegex(ValueError, "n_pixels_per_output"):
            life.get_detector_noise_rates(wavelengths, np.array([2.0, 0.0]), mode)

        with self.assertRaisesRegex(ValueError, "n_pixels_per_output"):
            life.get_detector_noise_rates(wavelengths, np.array([2.0, 3.0, 4.0]), mode)

        bad_mode = {"inst": {**mode["inst"], "texp": 0.0 * u.s}}

        with self.assertRaisesRegex(ValueError, "texp"):
            life.get_detector_noise_rates(wavelengths, 3.0, bad_mode)

        bad_mode = {"inst": {**mode["inst"], "sread": -1.0}}

        with self.assertRaisesRegex(ValueError, "sread"):
            life.get_detector_noise_rates(wavelengths, 3.0, bad_mode)

    def test_assemble_life_signal_noise(self):
        """Verify chopped signal, variances, and rotation weighting."""
        life = LIFEOpticalSystem.__new__(LIFEOpticalSystem)

        # Two rotations, one planet, one wavelength.
        planet_left = np.array([[[10.0]], [[6.0]]]) / u.s
        planet_right = np.array([[[4.0]], [[2.0]]]) / u.s

        stellar_left = np.array([[20.0], [30.0]]) / u.s
        stellar_right = np.array([[21.0], [31.0]]) / u.s

        # One extended-background component.
        bg_left = np.array([[7.0], [8.0]]) / u.s
        bg_right = np.array([[9.0], [10.0]]) / u.s

        fractions = np.array([0.25, 0.75])

        signal, variance = life.assemble_life_signal_noise(
            planet_left=planet_left,
            planet_right=planet_right,
            stellar_left=stellar_left,
            stellar_right=stellar_right,
            background_rates=[(bg_left, bg_right)],
            detector_dark_rate=np.array([1.0]) / u.s,
            detector_read_variance_rate=np.array([2.0]) / u.s,
            rotation_fractions=fractions,
        )

        # Rotation 1:
        # Signal = (10 - 4) * 0.25 = 1.5
        # Variance = (10+4+20+21+7+9+2*(1+2))*0.25 = 19.25
        #
        # Rotation 2:
        # Signal = (6 - 2) * 0.75 = 3.0
        # Variance = (6+2+30+31+8+10+2*(1+2))*0.75 = 69.75

        self.assertEqual(signal.shape, (1, 2))
        self.assertEqual(variance.shape, (1, 2))

        np.testing.assert_allclose(
            signal.to_value(1 / u.s),
            [[1.5, 3.0]],
        )

        np.testing.assert_allclose(
            variance.to_value(1 / u.s),
            [[19.25, 69.75]],
        )

        # Independently calculate the expected SNR.
        expected_snr = np.sqrt(10.0 * (0.25 * 6.0**2 / 77.0 + 0.75 * 4.0**2 / 93.0))

        actual_snr = life.calc_life_snr(signal, variance, 10.0 * u.s)

        self.assertEqual(actual_snr.shape, (1,))
        np.testing.assert_allclose(
            actual_snr[0],
            expected_snr,
            rtol=1e-6,
            atol=1e-12,
        )

        # Invalid exposure allocation must be rejected.
        with self.assertRaisesRegex(ValueError, "sum to one"):
            life.assemble_life_signal_noise(
                planet_left,
                planet_right,
                stellar_left,
                stellar_right,
                rotation_fractions=np.array([0.4, 0.4]),
            )

    def test_calc_life_snr(self):
        """Verify photon-limited and systematic-limited SNR."""
        life = LIFEOpticalSystem.__new__(LIFEOpticalSystem)

        signal = np.array([3.0, 4.0]) / u.s
        variance = np.array([4.0, 16.0]) / u.s
        time = 10.0 * u.s

        # Photon-noise-limited case:
        # SNR^2 = 9*100/40 + 16*100/160 = 32.5
        snr_photon = life.calc_life_snr(signal, variance, time)

        np.testing.assert_allclose(
            snr_photon,
            np.sqrt(32.5),
            rtol=1e-6,
            atol=1e-12,
        )

        # Add an independent systematic floor in each channel.
        systematic = np.array([0.5, 1.0]) / u.s

        snr_systematic = life.calc_life_snr(signal, variance, time, systematic)

        # Denominators: 40 + 25 = 65; 160 + 100 = 260.
        expected_squared = 900.0 / 65.0 + 1600.0 / 260.0

        np.testing.assert_allclose(
            snr_systematic,
            np.sqrt(expected_squared),
            rtol=1e-6,
            atol=1e-12,
        )

        self.assertLess(snr_systematic, snr_photon)

        # The systematic floor limits SNR at long times.
        snr_long = life.calc_life_snr(signal, variance, 1.0e10 * u.s, systematic)

        expected_limit = np.sqrt((3.0 / 0.5) ** 2 + (4.0 / 1.0) ** 2)

        np.testing.assert_allclose(
            snr_long,
            expected_limit,
            rtol=1e-08,
            atol=1e-12,
        )

        # Preserve the observing-configuration axis.
        signal_2d = (
            np.array(
                [
                    [3.0, 4.0],
                    [1.0, 2.0],
                ]
            )
            / u.s
        )

        variance_2d = (
            np.array(
                [
                    [4.0, 16.0],
                    [2.0, 4.0],
                ]
            )
            / u.s
        )

        result = life.calc_life_snr(signal_2d, variance_2d, time)

        self.assertEqual(result.shape, (2,))

        np.testing.assert_allclose(
            result,
            [np.sqrt(32.5), np.sqrt(15.0)],
        )

    def test_calc_life_snr_invalid_inputs(self):
        """Reject inconsistent rates and invalid integration times."""
        life = LIFEOpticalSystem.__new__(LIFEOpticalSystem)

        signal = np.array([3.0, 4.0]) / u.s
        variance = np.array([4.0, 16.0]) / u.s

        with self.assertRaisesRegex(ValueError, "positive"):
            life.calc_life_snr(signal, variance, 0.0 * u.s)

        with self.assertRaisesRegex(ValueError, "shape"):
            life.calc_life_snr(signal, np.array([4.0]) / u.s, 10.0 * u.s)

        with self.assertRaisesRegex(ValueError, "nonnegative"):
            life.calc_life_snr(
                signal,
                np.array([4.0, -1.0]) / u.s,
                10.0 * u.s,
            )

        with self.assertRaisesRegex(ValueError, "shape"):
            life.calc_life_snr(
                signal,
                variance,
                10.0 * u.s,
                np.array([0.5]) / u.s,
            )

        with self.assertRaisesRegex(ValueError, "positive"):
            life.calc_life_snr(
                signal,
                np.zeros(2) / u.s,
                10.0 * u.s,
            )

    def test_calc_life_int_time_photon_limited(self):
        """Verify the analytical photon-noise integration time."""
        life = LIFEOpticalSystem.__new__(LIFEOpticalSystem)

        signal = np.array([3.0, 4.0]) / u.s
        variance = np.array([4.0, 16.0]) / u.s

        required_time = life.calc_life_int_time(signal, variance, snr_threshold=5.0)

        expected = 25.0 / (9.0 / 4.0 + 16.0 / 16.0)

        np.testing.assert_allclose(
            required_time.to_value(u.s),
            expected,
            rtol=1e-6,
            atol=1e-12,
        )

        # Verify that the calculated time actually reaches SNR=5.
        actual_snr = life.calc_life_snr(signal, variance, required_time)

        np.testing.assert_allclose(actual_snr, 5.0, rtol=1e-6, atol=1e-12)

        # Multiple observing configurations.
        signals = (
            np.array(
                [
                    [3.0, 4.0],
                    [1.0, 2.0],
                ]
            )
            / u.s
        )

        variances = (
            np.array(
                [
                    [4.0, 16.0],
                    [2.0, 4.0],
                ]
            )
            / u.s
        )

        times = life.calc_life_int_time(signals, variances, snr_threshold=5.0)

        self.assertEqual(times.shape, (2,))

        np.testing.assert_allclose(
            times.to_value(u.s),
            [
                25.0 / 3.25,
                25.0 / 1.5,
            ],
        )

    def test_calc_life_int_time_systematics(self):
        """Verify integration time with a systematic noise floor."""
        life = LIFEOpticalSystem.__new__(LIFEOpticalSystem)

        signal = np.array([3.0]) / u.s
        variance = np.array([4.0]) / u.s
        systematic = np.array([0.5]) / u.s

        required_time = life.calc_life_int_time(
            signal,
            variance,
            snr_threshold=4.0,
            systematic_rates=systematic,
        )

        # Single-channel analytical solution:
        # t = SNR^2 * V / (S^2 - SNR^2 * Csp^2).
        expected = 16.0 * 4.0 / (9.0 - 16.0 * 0.25)

        np.testing.assert_allclose(
            required_time.to_value(u.s),
            expected,
            rtol=1e-6,
            atol=1e-12,
        )

        actual_snr = life.calc_life_snr(signal, variance, required_time, systematic)

        np.testing.assert_allclose(actual_snr, 4.0, rtol=1e-6, atol=1e-12)

        # Maximum achievable SNR = 3 / 0.5 = 6.
        unreachable = life.calc_life_int_time(
            signal,
            variance,
            snr_threshold=6.0,
            systematic_rates=systematic,
        )

        self.assertTrue(np.isnan(unreachable.to_value(u.s)))

        # Multiple spectral channels with systematic noise.
        signals = np.array([3.0, 4.0]) / u.s
        variances = np.array([4.0, 16.0]) / u.s
        systematics = np.array([0.5, 1.0]) / u.s

        multi_time = life.calc_life_int_time(
            signals,
            variances,
            snr_threshold=4.0,
            systematic_rates=systematics,
        )

        np.testing.assert_allclose(
            multi_time.to_value(u.s),
            7.111111111111111,
            rtol=1e-6,
            atol=1e-12,
        )

        np.testing.assert_allclose(
            life.calc_life_snr(signals, variances, multi_time, systematics),
            4.0,
            rtol=1e-6,
            atol=1e-12,
        )

    def test_calc_life_int_time_invalid_inputs(self):
        """Test invalid input and zero-signal cases."""
        life = LIFEOpticalSystem.__new__(LIFEOpticalSystem)

        signal = np.array([3.0, 4.0]) / u.s
        variance = np.array([4.0, 16.0]) / u.s

        with self.assertRaisesRegex(ValueError, "positive"):
            life.calc_life_int_time(signal, variance, snr_threshold=0.0)

        with self.assertRaisesRegex(ValueError, "shape"):
            life.calc_life_int_time(
                signal,
                np.array([4.0]) / u.s,
                snr_threshold=5.0,
            )

        with self.assertRaisesRegex(ValueError, "nonnegative"):
            life.calc_life_int_time(
                signal,
                np.array([-1.0, 16.0]) / u.s,
                snr_threshold=5.0,
            )

        with self.assertRaisesRegex(ValueError, "positive"):
            life.calc_life_int_time(
                signal,
                np.zeros(2) / u.s,
                snr_threshold=5.0,
            )

        # Without a planet signal, detection is impossible.
        no_detection = life.calc_life_int_time(
            np.zeros(2) / u.s,
            variance,
            snr_threshold=5.0,
        )

        self.assertTrue(np.isnan(no_detection.to_value(u.s)))

    def test_life_exosims_scalar_bridge_photon_only(self):
        """Scalar rates must reproduce the full spectral photon SNR."""
        life = LIFEOpticalSystem.__new__(LIFEOpticalSystem)

        signal = (
            np.array(
                [
                    [3.0, 4.0],
                    [0.0, 2.0],
                ]
            )
            / u.s
        )

        variance = (
            np.array(
                [
                    [4.0, 16.0],
                    [1.0, 4.0],
                ]
            )
            / u.s
        )

        # Synthetic provider; no planet spectrum required.
        life.life_rate_provider = lambda *args: (signal, variance)

        mode = {"SNR": 5.0}
        s_inds = np.array([0, 1])
        dmag = np.zeros(2)
        wa = np.ones(2) * u.arcsec

        args = (None, s_inds, None, None, dmag, wa, mode)

        cp, cb, csp = life.Cp_Cb_Csp(*args)

        np.testing.assert_allclose(
            cb.to_value(1 / u.s),
            [20.0, 5.0],
        )

        np.testing.assert_allclose(
            (cp**2 / cb).to_value(1 / u.s),
            [3.25, 1.0],
        )

        np.testing.assert_array_equal(
            csp.to_value(1 / u.s),
            [0.0, 0.0],
        )

        times = life.calc_intTime(*args)

        self.assertTrue(times.unit.is_equivalent(u.day))

        np.testing.assert_allclose(
            times.to_value(u.s),
            [25.0 / 3.25, 25.0],
        )

        for i, integration_time in enumerate(times):
            actual = life.calc_life_snr(
                signal[i],
                variance[i],
                integration_time,
            )
            np.testing.assert_allclose(actual, 5.0, rtol=1e-6, atol=1e-12)

        with self.assertRaisesRegex(NotImplementedError, "returnExtra"):
            life.Cp_Cb_Csp(*args, returnExtra=True)

    def test_life_exosims_adapter_systematics(self):
        """Spectral inversion supports systematics; scalar mapping rejects them."""
        life = LIFEOpticalSystem.__new__(LIFEOpticalSystem)

        mode = {"SNR": 4.0}
        s_inds = np.array([0])
        wa = np.array([1.0]) * u.arcsec

        args = (
            None,
            s_inds,
            None,
            None,
            np.array([0.0]),
            wa,
            mode,
        )

        with self.assertRaisesRegex(NotImplementedError, "life_rate_provider"):
            life.calc_intTime(*args)

        life.life_rate_provider = lambda *unused: (
            np.array([[3.0]]) / u.s,
            np.array([[4.0]]) / u.s,
            np.array([[0.5]]) / u.s,
        )

        result = life.calc_intTime(*args)

        # SNR = 4, S = 3/s, V = 4/s, Csp = 0.5/s.
        # t = 16*4 / (9 - 16*0.25) = 12.8 s.
        np.testing.assert_allclose(
            result.to_value(u.s)[0],
            12.8,
            rtol=1e-6,
            atol=1e-12,
        )

        with self.assertRaisesRegex(NotImplementedError, "single"):
            life.Cp_Cb_Csp(*args)

        with self.assertRaisesRegex(NotImplementedError, "numerical"):
            life.calc_dMag_per_intTime(
                10 * u.s,
                None,
                s_inds,
                None,
                None,
                wa,
                mode,
                analytic_only=True,
            )

    def test_life_thermal_dmag_inversion(self):
        """Verify thermal dMag inversion and its time derivative."""
        life = LIFEOpticalSystem.__new__(LIFEOpticalSystem)

        s_inds = np.array([0, 1])
        wa = np.array([1.0, 1.0]) * u.arcsec
        mode = {"SNR": 5.0}

        # Synthetic wavelength-dependent planet spectrum.
        # This will later be supplied by a blackbody model.
        spectral_shape = np.array([1.0, 0.7, 0.3])

        # Chopped transmission changes sign between channels.
        transmission = np.array([0.5, -0.3, 0.1])

        background = np.array([8.0, 12.0, 6.0])

        def provider(TL, sInds, fZ, JEZ, dMag, WA, mode, TK):
            """Generate synthetic spectral rates for a trial dMag."""
            dmag = np.asarray(dMag, dtype=float)

            # Planet brightness scales with magnitude contrast.
            contrast = 10 ** (-0.4 * dmag)

            planet = 1e9 * contrast[:, None] * spectral_shape[None, :]

            # Differential planet signal.
            signal = planet * transmission[None, :]

            # Include planet photon shot noise.
            variance = background[None, :] + 0.7 * planet

            return signal / u.s, variance / u.s

        life.life_rate_provider = provider

        original_dmag = np.array([19.0, 21.0])

        # Step 1: Calculate integration time for known dMag.
        integration_time = life.calc_intTime(
            None,
            s_inds,
            None,
            None,
            original_dmag,
            wa,
            mode,
        )

        self.assertTrue(np.all(np.isfinite(integration_time.value)))
        self.assertTrue(np.all(integration_time.value > 0))

        # Step 2: Recover dMag from that integration time.
        recovered_dmag = life.calc_dMag_per_intTime(
            integration_time,
            None,
            s_inds,
            None,
            None,
            wa,
            mode,
        )

        np.testing.assert_allclose(
            recovered_dmag,
            original_dmag,
            atol=1e-6,
        )

        # Step 3: Longer integration permits fainter planets.
        longer_dmag = life.calc_dMag_per_intTime(
            2 * integration_time,
            None,
            s_inds,
            None,
            None,
            wa,
            mode,
        )

        self.assertTrue(np.all(longer_dmag > recovered_dmag))

        # Step 4: Check derivative of limiting dMag.
        derivative = life.ddMag_dt(
            integration_time,
            None,
            s_inds,
            None,
            None,
            wa,
            mode,
        )

        self.assertTrue(derivative.unit.is_equivalent(1 / u.s))
        self.assertTrue(np.all(np.isfinite(derivative.value)))
        self.assertTrue(np.all(derivative.value > 0))

        # Step 5: Zero integration time is invalid.
        invalid = life.calc_dMag_per_intTime(
            np.array([0.0, 0.0]) * u.s,
            None,
            s_inds,
            None,
            None,
            wa,
            mode,
        )

        self.assertTrue(np.all(np.isnan(invalid)))

    def test_transmission_energy_conservation(self):
        """Verify transmission normalization using a unitary beam combiner."""
        life = LIFEOpticalSystem.__new__(LIFEOpticalSystem)

        rotations = np.array([0.0, 0.7, 1.8]) * u.rad
        sky = np.array([4.848e-7, 1.0e-7]) * u.rad
        wavelength = 10.0 * u.um

        positions = reference_collector_positions(rotations)

        left, right, _ = life.get_transmission(positions, wavelength, sky)

        # Four-output double-Bracewell combiner.
        # Columns follow the ordering in reference_collector_positions().
        combiner = 0.5 * np.array(
            [
                [0, np.sqrt(2), 0, np.sqrt(2)],
                [np.sqrt(2), 0, np.sqrt(2), 0],
                [1, 1j, -1, -1j],
                [1, -1j, -1, 1j],
            ],
            dtype=complex,
        )

        # A lossless beam-combiner matrix must be unitary.
        np.testing.assert_allclose(
            combiner.conj().T @ combiner,
            np.eye(4),
            atol=1e-14,
        )

        positions_m = positions.to_value(u.m)

        # Use collector 0 as the reference for optical phase.
        relative_positions = positions_m - positions_m[:, :1, :]
        geometric_phase = (
            2
            * np.pi
            / wavelength.to_value(u.m)
            * np.einsum(
                "rjc,c->rj",
                relative_positions,
                sky.to_value(u.rad),
            )
        )

        input_fields = np.exp(1j * geometric_phase)
        output_fields = input_fields @ combiner.T

        # Normalize output intensities by four telescope inputs.
        transmissions = np.abs(output_fields) ** 2 / 4.0

        # All collected light must be accounted for.
        np.testing.assert_allclose(
            transmissions.sum(axis=1),
            1.0,
            atol=1e-14,
        )

        # The two nulled outputs must match get_transmission().
        np.testing.assert_allclose(
            transmissions[:, 2],
            left[:, 0, 0].value,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            transmissions[:, 3],
            right[:, 0, 0].value,
            atol=1e-14,
        )

    # Sun-Earth reference regression tests.

    def test_life_sun_earth_reference_benchmark(self):
        """Check LIFE rates/SNR against the LIFEsimFT Sun-Earth reference.

        Reference: padandrea/LIFEsimFT, reference.py and detectability.py.
        This uses the reference's wavelength-bin convention (including its
        last overshooting bin), to compare the same physical configuration.
        """
        from scipy.constants import au as au_si
        from scipy.constants import c as c_si
        from scipy.constants import h as h_si
        from scipy.constants import k as k_si
        from scipy.special import j0

        life = LIFEOpticalSystem.__new__(LIFEOpticalSystem)
        diameter = 3.0  # m, per collector
        efficiency = 0.035  # overall photon-conversion efficiency
        life.pupilArea = np.pi * (diameter / 2) ** 2 * u.m**2
        mode = {
            "attenuation": efficiency,
            "inst": {"QE": lambda _lam: 1.0 / u.photon},
        }

        distance_m = (10 * u.pc).to_value(u.m)
        star_radius_m = (1 * u.R_sun).to_value(u.m)
        planet_radius_m = 6.3781e6

        # LIFEsimFT reference uses edges growing by (1 + 1/R), with no
        # truncation of the last bin. This differs slightly from the
        # current LIFEOpticalSystem.get_spectral_bins() convention.
        resolving_power = 10.0 / 0.3
        n_bins = int(np.ceil(np.log(18.5 / 4.0) / np.log1p(1 / resolving_power)))
        edges_m = 4e-6 * (1 + 1 / resolving_power) ** np.arange(n_bins + 1)
        lam_m = (edges_m[:-1] + edges_m[1:]) / 2
        widths_m = np.diff(edges_m)
        wavelengths = lam_m * u.m
        widths = widths_m * u.m

        # 2325 samples over 15 complete rotations in LIFEsimFT are
        # equivalent to 155 distinct equally weighted orientations.
        angles = np.arange(155) * (2 * np.pi / 155) * u.rad
        positions = reference_collector_positions(angles)
        sky = np.array([au_si / distance_m, 0.0]) * u.rad

        def blackbody_photon_flux(temperature_k, radius_m):
            """Observed photon flux density of a uniform blackbody sphere."""
            exponent = h_si * c_si / (lam_m * k_si * temperature_k)
            values = (
                2
                * np.pi
                * c_si
                / lam_m**4
                / np.expm1(exponent)
                * (radius_m / distance_m) ** 2
            )
            return (values * u.photon / (u.s * u.m**2 * u.m)).to(
                u.photon / (u.s * u.m**2 * u.um)
            )

        planet_flux = blackbody_photon_flux(254.0, planet_radius_m)
        stellar_flux = blackbody_photon_flux(5778.0, star_radius_m)

        planet_l, planet_r = life.get_planet_photon_rates(
            positions, wavelengths, widths, sky, planet_flux, mode
        )
        star_l, star_r = life.get_stellar_leakage_rates(
            positions,
            wavelengths,
            widths,
            (star_radius_m / distance_m) * u.rad,
            stellar_flux,
            mode,
        )

        self.assertEqual(planet_l.shape, (155, 1, n_bins))
        self.assertEqual(star_l.shape, (155, n_bins))

        # Reference astrophysical backgrounds (LIFEsimFT sources.py).
        # No atmospheric models or extra EXOSIMS source modules needed.
        def planck_photon_radiance(lam, temperature_k):
            return (
                2 * c_si / lam**4 / np.expm1(h_si * c_si / (lam * k_si * temperature_k))
            )

        one_collector_area = np.pi * (diameter / 2) ** 2
        # Uniform disk brightness cancels the chopped mean but adds noise.
        phase = np.arange(4) * np.pi / 2
        weights = np.cos(phase[:, None] - phase[None, :])
        collectors = positions[0].to_value(u.m)
        baselines = np.linalg.norm(
            collectors[:, None, :] - collectors[None, :, :], axis=-1
        )

        local_zodi_per_output = np.empty(n_bins)
        exozodi_per_output = np.empty(n_bins)

        zodi_lat = np.deg2rad(45.0)
        cos_lat = np.cos(zodi_lat)
        sin_lat = np.sin(zodi_lat)
        solar_elongation = np.arccos(np.cos(0.75 * np.pi) * cos_lat)
        dust_r_in = (278.3 / 1500.0) ** 2
        dust_r_out = (278.3 / 88.0) ** 2

        for i, (lam, band) in enumerate(zip(lam_m, widths_m)):
            geometry = np.sqrt(
                np.pi
                / solar_elongation
                / (sin_lat**2 + (0.6 * (lam / 11e-6) ** (-0.4) * cos_lat) ** 2)
            )
            local_radiance = (
                4e-8
                * (
                    planck_photon_radiance(lam, 265.0)
                    + 0.22
                    * (0.00465047 / 1.5) ** 2
                    * planck_photon_radiance(lam, 5777.0)
                )
                * geometry
            )
            fov_radius = lam / (2 * diameter)
            omega = np.pi * fov_radius**2
            local_zodi_per_output[i] = (
                local_radiance * omega * band * one_collector_area * efficiency
            )

            # Face-on, one-zodi Kennedy et al. (2015) dust disk.
            theta = np.linspace(dust_r_in * au_si / distance_m, fov_radius, 4000)
            radius_au = theta * distance_m / au_si
            inside = (radius_au >= dust_r_in) & (radius_au <= dust_r_out)
            radius_safe = np.where(inside, radius_au, 1.0)
            temperature = 278.3 / np.sqrt(radius_safe)
            sigma = 7.12e-8 * radius_safe ** (-0.34)
            radiance = np.where(
                inside,
                sigma * planck_photon_radiance(lam, temperature),
                0.0,
            )
            visibility = np.trapezoid(
                j0(2 * np.pi * baselines[:, :, None] * theta[None, None, :] / lam)
                * radiance[None, None, :]
                * (2 * np.pi * theta)[None, None, :],
                theta,
                axis=-1,
            )
            exozodi_per_output[i] = (
                np.sum(weights * visibility)
                * band
                * one_collector_area
                * efficiency
                / 4
            )

        # Both phase-chopped outputs receive independent background noise.
        n_rotations = len(angles)
        lz_rate = np.broadcast_to(local_zodi_per_output, (n_rotations, n_bins)) / u.s
        ez_rate = np.broadcast_to(exozodi_per_output, (n_rotations, n_bins)) / u.s

        signal, variance = life.assemble_life_signal_noise(
            planet_l,
            planet_r,
            star_l,
            star_r,
            background_rates=[(lz_rate, lz_rate), (ez_rate, ez_rate)],
        )
        snr_life = life.calc_life_snr(signal[0], variance[0], 16 * u.day)
        time_life = life.calc_life_int_time(
            signal, variance, snr_threshold=5.0
        ).to_value(u.h)[0]

        # LIFEsimFT uses one mean noise rate per wavelength bin, whereas
        # LIFEOpticalSystem uses per-rotation inverse-variance weighting.
        chopped = (planet_l[:, 0, :] - planet_r[:, 0, :]).to_value(1 / u.s)
        total_noise = (
            planet_l[:, 0, :] + planet_r[:, 0, :] + star_l + star_r
        ).to_value(1 / u.s) + 2 * (local_zodi_per_output + exozodi_per_output)[None, :]
        snr_reference_statistic = np.sqrt(
            (16 * u.day).to_value(u.s)
            * np.sum(np.mean(chopped**2, axis=0) / np.mean(total_noise, axis=0))
        )

        print(f"Sun-Earth reference:  {snr_reference_statistic:.4f} SNR / 16 days")
        print(f"EXOSIMS LIFE SNR:    {snr_life:.4f} / 16 days")
        print(f"EXOSIMS LIFE t5:     {time_life:.4f} h")
        print(f"Stellar leakage:     {star_l.to_value(1/u.s)[0].sum():.2f} 1/s")
        print(f"Local zodi:          {local_zodi_per_output.sum():.2f} 1/s")
        print(f"Exozodi:             {exozodi_per_output.sum():.2f} 1/s")

        np.testing.assert_allclose(
            snr_reference_statistic,
            42.18,
            rtol=0,
            atol=0.1,
        )
        np.testing.assert_allclose(snr_life, 42.18, rtol=0, atol=0.1)
        np.testing.assert_allclose(time_life, 5.4, rtol=0, atol=0.05)
        np.testing.assert_allclose(
            life.calc_life_snr(signal[0], variance[0], time_life * u.h),
            5.0,
            rtol=0,
            atol=1e-10,
        )

    def test_life_sun_earth_local_zodi_map_10um(self):
        """Check the sky-pixel background integrator with a physical FoV."""
        life = LIFEOpticalSystem.__new__(LIFEOpticalSystem)
        diameter = 3.0
        efficiency = 0.035
        life.pupilArea = np.pi * (diameter / 2) ** 2 * u.m**2
        mode = {
            "attenuation": efficiency,
            "inst": {"QE": lambda _lam: 1.0 / u.photon},
        }

        lam = 10e-6
        width = 0.3e-6
        fov = lam / (2 * diameter)
        n = 96
        grid = (np.arange(n) + 0.5) * (2 * fov / n) - fov
        xx, yy = np.meshgrid(grid, grid, indexing="ij")
        sky = np.stack([xx.ravel(), yy.ravel()], axis=-1) * u.rad
        solid_angle = np.full(n * n, (2 * fov / n) ** 2) * u.sr
        within_fov = (xx.ravel() ** 2 + yy.ravel() ** 2) <= fov**2

        from scipy.constants import c as c_si
        from scipy.constants import h as h_si
        from scipy.constants import k as k_si

        def planck_photon_radiance(temp):
            return 2 * c_si / lam**4 / np.expm1(h_si * c_si / (lam * k_si * temp))

        cos_lat = np.cos(np.pi / 4)
        sin_lat = np.sin(np.pi / 4)
        elong = np.arccos(np.cos(0.75 * np.pi) * cos_lat)
        geometry = np.sqrt(
            np.pi
            / elong
            / (sin_lat**2 + (0.6 * (lam / 11e-6) ** (-0.4) * cos_lat) ** 2)
        )
        radiance_value = (
            4e-8
            * (
                planck_photon_radiance(265.0)
                + 0.22 * (0.00465047 / 1.5) ** 2 * planck_photon_radiance(5777.0)
            )
            * geometry
        )
        pixel_radiance = (radiance_value * within_fov[:, None]) * (
            u.photon / (u.s * u.m**2 * u.m * u.sr)
        )

        left, right = life.get_extended_background_rates(
            reference_collector_positions(np.array([0.0]) * u.rad),
            np.array([lam]) * u.m,
            np.array([width]) * u.m,
            sky,
            pixel_radiance,
            solid_angle,
            mode,
        )
        expected = (
            radiance_value
            * width
            * np.pi
            * fov**2
            * np.pi
            * (diameter / 2) ** 2
            * efficiency
        )
        # Finite FoV of 96x96 pixels approximates the analytical foreground.
        np.testing.assert_allclose(
            left.to_value(1 / u.s)[0, 0],
            expected,
            rtol=0.08,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            right.to_value(1 / u.s)[0, 0],
            expected,
            rtol=0.08,
            atol=1e-12,
        )

        # The same spatial integrator should also accept a radial exozodi map.
        from scipy.constants import au as au_si
        from scipy.special import j0

        distance_m = (10 * u.pc).to_value(u.m)
        r_au = np.hypot(xx, yy).ravel() * distance_m / au_si
        r_in = (278.3 / 1500.0) ** 2
        r_out = (278.3 / 88.0) ** 2
        inside = within_fov & (r_au >= r_in) & (r_au <= r_out)
        r_safe = np.where(inside, r_au, 1.0)
        dust_radiance = np.where(
            inside,
            7.12e-8
            * r_safe ** (-0.34)
            * planck_photon_radiance(278.3 / np.sqrt(r_safe)),
            0.0,
        )
        disk_map = dust_radiance[:, None] * (u.photon / (u.s * u.m**2 * u.m * u.sr))
        ez_left, ez_right = life.get_extended_background_rates(
            reference_collector_positions(np.array([0.0]) * u.rad),
            np.array([lam]) * u.m,
            np.array([width]) * u.m,
            sky,
            disk_map,
            solid_angle,
            mode,
        )

        # Independent radial Hankel transform of the same face-on dust disk.
        theta = np.linspace(r_in * au_si / distance_m, fov, 4000)
        radius_au = theta * distance_m / au_si
        active = (radius_au >= r_in) & (radius_au <= r_out)
        radius_safe = np.where(active, radius_au, 1.0)
        disk_profile = np.where(
            active,
            7.12e-8
            * radius_safe ** (-0.34)
            * planck_photon_radiance(278.3 / np.sqrt(radius_safe)),
            0.0,
        )
        collector_xy = reference_collector_positions(np.array([0.0]) * u.rad)[
            0
        ].to_value(u.m)
        baselines = np.linalg.norm(
            collector_xy[:, None, :] - collector_xy[None, :, :], axis=-1
        )
        phase = np.arange(4) * np.pi / 2
        weights = np.cos(phase[:, None] - phase[None, :])
        visibility = np.trapezoid(
            j0(2 * np.pi * baselines[:, :, None] * theta[None, None, :] / lam)
            * disk_profile[None, None, :]
            * (2 * np.pi * theta)[None, None, :],
            theta,
            axis=-1,
        )
        ez_expected = (
            np.sum(weights * visibility)
            / 4
            * width
            * np.pi
            * (diameter / 2) ** 2
            * efficiency
        )
        np.testing.assert_allclose(
            ez_left.to_value(1 / u.s)[0, 0],
            ez_expected,
            rtol=0.08,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            ez_right.to_value(1 / u.s)[0, 0],
            ez_expected,
            rtol=0.08,
            atol=1e-12,
        )

    def test_life_mennesson_local_zodi_integration(self):
        """Test the existing Mennesson local-zodi model with LIFE optics."""

        # 1. Initialize the existing EXOSIMS zodiacal-light model.
        zodi = Mennesson(verbose=False)

        # 2. Set up the LIFE optical system.
        life = LIFEOpticalSystem.__new__(LIFEOpticalSystem)
        diameter = 3.0 * u.m
        life.pupilArea = np.pi * (diameter / 2) ** 2

        mode = {
            "lam": 11.25 * u.um,
            "deltaLam": 14.5 * u.um,
            "attenuation": 0.035,
            "inst": {
                "name": "LIFE-spectrometer",
                "Rs": 33.3,
                "QE": lambda _lam: 1.0 / u.photon,
            },
        }

        wavelengths, widths = life.get_spectral_bins_from_mode(mode)

        # 3. Calculate zodiacal spectral radiance with Mennesson.
        # Example sightline; real target geometry comes later.
        lon = np.array([135.0]) * u.deg
        lat = np.array([45.0]) * u.deg

        I_ref = zodi.zodi_intensity_at_location(lon, lat, photon_units=True)[0]

        color = zodi.zodi_color_correction_factor(wavelengths, photon_units=True)

        I_photon = (I_ref * color).to(u.photon / (u.s * u.m**2 * u.um * u.sr))

        self.assertEqual(I_photon.shape, wavelengths.shape)
        self.assertTrue(np.all(np.isfinite(I_photon.value)))
        self.assertTrue(np.all(I_photon.value > 0))

        # 4. Construct a finite sky grid.
        # Fixed reference FoV for this integration test.
        fov = (10.0 * u.um / (2 * diameter)).to_value(u.dimensionless_unscaled)

        n = 24
        axis = (np.arange(n) + 0.5) * (2 * fov / n) - fov
        xx, yy = np.meshgrid(axis, axis, indexing="ij")

        sky_positions = np.stack([xx.ravel(), yy.ravel()], axis=-1) * u.rad

        pixel_solid_angles = np.full(n * n, (2 * fov / n) ** 2) * u.sr

        within_fov = xx.ravel() ** 2 + yy.ravel() ** 2 <= fov**2

        # Local zodi is approximately uniform over the small field.
        radiance_map = within_fov[:, None] * I_photon[None, :]

        # 5. Obtain the reference LIFE collector geometry.
        collector_positions = reference_collector_positions(np.array([0.0]) * u.rad)

        # 6. Calculate left/right zodiacal photon rates.
        C_z_left, C_z_right = life.get_extended_background_rates(
            collector_positions,
            wavelengths,
            widths,
            sky_positions,
            radiance_map,
            pixel_solid_angles,
            mode,
        )

        # 7. Validate output shapes and physical behavior.
        self.assertEqual(C_z_left.shape, (1, len(wavelengths)))
        self.assertEqual(C_z_right.shape, (1, len(wavelengths)))

        self.assertTrue(np.all(C_z_left > 0 / u.s))
        self.assertTrue(np.all(C_z_right > 0 / u.s))

        # Symmetric uniform emission produces equal mean rates.
        np.testing.assert_allclose(
            C_z_left.to_value(1 / u.s),
            C_z_right.to_value(1 / u.s),
            rtol=1e-2,
        )

    # ================================================================
    # LIFE Gaussian field-of-view tests
    # ================================================================

    def test_fov_gaussian_on_axis_and_reference_scale(self):
        """Check on-axis response and the LIFEsim Gaussian formula."""
        life = make_fov_test_optical_system()

        wavelength = 10.0 * u.um
        diameter = life.pupilDiam

        hfov = (wavelength / (2.0 * diameter)).to_value(
            u.dimensionless_unscaled
        ) * u.rad

        sky = (
            np.array(
                [
                    [0.0, 0.0],
                    [hfov.to_value(u.rad), 0.0],
                    [2.0 * hfov.to_value(u.rad), 0.0],
                ]
            )
            * u.rad
        )

        attenuation = life.get_fov_attenuation(
            wavelength,
            sky,
            fov_taper="gaussian",
        )

        self.assertEqual(attenuation.shape, (3, 1))

        # At the field center, no attenuation.
        np.testing.assert_allclose(attenuation[0, 0], 1.0, rtol=1e-6, atol=1e-12)

        # At theta = hfov, G = exp(-(pi/4)^2).
        np.testing.assert_allclose(
            attenuation[1, 0], np.exp(-((np.pi / 4.0) ** 2)), rtol=1e-6, atol=1e-12
        )

        # At theta = 2*hfov, G = exp(-(pi/2)^2).
        np.testing.assert_allclose(
            attenuation[2, 0], np.exp(-((np.pi / 2.0) ** 2)), rtol=1e-6, atol=1e-12
        )

    def test_fov_gaussian_wavelength_dependence(self):
        """A longer wavelength produces a broader field of view."""
        life = make_fov_test_optical_system()

        wavelengths = np.array([10.0, 15.0]) * u.um
        sky = np.array([200.0, 0.0]) * u.mas

        attenuation = life.get_fov_attenuation(
            wavelengths,
            sky,
            fov_taper="gaussian",
        )

        self.assertEqual(attenuation.shape, (1, 2))
        self.assertLess(attenuation[0, 0], attenuation[0, 1])

        self.assertTrue(np.all(attenuation > 0))
        self.assertTrue(np.all(attenuation <= 1))

    def test_fov_none_is_identity(self):
        """Turning off attenuation must recover the original model."""
        life = make_fov_test_optical_system()

        sky = np.array([[0.0, 0.0], [100.0, 50.0], [-200.0, 30.0]]) * u.mas

        wavelengths = np.array([8.0, 10.0, 15.0]) * u.um

        attenuation = life.get_fov_attenuation(
            wavelengths,
            sky,
            fov_taper="none",
        )

        np.testing.assert_array_equal(
            attenuation,
            np.ones((3, 3)),
        )

    def test_fov_transmission_scaling(self):
        """Gaussian taper multiplies both outputs and the chopped map."""
        life = make_fov_test_optical_system()

        positions = reference_collector_positions(np.array([0.0, 0.7, 1.8]) * u.rad)

        wavelengths = np.array([10.0, 15.0]) * u.um

        sky = (
            np.array(
                [
                    [50.0, 20.0],
                    [120.0, 80.0],
                ]
            )
            * u.mas
        )

        ideal = life.get_transmission(
            positions,
            wavelengths,
            sky,
            fov_taper="none",
        )

        gaussian = life.get_transmission(
            positions,
            wavelengths,
            sky,
            fov_taper="gaussian",
        )

        attenuation = life.get_fov_attenuation(
            wavelengths,
            sky,
            fov_taper="gaussian",
        )

        for ideal_map, gaussian_map in zip(ideal, gaussian, strict=True):
            np.testing.assert_allclose(
                gaussian_map.value,
                ideal_map.value * attenuation[None, :, :],
                rtol=1e-12,
                atol=1e-14,
            )

        np.testing.assert_allclose(
            gaussian[2].value,
            (gaussian[0] - gaussian[1]).value,
            atol=1e-14,
        )

    def test_fov_planet_photon_rate_scaling(self):
        """Planet rates must inherit the field-of-view attenuation."""
        life = make_fov_test_optical_system()

        positions = reference_collector_positions(np.array([0.0]) * u.rad)

        wavelengths = np.array([10.0, 15.0]) * u.um
        widths = np.array([0.3, 0.45]) * u.um

        sky = np.array([140.0, 80.0]) * u.mas

        flux = np.array([100.0, 200.0]) * u.photon / (u.s * u.m**2 * u.um)

        mode_none = {
            **life.observingModes[0],
            "LIFE_fov_taper": "none",
        }

        mode_gaussian = {
            **life.observingModes[0],
            "LIFE_fov_taper": "gaussian",
        }

        rates_none = life.get_planet_photon_rates(
            positions, wavelengths, widths, sky, flux, mode_none
        )

        rates_gaussian = life.get_planet_photon_rates(
            positions, wavelengths, widths, sky, flux, mode_gaussian
        )

        attenuation = life.get_fov_attenuation(wavelengths, sky, fov_taper="gaussian")

        for original, tapered in zip(rates_none, rates_gaussian, strict=True):
            np.testing.assert_allclose(
                tapered.to_value(1 / u.s),
                original.to_value(1 / u.s) * attenuation[None, :, :],
                rtol=1e-12,
                atol=1e-14,
            )

    def test_fov_extended_background_scaling(self):
        """Verify Gaussian attenuation of extended-background rates."""
        for background in ("local", "exozodi"):
            with self.subTest(background=background):
                life = make_fov_test_optical_system()

                positions = reference_collector_positions(np.array([0.0, 0.7]) * u.rad)

                wavelengths = np.array([10.0, 12.0]) * u.um
                widths = np.array([0.3, 0.36]) * u.um

                # All pixels have the same radial distance: 200 mas.
                # Their Gaussian attenuation is therefore identical.
                sky = (
                    np.array(
                        [
                            [120.0, 160.0],
                            [-120.0, 160.0],
                            [0.0, -200.0],
                        ]
                    )
                    * u.mas
                )

                if background == "local":
                    # Spatially uniform local zodiacal background.
                    values = np.ones((3, 2)) * 100.0
                else:
                    # Spatially varying exozodiacal background.
                    values = np.array(
                        [
                            [100.0, 120.0],
                            [200.0, 250.0],
                            [150.0, 180.0],
                        ]
                    )

                radiance = values * u.photon / (u.s * u.m**2 * u.um * u.sr)

                solid_angles = np.full(3, 1.0e-12) * u.sr

                mode_none = {
                    **life.observingModes[0],
                    "LIFE_fov_taper": "none",
                }

                mode_gaussian = {
                    **life.observingModes[0],
                    "LIFE_fov_taper": "gaussian",
                }

                rates_none = life.get_extended_background_rates(
                    positions,
                    wavelengths,
                    widths,
                    sky,
                    radiance,
                    solid_angles,
                    mode_none,
                )

                rates_gaussian = life.get_extended_background_rates(
                    positions,
                    wavelengths,
                    widths,
                    sky,
                    radiance,
                    solid_angles,
                    mode_gaussian,
                )

                attenuation = life.get_fov_attenuation(
                    wavelengths, sky, fov_taper="gaussian"
                )

                # Equal-radius pixels have the same attenuation per channel.
                np.testing.assert_allclose(
                    attenuation,
                    np.broadcast_to(attenuation[0], attenuation.shape),
                )

                for original, tapered in zip(rates_none, rates_gaussian, strict=True):
                    np.testing.assert_allclose(
                        tapered.to_value(1 / u.s),
                        original.to_value(1 / u.s) * attenuation[0][None, :],
                        rtol=1e-12,
                        atol=1e-14,
                    )

    def test_fov_rejects_invalid_model(self):
        """Reject unknown field-of-view models."""
        life = make_fov_test_optical_system()

        with self.assertRaisesRegex(ValueError, "fov_taper"):
            life.get_fov_attenuation(
                10.0 * u.um,
                np.array([0.0, 0.0]) * u.mas,
                fov_taper="invalid",
            )

    def test_fov_observing_mode_configuration(self):
        """Verify Gaussian field-of-view selection via observing mode."""
        life = LIFEOpticalSystem(
            pupilDiam=2.0,
            scienceInstruments=[{"name": "LIFE-spectrometer"}],
            starlightSuppressionSystems=[{"name": "nuller", "lam": 10000}],
            observingModes=[
                {
                    "instName": "LIFE-spectrometer",
                    "systName": "nuller",
                    "LIFE_fov_taper": "gaussian",
                }
            ],
        )

        self.assertEqual(life.observingModes[0]["LIFE_fov_taper"], "gaussian")

    def test_circular_fov_is_wavelength_dependent_for_backgrounds(self):
        """A 640 mas dust patch is excluded at 8 um but included at 16 um."""
        life = make_fov_test_optical_system()  # D = 2 m
        positions = reference_collector_positions(np.array([0.3, 1.1]) * u.rad)
        wavelengths = np.array([8.0, 16.0]) * u.um
        widths = np.array([0.24, 0.48]) * u.um
        sky = np.array([[500.0, 400.0]]) * u.mas
        radiance = np.full((1, 2), 100.0) * u.photon / (u.s * u.m**2 * u.um * u.sr)
        areas = np.array([1.0e-12]) * u.sr
        square = {**life.observingModes[0], "LIFE_fov_boundary": "square"}
        circle = {**square, "LIFE_fov_boundary": "circle"}
        square_rates = life.get_extended_background_rates(
            positions, wavelengths, widths, sky, radiance, areas, square
        )
        circle_rates = life.get_extended_background_rates(
            positions, wavelengths, widths, sky, radiance, areas, circle
        )
        self.assertGreater(sum(np.sum(x.value) for x in square_rates), 0)
        for unrestricted, restricted in zip(square_rates, circle_rates, strict=True):
            np.testing.assert_allclose(restricted[:, 0].value, 0.0, atol=0.0)
            np.testing.assert_allclose(
                restricted[:, 1].to_value(1 / u.s),
                unrestricted[:, 1].to_value(1 / u.s),
                rtol=1e-12,
                atol=1e-14,
            )

    def test_circular_fov_does_not_mask_planet_signal(self):
        """The circle selects diffuse integration pixels, not point sources."""
        life = make_fov_test_optical_system()
        positions = reference_collector_positions(np.array([0.7]) * u.rad)
        wavelengths = np.array([8.0, 16.0]) * u.um
        widths = np.array([0.24, 0.48]) * u.um
        sky = np.array([500.0, 400.0]) * u.mas
        flux = np.array([100.0, 100.0]) * u.photon / (u.s * u.m**2 * u.um)
        square = {**life.observingModes[0], "LIFE_fov_boundary": "square"}
        circle = {**square, "LIFE_fov_boundary": "circle"}
        standard = life.get_planet_photon_rates(
            positions, wavelengths, widths, sky, flux, square
        )
        selected = life.get_planet_photon_rates(
            positions, wavelengths, widths, sky, flux, circle
        )
        for original, actual in zip(standard, selected, strict=True):
            np.testing.assert_array_equal(actual.value, original.value)

    def test_fov_boundary_rejects_invalid_model(self):
        """Invalid boundary choices must not silently revert to square."""
        life = make_fov_test_optical_system()
        positions = reference_collector_positions(np.array([0.7]) * u.rad)
        wavelengths = np.array([10.0]) * u.um
        widths = np.array([0.3]) * u.um
        sky = np.array([[100.0, 0.0]]) * u.mas
        radiance = np.full((1, 1), 100.0) * u.photon / (u.s * u.m**2 * u.um * u.sr)
        with self.assertRaisesRegex(ValueError, "LIFE_fov_boundary"):
            life.get_extended_background_rates(
                positions,
                wavelengths,
                widths,
                sky,
                radiance,
                np.array([1.0e-12]) * u.sr,
                {**life.observingModes[0], "LIFE_fov_boundary": "hexagon"},
            )

    def test_fourier_invalid_radial_grid(self):
        """Reject invalid Fourier radial sampling."""
        optical = LIFEOpticalSystem.__new__(LIFEOpticalSystem)
        optical.pupilDiam = 3.0 * u.m
        optical.pupilArea = np.pi * (1.5 * u.m) ** 2

        positions = reference_collector_positions(np.array([0.0]) * u.rad)[0]

        wavelengths = np.array([10.0, 12.0]) * u.um
        widths = np.array([0.3, 0.36]) * u.um

        intensity = np.ones((3, 2)) * (u.photon / (u.s * u.m**2 * u.um * u.sr))

        mode = {
            "LIFE_fov_boundary": "circle",
            "LIFE_fov_taper": "none",
        }

        for radii in (
            [-1e-6, 0.0, 2e-6],
            [0.0, 1e-6, np.inf],
            [0.0, 2e-6, 1e-6],
            [0.0, 1e-6, 1e-6],
        ):
            with self.subTest(radii=radii):
                with self.assertRaisesRegex(
                    ValueError, "Invalid radial source sampling"
                ):
                    optical.get_axisymmetric_background_rates(
                        positions,
                        wavelengths,
                        widths,
                        np.array(radii) * u.rad,
                        intensity,
                        mode,
                    )

    def test_fourier_uniform_annulus(self):
        """Compare Fourier quadrature with an analytical annulus."""
        from scipy.special import j1

        optical = LIFEOpticalSystem.__new__(LIFEOpticalSystem)
        optical.pupilDiam = 3.0 * u.m
        optical.pupilArea = np.pi * (1.5 * u.m) ** 2

        positions = reference_collector_positions(np.array([0.0]) * u.rad)[0]

        wavelengths = np.array([10.0, 12.0]) * u.um
        widths = np.array([0.3, 0.36]) * u.um

        mode = {
            "LIFE_fov_boundary": "circle",
            "LIFE_fov_taper": "none",
            "attenuation": 0.05,
            "inst": {"QE": lambda lam: (np.ones(len(np.atleast_1d(lam))) / u.photon)},
        }

        inner = 1.8e-6
        outer = wavelengths[1].to_value(u.m) / (2.0 * optical.pupilDiam.to_value(u.m))
        radii = np.linspace(inner, outer, 8193)

        intensity = np.full((len(radii), 2), 1e12) * (
            u.photon / (u.s * u.m**2 * u.um * u.sr)
        )

        actual = optical.get_axisymmetric_background_rates(
            positions,
            wavelengths,
            widths,
            radii * u.rad,
            intensity,
            mode,
        )

        p = positions.to_value(u.m)
        baselines = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)

        k = 2 * np.pi * baselines / wavelengths[1].to_value(u.m)

        integral = np.full_like(k, np.pi * (outer**2 - inner**2))

        nonzero = k != 0
        q = k[nonzero]

        integral[nonzero] = (
            2 * np.pi * (outer * j1(q * outer) - inner * j1(q * inner)) / q
        )

        for output, phases in enumerate(
            (
                [0, np.pi / 2, np.pi, 3 * np.pi / 2],
                [0, 3 * np.pi / 2, np.pi, np.pi / 2],
            )
        ):
            np.testing.assert_array_equal(actual[output].value[:, 0], 0.0)

            phases = np.asarray(phases)
            weights = np.cos(phases[:, None] - phases[None, :]) / 16.0

            expected = (
                np.sum(weights * integral)
                * u.sr
                * intensity[0, 1]
                * optical.total_collecting_area
                * widths[1]
                * mode["attenuation"]
                * mode["inst"]["QE"](wavelengths)[1]
            ).to_value(1 / u.s)

            self.assertGreater(expected, 0.0)

            np.testing.assert_allclose(
                actual[output].value[0, 1],
                expected,
                rtol=2e-5,
                atol=0,
            )


if __name__ == "__main__":
    unittest.main()
