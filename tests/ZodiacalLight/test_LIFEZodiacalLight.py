"""Tests for the LIFE zodiacal-light source model."""

import unittest

import astropy.units as u
import numpy as np
from astropy.constants import c, h, k_B

from EXOSIMS.ZodiacalLight.LIFEZodiacalLight import LIFEZodiacalLight
from EXOSIMS.ZodiacalLight.Mennesson import Mennesson


def sky_at_radius(radius_au, distance=10 * u.pc):
    """Return one sky position corresponding to a physical radius."""
    theta = (radius_au * u.au / distance).decompose().value
    return np.array([[theta, 0.0]]) * u.rad


class TestLIFEZodiacalLight(unittest.TestCase):
    """Test the LIFE zodiacal-light source model."""

    def setUp(self):
        """Create a fresh LIFE zodiacal-light model for each test."""
        self.zodi = LIFEZodiacalLight(verbose=False)

    def test_inherits_mennesson(self):
        """Verify inheritance from the Mennesson model."""
        self.assertIsInstance(self.zodi, Mennesson)
        self.assertTrue(callable(self.zodi.gen_systemnEZ))
        self.assertTrue(callable(self.zodi.zodi_intensity_at_location))

    def test_exozodi_radiance_units_and_shape(self):
        """Verify spectral radiance units, shape, and positivity."""
        sky = np.array([[0.0, 0.0], [1e-6, 0.0]]) * u.rad
        wavelengths = np.array([5.0, 10.0, 15.0]) * u.um

        radiance = self.zodi.get_exozodi_spectral_radiance(
            sky_positions=sky,
            wavelengths=wavelengths,
            nEZ=1.0,
            star_distance=10 * u.pc,
            star_luminosity=1.0,
        )

        expected_unit = u.photon / (u.s * u.m**2 * u.um * u.sr)

        self.assertEqual(radiance.shape, (2, 3))
        self.assertTrue(radiance.unit.is_equivalent(expected_unit))
        self.assertTrue(np.all(np.isfinite(radiance.value)))
        self.assertTrue(np.all(radiance.value >= 0))

    def test_exozodi_reference_at_one_au(self):
        """Check the Kennedy model independently at 1 AU around the Sun."""
        wavelength = 10.0 * u.um

        radiance = self.zodi.get_exozodi_spectral_radiance(
            sky_positions=sky_at_radius(1.0),
            wavelengths=np.array([10.0]) * u.um,
            nEZ=1.0,
            star_distance=10 * u.pc,
            star_luminosity=1.0,
        )

        exponent = (h * c / (wavelength * k_B * (278.3 * u.K))).decompose().value

        expected = (
            7.11889e-8
            * 2.0
            * c
            * u.photon
            / (wavelength**4 * u.sr * np.expm1(exponent))
        ).to(u.photon / (u.s * u.m**2 * u.um * u.sr))

        np.testing.assert_allclose(
            radiance[0, 0].to_value(expected.unit),
            expected.value,
            rtol=1e-10,
        )

    def test_exozodi_scales_linearly_with_nez(self):
        """Verify linear scaling with exozodiacal dust level."""
        arguments = dict(
            sky_positions=sky_at_radius(1.0),
            wavelengths=np.array([8.0, 12.0]) * u.um,
            star_distance=10 * u.pc,
            star_luminosity=1.0,
        )

        one_zodi = self.zodi.get_exozodi_spectral_radiance(nEZ=1.0, **arguments)
        three_zodis = self.zodi.get_exozodi_spectral_radiance(nEZ=3.0, **arguments)

        np.testing.assert_allclose(
            three_zodis.value,
            3.0 * one_zodi.value,
            rtol=1e-12,
        )

    def test_exozodi_inner_edge(self):
        """Emission must vanish inside the Kennedy inner edge."""
        sky = (
            np.array(
                [
                    [0.0, 0.0],
                    sky_at_radius(0.01)[0].to_value(u.rad),
                    sky_at_radius(0.1)[0].to_value(u.rad),
                ]
            )
            * u.rad
        )

        radiance = self.zodi.get_exozodi_spectral_radiance(
            sky_positions=sky,
            wavelengths=np.array([10.0]) * u.um,
            nEZ=1.0,
            star_distance=10 * u.pc,
            star_luminosity=1.0,
        )

        self.assertEqual(radiance.value[0, 0], 0)
        self.assertEqual(radiance.value[1, 0], 0)
        self.assertGreater(radiance.value[2, 0], 0)

    def test_no_additional_outer_temperature_cutoff(self):
        """Cold outer dust remains present in the source model."""
        radiance = self.zodi.get_exozodi_spectral_radiance(
            sky_positions=sky_at_radius(20.0),
            wavelengths=np.array([15.0]) * u.um,
            nEZ=1.0,
            star_distance=10 * u.pc,
            star_luminosity=1.0,
        )

        self.assertGreater(radiance.value[0, 0], 0)

    def test_local_zodi_mennesson_preserves_existing_model(self):
        """The new default must reproduce the inherited EXOSIMS spectrum."""
        wavelengths = np.array([8.0, 10.0, 15.0]) * u.um
        reference = self.zodi.zodi_intensity_at_location(
            np.array([135.0]) * u.deg,
            np.array([45.0]) * u.deg,
            photon_units=True,
        )[0] * self.zodi.zodi_color_correction_factor(wavelengths, photon_units=True)
        actual = self.zodi.get_local_zodi_spectral_radiance(
            wavelengths, 135 * u.deg, 45 * u.deg
        )
        np.testing.assert_allclose(actual.to_value(reference.unit), reference.value)

    def test_local_zodi_darwinsim_independent_reference(self):
        """Check LIFEsim's 265 K dust and scattered sunlight analytically."""
        from scipy.constants import c as c_si, h as h_si, k as k_si

        wavelength = 10.0e-6
        lon, lat = np.deg2rad([135.0, 45.0])
        thermal = lambda temp: (
            2
            * c_si
            / wavelength**4
            / np.expm1(h_si * c_si / (wavelength * k_si * temp))
            * 1.0e-6  # per metre -> per micrometre
        )
        geometry = np.sqrt(
            np.pi
            / np.arccos(np.cos(lon) * np.cos(lat))
            / (np.sin(lat) ** 2 + (0.6 * (10.0 / 11.0) ** (-0.4) * np.cos(lat)) ** 2)
        )
        expected = (
            4.0e-8
            * geometry
            * (thermal(265) + 0.22 * (0.00465047 / 1.5) ** 2 * thermal(5777))
        )
        actual = self.zodi.get_local_zodi_spectral_radiance(
            np.array([10.0]) * u.um,
            135 * u.deg,
            45 * u.deg,
            model="darwinsim",
        )
        self.assertEqual(actual.shape, (1,))
        np.testing.assert_allclose(
            actual.to_value(u.photon / (u.s * u.m**2 * u.um * u.sr))[0],
            expected,
            rtol=1e-12,
        )

    def test_local_zodi_invalid_model_and_solar_singularity(self):
        """Reject invalid model selections and undefined solar geometry."""
        with self.assertRaisesRegex(ValueError, "model"):
            self.zodi.get_local_zodi_spectral_radiance(
                10.0 * u.um, 135 * u.deg, 45 * u.deg, model="unknown"
            )
        with self.assertRaisesRegex(ValueError, "singular"):
            self.zodi.get_local_zodi_spectral_radiance(
                10.0 * u.um, 0 * u.deg, 0 * u.deg, model="darwinsim"
            )

    def test_invalid_exozodi_level(self):
        """Reject negative exozodiacal dust levels."""
        with self.assertRaises(ValueError):
            self.zodi.get_exozodi_spectral_radiance(
                sky_positions=sky_at_radius(1.0),
                wavelengths=np.array([10.0]) * u.um,
                nEZ=-1.0,
                star_distance=10 * u.pc,
                star_luminosity=1.0,
            )

    def test_inner_radius_scaling(self):
        """Verify the Kennedy inner dust radius scaling."""
        for luminosity in (0.01, 1.0, 4.0, 100.0):
            with self.subTest(luminosity=luminosity):
                expected = 0.034422617777777775 * np.sqrt(luminosity)

                for value in (luminosity, luminosity * u.Lsun):
                    actual = self.zodi.get_exozodi_inner_radius(value)
                    self.assertAlmostEqual(actual.to_value(u.au), expected, places=14)

    def test_exact_dust_boundary(self):
        """Verify the dust-side limit at the inner boundary."""
        distance = 10.0 * u.pc

        for luminosity in (0.01, 1.0, 4.0, 100.0):
            with self.subTest(luminosity=luminosity):
                edge = self.zodi.get_exozodi_inner_radius(luminosity).to_value(
                    u.m
                ) / distance.to_value(u.m)

                radii = edge * np.array([1.0 - 1e-8, 1.0, 1.0 + 1e-8])
                sky = np.column_stack((radii, np.zeros_like(radii))) * u.rad

                rates = self.zodi.get_exozodi_spectral_radiance(
                    sky_positions=sky,
                    wavelengths=np.array([10.0, 12.0]) * u.um,
                    nEZ=1.0,
                    star_distance=distance,
                    star_luminosity=luminosity,
                ).value

                np.testing.assert_array_equal(rates[0], 0.0)
                self.assertTrue(np.all(rates[1:] > 0))
                np.testing.assert_allclose(rates[1], rates[2], rtol=1e-6)
