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
