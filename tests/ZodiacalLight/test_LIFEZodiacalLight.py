"""Tests for the LIFE zodiacal-light source model."""

import astropy.units as u
import numpy as np
import pytest
from astropy.constants import c, h, k_B

from EXOSIMS.ZodiacalLight.LIFEZodiacalLight import (
    LIFEZodiacalLight,
)
from EXOSIMS.ZodiacalLight.Mennesson import Mennesson


@pytest.fixture
def zodi():
    """Create a LIFE zodiacal-light model for testing."""
    return LIFEZodiacalLight(verbose=False)


def sky_at_radius(radius_au, distance=10 * u.pc):
    """Return one sky position corresponding to a physical radius."""
    theta = (radius_au * u.au / distance).decompose().value
    return np.array([[theta, 0.0]]) * u.rad


def test_inherits_mennesson(zodi):
    """Verify inheritance from the Mennesson model."""
    assert isinstance(zodi, Mennesson)
    assert callable(zodi.gen_systemnEZ)
    assert callable(zodi.zodi_intensity_at_location)


def test_exozodi_radiance_units_and_shape(zodi):
    """Verify spectral radiance units, shape, and positivity."""
    sky = np.array([[0.0, 0.0], [1e-6, 0.0]]) * u.rad
    wavelengths = np.array([5.0, 10.0, 15.0]) * u.um

    radiance = zodi.get_exozodi_spectral_radiance(
        sky_positions=sky,
        wavelengths=wavelengths,
        nEZ=1.0,
        star_distance=10 * u.pc,
        star_luminosity=1.0,
    )

    expected_unit = u.photon / (u.s * u.m**2 * u.um * u.sr)

    assert radiance.shape == (2, 3)
    assert radiance.unit.is_equivalent(expected_unit)
    assert np.all(np.isfinite(radiance.value))
    assert np.all(radiance.value >= 0)


def test_exozodi_reference_at_one_au(zodi):
    """Check the Kennedy model independently at 1 AU around the Sun."""
    wavelength = 10.0 * u.um

    radiance = zodi.get_exozodi_spectral_radiance(
        sky_positions=sky_at_radius(1.0),
        wavelengths=np.array([10.0]) * u.um,
        nEZ=1.0,
        star_distance=10 * u.pc,
        star_luminosity=1.0,
    )

    exponent = (h * c / (wavelength * k_B * (278.3 * u.K))).decompose().value

    expected = (
        7.11889e-8 * 2.0 * c * u.photon / (wavelength**4 * u.sr * np.expm1(exponent))
    ).to(u.photon / (u.s * u.m**2 * u.um * u.sr))

    assert radiance[0, 0].to_value(expected.unit) == pytest.approx(
        expected.value,
        rel=1e-10,
    )


def test_exozodi_scales_linearly_with_nez(zodi):
    """Verify linear scaling with exozodiacal dust level."""
    arguments = dict(
        sky_positions=sky_at_radius(1.0),
        wavelengths=np.array([8.0, 12.0]) * u.um,
        star_distance=10 * u.pc,
        star_luminosity=1.0,
    )

    one_zodi = zodi.get_exozodi_spectral_radiance(nEZ=1.0, **arguments)
    three_zodis = zodi.get_exozodi_spectral_radiance(nEZ=3.0, **arguments)

    np.testing.assert_allclose(
        three_zodis.value,
        3.0 * one_zodi.value,
        rtol=1e-12,
    )


def test_exozodi_inner_edge(zodi):
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

    radiance = zodi.get_exozodi_spectral_radiance(
        sky_positions=sky,
        wavelengths=np.array([10.0]) * u.um,
        nEZ=1.0,
        star_distance=10 * u.pc,
        star_luminosity=1.0,
    )

    assert radiance.value[0, 0] == 0
    assert radiance.value[1, 0] == 0
    assert radiance.value[2, 0] > 0


def test_no_additional_outer_temperature_cutoff(zodi):
    """Cold outer dust remains present in the source model."""
    radiance = zodi.get_exozodi_spectral_radiance(
        sky_positions=sky_at_radius(20.0),
        wavelengths=np.array([15.0]) * u.um,
        nEZ=1.0,
        star_distance=10 * u.pc,
        star_luminosity=1.0,
    )

    assert radiance.value[0, 0] > 0


def test_invalid_exozodi_level(zodi):
    """Reject negative exozodiacal dust levels."""
    with pytest.raises(ValueError):
        zodi.get_exozodi_spectral_radiance(
            sky_positions=sky_at_radius(1.0),
            wavelengths=np.array([10.0]) * u.um,
            nEZ=-1.0,
            star_distance=10 * u.pc,
            star_luminosity=1.0,
        )
