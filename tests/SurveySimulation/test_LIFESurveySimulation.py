"""Tests for the LIFE survey simulation interface."""

from types import SimpleNamespace

import astropy.units as u
import numpy as np
import pytest

from EXOSIMS.OpticalSystem.LIFEOpticalSystem import LIFEOpticalSystem
from EXOSIMS.Prototypes.SurveySimulation import SurveySimulation
from EXOSIMS.SurveySimulation.LIFESurveySimulation import (
    LIFESurveySimulation,
)
from EXOSIMS.ZodiacalLight.LIFEZodiacalLight import (
    LIFEZodiacalLight,
)


@pytest.fixture
def survey():
    """Construct a lightweight survey with real LIFE source physics."""
    sim = LIFESurveySimulation.__new__(LIFESurveySimulation)

    optical_system = LIFEOpticalSystem.__new__(LIFEOpticalSystem)
    diameter = 3.0 * u.m
    optical_system.pupilArea = np.pi * (diameter / 2) ** 2

    sim.OpticalSystem = optical_system
    sim.ZodiacalLight = LIFEZodiacalLight(verbose=False)

    sim.TargetList = SimpleNamespace(
        L=np.array([1.0]),
        dist=np.array([10.0]) * u.pc,
    )

    sim.SimulatedUniverse = SimpleNamespace(
        plan2star=np.array([0, 0]),
        nEZ=np.array([1.0, 1.0]),
    )

    reference_positions = np.array(
        [
            [-7.25, -43.5],
            [-7.25, 43.5],
            [7.25, -43.5],
            [7.25, 43.5],
        ]
    )

    def collector_positions(
        rotation_angles,
        nulling_baseline=None,
    ):
        """Supply reference X-array positions at requested rotations."""
        angles = np.atleast_1d(rotation_angles.to_value(u.rad))

        if nulling_baseline is not None:
            baseline = nulling_baseline.to_value(u.m)
            positions = reference_positions * (baseline / 14.5)
        else:
            positions = reference_positions

        cosine = np.cos(angles)[:, None]
        sine = np.sin(angles)[:, None]

        x = positions[:, 0][None, :]
        y = positions[:, 1][None, :]

        rotated_x = cosine * x - sine * y
        rotated_y = sine * x + cosine * y

        return (
            np.stack(
                (rotated_x, rotated_y),
                axis=-1,
            )
            * u.m
        )

    sim.Observatory = SimpleNamespace(get_collector_positions=collector_positions)

    return sim


@pytest.fixture
def observation():
    """Create a small mid-infrared observation configuration."""
    wavelengths = np.array([10.0, 12.0]) * u.um
    bin_widths = np.array([0.3, 0.36]) * u.um

    sky_positions, pixel_solid_angles = LIFESurveySimulation.make_life_sky_grid(
        half_width=3.0e-6 * u.rad,
        pixels_per_axis=32,
    )

    mode = {
        "attenuation": 0.05,
        "inst": {
            "QE": lambda lam: (np.ones(len(np.atleast_1d(lam))) / u.photon),
        },
    }

    return {
        "sInd": 0,
        "wavelengths": wavelengths,
        "bin_widths": bin_widths,
        "sky_positions": sky_positions,
        "pixel_solid_angles": pixel_solid_angles,
        "rotation_angles": np.array([0.0, 90.0]) * u.deg,
        "mode": mode,
        "local_zodi_longitude": 135.0 * u.deg,
        "local_zodi_latitude": 45.0 * u.deg,
    }


def test_inherits_survey_simulation():
    """Check compatibility with the EXOSIMS survey interface."""
    assert issubclass(LIFESurveySimulation, SurveySimulation)


def test_life_sky_grid():
    """Check sky-grid dimensions and total solid angle."""
    half_width = 3.0 * u.arcsec

    sky, solid_angles = LIFESurveySimulation.make_life_sky_grid(
        half_width=half_width,
        pixels_per_axis=20,
    )

    assert sky.shape == (400, 2)
    assert solid_angles.shape == (400,)

    expected_solid_angle = 4.0 * half_width.to_value(u.rad) ** 2

    assert solid_angles.sum().to_value(u.sr) == pytest.approx(expected_solid_angle)


def test_zodiacal_rate_shapes(survey, observation):
    """Check output shapes and photon-rate units."""
    rates = survey.get_life_zodiacal_rates(**observation)

    assert set(rates) == {
        "local_left",
        "local_right",
        "exozodi_left",
        "exozodi_right",
    }

    for rate in rates.values():
        assert rate.shape == (2, 2)
        assert rate.unit.is_equivalent(1 / u.s)
        assert np.all(np.isfinite(rate.value))
        assert np.all(rate.value >= 0)


def test_local_zodi_positive(survey, observation):
    """Check the local zodiacal contribution is nonzero."""
    rates = survey.get_life_zodiacal_rates(**observation)

    assert np.all(rates["local_left"] > 0 / u.s)
    assert np.all(rates["local_right"] > 0 / u.s)


def test_exozodi_scaling(survey, observation):
    """Check that exozodi rates scale linearly with dust level."""
    one_zodi = survey.get_life_zodiacal_rates(
        **observation,
        nEZ=1.0,
    )

    three_zodis = survey.get_life_zodiacal_rates(
        **observation,
        nEZ=3.0,
    )

    for key in ("exozodi_left", "exozodi_right"):
        np.testing.assert_allclose(
            three_zodis[key].value,
            3.0 * one_zodi[key].value,
            rtol=1e-12,
            atol=0,
        )

    for key in ("local_left", "local_right"):
        np.testing.assert_allclose(
            three_zodis[key].value,
            one_zodi[key].value,
            rtol=1e-12,
            atol=0,
        )


def test_zero_exozodi(survey, observation):
    """Check zero dust level gives zero exozodiacal emission."""
    rates = survey.get_life_zodiacal_rates(
        **observation,
        nEZ=0.0,
    )

    np.testing.assert_allclose(
        rates["exozodi_left"].value,
        0.0,
        atol=1e-14,
    )

    np.testing.assert_allclose(
        rates["exozodi_right"].value,
        0.0,
        atol=1e-14,
    )

    assert np.all(rates["local_left"] > 0 / u.s)


def test_automatic_system_nez(survey, observation):
    """Check extraction of a shared system-level dust value."""
    automatic = survey.get_life_zodiacal_rates(**observation)

    explicit = survey.get_life_zodiacal_rates(
        **observation,
        nEZ=1.0,
    )

    for key in automatic:
        np.testing.assert_allclose(
            automatic[key].value,
            explicit[key].value,
            rtol=1e-12,
            atol=0,
        )


def test_reject_inconsistent_system_nez(survey, observation):
    """Reject ambiguous dust levels assigned to the same system."""
    survey.SimulatedUniverse.nEZ = np.array([1.0, 3.0])

    with pytest.raises(ValueError, match="Multiple nEZ"):
        survey.get_life_zodiacal_rates(**observation)


def test_reject_missing_system_nez(survey, observation):
    """Require explicit dust level when no planet record exists."""
    survey.SimulatedUniverse.plan2star = np.array([1, 1])

    with pytest.raises(ValueError, match="No planet-associated"):
        survey.get_life_zodiacal_rates(**observation)


def test_reject_invalid_star_index(survey, observation):
    """Reject indices outside the target list."""
    arguments = observation.copy()
    arguments["sInd"] = 4

    with pytest.raises(ValueError, match="valid target"):
        survey.get_life_zodiacal_rates(**arguments)


# ================================================================
# LIFE spectral detection assembly
# ================================================================


class _DetectionGeometry:
    def get_collector_positions(self, rotation_angles, *, nulling_baseline=None):
        b = 14.5 if nulling_baseline is None else nulling_baseline.to_value(u.m)
        p = np.array(
            [
                [-b / 2, -3 * b],
                [-b / 2, 3 * b],
                [b / 2, -3 * b],
                [b / 2, 3 * b],
            ]
        )
        a = np.atleast_1d(rotation_angles.to_value(u.rad))
        c, s = np.cos(a)[:, None], np.sin(a)[:, None]
        return (
            np.stack(
                (
                    c * p[None, :, 0] - s * p[None, :, 1],
                    s * p[None, :, 0] + c * p[None, :, 1],
                ),
                axis=-1,
            )
            * u.m
        )


def _detection_case(ez_scale=1.0):
    survey = LIFESurveySimulation.__new__(LIFESurveySimulation)
    survey.OpticalSystem = LIFEOpticalSystem(
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
        starlightSuppressionSystems=[{"name": "nuller", "lam": 10000, "optics": 0.75}],
        observingModes=[
            {
                "instName": "LIFE-spectrometer",
                "systName": "nuller",
                "SNR": 5.0,
                "LIFE_fov_taper": "gaussian",
            }
        ],
    )
    survey.Observatory = _DetectionGeometry()

    def background_rates(**_):
        return {
            "local_left": np.array([[6.0, 8.0], [7.0, 9.0]]) / u.s,
            "local_right": np.array([[5.0, 7.0], [6.0, 8.0]]) / u.s,
            "exozodi_left": ez_scale * np.array([[4.0, 5.0], [6.0, 7.0]]) / u.s,
            "exozodi_right": ez_scale * np.array([[3.0, 4.0], [5.0, 6.0]]) / u.s,
        }

    # Spatial dust integration is covered by separate existing tests.
    survey.get_life_zodiacal_rates = background_rates

    f = u.photon / (u.s * u.m**2 * u.um)
    inputs = dict(
        sInd=0,
        wavelengths=np.array([10.0, 12.0]) * u.um,
        bin_widths=np.array([0.3, 0.36]) * u.um,
        planet_sky_positions=np.array([[100.0, 40.0], [-70.0, 85.0]]) * u.mas,
        planet_spectral_fluxes=np.array([[200.0, 170.0], [90.0, 160.0]]) * f,
        stellar_angular_radius=0.5 * u.mas,
        stellar_spectral_flux=np.array([1e8, 8e7]) * f,
        sky_positions=np.array([[0.0, 0.0], [10.0, 0.0]]) * u.mas,
        pixel_solid_angles=np.array([1e-13, 1e-13]) * u.sr,
        rotation_angles=np.array([0.0, 0.7]) * u.rad,
        mode=survey.OpticalSystem.observingModes[0],
        local_zodi_longitude=135.0 * u.deg,
        local_zodi_latitude=30.0 * u.deg,
        integration_time=600.0 * u.s,
        detector_pixels_per_output=3.0,
        nEZ=1.0,
    )
    return survey, inputs


def test_life_detection_metrics_manual_photon_budget():
    """Verify the assembled detection signal and photon-noise budget."""
    survey, inputs = _detection_case()
    result = survey.get_life_detection_metrics(**inputs)

    def rate(key):
        return result[key].to_value(1 / u.s)

    assert result["signal_rates"].shape == (2, 4)
    assert result["noise_variance_rates"].shape == (2, 4)

    pl, pr = rate("planet_left"), rate("planet_right")

    background = sum(
        rate(key)
        for key in (
            "stellar_left",
            "stellar_right",
            "local_left",
            "local_right",
            "exozodi_left",
            "exozodi_right",
        )
    )
    detector = rate("detector_dark_rate") + rate("detector_read_variance_rate")
    background += 2 * detector[None, :]

    # Each candidate sees photon noise from ALL planets.
    expected_signal = (0.5 * (pl - pr)).transpose(1, 0, 2).reshape(2, 4)

    expected_variance = (
        np.broadcast_to(
            0.5 * ((pl + pr).sum(axis=1) + background)[:, None, :],
            pl.shape,
        )
        .transpose(1, 0, 2)
        .reshape(2, 4)
    )

    np.testing.assert_allclose(rate("signal_rates"), expected_signal)
    np.testing.assert_allclose(rate("noise_variance_rates"), expected_variance)

    information = np.sum(expected_signal**2 / expected_variance, axis=1)
    np.testing.assert_allclose(result["snr"], np.sqrt(600 * information))
    np.testing.assert_allclose(
        result["required_integration_time"].to_value(u.s),
        25 / information,
    )


def test_life_detection_companion_contributes_photon_noise():
    """Verify that companion planets contribute additional photon noise."""
    survey, inputs = _detection_case()
    full = survey.get_life_detection_metrics(**inputs)

    without = dict(inputs)
    without["planet_spectral_fluxes"] = inputs["planet_spectral_fluxes"].copy()
    without["planet_spectral_fluxes"][1] = 0 * u.photon / (u.s * u.m**2 * u.um)

    no_companion = survey.get_life_detection_metrics(**without)

    np.testing.assert_allclose(
        full["signal_rates"][0].to_value(1 / u.s),
        no_companion["signal_rates"][0].to_value(1 / u.s),
    )
    assert np.all(
        full["noise_variance_rates"][0] > no_companion["noise_variance_rates"][0]
    )

def test_life_detection_more_exozodi_reduces_snr():
    """Verify that increasing exozodi reduces SNR and increases detection time."""
    low_sim, low_inputs = _detection_case(ez_scale=1.0)
    high_sim, high_inputs = _detection_case(ez_scale=10.0)

    low = low_sim.get_life_detection_metrics(**low_inputs)
    high = high_sim.get_life_detection_metrics(**high_inputs)

    assert np.all(high["snr"] < low["snr"])
    assert np.all(high["required_integration_time"] > low["required_integration_time"])


def test_life_detection_requires_one_spectrum_per_planet():
    """Reject inconsistent planet spectrum dimensions."""
    survey, inputs = _detection_case()
    inputs["planet_spectral_fluxes"] = (
        np.ones((2, 3)) * u.photon / (u.s * u.m**2 * u.um)
    )

    with pytest.raises(ValueError, match="planet_spectral_fluxes"):
        survey.get_life_detection_metrics(**inputs)
