"""Tests for the LIFE nulling interferometer optical system."""

import json
import numpy as np

import astropy.units as u
import pytest

from EXOSIMS.OpticalSystem.LIFEOpticalSystem import LIFEOpticalSystem
from tests.TestSupport.Info import resource_path


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


def test_total_collecting_area():
    """Verify the combined area of four equal collectors."""
    life = LIFEOpticalSystem(
        pupilDiam=2.0,
        obscurFac=0.0,
    )

    # Each 2 m collector has an area of pi m².
    # Four collectors therefore have an area of 4*pi m².
    assert life.total_collecting_area.to_value(u.m**2) == pytest.approx(4.0 * np.pi)


def test_mid_ir_observing_mode():
    """Verify that LIFE supports a 15 um observing mode."""
    path = resource_path("test-scripts/template_minimal.json")

    with open(path, encoding="utf-8") as file:
        specs = json.load(file)

    specs["starlightSuppressionSystems"][0]["lam"] = 15000

    optical_system = LIFEOpticalSystem(**specs)

    assert optical_system.observingModes[0]["lam"].to_value(u.um) == pytest.approx(15.0)


def test_get_spectral_bins_reference():
    """Verify geometric spectral binning at constant resolving power."""
    life = LIFEOpticalSystem.__new__(LIFEOpticalSystem)

    centers, widths = life.get_spectral_bins(
        4.0 * u.um,
        18.5 * u.um,
        33.3,
    )

    assert len(centers) == 52
    assert len(widths) == 52

    # The first bin starts at 4 um.
    lower_edges = centers - widths / 2
    upper_edges = centers + widths / 2

    np.testing.assert_allclose(
        lower_edges[0].to_value(u.um),
        4.0,
        atol=1e-12,
    )

    # The final bin ends exactly at the upper wavelength limit.
    assert lower_edges[-1] < 18.5 * u.um
    np.testing.assert_allclose(
        upper_edges[-1].to_value(u.um),
        18.5,
        atol=1e-12,
    )

    # Check the geometric spacing.
    ratios = lower_edges[1:] / lower_edges[:-1]

    np.testing.assert_allclose(
        ratios.to_value(u.dimensionless_unscaled),
        1.0 + 1.0 / 33.3,
        rtol=1e-12,
    )

    assert np.all(widths > 0 * u.m)


def test_get_spectral_bins_from_mode():
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


def test_life_explicit_wavelength_limits():
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

    assert mode["lam"].to_value(u.um) == pytest.approx(11.25)
    assert mode["deltaLam"].to_value(u.um) == pytest.approx(14.5)
    assert mode["inst"]["Rs"] == pytest.approx(33.3)

    centers, widths = life.get_spectral_bins_from_mode(mode)

    lower_edges = centers - widths / 2
    upper_edges = centers + widths / 2

    assert lower_edges[0].to_value(u.um) == pytest.approx(4.0)
    assert upper_edges[-1].to_value(u.um) == pytest.approx(18.5)


def test_life_wavelength_limits_require_both_keys():
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

    with pytest.raises(ValueError, match="must be specified together"):
        LIFEOpticalSystem(**specs)


def test_transmission_shape_and_on_axis_null():
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
        assert transmission.shape == (2, 1, 2)
        assert transmission.unit == u.dimensionless_unscaled
        np.testing.assert_allclose(transmission.value, 0.0, atol=1e-14)


def test_transmission_matches_fourier_pair_sum():
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


def test_transmission_mirror_symmetry_and_odd_chop():
    """The right output mirrors the left; the chopped response is odd."""
    life = LIFEOpticalSystem.__new__(LIFEOpticalSystem)

    sky = np.array([[4.848e-7, 0.0], [-4.848e-7, 0.0]]) * u.rad
    rotations = np.linspace(0.0, 2.0 * np.pi, 13, endpoint=False) * u.rad
    positions = reference_collector_positions(rotations)

    left, right, chopped = life.get_transmission(positions, 10.0 * u.um, sky)

    np.testing.assert_allclose(left[:, 0, :].value, right[:, 1, :].value, atol=1e-14)
    np.testing.assert_allclose(
        chopped[:, 0, :].value, -chopped[:, 1, :].value, atol=1e-14
    )
    assert np.all(left.value >= 0.0)
    assert np.all(right.value >= 0.0)
    assert np.all(left.value <= 1.0 + 1e-14)
    assert np.all(right.value <= 1.0 + 1e-14)


def test_transmission_changes_with_collector_geometry():
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

    assert not np.isclose(
        left_6.value[0, 0, 0],
        left_8.value[0, 0, 0],
    )


def test_transmission_rejects_invalid_inputs():
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
    with pytest.raises(TypeError, match="collector_positions"):
        life.get_transmission(positions.value, wavelength, sky)

    with pytest.raises(ValueError, match="length units"):
        life.get_transmission(positions.value * u.rad, wavelength, sky)

    with pytest.raises(ValueError, match="collector_positions"):
        life.get_transmission(np.zeros((3, 2)) * u.m, wavelength, sky)

    with pytest.raises(ValueError, match="collector_positions"):
        life.get_transmission(np.full((4, 2), np.nan) * u.m, wavelength, sky)

    # Wavelengths must be positive physical quantities.
    with pytest.raises(TypeError, match="wavelengths"):
        life.get_transmission(positions, 10.0, sky)

    with pytest.raises(ValueError, match="wavelengths"):
        life.get_transmission(positions, -10.0 * u.um, sky)

    # Sky coordinates must have angular units and the correct shape.
    with pytest.raises(ValueError, match="sky_positions"):
        life.get_transmission(positions, wavelength, np.array([0, 0, 0]) * u.rad)

    with pytest.raises(ValueError, match="angular units"):
        life.get_transmission(positions, wavelength, np.array([0, 0]) * u.m)


def test_planet_photon_rates():
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

    assert rate_left.shape == (1, 1, 2)
    assert rate_right.shape == (1, 1, 2)

    np.testing.assert_allclose(
        rate_left.to_value(1 / u.s),
        (transmission_left * expected_rate).to_value(1 / u.s),
    )

    np.testing.assert_allclose(
        rate_right.to_value(1 / u.s),
        (transmission_right * expected_rate).to_value(1 / u.s),
    )


def test_stellar_leakage_rates():
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

    left_map, right_map, _ = life.get_transmission(positions, wavelengths, sky_points)

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

    assert left_rate.shape == (2, 2)
    assert right_rate.shape == (2, 2)

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

    assert np.all(left_rate > 0 * (1 / u.s))


def test_extended_background_rates():
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

    assert rate_left.shape == (2, 2)
    assert rate_right.shape == (2, 2)

    np.testing.assert_allclose(
        rate_left.to_value(1 / u.s),
        expected_left.to_value(1 / u.s),
    )
    np.testing.assert_allclose(
        rate_right.to_value(1 / u.s),
        expected_right.to_value(1 / u.s),
    )

    assert np.all(rate_left >= 0 / u.s)
    assert np.all(rate_right >= 0 / u.s)


def test_detector_noise_rates():
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

    assert dark.shape == (2,)
    assert read.shape == (2,)

    scalar_dark, scalar_read = life.get_detector_noise_rates(
        wavelengths, 3.0, life.observingModes[0]
    )

    np.testing.assert_allclose(scalar_dark.to_value(1 / u.s), [0.06, 0.06])
    np.testing.assert_allclose(scalar_read.to_value(1 / u.s), [0.30, 0.30])


def test_detector_noise_rejects_invalid_inputs():
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

    with pytest.raises(ValueError, match="n_pixels_per_output"):
        life.get_detector_noise_rates(wavelengths, np.array([2.0, 0.0]), mode)

    with pytest.raises(ValueError, match="n_pixels_per_output"):
        life.get_detector_noise_rates(wavelengths, np.array([2.0, 3.0, 4.0]), mode)

    bad_mode = {"inst": {**mode["inst"], "texp": 0.0 * u.s}}

    with pytest.raises(ValueError, match="texp"):
        life.get_detector_noise_rates(wavelengths, 3.0, bad_mode)

    bad_mode = {"inst": {**mode["inst"], "sread": -1.0}}

    with pytest.raises(ValueError, match="sread"):
        life.get_detector_noise_rates(wavelengths, 3.0, bad_mode)


def test_assemble_life_signal_noise():
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

    assert signal.shape == (1, 2)
    assert variance.shape == (1, 2)

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

    assert actual_snr.shape == (1,)
    assert actual_snr[0] == pytest.approx(expected_snr)

    # Invalid exposure allocation must be rejected.
    with pytest.raises(ValueError, match="sum to one"):
        life.assemble_life_signal_noise(
            planet_left,
            planet_right,
            stellar_left,
            stellar_right,
            rotation_fractions=np.array([0.4, 0.4]),
        )


def test_calc_life_snr():
    """Verify photon-limited and systematic-limited SNR."""
    life = LIFEOpticalSystem.__new__(LIFEOpticalSystem)

    signal = np.array([3.0, 4.0]) / u.s
    variance = np.array([4.0, 16.0]) / u.s
    time = 10.0 * u.s

    # Photon-noise-limited case:
    # SNR^2 = 9*100/40 + 16*100/160 = 32.5
    snr_photon = life.calc_life_snr(signal, variance, time)

    assert snr_photon == pytest.approx(np.sqrt(32.5))

    # Add an independent systematic floor in each channel.
    systematic = np.array([0.5, 1.0]) / u.s

    snr_systematic = life.calc_life_snr(signal, variance, time, systematic)

    # Denominators: 40 + 25 = 65; 160 + 100 = 260.
    expected_squared = 900.0 / 65.0 + 1600.0 / 260.0

    assert snr_systematic == pytest.approx(np.sqrt(expected_squared))

    assert snr_systematic < snr_photon

    # The systematic floor limits SNR at long times.
    snr_long = life.calc_life_snr(signal, variance, 1.0e10 * u.s, systematic)

    expected_limit = np.sqrt((3.0 / 0.5) ** 2 + (4.0 / 1.0) ** 2)

    assert snr_long == pytest.approx(expected_limit, rel=1e-8)

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

    assert result.shape == (2,)

    np.testing.assert_allclose(
        result,
        [np.sqrt(32.5), np.sqrt(15.0)],
    )


def test_calc_life_snr_invalid_inputs():
    """Reject inconsistent rates and invalid integration times."""
    life = LIFEOpticalSystem.__new__(LIFEOpticalSystem)

    signal = np.array([3.0, 4.0]) / u.s
    variance = np.array([4.0, 16.0]) / u.s

    with pytest.raises(ValueError, match="positive"):
        life.calc_life_snr(signal, variance, 0.0 * u.s)

    with pytest.raises(ValueError, match="shape"):
        life.calc_life_snr(signal, np.array([4.0]) / u.s, 10.0 * u.s)

    with pytest.raises(ValueError, match="nonnegative"):
        life.calc_life_snr(
            signal,
            np.array([4.0, -1.0]) / u.s,
            10.0 * u.s,
        )

    with pytest.raises(ValueError, match="shape"):
        life.calc_life_snr(
            signal,
            variance,
            10.0 * u.s,
            np.array([0.5]) / u.s,
        )

    with pytest.raises(ValueError, match="positive"):
        life.calc_life_snr(
            signal,
            np.zeros(2) / u.s,
            10.0 * u.s,
        )


def test_calc_life_int_time_photon_limited():
    """Verify the analytical photon-noise integration time."""
    life = LIFEOpticalSystem.__new__(LIFEOpticalSystem)

    signal = np.array([3.0, 4.0]) / u.s
    variance = np.array([4.0, 16.0]) / u.s

    required_time = life.calc_life_int_time(signal, variance, snr_threshold=5.0)

    expected = 25.0 / (9.0 / 4.0 + 16.0 / 16.0)

    assert required_time.to_value(u.s) == pytest.approx(expected)

    # Verify that the calculated time actually reaches SNR=5.
    actual_snr = life.calc_life_snr(signal, variance, required_time)

    assert actual_snr == pytest.approx(5.0)

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

    assert times.shape == (2,)

    np.testing.assert_allclose(
        times.to_value(u.s),
        [
            25.0 / 3.25,
            25.0 / 1.5,
        ],
    )


def test_calc_life_int_time_systematics():
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

    assert required_time.to_value(u.s) == pytest.approx(expected)

    actual_snr = life.calc_life_snr(signal, variance, required_time, systematic)

    assert actual_snr == pytest.approx(4.0)

    # Maximum achievable SNR = 3 / 0.5 = 6.
    unreachable = life.calc_life_int_time(
        signal,
        variance,
        snr_threshold=6.0,
        systematic_rates=systematic,
    )

    assert np.isnan(unreachable.to_value(u.s))

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

    assert multi_time.to_value(u.s) == pytest.approx(7.111111111111111)

    assert life.calc_life_snr(
        signals, variances, multi_time, systematics
    ) == pytest.approx(4.0)


def test_calc_life_int_time_invalid_inputs():
    """Test invalid input and zero-signal cases."""
    life = LIFEOpticalSystem.__new__(LIFEOpticalSystem)

    signal = np.array([3.0, 4.0]) / u.s
    variance = np.array([4.0, 16.0]) / u.s

    with pytest.raises(ValueError, match="positive"):
        life.calc_life_int_time(signal, variance, snr_threshold=0.0)

    with pytest.raises(ValueError, match="shape"):
        life.calc_life_int_time(
            signal,
            np.array([4.0]) / u.s,
            snr_threshold=5.0,
        )

    with pytest.raises(ValueError, match="nonnegative"):
        life.calc_life_int_time(
            signal,
            np.array([-1.0, 16.0]) / u.s,
            snr_threshold=5.0,
        )

    with pytest.raises(ValueError, match="positive"):
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

    assert np.isnan(no_detection.to_value(u.s))


def test_life_exosims_scalar_bridge_photon_only():
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

    assert times.unit.is_equivalent(u.day)

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
        assert actual == pytest.approx(5.0)

    with pytest.raises(NotImplementedError, match="returnExtra"):
        life.Cp_Cb_Csp(*args, returnExtra=True)


def test_life_exosims_adapter_systematics():
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

    with pytest.raises(NotImplementedError, match="life_rate_provider"):
        life.calc_intTime(*args)

    life.life_rate_provider = lambda *unused: (
        np.array([[3.0]]) / u.s,
        np.array([[4.0]]) / u.s,
        np.array([[0.5]]) / u.s,
    )

    result = life.calc_intTime(*args)

    # SNR = 4, S = 3/s, V = 4/s, Csp = 0.5/s.
    # t = 16*4 / (9 - 16*0.25) = 12.8 s.
    assert result.to_value(u.s)[0] == pytest.approx(12.8)

    with pytest.raises(NotImplementedError, match="single"):
        life.Cp_Cb_Csp(*args)

    with pytest.raises(NotImplementedError, match="thermal"):
        life.calc_dMag_per_intTime(
            10 * u.s,
            None,
            s_inds,
            None,
            None,
            wa,
            mode,
        )


def test_transmission_energy_conservation():
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
