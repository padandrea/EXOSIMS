"""Geometry and EXOSIMS integration tests for the LIFE observatory."""

import astropy.units as u
import numpy as np
import pytest

from EXOSIMS.Observatory.LIFEObservatory import LIFEObservatory
from EXOSIMS.Observatory.ObservatoryL2Halo import ObservatoryL2Halo
from EXOSIMS.OpticalSystem.LIFEOpticalSystem import LIFEOpticalSystem
from EXOSIMS.Prototypes.Observatory import Observatory


@pytest.fixture(scope="module")
def observatory(tmp_path_factory):
    """Construct one real L2 observatory using static ephemerides."""
    cache_dir = tmp_path_factory.mktemp("life_observatory")
    return LIFEObservatory(
        cachedir=str(cache_dir), forceStaticEphem=True, verbose=False
    )


def test_reference_geometry_and_collector_order(observatory):
    """Check all four reference coordinates without permitting reordering."""
    positions = observatory.get_collector_positions(0.0 * u.deg)
    expected = np.array(
        [
            [-7.25, -43.5],
            [-7.25, 43.5],
            [7.25, -43.5],
            [7.25, 43.5],
        ]
    )
    assert isinstance(positions, u.Quantity)
    assert positions.unit == u.m
    assert positions.shape == (1, 4, 2)
    np.testing.assert_allclose(positions[0].to_value(u.m), expected, atol=1e-14)
    np.testing.assert_allclose(positions.sum(axis=1).to_value(u.m), 0, atol=1e-14)


def test_reference_baselines_and_outspec(observatory):
    """Verify the nulling and imaging baselines and serializable settings."""
    coords = observatory.get_collector_positions()[0]
    np.testing.assert_allclose(
        np.linalg.norm((coords[0] - coords[2]).to_value(u.m)), 14.5
    )
    np.testing.assert_allclose(
        np.linalg.norm((coords[0] - coords[1]).to_value(u.m)), 87.0
    )
    assert observatory.nulling_baseline == 14.5 * u.m
    assert observatory.imaging_baseline == 87.0 * u.m
    assert observatory.baseline_ratio == pytest.approx(6.0)
    assert observatory._outspec["nulling_baseline"] == pytest.approx(14.5)
    assert observatory._outspec["baseline_ratio"] == pytest.approx(6.0)


def test_configurable_baselines_and_overheads(observatory):
    """Check unit conversion, ratio configuration, and EXOSIMS settling time."""
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
    assert custom.nulling_baseline == 20 * u.m
    assert custom.imaging_baseline == 60 * u.m
    assert custom.settlingTime == 0.125 * u.d
    assert observatory.settlingTime == 0 * u.d


def test_rotation_zero_ninety_and_full_turn(observatory):
    """A positive 90-degree roll maps (x, y) to (-y, x)."""
    positions = observatory.get_collector_positions(
        np.array([0.0, 90.0, 360.0]) * u.deg
    ).to_value(u.m)
    assert positions.shape == (3, 4, 2)
    ninety_expected = np.column_stack((-positions[0, :, 1], positions[0, :, 0]))
    np.testing.assert_allclose(positions[1], ninety_expected, atol=1e-12)
    np.testing.assert_allclose(positions[2], positions[0], atol=1e-12)


def test_scalar_and_vector_rotations(observatory):
    """Single and batched rotations use the same three-dimensional shape."""
    scalar = observatory.get_collector_positions(30.0 * u.deg)
    vector = observatory.get_collector_positions(np.array([30.0]) * u.deg)
    assert scalar.shape == vector.shape == (1, 4, 2)
    np.testing.assert_allclose(scalar.to_value(u.m), vector.to_value(u.m))


def test_rotation_preserves_all_pairwise_distances(observatory):
    """Rigid planar rotations must preserve all six collector separations."""
    positions = observatory.get_collector_positions(
        np.array([0.0, 23.0, 90.0, 180.0, 359.0]) * u.deg
    ).to_value(u.m)
    differences = positions[:, :, None, :] - positions[:, None, :, :]
    separations = np.linalg.norm(differences, axis=-1)
    np.testing.assert_allclose(
        separations, np.broadcast_to(separations[0], separations.shape), atol=1e-12
    )


@pytest.mark.parametrize(
    ("baseline", "error"),
    [
        (0.0, ValueError),
        (-14.5, ValueError),
        (np.nan, ValueError),
        (np.inf, ValueError),
        (2.0 * u.s, ValueError),
        (np.array([14.5, 20.0]) * u.m, ValueError),
        ("14.5", TypeError),
        (True, TypeError),
    ],
)
def test_invalid_nulling_baselines(baseline, error):
    """Reject invalid lengths before running the halo-orbit constructor."""
    with pytest.raises(error):
        LIFEObservatory(nulling_baseline=baseline)


@pytest.mark.parametrize(
    ("ratio", "error"),
    [
        (0.0, ValueError),
        (-6.0, ValueError),
        (np.nan, ValueError),
        (np.inf, ValueError),
        ("6", TypeError),
        (6.0 * u.m, TypeError),
        (False, TypeError),
    ],
)
def test_invalid_baseline_ratios(ratio, error):
    """Reject invalid ratios before running the halo-orbit constructor."""
    with pytest.raises(error):
        LIFEObservatory(baseline_ratio=ratio)


@pytest.mark.parametrize(
    ("angles", "error"),
    [
        (90.0, TypeError),
        (90.0 * u.m, ValueError),
        (np.array([]) * u.deg, ValueError),
        (np.array([0.0, np.nan]) * u.deg, ValueError),
        (np.array([0.0, np.inf]) * u.deg, ValueError),
        (np.zeros((2, 2)) * u.deg, ValueError),
    ],
)
def test_invalid_rotation_inputs(observatory, angles, error):
    """Reject ambiguous units, malformed shapes, and nonfinite angles."""
    with pytest.raises(error):
        observatory.get_collector_positions(angles)


def test_inherited_orbit_and_keepout_interfaces(observatory):
    """Verify that original observatory behavior remains inherited."""
    assert isinstance(observatory, ObservatoryL2Halo)
    assert isinstance(observatory, Observatory)
    assert LIFEObservatory.orbit is ObservatoryL2Halo.orbit
    assert LIFEObservatory.keepout is Observatory.keepout
    assert LIFEObservatory.calculate_observableTimes is (
        Observatory.calculate_observableTimes
    )


def test_geometry_with_life_transmission(observatory):
    """Verify geometry shape, null, and nonzero chopped off-axis response."""
    optics = LIFEOpticalSystem.__new__(LIFEOpticalSystem)
    positions = observatory.get_collector_positions(
        np.array([0.0, 60.0, 120.0]) * u.deg
    )
    sky = np.array([[0.0, 0.0], [4.848e-7, 1.0e-7]]) * u.rad
    wavelengths = np.array([10.0, 12.0]) * u.um

    left, right, chopped = optics.get_transmission(positions, wavelengths, sky)
    for output in (left, right, chopped):
        assert output.shape == (3, 2, 2)
        assert output.unit == u.dimensionless_unscaled
        np.testing.assert_allclose(output[:, 0, :].value, 0, atol=1e-14)
    np.testing.assert_allclose(chopped.value, (left - right).value, atol=1e-14)
    assert np.any(np.abs(chopped[:, 1, :].value) > 1e-5)
