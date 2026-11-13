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


def test_per_observation_baseline_override_without_limits(observatory):
    """Use a different geometry without changing the Observatory defaults."""
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
    assert positions.shape == (2, 4, 2)
    np.testing.assert_allclose(positions[0].to_value(u.m), expected)
    np.testing.assert_allclose(
        positions[1].to_value(u.m),
        np.column_stack((-expected[:, 1], expected[:, 0])),
        atol=1e-12,
    )
    assert observatory.nulling_baseline == 14.5 * u.m
    assert observatory.imaging_baseline == 87.0 * u.m
    assert observatory.baseline_ratio == 6.0


@pytest.fixture(scope="module")
def bounded_observatory(observatory):
    """Configure explicit, illustrative physical baseline limits."""
    return LIFEObservatory(
        nulling_baseline=14.5 * u.m,
        baseline_ratio=6.0,
        nulling_baseline_range=(10.0 * u.m, 30.0 * u.m),
        imaging_baseline_range=(60.0 * u.m, 180.0 * u.m),
        cachedir=observatory.cachedir,
        forceStaticEphem=True,
        verbose=False,
    )


def test_baseline_range_configuration_and_endpoints(bounded_observatory):
    """Accept inclusive range boundaries and record them in the outspec."""
    obs = bounded_observatory
    assert obs.nulling_baseline_range == (10.0 * u.m, 30.0 * u.m)
    assert obs.imaging_baseline_range == (60.0 * u.m, 180.0 * u.m)
    assert obs._outspec["nulling_baseline_range"] == [10.0, 30.0]
    assert obs._outspec["imaging_baseline_range"] == [60.0, 180.0]
    for value in (10.0, 30.0):
        positions = obs.get_collector_positions(nulling_baseline=value * u.m)
        assert positions.shape == (1, 4, 2)
        np.testing.assert_allclose(
            np.abs(positions[0, 0, 0].to_value(u.m)), value / 2.0
        )


def test_dynamic_ratio_checks_imaging_limit(bounded_observatory):
    """The imaging constraint applies to the product of ratio and baseline."""
    obs = bounded_observatory
    positions = obs.get_collector_positions(
        nulling_baseline=25 * u.m, baseline_ratio=5.0
    )
    np.testing.assert_allclose(
        positions[0, :, 1].to_value(u.m),
        [-62.5, 62.5, -62.5, 62.5],
    )
    with pytest.raises(ValueError, match="imaging baseline"):
        obs.get_collector_positions(nulling_baseline=25 * u.m, baseline_ratio=8.0)


@pytest.mark.parametrize(
    "override",
    [9.0 * u.m, 31.0 * u.m, 0.0 * u.m, np.nan * u.m],
)
def test_reject_out_of_range_nulling_baselines(bounded_observatory, override):
    """Reject unphysical or unavailable nulling baselines at call time."""
    with pytest.raises(ValueError):
        bounded_observatory.get_collector_positions(nulling_baseline=override)


@pytest.mark.parametrize(
    "overrides",
    [
        {"nulling_baseline": "20"},
        {"nulling_baseline": 20 * u.s},
        {"baseline_ratio": -2.0},
        {"baseline_ratio": float("inf")},
        {"baseline_ratio": "6"},
    ],
)
def test_reject_invalid_per_observation_configuration(observatory, overrides):
    """Validate optional parameters on every geometry request."""
    with pytest.raises((ValueError, TypeError)):
        observatory.get_collector_positions(**overrides)


@pytest.mark.parametrize(
    "settings",
    [
        {"nulling_baseline_range": (30.0, 10.0)},
        {"nulling_baseline_range": (0.0, 20.0)},
        {"nulling_baseline_range": (10.0, float("inf"))},
        {"nulling_baseline_range": (10.0,)},
        {"nulling_baseline_range": (10.0 * u.s, 20.0 * u.m)},
        {"nulling_baseline_range": (20.0, 30.0)},
        {"imaging_baseline_range": (100.0, 200.0)},
    ],
)
def test_reject_invalid_baseline_ranges(settings):
    """Reject malformed ranges or defaults outside configured bounds."""
    with pytest.raises((ValueError, TypeError)):
        LIFEObservatory(**settings)


def test_baseline_override_changes_optical_response(bounded_observatory):
    """Changing the requested geometry changes off-axis transmission."""
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
    assert left_small.shape == left_large.shape == (1, 1, 1)
    assert not np.allclose(left_small.value, left_large.value, atol=1e-8)


# ================================================================
# Dynamic baseline configuration and operating-range tests
# ================================================================


def test_default_baseline_operating_range(observatory):
    """Verify the default 10–100 m nulling-baseline operating range."""
    obs = observatory

    assert obs.nulling_baseline == 14.5 * u.m
    assert obs.baseline_ratio == pytest.approx(6.0)
    assert obs.imaging_baseline == 87.0 * u.m
    assert obs.nulling_baseline_range == (10.0 * u.m, 100.0 * u.m)
    assert obs.imaging_baseline_range is None
    assert obs._outspec["nulling_baseline_range"] == [10.0, 100.0]

    # Both endpoints must be permitted.
    for baseline in (10.0, 100.0):
        positions = obs.get_collector_positions(nulling_baseline=baseline * u.m)
        assert positions.shape == (1, 4, 2)
        np.testing.assert_allclose(
            abs(positions[0, 0, 0].to_value(u.m)),
            baseline / 2.0,
        )


def test_per_observation_baseline_override(observatory):
    """Change the array configuration without changing stored defaults."""
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

    assert positions.shape == (2, 4, 2)
    np.testing.assert_allclose(positions[0].to_value(u.m), expected)
    np.testing.assert_allclose(
        positions[1].to_value(u.m),
        np.column_stack((-expected[:, 1], expected[:, 0])),
        atol=1e-12,
    )

    # The stored reference configuration must remain unchanged.
    assert observatory.nulling_baseline == 14.5 * u.m
    assert observatory.imaging_baseline == 87.0 * u.m
    assert observatory.baseline_ratio == 6.0


def test_explicitly_disable_default_baseline_range(observatory):
    """Setting None must disable the default nulling-baseline limit."""
    obs = LIFEObservatory(
        nulling_baseline_range=None,
        cachedir=observatory.cachedir,
        forceStaticEphem=True,
        verbose=False,
    )

    assert obs.nulling_baseline_range is None
    assert obs._outspec["nulling_baseline_range"] is None

    positions = obs.get_collector_positions(nulling_baseline=120.0 * u.m)
    assert positions.shape == (1, 4, 2)
    np.testing.assert_allclose(abs(positions[0, 0, 0].to_value(u.m)), 60.0)


def test_custom_baseline_ranges_and_endpoints(bounded_observatory):
    """Accept both inclusive boundaries of configured baseline intervals."""
    obs = bounded_observatory

    assert obs.nulling_baseline_range == (10.0 * u.m, 30.0 * u.m)
    assert obs.imaging_baseline_range == (60.0 * u.m, 180.0 * u.m)
    assert obs._outspec["nulling_baseline_range"] == [10.0, 30.0]
    assert obs._outspec["imaging_baseline_range"] == [60.0, 180.0]

    for baseline in (10.0, 30.0):
        positions = obs.get_collector_positions(nulling_baseline=baseline * u.m)

        assert positions.shape == (1, 4, 2)
        np.testing.assert_allclose(
            abs(positions[0, 0, 0].to_value(u.m)),
            baseline / 2.0,
        )
        np.testing.assert_allclose(
            abs(positions[0, 0, 1].to_value(u.m)),
            baseline * 6.0 / 2.0,
        )


def test_dynamic_ratio_respects_imaging_limits(bounded_observatory):
    """Check the imaging baseline after applying the requested ratio."""
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
    with pytest.raises(ValueError, match="imaging baseline"):
        obs.get_collector_positions(
            nulling_baseline=25.0 * u.m,
            baseline_ratio=8.0,
        )


@pytest.mark.parametrize(
    "baseline",
    [
        9.0 * u.m,
        31.0 * u.m,
        0.0 * u.m,
        np.nan * u.m,
    ],
)
def test_reject_nulling_baselines_outside_custom_range(bounded_observatory, baseline):
    """Reject baselines outside the limits or physically invalid values."""
    with pytest.raises(ValueError):
        bounded_observatory.get_collector_positions(nulling_baseline=baseline)


@pytest.mark.parametrize(
    "overrides",
    [
        {"nulling_baseline": "20"},
        {"nulling_baseline": 20.0 * u.s},
        {"baseline_ratio": -2.0},
        {"baseline_ratio": float("inf")},
        {"baseline_ratio": "6"},
    ],
)
def test_reject_invalid_dynamic_baseline_inputs(observatory, overrides):
    """Reject unsupported types, units, and ratios during observations."""
    with pytest.raises((ValueError, TypeError)):
        observatory.get_collector_positions(**overrides)


@pytest.mark.parametrize(
    "settings",
    [
        {"nulling_baseline_range": (30.0, 10.0)},
        {"nulling_baseline_range": (0.0, 20.0)},
        {"nulling_baseline_range": (10.0, float("inf"))},
        {"nulling_baseline_range": (10.0,)},
        {"nulling_baseline_range": (10.0 * u.s, 20.0 * u.m)},
        {"nulling_baseline_range": (20.0, 30.0)},
        {"imaging_baseline_range": (100.0, 200.0)},
    ],
)
def test_reject_invalid_baseline_range_configuration(settings):
    """Reject malformed ranges and defaults outside permitted limits."""
    with pytest.raises((ValueError, TypeError)):
        LIFEObservatory(**settings)


def test_baseline_override_changes_optical_transmission(
    bounded_observatory,
):
    """Check that different collector separations alter transmission."""
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

    assert left_small.shape == (1, 1, 1)
    assert left_large.shape == (1, 1, 1)

    assert not np.allclose(
        left_small.value,
        left_large.value,
        atol=1e-8,
    )
