"""Tests for the LIFE nulling interferometer optical system."""

import json
import numpy as np

import astropy.units as u
import pytest

from EXOSIMS.OpticalSystem.LIFEOpticalSystem import LIFEOpticalSystem
from tests.TestSupport.Info import resource_path


def test_mid_ir_observing_mode():
    """Verify that LIFE supports a 15 um observing mode."""
    path = resource_path("test-scripts/template_minimal.json")

    with open(path, encoding="utf-8") as file:
        specs = json.load(file)

    specs["starlightSuppressionSystems"][0]["lam"] = 15000

    optical_system = LIFEOpticalSystem(**specs)

    assert optical_system.observingModes[0]["lam"].to_value(u.um) == pytest.approx(15.0)


@pytest.mark.parametrize(
    ("configured_ratio", "expected_ratio"),
    [
        (None, 6.0),
        (8.0, 8.0),
    ],
)
def test_baseline_ratio_configuration(configured_ratio, expected_ratio):
    """Verify the default ratio and configuration overrides."""
    path = resource_path("test-scripts/template_minimal.json")

    with open(path, encoding="utf-8") as file:
        specs = json.load(file)

    if configured_ratio is not None:
        specs["baseline_ratio"] = configured_ratio

    optical_system = LIFEOpticalSystem(**specs)

    assert optical_system.baseline_ratio == expected_ratio
    assert optical_system._outspec["baseline_ratio"] == expected_ratio


@pytest.mark.parametrize(
    "invalid_ratio",
    [0.0, -1.0, float("nan"), float("inf")],
)
def test_invalid_baseline_ratio(invalid_ratio):
    """Reject nonpositive and nonfinite baseline ratios."""
    with pytest.raises(ValueError, match="baseline_ratio"):
        LIFEOpticalSystem(baseline_ratio=invalid_ratio)

def test_get_collector_positions_reference():
    """Verify the LIFE reference X-array geometry."""
    life = LIFEOpticalSystem.__new__(LIFEOpticalSystem)
    life.baseline_ratio = 6.0

    positions = life.get_collector_positions(14.5 * u.m)

    expected = np.array(
        [
            [-7.25, -43.5],
            [-7.25, 43.5],
            [7.25, -43.5],
            [7.25, 43.5],
        ]
    ) * u.m

    np.testing.assert_allclose(
        positions.to_value(u.m),
        expected.to_value(u.m),
    )

def test_get_baselines_rotation():
    """Verify pairwise baselines and 90-degree array rotation."""
    life = LIFEOpticalSystem.__new__(LIFEOpticalSystem)
    life.baseline_ratio = 6.0

    baselines = life.get_baselines(
        14.5 * u.m,
        np.array([0, 90]) * u.deg,
    )

    assert baselines.shape == (2, 4, 4, 2)

    np.testing.assert_allclose(
        baselines[0, 0, 2].to_value(u.m),
        [-14.5, 0.0],
        atol=1e-12,
    )

    np.testing.assert_allclose(
        baselines[1, 0, 2].to_value(u.m),
        [0.0, -14.5],
        atol=1e-12,
    )

    np.testing.assert_allclose(
        baselines[0].to_value(u.m),
        -np.swapaxes(baselines[0].to_value(u.m), 0, 1),
        atol=1e-12,
    )

def test_get_spectral_bins_reference():
    """Verify LIFE spectral bins against the LIFEsimFT convention."""
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

    # The final bin covers the upper wavelength limit.
    assert lower_edges[-1] < 18.5 * u.um
    assert upper_edges[-1] >= 18.5 * u.um

    # Check the geometric spacing.
    ratios = lower_edges[1:] / lower_edges[:-1]

    np.testing.assert_allclose(
        ratios.to_value(u.dimensionless_unscaled),
        1.0 + 1.0 / 33.3,
        rtol=1e-12,
    )

    assert np.all(widths > 0 * u.m)