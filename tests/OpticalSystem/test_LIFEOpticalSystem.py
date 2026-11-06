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