"""Tests for the LIFE nulling interferometer optical system."""

import json

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
