"""Focused unittest checks for the LIFE TargetList blackbody implementation.

Run from the root of EXOSIMS:
    python -m unittest discover -s tests/TargetList -p 'test_LIFETargetList.py' -v

Optional cached-LTC4 photometry integration check:
    LIFE_LTC4_INTEGRATION=1 python -m unittest discover -s tests/TargetList -p 'test_LIFETargetList.py' -v
"""

import json
import os
import unittest
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.constants import c, h
from astropy.modeling.physical_models import BlackBody

from EXOSIMS.TargetList.LIFETargetList import LIFETargetList


class TestLIFETargetList(unittest.TestCase):
    """Unit tests for LIFE stellar blackbody flux calculations."""

    @staticmethod
    def make_list():
        """Build a minimal fixture without running the incomplete full constructor."""
        tl = object.__new__(LIFETargetList)
        tl.nStars = 3
        tl.Teff = np.array([5772.0, 5772.0, 4500.0]) * u.K
        tl.radius = np.array([1.0, 2.0, 0.7]) * u.Rsun
        tl.dist = np.array([10.0, 10.0, 15.0]) * u.pc
        tl.star_fluxes = {}
        return tl

    @staticmethod
    def flat_mode(lower_nm=4000.0, upper_nm=18500.0):
        """Create a flat-throughput LIFE observing mode."""
        return {
            "hex": "life-test-flat",
            "LIFE_lambda_min": lower_nm,
            "LIFE_lambda_max": upper_nm,
            "bandpass": lambda wavelength: np.ones(wavelength.size),
        }

    def test_monochromatic_flux_matches_astropy_blackbody(self):
        """Reference calculation via an independent Astropy BlackBody model."""
        wavelengths = np.array([4.0, 10.0, 18.5, 20.0]) * u.um
        actual = LIFETargetList.blackbody_photon_flux_density(
            wavelengths,
            np.array([5772.0]) * u.K,
            np.array([1.0]) * u.Rsun,
            np.array([10.0]) * u.pc,
        )[0]

        bb = BlackBody(temperature=5772.0 * u.K)
        # BlackBody returns energy per area, frequency, time and steradian.
        energy_at_earth = (
            bb(wavelengths) * np.pi * u.sr * (1 * u.Rsun / (10 * u.pc)) ** 2
        ).to(u.W / (u.m**2 * u.um), equivalencies=u.spectral_density(wavelengths))
        expected = (energy_at_earth * wavelengths / (h * c) * u.photon).to(
            u.photon / (u.s * u.m**2 * u.um)
        )

        np.testing.assert_allclose(
            actual.to_value(expected.unit), expected.value, rtol=1e-10
        )

    def test_flux_scales_with_radius_squared(self):
        """Verify flux scales with stellar radius squared."""
        tl = self.make_list()
        flux = tl.starFlux(np.array([0, 1, 2]), self.flat_mode())
        self.assertTrue(flux.unit.is_equivalent(u.photon / (u.s * u.m**2)))
        self.assertTrue(np.all(np.isfinite(flux.value)))
        self.assertTrue(np.all(flux.value > 0))
        self.assertAlmostEqual((flux[1] / flux[0]).value, 4.0, places=10)

    def test_20um_extension_is_finite(self):
        """Verify finite flux when extending the bandpass to 20 um."""
        tl = self.make_list()
        flux_185 = tl.starFlux([0], self.flat_mode())
        flux_200 = tl.starFlux(
            [0], {**self.flat_mode(4000, 20000), "hex": "life-test-flat-20"}
        )
        self.assertTrue(np.isfinite(flux_200[0].value))
        self.assertGreater(flux_200[0].value, flux_185[0].value)

    def test_cache_and_indexing(self):
        """Verify flux caching and arbitrary target indexing."""
        tl = self.make_list()
        mode = self.flat_mode()
        flux = tl.starFlux([2, 0, 2], mode)
        self.assertEqual(flux.shape, (3,))
        self.assertEqual(len(tl.star_fluxes), 1)
        self.assertAlmostEqual(flux[0].value, flux[2].value)
        np.testing.assert_array_equal(flux.value, tl.starFlux([2, 0, 2], mode).value)

    def test_invalid_parameters_raise(self):
        """Reject invalid stellar parameters and wavelength ranges."""
        with self.assertRaises(ValueError):
            LIFETargetList.blackbody_photon_flux_density(
                [10] * u.um, [0] * u.K, [1] * u.Rsun, [10] * u.pc
            )
        tl = self.make_list()
        with self.assertRaises(ValueError):
            tl.starFlux([0], self.flat_mode(20000, 18000))
        with self.assertRaises(IndexError):
            tl.starFlux([tl.nStars], self.flat_mode())

    def test_legacy_exozodi_is_explicitly_blocked(self):
        """Prevent use of the conventional exozodi calculation."""
        tl = self.make_list()
        with self.assertRaisesRegex(NotImplementedError, "LIFE-specific exozodi"):
            tl.calc_all_JEZ0()

    @unittest.skipUnless(
        os.environ.get("LIFE_LTC4_INTEGRATION") == "1",
        "Set LIFE_LTC4_INTEGRATION=1 for cached LTC4 integration check",
    )
    def test_ltc4_photometry_filter(self):
        """Confirm legacy photometry temporarily removes 14 known LTC4 stars."""
        expected = {
            "SIPS J1514-2254",
            "V* DY Psc",
            "SSSPM J2356-3426",
            "SIPS J2126-3143",
            "LP  944-20",
            "2MASSW J0149090+295613",
            "2MASS J11592585-3634495",
            "2MASSI J1743348+584411",
            "SIPS J1019-2707",
            "2MASS J22341394+2359559",
            "2MASS J21330794+2321590",
            "2MASS J15445518+3301447",
            "2MASS J08111040+1855280",
            "HD 222335",
        }

        class StopAfterPhotometry(Exception):
            pass

        class CheckpointLIFETargetList(LIFETargetList):
            def nan_filter(self):
                before = set(map(str, self.Name))
                super().nan_filter()
                after = set(map(str, self.Name))
                raise StopAfterPhotometry(before, after)

        root = Path(__file__).resolve().parents[2]
        template = (
            root / "tests" / "TestSupport" / "test-scripts" / "template_minimal.json"
        )
        specs = json.loads(template.read_text(encoding="utf-8"))
        specs["modules"]["StarCatalog"] = "LIFEStarCatalog"
        specs["modules"]["TargetList"] = "LIFETargetList"
        specs.update(
            fillPhotometry=True,
            fillMissingBandMags=False,
            filterBinaries=False,
            verbose=False,
            optional_filters={
                "binary_filter": {"enabled": False},
                "outside_IWA_filter": {"enabled": False},
                "completeness_filter": {"enabled": False},
            },
        )
        with self.assertRaises(StopAfterPhotometry) as captured:
            CheckpointLIFETargetList(**specs)
        before, after = captured.exception.args
        self.assertEqual(len(before), 3932)
        self.assertEqual(before - after, expected)
        self.assertEqual(len(after), 3918)


if __name__ == "__main__":
    unittest.main(verbosity=2)
