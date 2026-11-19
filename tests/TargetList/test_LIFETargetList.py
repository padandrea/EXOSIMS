"""Focused unittest checks for LIFE stellar properties and blackbody fluxes.

Run from the root of EXOSIMS:
    python -m unittest discover -s tests/TargetList -p 'test_LIFETargetList.py' -v

Optional cached-LTC4 physical-catalog integration check:
    LIFE_LTC4_INTEGRATION=1 python -m unittest discover -s tests/TargetList -p 'test_LIFETargetList.py' -v
"""

import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace

import astropy.units as u
import numpy as np
from astropy.constants import c, h
from astropy.modeling.physical_models import BlackBody

from EXOSIMS.TargetList.LIFETargetList import LIFETargetList


class TestLIFETargetList(unittest.TestCase):
    """Unit tests for LIFE stellar fluxes and physical catalog properties."""

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

    def test_catalog_attributes_do_not_require_optical_photometry(self):
        """Keep V-band fields optional and require physical LIFE inputs."""
        tl = self.make_list()
        tl.StarCatalog = SimpleNamespace(
            catalog_atts=[
                "Name",
                "L",
                "coords",
                "dist",
                "Binary_Cut",
                "Teff",
                "radius",
                "mass",
                "diameter",
                "Vmag",
                "BV",
                "MV",
                "BC",
            ]
        )
        tl.set_catalog_attributes()
        for name in ("Teff", "radius", "mass", "diameter"):
            self.assertIn(name, tl.required_catalog_atts)
        for name in ("Vmag", "BV", "MV", "BC"):
            self.assertNotIn(name, tl.required_catalog_atts)
            self.assertIn(name, tl.catalog_atts)

    def test_missing_physical_catalog_attribute_raises(self):
        """Reject a catalog missing essential LIFE physical properties."""
        tl = self.make_list()
        tl.StarCatalog = SimpleNamespace(
            catalog_atts=[
                "Name",
                "L",
                "coords",
                "dist",
                "Binary_Cut",
                "Teff",
                "radius",
                "mass",
            ]
        )
        with self.assertRaisesRegex(ValueError, "diameter"):
            tl.set_catalog_attributes()

    def test_photometry_is_not_synthesized(self):
        """Preserve missing optical magnitudes and source spectral labels."""
        tl = self.make_list()
        tl.fillPhotometry = False
        tl.fillMissingBandMags = False
        tl.Vmag = np.full(tl.nStars, np.nan)
        tl.BV = np.full(tl.nStars, np.nan)
        tl.Spec = np.array(["G2V", "M11", ""])
        tl.fillPhotometryVals()
        self.assertTrue(np.all(np.isnan(tl.Vmag)))
        self.assertTrue(np.all(np.isnan(tl.BV)))
        np.testing.assert_array_equal(tl.Spec, ["G2V", "M11", ""])

        tl.fillPhotometry = True
        with self.assertRaisesRegex(ValueError, "fillPhotometry=False"):
            tl.fillPhotometryVals()

    def test_catalog_stellar_mass_without_optical_magnitude(self):
        """Use supplied LTC4 masses without stochastic V-band inference."""
        tl = self.make_list()
        tl.mass = np.array([1.0, 0.3, 1.4]) * u.Msun
        tl.catalog_atts = ["mass"]
        tl.stellar_mass()
        np.testing.assert_allclose(tl.MsEst.to_value(u.Msun), [1.0, 0.3, 1.4])
        np.testing.assert_allclose(tl.MsTrue.to_value(u.Msun), [1.0, 0.3, 1.4])
        self.assertIn("MsEst", tl.catalog_atts)
        self.assertIn("MsTrue", tl.catalog_atts)

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
        """Verify stellar photon flux scales with radius squared."""
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
    def test_ltc4_physical_catalog_filter(self):
        """Verify physical filtering does not require reconstructed V-band data."""

        class StopAfterPhysicalFilter(Exception):
            """Stop construction once the physical-property filter completes."""

        class CheckpointLIFETargetList(LIFETargetList):
            """Inspect TargetList immediately after the physical-quantity filter."""

            def nan_filter(self):
                """Record expected physically valid rows before filtering."""
                before = self.nStars
                valid = (
                    (self.Name != "")
                    & np.isfinite(self.L)
                    & np.isfinite(self.dist.to_value(u.pc))
                    & np.isfinite(self.Teff.to_value(u.K))
                    & np.isfinite(self.radius.to_value(u.Rsun))
                    & np.isfinite(self.mass.to_value(u.Msun))
                    & np.isfinite(self.diameter.to_value(u.mas))
                )
                expected_names = self.Name[valid].copy()
                super().nan_filter()
                raise StopAfterPhysicalFilter(
                    before,
                    expected_names,
                    self.Name.copy(),
                    np.isnan(self.Vmag).all(),
                    np.isnan(self.BV).all(),
                )

        root = Path(__file__).resolve().parents[2]
        template = (
            root / "tests" / "TestSupport" / "test-scripts" / "template_minimal.json"
        )
        specs = json.loads(template.read_text(encoding="utf-8"))
        specs["modules"]["StarCatalog"] = "LIFEStarCatalog"
        specs["modules"]["TargetList"] = "LIFETargetList"
        specs.update(
            fillPhotometry=False,
            fillMissingBandMags=False,
            filterBinaries=False,
            verbose=False,
            optional_filters={
                "binary_filter": {"enabled": False},
                "outside_IWA_filter": {"enabled": False},
                "completeness_filter": {"enabled": False},
            },
        )
        with self.assertRaises(StopAfterPhysicalFilter) as captured:
            CheckpointLIFETargetList(**specs)
        before, expected_names, actual_names, missing_v, missing_bv = (
            captured.exception.args
        )
        self.assertEqual(before, 3932)
        np.testing.assert_array_equal(actual_names, expected_names)
        self.assertTrue(missing_v)
        self.assertTrue(missing_bv)


if __name__ == "__main__":
    unittest.main(verbosity=2)
