"""Tests for the catalog-backed LIFE PlanetPopulation interface."""

import tempfile
import unittest
from pathlib import Path

import astropy.units as u
import numpy as np

from EXOSIMS.PlanetPopulation.LIFEPlanetPopulation import LIFEPlanetPopulation
from EXOSIMS.Prototypes.PlanetPopulation import PlanetPopulation


class TestLIFEPlanetPopulation(unittest.TestCase):
    """Verify catalog statistics without replacing P-Pop's sampling."""

    def setUp(self):
        """Create a temporary P-Pop catalog and EXOSIMS configuration."""
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / "ppop.txt"
        self.header = (
            "Nuniverse Nstar name ap ep ip Omegap omegap "
            "AgeomVIS Rp Mp"
        ).split()
        self.rows = [
            (0, 0, "STAR_A", 1.0, 0.0, 0.0, 0.0, 0.0, 0.2, 1.0, 1.0),
            (0, 0, "STAR_A", 2.0, 0.1, 0.2, 0.4, 0.6, 0.4, 2.0, 3.0),
            (2, 1, "STAR_B", 3.0, 0.2, 1.0, 1.5, 2.0, 0.3, 3.0, 5.0),
        ]
        self.write_catalog()
        self.specs = {
            "ppop_catalog": self.path,
            "ppop_total_universes": 3,
            "ppop_nstars": 2,
            "modules": {"PlanetPhysicalModel": "PlanetPhysicalModel"},
            "verbose": False,
        }

    def write_catalog(self):
        """Write a two-header P-Pop catalog containing the current test rows."""
        with self.path.open("w", encoding="utf-8") as stream:
            stream.write("P-Pop old header\n")
            stream.write("\t".join(self.header) + "\n")
            for row in self.rows:
                stream.write("\t".join(map(str, row)) + "\n")

    def test_exosims_module_contract(self):
        """Verify the adapter follows the EXOSIMS population interface."""
        pop = LIFEPlanetPopulation(**self.specs)
        self.assertIsInstance(pop, PlanetPopulation)
        self.assertEqual(pop._modtype, "PlanetPopulation")
        self.assertIsNotNone(pop.PlanetPhysicalModel)
        self.assertFalse(pop.scaleOrbits)
        self.assertFalse(pop.constrainOrbits)

    def test_empirical_eta_counts_empty_universes(self):
        """Count all generated universes when calculating empirical occurrence."""
        pop = LIFEPlanetPopulation(**self.specs)
        self.assertEqual(pop.ppop_planet_count, 3)
        self.assertEqual(pop.ppop_nonempty_universes, 2)
        self.assertEqual(pop.ppop_total_universes, 3)
        self.assertEqual(pop.ppop_nstars, 2)
        self.assertAlmostEqual(pop.eta, 3 / (3 * 2))
        self.assertAlmostEqual(pop.ppop_empirical_eta, pop.eta)

    def test_empirical_ranges_and_units(self):
        """Check observed catalog parameter bounds and Astropy units."""
        pop = LIFEPlanetPopulation(**self.specs)
        np.testing.assert_allclose(pop.arange.to_value(u.AU), [1, 3])
        np.testing.assert_allclose(pop.Rprange.to_value(u.earthRad), [1, 3])
        np.testing.assert_allclose(pop.Mprange.to_value(u.earthMass), [1, 5])
        np.testing.assert_allclose(pop.erange, [0, 0.2])
        np.testing.assert_allclose(pop.prange, [0.2, 0.4])
        self.assertEqual(pop.Irange.unit, u.deg)

    def test_reproducibility_metadata(self):
        """Verify catalog provenance and empirical-range metadata."""
        pop = LIFEPlanetPopulation(**self.specs)
        self.assertEqual(len(pop.ppop_sha256), 64)
        self.assertEqual(pop._outspec["ppop_sha256"], pop.ppop_sha256)
        self.assertTrue(pop._outspec["ppop_ranges_are_empirical"])
        self.assertEqual(pop._outspec["ppop_catalog"], str(self.path.resolve()))

    def test_no_fallback_random_generator(self):
        """Ensure generation methods cannot silently draw new planets."""
        pop = LIFEPlanetPopulation(**self.specs)
        for call in (
            lambda: pop.gen_plan_params(1),
            lambda: pop.gen_mass(1),
            lambda: pop.gen_angles(1),
        ):
            with self.subTest(call=call):
                with self.assertRaises(NotImplementedError):
                    call()

    def test_no_fake_analytic_density(self):
        """Ensure unsupported density evaluations explicitly fail."""
        pop = LIFEPlanetPopulation(**self.specs)
        for call in (
            lambda: pop.dist_sma(np.array([1.0])),
            lambda: pop.dist_eccen(np.array([0.1])),
            lambda: pop.dist_eccen_from_sma(np.array([0.1]), np.array([1.0])),
            lambda: pop.dist_albedo(np.array([0.3])),
            lambda: pop.dist_radius(np.array([1.0])),
            lambda: pop.dist_mass(np.array([1.0])),
        ):
            with self.subTest(call=call):
                with self.assertRaises(NotImplementedError):
                    call()

    def test_missing_metadata_rejected(self):
        """Require catalog path, universe count, and generating star count."""
        for key in ("ppop_catalog", "ppop_total_universes", "ppop_nstars"):
            specs = self.specs.copy()
            del specs[key]
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    LIFEPlanetPopulation(**specs)

    def test_sampling_override_rejected(self):
        """Reject configuration that would change the catalog population."""
        with self.assertRaisesRegex(ValueError, "sampling overrides"):
            LIFEPlanetPopulation(**self.specs, eta=0.05)
        with self.assertRaisesRegex(ValueError, "sampling overrides"):
            LIFEPlanetPopulation(**self.specs, scaleOrbits=True)

    def test_invalid_star_index_rejected(self):
        """Reject host indices beyond the configured generating-star count."""
        self.rows[2] = (2, 2, "STAR_B", 3, 0.2, 1, 1.5, 2, 0.3, 3, 5)
        self.write_catalog()
        with self.assertRaisesRegex(ValueError, "star index"):
            LIFEPlanetPopulation(**self.specs)

    def test_invalid_universe_index_rejected(self):
        """Reject universe indices beyond the configured universe count."""
        self.rows[2] = (3, 1, "STAR_B", 3, 0.2, 1, 1.5, 2, 0.3, 3, 5)
        self.write_catalog()
        with self.assertRaisesRegex(ValueError, "universe ID"):
            LIFEPlanetPopulation(**self.specs)

    def test_inconsistent_star_identifier_rejected(self):
        """Reject inconsistent host names for one original P-Pop star index."""
        self.rows[1] = (0, 0, "OTHER", 2, 0.1, 0.2, 0.4, 0.6, 0.4, 2, 3)
        self.write_catalog()
        with self.assertRaisesRegex(ValueError, "Nstar-to-name"):
            LIFEPlanetPopulation(**self.specs)

    def test_zero_planet_catalog_rejected(self):
        """Reject an empty catalog because empirical ranges are undefined."""
        self.rows = []
        self.write_catalog()
        with self.assertRaisesRegex(ValueError, "no planets"):
            LIFEPlanetPopulation(**self.specs)


if __name__ == "__main__":
    unittest.main()
