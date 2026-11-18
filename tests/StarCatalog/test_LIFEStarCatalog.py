"""Unit tests for the LIFE LTC4 StarCatalog adapter."""

import tempfile
import unittest
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd

from EXOSIMS.StarCatalog.LIFEStarCatalog import LIFEStarCatalog


class TestLIFEStarCatalog(unittest.TestCase):
    """Test LTC4 catalog loading and validation."""

    def setUp(self):
        """Create temporary LTC4 test data."""
        self.temp_dir = tempfile.TemporaryDirectory(prefix="life_catalog_")
        self.addCleanup(self.temp_dir.cleanup)
        self.catalogpath = Path(self.temp_dir.name) / "LTC4.csv"
        self.rows = pd.DataFrame(
            {
                "main_id": ["Sun test", "M dwarf test"],
                "coo_ra": [10.0, 75.0],
                "coo_dec": [-20.0, 30.0],
                "plx_value": [100.0, 50.0],
                "dist_st_value": [10.0, 20.0],
                "sptype_string": ["G2V", "M3V"],
                "teff_st_value": [5900.0, 3200.0],
                "teff": [5772.0, 3300.0],
                "mass_st_value": [1.0, 0.3],
                "radius_st_value": [1.0, 0.3],
                "binary_flag": [False, True],
                "mag_i_value": [5.5, np.nan],
                "mag_j_value": [5.0, 9.0],
            }
        )
        self.rows.to_csv(self.catalogpath, index=False)

    def make_catalog(self):
        """Initialize a catalog from the test data."""
        return LIFEStarCatalog(
            catalogpath=self.catalogpath,
            cachedir=self.temp_dir.name,
            verbose=False,
        )

    def test_loads_every_row_without_filters(self):
        """Verify that no stars are filtered during loading."""
        catalog = self.make_catalog()
        self.assertEqual(catalog.ntargs, 2)
        self.assertEqual(list(catalog.Name), ["Sun test", "M dwarf test"])
        self.assertEqual(list(catalog.Spec), ["G2V", "M3V"])
        np.testing.assert_allclose(catalog.dist.to_value(u.pc), [10.0, 20.0])
        np.testing.assert_allclose(catalog.coords.ra.to_value(u.deg), [10.0, 75.0])

    def test_uses_ppop_temperature_and_physical_units(self):
        """Validate temperatures and physical units."""
        catalog = self.make_catalog()
        np.testing.assert_allclose(catalog.Teff.to_value(u.K), [5772.0, 3300.0])
        np.testing.assert_allclose(catalog.radius.to_value(u.Rsun), [1.0, 0.3])
        np.testing.assert_allclose(catalog.mass.to_value(u.Msun), [1.0, 0.3])
        self.assertAlmostEqual(catalog.L[0], 1.0, delta=0.01)
        self.assertTrue(np.all(catalog.diameter.to_value(u.mas) > 0))

    def test_binary_metadata_and_missing_optical_photometry(self):
        """Check binary flags and missing photometry."""
        catalog = self.make_catalog()
        np.testing.assert_array_equal(catalog.binary_flag, [False, True])
        np.testing.assert_array_equal(catalog.Binary_Cut, [False, True])
        np.testing.assert_allclose(catalog.Jmag, [5.0, 9.0])
        self.assertTrue(np.isnan(catalog.Vmag).all())
        self.assertTrue(np.isnan(catalog.BV).all())

    def test_rejects_missing_columns(self):
        """Reject catalogs lacking required columns."""
        self.rows.drop(columns=["teff"]).to_csv(self.catalogpath, index=False)
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            self.make_catalog()

    def test_rejects_missing_catalog_path(self):
        """Reject an explicitly supplied path that does not exist."""
        missing_path = self.catalogpath.with_name("missing.csv")

        with self.assertRaises(FileNotFoundError):
            LIFEStarCatalog(catalogpath=missing_path)


if __name__ == "__main__":
    unittest.main()
