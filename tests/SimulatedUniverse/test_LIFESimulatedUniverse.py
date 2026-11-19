"""Tests for original P-Pop universe import."""

import tempfile
import unittest
from pathlib import Path

import astropy.units as u
import numpy as np

from EXOSIMS.Prototypes.SimulatedUniverse import SimulatedUniverse
from EXOSIMS.SimulatedUniverse.LIFESimulatedUniverse import (
    LIFESimulatedUniverse,
)


class TestLIFESimulatedUniverse(unittest.TestCase):
    """Verify preservation of P-Pop planetary-system properties."""

    def setUp(self):
        """Construct a minimal two-header P-Pop catalog."""
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "population.txt"

        self.columns = (
            "Nuniverse Nstar name Rp Mp Porb ap ep ip Omegap "
            "omegap thetap rp AngSep Abond AgeomVIS AgeomMIR Tp Ds z"
        ).split()

        self.planets = [
            dict(
                Nuniverse=0,
                Nstar=5,
                name="HIP12345",
                Rp=1,
                Mp=1,
                Porb=365,
                ap=1,
                ep=0,
                ip=0,
                Omegap=0,
                omegap=0,
                thetap=0,
                rp=1,
                AngSep=0.1,
                Abond=0.3,
                AgeomVIS=0.2,
                AgeomMIR=0.05,
                Tp=288,
                Ds=10,
                z=3,
            ),
            dict(
                Nuniverse=1,
                Nstar=5,
                name="HIP12345",
                Rp=2,
                Mp=5,
                Porb=365,
                ap=1,
                ep=0,
                ip=0,
                Omegap=np.pi / 2,
                omegap=0,
                thetap=0,
                rp=1,
                AngSep=0.1,
                Abond=0.4,
                AgeomVIS=0.3,
                AgeomMIR=0.05,
                Tp=250,
                Ds=10,
                z=3,
            ),
        ]
        self.write_catalog(self.planets)

    def write_catalog(self, rows):
        """Write P-Pop's original two-line header structure."""
        with self.path.open("w", encoding="utf-8") as stream:
            stream.write("Original P-Pop header\n")
            stream.write("\t".join(self.columns) + "\n")
            for row in rows:
                stream.write(
                    "\t".join(str(row[column]) for column in self.columns) + "\n"
                )

    def test_inheritance(self):
        """Use the EXOSIMS SimulatedUniverse module interface."""
        self.assertTrue(issubclass(LIFESimulatedUniverse, SimulatedUniverse))
        self.assertEqual(LIFESimulatedUniverse._modtype, "SimulatedUniverse")

    def test_preserves_planet_properties(self):
        """Read original radius, mass, orbit, albedo and temperature."""
        data = LIFESimulatedUniverse.load_ppop_universe(self.path, 0)

        self.assertEqual(data["universe_id"], 0)
        self.assertEqual(data["Nstar"][0], 5)
        self.assertEqual(data["name"][0], "HIP12345")
        self.assertAlmostEqual(data["Rp"][0].to_value(u.earthRad), 1)
        self.assertAlmostEqual(data["Mp"][0].to_value(u.earthMass), 1)
        self.assertAlmostEqual(data["Tp"][0].to_value(u.K), 288)
        self.assertAlmostEqual(data["Abond"][0], 0.3)
        self.assertAlmostEqual(data["nEZ"][0], 3)

    def test_projected_position(self):
        """Reconstruct the orbital sky-plane coordinates."""
        first = LIFESimulatedUniverse.load_ppop_universe(self.path, 0)
        second = LIFESimulatedUniverse.load_ppop_universe(self.path, 1)

        np.testing.assert_allclose(
            first["planet_sky_positions"].to_value(u.arcsec),
            [[0.1, 0.0]],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            second["planet_sky_positions"].to_value(u.arcsec),
            [[0.0, 0.1]],
            atol=1e-6,
        )

    def test_separate_universes(self):
        """Do not mix planets belonging to different universes."""
        first = LIFESimulatedUniverse.load_ppop_universe(self.path, 0)
        second = LIFESimulatedUniverse.load_ppop_universe(self.path, 1)

        self.assertEqual(len(first["Rp"]), 1)
        self.assertEqual(len(second["Rp"]), 1)
        self.assertNotEqual(first["Tp"][0], second["Tp"][0])

    def test_explicit_empty_universe(self):
        """Preserve universes containing zero planets."""
        data = LIFESimulatedUniverse.load_ppop_universe(self.path, 2, total_universes=3)
        self.assertEqual(len(data["Rp"]), 0)
        self.assertEqual(data["planet_sky_positions"].shape, (0, 2))

        with self.assertRaises(ValueError):
            LIFESimulatedUniverse.load_ppop_universe(self.path, 2)

    def test_missing_required_column(self):
        """Reject catalogs without essential orbital information."""
        self.columns.remove("Omegap")
        self.write_catalog(self.planets)

        with self.assertRaisesRegex(ValueError, "Missing P-Pop columns"):
            LIFESimulatedUniverse.load_ppop_universe(self.path, 0)

    def test_invalid_orbit(self):
        """Reject nonphysical P-Pop orbital parameters."""
        invalid = [dict(self.planets[0], ep=1.2)]
        self.write_catalog(invalid)

        with self.assertRaisesRegex(ValueError, "Nonphysical"):
            LIFESimulatedUniverse.load_ppop_universe(self.path, 0)

    def test_prototype_random_generation_blocked(self):
        """Prevent accidental replacement by random EXOSIMS planets."""
        universe = object.__new__(LIFESimulatedUniverse)

        with self.assertRaises(NotImplementedError):
            universe.gen_physical_properties()


if __name__ == "__main__":
    unittest.main()
