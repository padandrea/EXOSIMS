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
from types import SimpleNamespace


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
        """Require P-Pop inputs instead of generating random EXOSIMS planets."""
        universe = object.__new__(LIFESimulatedUniverse)

        with self.assertRaisesRegex(
            ValueError, "ppop_catalog and ppop_universe_id are required"
        ):
            universe.gen_physical_properties()

    def make_initialized_universe(
        self,
        universe_id=0,
        names=("HIP12345",),
        distances=(10.0,),
        masses=(1.0,),
    ):
        """Build a LIFE orbital state with a lightweight physical TargetList."""
        universe = object.__new__(LIFESimulatedUniverse)

        universe.TargetList = SimpleNamespace(
            Name=np.asarray(names),
            dist=np.asarray(distances) * u.pc,
            MsTrue=np.asarray(masses) * u.Msun,
        )
        universe.Min = None
        universe.commonSystemPlane = False
        universe.fixedPlanPerStar = None
        universe.fixed_nEZ_val = None
        universe._outspec = {}
        universe.planet_atts = []

        universe.gen_physical_properties(
            ppop_catalog=self.path,
            ppop_universe_id=universe_id,
            ppop_total_universes=3,
        )
        universe.init_systems()

        return universe

    def test_exosims_planet_attributes(self):
        """Populate EXOSIMS properties using original P-Pop values."""
        su = self.make_initialized_universe()

        self.assertEqual(su.nPlans, 1)
        np.testing.assert_array_equal(su.plan2star, [0])
        np.testing.assert_array_equal(su.sInds, [0])

        self.assertAlmostEqual(su.Rp[0].to_value(u.earthRad), 1.0)
        self.assertAlmostEqual(su.Mp[0].to_value(u.earthMass), 1.0)
        self.assertAlmostEqual(su.a[0].to_value(u.AU), 1.0)
        self.assertAlmostEqual(su.Tp[0].to_value(u.K), 288.0)
        self.assertAlmostEqual(su.Abond[0], 0.3)
        self.assertAlmostEqual(su.nEZ[0], 3.0)
        self.assertAlmostEqual(su.M0[0].to_value(u.rad), 0.0)

        self.assertEqual(su.r.shape, (1, 3))
        self.assertEqual(su.v.shape, (1, 3))
        self.assertTrue(np.isnan(su.dMag).all())

        np.testing.assert_allclose(
            su.r[0].to_value(u.AU),
            [1.0, 0.0, 0.0],
            atol=1e-10,
        )

    def test_exosims_projected_position(self):
        """Preserve P-Pop orbital orientation and projected position."""
        su = self.make_initialized_universe(universe_id=1)

        np.testing.assert_allclose(
            su.planet_sky_positions.to_value(u.arcsec),
            [[0.0, 0.1]],
            atol=1e-6,
        )
        self.assertAlmostEqual(su.WA[0].to_value(u.arcsec), 0.1, places=5)

    def test_targetlist_mapping_uses_names(self):
        """Map stars by identifiers rather than original P-Pop indices."""
        su = self.make_initialized_universe(
            names=("OTHER", "HIP 12345"),
            distances=(30.0, 10.0),
            masses=(0.9, 1.0),
        )

        np.testing.assert_array_equal(su.plan2star, [1])
        np.testing.assert_array_equal(su.ppop_nstar, [5])

    def test_empty_exosims_universe(self):
        """Do not invent planets when one P-Pop universe is empty."""
        su = self.make_initialized_universe(universe_id=2)

        self.assertEqual(su.nPlans, 0)
        self.assertEqual(su.plan2star.size, 0)
        self.assertEqual(su.r.shape, (0, 3))
        self.assertEqual(su.v.shape, (0, 3))
        self.assertEqual(su.planet_sky_positions.shape, (0, 2))

        su.propag_system(0, 10 * u.day)
        self.assertEqual(su.nPlans, 0)

    def test_unmatched_star_is_rejected(self):
        """Reject missing target stars rather than silently dropping planets."""
        with self.assertRaisesRegex(ValueError, "absent from TargetList"):
            self.make_initialized_universe(names=("OTHER",))

    def test_stellar_distance_mismatch(self):
        """Reject mismatched stellar catalogs."""
        with self.assertRaisesRegex(ValueError, "distances disagree"):
            self.make_initialized_universe(distances=(20.0,))

    def test_original_separation_validation(self):
        """Check independent P-Pop angular-separation information."""
        invalid = [dict(self.planets[0], AngSep=0.2)]
        self.write_catalog(invalid)

        with self.assertRaisesRegex(ValueError, "orbital geometry"):
            self.make_initialized_universe()

    def test_keplerian_propagation(self):
        """Move a circular orbit approximately one quarter revolution."""
        su = self.make_initialized_universe()

        mu = su.mu_AU3_div_day2[0] * u.AU**3 / u.day**2
        period = (2 * np.pi * np.sqrt((1 * u.AU) ** 3 / mu)).to(u.day)

        su.propag_system(0, period / 4)

        np.testing.assert_allclose(
            su.r[0].to_value(u.AU),
            [0.0, 1.0, 0.0],
            atol=2e-3,
        )
        self.assertAlmostEqual(su.d[0].to_value(u.AU), 1.0, places=3)
        self.assertTrue(np.isnan(su.dMag[0]))

    def test_invalid_propagation_time(self):
        """Reject negative and nonfinite propagation times."""
        su = self.make_initialized_universe()

        for dt in (-1 * u.day, np.nan * u.day):
            with self.subTest(dt=dt):
                with self.assertRaises(ValueError):
                    su.propag_system(0, dt)


if __name__ == "__main__":
    unittest.main()
