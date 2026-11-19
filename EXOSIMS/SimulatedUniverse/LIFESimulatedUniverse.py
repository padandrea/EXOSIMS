"""Import Kammerer-Quanz P-Pop planetary-system snapshots."""

import csv
from pathlib import Path

import astropy.units as u
import numpy as np

from EXOSIMS.Prototypes.SimulatedUniverse import SimulatedUniverse


class LIFESimulatedUniverse(SimulatedUniverse):
    """LIFE simulated-universe adapter for original P-Pop catalogs.

    Full EXOSIMS initialization is not yet implemented. The catalog
    reader can be used independently without sampling new planets.
    """

    _modtype = "SimulatedUniverse"

    @staticmethod
    def load_ppop_universe(catalog_path, universe_id, total_universes=None):
        """Read one P-Pop universe without changing its planet properties.

        P-Pop stores two header lines, followed by tab-separated
        planet records. Original Nstar indices are preserved and
        are not yet mapped to filtered EXOSIMS TargetList indices.

        Parameters
        ----------
        catalog_path : str or pathlib.Path
            Original P-Pop text catalog.
        universe_id : int
            Zero-based P-Pop universe identifier.
        total_universes : int, optional
            Explicit universe count, needed to distinguish an empty
            universe from an identifier absent from the catalog.

        Returns
        -------
        dict
            Original planet properties with Astropy units, plus
            reconstructed two-dimensional angular sky positions.
        """
        if isinstance(universe_id, (bool, np.bool_)) or not isinstance(
            universe_id, (int, np.integer)
        ):
            raise TypeError("universe_id must be an integer.")
        if universe_id < 0:
            raise ValueError("universe_id must be nonnegative.")

        if total_universes is not None:
            if (
                isinstance(total_universes, (bool, np.bool_))
                or not isinstance(total_universes, (int, np.integer))
                or total_universes <= universe_id
            ):
                raise ValueError("Invalid total_universes.")

        required = {
            "Nuniverse",
            "Nstar",
            "name",
            "Rp",
            "Mp",
            "Porb",
            "ap",
            "ep",
            "ip",
            "Omegap",
            "omegap",
            "thetap",
            "rp",
            "AngSep",
            "Abond",
            "AgeomVIS",
            "AgeomMIR",
            "Tp",
            "Ds",
            "z",
        }

        path = Path(catalog_path).expanduser()
        rows = []

        with path.open(encoding="utf-8") as stream:
            if not stream.readline():
                raise ValueError("P-Pop catalog has no first header.")

            header = stream.readline().rstrip("\r\n\t").split("\t")
            missing = required.difference(header)
            if missing:
                raise ValueError(f"Missing P-Pop columns: {sorted(missing)}")

            for row in csv.DictReader(stream, fieldnames=header, delimiter="\t"):
                if int(row["Nuniverse"]) == universe_id:
                    rows.append(row)

        if not rows and total_universes is None:
            raise ValueError(
                "No planet records for this universe. Provide "
                "total_universes to permit an empty universe."
            )

        def numeric(column):
            """Read finite numerical values from the selected rows."""
            data = np.array([float(row[column]) for row in rows])
            if not np.all(np.isfinite(data)):
                raise ValueError(f"Invalid P-Pop values in {column}.")
            return data

        nstar = np.array([int(row["Nstar"]) for row in rows], dtype=int)
        if np.any(nstar < 0):
            raise ValueError("Negative P-Pop star index.")

        radius = numeric("Rp")
        mass = numeric("Mp")
        period = numeric("Porb")
        a = numeric("ap")
        e = numeric("ep")
        i = numeric("ip")
        omega_node = numeric("Omegap")
        omega_peri = numeric("omegap")
        true_anomaly = numeric("thetap")
        orbital_radius = numeric("rp")
        distance_pc = numeric("Ds")
        temperature = numeric("Tp")
        zodi = numeric("z")
        bond_albedo = numeric("Abond")
        visible_albedo = numeric("AgeomVIS")
        mir_albedo = numeric("AgeomMIR")

        if (
            np.any(radius <= 0)
            or np.any(mass <= 0)
            or np.any(period <= 0)
            or np.any(a <= 0)
            or np.any((e < 0) | (e >= 1))
            or np.any(orbital_radius <= 0)
            or np.any(distance_pc <= 0)
            or np.any(temperature <= 0)
            or np.any(zodi < 0)
            or np.any((bond_albedo < 0) | (bond_albedo > 1))
            or np.any((visible_albedo < 0) | (visible_albedo > 1))
            or np.any((mir_albedo < 0) | (mir_albedo > 1))
        ):
            raise ValueError("Nonphysical P-Pop planetary properties.")

        # Standard Keplerian projection onto the sky plane.
        # Use the original P-Pop orbital radius and angles.
        argument = omega_peri + true_anomaly

        x_au = orbital_radius * (
            np.cos(omega_node) * np.cos(argument)
            - np.sin(omega_node) * np.sin(argument) * np.cos(i)
        )
        y_au = orbital_radius * (
            np.sin(omega_node) * np.cos(argument)
            + np.cos(omega_node) * np.sin(argument) * np.cos(i)
        )

        # One AU at one pc subtends approximately one arcsecond.
        sky_arcsec = np.column_stack((x_au / distance_pc, y_au / distance_pc))
        sky_positions = (sky_arcsec * u.arcsec).to(u.rad)

        return {
            "universe_id": int(universe_id),
            "Nstar": nstar,
            "name": np.array([row["name"] for row in rows], dtype=str),
            "Rp": radius * u.earthRad,
            "Mp": mass * u.earthMass,
            "Porb": period * u.day,
            "a": a * u.AU,
            "e": e,
            "I": i * u.rad,
            "O": omega_node * u.rad,
            "w": omega_peri * u.rad,
            "nu": true_anomaly * u.rad,
            "r_orbit": orbital_radius * u.AU,
            "AngSep": numeric("AngSep") * u.arcsec,
            "Abond": bond_albedo,
            "p": visible_albedo,
            "AgeomMIR": mir_albedo,
            "Tp": temperature * u.K,
            "nEZ": zodi,
            "star_distance": distance_pc * u.pc,
            "planet_sky_positions": sky_positions,
        }

    def gen_physical_properties(self, **specs):
        """Block random prototype planets until catalog loading is wired."""
        raise NotImplementedError(
            "LIFESimulatedUniverse must initialize planets from a P-Pop "
            "catalog. The full EXOSIMS initialization is not yet implemented."
        )
