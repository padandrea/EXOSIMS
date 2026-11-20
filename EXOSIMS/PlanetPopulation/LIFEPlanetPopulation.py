"""EXOSIMS PlanetPopulation metadata adapter for fixed P-Pop catalogs.

This class does not implement the Kammerer-Quanz P-Pop generator or PDFs.
Planet realizations come exclusively from LIFESimulatedUniverse.
"""

import csv
import hashlib
from pathlib import Path

import numpy as np

from EXOSIMS.Prototypes.PlanetPopulation import PlanetPopulation


class LIFEPlanetPopulation(PlanetPopulation):
    """Expose fixed-catalog metadata to the EXOSIMS population interface.

    Parameters
    ----------
    ppop_catalog : str or Path
        Path to the *same* P-Pop .txt catalog used by LIFESimulatedUniverse.
    ppop_total_universes : int
        Number of universes generated, including empty ones.
    ppop_nstars : int
        Number of stars used by the original P-Pop generator per universe,
        before EXOSIMS target filtering. This is not necessarily the number
        of stars in the full LTC4 catalog.

    Notes
    -----
    The reported eta is the *empirical* mean planet count per generating
    star, not an estimate of the underlying P-Pop occurrence-rate PDF.
    Reported parameter ranges are observed minima/maxima, not sampling priors.
    Generation and distribution methods are explicitly unsupported.
    """

    _modtype = "PlanetPopulation"

    def __init__(
        self,
        ppop_catalog=None,
        ppop_total_universes=None,
        ppop_nstars=None,
        **specs,
    ):
        if ppop_catalog is None:
            raise ValueError("ppop_catalog is required.")

        for name, number in (
            ("ppop_total_universes", ppop_total_universes),
            ("ppop_nstars", ppop_nstars),
        ):
            if (
                isinstance(number, (bool, np.bool_))
                or not isinstance(number, (int, np.integer))
                or number < 1
            ):
                raise ValueError(f"{name} must be a positive integer.")

        # Avoid silently replacing catalog statistics with a different
        # model's occurrence rate, sampling ranges or orbit scaling.
        forbidden = {
            "eta", "arange", "erange", "Irange", "Orange", "wrange",
            "prange", "Rprange", "Mprange", "scaleOrbits",
            "constrainOrbits",
        }
        collision = forbidden.intersection(specs)
        if collision:
            raise ValueError(
                "Catalog-driven LIFEPlanetPopulation does not accept "
                f"sampling overrides: {sorted(collision)}"
            )

        path = Path(ppop_catalog).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"P-Pop catalog does not exist: {path}")
        if path.suffix.lower() != ".txt":
            raise ValueError("Use an uncompressed P-Pop .txt catalog.")

        self.ppop_catalog = str(path)
        self.ppop_total_universes = int(ppop_total_universes)
        self.ppop_nstars = int(ppop_nstars)

        columns = {
            "Nuniverse", "Nstar", "name", "ap", "ep", "ip",
            "Omegap", "omegap", "AgeomVIS", "Rp", "Mp",
        }
        observed = {
            column: [np.inf, -np.inf]
            for column in ("ap", "ep", "ip", "Omegap", "omegap", "AgeomVIS", "Rp", "Mp")
        }
        source_names = {}
        count = 0
        seen_universes = set()

        with path.open(encoding="utf-8") as stream:
            if not stream.readline():
                raise ValueError("Missing first P-Pop header line.")
            header = stream.readline().rstrip("\r\n\t").split("\t")
            missing = columns.difference(header)
            if missing:
                raise ValueError(f"Missing P-Pop columns: {sorted(missing)}")

            for row in csv.DictReader(stream, fieldnames=header, delimiter="\t"):
                if not row or not row["Nuniverse"]:
                    continue
                universe_id = int(row["Nuniverse"])
                star_id = int(row["Nstar"])
                if not 0 <= universe_id < self.ppop_total_universes:
                    raise ValueError("P-Pop universe ID outside configured range.")
                if not 0 <= star_id < self.ppop_nstars:
                    raise ValueError("P-Pop star index outside configured range.")
                star_name = "".join(row["name"].split()).casefold()
                if not star_name:
                    raise ValueError("Empty P-Pop host-star name.")
                if star_id in source_names and source_names[star_id] != star_name:
                    raise ValueError("Inconsistent P-Pop Nstar-to-name mapping.")
                source_names[star_id] = star_name

                for column, bounds in observed.items():
                    value = float(row[column])
                    if not np.isfinite(value):
                        raise ValueError(f"Invalid P-Pop {column} value.")
                    if column in ("ap", "Rp", "Mp") and value <= 0:
                        raise ValueError(f"Nonphysical P-Pop {column} value.")
                    if column == "ep" and not 0 <= value < 1:
                        raise ValueError("Nonphysical P-Pop eccentricity.")
                    if column == "AgeomVIS" and not 0 <= value <= 1:
                        raise ValueError("Nonphysical visible geometric albedo.")
                    bounds[0] = min(bounds[0], value)
                    bounds[1] = max(bounds[1], value)
                count += 1
                seen_universes.add(universe_id)

        if count == 0:
            raise ValueError("Catalog contains no planets; eta cannot be derived.")

        self.ppop_planet_count = count
        self.ppop_nonempty_universes = len(seen_universes)
        self.ppop_empirical_eta = count / (
            self.ppop_total_universes * self.ppop_nstars
        )

        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        self.ppop_sha256 = digest.hexdigest()

        # Constructor values are observed catalog bounds, not a P-Pop PDF.
        # Do not call the inherited distribution methods: overridden below.
        super().__init__(
            arange=observed["ap"],
            erange=observed["ep"],
            Irange=np.rad2deg(observed["ip"]),
            Orange=np.rad2deg(observed["Omegap"]),
            wrange=np.rad2deg(observed["omegap"]),
            prange=observed["AgeomVIS"],
            Rprange=observed["Rp"],
            Mprange=observed["Mp"],
            eta=self.ppop_empirical_eta,
            scaleOrbits=False,
            constrainOrbits=False,
            **specs,
        )
        self._outspec["ppop_sha256"] = self.ppop_sha256
        self._outspec["ppop_planet_count"] = self.ppop_planet_count
        self._outspec["ppop_nonempty_universes"] = self.ppop_nonempty_universes
        self._outspec["ppop_empirical_eta"] = self.ppop_empirical_eta
        self._outspec["ppop_ranges_are_empirical"] = True

    @staticmethod
    def _unsupported():
        raise NotImplementedError(
            "Catalog-driven LIFEPlanetPopulation has no independent sampler "
            "or analytic occurrence-rate density. Use P-Pop catalogs through "
            "LIFESimulatedUniverse."
        )

    def gen_plan_params(self, n):
        """Reject independent generation of planetary orbital and physical parameters."""
        return self._unsupported()

    def gen_angles(self, n, commonSystemPlane=False, commonSystemPlaneParams=None):
        """Reject resampling of P-Pop orbital orientation angles."""
        return self._unsupported()

    def gen_mass(self, n):
        """Reject independent sampling of planetary masses."""
        return self._unsupported()

    def dist_sma(self, a):
        """Reject evaluation of an unavailable semi-major-axis density."""
        return self._unsupported()

    def dist_eccen(self, e):
        """Reject evaluation of an unavailable eccentricity density."""
        return self._unsupported()

    def dist_eccen_from_sma(self, e, a):
        """Reject evaluation of an unavailable conditional eccentricity density."""
        return self._unsupported()

    def dist_albedo(self, p):
        """Reject evaluation of an unavailable geometric-albedo density."""
        return self._unsupported()

    def dist_radius(self, Rp):
        """Reject evaluation of an unavailable planet-radius density."""
        return self._unsupported()

    def dist_mass(self, Mp):
        """Reject evaluation of an unavailable planet-mass density."""
        return self._unsupported()
