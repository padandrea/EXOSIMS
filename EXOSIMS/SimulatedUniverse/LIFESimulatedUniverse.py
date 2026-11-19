"""Import Kammerer-Quanz P-Pop planetary-system snapshots."""

import csv
from pathlib import Path

import astropy.units as u
from astropy.constants import G
from keplertools.keplerSTM import planSys
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

    @staticmethod
    def _orbital_basis(inclination, ascending_node, periapsis):
        """Return the inertial periapsis and transverse unit vectors."""
        ci, si = np.cos(inclination), np.sin(inclination)
        co, so = np.cos(ascending_node), np.sin(ascending_node)
        cw, sw = np.cos(periapsis), np.sin(periapsis)

        p = np.column_stack((co * cw - so * sw * ci, so * cw + co * sw * ci, sw * si))
        q = np.column_stack((-co * sw - so * cw * ci, -so * sw + co * cw * ci, cw * si))
        return p, q

    def gen_physical_properties(self, **specs):
        """Populate EXOSIMS planet attributes from one fixed P-Pop universe."""
        path = specs.get("ppop_catalog")
        uid = specs.get("ppop_universe_id")

        if path is None or uid is None:
            raise ValueError("ppop_catalog and ppop_universe_id are required.")

        if (
            self.Min is not None
            or self.commonSystemPlane
            or self.fixedPlanPerStar is not None
            or self.fixed_nEZ_val is not None
        ):
            raise ValueError(
                "P-Pop planets cannot be combined with resampling options."
            )

        data = self.load_ppop_universe(
            path,
            uid,
            total_universes=specs.get("ppop_total_universes"),
        )

        # Map P-Pop stars by identifier rather than assuming original
        # Nstar indices survived EXOSIMS TargetList filtering.
        target_names = [
            "".join(str(name).split()).casefold() for name in self.TargetList.Name
        ]

        if len(set(target_names)) != len(target_names):
            raise ValueError("TargetList contains ambiguous normalized star names.")

        lookup = {name: index for index, name in enumerate(target_names)}
        source_to_name = {}
        name_to_source = {}
        indices = []

        for source, name in zip(data["Nstar"], data["name"]):
            key = "".join(str(name).split()).casefold()
            source = int(source)

            if source in source_to_name and source_to_name[source] != key:
                raise ValueError("One P-Pop star index maps to multiple names.")
            if key in name_to_source and name_to_source[key] != source:
                raise ValueError("One P-Pop star name maps to multiple indices.")

            source_to_name[source] = key
            name_to_source[key] = source

            if key not in lookup:
                raise ValueError(f"P-Pop host star {name!r} is absent from TargetList.")

            indices.append(lookup[key])

        self.plan2star = np.asarray(indices, dtype=int)

        if self.plan2star.size:
            catalog_dist = data["star_distance"].to_value(u.pc)
            target_dist = self.TargetList.dist[self.plan2star].to_value(u.pc)

            if not np.allclose(
                catalog_dist,
                target_dist,
                rtol=1e-4,
                atol=1e-4,
            ):
                raise ValueError("P-Pop and TargetList stellar distances disagree.")

        # EXOSIMS planetary population attributes.
        self.nPlans = len(self.plan2star)
        self.sInds = np.unique(self.plan2star)

        self.a = data["a"].copy()
        self.e = data["e"].copy()
        self.I = data["I"].copy()
        self.O = data["O"].copy()
        self.w = data["w"].copy()
        self.Rp = data["Rp"].copy()
        self.Mp = data["Mp"].copy()

        # The conventional p attribute is visible geometric albedo.
        # It must not be used as the planet's thermal Bond albedo.
        self.p = data["p"].copy()
        self.nEZ = data["nEZ"].copy()

        # Additional LIFE-relevant properties from P-Pop.
        self.Abond = data["Abond"].copy()
        self.AgeomMIR = data["AgeomMIR"].copy()
        self.Tp = data["Tp"].copy()
        self.nu = data["nu"].copy()

        # Preserve source quantities for traceability.
        self.ppop_r_orbit = data["r_orbit"].copy()
        self.ppop_ang_sep = data["AngSep"].copy()
        self.ppop_nstar = data["Nstar"].copy()
        self.ppop_star_name = data["name"].copy()

        self.phiIndex = np.array([], dtype=int)

        # Convert P-Pop true anomaly to EXOSIMS initial mean anomaly.
        nu = self.nu.to_value(u.rad)
        eccentric_anomaly = np.arctan2(
            np.sqrt(1.0 - self.e**2) * np.sin(nu),
            self.e + np.cos(nu),
        )
        self.M0 = (
            np.mod(
                eccentric_anomaly - self.e * np.sin(eccentric_anomaly),
                2 * np.pi,
            )
            * u.rad
        )

        self._outspec["ppop_catalog"] = str(Path(path).expanduser().resolve())
        self._outspec["ppop_universe_id"] = int(uid)
        self._outspec["ppop_total_universes"] = specs.get("ppop_total_universes")

        self.planet_atts.extend(
            [
                "Abond",
                "AgeomMIR",
                "Tp",
                "nu",
                "ppop_r_orbit",
                "ppop_nstar",
                "ppop_star_name",
                "ppop_ang_sep",
                "planet_sky_positions",
            ]
        )

    def _update_life_geometry(self, pinds):
        """Refresh orbital geometry without reflected-light photometry."""
        if not len(pinds):
            return

        r = self.r[pinds].to_value(u.AU)

        self.d[pinds] = np.linalg.norm(r, axis=1) * u.AU
        self.s[pinds] = np.linalg.norm(r[:, :2], axis=1) * u.AU

        ratio = self.r[pinds, :2] / self.TargetList.dist[self.plan2star[pinds], None]
        self.planet_sky_positions[pinds] = ratio.decompose().value * u.rad

        projected_ratio = (
            (self.s[pinds] / self.TargetList.dist[self.plan2star[pinds]])
            .decompose()
            .value
        )

        self.WA[pinds] = (np.arctan(projected_ratio) * u.rad).to(u.arcsec)

        p, q = self._orbital_basis(
            self.I[pinds].to_value(u.rad),
            self.O[pinds].to_value(u.rad),
            self.w[pinds].to_value(u.rad),
        )

        self.nu[pinds] = (
            np.mod(
                np.arctan2(
                    np.sum(r * q, axis=1),
                    np.sum(r * p, axis=1),
                ),
                2 * np.pi,
            )
            * u.rad
        )

    def init_systems(self):
        """Initialize 3D position and velocity from the P-Pop snapshot."""
        n = self.nPlans
        self.AU_div_day = u.AU / u.day
        self.AU3_div_day2 = u.AU**3 / u.day**2

        mu = G * (self.Mp + self.TargetList.MsTrue[self.plan2star])
        self.mu_AU3_div_day2 = mu.to_value(self.AU3_div_day2)

        nu = self.nu.to_value(u.rad)

        p, q = self._orbital_basis(
            self.I.to_value(u.rad),
            self.O.to_value(u.rad),
            self.w.to_value(u.rad),
        )

        direction = np.cos(nu)[:, None] * p + np.sin(nu)[:, None] * q

        # Preserve P-Pop's original, rounded orbital radius.
        self.r = self.ppop_r_orbit[:, None] * direction

        speed = (mu / (self.a * (1 - self.e**2))) ** 0.5

        self.v = (
            speed[:, None]
            * (-np.sin(nu)[:, None] * p + (self.e + np.cos(nu))[:, None] * q)
        ).to(self.AU_div_day)

        self.d = np.zeros(n) * u.AU
        self.s = np.zeros(n) * u.AU
        self.WA = np.zeros(n) * u.arcsec
        self.planet_sky_positions = np.zeros((n, 2)) * u.rad

        # Reflected-light quantities are deliberately unavailable.
        self.dMag = np.full(n, np.nan)
        self.phi = np.full(n, np.nan)

        self._update_life_geometry(np.arange(n))

        if not np.allclose(
            self.WA.to_value(u.arcsec),
            self.ppop_ang_sep.to_value(u.arcsec),
            atol=3e-5,
            rtol=0.0,
        ):
            raise ValueError(
                "Reconstructed orbital geometry disagrees with " "P-Pop AngSep."
            )

    def propag_system(self, sInd, dt):
        """Propagate one planetary system using EXOSIMS's Kepler integrator."""
        if not isinstance(sInd, (int, np.integer)) or isinstance(
            sInd, (bool, np.bool_)
        ):
            raise TypeError("sInd must be a scalar star index.")

        if sInd < 0 or sInd >= len(self.TargetList.Name):
            raise ValueError("sInd is outside TargetList.")

        if not isinstance(dt, u.Quantity):
            raise TypeError("dt must have time units.")

        days = np.asarray(dt.to_value(u.day), dtype=float)

        if days.ndim != 0 or not np.isfinite(days) or days < 0:
            raise ValueError("dt must be finite and nonnegative.")

        pinds = np.flatnonzero(self.plan2star == sInd)

        if len(pinds) == 0 or days == 0:
            return

        state = np.column_stack(
            (
                self.r[pinds].to_value(u.AU),
                self.v[pinds].to_value(self.AU_div_day),
            )
        ).ravel()

        mu = self.mu_AU3_div_day2[pinds]

        integrator = planSys(state, mu, epsmult=10.0)

        try:
            integrator.takeStep(float(days))
        except ValueError:
            integrator = planSys(state, mu, epsmult=100.0)
            integrator.takeStep(float(days) / 2)
            integrator.takeStep(float(days) / 2)

        evolved = np.asarray(integrator.x0).reshape(-1, 6)

        self.r[pinds] = evolved[:, :3] * u.AU
        self.v[pinds] = evolved[:, 3:] * self.AU_div_day

        self._update_life_geometry(pinds)
