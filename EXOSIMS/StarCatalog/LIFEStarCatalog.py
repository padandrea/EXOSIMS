"""EXOSIMS StarCatalog adapter for the LIFE Target Catalog (LTC4).

This module reads the source catalog without applying target-selection cuts.
LTC4 does not supply reliable optical V and B-V photometry; those fields
remain NaN until a separate, documented photometric strategy is implemented.
"""

from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.constants import L_sun, sigma_sb
from astropy.coordinates import SkyCoord

from EXOSIMS.Prototypes.StarCatalog import StarCatalog


class LIFEStarCatalog(StarCatalog):
    """Import all LTC4 stars into the EXOSIMS StarCatalog interface.

    Args:
        catalogpath: Path to P-Pop's ``LTC4.csv``. Required explicitly to
            avoid accidentally loading a different catalog version.
        **specs: EXOSIMS StarCatalog options.

    Notes:
        No spectral-type, distance, or binary-star selection is applied here.
        ``Binary_Cut`` provisionally mirrors ``binary_flag`` as a conservative
        policy; LTC4 does not supply the angular companion-separation criterion
        used by standard EXOSIMS. Set ``filterBinaries=False`` in TargetList
        if that provisional cut is not intended.

        Luminosity is estimated using radius and effective temperature via
        Stefan-Boltzmann. Missing optical magnitudes remain NaN, meaning the
        standard EXOSIMS TargetList cannot yet be initialized unchanged.
    """

    def __init__(self, catalogpath=None, **specs):
        if catalogpath is None:
            raise ValueError("Provide the path to LTC4.csv using catalogpath.")

        path = Path(catalogpath).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"LTC4 catalog not found: {path}")

        data = pd.read_csv(path)
        required = {
            "main_id",
            "coo_ra",
            "coo_dec",
            "dist_st_value",
            "plx_value",
            "sptype_string",
            "teff",
            "radius_st_value",
            "mass_st_value",
            "binary_flag",
        }
        missing = sorted(required.difference(data.columns))
        if missing:
            raise ValueError(f"LTC4 is missing required columns: {missing}")
        if data.empty:
            raise ValueError("LTC4 contains no stars.")

        n_stars = len(data)
        if "ntargs" in specs:
            if int(specs["ntargs"]) != n_stars:
                raise ValueError("ntargs does not match the LTC4 row count.")
            specs.pop("ntargs")

        super().__init__(ntargs=n_stars, **specs)
        self._outspec["catalogpath"] = str(path)
        self.data = data

        def numeric(column):
            return pd.to_numeric(data[column], errors="coerce").to_numpy(dtype=float)

        self.Name = data["main_id"].fillna("").astype(str).to_numpy()
        self.Spec = data["sptype_string"].fillna("").astype(str).to_numpy()
        self.dist = numeric("dist_st_value") * u.pc
        self.parx = numeric("plx_value") * u.mas
        self.Teff = numeric("teff") * u.K  # P-Pop uses 'teff', not 'teff_st_value'.
        self.radius = numeric("radius_st_value") * u.Rsun
        self.mass = numeric("mass_st_value") * u.Msun

        ra = numeric("coo_ra")
        dec = numeric("coo_dec")
        d = self.dist.to_value(u.pc)
        if (
            not np.all(np.isfinite(ra))
            or not np.all(np.isfinite(dec))
            or not np.all(np.isfinite(d) & (d > 0))
        ):
            raise ValueError("LTC4 has invalid coordinates or stellar distances.")

        # LTC4 does not include proper motions in these fields. Zero here means
        # "not modeled", not a claim of measured zero stellar proper motion.
        self.coords = SkyCoord(
            ra=ra * u.deg,
            dec=dec * u.deg,
            distance=self.dist,
            pm_ra_cosdec=np.zeros(n_stars) * u.mas / u.yr,
            pm_dec=np.zeros(n_stars) * u.mas / u.yr,
            radial_velocity=np.zeros(n_stars) * u.km / u.s,
            obstime=self.catalog_epoch,
        )

        self.diameter = (
            (2 * self.radius.to_value(u.m) / self.dist.to_value(u.m)) * u.rad
        ).to(u.mas)
        bolometric_luminosity = 4 * np.pi * sigma_sb * self.radius**2 * self.Teff**4
        self.L = (bolometric_luminosity / L_sun).decompose().value

        flags = data["binary_flag"].astype("string").str.strip().str.lower()
        if not flags.isin(["true", "false"]).all():
            raise ValueError("LTC4 binary_flag contains values other than True/False.")
        self.binary_flag = flags.eq("true").to_numpy(dtype=bool)
        self.Binary_Cut = self.binary_flag.copy()  # Provisional, see class notes.

        # Preserve the photometry LTC4 actually supplies. Optical V and B-V
        # are unavailable; do NOT use the prototype's numerical placeholders.
        self.Imag = (
            numeric("mag_i_value")
            if "mag_i_value" in data
            else np.full(n_stars, np.nan)
        )
        self.Jmag = (
            numeric("mag_j_value")
            if "mag_j_value" in data
            else np.full(n_stars, np.nan)
        )
        for attribute in (
            "Umag",
            "Bmag",
            "Vmag",
            "Rmag",
            "Hmag",
            "Kmag",
            "BV",
            "MV",
            "BC",
        ):
            setattr(self, attribute, np.full(n_stars, np.nan))

        self.catalog_atts += ["Teff", "radius", "diameter", "mass", "binary_flag"]
