"""LIFE-specific EXOSIMS target-list inputs and stellar blackbody fluxes.

The LIFE catalog uses physical stellar properties without reconstructing
optical photometry. The inherited exozodi/completeness initialization remains
blocked until the LIFE observation and completeness interfaces are connected.
"""

import astropy.units as u
import numpy as np
from astropy.constants import c, h, k_B
from scipy.integrate import trapezoid

from EXOSIMS.Prototypes.TargetList import TargetList


class _CatalogOnlyReady(Exception):
    """Signal completion of physical stellar initialization."""


class LIFETargetList(TargetList):
    """TargetList using catalogued LTC4 physical stellar properties.

    No V-band photometry is fabricated, and stellar masses are taken from LTC4.
    Full construction intentionally stops at ``calc_all_JEZ0`` until the LIFE
    observation and completeness interfaces replace reflected-light assumptions.
    """

    def __init__(self, *args, optional_filters=None, catalog_only=False, **specs):
        """Initialize LIFE stellar properties without legacy detection physics."""
        filters = {} if optional_filters is None else dict(optional_filters)

        for name in ("outside_IWA_filter", "completeness_filter"):
            if filters.get(name, {}).get("enabled", False):
                raise NotImplementedError(
                    f"{name} requires a LIFE-specific implementation."
                )
            filters[name] = {"enabled": False}

        if catalog_only:
            if specs.get("filterBinaries", True):
                raise ValueError("catalog_only requires filterBinaries=False.")

            active = [
                name for name, config in filters.items() if config.get("enabled", False)
            ]
            if active:
                raise ValueError(
                    f"catalog_only does not support active filters: {active}"
                )

            if (
                specs.get("getKnownPlanets", False)
                or specs.get("filter_for_char", False)
                or specs.get("earths_only", False)
            ):
                raise ValueError(
                    "catalog_only does not support known-planet queries "
                    "or characterization filtering."
                )

        self._catalog_only = bool(catalog_only)

        try:
            super().__init__(
                *args,
                optional_filters=filters,
                **specs,
            )
        except _CatalogOnlyReady:
            if not self._catalog_only:
                raise

            self.hasKnownPlanet = np.zeros(self.nStars, dtype=bool)
            self.catalog_atts.append("hasKnownPlanet")
            self._outspec["nStars"] = self.nStars
            self._outspec["catalog_only"] = True

    def set_catalog_attributes(self):
        """Require LIFE physical inputs, not unavailable optical magnitudes."""
        self.catalog_atts = list(self.StarCatalog.catalog_atts)
        self.required_catalog_atts = [
            "Name",
            "L",
            "coords",
            "dist",
            "Binary_Cut",
            "Teff",
            "radius",
            "mass",
            "diameter",
        ]
        missing = set(self.required_catalog_atts) - set(self.catalog_atts)
        if missing:
            raise ValueError(
                f"LIFEStarCatalog is missing attributes: {sorted(missing)}"
            )

    def fillPhotometryVals(self):
        """Preserve catalog properties without synthesizing optical magnitudes."""
        if self.fillPhotometry or self.fillMissingBandMags:
            raise ValueError(
                "LIFETargetList does not synthesize optical photometry. "
                "Set fillPhotometry=False and fillMissingBandMags=False."
            )

    def stellar_mass(self):
        """Use LTC4 stellar masses instead of a V-band mass-luminosity fit."""
        self.MsEst = self.mass.to(u.Msun).copy()
        self.MsTrue = self.MsEst.copy()
        for name in ("MsEst", "MsTrue"):
            if name not in self.catalog_atts:
                self.catalog_atts.append(name)

    def genStarCatalogHex(self):
        """Separate LIFE blackbody caches from legacy optical-template caches."""
        return f"LIFE_BB_v1_{super().genStarCatalogHex()}"

    @staticmethod
    def blackbody_photon_flux_density(wavelengths, temperatures, radii, distances):
        """Return stellar photon flux per unit wavelength at the observer.

        Implements F_lambda = pi B_lambda(T) (R/d)^2, expressed in photons.

        Parameters
        ----------
        wavelengths : astropy.units.Quantity
            One-dimensional wavelengths with length units.
        temperatures, radii, distances : astropy.units.Quantity
            Equal-length, one-dimensional arrays in K, length, and length.

        Returns
        -------
        astropy.units.Quantity
            Array with shape (n_stars, n_wavelengths) and units
            photon / (s m2 m).
        """
        wavelength_m = np.atleast_1d(wavelengths.to_value(u.m))
        temperature_k = np.atleast_1d(temperatures.to_value(u.K))
        radius_m = np.atleast_1d(radii.to_value(u.m))
        distance_m = np.atleast_1d(distances.to_value(u.m))

        if wavelength_m.ndim != 1:
            raise ValueError("wavelengths must be one-dimensional")
        if not (temperature_k.shape == radius_m.shape == distance_m.shape):
            raise ValueError(
                "temperatures, radii, and distances must have matching shapes"
            )
        if (
            not np.all(np.isfinite(wavelength_m) & (wavelength_m > 0))
            or not np.all(np.isfinite(temperature_k) & (temperature_k > 0))
            or not np.all(np.isfinite(radius_m) & (radius_m > 0))
            or not np.all(np.isfinite(distance_m) & (distance_m > 0))
        ):
            raise ValueError(
                "wavelength, temperature, radius, and distance must be finite and positive"
            )

        hc_over_k = (h * c / k_B).to_value(u.m * u.K)
        speed_of_light = c.to_value(u.m / u.s)
        exponent = hc_over_k / (temperature_k[:, None] * wavelength_m[None, :])
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            photons = (
                2.0
                * np.pi
                * speed_of_light
                / wavelength_m[None, :] ** 4
                / np.expm1(exponent)
                * (radius_m / distance_m)[:, None] ** 2
            )

        return photons * (u.photon / (u.s * u.m**3))

    def starFlux(self, sInds, mode):
        """Return band-integrated stellar blackbody photon flux [ph/(s m2)].

        Uses the physical LTC4 stellar parameters, not optical magnitudes.
        The instrument bandpass is applied over the explicitly configured LIFE
        wavelength interval (specified in nanometres in the observing mode).
        """
        sInds = np.atleast_1d(np.asarray(sInds, dtype=int))
        if np.any((sInds < 0) | (sInds >= self.nStars)):
            raise IndexError("stellar index is outside the TargetList")

        if not all(
            key in mode
            for key in ("hex", "bandpass", "LIFE_lambda_min", "LIFE_lambda_max")
        ):
            raise ValueError(
                "LIFE mode requires hex, bandpass, LIFE_lambda_min and LIFE_lambda_max"
            )

        flux_unit = u.photon / (u.s * u.m**2)
        if mode["hex"] not in self.star_fluxes:
            self.star_fluxes[mode["hex"]] = np.full(self.nStars, np.nan) * flux_unit

        missing = np.unique(sInds[np.isnan(self.star_fluxes[mode["hex"]][sInds].value)])
        if missing.size:
            lower = float(mode["LIFE_lambda_min"])
            upper = float(mode["LIFE_lambda_max"])
            if (
                not np.isfinite(lower)
                or not np.isfinite(upper)
                or lower <= 0
                or upper <= lower
            ):
                raise ValueError(
                    "LIFE wavelength bounds must be finite, positive and increasing"
                )

            wavelengths = np.linspace(lower, upper, 257) * u.nm
            response = mode["bandpass"](wavelengths)
            if isinstance(response, u.Quantity):
                response = response.to_value(u.dimensionless_unscaled)
            response = np.asarray(response, dtype=float)
            if response.shape != (wavelengths.size,) or not np.all(
                np.isfinite(response)
            ):
                raise ValueError("bandpass must return one finite value per wavelength")
            if np.any((response < 0) | (response > 1)):
                raise ValueError("bandpass transmission must lie between zero and one")

            density = self.blackbody_photon_flux_density(
                wavelengths,
                self.Teff[missing],
                self.radius[missing],
                self.dist[missing],
            )
            values = trapezoid(
                density.to_value(u.photon / (u.s * u.m**3)) * response[None, :],
                x=wavelengths.to_value(u.m),
                axis=1,
            )
            self.star_fluxes[mode["hex"]][missing] = values * flux_unit

        return self.star_fluxes[mode["hex"]][sInds]

    def calc_all_JEZ0(self):
        """Prevent the inherited visible-light exozodi prescription being used."""
        if getattr(self, "_catalog_only", False):
            raise _CatalogOnlyReady()
        raise NotImplementedError(
            "LIFETargetList requires a LIFE-specific exozodi/completeness "
            "interface. The inherited TargetList.calc_all_JEZ0 uses "
            "optical color factors and MV, which are not valid LIFE "
            "mid-IR background inputs. Stellar starFlux() is implemented "
            "and can be tested independently."
        )

    def calc_saturation_and_intCutoff_vals(self):
        """Prevent inherited reflected-light detection-limit calculations."""
        raise NotImplementedError(
            "LIFE-specific detection limits require observation-dependent "
            "spectral signal and noise rates, not inherited reflected-light "
            "saturation and dMag calculations."
        )
