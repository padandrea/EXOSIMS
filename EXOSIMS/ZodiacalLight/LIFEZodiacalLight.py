"""Zodiacal-light source models for the LIFE mid-infrared interferometer."""

import astropy.units as u
import numpy as np
from astropy.constants import c, h, k_B

from EXOSIMS.ZodiacalLight.Mennesson import Mennesson


class LIFEZodiacalLight(Mennesson):
    """LIFE zodiacal-light model extending Mennesson.

    Inherits the local-zodiacal-light methods and exozodi-level
    distribution from Mennesson.

    Adds the face-on, axisymmetric thermal exozodi prescription
    of Kennedy et al. (2015).
    """

    def get_exozodi_spectral_radiance(
        self,
        sky_positions: u.Quantity,
        wavelengths: u.Quantity,
        nEZ: float,
        star_distance: u.Quantity,
        star_luminosity: float | u.Quantity,
    ) -> u.Quantity:
        """Calculate the thermal radiance of a face-on exozodiacal disk.

        Args:
            sky_positions:
                Sky-plane angular offsets, shape (n_sky, 2).
            wavelengths:
                Wavelength-bin centers, shape (n_wavelengths,).
            nEZ:
                Exozodiacal dust level in zodis.
            star_distance:
                Distance to the target star.
            star_luminosity:
                Stellar luminosity in solar luminosities, either as
                a dimensionless float or an Astropy Quantity.

        Returns:
            Spectral photon radiance with shape
            (n_sky, n_wavelengths), in
            photon / (s m2 um sr).

        Notes:
            The disk is assumed to be face-on and axisymmetric.
            The model includes the Kennedy et al. inner dust edge.
            There is no additional outer-temperature cutoff.
            Field-of-view attenuation is handled by OpticalSystem.
        """
        radiance_unit = u.photon / (u.s * u.m**2 * u.um * u.sr)

        if not isinstance(sky_positions, u.Quantity):
            raise TypeError("sky_positions must be an Astropy Quantity.")

        if not isinstance(wavelengths, u.Quantity):
            raise TypeError("wavelengths must be an Astropy Quantity.")

        if not isinstance(star_distance, u.Quantity):
            raise TypeError("star_distance must be an Astropy Quantity.")

        theta = np.asarray(sky_positions.to_value(u.rad))
        if theta.ndim == 1:
            theta = theta[None, :]

        if (
            theta.ndim != 2
            or theta.shape[1] != 2
            or theta.shape[0] == 0
            or not np.all(np.isfinite(theta))
        ):
            raise ValueError("sky_positions must have shape (n_sky, 2).")

        wavelength_m = np.atleast_1d(wavelengths.to(u.m))
        if (
            wavelength_m.ndim != 1
            or wavelength_m.size == 0
            or not np.all(np.isfinite(wavelength_m.value))
            or np.any(wavelength_m.value <= 0)
        ):
            raise ValueError("wavelengths must be finite and positive.")

        distance_m = star_distance.to_value(u.m)
        if np.ndim(distance_m) != 0 or not np.isfinite(distance_m) or distance_m <= 0:
            raise ValueError("star_distance must be finite and positive.")

        if isinstance(star_luminosity, u.Quantity):
            luminosity = float(star_luminosity.to_value(u.Lsun))
        else:
            luminosity = float(star_luminosity)

        zodi_level = float(nEZ)

        if not np.isfinite(luminosity) or luminosity <= 0:
            raise ValueError("star_luminosity must be positive.")

        if not np.isfinite(zodi_level) or zodi_level < 0:
            raise ValueError("nEZ must be non-negative.")

        # Projected physical distance from the star in AU.
        radius_au = np.linalg.norm(theta, axis=1) * distance_m / u.au.to(u.m)

        # Kennedy et al. (2015) radial model.
        alpha = 0.34
        sigma_zero = 7.11889e-8
        reference_radius_au = np.sqrt(luminosity)
        inner_radius_au = 0.034422617777777775 * np.sqrt(luminosity)

        valid = radius_au >= inner_radius_au

        # Avoid division by zero at the star's position.
        safe_radius_au = np.maximum(radius_au, inner_radius_au)

        # Dust temperature in K.
        temperature = (278.3 * luminosity**0.25 / np.sqrt(safe_radius_au)) * u.K

        # Dimensionless dust surface density / optical depth.
        sigma = (
            sigma_zero * zodi_level * (safe_radius_au / reference_radius_au) ** (-alpha)
        )

        # Planck law expressed in photon spectral radiance.
        exponent = (
            (h * c / (wavelength_m[None, :] * k_B * temperature[:, None]))
            .decompose()
            .value
        )

        with np.errstate(over="ignore"):
            blackbody = (
                2.0
                * c
                * u.photon
                / (wavelength_m[None, :] ** 4 * u.sr * np.expm1(exponent))
            )

        radiance = np.where(valid, sigma, 0.0)[:, None] * blackbody

        return radiance.to(radiance_unit)
