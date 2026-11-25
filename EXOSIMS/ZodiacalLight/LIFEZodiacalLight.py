"""Zodiacal-light source models for the LIFE mid-infrared interferometer."""

import astropy.units as u
import numpy as np
from astropy.constants import c, h, k_B

from EXOSIMS.ZodiacalLight.Mennesson import Mennesson

KENNEDY_TEMPERATURE_1AU_K = 278.3
KENNEDY_SUBLIMATION_TEMPERATURE_K = 1500.0


class LIFEZodiacalLight(Mennesson):
    """LIFE zodiacal-light model extending Mennesson.

    Inherits the local-zodiacal-light methods and exozodi-level
    distribution from Mennesson.

    Adds the face-on, axisymmetric thermal exozodi prescription
    of Kennedy et al. (2015).
    """

    def get_local_zodi_spectral_radiance(
        self,
        wavelengths: u.Quantity,
        longitude: u.Quantity,
        latitude: u.Quantity,
        model: str = "mennesson",
    ) -> u.Quantity:
        """Local-zodiacal photon spectral radiance at the observer.

        The default reproduces the inherited EXOSIMS Mennesson/Leinert
        model. The optional ``darwinsim`` model reproduces LIFEsim's
        265 K dust + scattered-sunlight prescription, including its
        wavelength-dependent ecliptic geometry factor. Unlike the
        reference LIFEsim code, longitude is passed explicitly.
        """
        if model not in ("mennesson", "darwinsim"):
            raise ValueError("model must be 'mennesson' or 'darwinsim'.")
        if not all(
            isinstance(value, u.Quantity)
            for value in (wavelengths, longitude, latitude)
        ):
            raise TypeError("wavelengths, longitude, and latitude need units.")

        wavelength = np.atleast_1d(wavelengths.to(u.m))
        if (
            wavelength.ndim != 1
            or wavelength.size == 0
            or not np.all(np.isfinite(wavelength.value))
            or np.any(wavelength.value <= 0)
        ):
            raise ValueError("wavelengths must be finite and positive.")

        lon = np.asarray(longitude.to_value(u.rad))
        lat = np.asarray(latitude.to_value(u.rad))
        if lon.ndim != 0 or lat.ndim != 0:
            raise ValueError("Local-zodi longitude and latitude must be scalar.")
        if not np.isfinite(lon) or not np.isfinite(lat):
            raise ValueError("Local-zodi coordinates must be finite.")

        radiance_unit = u.photon / (u.s * u.m**2 * u.um * u.sr)
        if abs(lat) > np.pi / 2:
            raise ValueError("ecliptic latitude must be within [-90, 90] deg.")

        if model == "mennesson":
            # Leinert Table 17 covers solar-relative longitude
            # 0--180 degrees and absolute latitude 0--90 degrees.
            folded_lon = abs((longitude.to_value(u.deg) + 180.0) % 360.0 - 180.0)
            folded_lat = abs(latitude.to_value(u.deg))

            reference = self.zodi_intensity_at_location(
                np.atleast_1d(folded_lon) * u.deg,
                np.atleast_1d(folded_lat) * u.deg,
                photon_units=True,
            )
            color = self.zodi_color_correction_factor(wavelength, photon_units=True)
            return (reference[0] * color).to(radiance_unit)

        # LIFEsim pn_localzodi.py: elongation angle and dust scale height.
        cos_elongation = np.clip(np.cos(lon) * np.cos(lat), -1.0, 1.0)
        elongation = np.arccos(cos_elongation)
        if elongation <= 0:
            raise ValueError("Darwinsim is singular toward the Sun.")

        lambda_ratio = wavelength.to_value(u.um) / 11.0
        geometry_denominator = (
            np.sin(lat) ** 2 + (0.6 * lambda_ratio ** (-0.4) * np.cos(lat)) ** 2
        )
        geometry = np.sqrt(np.pi / elongation / geometry_denominator)

        # Planck photon spectral radiance, in photons / (s m2 um sr).
        def planck_photons(temperature: u.Quantity) -> u.Quantity:
            exponent = (h * c / (wavelength * k_B * temperature)).decompose().value
            with np.errstate(over="ignore"):
                return (
                    2.0 * c * u.photon / (wavelength**4 * u.sr * np.expm1(exponent))
                ).to(radiance_unit)

        thermal = planck_photons(265.0 * u.K)
        scattered = 0.22 * (0.00465047 / 1.5) ** 2 * planck_photons(5777.0 * u.K)
        return (4.0e-8 * geometry * (thermal + scattered)).to(radiance_unit)

    @staticmethod
    def get_exozodi_inner_radius(
        star_luminosity: float | u.Quantity,
    ) -> u.Quantity:
        """Return the Kennedy model's inner dust radius in AU."""
        luminosity = (
            float(star_luminosity.to_value(u.Lsun))
            if isinstance(star_luminosity, u.Quantity)
            else float(star_luminosity)
        )

        if not np.isfinite(luminosity) or luminosity <= 0:
            raise ValueError("star_luminosity must be positive.")

        return (
            (KENNEDY_TEMPERATURE_1AU_K / KENNEDY_SUBLIMATION_TEMPERATURE_K) ** 2
            * np.sqrt(luminosity)
            * u.au
        )

    def get_exozodi_inner_angle(self, star_distance, star_luminosity):
        """Return the angular dust sublimation radius."""
        distance = star_distance.to_value(u.m)

        if np.ndim(distance) != 0 or not np.isfinite(distance) or distance <= 0:
            raise ValueError("star_distance must be finite and positive.")

        radius = self.get_exozodi_inner_radius(star_luminosity)

        return radius.to_value(u.m) / distance * u.rad

    def get_exozodi_radial_profile(
        self,
        wavelengths,
        nEZ,
        star_distance,
        star_luminosity,
        outer_radius,
        n_samples,
        spacing="linear",
    ):
        """Return a boundary-aligned Kennedy radial brightness profile."""
        if spacing not in ("linear", "log"):
            raise ValueError("Invalid radial spacing.")

        if (
            not isinstance(n_samples, (int, np.integer))
            or isinstance(n_samples, (bool, np.bool_))
            or n_samples < 3
        ):
            raise ValueError("n_samples must be an integer >= 3.")

        inner = self.get_exozodi_inner_angle(star_distance, star_luminosity).to_value(
            u.rad
        )

        outer = outer_radius.to_value(u.rad)

        if np.ndim(outer) != 0 or not np.isfinite(outer) or outer <= 0:
            raise ValueError("Invalid outer_radius.")

        empty = inner >= outer

        if empty:
            radii = np.array([0.0])
        elif spacing == "linear":
            radii = np.linspace(inner, outer, n_samples)
        else:
            radii = np.geomspace(inner, outer, n_samples)

        sky = np.column_stack((radii, np.zeros_like(radii))) * u.rad

        intensity = self.get_exozodi_spectral_radiance(
            sky_positions=sky,
            wavelengths=wavelengths,
            nEZ=nEZ,
            star_distance=star_distance,
            star_luminosity=star_luminosity,
        )

        if empty:
            return radii[:0] * u.rad, intensity[:0]

        return radii * u.rad, intensity

    def get_exozodi_thin_rings(
        self,
        wavelengths,
        nEZ,
        star_distance,
        star_luminosity,
        fov_radii,
        n_rings,
    ):
        """Approximate Kennedy emission with logarithmic thin rings."""
        if (
            not isinstance(n_rings, (int, np.integer))
            or isinstance(n_rings, (bool, np.bool_))
            or n_rings < 3
        ):
            raise ValueError("n_rings must be an integer >= 3.")

        fov = np.atleast_1d(fov_radii.to_value(u.rad))

        if (
            fov.shape != np.atleast_1d(wavelengths.value).shape
            or not np.all(np.isfinite(fov))
            or np.any(fov <= 0)
        ):
            raise ValueError("Invalid FoV radii.")

        inner = self.get_exozodi_inner_angle(star_distance, star_luminosity).to_value(
            u.rad
        )

        flux_unit = u.photon / (u.s * u.m**2 * u.um)

        def radiance(radii):
            sky = np.column_stack((radii, np.zeros_like(radii))) * u.rad

            return self.get_exozodi_spectral_radiance(
                sky_positions=sky,
                wavelengths=wavelengths,
                nEZ=nEZ,
                star_distance=star_distance,
                star_luminosity=star_luminosity,
            ).to_value(flux_unit / u.sr)

        if inner >= fov.max():
            radiance(np.array([0.0]))

            return (
                np.empty((0, fov.size)) * u.rad,
                np.empty((0, fov.size)) * flux_unit,
            )

        edges = np.geomspace(inner, fov.max(), n_rings + 1)
        centres = np.sqrt(edges[:-1] * edges[1:])
        intensity = radiance(centres)

        lo = edges[:-1, None]
        hi = np.maximum(lo, np.minimum(edges[1:, None], fov[None, :]))

        radii = np.sqrt(lo * hi)

        index = np.clip(
            np.searchsorted(edges, fov, side="left") - 1,
            0,
            n_rings - 1,
        )

        column = np.arange(fov.size)

        intensity[index, column] = np.diag(radiance(radii[index, column]))

        fluxes = np.pi * (hi**2 - lo**2) * intensity

        radii = np.where(hi > lo, radii, 0.0)

        return radii * u.rad, fluxes * flux_unit

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
        angular_radius = np.linalg.norm(theta, axis=1)
        radius_au = angular_radius * distance_m / u.au.to(u.m)

        # Kennedy et al. (2015) radial model.
        alpha = 0.34
        sigma_zero = 7.11889e-8
        reference_radius_au = np.sqrt(luminosity)
        inner_radius = self.get_exozodi_inner_radius(luminosity)
        inner_radius_au = inner_radius.to_value(u.au)

        # Compare angular radii directly to evaluate the dust-side
        # limit consistently at the integration boundary.
        inner_angle = inner_radius.to_value(u.m) / distance_m
        valid = angular_radius >= inner_angle

        # Avoid division by zero at the star's position.
        safe_radius_au = np.maximum(radius_au, inner_radius_au)

        # Dust temperature in K.
        temperature = (
            KENNEDY_TEMPERATURE_1AU_K * luminosity**0.25 / np.sqrt(safe_radius_au)
        ) * u.K

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
