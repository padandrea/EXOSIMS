"""Survey simulation interface for the LIFE nulling interferometer."""

import astropy.units as u
import numpy as np

from EXOSIMS.Prototypes.SurveySimulation import SurveySimulation


class LIFESurveySimulation(SurveySimulation):
    """LIFE survey simulation extending the EXOSIMS SurveySimulation.

    The base class retains responsibility for mission scheduling,
    observations, and simulation state.

    LIFE-specific methods coordinate the Observatory, ZodiacalLight,
    and OpticalSystem modules without duplicating their physics.
    """

    @staticmethod
    def make_life_sky_grid(
        half_width: u.Quantity,
        pixels_per_axis: int,
    ) -> tuple[u.Quantity, u.Quantity]:
        """Generate a square sky grid and its pixel solid angles.

        Args:
            half_width:
                Angular half-width of the square field.
            pixels_per_axis:
                Number of pixels along each sky axis.

        Returns:
            sky_positions:
                Angular positions with shape (n_pixels, 2).
            pixel_solid_angles:
                Solid angle of each pixel with shape (n_pixels,).

        Notes:
            This method generates the integration grid only.
            It does not impose an optical field-of-view taper.
        """
        if not isinstance(half_width, u.Quantity):
            raise TypeError("half_width must be an Astropy Quantity.")

        width_rad = half_width.to_value(u.rad)

        if np.ndim(width_rad) != 0 or not np.isfinite(width_rad) or width_rad <= 0:
            raise ValueError("half_width must be finite and positive.")

        if (
            not isinstance(pixels_per_axis, (int, np.integer))
            or isinstance(pixels_per_axis, (bool, np.bool_))
            or pixels_per_axis < 2
        ):
            raise ValueError("pixels_per_axis must be an integer >= 2.")

        n = int(pixels_per_axis)
        pixel_width = 2.0 * width_rad / n

        axis = (np.arange(n) + 0.5) * pixel_width - width_rad

        xx, yy = np.meshgrid(axis, axis, indexing="ij")

        sky_positions = np.stack((xx.ravel(), yy.ravel()), axis=-1) * u.rad

        pixel_solid_angles = np.full(n * n, pixel_width**2) * u.sr

        return sky_positions, pixel_solid_angles

    def get_life_zodiacal_rates(
        self,
        sInd: int,
        wavelengths: u.Quantity,
        bin_widths: u.Quantity,
        sky_positions: u.Quantity,
        pixel_solid_angles: u.Quantity,
        rotation_angles: u.Quantity,
        mode: dict,
        local_zodi_longitude: u.Quantity,
        local_zodi_latitude: u.Quantity,
        nEZ: float | None = None,
        nulling_baseline: u.Quantity | None = None,
    ) -> dict[str, u.Quantity]:
        """Calculate local and exozodiacal photon rates for one star.

        Uses the configured TargetList and SimulatedUniverse to obtain
        stellar properties and, when possible, the system's dust level.

        Args:
            sInd:
                Index of the target star in TargetList.
            wavelengths:
                Spectral-bin centers.
            bin_widths:
                Spectral-bin widths.
            sky_positions:
                Sky-grid angular positions, shape (n_sky, 2).
            pixel_solid_angles:
                Sky-pixel solid angles, shape (n_sky,).
            rotation_angles:
                Requested array orientations.
            mode:
                LIFE observing mode.
            local_zodi_longitude:
                Target longitude relative to the Sun, as seen by
                the observatory.
            local_zodi_latitude:
                Target ecliptic latitude.
            nEZ:
                Exozodiacal dust level. If omitted, use the value
                from SimulatedUniverse when unambiguous.
            nulling_baseline:
                Requested nulling baseline, or the Observatory default.

        Returns:
            Dictionary containing:
                local_left, local_right,
                exozodi_left, exozodi_right.

            Each rate has units 1/s and shape
            (n_configurations, n_wavelengths).

        Notes:
            The exozodi is face-on and axisymmetric.
            The local foreground uses the inherited Mennesson model.
            Optical field-of-view attenuation is not added here.
        """
        if (
            not isinstance(sInd, (int, np.integer))
            or isinstance(sInd, (bool, np.bool_))
            or sInd < 0
            or sInd >= len(self.TargetList.L)
        ):
            raise ValueError("sInd must identify a valid target star.")

        TL = self.TargetList
        SU = self.SimulatedUniverse
        ZL = self.ZodiacalLight
        Obs = self.Observatory
        OS = self.OpticalSystem

        # ---------------------------------------------------------
        # 1. Stellar parameters
        # ---------------------------------------------------------
        star_distance = TL.dist[sInd]
        star_luminosity = TL.L[sInd]

        # ---------------------------------------------------------
        # 2. Exozodiacal dust level
        # ---------------------------------------------------------
        if nEZ is None:
            planet_indices = np.flatnonzero(np.asarray(SU.plan2star) == sInd)

            if planet_indices.size == 0:
                raise ValueError(
                    "No planet-associated nEZ value for this star. "
                    "Provide an explicit nEZ."
                )

            dust_levels = np.asarray(SU.nEZ, dtype=float)[planet_indices]

            if not np.all(dust_levels == dust_levels[0]):
                raise ValueError(
                    "Multiple nEZ values found for one star. "
                    "Provide an explicit system-level nEZ."
                )

            nEZ = float(dust_levels[0])

        # ---------------------------------------------------------
        # 3. Exozodiacal spectral radiance
        # ---------------------------------------------------------
        exozodi_radiance = ZL.get_exozodi_spectral_radiance(
            sky_positions=sky_positions,
            wavelengths=wavelengths,
            nEZ=nEZ,
            star_distance=star_distance,
            star_luminosity=star_luminosity,
        )

        # ---------------------------------------------------------
        # 4. Local-zodiacal spectral radiance
        # ---------------------------------------------------------
        lon = np.atleast_1d(local_zodi_longitude.to(u.deg))
        lat = np.atleast_1d(local_zodi_latitude.to(u.deg))

        if lon.size != 1 or lat.size != 1:
            raise ValueError("Local-zodi longitude and latitude must be scalar.")

        if not np.all(np.isfinite(lon.value)) or not np.all(np.isfinite(lat.value)):
            raise ValueError("Local-zodi coordinates must be finite.")

        # Local zodiacal brightness at the reference wavelength.
        local_reference = ZL.zodi_intensity_at_location(
            lon,
            lat,
            photon_units=True,
        )

        # Convert the inherited local-zodi spectrum to the
        # requested LIFE wavelength bins.
        color_correction = ZL.zodi_color_correction_factor(
            wavelengths,
            photon_units=True,
        )

        local_spectrum = (local_reference[0] * color_correction).to(
            u.photon / (u.s * u.m**2 * u.um * u.sr)
        )

        # Local zodi is approximately uniform over the
        # small interferometric field of view.
        local_radiance = (
            np.broadcast_to(
                local_spectrum.value[None, :],
                exozodi_radiance.shape,
            )
            * local_spectrum.unit
        )

        # ---------------------------------------------------------
        # 5. Collector geometry from LIFEObservatory
        # ---------------------------------------------------------
        collector_positions = Obs.get_collector_positions(
            rotation_angles,
            nulling_baseline=nulling_baseline,
        )

        # ---------------------------------------------------------
        # 6. Instrument photon rates
        # ---------------------------------------------------------
        local_left, local_right = OS.get_extended_background_rates(
            collector_positions=collector_positions,
            wavelengths=wavelengths,
            bin_widths=bin_widths,
            sky_positions=sky_positions,
            spectral_radiance=local_radiance,
            pixel_solid_angles=pixel_solid_angles,
            mode=mode,
        )

        exozodi_left, exozodi_right = OS.get_extended_background_rates(
            collector_positions=collector_positions,
            wavelengths=wavelengths,
            bin_widths=bin_widths,
            sky_positions=sky_positions,
            spectral_radiance=exozodi_radiance,
            pixel_solid_angles=pixel_solid_angles,
            mode=mode,
        )

        return {
            "local_left": local_left,
            "local_right": local_right,
            "exozodi_left": exozodi_left,
            "exozodi_right": exozodi_right,
        }
