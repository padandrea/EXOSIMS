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
            The local foreground uses Mennesson by default, or Darwinsim.
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

        if mode.get("LIFE_background_method", "grid") == "fourier":
            if (
                mode.get("LIFE_fov_taper", "none") != "none"
                or mode.get("LIFE_fov_boundary", "square") != "circle"
            ):
                raise ValueError(
                    "Fourier background mode requires an untapered "
                    "circular field of view."
                )

            n_radial = int(mode.get("LIFE_radial_samples", 8193))
            if n_radial < 3:
                raise ValueError("LIFE_radial_samples must be >= 3.")

            diameter = OS.pupilDiam.to_value(u.m)
            max_radius = np.max(wavelengths.to_value(u.m)) / (2.0 * diameter)

            radii = np.linspace(0.0, max_radius, n_radial)
            radial_positions = np.column_stack((radii, np.zeros_like(radii))) * u.rad

            local_spectrum = ZL.get_local_zodi_spectral_radiance(
                wavelengths=wavelengths,
                longitude=local_zodi_longitude,
                latitude=local_zodi_latitude,
                model=mode.get("LIFE_local_zodi_model", "mennesson"),
            )

            local_radial = (
                np.broadcast_to(
                    local_spectrum.value[None, :],
                    (n_radial, len(wavelengths)),
                )
                * local_spectrum.unit
            )

            exozodi_radial = ZL.get_exozodi_spectral_radiance(
                sky_positions=radial_positions,
                wavelengths=wavelengths,
                nEZ=nEZ,
                star_distance=star_distance,
                star_luminosity=star_luminosity,
            )

            positions = Obs.get_collector_positions(
                rotation_angles,
                nulling_baseline=nulling_baseline,
            )

            # Axisymmetric background: collector-pair lengths do
            # not change under rotation. Evaluate only once.
            reference_positions = positions[0]

            local_left, local_right = OS.get_axisymmetric_background_rates(
                reference_positions,
                wavelengths,
                bin_widths,
                radii * u.rad,
                local_radial,
                mode,
            )

            exozodi_left, exozodi_right = OS.get_axisymmetric_background_rates(
                reference_positions,
                wavelengths,
                bin_widths,
                radii * u.rad,
                exozodi_radial,
                mode,
            )

            n_rotations = positions.shape[0]

            def repeat_rotations(rate):
                return (
                    np.broadcast_to(
                        rate.value,
                        (n_rotations, rate.shape[1]),
                    )
                    * rate.unit
                )

            return {
                "local_left": repeat_rotations(local_left),
                "local_right": repeat_rotations(local_right),
                "exozodi_left": repeat_rotations(exozodi_left),
                "exozodi_right": repeat_rotations(exozodi_right),
            }

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
        local_spectrum = ZL.get_local_zodi_spectral_radiance(
            wavelengths=wavelengths,
            longitude=local_zodi_longitude,
            latitude=local_zodi_latitude,
            model=mode.get("LIFE_local_zodi_model", "mennesson"),
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

    def get_life_detection_metrics(
        self,
        *,
        sInd: int,
        wavelengths: u.Quantity,
        bin_widths: u.Quantity,
        planet_sky_positions: u.Quantity,
        planet_spectral_fluxes: u.Quantity,
        stellar_angular_radius: u.Quantity,
        stellar_spectral_flux: u.Quantity,
        sky_positions: u.Quantity,
        pixel_solid_angles: u.Quantity,
        rotation_angles: u.Quantity,
        mode: dict,
        local_zodi_longitude: u.Quantity,
        local_zodi_latitude: u.Quantity,
        integration_time: u.Quantity,
        detector_pixels_per_output: float | np.ndarray,
        nEZ: float | None = None,
        nulling_baseline: u.Quantity | None = None,
        rotation_fractions: np.ndarray | None = None,
        snr_threshold: float | None = None,
    ) -> dict:
        """Assemble LIFE detection SNR and integration times for one star.

        Planet spectra have shape (n_planets, n_wavelengths) and units
        photon/(s m2 um); planet sky offsets have shape (n_planets, 2).
        All spectra are flux densities at the observer, not surface fluxes.
        The supplied total integration time is distributed over rotations.
        Other planets contribute photon noise to each candidate detection.
        This method does not generate spectra or propagate orbits.
        """
        if not isinstance(planet_sky_positions, u.Quantity):
            raise TypeError("planet_sky_positions must be an Astropy Quantity.")
        if not isinstance(planet_spectral_fluxes, u.Quantity):
            raise TypeError("planet_spectral_fluxes must be an Astropy Quantity.")

        planet_positions = np.atleast_2d(planet_sky_positions.to(u.rad))
        planet_fluxes = np.atleast_2d(
            planet_spectral_fluxes.to(u.photon / (u.s * u.m**2 * u.um))
        )
        n_wavelengths = np.atleast_1d(wavelengths).size
        n_planets = planet_positions.shape[0]

        if (
            planet_positions.shape != (n_planets, 2)
            or n_planets == 0
            or planet_fluxes.shape != (n_planets, n_wavelengths)
        ):
            raise ValueError(
                "Expected planet_sky_positions (n_planets, 2) and "
                "planet_spectral_fluxes (n_planets, n_wavelengths), "
                "with at least one planet."
            )

        optical_system = self.OpticalSystem
        collector_positions = self.Observatory.get_collector_positions(
            rotation_angles,
            nulling_baseline=nulling_baseline,
        )

        # Each planet can have a different thermal spectrum.
        planet_left_list = []
        planet_right_list = []
        for planet_index in range(n_planets):
            left, right = optical_system.get_planet_photon_rates(
                collector_positions,
                wavelengths,
                bin_widths,
                planet_positions[planet_index],
                planet_fluxes[planet_index],
                mode,
            )
            planet_left_list.append(left)
            planet_right_list.append(right)

        # (n_rotations, n_planets, n_wavelengths)
        planet_left = np.concatenate(planet_left_list, axis=1)
        planet_right = np.concatenate(planet_right_list, axis=1)

        stellar_left, stellar_right = optical_system.get_stellar_leakage_rates(
            collector_positions,
            wavelengths,
            bin_widths,
            stellar_angular_radius,
            stellar_spectral_flux,
            mode,
        )

        zodiacal_rates = self.get_life_zodiacal_rates(
            sInd=sInd,
            wavelengths=wavelengths,
            bin_widths=bin_widths,
            sky_positions=sky_positions,
            pixel_solid_angles=pixel_solid_angles,
            rotation_angles=rotation_angles,
            mode=mode,
            local_zodi_longitude=local_zodi_longitude,
            local_zodi_latitude=local_zodi_latitude,
            nEZ=nEZ,
            nulling_baseline=nulling_baseline,
        )

        dark_rate, read_variance_rate = optical_system.get_detector_noise_rates(
            wavelengths,
            detector_pixels_per_output,
            mode,
        )

        # Other planets also contribute photon shot noise. Assemble one
        # detection candidate at a time so its companions enter only the
        # variance, not the candidate's signal template.
        signal_list = []
        variance_list = []
        for planet_index in range(n_planets):
            background_rates = [
                (zodiacal_rates["local_left"], zodiacal_rates["local_right"]),
                (zodiacal_rates["exozodi_left"], zodiacal_rates["exozodi_right"]),
            ]
            if n_planets > 1:
                other_planets = np.arange(n_planets) != planet_index
                background_rates.append(
                    (
                        planet_left[:, other_planets, :].sum(axis=1),
                        planet_right[:, other_planets, :].sum(axis=1),
                    )
                )

            signal, variance = optical_system.assemble_life_signal_noise(
                planet_left=planet_left[:, planet_index : planet_index + 1, :],
                planet_right=planet_right[:, planet_index : planet_index + 1, :],
                stellar_left=stellar_left,
                stellar_right=stellar_right,
                background_rates=background_rates,
                detector_dark_rate=dark_rate,
                detector_read_variance_rate=read_variance_rate,
                rotation_fractions=rotation_fractions,
            )
            signal_list.append(signal)
            variance_list.append(variance)

        # One row per planet; columns are rotation-wavelength channels.
        signal_rates = np.concatenate(signal_list, axis=0)
        noise_variance_rates = np.concatenate(variance_list, axis=0)

        threshold = mode["SNR"] if snr_threshold is None else snr_threshold
        snr = optical_system.calc_life_snr(
            signal_rates,
            noise_variance_rates,
            integration_time,
        )
        required_time = optical_system.calc_life_int_time(
            signal_rates,
            noise_variance_rates,
            snr_threshold=threshold,
        )

        return {
            "planet_left": planet_left,
            "planet_right": planet_right,
            "stellar_left": stellar_left,
            "stellar_right": stellar_right,
            **zodiacal_rates,
            "detector_dark_rate": dark_rate,
            "detector_read_variance_rate": read_variance_rate,
            "signal_rates": signal_rates,
            "noise_variance_rates": noise_variance_rates,
            "snr": snr,
            "required_integration_time": required_time,
        }
