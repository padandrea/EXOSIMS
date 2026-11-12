"""Optical system model for the LIFE nulling interferometer."""

import astropy.units as u
import numpy as np
from scipy.optimize import brentq
from scipy.special import j1
from synphot import Observation, SpectralElement
from synphot.models import Box1D, Gaussian1D

from EXOSIMS.Prototypes.OpticalSystem import OpticalSystem


class LIFEOpticalSystem(OpticalSystem):
    """LIFE mid-infrared nulling interferometer optical system.

    Args:
        **specs (dict):
            EXOSIMS OpticalSystem configuration parameters.
    """

    @property
    def total_collecting_area(self) -> u.Quantity:
        """Return the combined collecting area of four LIFE collectors."""
        return 4.0 * self.pupilArea

    # ================================================================
    # Stage 0 — Initialization
    # ================================================================
    # Configure the LIFE instrument and initialize EXOSIMS interfaces.

    def populate_observingModes(self, observingModes):
        """Helper method to parse input observingMode dictionaries and assign default
        values, as needed. Also creates the allowed_observingMode_kws attribute.

        Args:
            observingModes (list):
                List of observingMode dicts.

        """

        self.observingModes = observingModes
        self._outspec["observingModes"] = []
        for nmode, mode in enumerate(self.observingModes):
            assert isinstance(mode, dict), "Observing modes must be defined as dicts."
            assert (
                "instName" in mode and "systName" in mode
            ), "All observing modes must have keys 'instName' and 'systName'."
            assert np.any(
                [mode["instName"] == inst["name"] for inst in self.scienceInstruments]
            ), f"The mode's instrument name {mode['instName']} does not exist."
            assert np.any(
                [
                    mode["systName"] == syst["name"]
                    for syst in self.starlightSuppressionSystems
                ]
            ), f"The mode's system name {mode['systName']} does not exist."
            self._outspec["observingModes"].append(mode.copy())

            # loading mode specifications
            mode["SNR"] = float(mode.get("SNR", self.default_vals["SNR"]))
            mode["timeMultiplier"] = float(
                mode.get("timeMultiplier", self.default_vals["timeMultiplier"])
            )
            mode["detectionMode"] = mode.get("detectionMode", False)
            mode["inst"] = [
                inst
                for inst in self.scienceInstruments
                if inst["name"] == mode["instName"]
            ][0]
            mode["syst"] = [
                syst
                for syst in self.starlightSuppressionSystems
                if syst["name"] == mode["systName"]
            ][0]

            # start tracking allowed keywords

            self.allowed_observingMode_kws = [
                "instName",
                "systName",
                "SNR",
                "timeMultiplier",
                "detectionMode",
                "lam",
                "deltaLam",
                "BW",
                "LIFE_lambda_min",
                "LIFE_lambda_max",
                "bandpass_model",
                "bandpass_step",
            ]

            # Configure wavelength range.
            # LIFE-specific limits take precedence over standard EXOSIMS
            # central wavelength and bandwidth parameters.
            has_life_min = "LIFE_lambda_min" in mode
            has_life_max = "LIFE_lambda_max" in mode

            if has_life_min != has_life_max:
                raise ValueError(
                    "LIFE_lambda_min and LIFE_lambda_max " "must be specified together."
                )

            if has_life_min:
                # JSON wavelength limits are specified in nanometres.
                try:
                    lam_min = float(mode["LIFE_lambda_min"])
                    lam_max = float(mode["LIFE_lambda_max"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "LIFE wavelength limits must be numeric "
                        "values in nanometres."
                    ) from exc

                if (
                    not np.isfinite(lam_min)
                    or not np.isfinite(lam_max)
                    or lam_min <= 0
                    or lam_max <= lam_min
                ):
                    raise ValueError(
                        "LIFE wavelength limits must be finite, "
                        "positive, and satisfy min < max."
                    )

                # Derive standard EXOSIMS parameters.
                mode["lam"] = 0.5 * (lam_min + lam_max) * u.nm
                mode["deltaLam"] = (lam_max - lam_min) * u.nm

            else:
                # Original EXOSIMS behavior.
                syst_lam = mode["syst"]["lam"].to_value(u.nm)
                syst_BW = mode["syst"]["BW"]

                mode["lam"] = float(mode.get("lam", syst_lam)) * u.nm

                mode["deltaLam"] = (
                    float(
                        mode.get(
                            "deltaLam",
                            mode["lam"].value * mode.get("BW", syst_BW),
                        )
                    )
                    * u.nm
                )

            mode["BW"] = float(mode["deltaLam"] / mode["lam"])

            # get mode IWA and OWA: rescale if the mode wavelength is different from
            # the wavelength at which the system is defined
            mode["IWA"] = mode["syst"]["IWA"]
            mode["OWA"] = mode["syst"]["OWA"]
            if mode["lam"] != mode["syst"]["lam"]:
                mode["IWA"] = mode["IWA"] * mode["lam"] / mode["syst"]["lam"]
                mode["OWA"] = mode["OWA"] * mode["lam"] / mode["syst"]["lam"]

            # OWA must be bounded by FOV:
            if mode["OWA"] > mode["inst"]["FoV"]:
                mode["OWA"] = mode["inst"]["FoV"]

            # generate the mode's bandpass
            # TODO: Add support for custom filter profiles
            mode["bandpass_model"] = mode.get(
                "bandpass_model", self.default_vals["bandpass_model"]
            ).lower()
            assert mode["bandpass_model"] in [
                "gaussian",
                "box",
            ], "bandpass_model must be one of ['gaussian', 'box']"
            mode["bandpass_step"] = (
                float(mode.get("bandpass_step", self.default_vals["bandpass_step"]))
                * u.nm
            )
            if mode["bandpass_model"] == "box":
                mode["bandpass"] = SpectralElement(
                    Box1D,
                    x_0=mode["lam"],
                    width=mode["deltaLam"],
                    step=mode["bandpass_step"].to(u.AA).value,
                )
            else:
                mode["bandpass"] = SpectralElement(
                    Gaussian1D,
                    mean=mode["lam"],
                    stddev=mode["deltaLam"] / np.sqrt(2 * np.pi),
                )

            # evaluate zero-magnitude flux for this band from vega spectrum
            # NB: This is flux, not flux density! The bandpass is already factored in.
            mode["F0"] = Observation(
                self.vega_spectrum, mode["bandpass"], force="taper"
            ).integrate()

            # Evaluate the V-band zero magnitude flux density

            # populate system specifications to outspec
            for att in mode:
                if att not in [
                    "inst",
                    "syst",
                    "F0",
                    "bandpass",
                ]:
                    dat = mode[att]
                    self._outspec["observingModes"][nmode][att] = (
                        dat.value if isinstance(dat, u.Quantity) else dat
                    )

            # populate some final mode attributes (computed from the others)
            # define total mode attenution
            mode["attenuation"] = mode["inst"]["optics"] * mode["syst"]["optics"]

            # effective mode bandwidth (including any IFS spectral resolving power)
            mode["deltaLam_eff"] = (
                mode["lam"] / mode["inst"]["Rs"]
                if "spec" in mode["inst"]["name"].lower()
                else mode["deltaLam"]
            )

            # total attenuation due to non-coronagraphic optics:
            mode["losses"] = (
                self.pupilArea
                * mode["inst"]["QE"](mode["lam"])
                * mode["attenuation"]
                * mode["deltaLam_eff"]
                / mode["deltaLam"]
            )

        # check for only one detection mode
        allModes = self.observingModes
        detModes = list(filter(lambda mode: mode["detectionMode"], allModes))
        assert len(detModes) <= 1, "More than one detection mode specified."

        # if not specified, default detection mode is first imager mode
        if len(detModes) == 0:
            imagerModes = list(
                filter(lambda mode: "imag" in mode["inst"]["name"], allModes)
            )
            if imagerModes:
                imagerModes[0]["detectionMode"] = True
            # if no imager mode, default detection mode is first observing mode
            else:
                allModes[0]["detectionMode"] = True

        self.populate_observingModes_extra()

    # ================================================================
    # Stage 1 —  Spectral configuration
    # ================================================================

    # ----------------------------------------------------------------
    # Spectral configuration
    # ----------------------------------------------------------------

    def get_spectral_bins(
        self,
        wavelength_min: u.Quantity,
        wavelength_max: u.Quantity,
        resolving_power: float,
    ) -> tuple[u.Quantity, u.Quantity]:
        """Calculate LIFE wavelength bins at constant nominal resolution.

        Wavelength bins are geometrically spaced to maintain constant
        nominal resolving power R = lambda / delta_lambda.
        The final bin is truncated at wavelength_max.

        Args:
            wavelength_min (astropy.units.Quantity):
                Minimum wavelength.
            wavelength_max (astropy.units.Quantity):
                Maximum wavelength.
            resolving_power (float):
                Nominal spectral resolving power R = lambda / delta_lambda.

        Returns:
            tuple[astropy.units.Quantity, astropy.units.Quantity]:
                Wavelength bin centers and widths, both in metres.

        Raises:
            TypeError:
                If wavelength inputs are not Astropy Quantities.
            ValueError:
                If wavelengths or resolving power are invalid.
        """
        if not isinstance(wavelength_min, u.Quantity):
            raise TypeError("wavelength_min must be an Astropy Quantity.")

        if not isinstance(wavelength_max, u.Quantity):
            raise TypeError("wavelength_max must be an Astropy Quantity.")

        try:
            lam_min = wavelength_min.to_value(u.m)
            lam_max = wavelength_max.to_value(u.m)
        except u.UnitConversionError as exc:
            raise ValueError("Wavelengths must have length units.") from exc

        if (
            np.ndim(lam_min) != 0
            or np.ndim(lam_max) != 0
            or not np.isfinite(lam_min)
            or not np.isfinite(lam_max)
            or lam_min <= 0
            or lam_max <= lam_min
        ):
            raise ValueError("Wavelength limits must be finite and ordered.")

        resolving_power = float(resolving_power)

        if not np.isfinite(resolving_power) or resolving_power <= 0.5:
            raise ValueError("resolving_power must be finite and greater than 0.5.")

        # Exact resolving power at the arithmetic bin center:
        #
        # R = lambda_center / delta_lambda
        #
        # For a bin beginning at lambda_left:
        # delta_lambda = lambda_left / (R - 0.5)
        #
        # Therefore, consecutive bin edges have this ratio:
        relative_width = 1.0 / (resolving_power - 0.5)
        step = 1.0 + relative_width

        n_bins = int(np.ceil(np.log(lam_max / lam_min) / np.log1p(relative_width)))

        edges = lam_min * step ** np.arange(n_bins + 1)
        edges[-1] = lam_max

        centers = (edges[:-1] + edges[1:]) / 2
        widths = edges[1:] - edges[:-1]

        return centers * u.m, widths * u.m

    def get_spectral_bins_from_mode(self, mode: dict) -> tuple[u.Quantity, u.Quantity]:
        """Calculate wavelength bins from an EXOSIMS observing mode."""
        if "spec" not in mode["inst"]["name"].lower():
            raise ValueError("Spectral binning requires a spectrometer.")

        half_width = mode["deltaLam"] / 2

        return self.get_spectral_bins(
            mode["lam"] - half_width,
            mode["lam"] + half_width,
            mode["inst"]["Rs"],
        )

    # ================================================================
    # Stage 2 — Interferometry
    # ================================================================

    def get_transmission(
        self,
        collector_positions: u.Quantity,
        wavelengths: u.Quantity,
        sky_positions: u.Quantity,
    ) -> tuple[u.Quantity, u.Quantity, u.Quantity]:
        """Calculate point-source transmission of the LIFE double Bracewell.

        The two chopped outputs use the quadrature phase shifts of the
        double Bracewell beam combiner.
        Transmission is normalized to the total collecting area of the
        four telescopes, without throughput or photon-rate factors.

        Args:
            collector_positions (astropy.units.Quantity):
                Collector coordinates with shape (4, 2) or
                (n_configurations, 4, 2), in length units.
                Collector ordering must match the beam combiner.
                Coordinates use the same sky axes as sky_positions.
            wavelengths (astropy.units.Quantity):
                Scalar or 1D array of wavelengths with length units.
            sky_positions (astropy.units.Quantity):
                Angular offsets (theta_x, theta_y), shape (2,) or (n_sky, 2).

        Returns:
            tuple[astropy.units.Quantity, astropy.units.Quantity, astropy.units.Quantity]:
                Left, right, and left-minus-right transmission, dimensionless.
                Each has shape (n_configurations, n_sky, n_wavelengths).

        Raises:
            TypeError:
                If an input is not an Astropy Quantity.
            ValueError:
                If input units, shapes, or values are invalid.
        """
        if not isinstance(wavelengths, u.Quantity):
            raise TypeError("wavelengths must be an Astropy Quantity.")
        if not isinstance(sky_positions, u.Quantity):
            raise TypeError("sky_positions must be an Astropy Quantity.")

        try:
            wavelength_m = np.atleast_1d(wavelengths.to_value(u.m))
        except u.UnitConversionError as exc:
            raise ValueError("wavelengths must have length units.") from exc

        if (
            wavelength_m.ndim != 1
            or wavelength_m.size == 0
            or not np.all(np.isfinite(wavelength_m))
            or np.any(wavelength_m <= 0)
        ):
            raise ValueError("wavelengths must be finite, positive, and 1D.")

        try:
            theta = np.asarray(sky_positions.to_value(u.rad))
        except u.UnitConversionError as exc:
            raise ValueError("sky_positions must have angular units.") from exc

        if theta.ndim == 1:
            theta = theta[np.newaxis, :]

        if (
            theta.ndim != 2
            or theta.shape[1] != 2
            or theta.shape[0] == 0
            or not np.all(np.isfinite(theta))
        ):
            raise ValueError("sky_positions must have shape (2,) or (n_sky, 2).")

        if not isinstance(collector_positions, u.Quantity):
            raise TypeError("collector_positions must be an Astropy Quantity.")

        try:
            positions_m = np.asarray(collector_positions.to_value(u.m))
        except u.UnitConversionError as exc:
            raise ValueError("collector_positions must have length units.") from exc

        if positions_m.ndim == 2:
            positions_m = positions_m[np.newaxis, :, :]

        if (
            positions_m.ndim != 3
            or positions_m.shape[0] == 0
            or positions_m.shape[1:] != (4, 2)
            or not np.all(np.isfinite(positions_m))
        ):
            raise ValueError(
                "collector_positions must have shape (4, 2) "
                "or (n_configurations, 4, 2), with finite values."
            )

        # Remove the common optical phase using collector 0 as reference.
        relative_positions = positions_m - positions_m[:, :1, :]
        delays_m = np.einsum("rjc,sc->rsj", relative_positions, theta)
        geometric_phase = (
            2.0 * np.pi * delays_m[:, :, None, :] / wavelength_m[None, None, :, None]
        )

        phase_left = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
        phase_right = np.array([0.0, 3 * np.pi / 2, np.pi, np.pi / 2])

        field_left = np.exp(1j * (geometric_phase + phase_left)).sum(axis=-1)
        field_right = np.exp(1j * (geometric_phase + phase_right)).sum(axis=-1)

        # Division by 4**2 gives transmission relative to four collectors.
        left = np.abs(field_left) ** 2 / 16.0
        right = np.abs(field_right) ** 2 / 16.0

        unit = u.dimensionless_unscaled
        return left * unit, right * unit, (left - right) * unit

    # ================================================================
    # Stage 3 — Photon rates
    # ================================================================
    def get_planet_photon_rates(
        self,
        collector_positions: u.Quantity,
        wavelengths: u.Quantity,
        bin_widths: u.Quantity,
        sky_positions: u.Quantity,
        planet_spectral_flux: u.Quantity,
        mode: dict,
    ) -> tuple[u.Quantity, u.Quantity]:
        """Calculate planet count rates in the two chopped outputs.

        Args:
            collector_positions (astropy.units.Quantity):
                Collector coordinates in metres.
            wavelengths (astropy.units.Quantity):
                Wavelength-bin centers.
            bin_widths (astropy.units.Quantity):
                Width of each spectral bin.
            sky_positions (astropy.units.Quantity):
                Planet angular offsets from the star.
            planet_spectral_flux (astropy.units.Quantity):
                Planet photon flux density at the observer, with one
                value per wavelength bin, in ph / (s m2 um).
            mode (dict):
                EXOSIMS observing mode.

        Returns:
            tuple[astropy.units.Quantity, astropy.units.Quantity]:
                Left and right planet count rates in 1/s.
                Shape: (n_configurations, n_sky, n_wavelengths).
        """
        left, right, _ = self.get_transmission(
            collector_positions, wavelengths, sky_positions
        )

        flux = np.atleast_1d(planet_spectral_flux.to(u.photon / (u.s * u.m**2 * u.um)))
        widths = np.atleast_1d(bin_widths.to(u.um))
        n_wavelengths = left.shape[-1]

        if flux.shape != (n_wavelengths,):
            raise ValueError("planet_spectral_flux must have one value per wavelength.")

        if widths.shape != (n_wavelengths,):
            raise ValueError("bin_widths must have one value per wavelength.")

        if (
            not np.all(np.isfinite(flux.value))
            or np.any(flux.value < 0)
            or not np.all(np.isfinite(widths.value))
            or np.any(widths.value <= 0)
        ):
            raise ValueError("Fluxes and bin widths must be physically valid.")

        qe = mode["inst"]["QE"](np.atleast_1d(wavelengths))
        throughput = mode["attenuation"]

        incident_rate = (
            flux * widths * self.total_collecting_area * throughput * qe
        ).to(1 / u.s)

        return (
            (left * incident_rate).to(1 / u.s),
            (right * incident_rate).to(1 / u.s),
        )

    def get_stellar_leakage_rates(
        self,
        collector_positions: u.Quantity,
        wavelengths: u.Quantity,
        bin_widths: u.Quantity,
        stellar_angular_radius: u.Quantity,
        stellar_spectral_flux: u.Quantity,
        mode: dict,
    ) -> tuple[u.Quantity, u.Quantity]:
        """Calculate stellar leakage rates for a uniform stellar disk.

        The star is centered on the optical axis and has uniform surface
        brightness. Disk-averaged transmission is evaluated analytically
        using the uniform-disk visibility function.

        Args:
            collector_positions (astropy.units.Quantity):
                Collector coordinates, shape (4, 2) or
                (n_configurations, 4, 2).
            wavelengths (astropy.units.Quantity):
                Wavelength-bin centers.
            bin_widths (astropy.units.Quantity):
                Widths of the spectral bins.
            stellar_angular_radius (astropy.units.Quantity):
                Apparent stellar angular radius.
            stellar_spectral_flux (astropy.units.Quantity):
                Total stellar photon flux density at the observer,
                in ph / (s m2 um), one value per wavelength bin.
            mode (dict):
                EXOSIMS observing mode.

        Returns:
            tuple[astropy.units.Quantity, astropy.units.Quantity]:
                Leakage count rates in the left and right outputs,
                in 1/s, shape (n_configurations, n_wavelengths).
        """
        quantities = (
            collector_positions,
            wavelengths,
            bin_widths,
            stellar_angular_radius,
            stellar_spectral_flux,
        )
        if not all(isinstance(q, u.Quantity) for q in quantities):
            raise TypeError("All physical inputs must be Astropy Quantities.")

        positions = np.asarray(collector_positions.to_value(u.m))
        if positions.ndim == 2:
            positions = positions[np.newaxis, :, :]

        if (
            positions.ndim != 3
            or positions.shape[0] == 0
            or positions.shape[1:] != (4, 2)
            or not np.all(np.isfinite(positions))
        ):
            raise ValueError("Invalid collector_positions shape or values.")

        lam = np.atleast_1d(wavelengths.to_value(u.m))
        widths = np.atleast_1d(bin_widths.to(u.um))
        flux = np.atleast_1d(stellar_spectral_flux.to(u.photon / (u.s * u.m**2 * u.um)))
        radius = np.asarray(stellar_angular_radius.to_value(u.rad))

        if (
            lam.ndim != 1
            or lam.size == 0
            or not np.all(np.isfinite(lam))
            or np.any(lam <= 0)
            or widths.shape != lam.shape
            or flux.shape != lam.shape
            or not np.all(np.isfinite(widths.value))
            or not np.all(np.isfinite(flux.value))
            or np.any(widths.value <= 0)
            or np.any(flux.value < 0)
        ):
            raise ValueError("Invalid spectral inputs.")

        if radius.ndim != 0 or not np.isfinite(radius) or radius < 0:
            raise ValueError("stellar_angular_radius must be nonnegative.")

        # Pairwise projected collector separations.
        baselines = positions[:, :, None, :] - positions[:, None, :, :]
        baseline_lengths = np.linalg.norm(baselines, axis=-1)

        # Uniform-disk coherence: 2*J1(x)/x, with limit 1 at x=0.
        x = (
            2.0
            * np.pi
            * radius
            * baseline_lengths[:, None, :, :]
            / lam[None, :, None, None]
        )

        coherence = np.ones_like(x)
        np.divide(
            2.0 * j1(x),
            x,
            out=coherence,
            where=x != 0,
        )

        # Same beam-combiner phases as get_transmission().
        phase_left = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
        phase_right = np.array([0.0, 3 * np.pi / 2, np.pi, np.pi / 2])

        weights_left = np.cos(phase_left[:, None] - phase_left[None, :])
        weights_right = np.cos(phase_right[:, None] - phase_right[None, :])

        transmission_left = np.einsum("ij,cwij->cw", weights_left, coherence) / 16.0
        transmission_right = np.einsum("ij,cwij->cw", weights_right, coherence) / 16.0

        # Remove possible negative values from floating-point roundoff.
        transmission_left = np.maximum(transmission_left, 0.0)
        transmission_right = np.maximum(transmission_right, 0.0)

        qe = mode["inst"]["QE"](np.atleast_1d(wavelengths))

        incident_rate = (
            flux * widths * self.total_collecting_area * mode["attenuation"] * qe
        ).to(1 / u.s)

        return (
            (transmission_left * incident_rate).to(1 / u.s),
            (transmission_right * incident_rate).to(1 / u.s),
        )

    def get_extended_background_rates(
        self,
        collector_positions: u.Quantity,
        wavelengths: u.Quantity,
        bin_widths: u.Quantity,
        sky_positions: u.Quantity,
        spectral_radiance: u.Quantity,
        pixel_solid_angles: u.Quantity,
        mode: dict,
    ) -> tuple[u.Quantity, u.Quantity]:
        """Calculate photon rates from a spatially extended background.

        Args:
            collector_positions (astropy.units.Quantity):
                Collector coordinates in length units.
            wavelengths (astropy.units.Quantity):
                Wavelength-bin centers.
            bin_widths (astropy.units.Quantity):
                Width of each spectral bin.
            sky_positions (astropy.units.Quantity):
                Angular coordinates of sky pixels, shape (n_sky, 2).
            spectral_radiance (astropy.units.Quantity):
                Background photon radiance, shape (n_sky, n_wavelengths),
                in ph / (s m2 um sr).
            pixel_solid_angles (astropy.units.Quantity):
                Solid angle of each sky pixel, shape (n_sky,).
            mode (dict):
                EXOSIMS observing mode.

        Returns:
            tuple[astropy.units.Quantity, astropy.units.Quantity]:
                Left and right background count rates in 1/s,
                shape (n_configurations, n_wavelengths).
        """
        left, right, _ = self.get_transmission(
            collector_positions, wavelengths, sky_positions
        )

        n_sky = left.shape[1]
        n_wavelengths = left.shape[2]

        radiance = spectral_radiance.to(u.photon / (u.s * u.m**2 * u.um * u.sr))
        solid_angles = np.atleast_1d(pixel_solid_angles.to(u.sr))
        widths = np.atleast_1d(bin_widths.to(u.um))

        if radiance.shape != (n_sky, n_wavelengths):
            raise ValueError(
                "spectral_radiance must have shape " "(n_sky, n_wavelengths)."
            )

        if solid_angles.shape != (n_sky,):
            raise ValueError("pixel_solid_angles must have shape (n_sky,).")

        if widths.shape != (n_wavelengths,):
            raise ValueError("bin_widths must have one value per wavelength.")

        if (
            not np.all(np.isfinite(radiance.value))
            or np.any(radiance.value < 0)
            or not np.all(np.isfinite(solid_angles.value))
            or np.any(solid_angles.value <= 0)
            or not np.all(np.isfinite(widths.value))
            or np.any(widths.value <= 0)
        ):
            raise ValueError(
                "Background radiance, pixel areas, and bin widths "
                "must be finite and physically valid."
            )

        qe = mode["inst"]["QE"](np.atleast_1d(wavelengths))

        # Photon rate contributed by each sky pixel before
        # applying the interferometric transmission.
        patch_rates = (
            radiance
            * solid_angles[:, None]
            * widths[None, :]
            * self.total_collecting_area
            * mode["attenuation"]
            * qe
        ).to(1 / u.s)

        # Sum incoherent emission from all sky pixels.
        rate_left = (left * patch_rates[None, :, :]).sum(axis=1)

        rate_right = (right * patch_rates[None, :, :]).sum(axis=1)

        return rate_left.to(1 / u.s), rate_right.to(1 / u.s)

    def get_detector_noise_rates(
        self,
        wavelengths: u.Quantity,
        n_pixels_per_output: float | np.ndarray,
        mode: dict,
    ) -> tuple[u.Quantity, u.Quantity]:
        """Calculate detector noise contributions per spectral channel.

        Uses the EXOSIMS convention that sread represents the
        effective read-noise variance per pixel per exposure.

        Args:
            wavelengths (astropy.units.Quantity):
                Spectral-bin centers, used to determine output shape.
            n_pixels_per_output (float or numpy.ndarray):
                Effective number of detector pixels contributing
                to each spectral channel of one chopped output.
                May be a scalar or one value per wavelength.
            mode (dict):
                EXOSIMS observing mode.

        Returns:
            tuple[astropy.units.Quantity, astropy.units.Quantity]:
                Dark-current count rate and read-noise-equivalent
                variance rate per spectral channel, both expressed
                in EXOSIMS count-rate-equivalent units (1/s).
                Each has shape (n_wavelengths,).

        Notes:
            The returned quantities are for one chopped output.
            Both outputs are assumed to have identical detector
            properties and independent noise.

            Read noise is not a photon count rate. Its equivalent
            rate is used to compute accumulated noise variance.
        """
        if not isinstance(wavelengths, u.Quantity):
            raise TypeError("wavelengths must be an Astropy Quantity.")

        lam = np.atleast_1d(wavelengths.to_value(u.um))

        if (
            lam.ndim != 1
            or lam.size == 0
            or not np.all(np.isfinite(lam))
            or np.any(lam <= 0)
        ):
            raise ValueError("wavelengths must be finite and positive.")

        pixels = np.asarray(n_pixels_per_output, dtype=float)

        if pixels.ndim == 0:
            pixels = np.full(lam.shape, pixels.item())

        if pixels.shape != lam.shape:
            raise ValueError(
                "n_pixels_per_output must be a scalar or have "
                "one value per wavelength."
            )

        if not np.all(np.isfinite(pixels)) or np.any(pixels <= 0):
            raise ValueError("n_pixels_per_output must be finite and positive.")

        inst = mode["inst"]

        dark_current = inst["idark"].to(1 / u.s)
        exposure_time = inst["texp"].to(u.s)
        read_variance = float(inst["sread"])

        if (
            np.ndim(dark_current.value) != 0
            or not np.isfinite(dark_current.value)
            or dark_current.value < 0
        ):
            raise ValueError("idark must be finite and nonnegative.")

        if (
            np.ndim(exposure_time.value) != 0
            or not np.isfinite(exposure_time.value)
            or exposure_time.value <= 0
        ):
            raise ValueError("texp must be finite and positive.")

        if not np.isfinite(read_variance) or read_variance < 0:
            raise ValueError("sread must be finite and nonnegative.")

        dark_rate = pixels * dark_current
        read_rate = pixels * read_variance / exposure_time

        return dark_rate.to(1 / u.s), read_rate.to(1 / u.s)

    # ================================================================
    # Stage 4 — Detection
    # ================================================================

    @staticmethod
    def assemble_life_signal_noise(
        planet_left: u.Quantity,
        planet_right: u.Quantity,
        stellar_left: u.Quantity,
        stellar_right: u.Quantity,
        background_rates: list[tuple[u.Quantity, u.Quantity]] | None = None,
        detector_dark_rate: u.Quantity | None = None,
        detector_read_variance_rate: u.Quantity | None = None,
        rotation_fractions: np.ndarray | None = None,
    ) -> tuple[u.Quantity, u.Quantity]:
        """Assemble phase-chopped signal and noise-variance rates.

        Args:
            planet_left (astropy.units.Quantity):
                Planet rates with shape (n_rotations, n_planets, n_wavelengths).
            planet_right (astropy.units.Quantity):
                Planet rates in the right output, same shape.
            stellar_left (astropy.units.Quantity):
                Stellar leakage with shape (n_rotations, n_wavelengths).
            stellar_right (astropy.units.Quantity):
                Stellar leakage in the right output.
            background_rates (list, optional):
                Left/right pairs of extended-background rates.
                Each component has shape (n_rotations, n_wavelengths).
            detector_dark_rate (astropy.units.Quantity, optional):
                Dark-current rate per output, shape (n_wavelengths,).
            detector_read_variance_rate (astropy.units.Quantity, optional):
                Read-noise-equivalent variance rate per output.
            rotation_fractions (numpy.ndarray, optional):
                Fraction of total integration time per rotation.
                Defaults to equal exposure fractions.

        Returns:
            tuple[astropy.units.Quantity, astropy.units.Quantity]:
                Signal and variance rates in 1/s.
                Shape: (n_planets, n_rotations * n_wavelengths).

        Notes:
            Planet signal is the left-minus-right difference.
            Independent photon-noise variances from both outputs add.

            Rotation fractions weight both signal and variance rates.
            Each rotation-wavelength pair is treated as an
            independent measurement channel.

            Correlated noise and systematic residuals are excluded.
        """
        if not isinstance(planet_left, u.Quantity):
            raise TypeError("planet_left must be a Quantity.")

        shape = planet_left.shape

        if len(shape) != 3 or any(size == 0 for size in shape):
            raise ValueError(
                "planet_left must have shape "
                "(n_rotations, n_planets, n_wavelengths)."
            )

        n_rotations, n_planets, n_wavelengths = shape
        spectral_shape = (n_rotations, n_wavelengths)

        def check_rate(name, rate, expected_shape):
            if not isinstance(rate, u.Quantity):
                raise TypeError(f"{name} must be a Quantity.")

            values = np.asarray(rate.to_value(1 / u.s), dtype=float)

            if values.shape != expected_shape:
                raise ValueError(f"{name} must have shape {expected_shape}.")

            if not np.all(np.isfinite(values)) or np.any(values < 0):
                raise ValueError(f"{name} must be finite and nonnegative.")

            return values

        pl = check_rate("planet_left", planet_left, shape)
        pr = check_rate("planet_right", planet_right, shape)
        sl = check_rate("stellar_left", stellar_left, spectral_shape)
        sr = check_rate("stellar_right", stellar_right, spectral_shape)

        # Photon noise from the stellar leakage.
        background = sl + sr

        # Add each independent extended-background component.
        if background_rates is not None:
            for index, pair in enumerate(background_rates):
                if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                    raise ValueError("Background components must be left/right pairs.")

                bl = check_rate(
                    f"background[{index}].left",
                    pair[0],
                    spectral_shape,
                )
                br = check_rate(
                    f"background[{index}].right",
                    pair[1],
                    spectral_shape,
                )

                background += bl + br

        # Detector variance for one output.
        detector = np.zeros(n_wavelengths)

        if detector_dark_rate is not None:
            detector += check_rate(
                "detector_dark_rate",
                detector_dark_rate,
                (n_wavelengths,),
            )

        if detector_read_variance_rate is not None:
            detector += check_rate(
                "detector_read_variance_rate",
                detector_read_variance_rate,
                (n_wavelengths,),
            )

        # Left and right outputs have independent detector noise.
        background += 2.0 * detector[None, :]

        # Allocate the total integration time among rotations.
        if rotation_fractions is None:
            fractions = np.full(n_rotations, 1.0 / n_rotations)
        else:
            fractions = np.asarray(rotation_fractions, dtype=float)

            if (
                fractions.shape != (n_rotations,)
                or not np.all(np.isfinite(fractions))
                or np.any(fractions < 0)
                or not np.isclose(fractions.sum(), 1.0, rtol=0.0, atol=1e-12)
            ):
                raise ValueError(
                    "rotation_fractions must be nonnegative, "
                    "have one entry per rotation, and sum to one."
                )

        # Differential planet signal.
        signal = (pl - pr) * fractions[:, None, None]

        # Independent Poisson variances add, not subtract.
        variance = (pl + pr + background[:, None, :]) * fractions[:, None, None]

        # Merge rotation and wavelength into one channel axis.
        signal = signal.transpose(1, 0, 2).reshape(
            n_planets, n_rotations * n_wavelengths
        )

        variance = variance.transpose(1, 0, 2).reshape(
            n_planets, n_rotations * n_wavelengths
        )

        return signal / u.s, variance / u.s

    @staticmethod
    def calc_life_snr(
        signal_rates: u.Quantity,
        noise_variance_rates: u.Quantity,
        integration_time: u.Quantity,
        systematic_rates: u.Quantity | None = None,
    ) -> float | np.ndarray:
        """Calculate spectral matched-filter detection SNR.

        Assumes statistically independent spectral channels and
        a fixed signal template.

        Args:
            signal_rates (astropy.units.Quantity):
                Phase-chopped planet signal rates in 1/s.
                The final axis represents wavelength.
            noise_variance_rates (astropy.units.Quantity):
                Photon and detector noise-variance rates in 1/s,
                with the same shape as signal_rates.
            integration_time (astropy.units.Quantity):
                Positive scalar integration time.
            systematic_rates (astropy.units.Quantity, optional):
                Systematic noise amplitude rates in 1/s, with the
                same shape as signal_rates. Defaults to zero.

        Returns:
            float or numpy.ndarray:
                Dimensionless SNR, summed over the spectral axis.
                Leading dimensions are preserved.

        Notes:
            SNR**2 = sum((S*t)**2 / (V*t + (Csp*t)**2)).

            Systematic noise is assumed independent between
            spectral channels. Correlated systematics require
            a covariance-matrix formulation.
        """
        if not isinstance(signal_rates, u.Quantity):
            raise TypeError("signal_rates must be a Quantity.")

        if not isinstance(noise_variance_rates, u.Quantity):
            raise TypeError("noise_variance_rates must be a Quantity.")

        if not isinstance(integration_time, u.Quantity):
            raise TypeError("integration_time must be a Quantity.")

        signal = np.asarray(signal_rates.to_value(1 / u.s), dtype=float)
        variance = np.asarray(noise_variance_rates.to_value(1 / u.s), dtype=float)
        time = np.asarray(integration_time.to_value(u.s), dtype=float)

        if signal.ndim == 0 or signal.shape[-1] == 0:
            raise ValueError("signal_rates must have a spectral axis.")

        if variance.shape != signal.shape:
            raise ValueError("noise_variance_rates must match signal_rates shape.")

        if time.ndim != 0 or not np.isfinite(time) or time <= 0:
            raise ValueError("integration_time must be finite and positive.")

        if systematic_rates is None:
            systematic = np.zeros_like(signal)
        else:
            if not isinstance(systematic_rates, u.Quantity):
                raise TypeError("systematic_rates must be a Quantity.")

            systematic = np.asarray(
                systematic_rates.to_value(1 / u.s),
                dtype=float,
            )

            if systematic.shape != signal.shape:
                raise ValueError("systematic_rates must match signal_rates shape.")

        if (
            not np.all(np.isfinite(signal))
            or not np.all(np.isfinite(variance))
            or np.any(variance < 0)
            or not np.all(np.isfinite(systematic))
            or np.any(systematic < 0)
        ):
            raise ValueError("Rates must be finite, and noise rates nonnegative.")

        numerator = (signal * time) ** 2
        denominator = variance * time + (systematic * time) ** 2

        if np.any((denominator == 0) & (numerator > 0)):
            raise ValueError(
                "Noise variance must be positive wherever " "the signal is nonzero."
            )

        # Zero-signal, zero-noise channels contribute zero SNR.
        channel_snr_squared = np.zeros_like(numerator)

        np.divide(
            numerator,
            denominator,
            out=channel_snr_squared,
            where=denominator > 0,
        )

        snr_squared = np.sum(channel_snr_squared, axis=-1)

        result = np.sqrt(snr_squared)

        return float(result) if result.ndim == 0 else result

    @staticmethod
    def calc_life_int_time(
        signal_rates: u.Quantity,
        noise_variance_rates: u.Quantity,
        snr_threshold: float,
        systematic_rates: u.Quantity | None = None,
    ) -> u.Quantity:
        """Calculate integration time required for a spectral detection.

        Uses the same independent-channel noise model as calc_life_snr.

        Args:
            signal_rates (astropy.units.Quantity):
                Phase-chopped planet signal rates in 1/s.
                The final axis represents wavelength.
            noise_variance_rates (astropy.units.Quantity):
                Photon and detector noise-variance rates in 1/s.
                Must be positive wherever the signal is nonzero.
            snr_threshold (float):
                Required positive detection SNR.
            systematic_rates (astropy.units.Quantity, optional):
                Systematic noise amplitude rates in 1/s.
                Defaults to zero.

        Returns:
            astropy.units.Quantity:
                Required integration time in seconds.
                Leading input dimensions are preserved.
                Returns NaN for unreachable detection thresholds.

        Notes:
            SNR**2 = sum(S**2 * t / (V + Csp**2 * t)).

            Assumes independent spectral channels and time-independent
            signal and noise rates.
        """
        if not isinstance(signal_rates, u.Quantity):
            raise TypeError("signal_rates must be a Quantity.")

        if not isinstance(noise_variance_rates, u.Quantity):
            raise TypeError("noise_variance_rates must be a Quantity.")

        signal = np.asarray(signal_rates.to_value(1 / u.s), dtype=float)
        variance = np.asarray(noise_variance_rates.to_value(1 / u.s), dtype=float)

        if signal.ndim == 0 or signal.shape[-1] == 0:
            raise ValueError("signal_rates must have a spectral axis.")

        if variance.shape != signal.shape:
            raise ValueError("noise_variance_rates must match signal_rates shape.")

        if systematic_rates is None:
            systematic = np.zeros_like(signal)
        else:
            if not isinstance(systematic_rates, u.Quantity):
                raise TypeError("systematic_rates must be a Quantity.")

            systematic = np.asarray(
                systematic_rates.to_value(1 / u.s),
                dtype=float,
            )

            if systematic.shape != signal.shape:
                raise ValueError("systematic_rates must match signal_rates shape.")

        threshold = np.asarray(snr_threshold, dtype=float)

        if threshold.ndim != 0 or not np.isfinite(threshold) or threshold <= 0:
            raise ValueError("snr_threshold must be finite and positive.")

        if (
            not np.all(np.isfinite(signal))
            or not np.all(np.isfinite(variance))
            or np.any(variance < 0)
            or not np.all(np.isfinite(systematic))
            or np.any(systematic < 0)
        ):
            raise ValueError("Rates must be finite, and noise rates nonnegative.")

        if np.any((signal != 0) & (variance <= 0)):
            raise ValueError(
                "Noise variance must be positive wherever " "the signal is nonzero."
            )

        target_squared = float(threshold) ** 2
        n_wavelengths = signal.shape[-1]

        # One result per observing configuration.
        results = np.full(signal.shape[:-1], np.nan, dtype=float)

        signal_rows = signal.reshape(-1, n_wavelengths)
        variance_rows = variance.reshape(-1, n_wavelengths)
        systematic_rows = systematic.reshape(-1, n_wavelengths)

        for index, (s, v, csp) in enumerate(
            zip(signal_rows, variance_rows, systematic_rows)
        ):
            active = s != 0

            # No planet signal means no detection.
            if not np.any(active):
                continue

            signal_squared = s[active] ** 2
            photon_variance = v[active]
            systematic_squared = csp[active] ** 2

            # A finite systematic floor can make the required
            # detection SNR impossible.
            if np.all(systematic_squared > 0):
                maximum_snr_squared = np.sum(signal_squared / systematic_squared)

                if target_squared >= maximum_snr_squared:
                    continue

            # Exact photon-noise-only solution.
            photon_time = target_squared / np.sum(signal_squared / photon_variance)

            if np.all(systematic_squared == 0):
                results.flat[index] = photon_time
                continue

            # General case: numerically invert the SNR equation.
            def residual(time):
                if time == 0:
                    return -target_squared

                snr_squared = np.sum(
                    signal_squared / (photon_variance / time + systematic_squared)
                )

                return snr_squared - target_squared

            # Systematic noise cannot shorten the required
            # integration time relative to the photon-only model.
            upper = photon_time

            for _ in range(512):
                if not np.isfinite(upper):
                    break

                if residual(upper) >= 0:
                    results.flat[index] = brentq(
                        residual,
                        0.0,
                        upper,
                        xtol=1e-12,
                        rtol=1e-12,
                    )
                    break

                upper *= 2.0

        return results * u.s

    # ================================================================
    # Stage 5 — EXOSIMS compatibility (photon-noise reference)
    # ================================================================

    def _get_life_spectral_rates(self, TL, sInds, fZ, JEZ, dMag, WA, mode, TK):
        """Obtain observation-specific spectral rates.

        A callable life_rate_provider must be attached to this
        instance. It receives the standard EXOSIMS observation
        arguments and returns the spectral rates.

        Returns:
            tuple:
                Signal, noise-variance, and systematic rates in 1/s,
                each shaped (n_candidates, n_channels).

        Notes:
            The provider is responsible for constructing the
            thermal-IR observation. This method does not generate
            stellar or planetary spectra.
        """
        provider = getattr(self, "life_rate_provider", None)

        if not callable(provider):
            raise NotImplementedError(
                "LIFE requires observation-specific spectral rates. "
                "Attach a callable life_rate_provider before using "
                "the standard EXOSIMS OpticalSystem interfaces."
            )

        result = provider(TL, sInds, fZ, JEZ, dMag, WA, mode, TK)

        if not isinstance(result, (tuple, list)) or len(result) not in (2, 3):
            raise ValueError(
                "life_rate_provider must return (signal, variance) "
                "or (signal, variance, systematic)."
            )

        signal, variance = result[:2]
        systematic = result[2] if len(result) == 3 else None

        for name, value in (
            ("signal", signal),
            ("variance", variance),
        ):
            if not isinstance(value, u.Quantity):
                raise TypeError(f"{name} must be an Astropy Quantity.")

        signal = signal.to(1 / u.s)
        variance = variance.to(1 / u.s)

        if (
            signal.ndim != 2
            or signal.shape[0] == 0
            or signal.shape[1] == 0
            or variance.shape != signal.shape
        ):
            raise ValueError(
                "signal and variance must have identical, nonempty "
                "(n_candidates, n_channels) shapes."
            )

        if systematic is None:
            systematic = np.zeros(signal.shape) / u.s
        elif not isinstance(systematic, u.Quantity):
            raise TypeError("systematic must be an Astropy Quantity.")
        else:
            systematic = systematic.to(1 / u.s)

        if systematic.shape != signal.shape:
            raise ValueError("systematic must match the signal shape.")

        s = signal.to_value(1 / u.s)
        v = variance.to_value(1 / u.s)
        csp = systematic.to_value(1 / u.s)

        if (
            not np.all(np.isfinite(s))
            or not np.all(np.isfinite(v))
            or not np.all(np.isfinite(csp))
            or np.any(v < 0)
            or np.any(csp < 0)
        ):
            raise ValueError("Signal and noise rates must be finite and valid.")

        if np.any((v == 0) & (s != 0)):
            raise ValueError("Nonzero signal requires positive variance.")

        return signal, variance, systematic

    def Cp_Cb_Csp(
        self,
        TL,
        sInds,
        fZ,
        JEZ,
        dMag,
        WA,
        mode,
        returnExtra=False,
        TK=None,
    ):
        """Return SNR-equivalent count rates for photon-noise-only LIFE.

        The effective signal rate is defined by

            C_p**2 / C_b = sum_i(S_i**2 / V_i).

        C_b is the sum of independent noise-variance rates.
        C_p is an effective matched-filter signal, not the
        physical planet photon count rate.

        The mapping preserves the spectral SNR for all positive
        integration times when systematic rates are zero.
        """
        if returnExtra:
            raise NotImplementedError(
                "LIFE returnExtra needs a validated per-source " "rate breakdown."
            )

        signal, variance, systematic = self._get_life_spectral_rates(
            TL, sInds, fZ, JEZ, dMag, WA, mode, TK
        )

        if np.any(systematic.to_value(1 / u.s) != 0):
            raise NotImplementedError(
                "A single (C_p, C_b, C_sp) triple cannot generally "
                "preserve LIFE's multi-channel SNR with "
                "systematic noise."
            )

        s = signal.to_value(1 / u.s)
        v = variance.to_value(1 / u.s)

        total_variance = np.sum(v, axis=1)

        if np.any(total_variance <= 0):
            raise ValueError("Total noise variance rate must be positive.")

        weighted_snr_rate = np.zeros_like(s)

        np.divide(
            s**2,
            v,
            out=weighted_snr_rate,
            where=v > 0,
        )

        coefficient = np.sum(weighted_snr_rate, axis=1)

        C_b = total_variance / u.s
        C_p = np.sqrt(coefficient * total_variance) / u.s
        C_sp = np.zeros_like(coefficient) / u.s

        return C_p, C_b, C_sp

    def calc_intTime(self, TL, sInds, fZ, JEZ, dMag, WA, mode, TK=None):
        """Compute spectral LIFE integration times in days.

        Preserves the EXOSIMS method signature while using
        the LIFE spectral matched-filter calculation directly.
        """
        signal, variance, systematic = self._get_life_spectral_rates(
            TL, sInds, fZ, JEZ, dMag, WA, mode, TK
        )

        return self.calc_life_int_time(
            signal,
            variance,
            snr_threshold=mode["SNR"],
            systematic_rates=systematic,
        ).to(u.day)

    def calc_dMag_per_intTime(
        self,
        intTimes,
        TL,
        sInds,
        fZ,
        JEZ,
        WA,
        mode,
        C_b=None,
        C_sp=None,
        TK=None,
        analytic_only=False,
    ):
        """Reject reflected-light dMag inversion for thermal LIFE."""
        raise NotImplementedError(
            "LIFE needs a defined thermal brightness/spectral "
            "template before calc_dMag_per_intTime can be implemented."
        )

    def ddMag_dt(
        self,
        intTimes,
        TL,
        sInds,
        fZ,
        JEZ,
        WA,
        mode,
        C_b=None,
        C_sp=None,
        TK=None,
    ):
        """Reject the inherited reflected-light dMag derivative."""
        raise NotImplementedError(
            "LIFE ddMag_dt requires a thermal brightness convention."
        )

    def calc_saturation_dMag(self, TL, sInds, fZ, JEZ, WA, mode, TK=None):
        """Reject an unsupported coronagraphic saturation threshold."""
        raise NotImplementedError(
            "No finite LIFE systematic floor is defined; "
            "an inherited contrast-limited saturation dMag "
            "would be unphysical."
        )
