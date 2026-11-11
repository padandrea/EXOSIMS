"""Optical system model for the LIFE nulling interferometer."""

import astropy.units as u
import numpy as np
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
                "bandpass_model",
                "bandpass_step",
            ]

            # get mode wavelength and bandwidth (get system's values by default)
            # when provided, always use deltaLam instead of BW (bandwidth fraction)
            syst_lam = mode["syst"]["lam"].to("nm").value
            syst_BW = mode["syst"]["BW"]
            mode["lam"] = float(mode.get("lam", syst_lam)) * u.nm
            mode["deltaLam"] = (
                float(mode.get("deltaLam", mode["lam"].value * mode.get("BW", syst_BW)))
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

        if not np.isfinite(resolving_power) or resolving_power <= 0:
            raise ValueError("resolving_power must be finite and positive.")

        # Geometric wavelength bins with constant nominal resolving power.
        step = 1.0 + 1.0 / resolving_power

        n_bins = int(
            np.ceil(np.log(lam_max / lam_min) / np.log1p(1.0 / resolving_power))
        )

        edges = lam_min * step ** np.arange(n_bins + 1)
        edges[-1] = lam_max

        centers = (edges[:-1] + edges[1:]) / 2
        widths = edges[1:] - edges[:-1]

        return centers * u.m, widths * u.m

    def get_spectral_bins_from_mode(
        self, mode: dict
    ) -> tuple[u.Quantity, u.Quantity]:
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
            raise ValueError(
                "collector_positions must have length units."
            ) from exc

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
            2.0 * np.pi * delays_m[:, :, None, :]
            / wavelength_m[None, None, :, None]
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

        flux = np.atleast_1d(
            planet_spectral_flux.to(
                u.photon / (u.s * u.m**2 * u.um)
            )
        )
        widths = np.atleast_1d(bin_widths.to(u.um))
        n_wavelengths = left.shape[-1]

        if flux.shape != (n_wavelengths,):
            raise ValueError(
                "planet_spectral_flux must have one value per wavelength."
            )

        if widths.shape != (n_wavelengths,):
            raise ValueError(
                "bin_widths must have one value per wavelength."
            )

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
            flux
            * widths
            * self.total_collecting_area
            * throughput
            * qe
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
        flux = np.atleast_1d(
            stellar_spectral_flux.to(
                u.photon / (u.s * u.m**2 * u.um)
            )
        )
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
        baselines = (
            positions[:, :, None, :]
            - positions[:, None, :, :]
        )
        baseline_lengths = np.linalg.norm(baselines, axis=-1)

        # Uniform-disk coherence: 2*J1(x)/x, with limit 1 at x=0.
        x = (
            2.0 * np.pi * radius
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

        weights_left = np.cos(
            phase_left[:, None] - phase_left[None, :]
        )
        weights_right = np.cos(
            phase_right[:, None] - phase_right[None, :]
        )

        transmission_left = np.einsum(
            "ij,cwij->cw", weights_left, coherence
        ) / 16.0
        transmission_right = np.einsum(
            "ij,cwij->cw", weights_right, coherence
        ) / 16.0

        # Remove possible negative values from floating-point roundoff.
        transmission_left = np.maximum(transmission_left, 0.0)
        transmission_right = np.maximum(transmission_right, 0.0)

        qe = mode["inst"]["QE"](np.atleast_1d(wavelengths))

        incident_rate = (
            flux
            * widths
            * self.total_collecting_area
            * mode["attenuation"]
            * qe
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

        radiance = spectral_radiance.to(
            u.photon / (u.s * u.m**2 * u.um * u.sr)
        )
        solid_angles = np.atleast_1d(
            pixel_solid_angles.to(u.sr)
        )
        widths = np.atleast_1d(bin_widths.to(u.um))

        if radiance.shape != (n_sky, n_wavelengths):
            raise ValueError(
                "spectral_radiance must have shape "
                "(n_sky, n_wavelengths)."
            )

        if solid_angles.shape != (n_sky,):
            raise ValueError(
                "pixel_solid_angles must have shape (n_sky,)."
            )

        if widths.shape != (n_wavelengths,):
            raise ValueError(
                "bin_widths must have one value per wavelength."
            )

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

        qe = mode["inst"]["QE"](
            np.atleast_1d(wavelengths)
        )

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
        rate_left = (
            left * patch_rates[None, :, :]
        ).sum(axis=1)

        rate_right = (
            right * patch_rates[None, :, :]
        ).sum(axis=1)

        return rate_left.to(1 / u.s), rate_right.to(1 / u.s)


    # ================================================================
    # Stage 4 — Detection
    # ================================================================
    # TODO: calc_life_snr()
    # TODO: calc_life_int_time()
    # Calculate matched-filter SNR and required integration time.

    # ================================================================
    # Stage 5 — EXOSIMS compatibility
    # ================================================================
    # TODO: Cp_Cb_Csp()
    # TODO: calc_intTime()
    # TODO: calc_dMag_per_intTime()
    # TODO: ddMag_dt()
    # TODO: calc_saturation_dMag()
    # Connect LIFE calculations to the EXOSIMS OpticalSystem API.
