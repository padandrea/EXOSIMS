"""Optical system model for the LIFE nulling interferometer."""

import astropy.units as u
import numpy as np
from synphot import Observation, SpectralElement
from synphot.models import Box1D, Gaussian1D

from EXOSIMS.Prototypes.OpticalSystem import OpticalSystem


class LIFEOpticalSystem(OpticalSystem):
    """LIFE mid-infrared nulling interferometer optical system.

    Args:
        baseline_ratio (float):
            Imaging-to-nulling baseline ratio. Defaults to 6.0.
        **specs (dict):
            Additional EXOSIMS OpticalSystem configuration parameters.

    Attributes:
        baseline_ratio (float):
            Ratio of imaging baseline to nulling baseline.
    """

    # ================================================================
    # Stage 0 — Initialization
    # ================================================================
    # Configure the LIFE instrument and initialize EXOSIMS interfaces.


    def __init__(self, baseline_ratio=6.0, **specs):
        """Initialize the LIFE optical system."""
        baseline_ratio = float(baseline_ratio)

        if not np.isfinite(baseline_ratio) or baseline_ratio <= 0:
            raise ValueError("baseline_ratio must be finite and positive.")

        super().__init__(**specs)

        self.baseline_ratio = baseline_ratio
        self._outspec["baseline_ratio"] = baseline_ratio

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
    # Stage 1 — Geometry and spectrum
    # ================================================================

    # ----------------------------------------------------------------
    # Array geometry
    # ----------------------------------------------------------------

    def get_collector_positions(self, nulling_baseline: u.Quantity) -> u.Quantity:
        """Calculate the four collector positions of the LIFE X-array.

        The array is centered at the origin. The imaging baseline is
        calculated from the nulling baseline and baseline_ratio.

        Args:
            nulling_baseline (astropy.units.Quantity):
                Nulling baseline with physical length units.

        Returns:
            astropy.units.Quantity:
                Collector positions with shape (4, 2), in metres.
                Each row contains the (x, y) coordinates of a collector.

        Raises:
            TypeError:
                If nulling_baseline is not an Astropy Quantity.
            ValueError:
                If the baseline is not a positive finite scalar length.
        """
        if not isinstance(nulling_baseline, u.Quantity):
            raise TypeError("nulling_baseline must be an Astropy Quantity.")

        try:
            b_null = nulling_baseline.to_value(u.m)
        except u.UnitConversionError as exc:
            raise ValueError("nulling_baseline must have length units.") from exc

        if np.ndim(b_null) != 0 or not np.isfinite(b_null) or b_null <= 0:
            raise ValueError("nulling_baseline must be finite and positive.")

        b_img = self.baseline_ratio * b_null

        positions = np.array(
            [
                [-b_null / 2, -b_img / 2],
                [-b_null / 2, +b_img / 2],
                [+b_null / 2, -b_img / 2],
                [+b_null / 2, +b_img / 2],
            ],
            dtype=float,
        )

        return positions * u.m

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
        The final bin may extend beyond wavelength_max.

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

        centers = (edges[:-1] + edges[1:]) / 2
        widths = edges[1:] - edges[:-1]

        return centers * u.m, widths * u.m

    # ----------------------------------------------------------------
    # Baseline geometry and rotation
    # ----------------------------------------------------------------

    def get_baselines(
        self,
        nulling_baseline: u.Quantity,
        rotation_angles: u.Quantity,
    ) -> u.Quantity:
        """Calculate pairwise baseline vectors for the rotating LIFE array.

        Args:
            nulling_baseline (astropy.units.Quantity):
                Nulling baseline with length units.
            rotation_angles (astropy.units.Quantity):
                Scalar or 1D array of rotation angles.

        Returns:
            astropy.units.Quantity:
                Baseline vectors with shape (n_rotations, 4, 4, 2),
                in metres. Entry [a, j, k] is r_j - r_k.
        """
        if not isinstance(rotation_angles, u.Quantity):
            raise TypeError("rotation_angles must be an Astropy Quantity.")

        angles = np.atleast_1d(rotation_angles.to_value(u.rad))

        if angles.ndim != 1 or not np.all(np.isfinite(angles)):
            raise ValueError("rotation_angles must be finite and 1D.")

        positions = self.get_collector_positions(
            nulling_baseline
        ).to_value(u.m)

        x = positions[None, :, 0]
        y = positions[None, :, 1]

        cos_a = np.cos(angles)[:, None]
        sin_a = np.sin(angles)[:, None]

        rotated = np.stack(
            [
                x * cos_a - y * sin_a,
                x * sin_a + y * cos_a,
            ],
            axis=-1,
        )

        baselines = (
            rotated[:, :, None, :] - rotated[:, None, :, :]
        )

        return baselines * u.m


    # ================================================================
    # Stage 2 — Interferometry
    # ================================================================

    def get_transmission(
        self,
        nulling_baseline: u.Quantity,
        rotation_angles: u.Quantity,
        wavelengths: u.Quantity,
        sky_positions: u.Quantity,
    ) -> tuple[u.Quantity, u.Quantity, u.Quantity]:
        """Calculate point-source transmission of the LIFE double Bracewell.

        The two chopped outputs use the quadrature phase shifts of the
        double Bracewell beam combiner.
        Transmission is normalized to the total collecting area of the
        four telescopes, without throughput or photon-rate factors.

        Args:
            nulling_baseline (astropy.units.Quantity):
                Nulling baseline with length units.
            rotation_angles (astropy.units.Quantity):
                Scalar or 1D array of array-rotation angles.
            wavelengths (astropy.units.Quantity):
                Scalar or 1D array of wavelengths with length units.
            sky_positions (astropy.units.Quantity):
                Angular offsets (theta_x, theta_y), shape (2,) or (n_sky, 2).

        Returns:
            tuple[astropy.units.Quantity, astropy.units.Quantity, astropy.units.Quantity]:
                Left, right, and left-minus-right transmission, dimensionless.
                Each has shape (n_rotations, n_sky, n_wavelengths).

        Raises:
            TypeError:
                If wavelengths or sky_positions are not Astropy Quantities.
            ValueError:
                If wavelengths or sky positions have invalid units or values.
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

        baselines_m = self.get_baselines(
            nulling_baseline, rotation_angles
        ).to_value(u.m)
        if baselines_m.shape[0] == 0:
            raise ValueError("At least one rotation angle is required.")

        # A common optical phase does not affect intensity. Use collector 0
        # as the reference and avoid explicitly summing all 16 pairs.
        relative_positions = baselines_m[:, :, 0, :]
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
    # TODO: get_photon_rates()
    # Calculate planet, stellar leakage, zodiacal, exozodiacal
    # and detector photon rates.

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
