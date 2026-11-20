"""LIFE-specific completeness and Monte Carlo detection statistics."""

import astropy.units as u
import numpy as np

from EXOSIMS.Prototypes.Completeness import Completeness
from EXOSIMS.OpticalSystem.LIFEOpticalSystem import LIFEOpticalSystem


class LIFECompleteness(Completeness):
    """LIFE completeness using thermal-IR detection calculations.

    The detection-statistics estimator is implemented first.
    EXOSIMS completeness interfaces remain explicitly unsupported
    until LIFE planet populations and spectral rates are connected.
    """

    _modtype = "Completeness"

    @staticmethod
    def detection_times_from_rates(
        signal_rates,
        noise_variance_rates,
        snr_threshold,
        systematic_rates=None,
    ):
        """Calculate planet detection times from LIFE spectral rates.

        Parameters
        ----------
        signal_rates, noise_variance_rates : astropy.units.Quantity
            Arrays of shape (n_universes, n_planet_slots, n_channels)
            with units of 1/s. Channels may contain signed signals.
        snr_threshold : float
            Required detection SNR.
        systematic_rates : astropy.units.Quantity, optional
            Systematic noise amplitude rates, same shape and units.

        Returns
        -------
        astropy.units.Quantity
            Detection times with shape (n_universes, n_planet_slots).
            Undetectable planets are represented by positive infinity.

        Notes
        -----
        Each row represents one simulated universe for one target.
        Photon rates must already include the appropriate LIFE
        transmission, rotation, spectral and background calculations.
        """
        if not isinstance(signal_rates, u.Quantity):
            raise TypeError("signal_rates must be an Astropy Quantity.")

        if (
            signal_rates.ndim != 3
            or signal_rates.shape[0] == 0
            or signal_rates.shape[2] == 0
        ):
            raise ValueError(
                "signal_rates must have shape "
                "(n_universes, n_planet_slots, n_channels), "
                "with nonempty universe and channel axes."
            )

        required_times = LIFEOpticalSystem.calc_life_int_time(
            signal_rates,
            noise_variance_rates,
            snr_threshold=snr_threshold,
            systematic_rates=systematic_rates,
        )

        seconds = required_times.to_value(u.s)

        if np.any(np.isinf(seconds)) or np.any(np.isfinite(seconds) & (seconds <= 0)):
            raise ValueError("Instrument returned invalid detection times.")

        # LIFEOpticalSystem returns NaN when SNR is unreachable.
        # The Monte Carlo statistics use +inf for such planets.
        seconds = np.where(np.isnan(seconds), np.inf, seconds)

        return seconds * u.s

    @staticmethod
    def detection_times_from_survey(
        survey, observing_inputs, universes, observation_time
    ):
        """Collect physical LIFE detection times across simulated universes.

        observing_inputs contains the shared observing configuration for
        one target star, excluding planet inputs and integration_time.

        Each universe contains planet_sky_positions and
        planet_spectral_fluxes, plus optional system-level nEZ.

        Returns a Quantity of shape (n_universes, max_planets).
        Missing or undetectable planets are represented by +inf.
        """
        if not callable(getattr(survey, "get_life_detection_metrics", None)):
            raise TypeError("survey must provide get_life_detection_metrics.")

        if not isinstance(observing_inputs, dict):
            raise TypeError("observing_inputs must be a dictionary.")

        if not isinstance(universes, (list, tuple)) or not universes:
            raise ValueError("Provide at least one simulated universe.")

        if not isinstance(observation_time, u.Quantity):
            raise TypeError("observation_time must be a Quantity.")

        budget = np.asarray(observation_time.to_value(u.s))
        if budget.ndim != 0 or not np.isfinite(budget) or budget < 0:
            raise ValueError("observation_time must be finite and nonnegative.")

        forbidden = {
            "planet_sky_positions",
            "planet_spectral_fluxes",
            "integration_time",
        }
        if forbidden.intersection(observing_inputs):
            raise ValueError(
                "Remove planet inputs and integration_time from observing_inputs."
            )

        wavelengths = observing_inputs.get("wavelengths")
        if not isinstance(wavelengths, u.Quantity) or wavelengths.ndim != 1:
            raise ValueError("observing_inputs requires a wavelength array.")
        if wavelengths.size == 0:
            raise ValueError("The wavelength array cannot be empty.")

        flux_unit = u.photon / (u.s * u.m**2 * u.um)
        rows = []

        for universe in universes:
            if not isinstance(universe, dict):
                raise TypeError("Each universe must be a dictionary.")

            allowed = {
                "planet_sky_positions",
                "planet_spectral_fluxes",
                "nEZ",
            }
            if set(universe) - allowed:
                raise ValueError("Unexpected universe input.")
            if "nEZ" in universe and "nEZ" in observing_inputs:
                raise ValueError("Specify nEZ in only one input dictionary.")

            positions = universe.get("planet_sky_positions")
            spectra = universe.get("planet_spectral_fluxes")

            if not isinstance(positions, u.Quantity):
                raise TypeError("Planet positions must be Quantities.")
            if not isinstance(spectra, u.Quantity):
                raise TypeError("Planet spectra must be Quantities.")

            positions.to(u.rad)
            spectra.to(flux_unit)

            if positions.ndim != 2 or positions.shape[1] != 2:
                raise ValueError("Planet positions must have shape (n_planets, 2).")

            n_planets = positions.shape[0]
            if spectra.shape != (n_planets, wavelengths.size):
                raise ValueError(
                    "Planet spectra must have shape (n_planets, n_wavelengths)."
                )

            if n_planets == 0:
                rows.append(np.empty(0))
                continue

            result = survey.get_life_detection_metrics(
                **observing_inputs,
                **universe,
                integration_time=observation_time,
            )

            times = result["required_integration_time"]
            if not isinstance(times, u.Quantity):
                raise TypeError("Survey detection times must be Quantities.")

            values = np.asarray(times.to_value(u.s), dtype=float)
            if values.shape != (n_planets,):
                raise ValueError("Survey returned the wrong detection-time shape.")
            if np.any(np.isneginf(values) | (values <= 0)):
                raise ValueError("Survey returned invalid detection times.")

            rows.append(np.where(np.isnan(values), np.inf, values))

        max_planets = max(map(len, rows))
        result = np.full((len(rows), max_planets), np.inf)

        for index, values in enumerate(rows):
            result[index, : len(values)] = values

        return result * u.s

    @staticmethod
    def detection_statistics(detection_times, observation_time):
        """Estimate LIFE detection statistics across simulated universes.

        Parameters
        ----------
        detection_times : astropy.units.Quantity
            Shape (n_universes, n_planet_slots).
            Each row represents one simulated planetary system.
            Positive infinity represents an absent or undetectable planet.
        observation_time : astropy.units.Quantity
            Scalar observing-time budget.

        Returns
        -------
        dict
            Expected detections and probability of at least one detection.
            All simulated universes have equal statistical weight.
        """
        if not isinstance(detection_times, u.Quantity):
            raise TypeError("detection_times must be an Astropy Quantity.")

        if not isinstance(observation_time, u.Quantity):
            raise TypeError("observation_time must be an Astropy Quantity.")

        times = np.asarray(detection_times.to_value(u.s), dtype=float)
        budget = np.asarray(observation_time.to_value(u.s), dtype=float)

        if times.ndim != 2 or times.shape[0] == 0:
            raise ValueError(
                "detection_times must have shape "
                "(n_universes, n_planet_slots), with at least one universe."
            )

        if np.any(np.isnan(times) | np.isneginf(times) | (times <= 0)):
            raise ValueError("Detection times must be positive or positive infinity.")

        if budget.ndim != 0 or not np.isfinite(budget) or budget < 0:
            raise ValueError("observation_time must be a finite nonnegative scalar.")

        detected = times <= float(budget)
        counts = np.count_nonzero(detected, axis=1)

        return {
            "expected_detections": float(np.mean(counts)),
            "probability_at_least_one": float(np.mean(counts > 0)),
        }

    def target_completeness(self, TL):
        """Require LIFE-specific planet-population detection calculations."""
        raise NotImplementedError(
            "LIFE target completeness requires simulated planetary systems "
            "and physically calculated detection times."
        )

    def comp_per_intTime(
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
        """Require LIFE completeness as a function of integration time."""
        raise NotImplementedError(
            "LIFE comp_per_intTime requires spectral detection times."
        )

    def completeness_update(self, TL, sInds, visits, dt):
        """Require LIFE-specific revisit completeness."""
        raise NotImplementedError("LIFE revisit completeness is not implemented.")

    def comp_calc(self, smin, smax, dMag):
        """Reject the inherited reflected-light completeness calculation."""
        raise NotImplementedError(
            "LIFE completeness cannot be calculated from "
            "reflected-light separation and dMag alone."
        )

    def dcomp_dt(
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
        """Require a physical time-dependent LIFE completeness model."""
        raise NotImplementedError(
            "LIFE completeness time derivatives are not implemented."
        )
