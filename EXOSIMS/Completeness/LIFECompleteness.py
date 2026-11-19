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
