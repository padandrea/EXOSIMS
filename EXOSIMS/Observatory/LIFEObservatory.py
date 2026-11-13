"""LIFE observatory with a Sun--Earth L2 halo orbit and ideal array geometry."""

from numbers import Real

import astropy.units as u
import numpy as np

from EXOSIMS.Observatory.ObservatoryL2Halo import ObservatoryL2Halo


class LIFEObservatory(ObservatoryL2Halo):
    """Model the LIFE observatory and its ideal projected four-collector X-array.

    Orbital positions and target keepout are inherited from ObservatoryL2Halo.
    Collector coordinates are defined in an orthonormal tangent plane to the
    target direction, with axes matching the angular offsets supplied to the
    optical system. The idealized array is perpendicular to the line of sight;
    three-dimensional formation dynamics and pointing constraints are omitted.

    Collector ordering is fixed to match the double-Bracewell beam combiner:
    (-x, -y), (-x, +y), (+x, -y), (+x, +y).

    Args:
        nulling_baseline (float or astropy.units.Quantity):
            Separation along the projected x-axis. A unitless number is
            interpreted in metres. Defaults to 14.5 m.
        baseline_ratio (float):
            Imaging-baseline to nulling-baseline ratio. Defaults to 6.0.
        **specs (dict):
            ObservatoryL2Halo configuration. In the absence of a specified
            settlingTime, zero overhead is assumed rather than an unvalidated
            LIFE-specific retargeting time.
    """

    def __init__(
        self,
        nulling_baseline: float | u.Quantity = 14.5,
        baseline_ratio: float = 6.0,
        **specs,
    ) -> None:
        """Validate array configuration and initialize the inherited L2 orbit."""
        if isinstance(nulling_baseline, u.Quantity):
            try:
                baseline = nulling_baseline.to(u.m)
            except u.UnitConversionError as exc:
                raise ValueError("nulling_baseline must have length units.") from exc
        elif isinstance(nulling_baseline, Real) and not isinstance(
            nulling_baseline, bool
        ):
            baseline = float(nulling_baseline) * u.m
        else:
            raise TypeError(
                "nulling_baseline must be a real scalar or length Quantity."
            )

        if (
            not baseline.isscalar
            or not np.isfinite(baseline.to_value(u.m))
            or baseline <= 0 * u.m
        ):
            raise ValueError("nulling_baseline must be finite, scalar, and positive.")

        if not isinstance(baseline_ratio, Real) or isinstance(baseline_ratio, bool):
            raise TypeError("baseline_ratio must be a real scalar.")
        ratio = float(baseline_ratio)
        if not np.isfinite(ratio) or ratio <= 0:
            raise ValueError("baseline_ratio must be finite and positive.")

        # The prototype's default settling time is not a LIFE calibration.
        # Zero means unmodelled overhead, not instantaneous real retargeting.
        specs.setdefault("settlingTime", 0.0)
        super().__init__(**specs)

        self.nulling_baseline = baseline
        self.baseline_ratio = ratio
        self.imaging_baseline = ratio * baseline

        self._outspec["nulling_baseline"] = baseline.to_value(u.m)
        self._outspec["baseline_ratio"] = ratio

    def get_collector_positions(
        self, rotation_angles: u.Quantity = 0.0 * u.deg
    ) -> u.Quantity:
        """Return ordered collector positions in the target tangent sky plane.

        The origin is the array centroid. Positive angles rotate the array
        counterclockwise about the target line of sight, from +x towards +y.
        The x-axis spans the nulling baseline and the y-axis the imaging
        baseline. This is ideal projected geometry, not a 3D orbit solution.

        Args:
            rotation_angles (astropy.units.Quantity):
                Scalar or nonempty 1D array of angular roll positions.

        Returns:
            astropy.units.Quantity:
                Metre-valued array shaped (n_rotations, 4, 2), including for
                a scalar input. Collector ordering is preserved under rotation.

        Raises:
            TypeError:
                If rotation_angles is not an Astropy Quantity.
            ValueError:
                If angles have incompatible units, shape, or nonfinite values.
        """
        if not isinstance(rotation_angles, u.Quantity):
            raise TypeError("rotation_angles must be an Astropy Quantity.")
        try:
            angles = np.atleast_1d(rotation_angles.to_value(u.rad))
        except u.UnitConversionError as exc:
            raise ValueError("rotation_angles must have angular units.") from exc

        if angles.ndim != 1 or angles.size == 0 or not np.all(np.isfinite(angles)):
            raise ValueError("rotation_angles must be a nonempty, finite 1D array.")

        half_null = self.nulling_baseline.to_value(u.m) / 2.0
        half_imaging = self.imaging_baseline.to_value(u.m) / 2.0
        reference = np.array(
            [
                [-half_null, -half_imaging],
                [-half_null, half_imaging],
                [half_null, -half_imaging],
                [half_null, half_imaging],
            ],
            dtype=float,
        )

        cosine = np.cos(angles)[:, None]
        sine = np.sin(angles)[:, None]
        x = reference[None, :, 0]
        y = reference[None, :, 1]
        positions = np.stack((cosine * x - sine * y, sine * x + cosine * y), axis=-1)
        return positions * u.m
