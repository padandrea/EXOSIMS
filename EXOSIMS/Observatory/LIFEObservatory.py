"""LIFE observatory with a Sun--Earth L2 halo orbit and ideal array geometry."""

from numbers import Real

import astropy.units as u
import numpy as np

from EXOSIMS.Observatory.ObservatoryL2Halo import ObservatoryL2Halo


class LIFEObservatory(ObservatoryL2Halo):
    """Model an L2 observatory with a configurable projected LIFE X-array.

    Collector coordinates lie in the target's tangent sky plane. They describe
    ideal projected geometry, not the three-dimensional spacecraft formation.
    Collector ordering matches the double-Bracewell phase convention:
    (-x, -y), (-x, +y), (+x, -y), (+x, +y).

    Args:
        nulling_baseline (float or astropy.units.Quantity):
            Default short baseline, in metres for unitless inputs.
        baseline_ratio (float):
            Default imaging-to-nulling baseline ratio.
        nulling_baseline_range (tuple or None):
            Inclusive (minimum, maximum) permitted nulling baseline.
            Unitless endpoints are interpreted in metres.
            Defaults to (10, 100) m. Set None to disable the constraint.
        **specs (dict):
            Configuration passed to ObservatoryL2Halo. Settling time defaults
            to zero because a LIFE-specific overhead is not yet calibrated.
    """

    @staticmethod
    def _positive_length(value: float | u.Quantity, name: str) -> u.Quantity:
        """Convert a scalar length to metres and require a finite positive value."""
        if isinstance(value, u.Quantity):
            try:
                length = value.to(u.m)
            except u.UnitConversionError as exc:
                raise ValueError(f"{name} must have length units.") from exc
        elif isinstance(value, Real) and not isinstance(value, bool):
            length = float(value) * u.m
        else:
            raise TypeError(f"{name} must be a real scalar or length Quantity.")

        if (
            not length.isscalar
            or not np.isfinite(length.to_value(u.m))
            or length <= 0 * u.m
        ):
            raise ValueError(f"{name} must be finite, scalar, and positive.")
        return length

    @staticmethod
    def _positive_ratio(value: float) -> float:
        """Validate a positive, dimensionless baseline ratio."""
        if not isinstance(value, Real) or isinstance(value, bool):
            raise TypeError("baseline_ratio must be a real scalar.")
        ratio = float(value)
        if not np.isfinite(ratio) or ratio <= 0:
            raise ValueError("baseline_ratio must be finite and positive.")
        return ratio

    @classmethod
    def _length_range(
        cls,
        bounds: tuple[float | u.Quantity, float | u.Quantity] | None,
        name: str,
    ) -> tuple[u.Quantity, u.Quantity] | None:
        """Validate an optional inclusive interval of physical lengths."""
        if bounds is None:
            return None
        if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
            raise TypeError(f"{name} must be a pair (minimum, maximum) or None.")
        minimum = cls._positive_length(bounds[0], f"{name} minimum")
        maximum = cls._positive_length(bounds[1], f"{name} maximum")
        if minimum > maximum:
            raise ValueError(f"{name} minimum cannot exceed its maximum.")
        return minimum, maximum

    def __init__(
        self,
        nulling_baseline: float | u.Quantity = 14.5,
        baseline_ratio: float = 6.0,
        nulling_baseline_range: tuple[float | u.Quantity, float | u.Quantity] | None = (
            10.0,
            100.0,
        ),
        imaging_baseline_range: (
            tuple[float | u.Quantity, float | u.Quantity] | None
        ) = None,
        **specs,
    ) -> None:
        """Configure the array, validate allowed baselines, and initialize L2."""
        baseline = self._positive_length(nulling_baseline, "nulling_baseline")
        ratio = self._positive_ratio(baseline_ratio)
        imaging_baseline = ratio * baseline
        nulling_range = self._length_range(
            nulling_baseline_range, "nulling_baseline_range"
        )
        imaging_range = self._length_range(
            imaging_baseline_range, "imaging_baseline_range"
        )

        if nulling_range is not None and not (
            nulling_range[0] <= baseline <= nulling_range[1]
        ):
            raise ValueError("Default nulling_baseline is outside its allowed range.")
        if imaging_range is not None and not (
            imaging_range[0] <= imaging_baseline <= imaging_range[1]
        ):
            raise ValueError("Default imaging_baseline is outside its allowed range.")

        specs.setdefault("settlingTime", 0.0)
        super().__init__(**specs)

        self.nulling_baseline = baseline
        self.baseline_ratio = ratio
        self.imaging_baseline = imaging_baseline
        self.nulling_baseline_range = nulling_range
        self.imaging_baseline_range = imaging_range

        self._outspec["nulling_baseline"] = baseline.to_value(u.m)
        self._outspec["baseline_ratio"] = ratio
        self._outspec["nulling_baseline_range"] = (
            None
            if nulling_range is None
            else [limit.to_value(u.m) for limit in nulling_range]
        )
        self._outspec["imaging_baseline_range"] = (
            None
            if imaging_range is None
            else [limit.to_value(u.m) for limit in imaging_range]
        )

    def get_collector_positions(
        self,
        rotation_angles: u.Quantity = 0.0 * u.deg,
        *,
        nulling_baseline: float | u.Quantity | None = None,
        baseline_ratio: float | None = None,
    ) -> u.Quantity:
        """Return ordered four-collector coordinates for requested configurations.

        Coordinates are projected onto the target tangent sky plane. The origin
        is the array centroid. Positive rotations are counterclockwise about
        the line of sight. The x-axis spans the nulling baseline; the y-axis
        spans the imaging baseline. No physical maneuver time is calculated.

        Args:
            rotation_angles (astropy.units.Quantity):
                Scalar or nonempty 1D array of rotation angles.
            nulling_baseline (float or astropy.units.Quantity, optional):
                Baseline used in this calculation, or the configured default.
            baseline_ratio (float, optional):
                Imaging-to-nulling ratio used in this calculation, or default.

        Returns:
            astropy.units.Quantity:
                Collector positions with shape (n_rotations, 4, 2), in metres.

        Raises:
            TypeError:
                If arguments have unsupported types.
            ValueError:
                If arguments are invalid or requested baselines exceed limits.
        """
        if not isinstance(rotation_angles, u.Quantity):
            raise TypeError("rotation_angles must be an Astropy Quantity.")
        try:
            angles = np.atleast_1d(rotation_angles.to_value(u.rad))
        except u.UnitConversionError as exc:
            raise ValueError("rotation_angles must have angular units.") from exc
        if angles.ndim != 1 or angles.size == 0 or not np.all(np.isfinite(angles)):
            raise ValueError("rotation_angles must be a nonempty, finite 1D array.")

        baseline = (
            self.nulling_baseline
            if nulling_baseline is None
            else self._positive_length(nulling_baseline, "nulling_baseline")
        )
        ratio = (
            self.baseline_ratio
            if baseline_ratio is None
            else self._positive_ratio(baseline_ratio)
        )
        imaging_baseline = baseline * ratio

        if self.nulling_baseline_range is not None and not (
            self.nulling_baseline_range[0] <= baseline <= self.nulling_baseline_range[1]
        ):
            raise ValueError("Requested nulling baseline is outside its allowed range.")
        if self.imaging_baseline_range is not None and not (
            self.imaging_baseline_range[0]
            <= imaging_baseline
            <= self.imaging_baseline_range[1]
        ):
            raise ValueError("Requested imaging baseline is outside its allowed range.")

        half_null = baseline.to_value(u.m) / 2.0
        half_imaging = imaging_baseline.to_value(u.m) / 2.0
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
