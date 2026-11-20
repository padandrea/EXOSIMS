"""Physical Sun-Earth integration check for the LIFE modules."""

import tempfile
import unittest
from types import SimpleNamespace

import astropy.units as u
import numpy as np
from scipy.constants import c, h, k

from EXOSIMS.Observatory.LIFEObservatory import LIFEObservatory
from EXOSIMS.OpticalSystem.LIFEOpticalSystem import LIFEOpticalSystem
from EXOSIMS.SurveySimulation.LIFESurveySimulation import LIFESurveySimulation
from EXOSIMS.ZodiacalLight.LIFEZodiacalLight import LIFEZodiacalLight


def blackbody_flux(wavelengths, temperature, radius, distance):
    """Observed blackbody photon flux density."""
    lam = wavelengths.to_value(u.m)
    r = radius.to_value(u.m)
    d = distance.to_value(u.m)

    flux = (
        2 * np.pi * c
        / (lam**4 * np.expm1(h * c / (lam * k * temperature)))
        * (r / d) ** 2
    )

    return (
        flux * u.photon / (u.s * u.m**2 * u.m)
    ).to(u.photon / (u.s * u.m**2 * u.um))


class TestLIFESunEarthComparison(unittest.TestCase):
    """Exercise the real LIFE detection calculation."""

    def test_sun_earth_physical_pipeline(self):
        distance = 10.0 * u.pc
        star_radius = 1.0 * u.R_sun
        planet_radius = 6378100.0 * u.m

        optical = LIFEOpticalSystem.__new__(LIFEOpticalSystem)
        optical.pupilArea = np.pi * (1.5 * u.m) ** 2
        optical.pupilDiam = 3.0 * u.m

        survey = LIFESurveySimulation.__new__(LIFESurveySimulation)
        survey.OpticalSystem = optical
        survey.ZodiacalLight = LIFEZodiacalLight(verbose=False)
        survey.TargetList = SimpleNamespace(
            L=np.array([1.0]),
            dist=np.array([10.0]) * u.pc,
        )
        survey.SimulatedUniverse = SimpleNamespace(
            plan2star=np.array([0]),
            nEZ=np.array([1.0]),
        )

        wavelengths, widths = optical.get_spectral_bins(
            4.0 * u.um,
            18.5 * u.um,
            10.0 / 0.3,
        )

        mode = {
            "attenuation": 0.05,
            "SNR": 5.0,
            "LIFE_fov_taper": "none",
            "LIFE_fov_boundary": "circle",
            "LIFE_local_zodi_model": "darwinsim",
            "inst": {
                "QE": lambda _: 0.7 / u.photon,
                "idark": 0.0 / u.s,
                "sread": 0.0,
                "texp": 100.0 * u.s,
            },
        }

        sky, pixel_areas = survey.make_life_sky_grid(
            half_width=4.0e-6 * u.rad,
            pixels_per_axis=64,
        )

        rotation_angles = (
            np.linspace(0.0, 2.0 * np.pi, 180, endpoint=False) * u.rad
        )

        with tempfile.TemporaryDirectory() as cachedir:
            survey.Observatory = LIFEObservatory(
                nulling_baseline=14.5,
                baseline_ratio=6.0,
                cachedir=cachedir,
                forceStaticEphem=True,
                verbose=False,
            )

            result = survey.get_life_detection_metrics(
                sInd=0,
                wavelengths=wavelengths,
                bin_widths=widths,
                planet_sky_positions=np.array([
                    [(1.0 * u.au / distance).decompose().value, 0.0]
                ]) * u.rad,
                planet_spectral_fluxes=blackbody_flux(
                    wavelengths, 254.0, planet_radius, distance
                )[None, :],
                stellar_angular_radius=(
                    (star_radius / distance).decompose().value * u.rad
                ),
                stellar_spectral_flux=blackbody_flux(
                    wavelengths, 5778.0, star_radius, distance
                ),
                sky_positions=sky,
                pixel_solid_angles=pixel_areas,
                rotation_angles=rotation_angles,
                mode=mode,
                local_zodi_longitude=135.0 * u.deg,
                local_zodi_latitude=45.0 * u.deg,
                integration_time=16.0 * u.day,
                detector_pixels_per_output=1.0,
                nEZ=1.0,
                nulling_baseline=14.5 * u.m,
            )

        snr = float(np.asarray(result["snr"]).item())
        time_h = float(
            result["required_integration_time"].to_value(u.h)[0]
        )

        for label, left_key, right_key in (
            ("Planet", "planet_left", "planet_right"),
            ("Stellar leakage", "stellar_left", "stellar_right"),
            ("Local zodi", "local_left", "local_right"),
            ("Exozodi", "exozodi_left", "exozodi_right"),
        ):
            left = result[left_key]
            right = result[right_key]
            if left.ndim == 3:
                left = left[:, 0, :]
                right = right[:, 0, :]
            mean_rate = (0.5 * (left + right)).sum(axis=-1).mean()
            print(f"{label}: {mean_rate.to_value(1 / u.s):.2f} 1/s")

        # Diagnostic: rotation-averaged noise estimator.
        # Compare with the existing inverse-variance SNR.
        unit = 1 / u.s

        planet_left = result["planet_left"][:, 0, :].to_value(unit)
        planet_right = result["planet_right"][:, 0, :].to_value(unit)

        chopped = planet_left - planet_right
        noise = planet_left + planet_right

        for key in (
            "stellar_left", "stellar_right",
            "local_left", "local_right",
            "exozodi_left", "exozodi_right",
        ):
            noise = noise + result[key].to_value(unit)

        snr_mean_noise = np.sqrt(
            (16.0 * u.day).to_value(u.s)
            * np.sum(
                np.mean(chopped**2, axis=0)
                / np.mean(noise, axis=0)
            )
        )

        print(f"Rotation-averaged SNR: {snr_mean_noise:.6f}")
        print(f"Inverse-variance SNR: {snr:.6f}")

        # Spectral diagnostic: individual photon rates and SNR contributions.
        from pathlib import Path

        rate_unit = 1 / u.s
        n_config = result["planet_left"].shape[0]
        n_wavelengths = len(wavelengths)

        signal = result["signal_rates"][0].to_value(rate_unit).reshape(
            n_config, n_wavelengths
        )
        variance = result["noise_variance_rates"][0].to_value(
            rate_unit
        ).reshape(n_config, n_wavelengths)

        snr2_per_wavelength = (
            (16 * u.day).to_value(u.s)
            * np.sum(
                np.divide(
                    signal**2,
                    variance,
                    out=np.zeros_like(signal),
                    where=variance > 0,
                ),
                axis=0,
            )
        )

        snr_reconstructed = np.sqrt(snr2_per_wavelength.sum())
        np.testing.assert_allclose(snr_reconstructed, snr, rtol=1e-10)

        planet_left = result["planet_left"][:, 0, :].to_value(rate_unit)
        planet_right = result["planet_right"][:, 0, :].to_value(rate_unit)

        def combined_rate(name):
            left = result[f"{name}_left"].to_value(rate_unit)
            right = result[f"{name}_right"].to_value(rate_unit)
            return (left + right).mean(axis=0)

        table = np.column_stack(
            (
                wavelengths.to_value(u.um),
                widths.to_value(u.um),
                np.sqrt(np.mean((planet_left - planet_right)**2, axis=0)),
                (planet_left + planet_right).mean(axis=0),
                combined_rate("stellar"),
                combined_rate("local"),
                combined_rate("exozodi"),
                snr2_per_wavelength,
            )
        )

        output = Path("/tmp/life_exosims_spectral.csv")
        np.savetxt(
            output,
            table,
            delimiter=",",
            header=(
                "wavelength_um,bin_width_um,"
                "planet_chopped_rms_per_s,planet_noise_rate_per_s,"
                "stellar_noise_rate_per_s,local_noise_rate_per_s,"
                "exozodi_noise_rate_per_s,snr2_16days"
            ),
            comments="",
            fmt="%.12g",
        )

        print(f"Spectral bins exported: {len(table)}")
        print(f"Reconstructed SNR: {snr_reconstructed:.6f}")
        print(f"Spectral CSV: {output}")

        print(f"\nIntegrated LIFE-EXOSIMS SNR (16 days): {snr:.6f}")
        print(f"Integrated LIFE-EXOSIMS t5: {time_h:.6f} h")

        self.assertEqual(len(wavelengths), 52)
        self.assertTrue(np.isfinite(snr))
        self.assertGreater(snr, 0.0)
        self.assertTrue(np.isfinite(time_h))
        self.assertGreater(time_h, 0.0)

        expected_snr = 5.0 * np.sqrt(16.0 * 24.0 / time_h)
        self.assertAlmostEqual(snr, expected_snr, places=8)


if __name__ == "__main__":
    unittest.main()
