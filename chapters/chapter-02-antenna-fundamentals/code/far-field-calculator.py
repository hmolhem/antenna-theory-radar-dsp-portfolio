"""
Far-field distance calculator for antenna problems.

This script computes the far-field distance using the three criteria

    r_ff > 2D^2 / lambda
    r_ff > 5D
    r_ff > 1.6 lambda

where
    D      = largest antenna dimension
    lambda = wavelength

The required far-field distance is the maximum of the three criteria.

Test case:
Problem 2.4-10:
A 31-in. fender-mount car radio antenna at 1 MHz.
"""

from dataclasses import dataclass
from typing import Optional


C0 = 299_792_458.0  # Speed of light in free space, m/s


@dataclass
class FarFieldResult:
    """
    Container for far-field calculation results.

    Attributes
    ----------
    D_m : float
        Largest physical dimension of the antenna in meters.

    wavelength_m : float
        Operating wavelength in meters.

    electrical_size : float
        Electrical size D/lambda.

    rayleigh_distance_m : float
        Rayleigh far-field criterion: 2D^2/lambda.

    size_distance_m : float
        Size-based far-field criterion: 5D.

    wavelength_distance_m : float
        Wavelength-based far-field criterion: 1.6 lambda.

    r_ff_m : float
        Required far-field distance.

    limiting_criterion : str
        Criterion that controls the far-field distance.

    regime : str
        Electrical-size regime based on D/lambda.
    """

    D_m: float
    wavelength_m: float
    electrical_size: float
    rayleigh_distance_m: float
    size_distance_m: float
    wavelength_distance_m: float
    r_ff_m: float
    limiting_criterion: str
    regime: str


def wavelength_from_frequency(
    frequency_hz: float,
    wave_speed_mps: float = C0
) -> float:
    """
    Compute wavelength from frequency.

    Parameters
    ----------
    frequency_hz : float
        Frequency in hertz.

    wave_speed_mps : float
        Wave propagation speed in meters per second.

    Returns
    -------
    float
        Wavelength in meters.
    """

    if frequency_hz <= 0:
        raise ValueError("Frequency must be positive.")

    return wave_speed_mps / frequency_hz


def classify_electrical_size(D_over_lambda: float) -> str:
    """
    Classify the antenna electrical-size regime.

    The controlling far-field criterion changes approximately at

        D/lambda = 0.32
        D/lambda = 2.5

    Parameters
    ----------
    D_over_lambda : float
        Electrical size of the antenna.

    Returns
    -------
    str
        Description of the electrical-size regime.
    """

    if D_over_lambda < 0:
        raise ValueError("Electrical size must be nonnegative.")

    if D_over_lambda < 0.32:
        return "electrically small: 1.6 lambda criterion controls"

    if D_over_lambda <= 2.5:
        return "intermediate size: 5D criterion controls"

    return "electrically large: 2D^2/lambda criterion controls"


def compute_far_field_distance(
    D_m: float,
    wavelength_m: Optional[float] = None,
    frequency_hz: Optional[float] = None,
    wave_speed_mps: float = C0
) -> FarFieldResult:
    """
    Compute the required far-field distance for an antenna.

    Either wavelength_m or frequency_hz must be provided. If wavelength_m
    is not provided, it is computed from frequency_hz.

    Parameters
    ----------
    D_m : float
        Largest physical dimension of the antenna in meters.

    wavelength_m : float, optional
        Wavelength in meters.

    frequency_hz : float, optional
        Frequency in hertz.

    wave_speed_mps : float
        Wave propagation speed in meters per second.

    Returns
    -------
    FarFieldResult
        Dataclass containing the three criteria and the limiting condition.
    """

    if D_m <= 0:
        raise ValueError("Antenna dimension D_m must be positive.")

    if wavelength_m is None:
        if frequency_hz is None:
            raise ValueError(
                "Either wavelength_m or frequency_hz must be provided."
            )
        wavelength_m = wavelength_from_frequency(
            frequency_hz,
            wave_speed_mps
        )

    if wavelength_m <= 0:
        raise ValueError("Wavelength must be positive.")

    D_over_lambda = D_m / wavelength_m

    rayleigh_distance_m = 2.0 * D_m**2 / wavelength_m
    size_distance_m = 5.0 * D_m
    wavelength_distance_m = 1.6 * wavelength_m

    criteria = {
        "Rayleigh criterion: 2D^2/lambda": rayleigh_distance_m,
        "Size criterion: 5D": size_distance_m,
        "Wavelength criterion: 1.6 lambda": wavelength_distance_m,
    }

    limiting_criterion = max(criteria, key=criteria.get)
    r_ff_m = criteria[limiting_criterion]

    regime = classify_electrical_size(D_over_lambda)

    return FarFieldResult(
        D_m=D_m,
        wavelength_m=wavelength_m,
        electrical_size=D_over_lambda,
        rayleigh_distance_m=rayleigh_distance_m,
        size_distance_m=size_distance_m,
        wavelength_distance_m=wavelength_distance_m,
        r_ff_m=r_ff_m,
        limiting_criterion=limiting_criterion,
        regime=regime
    )


def print_far_field_report(result: FarFieldResult) -> None:
    """
    Print a formatted far-field calculation report.

    Parameters
    ----------
    result : FarFieldResult
        Output of compute_far_field_distance().
    """

    print("Far-Field Distance Report")
    print("-" * 60)
    print(f"Antenna maximum dimension, D      = {result.D_m:.6g} m")
    print(f"Wavelength, lambda                = {result.wavelength_m:.6g} m")
    print(f"Electrical size, D/lambda         = {result.electrical_size:.6g}")
    print()
    print("Far-field criteria:")
    print(f"  2D^2/lambda                     = {result.rayleigh_distance_m:.6g} m")
    print(f"  5D                              = {result.size_distance_m:.6g} m")
    print(f"  1.6 lambda                      = {result.wavelength_distance_m:.6g} m")
    print()
    print(f"Required far-field distance, r_ff = {result.r_ff_m:.6g} m")
    print(f"Limiting criterion                = {result.limiting_criterion}")
    print(f"Electrical-size regime            = {result.regime}")
    print("-" * 60)


def run_problem_2_4_10_test() -> None:
    """
    Test case based on Problem 2.4-10.

    Given:
        D = 31 in.
        f = 1 MHz

    Conversion:
        1 inch = 0.0254 m
    """

    D_inch = 31.0
    D_m = D_inch * 0.0254
    frequency_hz = 1.0e6

    result = compute_far_field_distance(
        D_m=D_m,
        frequency_hz=frequency_hz
    )

    print_far_field_report(result)


if __name__ == "__main__":
    run_problem_2_4_10_test()