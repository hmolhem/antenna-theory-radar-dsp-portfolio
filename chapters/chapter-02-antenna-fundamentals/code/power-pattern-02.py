"""
Problem 2.5-2: Left and right half-power points of cosine power patterns.

This script analyzes the normalized power pattern

    P(theta) = cos^n(theta)

around the main-beam direction using a signed angular coordinate

    psi in [-90 deg, +90 deg]

where

    psi = 0 deg

is the main-beam axis.

The purpose is to visualize:

    1. Main beam direction
    2. Left half-power point
    3. Right half-power point
    4. Half-power beamwidth (HPBW)

The figures are saved to

    .\\chapters\\chapter-02-antenna-fundamentals\\figures

relative to the repository root.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# Output folder
# =========================================================

FIGURE_DIR = Path(
    r".\chapters\chapter-02-antenna-fundamentals\figures"
)

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# Data structure
# =========================================================

@dataclass
class BeamMetrics:
    """
    Beam metrics for the cosine power pattern.

    Attributes
    ----------
    n : int
        Exponent in the power pattern P(psi) = cos^n(psi).

    directivity_linear : float
        Directivity in linear scale.

    directivity_dbi : float
        Directivity in dBi.

    theta_hp_deg : Optional[float]
        One-sided half-power angle in degrees.

    left_hp_deg : Optional[float]
        Left half-power point in signed degrees.

    right_hp_deg : Optional[float]
        Right half-power point in signed degrees.

    hpbw_deg : Optional[float]
        Half-power beamwidth in degrees.
    """

    n: int
    directivity_linear: float
    directivity_dbi: float
    theta_hp_deg: Optional[float]
    left_hp_deg: Optional[float]
    right_hp_deg: Optional[float]
    hpbw_deg: Optional[float]


# =========================================================
# Pattern and metric functions
# =========================================================

def signed_power_pattern(psi_rad: np.ndarray, n: int) -> np.ndarray:
    """
    Compute the signed-angle normalized power pattern.

    The original pattern is

        P(theta) = cos^n(theta),  0 <= theta <= pi/2.

    To show the left and right sides of the main beam, we use

        psi in [-pi/2, +pi/2],

    where psi = 0 is the main-beam direction.

    Parameters
    ----------
    psi_rad : np.ndarray
        Signed angle from the main-beam axis in radians.

    n : int
        Nonnegative exponent.

    Returns
    -------
    np.ndarray
        Normalized power pattern.
    """

    if n < 0:
        raise ValueError("n must be nonnegative.")

    p = np.zeros_like(psi_rad, dtype=float)

    visible_region = np.abs(psi_rad) <= np.pi / 2

    if n == 0:
        p[visible_region] = 1.0
    else:
        p[visible_region] = np.cos(psi_rad[visible_region]) ** n

    return p


def directivity_linear(n: int) -> float:
    """
    Compute the directivity of the power pattern.

    For

        P(theta) = cos^n(theta),  0 <= theta <= pi/2,

    the beam solid angle is

        Omega_A = 2*pi/(n + 1),

    and therefore

        D = 4*pi/Omega_A = 2(n + 1).

    Parameters
    ----------
    n : int
        Nonnegative exponent.

    Returns
    -------
    float
        Directivity in linear scale.
    """

    if n < 0:
        raise ValueError("n must be nonnegative.")

    return 2.0 * (n + 1)


def half_power_angle_deg(n: int) -> Optional[float]:
    """
    Compute the one-sided half-power angle.

    The half-power condition is

        cos^n(theta_HP) = 1/2.

    Thus,

        theta_HP = arccos(2^(-1/n)).

    For n = 0, the pattern is constant over the forward hemisphere,
    so a conventional half-power point does not exist.

    Parameters
    ----------
    n : int
        Nonnegative exponent.

    Returns
    -------
    Optional[float]
        One-sided half-power angle in degrees.
        Returns None for n = 0.
    """

    if n < 0:
        raise ValueError("n must be nonnegative.")

    if n == 0:
        return None

    theta_hp_rad = np.arccos(2.0 ** (-1.0 / n))
    return float(np.degrees(theta_hp_rad))


def compute_beam_metrics(n: int) -> BeamMetrics:
    """
    Compute directivity, half-power points, and HPBW.

    Parameters
    ----------
    n : int
        Nonnegative exponent.

    Returns
    -------
    BeamMetrics
        Calculated beam metrics.
    """

    directivity = directivity_linear(n)
    directivity_dbi = 10.0 * np.log10(directivity)

    theta_hp = half_power_angle_deg(n)

    if theta_hp is None:
        left_hp = None
        right_hp = None
        hpbw = None
    else:
        left_hp = -theta_hp
        right_hp = theta_hp
        hpbw = right_hp - left_hp

    return BeamMetrics(
        n=n,
        directivity_linear=directivity,
        directivity_dbi=directivity_dbi,
        theta_hp_deg=theta_hp,
        left_hp_deg=left_hp,
        right_hp_deg=right_hp,
        hpbw_deg=hpbw,
    )


# =========================================================
# Reporting
# =========================================================

def print_beam_metrics_table(n_values: Sequence[int]) -> None:
    """
    Print a formatted table of beam metrics.

    Parameters
    ----------
    n_values : Sequence[int]
        Values of n to evaluate.
    """

    print("\nProblem 2.5-2: Left and Right Half-Power Beam Metrics")
    print("-" * 112)
    print(
        f"{'n':>3} | "
        f"{'D':>10} | "
        f"{'D (dBi)':>10} | "
        f"{'Left HP (deg)':>15} | "
        f"{'Right HP (deg)':>16} | "
        f"{'HPBW (deg)':>12}"
    )
    print("-" * 112)

    for n in n_values:
        metrics = compute_beam_metrics(n)

        left_text = (
            f"{metrics.left_hp_deg:.2f}"
            if metrics.left_hp_deg is not None
            else "N/A"
        )

        right_text = (
            f"{metrics.right_hp_deg:.2f}"
            if metrics.right_hp_deg is not None
            else "N/A"
        )

        hpbw_text = (
            f"{metrics.hpbw_deg:.2f}"
            if metrics.hpbw_deg is not None
            else "N/A"
        )

        print(
            f"{metrics.n:>3} | "
            f"{metrics.directivity_linear:>10.3f} | "
            f"{metrics.directivity_dbi:>10.2f} | "
            f"{left_text:>15} | "
            f"{right_text:>16} | "
            f"{hpbw_text:>12}"
        )

    print("-" * 112)


# =========================================================
# Figure saving
# =========================================================

def save_current_figure(output_path: Path) -> None:
    """
    Save the current Matplotlib figure.

    Parameters
    ----------
    output_path : Path
        Full output path for the figure.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {output_path}")


# =========================================================
# Plot functions
# =========================================================

def plot_left_right_beam_for_single_n(n: int) -> None:
    """
    Plot the signed-angle beam for one value of n.

    The plot shows:
        - main beam at psi = 0 deg
        - left half-power point
        - right half-power point
        - HPBW between the two half-power points

    Parameters
    ----------
    n : int
        Nonnegative exponent.
    """

    psi_deg = np.linspace(-90.0, 90.0, 2000)
    psi_rad = np.radians(psi_deg)

    pattern = signed_power_pattern(psi_rad, n)
    metrics = compute_beam_metrics(n)

    plt.figure(figsize=(9.0, 5.8))

    plt.plot(
        psi_deg,
        pattern,
        linewidth=2.5,
        label=rf"$P(\psi)=\cos^{n}(\psi)$"
    )

    # Half-power level
    plt.axhline(
        0.5,
        linestyle="--",
        linewidth=1.2,
        label="Half-power level"
    )

    # Main-beam axis
    plt.axvline(
        0.0,
        linestyle=":",
        linewidth=1.2,
        label="Main-beam axis"
    )

    # Main-beam peak
    plt.plot(
        [0.0],
        [1.0],
        marker="o",
        markersize=7,
        label="Main beam"
    )

    plt.text(
        3.0,
        0.95,
        "Main beam\n0 deg",
        fontsize=10
    )

    if metrics.left_hp_deg is not None and metrics.right_hp_deg is not None:
        # Left and right half-power points
        plt.plot(
            [metrics.left_hp_deg],
            [0.5],
            marker="o",
            markersize=7,
            label="Left HP point"
        )

        plt.plot(
            [metrics.right_hp_deg],
            [0.5],
            marker="o",
            markersize=7,
            label="Right HP point"
        )

        # Vertical reference lines at half-power points
        plt.axvline(
            metrics.left_hp_deg,
            linestyle=":",
            linewidth=1.0
        )

        plt.axvline(
            metrics.right_hp_deg,
            linestyle=":",
            linewidth=1.0
        )

        # HPBW line segment
        plt.plot(
            [metrics.left_hp_deg, metrics.right_hp_deg],
            [0.5, 0.5],
            linewidth=3.0,
            label=rf"HPBW = {metrics.hpbw_deg:.2f} deg"
        )

        plt.text(
            metrics.left_hp_deg - 4.0,
            0.58,
            f"Left HP\n{metrics.left_hp_deg:.2f} deg",
            fontsize=9,
            ha="right"
        )

        plt.text(
            metrics.right_hp_deg + 4.0,
            0.58,
            f"Right HP\n{metrics.right_hp_deg:.2f} deg",
            fontsize=9,
            ha="left"
        )

        plt.text(
            0.0,
            0.38,
            f"HPBW = {metrics.hpbw_deg:.2f} deg",
            fontsize=10,
            ha="center",
            bbox=dict(boxstyle="round", fc="white", ec="gray")
        )

    else:
        plt.text(
            0.0,
            0.45,
            "n = 0: uniform forward-hemisphere pattern\n"
            "No conventional half-power points",
            fontsize=10,
            ha="center",
            bbox=dict(boxstyle="round", fc="white", ec="gray")
        )

    plt.xlabel(r"Signed angle from main beam, $\psi$ (degrees)")
    plt.ylabel(r"Normalized power pattern, $P(\psi)$")
    plt.title(rf"Left and Right Sides of Main Beam for $n={n}$")

    plt.xlim(-90.0, 90.0)
    plt.ylim(0.0, 1.08)

    plt.grid(True, linestyle=":", linewidth=0.7)
    plt.legend(loc="upper right")
    plt.tight_layout()

    output_path = FIGURE_DIR / f"problem_2_5_2_left_right_beam_n{n}.png"
    save_current_figure(output_path)

    plt.show()


def plot_left_right_beams_combined(n_values: Sequence[int]) -> None:
    """
    Plot signed-angle power patterns for several n values on one Cartesian plot.

    This combined plot is useful for comparing how the main beam becomes narrower
    as n increases.

    Parameters
    ----------
    n_values : Sequence[int]
        Values of n to plot.
    """

    psi_deg = np.linspace(-90.0, 90.0, 2000)
    psi_rad = np.radians(psi_deg)

    plt.figure(figsize=(9.2, 5.8))

    # Half-power level
    plt.axhline(
        0.5,
        linestyle="--",
        linewidth=1.2,
        label="Half-power level"
    )

    # Main-beam axis
    plt.axvline(
        0.0,
        linestyle=":",
        linewidth=1.2,
        label="Main-beam axis"
    )

    for n in n_values:
        pattern = signed_power_pattern(psi_rad, n)
        metrics = compute_beam_metrics(n)

        if metrics.hpbw_deg is None:
            label = rf"$n={n}$, $D={metrics.directivity_linear:.1f}$"
        else:
            label = (
                rf"$n={n}$, "
                rf"$D={metrics.directivity_linear:.1f}$, "
                rf"HPBW={metrics.hpbw_deg:.2f}$^\circ$"
            )

        plt.plot(
            psi_deg,
            pattern,
            linewidth=2.2,
            label=label
        )

        # Mark left/right HP points for n > 0
        if metrics.left_hp_deg is not None and metrics.right_hp_deg is not None:
            plt.plot(
                [metrics.left_hp_deg, metrics.right_hp_deg],
                [0.5, 0.5],
                marker="o",
                linestyle="None",
                markersize=5
            )

    plt.xlabel(r"Signed angle from main beam, $\psi$ (degrees)")
    plt.ylabel(r"Normalized power pattern, $P(\psi)$")
    plt.title(r"Comparison of Left and Right Main-Beam Sides")

    plt.xlim(-90.0, 90.0)
    plt.ylim(0.0, 1.08)

    plt.grid(True, linestyle=":", linewidth=0.7)
    plt.legend(loc="upper right")
    plt.tight_layout()

    output_path = FIGURE_DIR / "problem_2_5_2_left_right_beams_combined.png"
    save_current_figure(output_path)

    plt.show()


# =========================================================
# Main driver
# =========================================================

def main() -> None:
    """
    Main driver for left/right beam visualization.

    Generates:
        1. A printed table of beam metrics.
        2. One combined left/right beam comparison figure.
        3. One detailed left/right beam figure for each n.
    """

    n_values = [0, 1, 2, 3]

    print_beam_metrics_table(n_values)

    plot_left_right_beams_combined(n_values)

    for n in n_values:
        plot_left_right_beam_for_single_n(n)


if __name__ == "__main__":
    main()