import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


# =========================================================
# Output path
# =========================================================
# Run this script from the repository root:
# C:\MyDocument\antenna-theory-radar-dsp-portfolio

FIGURE_DIR = Path(
    r".\chapters\chapter-02-antenna-fundamentals\figures"
)

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# Beam metrics data structure
# =========================================================

@dataclass
class BeamMetrics:
    n: int
    directivity_linear: float
    directivity_dBi: float
    theta_hp_deg: Optional[float]
    hpbw_deg: Optional[float]


# =========================================================
# Antenna pattern functions
# =========================================================

def power_pattern(theta_rad: np.ndarray, n: int) -> np.ndarray:
    """
    Compute the normalized power pattern

        P(theta) = cos^n(theta),  0 <= theta <= pi/2
                 = 0,             pi/2 < theta <= pi

    Parameters
    ----------
    theta_rad : np.ndarray
        Angle theta in radians.

    n : int
        Exponent of the cosine power pattern.

    Returns
    -------
    np.ndarray
        Normalized power pattern.
    """

    if n < 0:
        raise ValueError("n must be nonnegative.")

    p = np.zeros_like(theta_rad, dtype=float)

    forward_region = (theta_rad >= 0.0) & (theta_rad <= np.pi / 2)

    if n == 0:
        p[forward_region] = 1.0
    else:
        p[forward_region] = np.cos(theta_rad[forward_region]) ** n

    return p


def directivity(n: int) -> float:
    """
    Compute directivity for the power pattern

        P(theta) = cos^n(theta),  0 <= theta <= pi/2.

    The beam solid angle is

        Omega_A = 2*pi/(n+1)

    so

        D = 4*pi/Omega_A = 2(n+1).

    Parameters
    ----------
    n : int
        Exponent of the cosine power pattern.

    Returns
    -------
    float
        Directivity in linear scale.
    """

    if n < 0:
        raise ValueError("n must be nonnegative.")

    return 2.0 * (n + 1)


def half_power_angle(n: int) -> Optional[float]:
    """
    Compute the half-power angle in degrees.

    The half-power condition is

        cos^n(theta_HP) = 1/2.

    Therefore,

        theta_HP = arccos(2^(-1/n)).

    For n = 0, the pattern is constant over the forward hemisphere,
    so no conventional half-power point exists.

    Parameters
    ----------
    n : int
        Exponent of the cosine power pattern.

    Returns
    -------
    Optional[float]
        Half-power angle in degrees. Returns None for n = 0.
    """

    if n < 0:
        raise ValueError("n must be nonnegative.")

    if n == 0:
        return None

    theta_hp_rad = np.arccos(2.0 ** (-1.0 / n))
    return np.degrees(theta_hp_rad)


def compute_beam_metrics(n: int) -> BeamMetrics:
    """
    Compute directivity, directivity in dBi, half-power angle,
    and half-power beamwidth.

    Parameters
    ----------
    n : int
        Exponent of the cosine power pattern.

    Returns
    -------
    BeamMetrics
        Beam metrics for the selected n.
    """

    D = directivity(n)
    D_dBi = 10.0 * np.log10(D)

    theta_hp_deg = half_power_angle(n)

    if theta_hp_deg is None:
        hpbw = None
    else:
        hpbw = 2.0 * theta_hp_deg

    return BeamMetrics(
        n=n,
        directivity_linear=D,
        directivity_dBi=D_dBi,
        theta_hp_deg=theta_hp_deg,
        hpbw_deg=hpbw,
    )


# =========================================================
# Reporting function
# =========================================================

def print_metrics_table(n_values: list[int]) -> None:
    """
    Print a formatted table of beam metrics.

    Parameters
    ----------
    n_values : list[int]
        List of n values.
    """

    print("\nProblem 2.5-2 Beam Metrics")
    print("-" * 90)
    print(
        f"{'n':>3} | {'D':>10} | {'D (dBi)':>10} | "
        f"{'theta_HP (deg)':>16} | {'HPBW (deg)':>12}"
    )
    print("-" * 90)

    for n in n_values:
        metrics = compute_beam_metrics(n)

        theta_hp_text = (
            f"{metrics.theta_hp_deg:.2f}"
            if metrics.theta_hp_deg is not None
            else "N/A"
        )

        hpbw_text = (
            f"{metrics.hpbw_deg:.2f}"
            if metrics.hpbw_deg is not None
            else "N/A"
        )

        print(
            f"{n:>3} | "
            f"{metrics.directivity_linear:>10.3f} | "
            f"{metrics.directivity_dBi:>10.2f} | "
            f"{theta_hp_text:>16} | "
            f"{hpbw_text:>12}"
        )

    print("-" * 90)


# =========================================================
# Plotting functions
# =========================================================

def save_figure(path: Path) -> None:
    """
    Save the current Matplotlib figure.

    Parameters
    ----------
    path : Path
        Output file path.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {path}")


def plot_combined_patterns(n_values: list[int]) -> None:
    """
    Plot all power patterns on one polar plot and save the figure.

    Parameters
    ----------
    n_values : list[int]
        List of n values to plot.
    """

    theta = np.linspace(0.0, np.pi, 2000)

    fig, ax = plt.subplots(
        subplot_kw={"projection": "polar"},
        figsize=(9, 7)
    )

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetamin(0)
    ax.set_thetamax(180)

    ax.set_title(
        r"Power Patterns for $P(\theta)=\cos^n\theta$",
        pad=20
    )

    # Half-power reference circle
    ax.plot(
        theta,
        0.5 * np.ones_like(theta),
        linestyle="--",
        linewidth=1.0,
        label="Half-power level"
    )

    for n in n_values:
        p = power_pattern(theta, n)
        metrics = compute_beam_metrics(n)

        label = (
            f"n={n}, "
            f"D={metrics.directivity_linear:.1f}, "
            f"D={metrics.directivity_dBi:.2f} dBi"
        )

        ax.plot(theta, p, linewidth=2.0, label=label)

        # Main beam marker
        ax.plot([0.0], [1.0], marker="o", markersize=5)

        # Half-power marker
        if metrics.theta_hp_deg is not None:
            theta_hp_rad = np.radians(metrics.theta_hp_deg)
            ax.plot([theta_hp_rad], [0.5], marker="o", markersize=5)

    ax.set_rmax(1.05)
    ax.set_rticks([0.25, 0.5, 0.75, 1.0])
    ax.grid(True)
    ax.legend(loc="lower left", bbox_to_anchor=(1.05, 0.05))

    plt.tight_layout()

    output_path = FIGURE_DIR / "problem_2_5_2_power_patterns_all.png"
    save_figure(output_path)

    plt.show()


def plot_detailed_pattern(n: int) -> None:
    """
    Plot one power pattern with main beam and half-power annotation.

    Parameters
    ----------
    n : int
        Exponent of the cosine power pattern.
    """

    theta = np.linspace(0.0, np.pi, 2000)
    p = power_pattern(theta, n)
    metrics = compute_beam_metrics(n)

    fig, ax = plt.subplots(
        subplot_kw={"projection": "polar"},
        figsize=(8, 7)
    )

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetamin(0)
    ax.set_thetamax(180)

    ax.set_title(
        rf"Detailed Beam Plot for $n={n}$",
        pad=20
    )

    ax.plot(theta, p, linewidth=2.5, label=rf"$n={n}$")

    # Half-power level
    ax.plot(
        theta,
        0.5 * np.ones_like(theta),
        linestyle="--",
        linewidth=1.0,
        label="Half-power level"
    )

    # Main beam marker
    ax.plot([0.0], [1.0], marker="o", markersize=7, label="Main beam")

    ax.annotate(
        "Main beam\n0 deg",
        xy=(0.0, 1.0),
        xytext=(np.radians(18), 0.95),
        textcoords="data"
    )

    if metrics.theta_hp_deg is not None:
        theta_hp_rad = np.radians(metrics.theta_hp_deg)

        ax.plot(
            [theta_hp_rad],
            [0.5],
            marker="o",
            markersize=7,
            label="Half-power point"
        )

        ax.annotate(
            f"HP point\n{metrics.theta_hp_deg:.2f} deg",
            xy=(theta_hp_rad, 0.5),
            xytext=(theta_hp_rad + np.radians(15), 0.65),
            textcoords="data"
        )

        ax.text(
            np.radians(105),
            0.90,
            f"HPBW = {metrics.hpbw_deg:.2f} deg",
            fontsize=10,
            bbox=dict(boxstyle="round", fc="white", ec="gray")
        )

    else:
        ax.text(
            np.radians(100),
            0.85,
            "n = 0\nUniform forward hemisphere\nNo conventional HP point",
            fontsize=10,
            bbox=dict(boxstyle="round", fc="white", ec="gray")
        )

    ax.set_rmax(1.05)
    ax.set_rticks([0.25, 0.5, 0.75, 1.0])
    ax.grid(True)
    ax.legend(loc="lower left", bbox_to_anchor=(1.05, 0.05))

    plt.tight_layout()

    output_path = FIGURE_DIR / f"problem_2_5_2_power_pattern_n{n}.png"
    save_figure(output_path)

    plt.show()


# =========================================================
# Main program
# =========================================================

def main() -> None:
    """
    Main driver for Problem 2.5-2.

    This program computes and plots the power patterns for

        n = 0, 1, 2, 3.

    All figures are saved to

        chapters/chapter-02-antenna-fundamentals/figures
    """

    n_values = [0, 1, 2, 3]

    print_metrics_table(n_values)

    plot_combined_patterns(n_values)

    for n in n_values:
        plot_detailed_pattern(n)


if __name__ == "__main__":
    main()