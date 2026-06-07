import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------
# Far-field criteria for a line-source antenna
# ---------------------------------------------------------

# Electrical size: x = D/lambda
x = np.logspace(-2, 2, 1000)

# Normalized far-field distances: r/lambda
r_rayleigh = 2 * x**2          # r/lambda = 2(D/lambda)^2
r_size = 5 * x                 # r/lambda = 5(D/lambda)
r_wavelength = 1.6 * np.ones_like(x)  # r/lambda = 1.6

# Limiting envelope: required far-field distance
r_envelope = np.maximum.reduce([r_rayleigh, r_size, r_wavelength])

# ---------------------------------------------------------
# Plot
# ---------------------------------------------------------

plt.figure(figsize=(8, 5.2))

plt.loglog(x, r_rayleigh, label=r"$r/\lambda = 2(D/\lambda)^2$")
plt.loglog(x, r_size, label=r"$r/\lambda = 5(D/\lambda)$")
plt.loglog(x, r_wavelength, label=r"$r/\lambda = 1.6$")
plt.loglog(x, r_envelope, "k--", linewidth=2.0, label="Limiting envelope")

# Shade far-field region above the envelope
plt.fill_between(
    x,
    r_envelope,
    1e4,
    alpha=0.18,
    label="Far-field region"
)

plt.xlabel(r"Electrical size $D/\lambda$")
plt.ylabel(r"Normalized distance $r/\lambda$")
plt.title("Far-field criteria for a line-source antenna")

plt.xlim(1e-2, 1e2)
plt.ylim(1e-2, 1e4)

plt.grid(True, which="both", linestyle=":", linewidth=0.7)
plt.legend(loc="upper left")

plt.tight_layout()



# Output path
output_path = Path(r".\chapters\chapter-02-antenna-fundamentals\figures\far_field_criteria_line_source.png")

# Create folder if it does not exist
output_path.parent.mkdir(parents=True, exist_ok=True)

# Save figure
plt.savefig(output_path, dpi=300, bbox_inches="tight")

plt.show()