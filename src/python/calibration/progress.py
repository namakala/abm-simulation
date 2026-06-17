"""
Progress reporter for calibration runs.

Provides a lightweight CalibrationProgress class that prints per-iteration
progress with timestamps to stderr.
"""

import sys
from datetime import datetime


class CalibrationProgress:
    """Prints per-iteration progress during calibration.

    Parameters
    ----------
    total : int
        Total number of iterations expected.
    label : str, optional
        A descriptive label (default: "Calibration").
    """

    def __init__(self, total: int, label: str = "Calibration") -> None:
        self.total = total
        self.label = label

    def update(self, iteration: int, mean: float, std: float, *, converged: bool = False) -> None:
        """Print a progress line for the current iteration to stderr.

        Parameters
        ----------
        iteration : int
            Current iteration number (1-based).
        mean : float
            Current PSS-10 population mean.
        std : float
            Current PSS-10 population standard deviation.
        converged : bool, optional
            Whether the calibration has converged.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status = " (converged)" if converged else ""
        line = f"[{timestamp}] {self.label} iteration {iteration}/{self.total} | mean={mean:.1f}, std={std:.1f}{status}"
        print(line, file=sys.stderr, flush=True)

    def close(self) -> None:
        """Finalise the progress reporter (no-op for now)."""
        pass
