"""Real-time smoothing for the live aim point.

Two strategies are available:

- ``EMASmoother`` applies a single-pole IIR low-pass filter to each axis.
  No external dependencies; very low latency.
- ``SavGolSmoother`` fits a low-degree polynomial across a rolling window,
  preserving the shape of genuine movement while suppressing tremor.
  Falls back to EMA when SciPy is unavailable.

Both expose the same ``update(aim) -> aim`` interface and are designed to
be called once per camera frame.
"""

from __future__ import annotations

from collections import deque
from typing import Optional, Protocol, Tuple

import numpy as np


Aim = Tuple[float, float]


class Smoother(Protocol):
    """Minimal interface every smoother implements."""

    def update(self, aim: Aim) -> Aim: ...

    def reset(self) -> None: ...


class EMASmoother:
    """Independent exponential moving average on each axis.

    ``alpha`` controls responsiveness: 1.0 disables smoothing, lower
    values smooth more heavily at the cost of latency.
    """

    def __init__(self, alpha: float = 0.35):
        self.alpha = max(0.05, min(1.0, alpha))
        self._sx: Optional[float] = None
        self._sy: Optional[float] = None

    def reset(self) -> None:
        self._sx = None
        self._sy = None

    def update(self, aim: Aim) -> Aim:
        x, y = aim
        if self._sx is None:
            self._sx, self._sy = x, y
        else:
            a = self.alpha
            self._sx = a * x + (1.0 - a) * self._sx
            self._sy = a * y + (1.0 - a) * self._sy
        return self._sx, self._sy


class SavGolSmoother:
    """Rolling Savitzky-Golay filter with EMA fallback.

    Buffers the last ``window`` aim points and fits a polynomial of degree
    ``poly`` across them. The window must be odd and larger than ``poly``.
    Recommended starting points: ``window=11, poly=2`` at 30 fps,
    ``window=7, poly=2`` at 60 fps.
    """

    def __init__(self, window: int = 11, poly: int = 2,
                 fallback_alpha: float = 0.35):
        if window % 2 == 0:
            window += 1
        self.window = max(poly + 2, window)
        self.poly = poly
        self._buf_x: deque = deque(maxlen=self.window)
        self._buf_y: deque = deque(maxlen=self.window)
        self._fallback = EMASmoother(fallback_alpha)
        self._savgol_available = self._check_savgol()

    @staticmethod
    def _check_savgol() -> bool:
        try:
            from scipy.signal import savgol_filter  # noqa: F401
            return True
        except ImportError:
            return False

    def reset(self) -> None:
        self._buf_x.clear()
        self._buf_y.clear()
        self._fallback.reset()

    def update(self, aim: Aim) -> Aim:
        x, y = aim
        self._buf_x.append(x)
        self._buf_y.append(y)

        if not self._savgol_available:
            return self._fallback.update(aim)

        n = len(self._buf_x)
        if n < self.poly + 2:
            return self._fallback.update(aim)

        from scipy.signal import savgol_filter

        # Use as much of the buffer as is available, but the window must
        # be odd and at least ``poly + 2``.
        w = n if n % 2 == 1 else n - 1
        min_w = self.poly + 2 if (self.poly + 2) % 2 == 1 else self.poly + 3
        w = max(min_w, w)
        w = min(w, self.window if self.window % 2 == 1 else self.window - 1)

        try:
            sx = savgol_filter(np.array(self._buf_x), w, self.poly)
            sy = savgol_filter(np.array(self._buf_y), w, self.poly)
            return float(sx[-1]), float(sy[-1])
        except Exception:
            return self._fallback.update(aim)


class _PassThrough:
    """No-op smoother used for ``mode='none'``."""

    def update(self, aim: Aim) -> Aim:
        return aim

    def reset(self) -> None:
        pass


def make_smoother(mode: str, **kwargs) -> Smoother:
    """Build a smoother by name.

    Args:
        mode: ``'none'``, ``'ema'`` or ``'savgol'``.
        **kwargs: Forwarded to the selected smoother
            (``alpha``, ``window``, ``poly``).
    """
    if mode == "ema":
        return EMASmoother(alpha=kwargs.get("alpha", 0.35))
    if mode == "savgol":
        return SavGolSmoother(
            window=kwargs.get("window", 11),
            poly=kwargs.get("poly", 2),
            fallback_alpha=kwargs.get("alpha", 0.35),
        )
    return _PassThrough()
