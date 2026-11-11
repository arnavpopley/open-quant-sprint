"""Activation-PR feature by Arnav Popley (Track 04: funding-rate carry).

Same conventions as src/features.py: causal, aligned with df.index, one symbol's history never
reaches another symbol's output.

``vol_scaled_deadband`` is ``rebalance_deadband`` with a no-trade band per symbol and row that widens
when the symbol's Parkinson volatility is high relative to its own recent history:

    b[i, t] = base_buffer * clip(sigma[i, t] / median(sigma[i, t-168 .. t-1]), 1, max_ratio)

- sigma is ``parkinson_volatility(df, window=vol_window)`` (24 rows by default, min_periods 12).
- The baseline is the rolling median of sigma over the previous ``baseline_window`` rows of the same
  symbol, 168 rows (one week of hourly bars) by default. The current row is not part of it, so a spike
  is compared with the week before it rather than with itself. It needs half the window
  (84 rows) before it is defined, the same min_periods rule as ``parkinson_volatility``.
- The floor at 1 means the band only ever widens, and a series with flat volatility gets the fixed
  band exactly. The cap (``max_ratio``, 4 by default) stops one extreme stretch of bars from freezing a
  symbol for days.
- Warm-up: where sigma or its baseline is NaN (the first rows of each symbol), or the ratio is
  undefined (0 / 0, a symbol whose high equals its low for a week), the ratio is 1 and the band is the
  fixed one. A baseline of 0 under a positive sigma gives the cap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import parkinson_volatility, rebalance_deadband


def vol_band_ratio(
    df: pd.DataFrame,
    vol_window: int = 24,
    baseline_window: int = 168,
    max_ratio: float = 4.0,
) -> pd.Series:
    """
    Per-row multiplier of the no-trade band: sigma / trailing median of sigma, clipped to [1, max_ratio].
    The median covers the previous ``baseline_window`` rows of the same symbol, not the current one.
    1.0 wherever the ratio is undefined (warm-up, zero range).
    """
    if max_ratio < 1.0:
        raise ValueError(f"max_ratio must be >= 1, got {max_ratio}")
    sym = df["symbol"].to_numpy()
    sigma = parkinson_volatility(df, window=vol_window)
    baseline = sigma.groupby(sym, sort=False).transform(
        lambda x: x.shift(1).rolling(window=baseline_window, min_periods=max(2, baseline_window // 2)).median()
    )
    s = sigma.to_numpy(dtype=np.float64)
    b = baseline.to_numpy(dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = s / b
    ratio = np.where(np.isnan(ratio), 1.0, ratio)
    return pd.Series(np.clip(ratio, 1.0, max_ratio), index=df.index, name="vol_band_ratio")


def vol_scaled_deadband(
    df: pd.DataFrame,
    target_weights: pd.Series,
    base_buffer: float = 0.015,
    vol_window: int = 24,
    baseline_window: int = 168,
    max_ratio: float = 4.0,
) -> pd.Series:
    """
    Causal no-trade zone per symbol with a volatility-scaled band: the weight moves to the target only
    when |w_target[t] - w_actual[t-1]| > base_buffer * vol_band_ratio[t]. NaN targets count as 0, as
    in ``rebalance_deadband``. With ``base_buffer <= 0`` it returns ``rebalance_deadband`` unchanged.
    """
    if base_buffer <= 0.0:
        return rebalance_deadband(df, target_weights, buffer=base_buffer)

    w_arr = pd.Series(target_weights, index=df.index).fillna(0.0).to_numpy(dtype=np.float64)
    band = base_buffer * vol_band_ratio(df, vol_window, baseline_window, max_ratio).to_numpy(dtype=np.float64)
    out = np.zeros_like(w_arr)

    for _, pos in df.groupby("symbol", sort=False).indices.items():
        cur = 0.0
        for i in pos:
            tgt = w_arr[i]
            if abs(tgt - cur) > band[i]:
                cur = tgt
            out[i] = cur

    return pd.Series(out, index=df.index)
