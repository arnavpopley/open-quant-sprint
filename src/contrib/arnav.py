"""Activation-PR feature by Arnav Popley (Track 04, funding-rate carry).

The conventions are the same as src/features.py. Every output is causal and aligned with df.index,
and one symbol's history never reaches another symbol's output.

``vol_scaled_deadband`` is ``rebalance_deadband`` with a no-trade band that is set per symbol and per
row. The band widens when a symbol's Parkinson volatility is high relative to its own recent history.

    b[i, t] = base_buffer * clip(sigma[i, t] / median(sigma[i, t-168 .. t-1]), 1, max_ratio)

- sigma is ``parkinson_volatility(df, window=vol_window)`` (24 rows by default, min_periods 12).
- The baseline is the rolling median of sigma over the previous ``baseline_window`` rows of the same
  symbol (168 rows, 1 week of hourly bars, by default). The current row is left out of it, hence a
  spike is compared with the week before it rather than with itself. A median is used instead of a
  mean so that 1 extreme bar does not drag the baseline up with it. The baseline needs half of its
  window (84 rows) before it is defined, which is the same min_periods rule as
  ``parkinson_volatility``.
- The floor at 1 means the band only ever widens, hence a series with flat volatility gets the fixed
  band exactly. However, sigma sits above its own median on about half the rows of any series, so
  the band is wider than the fixed one on about half the rows by construction. The size of the
  widening is what matters, not how often it happens.
- The cap (``max_ratio``, 4 by default) stops 1 extreme stretch of bars from freezing a symbol for days.
- During warm-up (the first 95 rows of each symbol), sigma or its baseline is NaN, and the ratio is
  set to 1 so that the band is the fixed one. The same applies where the ratio is undefined (0 / 0,
  a symbol whose high equals its low for a week). A baseline of 0 under a positive sigma gives the cap.
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
    Per-row multiplier of the no-trade band, sigma / trailing median of sigma, clipped to [1, max_ratio].
    The median covers the previous ``baseline_window`` rows of the same symbol, not the current one.
    It is 1.0 wherever the ratio is undefined (warm-up, zero range).
    """
    if max_ratio < 1.0:
        raise ValueError(f"max_ratio must be >= 1, got {max_ratio}")
    sym = df["symbol"].to_numpy()
    sigma = parkinson_volatility(df, window=vol_window)
    # The shift by 1 row keeps the current bar out of its own baseline. Grouping by symbol keeps
    # one symbol's history out of another's.
    baseline = sigma.groupby(sym, sort=False).transform(
        lambda x: x.shift(1).rolling(window=baseline_window, min_periods=max(2, baseline_window // 2)).median()
    )
    s = sigma.to_numpy(dtype=np.float64)
    b = baseline.to_numpy(dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = s / b
    # NaN covers the warm-up rows and 0 / 0, and both fall back to the fixed band. A positive sigma
    # over a zero baseline gives inf, which the clip turns into the cap.
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
    Causal no-trade zone per symbol with a volatility-scaled band. The weight moves to the target only
    when |w_target[t] - w_actual[t-1]| > base_buffer * vol_band_ratio[t]. NaN targets count as 0, as
    in ``rebalance_deadband``. With ``base_buffer <= 0`` it returns ``rebalance_deadband`` unchanged.
    """
    # With no band there is nothing to scale, hence the fixed version is returned as it is.
    if base_buffer <= 0.0:
        return rebalance_deadband(df, target_weights, buffer=base_buffer)

    w_arr = pd.Series(target_weights, index=df.index).fillna(0.0).to_numpy(dtype=np.float64)
    band = base_buffer * vol_band_ratio(df, vol_window, baseline_window, max_ratio).to_numpy(dtype=np.float64)
    out = np.zeros_like(w_arr)

    # The same loop as rebalance_deadband, with band[i] in place of a single buffer. Each symbol
    # starts flat and moves only when its target is further than its band from the current weight.
    for _, pos in df.groupby("symbol", sort=False).indices.items():
        cur = 0.0
        for i in pos:
            tgt = w_arr[i]
            if abs(tgt - cur) > band[i]:
                cur = tgt
            out[i] = cur

    return pd.Series(out, index=df.index)
