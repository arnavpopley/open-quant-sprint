"""
Track 04 (#4), does funding-rate carry pay after fees?

On a perpetual future, longs pay shorts at every settlement when the funding rate is positive. A
cross-sectional carry book is short the perps whose longs pay high funding and long the perps with low
or negative funding. The question is whether the funding it collects covers the price moves against it
and the cost of trading in and out.

The book is dollar-neutral, hence the level of funding that every symbol pays cancels out, and the
book earns only the spread between the rates of its longs and its shorts.

This file is built up 1 piece at a time. So far it picks out the settlement rows, and compute_weights
still returns a flat book.
"""

from __future__ import annotations

import pandas as pd

SETTLEMENT_HOURS = (0, 8, 16)


def settlement_rows(df: pd.DataFrame) -> pd.Series:
    """True on the 00:00, 08:00, and 16:00 UTC rows, the 3 funding settlements of each day."""
    # The rows are picked by the clock, not by a non-zero rate. Hence a settlement that shows a rate of
    # exactly 0 (1,130 of them for BNBUSDT on the training panel) still counts as a rate of 0.
    return pd.Series(df["timestamp"].dt.hour.isin(SETTLEMENT_HOURS).to_numpy(), index=df.index)


def compute_weights(df: pd.DataFrame) -> pd.Series:
    return pd.Series(0.0, index=df.index)
