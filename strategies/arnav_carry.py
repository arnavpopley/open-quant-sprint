"""
Track 04 (#4), does funding-rate carry pay after fees?

On a perpetual future, longs pay shorts at every settlement when the funding rate is positive. A
cross-sectional carry book is short the perps whose longs pay high funding and long the perps with low
or negative funding. The question is whether the funding it collects covers the price moves against it
and the cost of trading in and out.

The book is dollar-neutral, hence the level of funding that every symbol pays cancels out, and the
book earns only the spread between the rates of its longs and its shorts.

This file is built up 1 piece at a time. For now compute_weights returns a flat book.
"""

from __future__ import annotations

import pandas as pd


def compute_weights(df: pd.DataFrame) -> pd.Series:
    return pd.Series(0.0, index=df.index)
