"""
Track 04 (#4), does funding-rate carry pay after fees?

On a perpetual future, longs pay shorts at every settlement when the funding rate is positive. A
cross-sectional carry book is short the perps whose longs pay high funding and long the perps with low
or negative funding. The question is whether the funding it collects covers the price moves against it
and the cost of trading in and out.

The book is dollar-neutral, hence the level of funding that every symbol pays cancels out, and the
book earns only the spread between the rates of its longs and its shorts.

This file is built up 1 piece at a time. So far it picks out the settlement rows and averages each
symbol's settled rates, and compute_weights still returns a flat book.
"""

from __future__ import annotations

import pandas as pd

SETTLEMENT_HOURS = (0, 8, 16)
SPAN = 21  # in settlements, so 1 week; a settlement's weight halves about every 7 settlements (2.4 days)
MIN_SETTLEMENTS = 3  # a symbol needs 1 day of settlements before it gets a signal


def settlement_rows(df: pd.DataFrame) -> pd.Series:
    """True on the 00:00, 08:00, and 16:00 UTC rows, the 3 funding settlements of each day."""
    # picked by the clock, not by a non-zero rate, so BNB's 1,130 zero settlements still count as 0
    return pd.Series(df["timestamp"].dt.hour.isin(SETTLEMENT_HOURS).to_numpy(), index=df.index)


def settled_funding_ewm(df: pd.DataFrame, span: int = SPAN, min_settlements: int = MIN_SETTLEMENTS) -> pd.Series:
    """EWM of each symbol's settled funding rates, on the settlement rows only (NaN on the hours in between)."""
    is_settle = settlement_rows(df).to_numpy()
    # only the settlement rows go in, so the span counts settlements and not hours
    # (the 0s on the hours in between would otherwise drag every average towards 0)
    settled = df.loc[is_settle, "funding_rate"].astype("float64")
    sym = df["symbol"].to_numpy()[is_settle]
    # grouped by symbol, so one coin's funding never leaks into another's average
    ewm = settled.groupby(sym, sort=False).transform(lambda x: x.ewm(span=span, min_periods=min_settlements).mean())
    return ewm.reindex(df.index)


def compute_weights(df: pd.DataFrame) -> pd.Series:
    return pd.Series(0.0, index=df.index)
