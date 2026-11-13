"""
Track 04 (#4), does funding-rate carry pay after fees?

On a perpetual future, longs pay shorts at every settlement when the funding rate is positive. A
cross-sectional carry book is short the perps whose longs pay high funding and long the perps with low
or negative funding. The question is whether the funding it collects covers the price moves against it
and the cost of trading in and out.

The book is dollar-neutral, hence the level of funding that every symbol pays cancels out, and the
book earns only the spread between the rates of its longs and its shorts.

Timing. The weight returned on a row is filled at the close of that bar and is the position held
across the next settlement. Hence a weight set on the 00:00 row is filled at 01:00 and is charged or
paid the 08:00 rate, which is not in the data at 01:00. The signal on a row uses the settlements up to
and including that row's own (the rate settles at the start of the bar, so it is known by its close)
and never the next one. Binance shows a live estimate of the next rate, however, the dataset has no
such estimate, so the strategy uses settled rates only and carry pays only if funding persists.

This file is built up 1 piece at a time. So far it builds the funding signal for every row and ranks
it across symbols, and compute_weights still returns a flat book.
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


def funding_signal(df: pd.DataFrame, span: int = SPAN, min_settlements: int = MIN_SETTLEMENTS) -> pd.Series:
    """The latest settled EWM on every row: a row sees its own settlement and earlier ones, never the next."""
    ewm = settled_funding_ewm(df, span=span, min_settlements=min_settlements)
    # each hour keeps the last settled value of its own symbol, nothing from a later row
    return ewm.groupby(df["symbol"].to_numpy(), sort=False).ffill()


def rank_signal(df: pd.DataFrame, signal: pd.Series) -> pd.Series:
    """Rank of the signal across symbols at each timestamp, 1 = lowest funding, NaN where there is no signal yet."""
    # ties share the average rank, so the result does not depend on the order of the rows
    # symbols without a signal stay NaN and are simply left out of that timestamp's ranking
    return signal.groupby(df["timestamp"]).rank(method="average")


def compute_weights(df: pd.DataFrame) -> pd.Series:
    return pd.Series(0.0, index=df.index)
