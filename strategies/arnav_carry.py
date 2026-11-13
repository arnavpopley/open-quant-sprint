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

This file is built up 1 piece at a time. So far it builds the funding signal, ranks it across
symbols, turns the ranks into weights, holds them between rebalance rows, and filters them with
vol_scaled_deadband (src/contrib/arnav.py), then centres the book again so that sum w = 0 at every
timestamp.
"""

from __future__ import annotations

import pandas as pd

from src.contrib.arnav import vol_scaled_deadband

SETTLEMENT_HOURS = (0, 8, 16)
SPAN = 21  # in settlements, so 1 week; a settlement's weight halves about every 7 settlements (2.4 days)
MIN_SETTLEMENTS = 3  # a symbol needs 1 day of settlements before it gets a signal
# rebalance on the settlement rows themselves, the first row each new rate is in the data
# filled at 01:00, 09:00, 17:00, so 7 hours before the next settlement, and a 1h delay still holds it
REBALANCE_HOURS = (0, 8, 16)
# the Part 1 band: 1.5 %, widening up to 4x when a symbol's vol is high against its own last week
BASE_BUFFER = 0.015
VOL_WINDOW = 24


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


def rank_target(df: pd.DataFrame, signal: pd.Series) -> pd.Series:
    """Short high funding, long low funding, linear in rank, with sum |w| = 1 and sum w = 0 at each timestamp."""
    ts = df["timestamp"]
    r = rank_signal(df, signal)
    n = signal.notna().groupby(ts).transform("sum")
    # rank 1 is the lowest funding, so (n + 1) / 2 - rank is + for the longs and - for the shorts
    # with all 20 ranked the extremes get 0.095 and 1 place apart is 0.01, so small swaps barely move it
    u = ((n + 1.0) / 2.0 - r).where(n >= 2)
    # fewer than 2 ranked symbols means no spread to trade, so no weight
    l1 = u.abs().groupby(ts).transform("sum")
    return (u / l1.where(l1 > 0)).fillna(0.0)


def hold_between_rebalances(df: pd.DataFrame, target: pd.Series) -> pd.Series:
    """The target moves only on the rebalance rows and is held on the hours in between."""
    is_reb = df["timestamp"].dt.hour.isin(REBALANCE_HOURS).to_numpy()
    # with rebalances on the settlement rows this changes nothing, since the signal only moves there
    # anyway; it keeps the schedule explicit in case the two ever differ
    return target.where(is_reb).groupby(df["symbol"].to_numpy(), sort=False).ffill().fillna(0.0)


def build_weights(df: pd.DataFrame, span: int = SPAN, base_buffer: float = BASE_BUFFER) -> pd.Series:
    """The whole book from the funding signal, with the span and the band as arguments."""
    # the arguments are only there so the breakdown can switch the deadband off (base_buffer=0) on the
    # same code; compute_weights always calls it with the fixed values above
    target = hold_between_rebalances(df, rank_target(df, funding_signal(df, span=span)))
    # small moves in the target (1-place rank swaps are 0.01) stay inside the band and cost no fees
    # the band acts per symbol, so a blocked trade can go through on a later hour once the band narrows
    w = vol_scaled_deadband(df, target, base_buffer=base_buffer, vol_window=VOL_WINDOW)
    # the band can also leave the book a little net long or short, so take the mean back out
    # this costs a small trade in every name whenever the net changes, which is the price of staying neutral
    return w - w.groupby(df["timestamp"]).transform("mean")


def compute_weights(df: pd.DataFrame) -> pd.Series:
    return build_weights(df)
