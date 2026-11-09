"""
Track 07 (#7): where in the hour the price moved.

Signal: the gap between the bar close and the bar VWAP, after removing what the hour's own return
explains (src/contrib/angelo.close_location_residual). It is averaged over SPAN hours and
residualized again on the SPAN-hour return, so the average doesn't turn into plain reversal.
A close far above the VWAP for its return is expected to give some of that back: weights are
the negative of the signal.

Pre-registered primary cell (PR #9): SPAN = 6, DEADBAND = 0.015. These values are fixed and are not
tuned on results. The other eight grid cells call build_weights with the other (span, deadband)
pairs; results are in the PR #9 description.
"""

from __future__ import annotations

import pandas as pd

from src.contrib.angelo import smoothed_close_location_residual
from src.features import cross_sectional_zscore, rebalance_deadband

SPAN = 6
DEADBAND = 0.015


def build_weights(df: pd.DataFrame, span: int, deadband: float) -> pd.Series:
    signal = -smoothed_close_location_residual(df, span)
    target = cross_sectional_zscore(df, signal.fillna(0.0))
    return rebalance_deadband(df, target, buffer=deadband)


def compute_weights(df: pd.DataFrame) -> pd.Series:
    return build_weights(df, SPAN, DEADBAND)
