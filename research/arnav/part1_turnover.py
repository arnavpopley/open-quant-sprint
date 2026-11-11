"""Track 04 Part 1: turnover of vol_scaled_deadband against the fixed band on the training panel.

Targets: strategies/template.py before its deadband (24h momentum over Parkinson vol, z-scored,
EWM span 18, z-scored again). Both filters use a 1.5 % base band. Turnover and costs come from
src.backtest.run_backtest at 5 bps (1-bar lag, gross scaled to 1), as in the audit.

Run: python -m research.arnav.part1_turnover
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest import run_backtest
from src.contrib.arnav import vol_band_ratio, vol_scaled_deadband
from src.features import cross_sectional_zscore, parkinson_volatility, rebalance_deadband


def template_target(df: pd.DataFrame) -> pd.Series:
    p_vol = parkinson_volatility(df, window=24).replace(0.0, np.nan)
    raw = df.groupby("symbol")["close"].pct_change(24) / p_vol
    z = cross_sectional_zscore(df, raw.fillna(0.0))
    smoothed = z.groupby(df["symbol"]).transform(lambda x: x.ewm(span=18, min_periods=1).mean())
    return cross_sectional_zscore(df, smoothed)


def main() -> None:
    df = pd.read_parquet("data/train_1h.parquet")
    target = template_target(df)
    ratio = vol_band_ratio(df)
    books = {
        "fixed 1.5 %": rebalance_deadband(df, target, buffer=0.015),
        "vol-scaled 1.5 %": vol_scaled_deadband(df, target, base_buffer=0.015),
    }
    print(f"rows {len(df)}, band widened on {(ratio > 1).mean() * 100:.1f} % of rows, "
          f"capped at 4 on {(ratio >= 4).mean() * 100:.2f} %, median ratio where widened {ratio[ratio > 1].median():.2f}")
    print(f"{'book':<18} {'turnover/h':>10} {'fee cost':>9} {'gross':>8} {'Sharpe@5':>9}")
    res = {}
    for name, w in books.items():
        m = run_backtest(df, w, fee_bps=5.0)
        res[name] = m
        print(f"{name:<18} {m.mean_turnover:>10.4f} {m.fee_cost_cum * 100:>8.2f}% {m.gross_pnl_cum * 100:>7.2f}% {m.sharpe:>+9.2f}")
    a, b = res["fixed 1.5 %"], res["vol-scaled 1.5 %"]
    print(f"turnover change: {(b.mean_turnover / a.mean_turnover - 1) * 100:+.1f} %")

    # Turnover on widened rows only, from the weights themselves.
    sym = df["symbol"].to_numpy()
    for name, w in books.items():
        to = w.groupby(sym).diff().fillna(w).abs()
        print(f"{name:<18} turnover on widened rows {to[ratio > 1].sum():.2f}, on other rows {to[ratio == 1].sum():.2f}")


if __name__ == "__main__":
    main()
