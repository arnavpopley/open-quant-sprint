"""Track 04 Part 2: breakdown of the pre-registered carry book after its single audit run.

Everything here uses strategies/arnav_carry.py at the pre-registration commit (0012090) unchanged,
through its build_weights switches only (scheme="top5", base_buffer=0). No parameter is tuned. The
split, lag, null, and wick numbers come from the audit's own functions, so they match the report.

Run: python -m research.arnav.part2_breakdown
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd

from src.audit import audit_wick_vulnerability, evaluate_lag_sensitivity, evaluate_shuffled_null, purged_oos_split_eval
from src.backtest import BacktestMetrics, coerce_weights, run_backtest, tiered_fee_bps
from src.schema import enforce_schema

FEE_GRID = np.arange(0.0, 20.0 + 1e-9, 0.5)


def load_strategy():
    spec = importlib.util.spec_from_file_location("arnav_carry", "strategies/arnav_carry.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def years(df: pd.DataFrame) -> float:
    return df["timestamp"].nunique() / 8760.0


def split_line(name: str, m: BacktestMetrics, yrs: float) -> str:
    # the audit prints funding as a cost, so the funding earned is minus funding_cost_cum
    funding = -m.funding_cost_cum
    return (
        f"{name:<22} Sharpe {m.sharpe:+6.2f} | price {m.gross_pnl_cum * 100:+7.2f}% | funding {funding * 100:+6.2f}% "
        f"({funding / yrs * 100:+5.2f}%/yr) | fees {-m.fee_cost_cum * 100:+7.2f}% | net {m.net_pnl_cum * 100:+7.2f}% | "
        f"turnover {m.mean_turnover:.4f}/h"
    )


def break_even(df: pd.DataFrame, w: pd.Series) -> tuple[float | None, list[tuple[float, float]]]:
    """First fee on the grid where net Sharpe drops to 0 or below, linearly interpolated."""
    curve = [(float(f), run_backtest(df, w, fee_bps=float(f)).sharpe) for f in FEE_GRID]
    for (f0, s0), (f1, s1) in zip(curve, curve[1:]):
        if s0 > 0.0 >= s1:
            return f0 + (f1 - f0) * s0 / (s0 - s1), curve
    return None, curve


def tilt_and_timing(df: pd.DataFrame, w: pd.Series) -> tuple[float, float, pd.DataFrame]:
    """
    Price PnL split into a static tilt and timing, on the held weights after the harness scaling:
    sum_t w_t r_t = sum_i mean(w_i) * sum_t r_i,t + sum_t (w_t - mean(w)) r_t
    The first term is what check [8] measures; the second is what shuffling the days destroys.
    """
    work = df[["timestamp", "symbol", "close"]].copy()
    work["w"] = coerce_weights(df, w).to_numpy()
    gross = work["w"].abs().groupby(work["timestamp"]).transform("sum")
    work["w"] = work["w"] * np.where(gross > 1.0, 1.0 / gross, 1.0)
    work["w_held"] = work.groupby("symbol")["w"].shift(1).fillna(0.0)
    work["ret"] = work.groupby("symbol")["close"].pct_change().fillna(0.0)
    per_sym = work.groupby("symbol").agg(mean_w=("w_held", "mean"), sum_ret=("ret", "sum"))
    per_sym["tilt_pnl"] = per_sym["mean_w"] * per_sym["sum_ret"]
    total = float((work["w_held"] * work["ret"]).sum())
    tilt = float(per_sym["tilt_pnl"].sum())
    return tilt, total - tilt, per_sym.sort_values("mean_w")


def report_book(name: str, df: pd.DataFrame, w: pd.Series) -> None:
    print(f"\n=== {name} ===")
    full = run_backtest(df, w, fee_bps=5.0)
    split = purged_oos_split_eval(df, w, fee_bps=5.0)
    lag = evaluate_lag_sensitivity(df, w, fee_bps=5.0)
    n_is = int(df["timestamp"].nunique() * 0.70)
    n_oos = df["timestamp"].nunique() - n_is - 24
    print("PnL split at 5 bps (totals over each period, funding earned is positive)")
    print(split_line("full panel", full, years(df)))
    print(split_line("first 70% (in-sample)", split["IS"], n_is / 8760))
    print(split_line("last 30% (out-of-sample)", split["OOS"], n_oos / 8760))
    print(split_line("full, lag test (+1h)", lag["m_lag2"], years(df)))

    no_fee_oos = purged_oos_split_eval(df, w, fee_bps=0.0)["OOS"]
    print(f"last 30% at 0 bps: Sharpe {no_fee_oos.sharpe:+.2f} (does it fail even without fees?)")

    be, curve = break_even(df, w)
    print(f"break-even fee, full panel: {be:.2f} bps" if be is not None else "break-even fee: none on the 0-20 bps grid")
    print("fee curve: " + ", ".join(f"{f:g}:{s:+.2f}" for f, s in curve if f in (0, 2, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20)))
    tiered = run_backtest(df, w, fee_bps=tiered_fee_bps(df))
    print(f"tiered fees (~{tiered.fee_bps:.1f} bps mean): Sharpe {tiered.sharpe:+.2f}")

    null = evaluate_shuffled_null(df, w, seed=42)
    wick = audit_wick_vulnerability(df, w)
    print(f"null check [8]: mean {null['null_sharpe']:+.2f} (sd {null['null_sd']:.2f}, passes if |mean| < 0.75), "
          f"actual gross SR {null['actual_gross_sharpe']:+.2f}, p={null['p_value']:.4f} | wick share {wick['wick_pnl_share'] * 100:.1f}%")
    tilt, timing, per_sym = tilt_and_timing(df, w)
    print(f"price PnL = static tilt {tilt * 100:+.2f}% + timing {timing * 100:+.2f}% (= {full.gross_pnl_cum * 100:+.2f}%)")
    print("average held weight and summed hourly return, most short and most long names:")
    for sym, row in pd.concat([per_sym.head(4), per_sym.tail(4)]).iterrows():
        print(f"    {sym:<9} mean w {row['mean_w']:+.4f} | summed return {row['sum_ret'] * 100:+7.1f}% | tilt PnL {row['tilt_pnl'] * 100:+6.2f}%")

    ts = df["timestamp"]
    dw = w.groupby(df["symbol"]).diff().fillna(w).abs()
    reb = ts.dt.hour.isin((0, 8, 16))
    print(f"turnover on the 00/08/16 rows: {dw[reb].sum() / dw.sum() * 100:.1f}% of the total")


def main() -> None:
    df = enforce_schema(pd.read_parquet("data/train_1h.parquet"))
    c = load_strategy()
    primary = c.compute_weights(df)
    assert np.array_equal(primary.to_numpy(), c.build_weights(df).to_numpy())
    print(f"rows {len(df)}, {df['timestamp'].nunique()} hours ({years(df):.2f} years)")

    report_book("primary: rank, deadband on (pre-registered)", df, primary)

    no_band = c.build_weights(df, base_buffer=0.0)
    m_on, m_off = run_backtest(df, primary, fee_bps=5.0), run_backtest(df, no_band, fee_bps=5.0)
    print("\n=== deadband on vs off (rank book, 5 bps) ===")
    print(split_line("deadband on", m_on, years(df)))
    print(split_line("deadband off", m_off, years(df)))
    print(f"turnover change from the deadband: {(m_on.mean_turnover / m_off.mean_turnover - 1) * 100:+.1f}%")

    report_book("secondary: top5 (pre-registered, not used to choose)", df, c.build_weights(df, scheme="top5"))


if __name__ == "__main__":
    main()
