from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.contrib.arnav import vol_band_ratio, vol_scaled_deadband
from src.features import cross_sectional_zscore, parkinson_volatility, rebalance_deadband

TRAIN = Path("data/train_1h.parquet")


@pytest.fixture(scope="module")
def train() -> pd.DataFrame:
    if not TRAIN.exists():
        pytest.skip("training panel not available")
    return pd.read_parquet(TRAIN)


def _panel(ranges: np.ndarray, seed: int = 0) -> pd.DataFrame:
    """Hourly panel with one column of ``ranges`` per symbol: high = close * (1 + range), low = close."""
    n_hours, n_sym = ranges.shape
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.002, size=(n_hours, n_sym)), axis=0))
    ts = pd.date_range("2025-01-01", periods=n_hours, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": np.repeat(ts, n_sym),
            "symbol": np.tile([f"S{i}" for i in range(n_sym)], n_hours),
            "open": close.ravel(),
            "high": (close * (1.0 + ranges)).ravel(),
            "low": close.ravel(),
            "close": close.ravel(),
            "volume": 1.0,
            "quote_volume": close.ravel(),
            "taker_buy_quote_volume": 0.5 * close.ravel(),
            "trades": 1,
            "funding_rate": 0.0,
        }
    )
    return df


def _noisy_targets(df: pd.DataFrame, scale: float = 0.02, seed: int = 1) -> pd.Series:
    """A slow level per symbol plus hourly noise around it, so a 1.5 % band trades often."""
    rng = np.random.default_rng(seed)
    n = len(df)
    level = pd.Series(rng.normal(0.0, 0.002, size=n)).groupby(df["symbol"].to_numpy()).cumsum().to_numpy()
    return pd.Series(level + rng.normal(0.0, scale, size=n), index=df.index)


def _turnover(df: pd.DataFrame, w: pd.Series) -> pd.Series:
    """|w[t] - w[t-1]| per row within each symbol, starting from a flat book."""
    return w.groupby(df["symbol"].to_numpy()).diff().fillna(w).abs()


# --- 1. trades less while volatility is high ------------------------------------------------


def test_trades_less_during_volatility_spike() -> None:
    n_hours, n_sym = 600, 4
    ranges = np.full((n_hours, n_sym), 0.01)
    spike = slice(300, 348)  # 48 hours with 5 times the usual high-low range
    ranges[spike] = 0.05
    df = _panel(ranges)
    target = _noisy_targets(df)

    fixed = rebalance_deadband(df, target, buffer=0.015)
    scaled = vol_scaled_deadband(df, target, base_buffer=0.015)

    ratio = vol_band_ratio(df)
    hour = df.groupby("symbol").cumcount().to_numpy()
    in_spike = (ratio > 1.0).to_numpy()
    # The band widens only once the spike is inside the 24-row window, and only while it stays there.
    # The first spike bar already enters sigma on its own row, hence nearly every spike row is wider.
    assert in_spike[hour < spike.start].sum() == 0
    assert in_spike[(hour >= spike.start) & (hour < spike.stop)].mean() > 0.95
    assert ratio.max() == pytest.approx(4.0)

    # Same weights up to the spike: the band is the fixed one there.
    before = hour < spike.start
    np.testing.assert_array_equal(scaled[before].to_numpy(), fixed[before].to_numpy())

    # Strictly lower for every symbol and below half in total, so the result does not rest on 1 symbol.
    to_fixed = _turnover(df, fixed)
    to_scaled = _turnover(df, scaled)
    for sym in df["symbol"].unique():
        m = in_spike & (df["symbol"] == sym).to_numpy()
        assert to_scaled[m].sum() < to_fixed[m].sum()
    assert to_scaled[in_spike].sum() < 0.5 * to_fixed[in_spike].sum()


# --- 2. equals the fixed band when volatility is flat ---------------------------------------


@pytest.mark.parametrize("base_buffer", [0.005, 0.015, 0.05])
def test_flat_volatility_equals_fixed_band(base_buffer: float) -> None:
    n_hours, n_sym = 500, 3
    # Constant range per symbol (different across symbols), so sigma / baseline = 1 on every row.
    ranges = np.tile(np.array([0.004, 0.01, 0.03]), (n_hours, 1))
    df = _panel(ranges)
    target = _noisy_targets(df)
    target.iloc[::37] = np.nan  # NaN targets count as 0 in both

    # The ratio has to be exactly 1 on every row for the 2 outputs to match bit for bit, hence it is
    # checked first.
    assert (vol_band_ratio(df) == 1.0).all()
    pd.testing.assert_series_equal(
        vol_scaled_deadband(df, target, base_buffer=base_buffer),
        rebalance_deadband(df, target, buffer=base_buffer),
    )


# --- 3. causal ------------------------------------------------------------------------------


def _random_vol_panel(seed: int = 3, n_hours: int = 900, n_sym: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    regime = np.repeat(rng.lognormal(0.0, 0.6, size=(n_hours // 50 + 1, n_sym)), 50, axis=0)[:n_hours]
    ranges = 0.008 * regime * rng.lognormal(0.0, 0.3, size=(n_hours, n_sym))
    return _panel(ranges, seed=seed)


def test_causal_on_prefixes() -> None:
    df = _random_vol_panel()
    target = _noisy_targets(df, seed=4)
    full = vol_scaled_deadband(df, target)
    full_ratio = vol_band_ratio(df)
    assert (full_ratio > 1.0).mean() > 0.1  # the band does widen on this panel

    # Each cut keeps whole timestamps, the same as the cuts in the audit. The cuts fall in the warm-up,
    # inside widened stretches, and near the end. The target is sliced with the panel, since only the
    # deadband is under test here.
    ts = df["timestamp"].drop_duplicates().sort_values().reset_index(drop=True)
    for k in (5, 30, 100, 180, 333, 512, 777, len(ts) - 2):
        cut = ts.iloc[k]
        head = df[df["timestamp"] <= cut]
        part = vol_scaled_deadband(head, target.loc[head.index])
        np.testing.assert_allclose(part.to_numpy(), full.loc[head.index].to_numpy(), rtol=0, atol=1e-12)
        np.testing.assert_allclose(
            vol_band_ratio(head).to_numpy(), full_ratio.loc[head.index].to_numpy(), rtol=0, atol=1e-12
        )


def test_no_leak_across_symbols() -> None:
    df = _random_vol_panel()
    target = _noisy_targets(df, seed=4)
    base = vol_scaled_deadband(df, target)

    other = df.copy()
    is_s1 = other["symbol"] == "S1"
    other.loc[is_s1, "high"] = other.loc[is_s1, "close"] * 1.2  # S1 becomes very volatile
    changed = vol_scaled_deadband(other, target)
    # Only S1 is changed, hence every other symbol has to give exactly the same weights as before.
    assert not np.allclose(changed[is_s1], base[is_s1])
    pd.testing.assert_series_equal(changed[~is_s1], base[~is_s1])


# --- alignment, NaN and edge cases ----------------------------------------------------------


def test_index_alignment_and_nan_targets() -> None:
    df = _random_vol_panel(n_hours=400)
    target = _noisy_targets(df)
    target.iloc[::11] = np.nan
    ref = vol_scaled_deadband(df, target)

    # The same panel under a non-default index has to give the same numbers, under the new labels.
    shifted = df.copy()
    shifted.index = shifted.index * 10 + 7
    t_shifted = pd.Series(target.to_numpy(), index=shifted.index)
    out = vol_scaled_deadband(shifted, t_shifted)
    assert out.index.equals(shifted.index)
    assert len(out) == len(shifted)
    assert out.notna().all()
    np.testing.assert_array_equal(out.to_numpy(), ref.to_numpy())

    # Deterministic.
    pd.testing.assert_series_equal(vol_scaled_deadband(shifted, t_shifted), out)


def test_ratio_warm_up_bounds_and_zero_range() -> None:
    ranges = np.full((300, 2), 0.01)
    ranges[:, 1] = 0.0  # high == low: sigma 0, ratio 0 / 0
    ranges[200:230, 0] = 0.5
    df = _panel(ranges)
    ratio = vol_band_ratio(df, max_ratio=4.0)
    hour = df.groupby("symbol").cumcount().to_numpy()

    # sigma needs 12 rows and its baseline another 84 after the shift, hence the first 95 rows of each
    # symbol use the fixed band.
    sigma = parkinson_volatility(df)
    assert sigma[hour < 11].isna().all()
    assert (ratio[hour < 11 + 84] == 1.0).all()  # sigma or its baseline not defined yet
    assert (ratio[(df["symbol"] == "S1").to_numpy()] == 1.0).all()
    assert ratio.between(1.0, 4.0).all()
    assert ratio.max() == 4.0

    with pytest.raises(ValueError):
        vol_band_ratio(df, max_ratio=0.5)


def test_zero_buffer_passes_target_through() -> None:
    df = _random_vol_panel(n_hours=200)
    target = _noisy_targets(df)
    pd.testing.assert_series_equal(vol_scaled_deadband(df, target, base_buffer=0.0), target)


# --- training panel -------------------------------------------------------------------------


def _template_target(df: pd.DataFrame) -> pd.Series:
    """The target of strategies/template.py before its deadband."""
    p_vol = parkinson_volatility(df, window=24).replace(0.0, np.nan)
    raw = df.groupby("symbol")["close"].pct_change(24) / p_vol
    z = cross_sectional_zscore(df, raw.fillna(0.0))
    smoothed = z.groupby(df["symbol"]).transform(lambda x: x.ewm(span=18, min_periods=1).mean())
    return cross_sectional_zscore(df, smoothed)


def test_train_panel_alignment_and_turnover(train: pd.DataFrame) -> None:
    target = _template_target(train)
    scaled = vol_scaled_deadband(train, target)
    fixed = rebalance_deadband(train, target, buffer=0.015)
    assert scaled.index.equals(train.index)
    assert scaled.notna().all()

    ratio = vol_band_ratio(train)
    assert ratio.between(1.0, 4.0).all()
    widened = (ratio > 1.0).to_numpy()
    # About half the rows are widened, which follows from the median baseline (module docstring).
    assert 0.2 < widened.mean() < 0.8
    assert _turnover(train, scaled)[widened].sum() < _turnover(train, fixed)[widened].sum()
    assert _turnover(train, scaled).sum() < _turnover(train, fixed).sum()


@pytest.mark.parametrize("frac", [0.4, 0.85])
def test_train_panel_causal(train: pd.DataFrame, frac: float) -> None:
    ts = train["timestamp"].drop_duplicates().sort_values()
    head = train[train["timestamp"] <= ts.iloc[int(len(ts) * frac)]]
    target = _template_target(train)
    full = vol_scaled_deadband(train, target).loc[head.index].to_numpy()
    part = vol_scaled_deadband(head, target.loc[head.index]).to_numpy()
    np.testing.assert_allclose(part, full, rtol=0, atol=1e-12)
