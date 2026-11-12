import numpy as np
import pandas as pd
import pytest

from src.contrib.janice import leader_flow_impulse


def make_panel():
    """Small synthetic panel; no downloads or real data needed."""
    rows = []

    for timestamp in pd.date_range(
        "2026-01-01", periods=6, freq="h", tz="UTC"
    ):
        rows.extend([
            (timestamp, "BTCUSDT", 100.0, 75.0),
            (timestamp, "ETHUSDT", 300.0, 150.0),
            (timestamp, "SOLUSDT", 50.0, 5.0),
        ])

    df = pd.DataFrame(rows, columns=[
        "timestamp",
        "symbol",
        "quote_volume",
        "taker_buy_quote_volume",
    ])

    # Deliberately avoid a default index to test alignment.
    df.index = pd.Index(
        np.arange(len(df)) * 3 + 10, name="row"
    )
    return df


def make_varying_panel():
    df = make_panel()
    raw_values = [0.125, -0.5, 1.0, -1.0, 0.0, 0.5]

    for timestamp, imbalance in zip(
        df["timestamp"].unique(), raw_values
    ):
        mask = (
            df["timestamp"].eq(timestamp)
            & df["symbol"].isin(["BTCUSDT", "ETHUSDT"])
        )
        df.loc[mask, "taker_buy_quote_volume"] = (
            df.loc[mask, "quote_volume"]
            * (1.0 + imbalance)
            / 2.0
        )

    return df, raw_values


def test_volume_weighting():
    df = make_panel()

    # BTC signed quote = 2*75 - 100 = 50.
    # ETH signed quote = 2*150 - 300 = 0.
    # Combined imbalance = 50 / 400 = 0.125.
    result = leader_flow_impulse(df, span=1)

    np.testing.assert_allclose(result, 0.125)


def test_broadcast_bounds_and_index():
    df, _ = make_varying_panel()
    result = leader_flow_impulse(df)

    assert result.index.equals(df.index)
    assert result.between(-1.0, 1.0).all()
    assert (
        result.groupby(df["timestamp"])
        .nunique()
        .eq(1)
        .all()
    )


def test_ewm_matches_expected():
    df, raw_values = make_varying_panel()

    expected = pd.Series(raw_values).ewm(
        span=3, adjust=True, min_periods=1
    ).mean()

    actual = (
        leader_flow_impulse(df, span=3)
        .groupby(df["timestamp"])
        .first()
    )

    np.testing.assert_allclose(
        actual.to_numpy(), expected.to_numpy()
    )


def test_causality_on_every_complete_hour_prefix():
    df, _ = make_varying_panel()
    full = leader_flow_impulse(df, span=3)

    for cutoff in df["timestamp"].unique():
        mask = df["timestamp"].le(cutoff)
        prefix = leader_flow_impulse(
            df.loc[mask].copy(), span=3
        )

        pd.testing.assert_series_equal(
            prefix,
            full.loc[mask],
            check_exact=False,
            atol=1e-12,
            rtol=0.0,
        )


def test_future_changes_do_not_change_past():
    df, _ = make_varying_panel()
    cutoff = df["timestamp"].unique()[2]
    past = df["timestamp"].le(cutoff)

    changed = df.copy()
    changed.loc[
        ~past, "taker_buy_quote_volume"
    ] = 0.0

    pd.testing.assert_series_equal(
        leader_flow_impulse(df, span=3).loc[past],
        leader_flow_impulse(changed, span=3).loc[past],
    )


def test_altcoin_flow_is_ignored():
    df, _ = make_varying_panel()
    changed = df.copy()
    changed.loc[
        changed["symbol"].eq("SOLUSDT"),
        "taker_buy_quote_volume",
    ] = 50.0

    pd.testing.assert_series_equal(
        leader_flow_impulse(df),
        leader_flow_impulse(changed),
    )


def test_zero_volume():
    df = make_panel()
    df[["quote_volume", "taker_buy_quote_volume"]] = 0.0

    assert leader_flow_impulse(df).eq(0.0).all()


def test_missing_leader_hour_is_nan_and_decays():
    df, raw_values = make_varying_panel()
    gap = df["timestamp"].unique()[3]
    dropped = df.loc[
        ~(df["timestamp"].eq(gap) & df["symbol"].eq("ETHUSDT"))
    ]
    result = leader_flow_impulse(dropped, span=3)

    assert result[dropped["timestamp"].eq(gap)].isna().all()

    raw = pd.Series(raw_values, dtype=float)
    raw[3] = np.nan
    expected = raw.ewm(span=3, adjust=True, min_periods=1).mean()
    expected[3] = np.nan
    actual = result.groupby(dropped["timestamp"]).first()
    np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy())


def test_missing_leader_everywhere_is_all_nan():
    df = make_panel()
    df = df.loc[df["symbol"].ne("BTCUSDT")]

    assert leader_flow_impulse(df).isna().all()


def test_duplicate_leader_row_raises():
    df = make_panel()
    df = pd.concat([df, df.iloc[[0]]])

    with pytest.raises(ValueError, match="at most one"):
        leader_flow_impulse(df)
