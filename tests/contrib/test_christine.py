import numpy as np
import pytest
from src.contrib.christine import effective_trials, sharpe_spread


def test_identical_strategies():
    rng = np.random.default_rng(seed=42)
    rt = rng.standard_normal(14871)
    returns_matrix = np.column_stack([rt]* 9)
    N_eff = effective_trials(returns=returns_matrix)

    assert np.isclose(N_eff, 1.0, atol=1e-6, rtol=0)

def test_independent_strategies():
    rng = np.random.default_rng(seed=42)
    independent_returns = rng.standard_normal(size=(14871, 9))
    N_eff = effective_trials(returns=independent_returns)

    assert 8.5 <= N_eff <= 9.5

def test_grouped_strategies():
    rng = np.random.default_rng(seed=42)
    group1 = rng.standard_normal(size=14871)
    group2 = rng.standard_normal(size=14871)
    group3 = rng.standard_normal(size=14871)
    returns_matrix = np.column_stack(
        [group1] * 3 + [group2] * 3 + [group3] * 3
    )

    assert returns_matrix.shape == (14871, 9)

    N_eff = effective_trials(returns=returns_matrix)

    assert np.isclose(N_eff, 3.0, atol=0.01)
    

def test_negatively_correlated_strategies():
    rng = np.random.default_rng(seed=42)
    rt = rng.standard_normal(14871)
    noise = rng.normal(loc=0, scale=0.01, size=14871)
    rt_neg = -rt + noise
    returns_matrix = np.column_stack([rt] + [rt_neg])
    N_eff = effective_trials(returns=returns_matrix)

    assert np.isclose(N_eff, 1.0, atol=0.01)
    assert np.corrcoef(rt, rt_neg)[0, 1] < -0.98

def test_sharpe_spread():
    rng = np.random.default_rng(seed=42)
    rt = rng.standard_normal(14871)
    rt_st = (rt - np.mean(rt))/np.std(rt, ddof=1)
    strategy_A = rt_st + 0.1
    strategy_B = rt_st + 0.2
    strategy_C = rt_st + 0.3
    returns_matrix = np.column_stack([strategy_A] + [strategy_B] + [strategy_C])
    sample_var = sharpe_spread(returns=returns_matrix)

    assert np.isclose(sample_var, 0.01, atol=1e-12, rtol=0)


def test_invalid_dimensions():
    returns = np.array([0.1, 0.2, 0.3])

    with pytest.raises(ValueError, match="two-dimensional"):
        effective_trials(returns)

    with pytest.raises(ValueError, match="two-dimensional"):
        sharpe_spread(returns)


def test_insufficient_observations():
    returns = np.array([
        [0.01, np.nan],
        [np.nan, 0.02],
        [0.03, 0.04]
    ])

    with pytest.raises(ValueError, match="At least two observations"):
        effective_trials(returns)

    with pytest.raises(ValueError, match="At least two observations"):
        sharpe_spread(returns)


def test_constant_strategy():
    returns = np.array([
        [0.1, 0.02],
        [0.1, 0.03],
        [0.1, 0.04]
    ])

    with pytest.raises(ValueError, match="constant strategy"):
        effective_trials(returns)

    with pytest.raises(ValueError, match="constant strategy"):
        sharpe_spread(returns)


def test_single_strategy_effective_trials():
    returns = np.array([
        [0.01],
        [0.02],
        [-0.01],
        [0.03]
    ])

    assert effective_trials(returns) == 1.0


def test_single_strategy_sharpe_spread():
    returns = np.array([
        [0.01],
        [0.02],
        [-0.01],
        [0.03]
    ])

    with pytest.raises(ValueError, match="At least two strategies"):
        sharpe_spread(returns)


def test_nan_rows_removed():
    clean_returns = np.array([
        [0.01, 0.02],
        [0.03, -0.01],
        [-0.02, 0.04],
        [0.02, 0.01]
    ])

    returns_with_nan = np.vstack([
        clean_returns,
        [np.nan, 0.05],
        [0.03, np.nan]
    ])

    assert np.isclose(
        effective_trials(returns_with_nan),
        effective_trials(clean_returns),
        atol=1e-12,
        rtol=0
    )

    assert np.isclose(
        sharpe_spread(returns_with_nan),
        sharpe_spread(clean_returns),
        atol=1e-12,
        rtol=0
    )


def test_no_strategies():
    returns = np.empty((10, 0))

    with pytest.raises(ValueError, match="At least one strategy"):
        effective_trials(returns)

    with pytest.raises(ValueError, match="At least one strategy"):
        sharpe_spread(returns)


def test_infinite_returns():
    returns = np.array([
        [0.01, 0.02],
        [np.inf, 0.03],
        [0.02, 0.04]
    ])

    with pytest.raises(ValueError, match="finite values"):
        effective_trials(returns)

    with pytest.raises(ValueError, match="finite values"):
        sharpe_spread(returns)
