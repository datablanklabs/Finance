"""Research and hypothesis formation.

The tools here exist to answer a question *before* you write a strategy: does
this universe, over this period, actually reward the behaviour I am about to
code? Running a backtest first and reasoning backwards from the equity curve is
how you end up fitting noise with great conviction.

The centrepiece is :func:`ic_grid`, which measures the rank correlation
between a trailing-return signal and a forward return across a grid of
formation and holding horizons. Positive regions are momentum. Negative
regions are reversal. If your intended strategy sits in a region where the
sign is inconsistent, that is worth knowing before you spend a weekend on it.

:func:`expected_max_sharpe` is the antidote to the parameter sweep. Search
enough configurations and the best one looks good by construction; this tells
you how good "good by construction" is, so you can subtract it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .data import PricePanel

__all__ = [
    "forward_returns",
    "trailing_signal",
    "information_coefficient",
    "ic_summary",
    "ic_grid",
    "return_autocorrelation",
    "walk_forward_splits",
    "expected_max_sharpe",
    "sharpe_haircut",
    "ParameterSweep",
]

EULER_GAMMA = 0.5772156649015329


# ---------------------------------------------------------------------------
# Signal / forward return construction
# ---------------------------------------------------------------------------


def forward_returns(panel: PricePanel, horizon: int) -> pd.DataFrame:
    """Return over the *next* ``horizon`` bars, aligned to the decision bar.

    Row ``t`` holds the return from close ``t`` to close ``t + horizon``. The
    last ``horizon`` rows are NaN by construction -- if they were not, you
    would be looking at the future.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    c = panel.close
    return (c.shift(-horizon) / c) - 1.0


def trailing_signal(panel: PricePanel, lookback: int, skip: int = 0) -> pd.DataFrame:
    """Trailing return over ``lookback`` bars ending ``skip`` bars ago."""
    c = panel.close
    end = c.shift(skip)
    start = c.shift(skip + lookback)
    return (end / start) - 1.0


# ---------------------------------------------------------------------------
# Information coefficient
# ---------------------------------------------------------------------------


def information_coefficient(
    signal: pd.DataFrame,
    fwd: pd.DataFrame,
    method: str = "spearman",
    min_names: int = 8,
    step: int | None = None,
) -> pd.Series:
    """Cross-sectional rank correlation between signal and forward return.

    One number per date: how well the signal ordered the universe that day.

    ``step`` subsamples dates. Set it to the forward horizon to get
    non-overlapping observations -- overlapping windows are serially
    correlated, which inflates the t-statistic of the mean IC, sometimes by a
    factor of two or three. The default does this for you when you use
    :func:`ic_summary`.
    """
    common_cols = signal.columns.intersection(fwd.columns)
    idx = signal.index.intersection(fwd.index)
    sig, fw = signal.loc[idx, common_cols], fwd.loc[idx, common_cols]

    if step and step > 1:
        sig, fw = sig.iloc[::step], fw.iloc[::step]

    out = {}
    for date in sig.index:
        a, b = sig.loc[date], fw.loc[date]
        ok = a.notna() & b.notna()
        if int(ok.sum()) < min_names:
            continue
        x, y = a[ok].to_numpy(), b[ok].to_numpy()
        if np.std(x) == 0 or np.std(y) == 0:
            continue
        if method == "spearman":
            rho, _ = stats.spearmanr(x, y)
        elif method == "pearson":
            rho, _ = stats.pearsonr(x, y)
        elif method == "kendall":
            rho, _ = stats.kendalltau(x, y)
        else:
            raise ValueError(f"unknown method {method!r}")
        if np.isfinite(rho):
            out[date] = float(rho)
    return pd.Series(out, name="ic").sort_index()


def ic_summary(ic: pd.Series) -> pd.Series:
    """Mean IC with a t-statistic and information ratio.

    The t-stat is the honest headline. A mean IC of 0.03 sounds tiny but can be
    highly significant with enough independent observations; a mean IC of 0.08
    over 14 overlapping observations is noise.
    """
    ic = ic.dropna()
    n = len(ic)
    if n < 2:
        return pd.Series({"n_obs": n}, dtype=float)
    mean, sd = float(ic.mean()), float(ic.std(ddof=1))
    t = mean / (sd / np.sqrt(n)) if sd > 0 else np.nan
    return pd.Series(
        {
            "n_obs": float(n),
            "mean_ic": mean,
            "std_ic": sd,
            "t_stat": t,
            "p_value": float(2 * (1 - stats.t.cdf(abs(t), df=n - 1))) if np.isfinite(t) else np.nan,
            "ir": mean / sd if sd > 0 else np.nan,
            "hit_rate": float((ic > 0).mean()),
        }
    )


def ic_grid(
    panel: PricePanel,
    lookbacks: Sequence[int] = (5, 10, 21, 63, 126, 252),
    horizons: Sequence[int] = (1, 5, 10, 21, 63),
    skip: int = 0,
    method: str = "spearman",
    non_overlapping: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mean IC and t-stat for every (formation, holding) pair.

    Returns ``(mean_ic, t_stat)`` frames with lookbacks as rows and horizons as
    columns. Read it as a map of where the universe pays for momentum
    (positive) versus reversal (negative), and how confidently.
    """
    mean_rows, t_rows = {}, {}
    for lb in lookbacks:
        sig = trailing_signal(panel, lb, skip)
        m_row, t_row = {}, {}
        for h in horizons:
            fwd = forward_returns(panel, h)
            step = h if non_overlapping else None
            ic = information_coefficient(sig, fwd, method=method, step=step)
            summ = ic_summary(ic)
            m_row[h] = summ.get("mean_ic", np.nan)
            t_row[h] = summ.get("t_stat", np.nan)
        mean_rows[lb], t_rows[lb] = m_row, t_row

    mean_df = pd.DataFrame(mean_rows).T
    t_df = pd.DataFrame(t_rows).T
    for df in (mean_df, t_df):
        df.index.name = "lookback"
        df.columns.name = "horizon"
    return mean_df, t_df


# ---------------------------------------------------------------------------
# Autocorrelation
# ---------------------------------------------------------------------------


def return_autocorrelation(
    panel: PricePanel, lags: Iterable[int] = range(1, 22), pooled: bool = True
) -> pd.DataFrame:
    """Autocorrelation of daily returns at several lags.

    With ``pooled=True`` each symbol's series is standardised and stacked, so
    the estimate reflects the typical name rather than the index. Negative
    autocorrelation at short lags is the statistical signature of the reversal
    effect; positive at longer lags hints at trend.
    """
    rets = panel.returns()
    n_dates, n_symbols = rets.shape
    date_idx = np.repeat(np.arange(n_dates), n_symbols)
    # Standardise each symbol before pooling, which is what makes this "the
    # typical name" rather than "whichever name is loudest". Without it the
    # pooled correlation is dominated by the highest-variance symbols: on a
    # universe of one high-vol trending name and three low-vol mean-reverting
    # ones, the unstandardised pooled figure reads +0.57 while the typical
    # name is -0.29 -- not merely imprecise, the opposite sign, and this
    # function's whole documented use is reading that sign as momentum
    # versus reversal.
    sd = rets.std(ddof=0)
    z = (rets - rets.mean()) / sd.replace(0.0, np.nan)
    rows = {}
    for lag in lags:
        if pooled:
            a = z.shift(lag)
            x = a.to_numpy().ravel()
            y = z.to_numpy().ravel()
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 100:
                rows[lag] = {"autocorr": np.nan, "n": int(ok.sum())}
                continue
            rho = float(np.corrcoef(x[ok], y[ok])[0, 1])
            n = int(ok.sum())
            # n itself overstates independence: returns are strongly
            # cross-sectionally correlated (market-wide moves), so pooling
            # symbols x dates and treating every pair as an i.i.d. draw
            # inflates the effective sample size by roughly the symbol
            # count. The effective number of independent observations is
            # much closer to the number of distinct dates than to n.
            n_effective = int(np.unique(date_idx[ok]).size)
            se = 1.0 / np.sqrt(n_effective)
            rows[lag] = {
                "autocorr": rho, "t_stat": rho / se, "n": n,
                "n_effective": n_effective,
            }
        else:
            per = rets.apply(lambda s: s.autocorr(lag=lag))
            rows[lag] = {"autocorr": float(per.mean()), "n": int(per.notna().sum())}
    out = pd.DataFrame(rows).T
    out.index.name = "lag"
    return out


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------


def walk_forward_splits(
    dates: pd.DatetimeIndex,
    train_bars: int = 756,
    test_bars: int = 252,
    anchored: bool = True,
    step: int | None = None,
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Sequential train/test date ranges, never overlapping forward in time.

    ``anchored=True`` grows the training window from a fixed start; ``False``
    rolls a fixed-length window. Anchored uses more data, rolling adapts faster
    to regime change. Neither is obviously right.
    """
    step = step or test_bars
    out = []
    start = 0
    while True:
        train_end = start + train_bars
        test_end = train_end + test_bars
        if test_end > len(dates):
            break
        train_start = 0 if anchored else start
        out.append((dates[train_start:train_end], dates[train_end:test_end]))
        start += step
    return out


# ---------------------------------------------------------------------------
# Multiple testing
# ---------------------------------------------------------------------------


def expected_max_sharpe(n_trials: int, sharpe_dispersion: float) -> float:
    """Expected best Sharpe from ``n_trials`` configurations with *zero* edge.

    Uses the extreme-value approximation from Bailey and Lopez de Prado: given
    N independent trials whose Sharpe estimates have cross-sectional standard
    deviation ``sharpe_dispersion``, the maximum you expect to observe under
    the null of no skill is::

        sigma * [ (1 - g) * Z(1 - 1/N) + g * Z(1 - 1/(N*e)) ]

    with ``g`` the Euler-Mascheroni constant. Compare your sweep's winner to
    this number. If it does not clear it comfortably, you have measured your
    search effort, not an edge.
    """
    if n_trials < 2 or sharpe_dispersion <= 0:
        return 0.0
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(sharpe_dispersion * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def sharpe_haircut(observed_sharpe: float, n_trials: int, sharpe_dispersion: float) -> dict:
    """Adjust an observed Sharpe for the size of the search that produced it."""
    threshold = expected_max_sharpe(n_trials, sharpe_dispersion)
    return {
        "observed_sharpe": observed_sharpe,
        "n_trials": n_trials,
        "expected_max_under_null": threshold,
        "excess": observed_sharpe - threshold,
        "survives": bool(observed_sharpe > threshold),
    }


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------


@dataclass
class ParameterSweep:
    """Evaluate a callable across a parameter grid and collect the results.

    ``evaluate`` receives one kwargs dict and returns a mapping of metrics.
    Deliberately dumb: the value is in what you do with the output, namely
    looking at the *dispersion* of results rather than only the maximum. A
    strategy whose Sharpe collapses when you nudge a lookback by 10% is not a
    strategy, it is a coincidence.
    """

    grid: dict[str, Sequence]
    evaluate: Callable[..., dict]

    def combinations(self) -> list[dict]:
        keys = list(self.grid)
        out: list[dict] = [{}]
        for k in keys:
            out = [dict(base, **{k: v}) for base in out for v in self.grid[k]]
        return out

    def run(self, progress: bool = False) -> pd.DataFrame:
        rows = []
        combos = self.combinations()
        n_failed = 0
        for i, params in enumerate(combos, 1):
            if progress:
                print(f"  [{i}/{len(combos)}] {params}", flush=True)
            try:
                metrics = self.evaluate(**params)
            except Exception as exc:  # keep the sweep alive
                metrics = {"error": repr(exc)}
                n_failed += 1
            rows.append({**params, **metrics})
        # Say so. Swallowing the exception is right -- one bad corner of the
        # grid shouldn't cost you the other 35 results -- but swallowing it
        # *silently* means a sweep where most configurations blew up reads
        # as a clean, narrow sweep, and the dispersion this class exists to
        # show you is computed over whatever happened to survive.
        if n_failed:
            print(f"  WARNING: {n_failed} of {len(combos)} configurations "
                  f"raised and have no metrics; see the 'error' column. "
                  f"stability() reports only the {len(combos) - n_failed} "
                  f"that completed.", flush=True)
        return pd.DataFrame(rows)

    @staticmethod
    def stability(df: pd.DataFrame, metric: str = "sharpe") -> pd.Series:
        """Distribution of a metric across the grid.

        ``n_failed`` counts rows that carry an ``error`` instead of a
        metric. Read it before the rest: every other number here describes
        only the configurations that completed, so a healthy-looking spread
        over 6 survivors of a 36-config grid is not a healthy grid.
        """
        n_failed = (
            int(df["error"].notna().sum()) if "error" in df.columns else 0
        )
        # If every configuration raised, `metric` was never produced by
        # anything and the column does not exist -- df[metric] would raise a
        # bare KeyError naming the metric, which reads like "you asked for
        # the wrong column" rather than "your whole sweep failed". That is
        # the one case where this summary is most worth having.
        if metric not in df.columns:
            return pd.Series({"n": 0.0, "n_failed": float(n_failed)})
        s = pd.to_numeric(df[metric], errors="coerce").dropna()
        if s.empty:
            return pd.Series({"n": 0.0, "n_failed": float(n_failed)})
        return pd.Series(
            {
                "n": float(len(s)),
                "n_failed": float(n_failed),
                "best": float(s.max()),
                "median": float(s.median()),
                "worst": float(s.min()),
                "std": float(s.std(ddof=1)) if len(s) > 1 else np.nan,
                "frac_positive": float((s > 0).mean()),
            }
        )
