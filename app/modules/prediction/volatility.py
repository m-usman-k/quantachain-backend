"""Volatility estimators, next-24h forecast and regime classification (pure numpy).

Estimators (log returns ``r_t`` on 1h candles, annualised with ``sqrt(8760)``):

* realised close-to-close ``sigma = std(r)`` over 24h / 7d windows
* Parkinson ``sigma^2 = mean(ln(H/L)^2) / (4 ln 2)``
* EWMA (RiskMetrics) ``sigma^2_t = lambda * sigma^2_{t-1} + (1 - lambda) * r^2_{t-1}``, ``lambda = 0.94``
* GARCH(1,1)-lite ``sigma^2_t = omega + alpha r^2_{t-1} + beta sigma^2_{t-1}`` fitted by Gaussian
  maximum likelihood over a coarse (alpha, beta) grid with variance targeting
  ``omega = v_bar (1 - alpha - beta)``; multi-step ``sigma^2_{t+k} = v_bar + (alpha+beta)^(k-1) (sigma^2_{t+1} - v_bar)``.

The next-24h forecast averages the EWMA and GARCH variance paths. The regime comes from the
percentile of the current 30-day realised volatility (HV30) within the last 365 daily HV30
observations - the "historic volatility index": ``<25 low, <75 normal, <90 high, else extreme``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from app.modules.prediction.features import ewm
from app.modules.prediction.models import VolRegime

HOURS_PER_YEAR = 24.0 * 365.0
DAYS_PER_YEAR = 365.0
EWMA_LAMBDA = 0.94
HVI_WINDOW_DAYS = 365
HVI_ROLLING_DAYS = 30
MIN_HVI_OBSERVATIONS = 10
GARCH_MIN_RETURNS = 100
GARCH_ALPHAS = np.linspace(0.02, 0.20, 10)
GARCH_BETAS = np.linspace(0.70, 0.97, 10)
FORECAST_STEPS = 24
REGIME_PERCENTILES: tuple[tuple[float, VolRegime], ...] = (
    (25.0, VolRegime.LOW),
    (75.0, VolRegime.NORMAL),
    (90.0, VolRegime.HIGH),
)
ABSOLUTE_REGIMES: tuple[tuple[float, VolRegime], ...] = (  # annualised decimals, used without history
    (0.40, VolRegime.LOW),
    (0.80, VolRegime.NORMAL),
    (1.20, VolRegime.HIGH),
)


def annualise(sigma: float, periods_per_year: float) -> float:
    return float(sigma) * math.sqrt(periods_per_year)


def realized_vol(returns: np.ndarray, window: int | None = None) -> float:
    """Sample standard deviation of the last ``window`` returns (NaN with fewer than 2 points)."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if window is not None:
        r = r[-window:]
    return float(r.std(ddof=1)) if r.size >= 2 else math.nan


def parkinson_vol(high: np.ndarray, low: np.ndarray, window: int | None = None) -> float:
    """Range-based estimator ``sqrt(mean(ln(H/L)^2) / (4 ln 2))`` per period."""
    h = np.asarray(high, dtype=float)
    lo = np.asarray(low, dtype=float)
    if window is not None:
        h, lo = h[-window:], lo[-window:]
    valid = (h > 0) & (lo > 0) & (h >= lo)
    if valid.sum() == 0:
        return math.nan
    squared = np.log(h[valid] / lo[valid]) ** 2
    return float(math.sqrt(squared.mean() / (4.0 * math.log(2.0))))


def ewma_variance_path(returns: np.ndarray, lam: float = EWMA_LAMBDA) -> np.ndarray:
    """Conditional variances ``sigma^2_1..sigma^2_{n+1}`` (the last element is the next-period forecast)."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return np.array([math.nan])
    seed = float(r.var(ddof=1))
    return np.concatenate([[seed], ewm(r * r, 1.0 - lam, seed=seed)])


def ewma_vol(returns: np.ndarray, lam: float = EWMA_LAMBDA) -> float:
    """Next-period EWMA volatility (per period)."""
    return float(math.sqrt(ewma_variance_path(returns, lam)[-1]))


@dataclass(slots=True)
class GarchFit:
    omega: float
    alpha: float
    beta: float
    long_run_variance: float
    next_variance: float
    loglik: float

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta


def garch11_fit(returns: np.ndarray) -> GarchFit | None:
    """Grid maximum likelihood for GARCH(1,1) with variance targeting; None with too few returns."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < GARCH_MIN_RETURNS:
        return None
    r = r - r.mean()
    v_bar = float(r.var(ddof=1))
    if v_bar <= 0:
        return None
    a_grid, b_grid = np.meshgrid(GARCH_ALPHAS, GARCH_BETAS, indexing="ij")
    alpha = a_grid.ravel()
    beta = b_grid.ravel()
    keep = alpha + beta < 0.999
    alpha, beta = alpha[keep], beta[keep]
    omega = v_bar * (1.0 - alpha - beta)
    variance = np.full(alpha.size, v_bar)
    loglik = np.zeros(alpha.size)
    r2 = r * r
    for t in range(r.size):
        loglik -= 0.5 * (np.log(variance) + r2[t] / variance)
        variance = omega + alpha * r2[t] + beta * variance  # variance for t + 1
    best = int(np.argmax(loglik))
    return GarchFit(
        omega=float(omega[best]),
        alpha=float(alpha[best]),
        beta=float(beta[best]),
        long_run_variance=v_bar,
        next_variance=float(variance[best]),
        loglik=float(loglik[best]),
    )


def garch_variance_path(fit: GarchFit, steps: int = FORECAST_STEPS) -> np.ndarray:
    """``sigma^2_{t+k} = v_bar + (alpha + beta)^(k-1) * (sigma^2_{t+1} - v_bar)`` for ``k = 1..steps``."""
    k = np.arange(steps, dtype=float)
    return fit.long_run_variance + (fit.persistence**k) * (fit.next_variance - fit.long_run_variance)


def horizon_sigmas(step_variances: np.ndarray) -> np.ndarray:
    """Cumulative sigma over ``k`` steps: ``sqrt(sum_{j<=k} sigma^2_j)`` (log-return units)."""
    return np.sqrt(np.cumsum(np.asarray(step_variances, dtype=float)))


def rolling_daily_vol(daily_closes: np.ndarray, window: int = HVI_ROLLING_DAYS) -> np.ndarray:
    """Annualised rolling realised volatility of daily log returns (one value per complete window)."""
    c = np.asarray(daily_closes, dtype=float)
    if c.size <= window:
        return np.array([])
    r = np.diff(np.log(c))
    return sliding_window_view(r, window).std(axis=1, ddof=1) * math.sqrt(DAYS_PER_YEAR)


def daily_vol_from_hourly(hourly_returns: np.ndarray, hours: int = 24) -> np.ndarray:
    """Fallback daily series from hourly returns: ``sqrt(sum r^2)`` per complete 24h block, annualised."""
    r = np.asarray(hourly_returns, dtype=float)
    r = r[np.isfinite(r)]
    days = r.size // hours
    if days == 0:
        return np.array([])
    blocks = r[-days * hours :].reshape(days, hours)
    return np.sqrt((blocks**2).sum(axis=1)) * math.sqrt(DAYS_PER_YEAR)


def percentile_rank(history: np.ndarray, value: float) -> float:
    """Share of ``history`` at or below ``value`` in percent."""
    h = np.asarray(history, dtype=float)
    h = h[np.isfinite(h)]
    if h.size == 0 or not math.isfinite(value):
        return math.nan
    return float(100.0 * np.mean(h <= value))


def regime_for(percentile: float | None, annualised: float) -> VolRegime:
    """Regime from the HVI percentile; absolute annualised thresholds when no history exists."""
    if percentile is not None and math.isfinite(percentile):
        for bound, regime in REGIME_PERCENTILES:
            if percentile < bound:
                return regime
        return VolRegime.EXTREME
    for bound, regime in ABSOLUTE_REGIMES:
        if annualised < bound:
            return regime
    return VolRegime.EXTREME


@dataclass(slots=True)
class VolatilitySnapshot:
    """All volatility numbers for one symbol; volatilities are annualised decimals unless noted."""

    realized_24h: float
    realized_7d: float
    parkinson_24h: float
    ewma: float
    garch: float | None
    forecast_24h_annualised: float
    forecast_24h_horizon: float  # one-sigma move over the next 24h (decimal, not annualised)
    horizon_sigmas: list[float]  # cumulative hourly sigma for k = 1..24 (decimals)
    current_hv30: float
    percentile: float | None
    regime: VolRegime
    observations: int
    history_times: list[datetime] = field(default_factory=list)
    history_values: list[float] = field(default_factory=list)
    params: dict[str, float] = field(default_factory=dict)

    def horizon_sigma(self, steps: int) -> float:
        if not self.horizon_sigmas:
            return math.nan
        return self.horizon_sigmas[min(len(self.horizon_sigmas), max(1, steps)) - 1]


def _clean(value: float, default: float = 0.0) -> float:
    return float(value) if value is not None and math.isfinite(value) else default


def forecast_volatility(
    *,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    times: Sequence[datetime],
    daily_closes: np.ndarray | None = None,
    daily_times: Sequence[datetime] | None = None,
    steps: int = FORECAST_STEPS,
    lam: float = EWMA_LAMBDA,
) -> VolatilitySnapshot:
    """Full volatility picture from 1h candles plus (optionally) daily closes for the historic index."""
    c = np.asarray(close, dtype=float)
    returns = np.diff(np.log(c)) if c.size > 1 else np.array([])
    realized_24h = annualise(realized_vol(returns, 24), HOURS_PER_YEAR)
    realized_7d = annualise(realized_vol(returns, 168), HOURS_PER_YEAR)
    parkinson_24h = annualise(parkinson_vol(high, low, 24), HOURS_PER_YEAR)

    ewma_path = ewma_variance_path(returns, lam)
    ewma_next = float(ewma_path[-1]) if math.isfinite(ewma_path[-1]) else _clean(realized_vol(returns)) ** 2
    ewma_steps = np.full(steps, ewma_next)
    fit = garch11_fit(returns)
    params: dict[str, float] = {"ewma_lambda": lam}
    if fit is not None:
        garch_steps = garch_variance_path(fit, steps)
        blended = 0.5 * (ewma_steps + garch_steps)
        params.update({"omega": fit.omega, "alpha": fit.alpha, "beta": fit.beta, "persistence": fit.persistence})
        garch_annualised: float | None = annualise(math.sqrt(fit.next_variance), HOURS_PER_YEAR)
    else:
        blended = ewma_steps
        garch_annualised = None
    sigmas = horizon_sigmas(blended)
    horizon_move = float(sigmas[-1]) if sigmas.size else math.nan
    forecast_annualised = annualise(math.sqrt(float(blended.mean())), HOURS_PER_YEAR) if blended.size else math.nan

    # Historic volatility index: HV30 from daily closes, or daily realised vol blocks from hourly data.
    history_times: list[datetime] = []
    if daily_closes is not None and np.asarray(daily_closes).size > HVI_ROLLING_DAYS + 1:
        series = rolling_daily_vol(np.asarray(daily_closes, dtype=float), HVI_ROLLING_DAYS)
        if daily_times is not None and len(daily_times) == np.asarray(daily_closes).size:
            history_times = list(daily_times)[HVI_ROLLING_DAYS:]
    else:
        series = daily_vol_from_hourly(returns)
        if times and series.size:
            history_times = [
                times[min(len(times) - 1, len(times) - 1 - 24 * (series.size - 1 - i))] for i in range(series.size)
            ]
    series = series[-HVI_WINDOW_DAYS:]
    history_times = history_times[-series.size :] if series.size else []
    current_hv30 = float(series[-1]) if series.size else _clean(realized_7d, _clean(realized_24h))
    observations = int(series.size)
    percentile = percentile_rank(series, current_hv30) if observations >= MIN_HVI_OBSERVATIONS else None
    regime = regime_for(percentile, _clean(forecast_annualised, current_hv30))

    return VolatilitySnapshot(
        realized_24h=_clean(realized_24h),
        realized_7d=_clean(realized_7d, _clean(realized_24h)),
        parkinson_24h=_clean(parkinson_24h),
        ewma=annualise(math.sqrt(ewma_next), HOURS_PER_YEAR),
        garch=garch_annualised,
        forecast_24h_annualised=_clean(forecast_annualised),
        forecast_24h_horizon=_clean(horizon_move),
        horizon_sigmas=[float(v) for v in sigmas],
        current_hv30=current_hv30,
        percentile=None if percentile is None or not math.isfinite(percentile) else round(percentile, 2),
        regime=regime,
        observations=observations,
        history_times=history_times,
        history_values=[float(v) for v in series],
        params=params,
    )


__all__ = [
    "EWMA_LAMBDA",
    "FORECAST_STEPS",
    "HVI_ROLLING_DAYS",
    "HVI_WINDOW_DAYS",
    "GarchFit",
    "VolatilitySnapshot",
    "annualise",
    "daily_vol_from_hourly",
    "ewma_variance_path",
    "ewma_vol",
    "forecast_volatility",
    "garch11_fit",
    "garch_variance_path",
    "horizon_sigmas",
    "parkinson_vol",
    "percentile_rank",
    "realized_vol",
    "regime_for",
    "rolling_daily_vol",
]
