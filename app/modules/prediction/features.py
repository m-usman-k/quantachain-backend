"""Feature engineering for the forecasting engine (pure numpy).

Every indicator operates on 1h candles sorted by ``open_time``. Columns are
scale-free (returns, ratios, z-scores, bounded oscillators) so models are not
dominated by price level and can be compared across assets. Warm-up values are
NaN; ``FeatureFrame.filled()`` replaces them for models that cannot take NaNs.

The exponential recursions (EMA, Wilder smoothing) use a blocked closed form
rather than a per-element Python loop::

    ema_t = (1-a)^t * ema_0 + a * sum_{k=1..t} (1-a)^(t-k) * x_k

evaluated per block of 256 observations with cumulative sums, which keeps the
powers of ``(1-a)`` far from floating-point under/overflow.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from app.core.timeutils import ensure_utc
from app.modules.market.models import Candle

PERIODS_PER_YEAR_1H = 24 * 365

# Columns in the order they appear in ``FeatureFrame.values``.
FEATURE_COLUMNS: tuple[str, ...] = (
    "log_return",
    "vol_12h",
    "vol_48h",
    "rsi_14",
    "macd_hist",
    "ema20_ratio",
    "ema50_ratio",
    "bb_pct_b",
    "volume_z",
    "atr_14",
    "atr_pct",
    "sentiment_index",
    "sentiment_z",
    "netflow_usd",
    "netflow_z",
)

# Exogenous inputs for the LSTM (log return first, then standardised indicators).
MODEL_FEATURES: tuple[str, ...] = (
    "log_return",
    "vol_12h",
    "vol_48h",
    "rsi_14",
    "macd_hist",
    "ema20_ratio",
    "ema50_ratio",
    "bb_pct_b",
    "volume_z",
    "sentiment_z",
    "netflow_z",
)

SENTIMENT_NEUTRAL = 50.0
SENTIMENT_SIGMA = 20.0  # one label band on the 0..100 index counts as one sigma
NETFLOW_WINDOW = 168  # one week of hourly observations
_EWM_BLOCK = 256


# ------------------------------------------------------------------ primitives
def ewm(values: np.ndarray, alpha: float, *, seed: float | None = None) -> np.ndarray:
    """Exponentially weighted recursion ``s_t = alpha*x_t + (1-alpha)*s_{t-1}`` (blocked closed form).

    Without ``seed`` the recursion starts at ``x_0``; with it the first output already
    blends ``x_0`` into the seed. NaN inputs propagate from their position onwards.
    """
    x = np.asarray(values, dtype=float)
    n = x.size
    out = np.empty(n, dtype=float)
    if n == 0:
        return out
    decay = 1.0 - alpha
    if seed is None:
        state = float(x[0])
        out[0] = state
        start = 1
    else:
        state = float(seed)
        start = 0
    for begin in range(start, n, _EWM_BLOCK):
        chunk = x[begin : begin + _EWM_BLOCK]
        powers = decay ** np.arange(1, chunk.size + 1)
        weighted = np.cumsum(chunk / powers) * powers
        block = powers * state + alpha * weighted
        out[begin : begin + chunk.size] = block
        state = float(block[-1])
    return out


def ema(values: np.ndarray, period: int) -> np.ndarray:
    """Classic EMA with ``alpha = 2 / (period + 1)``."""
    return ewm(values, 2.0 / (period + 1.0))


def wilder(values: np.ndarray, period: int) -> np.ndarray:
    """Wilder smoothing (``alpha = 1/period``) seeded with the simple mean of the first ``period`` values."""
    x = np.asarray(values, dtype=float)
    out = np.full(x.size, np.nan)
    if x.size < period:
        return out
    seed = float(np.mean(x[:period]))
    out[period - 1] = seed
    if x.size > period:
        out[period:] = ewm(x[period:], 1.0 / period, seed=seed)
    return out


def log_returns(close: np.ndarray) -> np.ndarray:
    """``r_t = ln(c_t / c_{t-1})``; the first element is NaN."""
    c = np.asarray(close, dtype=float)
    out = np.full(c.size, np.nan)
    if c.size > 1:
        out[1:] = np.diff(np.log(c))
    return out


def rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    """Sample standard deviation over a trailing ``window`` (NaN during warm-up)."""
    x = np.asarray(values, dtype=float)
    out = np.full(x.size, np.nan)
    if window < 2 or x.size < window:
        return out
    out[window - 1 :] = sliding_window_view(x, window).std(axis=1, ddof=1)
    return out


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    out = np.full(x.size, np.nan)
    if window < 1 or x.size < window:
        return out
    out[window - 1 :] = sliding_window_view(x, window).mean(axis=1)
    return out


def zscore(values: np.ndarray, window: int) -> np.ndarray:
    """``(x - rolling_mean) / rolling_std``; zero where the window has no dispersion."""
    x = np.asarray(values, dtype=float)
    mean = rolling_mean(x, window)
    std = rolling_std(x, window)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (x - mean) / std
    z[(std == 0) & ~np.isnan(mean)] = 0.0
    return z


# ------------------------------------------------------------------ indicators
def rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Relative Strength Index in [0, 100] with Wilder smoothing of gains and losses."""
    c = np.asarray(close, dtype=float)
    out = np.full(c.size, np.nan)
    if c.size <= period:
        return out
    delta = np.diff(c)
    gains = wilder(np.maximum(delta, 0.0), period)
    losses = wilder(np.maximum(-delta, 0.0), period)
    with np.errstate(invalid="ignore", divide="ignore"):
        rs = gains / losses
        value = 100.0 - 100.0 / (1.0 + rs)
    value = np.where(losses == 0, np.where(gains > 0, 100.0, 50.0), value)
    out[1:] = value
    out[:period] = np.nan
    return np.clip(out, 0.0, 100.0, out=out)


def macd_histogram(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> np.ndarray:
    """MACD histogram ``(EMA_fast - EMA_slow) - EMA_signal(MACD)`` divided by price (scale-free)."""
    c = np.asarray(close, dtype=float)
    line = ema(c, fast) - ema(c, slow)
    hist = (line - ema(line, signal)) / c
    hist[: min(c.size, slow + signal - 2)] = np.nan
    return hist


def ema_ratio(close: np.ndarray, period: int) -> np.ndarray:
    """``close / EMA(period) - 1``: positive when price trades above its moving average."""
    c = np.asarray(close, dtype=float)
    ratio = c / ema(c, period) - 1.0
    ratio[: min(c.size, period - 1)] = np.nan
    return ratio


def bollinger_pct_b(close: np.ndarray, window: int = 20, k: float = 2.0) -> np.ndarray:
    """Bollinger %b: 0 at the lower band, 1 at the upper band (0.5 when the band has no width)."""
    c = np.asarray(close, dtype=float)
    mid = rolling_mean(c, window)
    sd = rolling_std(c, window)
    width = 2.0 * k * sd
    with np.errstate(invalid="ignore", divide="ignore"):
        pct = (c - (mid - k * sd)) / width
    pct[(width == 0) & ~np.isnan(mid)] = 0.5
    return pct


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Average True Range (Wilder) where ``TR = max(h-l, |h-c_prev|, |l-c_prev|)``."""
    h = np.asarray(high, dtype=float)
    lo = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)
    tr = h - lo
    if c.size > 1:
        prev = c[:-1]
        tr[1:] = np.maximum.reduce([h[1:] - lo[1:], np.abs(h[1:] - prev), np.abs(lo[1:] - prev)])
    return wilder(tr, period)


# ------------------------------------------------------------- exogenous series
def _epoch(times: Sequence[datetime]) -> np.ndarray:
    return np.array([ensure_utc(t).timestamp() for t in times], dtype=float)


def forward_fill(
    source_times: Sequence[datetime], source_values: Sequence[float], target_times: Sequence[datetime]
) -> np.ndarray:
    """Align an irregular series onto ``target_times`` using the latest observation at or before each time."""
    out = np.full(len(target_times), np.nan)
    if not source_times or not target_times:
        return out
    src_t = _epoch(source_times)
    order = np.argsort(src_t, kind="stable")
    src_t = src_t[order]
    src_v = np.asarray(source_values, dtype=float)[order]
    idx = np.searchsorted(src_t, _epoch(target_times), side="right") - 1
    valid = idx >= 0
    out[valid] = src_v[idx[valid]]
    return out


def sentiment_z(index: np.ndarray) -> np.ndarray:
    """Sentiment index 0..100 mapped to sigma units: ``(index - 50) / 20`` (90 -> +2, 10 -> -2)."""
    return (np.asarray(index, dtype=float) - SENTIMENT_NEUTRAL) / SENTIMENT_SIGMA


def netflow_z(values: np.ndarray, window: int = NETFLOW_WINDOW) -> np.ndarray:
    """Exchange-netflow z-score against its trailing week; falls back to the full-sample statistics early on."""
    x = np.asarray(values, dtype=float)
    z = zscore(x, window)
    observed = ~np.isnan(x)
    if observed.sum() >= 2:
        mean = float(np.nanmean(x))
        std = float(np.nanstd(x, ddof=1))
        fallback = (x - mean) / std if std > 0 else np.zeros_like(x)
        z = np.where(np.isnan(z) & observed, fallback, z)
    return z


# ------------------------------------------------------------------ frame
@dataclass(slots=True)
class FeatureFrame:
    """Named feature matrix aligned with candle open times (rows = candles, columns = ``FEATURE_COLUMNS``)."""

    index: list[datetime]
    columns: list[str]
    values: np.ndarray  # shape (n, len(columns)), float, NaN where undefined

    def __len__(self) -> int:
        return len(self.index)

    def column(self, name: str) -> np.ndarray:
        return self.values[:, self.columns.index(name)]

    def matrix(self, columns: Sequence[str] | None = None, *, fill: float | None = 0.0) -> np.ndarray:
        """Sub-matrix of ``columns``; NaNs replaced with ``fill`` unless ``fill`` is None."""
        cols = list(columns) if columns is not None else self.columns
        out = np.column_stack([self.column(c) for c in cols]) if cols else np.empty((len(self), 0))
        if fill is not None:
            out = np.where(np.isnan(out), fill, out)
        return out

    def filled(self, fill: float = 0.0) -> FeatureFrame:
        return FeatureFrame(self.index, list(self.columns), np.where(np.isnan(self.values), fill, self.values))

    def latest(self, *, digits: int = 6) -> dict[str, float | None]:
        """Last row as JSON-safe primitives (NaN -> None)."""
        if not len(self):
            return dict.fromkeys(self.columns)
        row = self.values[-1]
        return {
            name: (None if math.isnan(v) else round(float(v), digits))
            for name, v in zip(self.columns, row, strict=True)
        }

    def last_value(self, name: str, default: float = 0.0) -> float:
        value = float(self.column(name)[-1]) if len(self) else math.nan
        return default if math.isnan(value) else value


def candle_arrays(candles: Sequence[Candle]) -> dict[str, Any]:
    """Columnar numpy view of a candle list (sorted by open time)."""
    ordered = sorted(candles, key=lambda c: c.open_time)
    return {
        "times": [c.open_time for c in ordered],
        "open": np.array([c.open for c in ordered], dtype=float),
        "high": np.array([c.high for c in ordered], dtype=float),
        "low": np.array([c.low for c in ordered], dtype=float),
        "close": np.array([c.close for c in ordered], dtype=float),
        "volume": np.array([c.volume for c in ordered], dtype=float),
    }


def build_features(
    candles: Sequence[Candle],
    *,
    sentiment: Sequence[tuple[datetime, float]] | None = None,
    netflow: Sequence[tuple[datetime, float]] | None = None,
) -> FeatureFrame:
    """Build the full feature frame from 1h candles plus optional sentiment / exchange-netflow series.

    ``sentiment`` holds ``(ts, index 0..100)`` snapshots (Module 4 asset scope), ``netflow``
    holds ``(ts, exchange_netflow_usd)`` points (Module 3). Both are forward-filled onto the
    candle grid; when absent the related columns are NaN (later filled with 0 = neutral).
    """
    arrays = candle_arrays(candles)
    close, high, low, volume = arrays["close"], arrays["high"], arrays["low"], arrays["volume"]
    times: list[datetime] = arrays["times"]
    n = close.size
    returns = log_returns(close)
    atr_values = atr(high, low, close, 14)

    def rolling_vol(window: int) -> np.ndarray:
        out = np.full(n, np.nan)
        if n > window:
            out[1:] = rolling_std(returns[1:], window)
        return out

    columns: dict[str, np.ndarray] = {
        "log_return": returns,
        "vol_12h": rolling_vol(12),
        "vol_48h": rolling_vol(48),
        "rsi_14": rsi(close, 14),
        "macd_hist": macd_histogram(close),
        "ema20_ratio": ema_ratio(close, 20),
        "ema50_ratio": ema_ratio(close, 50),
        "bb_pct_b": bollinger_pct_b(close),
        "volume_z": zscore(volume, 48),
        "atr_14": atr_values,
        "atr_pct": atr_values / np.where(close > 0, close, np.nan),
    }

    if sentiment:
        index = forward_fill([t for t, _ in sentiment], [v for _, v in sentiment], times)
        columns["sentiment_index"] = index
        columns["sentiment_z"] = sentiment_z(index)
    else:
        columns["sentiment_index"] = np.full(n, np.nan)
        columns["sentiment_z"] = np.full(n, np.nan)

    if netflow:
        flow = forward_fill([t for t, _ in netflow], [v for _, v in netflow], times)
        columns["netflow_usd"] = flow
        columns["netflow_z"] = netflow_z(flow)
    else:
        columns["netflow_usd"] = np.full(n, np.nan)
        columns["netflow_z"] = np.full(n, np.nan)

    values = np.column_stack([columns[name] for name in FEATURE_COLUMNS]) if n else np.empty((0, len(FEATURE_COLUMNS)))
    values = np.where(np.isinf(values), np.nan, values)
    return FeatureFrame(index=times, columns=list(FEATURE_COLUMNS), values=values)


__all__ = [
    "FEATURE_COLUMNS",
    "MODEL_FEATURES",
    "PERIODS_PER_YEAR_1H",
    "FeatureFrame",
    "atr",
    "bollinger_pct_b",
    "build_features",
    "candle_arrays",
    "ema",
    "ema_ratio",
    "ewm",
    "forward_fill",
    "log_returns",
    "macd_histogram",
    "netflow_z",
    "rolling_mean",
    "rolling_std",
    "rsi",
    "sentiment_z",
    "wilder",
    "zscore",
]
