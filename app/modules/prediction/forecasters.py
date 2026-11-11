"""Price forecasters: the always-available statistical ensemble and the optional TensorFlow LSTM.

Both implement the ``Forecaster`` protocol so the ensemble layer can swap them
transparently. ``StatisticalForecaster`` works on log prices and is fully
deterministic for a given seed; ``LSTMForecaster`` imports TensorFlow lazily and
only when a trained model exists on disk.
"""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from statistics import NormalDist
from typing import Any, Protocol

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from app.modules.prediction.features import MODEL_FEATURES

MIN_FIT_POINTS = 30

# Damped Holt grid (alpha = level smoothing, beta = trend smoothing, phi = damping).
HOLT_ALPHAS: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 0.95)
HOLT_BETAS: tuple[float, ...] = (0.02, 0.05, 0.1, 0.2)
HOLT_PHIS: tuple[float, ...] = (0.85, 0.95)
HOLT_WARMUP = 10  # one-step errors ignored while the state settles
AR_DRIFT_SHRINKAGE = 200.0  # pseudo-observations pulling the mean return towards zero
AR_RHO_LIMIT = 0.5

_TF_AVAILABLE: bool | None = None


@dataclass(slots=True)
class ForecastResult:
    """Point path with a prediction band, all in price units; ``residual_std`` is in log-return units."""

    points: list[float]
    lower: list[float]
    upper: list[float]
    residual_std: float
    model: str
    level: float = 0.9
    details: dict[str, float] = field(default_factory=dict)


class Forecaster(Protocol):
    name: str

    def fit(self, closes: np.ndarray, features: np.ndarray | None = None) -> Forecaster: ...

    def predict(self, steps: int) -> ForecastResult: ...


# ------------------------------------------------------------------ statistical
@dataclass(slots=True)
class _HoltState:
    alpha: float
    beta: float
    phi: float
    level: float
    trend: float
    fitted: np.ndarray  # one-step-ahead predictions of y[t] (index 0 = y[0])

    def path(self, steps: int) -> np.ndarray:
        """``y_{n+k} = l_n + b_n * sum_{j=1..k} phi^j`` for ``k = 1..steps``."""
        k = np.arange(1, steps + 1, dtype=float)
        damping = self.phi * (1.0 - self.phi**k) / (1.0 - self.phi) if self.phi < 1 else k
        return self.level + self.trend * damping


@dataclass(slots=True)
class _ARState:
    mu: float
    rho: float
    last_return: float
    last_level: float
    fitted: np.ndarray  # predictions of y[t] for t >= 2 (same length as y, NaN for t < 2)

    def path(self, steps: int) -> np.ndarray:
        """``r_{n+j} = mu + rho^j (r_n - mu)``; the log-price path is the cumulative sum from ``y_n``."""
        j = np.arange(1, steps + 1, dtype=float)
        increments = self.mu + (self.rho**j) * (self.last_return - self.mu)
        return self.last_level + np.cumsum(increments)


def _fit_holt(y: np.ndarray) -> _HoltState:
    """Grid search over (alpha, beta, phi) minimising one-step SSE; the recursion runs vectorised over the grid."""
    grid = np.array(list(itertools.product(HOLT_ALPHAS, HOLT_BETAS, HOLT_PHIS)), dtype=float)
    a, b, p = grid[:, 0], grid[:, 1], grid[:, 2]
    n, m = y.size, grid.shape[0]
    level = np.full(m, y[0])
    trend = np.full(m, y[1] - y[0])
    sse = np.zeros(m)
    preds = np.empty((n, m))
    preds[0] = y[0]
    for t in range(1, n):
        pred = level + p * trend
        preds[t] = pred
        err = y[t] - pred
        if t >= HOLT_WARMUP:
            sse += err * err
        new_level = a * y[t] + (1.0 - a) * pred
        trend = b * (new_level - level) + (1.0 - b) * p * trend
        level = new_level
    best = int(np.argmin(sse))
    return _HoltState(
        alpha=float(a[best]),
        beta=float(b[best]),
        phi=float(p[best]),
        level=float(level[best]),
        trend=float(trend[best]),
        fitted=preds[:, best],
    )


def _fit_ar1(y: np.ndarray) -> _ARState:
    """AR(1) on log returns by OLS with the mean shrunk towards zero (``mu = mean * n / (n + k)``)."""
    r = np.diff(y)
    n = r.size
    mu = float(r.mean() * n / (n + AR_DRIFT_SHRINKAGE))
    x = r[:-1] - mu
    z = r[1:] - mu
    denom = float(x @ x)
    rho = float(np.clip((x @ z) / denom, -AR_RHO_LIMIT, AR_RHO_LIMIT)) if denom > 0 else 0.0
    fitted = np.full(y.size, np.nan)
    fitted[2:] = y[1:-1] + mu + rho * (r[:-1] - mu)
    return _ARState(mu=mu, rho=rho, last_return=float(r[-1]), last_level=float(y[-1]), fitted=fitted)


class StatisticalForecaster:
    """Equal-weight ensemble of a damped Holt trend and an AR(1) return model on log prices.

    Prediction intervals come from a residual bootstrap: centred one-step residuals of the
    ensemble are resampled with replacement, accumulated over the horizon and the
    ``(1-level)/2`` / ``1-(1-level)/2`` quantiles of the simulated paths form the band.
    The per-step drift is capped at ``drift_cap_sigma`` residual standard deviations so a
    short burst cannot be extrapolated into an implausible 24h move.
    """

    name = "statistical"

    def __init__(
        self,
        *,
        seed: int = 7,
        level: float = 0.90,
        bootstrap_samples: int = 500,
        drift_cap_sigma: float = 2.5,
        max_history: int = 1000,
    ) -> None:
        self.seed = seed
        self.level = level
        self.bootstrap_samples = bootstrap_samples
        self.drift_cap_sigma = drift_cap_sigma
        self.max_history = max_history
        self._holt: _HoltState | None = None
        self._ar: _ARState | None = None
        self._residuals: np.ndarray | None = None
        self._last_log: float = 0.0

    def fit(self, closes: np.ndarray, features: np.ndarray | None = None) -> StatisticalForecaster:
        c = np.asarray(closes, dtype=float)[-self.max_history :]
        if c.size < MIN_FIT_POINTS or not np.all(np.isfinite(c)) or np.any(c <= 0):
            raise ValueError(f"StatisticalForecaster needs at least {MIN_FIT_POINTS} positive finite closes")
        y = np.log(c)
        self._holt = _fit_holt(y)
        self._ar = _fit_ar1(y)
        combined = 0.5 * (self._holt.fitted + self._ar.fitted)
        residuals = (y - combined)[2:]
        residuals = residuals[np.isfinite(residuals)]
        self._residuals = residuals - residuals.mean() if residuals.size else np.zeros(1)
        self._last_log = float(y[-1])
        return self

    def predict(self, steps: int) -> ForecastResult:
        if self._holt is None or self._ar is None or self._residuals is None:
            raise RuntimeError("fit() must be called before predict()")
        steps = max(1, int(steps))
        residual_std = float(self._residuals.std(ddof=1)) if self._residuals.size > 1 else 0.0
        path = 0.5 * (self._holt.path(steps) + self._ar.path(steps))
        increments = np.diff(np.concatenate([[self._last_log], path]))
        cap = self.drift_cap_sigma * residual_std
        if cap > 0:
            increments = np.clip(increments, -cap, cap)
        path = self._last_log + np.cumsum(increments)

        rng = np.random.default_rng(self.seed)
        idx = rng.integers(0, self._residuals.size, size=(self.bootstrap_samples, steps))
        simulated = path[None, :] + np.cumsum(self._residuals[idx], axis=1)
        tail = (1.0 - self.level) / 2.0
        lower = np.quantile(simulated, tail, axis=0)
        upper = np.quantile(simulated, 1.0 - tail, axis=0)
        floor = 1e-6 * np.arange(1, steps + 1)  # keeps lower < point < upper on degenerate series
        lower = np.minimum(lower, path - floor)
        upper = np.maximum(upper, path + floor)
        return ForecastResult(
            points=[float(v) for v in np.exp(path)],
            lower=[float(v) for v in np.exp(lower)],
            upper=[float(v) for v in np.exp(upper)],
            residual_std=residual_std,
            model=self.name,
            level=self.level,
            details={
                "holt_alpha": self._holt.alpha,
                "holt_beta": self._holt.beta,
                "holt_phi": self._holt.phi,
                "holt_trend": self._holt.trend,
                "ar_mu": self._ar.mu,
                "ar_rho": self._ar.rho,
            },
        )


# ------------------------------------------------------------------------ LSTM
@dataclass(slots=True)
class LSTMTrainingResult:
    path: str
    samples: int
    epochs: int
    lookback: int
    features: list[str]
    metrics: dict[str, float]
    scaler: dict[str, Any]


def _standardise(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std = np.where(std > 1e-12, std, 1.0)
    return (matrix - mean) / std, mean, std


def _windows(matrix: np.ndarray, lookback: int) -> np.ndarray:
    """Sliding windows ``(samples, lookback, columns)`` where window ``i`` ends just before row ``i + lookback``."""
    return np.transpose(sliding_window_view(matrix, lookback, axis=0), (0, 2, 1))


class LSTMForecaster:
    """Sequential ``LSTM(64) -> Dropout(0.2) -> Dense(1)`` predicting the next standardised log return.

    Inputs are sliding windows of ``lookback`` rows of ``[log_return, *features]`` (``features``
    are the exogenous ``MODEL_FEATURES`` without ``log_return``), standardised with constants
    stored next to the model. Multi-step forecasts are
    produced recursively; exogenous features are held at their last observed row
    (documented limitation). The band is a normal approximation
    ``+/- z_level * residual_std * sqrt(k)`` using the validation RMSE of returns.
    Model files: ``{model_dir}/lstm_{symbol}_{interval}.keras`` plus a ``.json`` sidecar.
    """

    name = "lstm"

    def __init__(
        self,
        model_dir: Path | str,
        *,
        symbol: str,
        interval: str = "1h",
        lookback: int = 48,
        level: float = 0.90,
        features: tuple[str, ...] = MODEL_FEATURES[1:],
    ) -> None:
        self.model_dir = Path(model_dir)
        self.symbol = symbol.upper()
        self.interval = interval
        self.lookback = lookback
        self.level = level
        self.features = tuple(features)
        self._model: Any = None
        self._scaler: dict[str, Any] | None = None
        self._matrix: np.ndarray | None = None
        self._last_close: float = 0.0

    # -------------------------------------------------------------- availability
    @classmethod
    def available(cls) -> bool:
        """True when TensorFlow can be imported (cached after the first attempt)."""
        global _TF_AVAILABLE
        if _TF_AVAILABLE is None:
            try:
                import importlib

                importlib.import_module("tensorflow")
                _TF_AVAILABLE = True
            except Exception:  # pragma: no cover - depends on the environment
                _TF_AVAILABLE = False
        return _TF_AVAILABLE

    @classmethod
    def reset_availability_cache(cls) -> None:
        global _TF_AVAILABLE
        _TF_AVAILABLE = None

    @staticmethod
    def model_path(model_dir: Path | str, symbol: str, interval: str) -> Path:
        return Path(model_dir) / f"lstm_{symbol.upper()}_{interval}.keras"

    @property
    def path(self) -> Path:
        return self.model_path(self.model_dir, self.symbol, self.interval)

    @property
    def sidecar_path(self) -> Path:
        return self.path.with_suffix(".json")

    def has_model(self) -> bool:
        return self.path.exists() and self.sidecar_path.exists()

    # ------------------------------------------------------------------ helpers
    def _input_matrix(self, closes: np.ndarray, features: np.ndarray | None) -> np.ndarray:
        """``[log_return, *exogenous]`` per candle; ``features`` has one row per close (row t describes candle t)."""
        c = np.asarray(closes, dtype=float)
        returns = np.diff(np.log(c))
        if features is None or np.asarray(features).size == 0:
            return returns[:, None]
        f = np.asarray(features, dtype=float)
        if f.ndim == 1:
            f = f[:, None]
        if f.shape[0] != c.size:
            raise ValueError("features must have one row per close")
        f = np.where(np.isnan(f[1:]), 0.0, f[1:])  # drop the first row: it has no return
        return np.column_stack([returns, f])

    @staticmethod
    def _keras() -> Any:
        from tensorflow import keras  # type: ignore[import-not-found]

        return keras

    def _build(self, columns: int) -> Any:
        keras = self._keras()
        model = keras.Sequential(
            [
                keras.layers.Input(shape=(self.lookback, columns)),
                keras.layers.LSTM(64),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="mse")
        return model

    # ------------------------------------------------------------------- train
    def train(
        self,
        closes: np.ndarray,
        features: np.ndarray | None = None,
        *,
        epochs: int = 20,
        batch_size: int = 32,
        validation_split: float = 0.2,
    ) -> LSTMTrainingResult:
        """Fit the network on sliding windows, score the validation tail, save model + scaler sidecar."""
        matrix = self._input_matrix(closes, features)
        if matrix.shape[0] < self.lookback + 50:
            raise ValueError(f"Need at least {self.lookback + 50} returns to train the LSTM")
        scaled, mean, std = _standardise(matrix)
        windows = _windows(scaled[:-1], self.lookback)  # window i -> predicts scaled return at i + lookback
        targets = scaled[self.lookback :, 0]
        split = max(self.lookback, int(windows.shape[0] * (1.0 - validation_split)))
        model = self._build(matrix.shape[1])
        model.fit(
            windows[:split].astype(np.float32),
            targets[:split].astype(np.float32),
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(windows[split:].astype(np.float32), targets[split:].astype(np.float32))
            if windows.shape[0] - split > 0
            else None,
            shuffle=True,
            verbose=0,
        )
        predicted = model.predict(windows[split:].astype(np.float32), verbose=0).reshape(-1)
        pred_returns = predicted * std[0] + mean[0]
        true_returns = matrix[self.lookback + split :, 0]
        prev_close = np.asarray(closes, dtype=float)[self.lookback + split : -1]
        true_close = np.asarray(closes, dtype=float)[self.lookback + split + 1 :]
        pred_close = prev_close * np.exp(pred_returns)
        metrics = {
            "directional_accuracy": float(np.mean(np.sign(pred_returns) == np.sign(true_returns))),
            "mape": float(np.mean(np.abs(pred_close - true_close) / true_close) * 100.0),
            "rmse": float(np.sqrt(np.mean((pred_returns - true_returns) ** 2))),
            "validation_samples": float(true_returns.size),
        }
        scaler = {
            "mean": [float(v) for v in mean],
            "std": [float(v) for v in std],
            "lookback": self.lookback,
            "features": list(self.features),
            "columns": int(matrix.shape[1]),
            "residual_std": metrics["rmse"],
            "level": self.level,
        }
        self.model_dir.mkdir(parents=True, exist_ok=True)
        model.save(str(self.path))
        self.sidecar_path.write_text(json.dumps(scaler), encoding="utf-8")
        self._model, self._scaler = model, scaler
        return LSTMTrainingResult(
            path=str(self.path),
            samples=int(windows.shape[0]),
            epochs=epochs,
            lookback=self.lookback,
            features=list(self.features),
            metrics=metrics,
            scaler=scaler,
        )

    # ----------------------------------------------------------------- predict
    def load(self) -> LSTMForecaster:
        if self._model is None:
            if not self.has_model():
                raise FileNotFoundError(f"No trained LSTM for {self.symbol} {self.interval} at {self.path}")
            self._model = self._keras().models.load_model(str(self.path))
            self._scaler = json.loads(self.sidecar_path.read_text(encoding="utf-8"))
        return self

    def fit(self, closes: np.ndarray, features: np.ndarray | None = None) -> LSTMForecaster:
        """Load the trained model and keep the most recent window; no re-training happens here."""
        self.load()
        assert self._scaler is not None
        matrix = self._input_matrix(closes, features)
        if matrix.shape[1] != self._scaler["columns"]:
            raise ValueError("Feature layout differs from the trained model; re-train the LSTM")
        if matrix.shape[0] < self._scaler["lookback"]:
            raise ValueError("Not enough history for the LSTM lookback window")
        self._matrix = matrix
        self._last_close = float(np.asarray(closes, dtype=float)[-1])
        return self

    def predict(self, steps: int) -> ForecastResult:
        if self._model is None or self._scaler is None or self._matrix is None:
            raise RuntimeError("fit() must be called before predict()")
        mean = np.asarray(self._scaler["mean"], dtype=float)
        std = np.asarray(self._scaler["std"], dtype=float)
        lookback = int(self._scaler["lookback"])
        window = ((self._matrix - mean) / std)[-lookback:].copy()
        log_price = math.log(self._last_close)
        path: list[float] = []
        for _ in range(max(1, int(steps))):
            scaled = float(np.asarray(self._model(window[None].astype(np.float32), training=False))[0, 0])
            log_price += scaled * std[0] + mean[0]
            path.append(log_price)
            next_row = window[-1].copy()
            next_row[0] = scaled
            window = np.vstack([window[1:], next_row])
        arr = np.array(path)
        residual_std = float(self._scaler.get("residual_std", 0.0))
        z = NormalDist().inv_cdf(1.0 - (1.0 - self.level) / 2.0)
        half = z * residual_std * np.sqrt(np.arange(1, arr.size + 1)) + 1e-6
        return ForecastResult(
            points=[float(v) for v in np.exp(arr)],
            lower=[float(v) for v in np.exp(arr - half)],
            upper=[float(v) for v in np.exp(arr + half)],
            residual_std=residual_std,
            model=self.name,
            level=self.level,
        )


__all__ = [
    "MIN_FIT_POINTS",
    "ForecastResult",
    "Forecaster",
    "LSTMForecaster",
    "LSTMTrainingResult",
    "StatisticalForecaster",
]
