from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error


@dataclass
class ArimaConfig:
    cutoff: str = "2017-01-01"
    horizon: int = 1  # lab default: 1-step ahead
    p_max: int = 3
    q_max: int = 3
    d: int | None = None  # if None, decide by tests
    max_train_points: int | None = None  # optional for speed
    target_col: str = "pm2.5"


def stationarity_tests(series: pd.Series) -> dict:
    x = series.dropna()
    if len(x) < 50:
        return {"adf_pvalue": None, "kpss_pvalue": None, "note": "Too few points"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adf_p = adfuller(x, autolag="AIC")[1]
        kpss_p = kpss(x, regression="c", nlags="auto")[1]
    return {"adf_pvalue": float(adf_p), "kpss_pvalue": float(kpss_p)}


def choose_d_from_tests(series: pd.Series, max_d: int = 2) -> tuple[int, list[dict]]:
    """Simple rule:
    - prefer smallest d such that ADF p < 0.05 AND KPSS p > 0.05 (both support stationarity)
    - otherwise increment d up to max_d
    """
    logs = []
    s = series.copy()
    for d in range(0, max_d + 1):
        res = stationarity_tests(s)
        res["d"] = d
        logs.append(res)
        adf_ok = (res["adf_pvalue"] is not None) and (res["adf_pvalue"] < 0.05)
        kpss_ok = (res["kpss_pvalue"] is not None) and (res["kpss_pvalue"] > 0.05)
        if adf_ok and kpss_ok:
            return d, logs
        s = s.diff().dropna()
    return max_d, logs


def grid_search_arima(train: pd.Series, d: int, p_max: int, q_max: int) -> dict:
    best = {"aic": np.inf, "order": None, "model": None}
    for p in range(p_max + 1):
        for q in range(q_max + 1):
            try:
                m = ARIMA(train, order=(p, d, q))
                r = m.fit()
                if r.aic < best["aic"]:
                    best = {"aic": float(r.aic), "order": (p, d, q), "model": r}
            except Exception:
                continue
    if best["order"] is None:
        raise RuntimeError("ARIMA grid search failed for all candidates.")
    return best


def rolling_forecast_1step(train: pd.Series, test: pd.Series, order: tuple[int, int, int], max_train_points: int | None = None) -> pd.Series:
    history = train.copy()
    preds = []
    idx = []
    for t, y in test.items():
        if max_train_points is not None and len(history) > max_train_points:
            history_use = history.iloc[-max_train_points:]
        else:
            history_use = history
        model = ARIMA(history_use, order=order).fit()
        yhat = float(model.forecast(steps=1).iloc[0])
        preds.append(yhat)
        idx.append(t)
        history = pd.concat([history, pd.Series([y], index=[t])])
    return pd.Series(preds, index=idx, name="forecast")


def eval_forecast(y_true: pd.Series, y_pred: pd.Series) -> dict:
    y_true = y_true.loc[y_pred.index]
    mae = float(mean_absolute_error(y_true.values, y_pred.values))
    rmse = float(np.sqrt(mean_squared_error(y_true.values, y_pred.values)))
    return {"mae": mae, "rmse": rmse, "n_test": int(len(y_pred))}


def save_metrics(metrics: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
