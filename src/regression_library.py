from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


@dataclass
class RegressionConfig:
    cutoff: str = "2017-01-01"
    horizon: int = 1
    lag_hours: tuple[int, ...] = (1, 3, 24)
    target_col: str = "pm2.5"


def prepare_regression_frame(df: pd.DataFrame, cfg: RegressionConfig) -> pd.DataFrame:
    """Prepare supervised dataset for forecasting:
    y(t+h) from X(t) using lag + time features.
    Requires columns: datetime, station, pm2.5 (target)
    """
    d = df.copy()
    d["datetime"] = pd.to_datetime(d["datetime"])
    d = d.sort_values(["station", "datetime"])

    # target shift
    d["pm25_target"] = d.groupby("station")[cfg.target_col].shift(-cfg.horizon)

    # lag features
    for lag in cfg.lag_hours:
        d[f"pm25_lag{lag}"] = d.groupby("station")[cfg.target_col].shift(lag)

    # time features
    dt = d["datetime"]
    d["hour"] = dt.dt.hour
    d["dow"] = dt.dt.dayofweek
    d["month"] = dt.dt.month

    feat_cols = [f"pm25_lag{lag}" for lag in cfg.lag_hours] + ["hour", "dow", "month"]
    d = d.dropna(subset=feat_cols + ["pm25_target"])
    return d, feat_cols


def time_split(df: pd.DataFrame, cutoff: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff_ts = pd.to_datetime(cutoff)
    train = df[df["datetime"] <= cutoff_ts].copy()
    test = df[df["datetime"] > cutoff_ts].copy()
    return train, test


def train_baseline_rf(train: pd.DataFrame, feat_cols: list[str]) -> RandomForestRegressor:
    X = train[feat_cols]
    y = train["pm25_target"]
    model = RandomForestRegressor(
        n_estimators=400,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    return model


def predict_and_eval(model, test: pd.DataFrame, feat_cols: list[str]) -> dict:
    Xte = test[feat_cols]
    yte = test["pm25_target"].values
    pred = model.predict(Xte)
    mae = float(mean_absolute_error(yte, pred))
    rmse = float(np.sqrt(mean_squared_error(yte, pred)))
    return {
        "mae": mae,
        "rmse": rmse,
        "n_test": int(len(test)),
    }, pred


def save_metrics(metrics: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
