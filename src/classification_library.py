from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import zipfile
import pandas as pd
import numpy as np


POLLUTANT_COLS = ["pm2.5", "pm10", "so2", "no2", "co", "o3"]
METEO_COLS = ["TEMP", "PRES", "DEWP", "RAIN", "wd", "WSPM"]


def load_prsa_from_zip(zip_path: str | Path) -> pd.DataFrame:
    """Load Beijing PRSA2017 multi-site dataset from a ZIP file containing 12 station CSVs.

    Expected: data/raw/PRSA2017_Data_20130301-20170228.zip
    """
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip not found: {zip_path}")

    frames: list[pd.DataFrame] = []
    with zipfile.ZipFile(zip_path, "r") as z:
        csv_names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError("No CSV files found inside zip.")
        for name in sorted(csv_names):
            with z.open(name) as f:
                df = pd.read_csv(f)
            # station name: either in file name or in column 'station'
            if "station" not in df.columns:
                # derive from filename: PRSA_Data_Aotizhongxin_20130301-20170228.csv
                station = Path(name).stem.split("_")[2] if "_" in Path(name).stem else Path(name).stem
                df["station"] = station
            frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    return out


def add_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ["year", "month", "day", "hour"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
    df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]], errors="coerce")
    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Light cleaning:
    - parse datetime
    - sort by station & datetime
    - unify column names to lower for pollutants (pm2.5 etc.)
    """
    df = df.copy()

    # Normalize common pollutant column spellings
    rename_map = {}
    for c in df.columns:
        lc = c.lower()
        if lc in ["pm2.5", "pm10", "so2", "no2", "co", "o3"]:
            rename_map[c] = lc
    if rename_map:
        df = df.rename(columns=rename_map)

    df = add_datetime(df)
    df = df.dropna(subset=["datetime"])
    df = df.sort_values(["station", "datetime"]).reset_index(drop=True)
    return df


def ensure_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure hourly frequency per station by reindexing to hourly grid.
    Does not aggressively impute; leaves NaN for missing hours.
    """
    if "datetime" not in df.columns:
        df = add_datetime(df)

    stations = []
    for st, d in df.groupby("station"):
        d = d.sort_values("datetime").set_index("datetime")
        full = pd.date_range(d.index.min(), d.index.max(), freq="H")
        d = d.reindex(full)
        d["station"] = st
        # re-add time columns for compatibility
        d["year"] = d.index.year
        d["month"] = d.index.month
        d["day"] = d.index.day
        d["hour"] = d.index.hour
        d = d.reset_index().rename(columns={"index": "datetime"})
        stations.append(d)

    return pd.concat(stations, ignore_index=True)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "datetime" not in df.columns:
        df = add_datetime(df)
    dt = pd.to_datetime(df["datetime"])
    df["hour_of_day"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek
    df["month_of_year"] = dt.dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


def add_lag_features(df: pd.DataFrame, lags: list[int], group_col: str = "station", target_col: str = "pm2.5") -> pd.DataFrame:
    df = df.copy()
    if "datetime" not in df.columns:
        df = add_datetime(df)
    df = df.sort_values([group_col, "datetime"])
    for lag in lags:
        df[f"{target_col}_lag{lag}"] = df.groupby(group_col)[target_col].shift(lag)
    return df


def save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)
