"""Turn raw Statcast rows into the shared feature table (data/statcast_features.parquet).

Usage: python scripts/prepare_statcast.py 2024 2025
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from features import FEATURES, LABELS, add_relative

ROOT = Path(__file__).resolve().parents[1]
MIN_PITCHES = 100  # skip position players pitching and cup-of-coffee arms


def season(year):
    raw = pd.read_parquet(ROOT / "data" / "raw" / f"statcast_{year}.parquet")
    raw = raw.dropna(subset=["pitch_type", "release_speed", "release_spin_rate", "pfx_x", "pfx_z", "spin_axis"])
    # Statcast pfx is in feet, from the catcher's view (positive = toward first base).
    # A righty's arm side is toward third base, so his arm-side break is -pfx_x.
    # Lefties' spin axis is mirrored so both sides share one scale.
    righty = raw["p_throws"] == "R"
    df = pd.DataFrame({
        "pitcher": raw["pitcher"].astype(str) + f"_{year}",  # one fastball reference per season
        "pid": raw["pitcher"], "name": raw["player_name"], "season": year,
        "pitch_type": raw["pitch_type"],
        "velo": raw["release_speed"], "spin": raw["release_spin_rate"],
        "ivb": raw["pfx_z"] * 12, "hb": raw["pfx_x"] * 12 * np.where(righty, -1, 1),
        "axis": np.where(righty, raw["spin_axis"], 360 - raw["spin_axis"]),
    })
    counts = df["pitcher"].value_counts()
    df = df[df["pitcher"].isin(counts[counts >= MIN_PITCHES].index)]
    return add_relative(df)  # computed on every pitch, before rare types are dropped


years = [int(y) for y in sys.argv[1:]] or [2024, 2025]
df = pd.concat([season(y) for y in years], ignore_index=True)
df["label"] = df["pitch_type"].map(LABELS)
df = df.dropna(subset=["label"] + FEATURES)
df.to_parquet(ROOT / "data" / "statcast_features.parquet", index=False)
print(f"{len(df):,} pitches, {df['pid'].nunique():,} pitchers, seasons {years}")
print(df["label"].value_counts().to_string())
