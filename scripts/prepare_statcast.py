"""Turn raw Statcast rows into the shared feature table (data/statcast_features.parquet)."""
import sys
from pathlib import Path

import pandas as pd

from features import FEATURES, LABELS, add_relative

ROOT = Path(__file__).resolve().parents[1]
MIN_PITCHES = 100  # skip position players pitching and cup-of-coffee arms

year = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
raw = pd.read_parquet(ROOT / "data" / "raw" / f"statcast_{year}.parquet")
raw = raw.dropna(subset=["pitch_type", "release_speed", "release_spin_rate", "pfx_x", "pfx_z"])

# Statcast pfx is in feet, from the catcher's view (positive = toward first base).
# A righty's arm side is toward third base, so his arm-side break is -pfx_x.
arm = raw["p_throws"].map({"R": -1, "L": 1})
df = pd.DataFrame({
    "pitcher": raw["pitcher"], "name": raw["player_name"], "throws": raw["p_throws"],
    "pitch_type": raw["pitch_type"],
    "velo": raw["release_speed"], "spin": raw["release_spin_rate"],
    "ivb": raw["pfx_z"] * 12, "hb": raw["pfx_x"] * 12 * arm,
})
counts = df["pitcher"].value_counts()
df = df[df["pitcher"].isin(counts[counts >= MIN_PITCHES].index)]
df = add_relative(df)  # computed on every pitch, before rare types are dropped
df["label"] = df["pitch_type"].map(LABELS)
df = df.dropna(subset=["label"] + FEATURES)
df.to_parquet(ROOT / "data" / "statcast_features.parquet", index=False)
print(f"{len(df):,} pitches, {df['pitcher'].nunique():,} pitchers")
print(df["label"].value_counts().to_string())
