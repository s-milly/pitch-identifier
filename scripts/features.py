"""Shared feature engineering for Statcast (MLB) and Rapsodo (Clarion) pitches.

Both sources are reduced to the same columns, measured the same way:
velo (mph), spin (rpm), ivb and hb (inches of induced break, with hb
positive toward the pitcher's arm side), spin axis (degrees, 180 = pure
backspin, mirrored for lefties so 210 means a 1:00 tilt on either side),
plus each pitch's difference from that pitcher's own fastball. Arm-side
terms make righties and lefties comparable, and working relative to the
fastball lets a model trained on big leaguers read a college arm that
throws a few mph slower.
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

BASE = ["velo", "spin", "ivb", "hb"]
FEATURES = BASE + ["d_" + c for c in BASE] + ["axis_s", "axis_c", "d_axis_s", "d_axis_c"]

# Statcast pitch_type codes grouped into the classes the model predicts.
# Rare or ambiguous types (knuckleball, eephus, screwball, pitchouts) are dropped.
LABELS = {
    "FF": "Four-seam", "SI": "Sinker", "FC": "Cutter",
    "SL": "Slider", "ST": "Sweeper",
    "CU": "Curveball", "KC": "Curveball", "SV": "Curveball", "CS": "Curveball",
    "CH": "Changeup", "FS": "Splitter", "FO": "Splitter",
}

FASTBALL_SHARE = 0.15  # a pitcher's hardest 15% of pitches define his fastball


def _circular_mean(deg):
    r = np.radians(deg)
    return np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean())) % 360


def add_relative(df: pd.DataFrame, by: str = "pitcher") -> pd.DataFrame:
    """Add model features: spin axis as sin/cos, and each pitch minus its pitcher's fastball."""
    cut = df.groupby(by)["velo"].transform(lambda v: v.quantile(1 - FASTBALL_SHARE))
    hard = df[df["velo"] >= cut]
    ref = hard.groupby(by)[BASE].median().add_prefix("ref_")
    ref["ref_axis"] = hard.groupby(by)["axis"].agg(_circular_mean)
    out = df.join(ref, on=by)
    for c in BASE:
        out["d_" + c] = out[c] - out["ref_" + c]
    axis, d_axis = np.radians(out["axis"]), np.radians(out["axis"] - out["ref_axis"])
    out["axis_s"], out["axis_c"] = np.sin(axis), np.cos(axis)
    out["d_axis_s"], out["d_axis_c"] = np.sin(d_axis), np.cos(d_axis)
    return out.drop(columns=ref.columns)


# Scales that put mph, rpm, inches and axis on roughly equal footing for neighbors.
_SCALE = np.array([1 / 2, 1 / 300, 1 / 2, 1 / 2, 3, 3])


def smooth(df: pd.DataFrame, prob: np.ndarray, k: int = 10, by: str = "pitcher") -> np.ndarray:
    """Average each pitch's class probabilities with its k most similar pitches
    from the same pitcher. One noisy reading gets outvoted by the pitches it
    looks like, which is how a person would tag a bullpen too."""
    out = prob.copy()
    for idx in df.groupby(by).indices.values():
        g = df.iloc[idx]
        z = g[BASE + ["axis_s", "axis_c"]].to_numpy() * _SCALE
        _, nb = NearestNeighbors(n_neighbors=min(k, len(g))).fit(z).kneighbors(z)
        out[idx] = prob[idx][nb].mean(1)
    return out
