"""Shared feature engineering for Statcast (MLB) and Rapsodo (Clarion) pitches.

Both sources are reduced to the same columns, measured the same way:
velo (mph), spin (rpm), ivb and hb (inches of induced break, with hb
positive toward the pitcher's arm side), plus each pitch's difference from
that pitcher's own fastball. Arm-side terms make righties and lefties
comparable, and working relative to the fastball lets a model trained on
big leaguers read a college arm that throws a few mph slower.
"""
import pandas as pd

BASE = ["velo", "spin", "ivb", "hb"]
REL = ["d_velo", "d_spin", "d_ivb", "d_hb"]
FEATURES = BASE + REL

# Statcast pitch_type codes grouped into the classes the model predicts.
# Rare or ambiguous types (knuckleball, eephus, screwball, pitchouts) are dropped.
LABELS = {
    "FF": "Four-seam", "SI": "Sinker", "FC": "Cutter",
    "SL": "Slider", "ST": "Sweeper",
    "CU": "Curveball", "KC": "Curveball", "SV": "Curveball", "CS": "Curveball",
    "CH": "Changeup", "FS": "Splitter", "FO": "Splitter",
}

FASTBALL_SHARE = 0.15  # a pitcher's hardest 15% of pitches define his fastball


def add_relative(df: pd.DataFrame, by: str = "pitcher") -> pd.DataFrame:
    """Add d_* columns: each pitch minus the median of its pitcher's hardest pitches."""
    cut = df.groupby(by)["velo"].transform(lambda v: v.quantile(1 - FASTBALL_SHARE))
    ref = df[df["velo"] >= cut].groupby(by)[BASE].median()
    ref.columns = ["ref_" + c for c in BASE]
    out = df.join(ref, on=by)
    for c in BASE:
        out["d_" + c] = out[c] - out["ref_" + c]
    return out.drop(columns=ref.columns)
