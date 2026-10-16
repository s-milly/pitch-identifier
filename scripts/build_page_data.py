"""Collect everything the web page shows into site/page_data.json.

Run after train.py and college.py. Every number on the page comes from here.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_parquet(ROOT / "data" / "statcast_features.parquet")
mlb = json.loads((ROOT / "data" / "metrics.json").read_text())
college = json.loads((ROOT / "data" / "college_metrics.json").read_text())
classes = mlb["classes"]
recent = df[df["season"] == df["season"].max()]


def circ_median(deg):
    r = np.radians(deg)
    return float(np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean())) % 360)


# Presets: for each pitch type, the median pitch of that type and the median
# fastball of the pitchers who throw it (a sinkerballer's fastball is a sinker).
def shape(g, axis):
    return {"velo": round(float(g["velo"].median()), 1), "spin": int(round(float(g["spin"].median()) / 10) * 10),
            "ivb": round(float(g["ivb"].median()), 1), "hb": round(float(g["hb"].median()), 1),
            "axis": round(axis)}


presets = {}
for c in classes:
    g = recent[recent["label"] == c]
    ref = pd.DataFrame({k: g[k] - g["d_" + k] for k in ["velo", "spin", "ivb", "hb"]})
    ref_axis = (g["axis"] - np.degrees(np.arctan2(g["d_axis_s"], g["d_axis_c"]))) % 360
    presets[c] = {"pitch": shape(g, circ_median(g["axis"])), "fastball": shape(ref, circ_median(ref_axis))}
fastball = presets["Four-seam"]["fastball"]

# Movement map: a random sample of each type, plus where each type centers.
rng = np.random.default_rng(7)
sample = []
for i, c in enumerate(classes):
    g = recent[recent["label"] == c]
    s = g.iloc[rng.choice(len(g), 700, replace=False)]
    sample += [[round(float(h), 1), round(float(v), 1), i] for h, v in zip(s["hb"], s["ivb"])]
centers = {c: [round(float(recent.loc[recent["label"] == c, "hb"].median()), 1),
               round(float(recent.loc[recent["label"] == c, "ivb"].median()), 1)] for c in classes}

# Most common mix-ups on MLB data (smoothed calls, held-out pitchers).
conf = np.array(mlb["confusion"])
mixups = sorted(((conf[i, j] / conf[i].sum(), classes[i], classes[j])
                 for i in range(len(classes)) for j in range(len(classes)) if i != j), reverse=True)[:5]

out = {
    "classes": classes,
    "fastball": fastball, "presets": presets,
    "sample": sample, "centers": centers, "sample_season": int(recent["season"].iloc[0]),
    "mlb": {
        "pitches": mlb["pitches"], "pitchers": mlb["pitchers"], "seasons": mlb["seasons"],
        "accuracy_single": round(mlb["accuracy_single"], 3), "accuracy": round(mlb["accuracy"], 3),
        "recall": {c: round(mlb["report"][c]["recall"], 3) for c in classes},
        "support": {c: int(mlb["report"][c]["support"]) for c in classes},
        "mixups": [{"actual": a, "called": b, "share": round(float(p), 3)} for p, a, b in mixups],
    },
    "college": {
        "pitches": college["pitches"], "pitchers": college["pitchers"],
        "model": round(college["model_smoothed"]["accuracy"], 3),
        "model_single": round(college["model"]["accuracy"], 3),
        "trackman": round(college["trackman"]["accuracy"], 3),
        "by_type": {k: {"model": college["model_smoothed"]["by_type"][k], "trackman": college["trackman"]["by_type"][k]}
                    for k in college["model_smoothed"]["by_type"]},
        "fastballs_called_changeup": college["fastballs_called_changeup"],
    },
}
(ROOT / "site" / "page_data.json").write_text(json.dumps(out, separators=(",", ":")))
print(json.dumps({k: out[k] for k in ["fastball", "presets", "centers"]}, indent=1))
print(out["mlb"]["mixups"])
