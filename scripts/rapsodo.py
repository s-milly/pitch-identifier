"""Run a Rapsodo pitching export through the trained classifier.

Usage: python scripts/rapsodo.py data/rapsodo/session.csv --throws R
Prints each pitch with the model's call, and, when the export has a
Pitch Type column, how often the model agrees with the tagged type.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from features import BASE, add_relative, smooth

ROOT = Path(__file__).resolve().parents[1]

# Rapsodo column names differ a little between app versions; first match wins.
COLUMNS = {
    "velo": ["Velocity"],
    "spin": ["Total Spin", "Spin Rate"],
    "ivb": ["VB (trajectory)", "Induced Vertical Break", "Vertical Break"],
    "hb": ["HB (trajectory)", "Horizontal Break"],
    "axis": ["Spin Direction", "Spin Axis"],
    "tagged": ["Pitch Type"],
}


def read_export(path):
    """Rapsodo puts a few lines of player info above the real header row."""
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    header = next(i for i, l in enumerate(lines) if "Velocity" in l)
    df = pd.read_csv(path, skiprows=header, encoding="utf-8-sig", na_values=["-", "--", ""])
    df.columns = [c.strip().strip('"') for c in df.columns]
    picked = {}
    for key, options in COLUMNS.items():
        hit = next((c for c in options if c in df.columns), None)
        if hit is None and key != "tagged":
            raise SystemExit(f"no column for {key}; columns are: {list(df.columns)}")
        picked[key] = hit
    print("using columns:", {k: v for k, v in picked.items() if v})
    out = pd.DataFrame({k: pd.to_numeric(df[v], errors="coerce")
                        for k, v in picked.items() if k not in ("tagged", "axis")})
    out["axis"] = df[picked["axis"]].map(clock_to_degrees)
    if picked["tagged"]:
        out["tagged"] = df[picked["tagged"]]
    return out.dropna(subset=BASE + ["axis"]).reset_index(drop=True)


def clock_to_degrees(value):
    """Rapsodo reports spin direction as a clock tilt like "1:15"; Statcast uses
    degrees with 12:00 (pure backspin) at 180 and each hour worth 30."""
    try:
        if ":" not in str(value):
            return float(value)
        h, m = str(value).split(":")[:2]
        return (180 + 30 * (int(h) % 12 + int(m) / 60)) % 360
    except ValueError:
        return np.nan


def predict(model, X):
    """Evaluate the exported LightGBM trees, the same way the web page does."""
    def walk(node, row):
        while "leaf_value" not in node:
            go_left = row[node["split_feature"]] <= node["threshold"]
            node = node["left_child"] if go_left else node["right_child"]
        return node["leaf_value"]
    k = len(model["classes"])
    scores = np.zeros((len(X), k))
    for i, tree in enumerate(model["trees"]):
        scores[:, i % k] += [walk(tree, row) for row in X]
    p = np.exp(scores - scores.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


ap = argparse.ArgumentParser()
ap.add_argument("csv")
ap.add_argument("--throws", choices=["R", "L"], required=True)
args = ap.parse_args()

df = read_export(args.csv)
# Rapsodo's horizontal break sign depends on the view it reports from. Every
# fastball breaks to the arm side, so the hardest pitches settle it.
hard = df["velo"] >= df["velo"].quantile(0.85)
if df.loc[hard, "hb"].median() < 0:
    df["hb"] = -df["hb"]
    print("flipped horizontal break so the fastball runs arm side")
# Same check for spin direction: a fastball's tilt leans to the arm side (past 180).
if ((df.loc[hard, "axis"] - 180) % 360).median() > 180:
    df["axis"] = (360 - df["axis"]) % 360
    print("mirrored spin direction so the fastball tilts arm side")
df["pitcher"] = "me"
df = add_relative(df)

model = json.loads((ROOT / "site" / "model.json").read_text())
probs = predict(model, df[model["features"]].to_numpy())
# A bullpen is short, so vote over fewer neighbors than the 10 used on MLB seasons,
# enough to steady a call without pulling in a different pitch type.
probs = smooth(df, probs, k=max(1, min(10, len(df) // 8)))
df["call"] = [model["classes"][i] for i in probs.argmax(1)]
df["confidence"] = probs.max(1).round(2)
cols = ["velo", "spin", "ivb", "hb"] + (["tagged"] if "tagged" in df else []) + ["call", "confidence"]
print(df[cols].round(1).to_string())
if "tagged" in df:
    print(pd.crosstab(df["tagged"], df["call"]))
