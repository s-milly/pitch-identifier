"""Test the MLB-trained model on college pitchers it has never seen.

Data: "Optical Tracking Data from College Baseball Scrimmages" by Nathan
David Pifer, Mendeley Data, V3, doi:10.17632/xfnz6mkdzm.3, licensed CC BY 4.0.
3,344 TrackMan pitches from 21 exhibition games between US college teams,
each pitch type tagged by hand by a trained operator.

The hand tags use five broader groups (a sinker is just "Fastball", a
sweeper is a "Slider"), so the model's calls are folded into the same
groups before scoring. TrackMan's own automatic tags are scored the same
way as a benchmark.

Usage: python scripts/college.py   (downloads the CSV on first run)
Writes data/college_metrics.json.
"""
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from classify import load_model, predict
from features import BASE, add_relative, smooth

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "college" / "Track_Combo.csv"
URL = ("https://data.mendeley.com/public-files/datasets/xfnz6mkdzm/files/"
       "0d5fcc5a-1429-4a46-ad9c-33160faca6eb/file_downloaded")

GROUP = {  # every label set folded into the hand-tag groups
    "Four-seam": "Fastball", "Four-Seam": "Fastball", "Sinker": "Fastball", "Fastball": "Fastball",
    "Slider": "Slider", "Sweeper": "Slider",
    "Changeup": "ChangeUp", "ChangeUp": "ChangeUp", "Splitter": "ChangeUp",
    "Curveball": "Curveball", "Cutter": "Cutter",
}

if not CSV.exists():
    CSV.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(URL, CSV)
raw = pd.read_csv(CSV)
raw = raw.dropna(subset=["RelSpeed", "SpinRate", "SpinAxis", "InducedVertBreak", "HorzBreak"])

# TrackMan reports horizontal break from the pitcher's view (a righty's arm side
# is positive) and spin axis on the same scale as Statcast. One pitcher's hand is
# "Undefined", so fill it from his fastball's break.
throws = raw["PitcherThrows"].where(raw["PitcherThrows"] != "Undefined")
fb_side = raw[raw["TaggedPitchType"] == "Fastball"].groupby("PitcherId")["HorzBreak"].median()
throws = throws.fillna(raw["PitcherId"].map(np.sign(fb_side)).map({1: "Right", -1: "Left"}))
righty = throws == "Right"
df = pd.DataFrame({
    "pitcher": raw["PitcherId"],
    "velo": raw["RelSpeed"], "spin": raw["SpinRate"], "ivb": raw["InducedVertBreak"],
    "hb": raw["HorzBreak"] * np.where(righty, 1, -1),
    "axis": np.where(righty, raw["SpinAxis"], 360 - raw["SpinAxis"]),
    "tagged": raw["TaggedPitchType"].map(GROUP),
    "trackman": raw["AutoPitchType"].map(GROUP),
})
df = add_relative(df.reset_index(drop=True))
keep = df["tagged"].notna().to_numpy()  # drops the one knuckleball

model = load_model()
classes = model["classes"]
groups = sorted(set(GROUP[c] for c in classes))
fold = np.zeros((len(classes), len(groups)))
for i, c in enumerate(classes):
    fold[i, groups.index(GROUP[c])] = 1
prob = predict(model, df[model["features"]].to_numpy())
df["model"] = [groups[i] for i in (prob @ fold).argmax(1)]
df["model_smoothed"] = [groups[i] for i in (smooth(df, prob) @ fold).argmax(1)]

t = df[keep]
results = {"pitches": int(len(t)), "pitchers": int(t["pitcher"].nunique())}
for col in ["model", "model_smoothed", "trackman"]:
    ok = t[col] == t["tagged"]
    results[col] = {"accuracy": float(ok.mean()),
                    "by_type": t.assign(ok=ok).groupby("tagged")["ok"].mean().round(3).to_dict()}
    print(f"{col:15s} {ok.mean():.3f}  " + "  ".join(f"{k} {v:.2f}" for k, v in results[col]["by_type"].items()))
print(f"\n{results['pitches']:,} pitches, {results['pitchers']} pitchers")
print("\nhand tag (rows) vs smoothed model call (columns):")
print(pd.crosstab(t["tagged"], t["model_smoothed"]))
results["confusion"] = pd.crosstab(t["tagged"], t["model_smoothed"]).to_dict()
(ROOT / "data" / "college_metrics.json").write_text(json.dumps(results, indent=1))
