"""Train the pitch classifier and export it for the browser.

Evaluation splits by pitcher (GroupKFold), so every score comes from
pitchers the model never saw in training, in either season. Splitting by
pitch would leak: a pitcher's sliders look like his other sliders.

Two scores are reported: one pitch on its own (what the web form does),
and with each pitch's call smoothed over the pitcher's similar pitches
(what the Rapsodo script does with a whole bullpen).

Writes data/metrics.json and site/model.json (LightGBM trees as JSON).
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold

from classify import compact
from features import FEATURES, smooth

ROOT = Path(__file__).resolve().parents[1]
PARAMS = dict(objective="multiclass", n_estimators=150, num_leaves=15,
              learning_rate=0.1, min_child_samples=200, verbose=-1)

df = pd.read_parquet(ROOT / "data" / "statcast_features.parquet")
classes = sorted(df["label"].unique())
X, y, groups = df[FEATURES], df["label"].map(classes.index), df["pid"]

prob = np.zeros((len(df), len(classes)))
for fold, (tr, te) in enumerate(GroupKFold(n_splits=5).split(X, y, groups)):
    m = lgb.LGBMClassifier(**PARAMS).fit(X.iloc[tr], y.iloc[tr])
    prob[te] = m.predict_proba(X.iloc[te])
    print(f"fold {fold + 1}: {accuracy_score(y.iloc[te], prob[te].argmax(1)):.3f}")

single = accuracy_score(y, prob.argmax(1))
pred = smooth(df, prob).argmax(1)
acc = accuracy_score(y, pred)
print(f"\nheld-out-pitcher accuracy: {single:.3f} one pitch at a time, {acc:.3f} smoothed\n")
print(classification_report(y, pred, target_names=classes, digits=3))
metrics = {
    "accuracy_single": single, "accuracy": acc, "pitches": len(df), "pitchers": int(groups.nunique()),
    "seasons": sorted(int(s) for s in df["season"].unique()),
    "classes": classes,
    "report": classification_report(y, pred, target_names=classes, output_dict=True),
    "confusion": confusion_matrix(y, pred).tolist(),
}
(ROOT / "data" / "metrics.json").write_text(json.dumps(metrics, indent=1))

final = lgb.LGBMClassifier(**PARAMS).fit(X, y)
dump = final.booster_.dump_model()
model = {"features": FEATURES, "classes": classes,
         "trees": [compact(t["tree_structure"]) for t in dump["tree_info"]]}
(ROOT / "site" / "model.json").write_text(json.dumps(model, separators=(",", ":")))
print(f"model: {len(model['trees'])} trees -> site/model.json")
