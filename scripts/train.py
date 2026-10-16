"""Train the pitch classifier and export it for the browser.

Evaluation splits by pitcher (GroupKFold), so every score comes from
pitchers the model never saw in training. Splitting by pitch would leak:
a pitcher's sliders look like his other sliders.

Writes data/metrics.json and site/model.json (LightGBM trees as JSON).
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold

from features import FEATURES

ROOT = Path(__file__).resolve().parents[1]
PARAMS = dict(objective="multiclass", n_estimators=150, num_leaves=15,
              learning_rate=0.1, min_child_samples=200, verbose=-1)

df = pd.read_parquet(ROOT / "data" / "statcast_features.parquet")
classes = sorted(df["label"].unique())
X, y, groups = df[FEATURES], df["label"].map(classes.index), df["pitcher"]

pred = np.empty(len(df), dtype=int)
for fold, (tr, te) in enumerate(GroupKFold(n_splits=5).split(X, y, groups)):
    m = lgb.LGBMClassifier(**PARAMS).fit(X.iloc[tr], y.iloc[tr])
    pred[te] = m.predict(X.iloc[te])
    print(f"fold {fold + 1}: {accuracy_score(y.iloc[te], pred[te]):.3f}")

acc = accuracy_score(y, pred)
print(f"\nheld-out-pitcher accuracy: {acc:.3f}\n")
print(classification_report(y, pred, target_names=classes, digits=3))
metrics = {
    "accuracy": acc, "pitches": len(df), "pitchers": int(groups.nunique()),
    "classes": classes,
    "report": classification_report(y, pred, target_names=classes, output_dict=True),
    "confusion": confusion_matrix(y, pred).tolist(),
}
(ROOT / "data" / "metrics.json").write_text(json.dumps(metrics, indent=1))

final = lgb.LGBMClassifier(**PARAMS).fit(X, y)
dump = final.booster_.dump_model()
model = {"features": FEATURES, "classes": classes,
         "trees": [t["tree_structure"] for t in dump["tree_info"]]}
(ROOT / "site" / "model.json").write_text(json.dumps(model, separators=(",", ":")))
print(f"model: {len(model['trees'])} trees -> site/model.json")
