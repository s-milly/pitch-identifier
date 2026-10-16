"""Load the exported model and run it, the same way the web page does."""
import json
from pathlib import Path

import numpy as np

MODEL_PATH = Path(__file__).resolve().parents[1] / "site" / "model.json"


def load_model():
    return json.loads(MODEL_PATH.read_text())


def predict(model, X):
    """Evaluate the exported LightGBM trees and return class probabilities."""
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
