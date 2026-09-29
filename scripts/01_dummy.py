"""Baseline DummyRegressor — always predicts the training-set mean.

Run with:
    source .venv/bin/activate
    python scripts/01_dummy.py
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyRegressor

from skore import Project, evaluate, login

# ── credentials ────────────────────────────────────────────────────────────
cfg = json.loads(Path(".skore").read_text())
login(mode="hub")

# ── data ───────────────────────────────────────────────────────────────────
X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv")

visits = X_train.merge(y_train, on="Index")
y = visits["target"]

feature_cols = [
    "sexM",
    "age_at_diagnosis",
    "age",
    "ledd",
    "time_since_intake_on",
    "time_since_intake_off",
    "on",
    "off",
]
X = visits[feature_cols]

# ── model ──────────────────────────────────────────────────────────────────
dummy = DummyRegressor(strategy="mean")
report = evaluate(dummy, X, y)
print("RMSE:", report.metrics.rmse())

# ── push to hub ────────────────────────────────────────────────────────────
project = Project(name="ibm-hackathon", mode="hub", workspace=cfg["workspace"])
project.put("01_dummy", report)
