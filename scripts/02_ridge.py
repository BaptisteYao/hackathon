"""Ridge regression compared to DummyRegressor baseline.

Run with:
    source .venv/bin/activate
    python scripts/02_ridge.py
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

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

# ── models ─────────────────────────────────────────────────────────────────
dummy = DummyRegressor(strategy="mean")

ridge = make_pipeline(
    SimpleImputer(strategy="median"),
    Ridge(alpha=1.0),
)

# ── evaluate individually ──────────────────────────────────────────────────
dummy_report = evaluate(dummy, X, y)
ridge_report = evaluate(ridge, X, y)

# ── compare metrics ────────────────────────────────────────────────────────
print(f"{'':25s} {'dummy':>10s} {'ridge':>10s}")
print(f"{'RMSE':25s} {dummy_report.metrics.rmse():>10.4f} {ridge_report.metrics.rmse():>10.4f}")
print(f"{'MAE':25s} {dummy_report.metrics.mae():>10.4f} {ridge_report.metrics.mae():>10.4f}")
print(f"{'R²':25s} {dummy_report.metrics.r2():>10.4f} {ridge_report.metrics.r2():>10.4f}")

# ── push ridge report to hub ───────────────────────────────────────────────
project = Project(name="ibm-hackathon", mode="hub", workspace=cfg["workspace"])
project.put("02_ridge", ridge_report)
print("Done.")
