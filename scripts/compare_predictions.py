"""Compare y_train vs in-sample predictions for every model.

Fits each model on the full training set, saves per-model prediction CSVs
to predictions/, then writes predictions/comparison_report.html.

Run with:
    source .venv/bin/activate
    python scripts/compare_predictions.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline

# ── data ───────────────────────────────────────────────────────────────────
X_train_raw = pd.read_csv("data/X_train.csv")
y_train_raw = pd.read_csv("data/y_train.csv")

visits = X_train_raw.merge(y_train_raw, on="Index")
y = visits["target"].values

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
MODELS = {
    "01_dummy": DummyRegressor(strategy="mean"),
    "02_ridge": make_pipeline(SimpleImputer(strategy="median"), Ridge(alpha=1.0)),
}

# ── output directory ────────────────────────────────────────────────────────
OUT = Path("predictions")
OUT.mkdir(exist_ok=True)

# ── fit, predict, save CSVs ─────────────────────────────────────────────────
results = {}
for name, model in MODELS.items():
    model.fit(X, y)
    y_pred = model.predict(X)

    df = pd.DataFrame({
        "Index": visits["Index"].values,
        "patient_id": visits["patient_id"].values,
        "y_true": y,
        "y_pred": y_pred,
        "residual": y - y_pred,
    })
    csv_path = OUT / f"{name}_predictions.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    results[name] = {"df": df, "rmse": rmse, "mae": mae, "r2": r2}
    print(f"  RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}")

# ── build HTML report ────────────────────────────────────────────────────────
COLORS = {
    "01_dummy": "#e07b39",
    "02_ridge": "#3b82d4",
    "03_hgbr":  "#7c5cd8",
    "04_skrub": "#27ae60",
}

def clr(name):
    for k, v in COLORS.items():
        if name.startswith(k[:7]):
            return v
    return "#888"


def scatter_svg(results, width=480, height=340):
    """Actual vs predicted scatter — one series per model."""
    all_vals = [v for r in results.values() for v in list(r["df"]["y_true"]) + list(r["df"]["y_pred"])]
    lo, hi = min(all_vals) - 2, max(all_vals) + 2
    pad = 44

    def scale(v, lo, hi, out_lo, out_hi):
        return out_lo + (v - lo) / (hi - lo) * (out_hi - out_lo)

    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" style="font-family:system-ui,sans-serif">']
    # axes
    lines.append(f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#ccc" stroke-width="1"/>')
    lines.append(f'<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#ccc" stroke-width="1"/>')
    # diagonal perfect line
    x0 = scale(lo, lo, hi, pad, width-pad)
    x1 = scale(hi, lo, hi, pad, width-pad)
    y0 = scale(lo, lo, hi, height-pad, pad)
    y1 = scale(hi, lo, hi, height-pad, pad)
    lines.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="#ddd" stroke-width="1.5" stroke-dasharray="4,3"/>')
    # axis labels
    for tick in np.linspace(lo, hi, 5):
        tx = scale(tick, lo, hi, pad, width-pad)
        ty = scale(tick, lo, hi, height-pad, pad)
        lines.append(f'<text x="{tx:.1f}" y="{height-pad+14}" text-anchor="middle" font-size="9" fill="#888">{tick:.0f}</text>')
        lines.append(f'<text x="{pad-6}" y="{ty+3:.1f}" text-anchor="end" font-size="9" fill="#888">{tick:.0f}</text>')
    # axis titles
    lines.append(f'<text x="{(pad+width-pad)/2}" y="{height-2}" text-anchor="middle" font-size="11" fill="#555">Predicted</text>')
    lines.append(f'<text transform="rotate(-90)" x="{-(pad+(height-pad))/2}" y="13" text-anchor="middle" font-size="11" fill="#555">Actual</text>')
    # dots
    for name, r in results.items():
        color = clr(name)
        for yt, yp in zip(r["df"]["y_true"], r["df"]["y_pred"]):
            cx = scale(yp, lo, hi, pad, width-pad)
            cy = scale(yt, lo, hi, height-pad, pad)
            lines.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3" fill="{color}" fill-opacity="0.55"/>')
    lines.append("</svg>")
    return "\n".join(lines)


def residual_svg(results, width=480, height=280):
    """Residual distribution — one bar chart per model side by side."""
    n = len(results)
    pad = 44
    chart_w = (width - pad - 10) // n
    all_res = [v for r in results.values() for v in r["df"]["residual"]]
    lo, hi = min(all_res) - 1, max(all_res) + 1
    bins = np.linspace(lo, hi, 25)

    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" style="font-family:system-ui,sans-serif">']

    for i, (name, r) in enumerate(results.items()):
        color = clr(name)
        x_off = pad + i * chart_w
        counts, edges = np.histogram(r["df"]["residual"], bins=bins)
        max_c = max(counts) if max(counts) > 0 else 1
        bar_w = chart_w / len(counts) - 1

        for j, (cnt, edge) in enumerate(zip(counts, edges[:-1])):
            bx = x_off + j * (bar_w + 1)
            bh = (cnt / max_c) * (height - pad - 20)
            by = height - pad - bh
            lines.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" fill="{color}" fill-opacity="0.7"/>')

        label_x = x_off + chart_w / 2
        lines.append(f'<text x="{label_x:.1f}" y="{height-4}" text-anchor="middle" font-size="10" fill="#444">{name}</text>')

    lines.append(f'<line x1="{pad}" y1="{height-pad}" x2="{width-10}" y2="{height-pad}" stroke="#ccc" stroke-width="1"/>')
    lines.append(f'<text x="{(pad+width)/2}" y="{height+2}" text-anchor="middle" font-size="11" fill="#555">Residuals (y_true − y_pred)</text>')
    lines.append("</svg>")
    return "\n".join(lines)


scatter = scatter_svg(results)
residuals = residual_svg(results)

# metric table rows
metric_rows = ""
for name, r in results.items():
    color = clr(name)
    badge = f'<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:{color};margin-right:6px"></span>'
    metric_rows += f"""
    <tr>
      <td>{badge}{name}</td>
      <td>{r['rmse']:.4f}</td>
      <td>{r['mae']:.4f}</td>
      <td>{r['r2']:.4f}</td>
      <td><a href="{name}_predictions.csv" style="color:#3b82d4;font-size:12px">CSV ↓</a></td>
    </tr>"""

n_models = len(results)
n_rows = len(next(iter(results.values()))["df"])

html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>Comparaison modèles — Parkinson OFF score</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system,"Segoe UI",system-ui,sans-serif; font-size: 14px;
         line-height: 1.6; background: #fff; color: #1f2328; }}
  .wrap {{ max-width: 900px; margin: 0 auto; padding: 32px 20px; }}
  h1 {{ font-size: 20px; font-weight: 600; margin-bottom: 4px; }}
  .subtitle {{ color: #57606a; font-size: 13px; margin-bottom: 28px; }}
  h2 {{ font-size: 15px; font-weight: 600; margin: 28px 0 10px; color: #1f2328; border-bottom: 1px solid #e5e7eb; padding-bottom: 6px; }}
  .stat-row {{ display: flex; gap: 16px; margin-bottom: 28px; flex-wrap: wrap; }}
  .stat {{ background: #f7f8fa; border: 1px solid #e5e7eb; border-radius: 6px;
           padding: 12px 18px; flex: 1; min-width: 120px; }}
  .stat-val {{ font-size: 22px; font-weight: 700; color: #1f2328; }}
  .stat-lbl {{ font-size: 11px; color: #57606a; margin-top: 2px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{ background: #f7f8fa; text-align: left; padding: 8px 10px;
        border-bottom: 1px solid #e5e7eb; font-weight: 600; color: #57606a; font-size: 12px; }}
  td {{ padding: 8px 10px; border-bottom: 1px solid #f0f0f0; }}
  tr:last-child td {{ border-bottom: none; }}
  .charts {{ display: flex; gap: 24px; flex-wrap: wrap; margin-top: 8px; }}
  .chart-box {{ background: #f7f8fa; border: 1px solid #e5e7eb; border-radius: 6px;
                padding: 14px; flex: 1; min-width: 300px; }}
  .chart-title {{ font-size: 12px; color: #57606a; margin-bottom: 8px; font-weight: 500; }}
  .legend {{ display: flex; gap: 16px; flex-wrap: wrap; margin-top: 10px; }}
  .legend-item {{ display: flex; align-items: center; gap: 5px; font-size: 12px; color: #444; }}
  .dot {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }}
  footer {{ margin-top: 40px; padding-top: 14px; border-top: 1px solid #e5e7eb;
            text-align: center; font-size: 12px; color: #57606a; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>Comparaison y_train vs prédictions</h1>
  <p class="subtitle">Score moteur OFF MDS-UPDRS — {n_models} modèles · {n_rows} visites d'entraînement</p>

  <div class="stat-row">
    <div class="stat"><div class="stat-val">{n_rows}</div><div class="stat-lbl">Visites (lignes train)</div></div>
    <div class="stat"><div class="stat-val">{n_models}</div><div class="stat-lbl">Modèles comparés</div></div>
    <div class="stat"><div class="stat-val">{min(r['rmse'] for r in results.values()):.2f}</div><div class="stat-lbl">Meilleur RMSE</div></div>
    <div class="stat"><div class="stat-val">{max(r['r2'] for r in results.values()):.2f}</div><div class="stat-lbl">Meilleur R²</div></div>
  </div>

  <h2>Métriques par modèle</h2>
  <table>
    <thead><tr><th>Modèle</th><th>RMSE ↓</th><th>MAE ↓</th><th>R² ↑</th><th>Prédictions</th></tr></thead>
    <tbody>{metric_rows}</tbody>
  </table>

  <h2>Actual vs Predicted &amp; Résidus</h2>
  <div class="charts">
    <div class="chart-box">
      <div class="chart-title">Actual vs Predicted (toutes visites)</div>
      {scatter}
      <div class="legend">
        {"".join(f'<div class="legend-item"><div class="dot" style="background:{clr(n)}"></div>{n}</div>' for n in results)}
        <div class="legend-item" style="color:#bbb">— ligne parfaite</div>
      </div>
    </div>
    <div class="chart-box">
      <div class="chart-title">Distribution des résidus (y_true − y_pred)</div>
      {residuals}
    </div>
  </div>

  <h2>Fichiers générés</h2>
  <ul style="font-size:13px;padding-left:18px;line-height:2">
    {"".join(f"<li><a href='{n}_predictions.csv' style='color:#3b82d4'>{n}_predictions.csv</a> — Index, patient_id, y_true, y_pred, residual</li>" for n in results)}
  </ul>

  <footer>Made with IBM Bob</footer>
</div>
</body>
</html>"""

html_path = OUT / "comparison_report.html"
html_path.write_text(html, encoding="utf-8")
print(f"\nHTML report → {html_path}")
