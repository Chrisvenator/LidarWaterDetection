"""Canopy classifier v1: XGBoost on echo/DTM/structure/waveform features.

Labels (z-based, training only — z is never a feature):
  canopy (1):          z > 263.283
  not canopy (0):      z <= 260.1
  pseudo-negative (2): v10 water point sitting at the water surface (trains as 0)
  pseudo-negative (3): open-sky-low point in the transition band (trains as 0) —
                       almost nothing >= 2 m overhead and < 1.2 m above the
                       ground/water reference = ground cover (grass, gravel,
                       water edge), not tree canopy. Dipping branches keep
                       their crown overhead, so they are not caught by this.
                       Canopy-band points matching the same rule are label
                       noise and excluded from training (-1).
  unlabeled (-1):      in between — model predicts

height_above_dtm is excluded from model features: under water DTM = riverbed, so
water depth masquerades as vegetation height. height_above_ref (water-aware) is
used instead; the raw column stays in the output CSV for inspection.

Outputs: models/canopy_v1/ (model, metrics, plots),
         pointclouds/labeled_pointcloud_canopy.csv (CloudCompare).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score, roc_auc_score

from baseline_model import spatial_cv_split

ROOT = Path(__file__).resolve().parent.parent.parent
FEATURES_PATH = ROOT / "data_processed" / "canopy_features.csv"
V10_PATH = ROOT / "pointclouds" / "labeled_pointcloud_v10.csv"
OUT_DIR = ROOT / "models" / "canopy_v1"
CLOUD_OUT = ROOT / "pointclouds" / "labeled_pointcloud_canopy.csv"

Z_CANOPY_MIN = 263.283   # user-verified: above this = 100% canopy
Z_CLEAR_MAX = 260.1      # user-verified: below this = 0% canopy
SURFACE_TOL_M = 0.15     # |z - water surface| below this → pseudo-negative
CANOPY_THRESHOLD = 0.5
N_FOLDS = 5

# open-sky-low rule: no crown overhead + near the ground reference → not foliage
OPEN_SKY_MAX_ABOVE = 1    # max neighbors >= 2 m overhead in the 1 m cylinder
LOW_HEIGHT_MAX_M = 1.2    # max height above the ground/water reference

NON_FEATURES = {"x", "y", "z", "height_above_dtm"}
XGB_PARAMS = dict(
    n_estimators=500, max_depth=7, learning_rate=0.06,
    subsample=0.8, colsample_bytree=0.8, tree_method="hist",
    eval_metric="logloss", n_jobs=-1,
)

# carried into the output cloud for CloudCompare inspection
INSPECT_COLS = ["height_above_dtm", "depth_below_dsm", "echo_ratio", "n_echoes",
                "echo_rank", "n_above_2m", "z_range_cyl", "planarity", "n_peaks"]


def open_sky_low_mask(df: pd.DataFrame) -> np.ndarray:
    """No crown overhead + near the ground/water reference — not tree foliage."""
    return ((df["n_above_2m"] <= OPEN_SKY_MAX_ABOVE)
            & (df["height_above_ref"] < LOW_HEIGHT_MAX_M)).to_numpy()


def make_labels(df: pd.DataFrame, v10: pd.DataFrame) -> np.ndarray:
    """-1 unlabeled, 0 user-negative, 1 user-canopy, 2/3 pseudo-negatives."""
    z = df["z"].to_numpy()
    labels = np.full(len(z), -1, dtype=int)
    labels[z > Z_CANOPY_MIN] = 1
    labels[z <= Z_CLEAR_MAX] = 0
    at_surface = ((v10["merged_label"] == 1) & (v10["in_footprint"] == 1)
                  & (v10["z_above_surface"].abs() < SURFACE_TOL_M)).to_numpy()
    labels[(labels == -1) & at_surface] = 2
    open_sky_low = open_sky_low_mask(df)
    labels[(labels == -1) & open_sky_low] = 3
    labels[(labels == 1) & open_sky_low] = -1  # canopy-band label noise — exclude
    return labels


def cross_validate(X: pd.DataFrame, y: np.ndarray, splits: list) -> dict:
    f1s, aucs = [], []
    pos_w = (y == 0).sum() / (y == 1).sum()
    for k, (tr, va) in enumerate(splits):
        model = xgb.XGBClassifier(**XGB_PARAMS, scale_pos_weight=pos_w)
        model.fit(X.iloc[tr], y[tr])
        proba = model.predict_proba(X.iloc[va])[:, 1]
        f1s.append(f1_score(y[va], proba > CANOPY_THRESHOLD, average="macro"))
        aucs.append(roc_auc_score(y[va], proba))
        print(f"  fold {k}: macro-F1={f1s[-1]:.4f}  AUC={aucs[-1]:.4f}")
    return {"cv_macro_f1": float(np.mean(f1s)), "cv_auc": float(np.mean(aucs)),
            "cv_f1_per_fold": [float(v) for v in f1s]}


def train_final(X: pd.DataFrame, y: np.ndarray) -> xgb.XGBClassifier:
    pos_w = (y == 0).sum() / (y == 1).sum()
    model = xgb.XGBClassifier(**XGB_PARAMS, scale_pos_weight=pos_w)
    model.fit(X, y)
    return model


def _save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT_DIR / name, dpi=120)
    plt.close(fig)


def plot_importance(model: xgb.XGBClassifier, feat_names: list[str]) -> None:
    imp = pd.Series(model.feature_importances_, index=feat_names).sort_values()
    fig, ax = plt.subplots(figsize=(8, 10))
    imp.plot.barh(ax=ax)
    ax.set_title("Canopy v1 — XGBoost feature importance")
    _save(fig, "feature_importance.png")


def plot_proba_hist(z: np.ndarray, proba: np.ndarray, labels: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for val, name in [(1, "canopy band (z>263.28)"), (-1, "transition band"),
                      (0, "water band (z<260.1)"), (2, "water-surface pseudo-neg"),
                      (3, "open-sky-low pseudo-neg")]:
        ax.hist(proba[labels == val], bins=50, alpha=0.55, label=name, density=True)
    ax.axvline(CANOPY_THRESHOLD, color="k", ls="--", lw=1)
    ax.set_xlabel("canopy probability")
    ax.legend()
    _save(fig, "proba_hist.png")


def plot_cross_sections(df: pd.DataFrame, pred: np.ndarray) -> None:
    fig, axes = plt.subplots(5, 1, figsize=(14, 18), sharex=True)
    y_cuts = np.percentile(df["y"], [10, 30, 50, 70, 90])
    colors = np.where(pred == 1, "forestgreen", "royalblue")
    for ax, yc in zip(axes, y_cuts):
        m = (df["y"] > yc - 1) & (df["y"] < yc + 1)
        ax.scatter(df.loc[m, "x"], df.loc[m, "z"], c=colors[m.to_numpy()], s=1.5)
        for zline, style in [(Z_CANOPY_MIN, "g--"), (Z_CLEAR_MAX, "b--")]:
            ax.axhline(zline, color=style[0], ls=style[1:], lw=0.8)
        ax.set_title(f"y = {yc:.1f} ± 1 m  (green=canopy, blue=not)")
        ax.set_ylabel("z [m]")
    axes[-1].set_xlabel("x [m]")
    _save(fig, "cross_sections.png")


def plot_topdown(df: pd.DataFrame, pred: np.ndarray) -> None:
    rng = np.random.default_rng(0)
    idx = rng.choice(len(df), min(120_000, len(df)), replace=False)
    fig, ax = plt.subplots(figsize=(12, 10))
    colors = np.where(pred[idx] == 1, "forestgreen", "royalblue")
    xy = df[["x", "y"]].to_numpy()[idx]
    ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=1)
    ax.set_aspect("equal")
    ax.set_title("Canopy v1 top-down (green=canopy, blue=not)")
    _save(fig, "topdown.png")


def v10_water_sanity(v10: pd.DataFrame, pred: np.ndarray, labels: np.ndarray) -> dict:
    """Mid-band points that v10 calls water should rarely be predicted canopy."""
    mid_water = (labels == -1) & (v10["merged_label"].to_numpy() == 1)
    frac = float(pred[mid_water].mean()) if mid_water.any() else float("nan")
    return {"mid_band_v10_water_points": int(mid_water.sum()),
            "predicted_canopy_fraction": frac}


def export_cloud(df: pd.DataFrame, proba: np.ndarray, pred: np.ndarray,
                 labels: np.ndarray) -> None:
    out = pd.DataFrame({"X": df["x"], "Y": df["y"], "Z": df["z"],
                        "reflectance_dB": df["reflectance_dB"],
                        "label_src": labels, "canopy_proba": np.round(proba, 4),
                        "canopy_pred": pred})
    for col in INSPECT_COLS:
        out[col] = df[col]
    out.to_csv(CLOUD_OUT, index=False)
    print(f"wrote {CLOUD_OUT}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(FEATURES_PATH)
    v10 = pd.read_csv(V10_PATH, usecols=["X", "Y", "Z", "merged_label",
                                         "in_footprint", "z_above_surface"])
    if not np.allclose(v10[["X", "Y", "Z"]].to_numpy(),
                       df[["x", "y", "z"]].to_numpy()):
        raise ValueError("labeled_pointcloud_v10.csv row order mismatch with features")
    labels = make_labels(df, v10)
    feat_names = [c for c in df.columns if c not in NON_FEATURES]
    X = df[feat_names]
    labeled = labels >= 0
    y = (labels == 1).astype(int)
    print(f"labeled: {labeled.sum()} (canopy={np.sum(labels == 1)}, "
          f"clear={np.sum(labels == 0)}, surface-pseudo-neg={np.sum(labels == 2)}, "
          f"open-sky-low-pseudo-neg={np.sum(labels == 3)}), "
          f"unlabeled: {(~labeled).sum()}")

    splits = spatial_cv_split(df.loc[labeled], N_FOLDS)
    print("spatial CV:")
    metrics = cross_validate(X[labeled], y[labeled], splits)

    model = train_final(X[labeled], y[labeled])
    model.save_model(OUT_DIR / "canopy_xgb.json")

    proba = model.predict_proba(X)[:, 1]
    pred = (proba > CANOPY_THRESHOLD).astype(int)

    mid = labels == -1
    metrics["mid_band_canopy_fraction"] = float(pred[mid].mean())
    metrics["v10_water_sanity"] = v10_water_sanity(v10, pred, labels)
    metrics["threshold"] = CANOPY_THRESHOLD
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

    plot_importance(model, feat_names)
    plot_proba_hist(df["z"].to_numpy(), proba, labels)
    plot_cross_sections(df, pred)
    plot_topdown(df, pred)
    export_cloud(df, proba, pred, labels)


if __name__ == "__main__":
    main()
