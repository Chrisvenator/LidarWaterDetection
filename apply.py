"""
apply.py — Apply trained models to classify river vs. land. No retraining.

Runs the full inference pipeline using existing models:
  v6  waveform-only classifier  (models/labeling/)
  v8  surface geometry model    (models/current/)
  v9  WCN transformer           (models/wcn_v9/)

Skips feature extraction stages only if their outputs already exist.
Always re-runs inference stages (v6 → v8 → v9) for a fresh classification.

Usage
-----
    python apply.py            # classify using all existing models
    python apply.py --force    # also redo feature extraction + WCN preprocessing

Output
------
    pointclouds/labeled_pointcloud_wcn.csv
      ensemble  0=land  1=water  2=uncertain
      wcn_pred / wcn_proba   (WCN v9 transformer)
      xgb_pred / xgb_proba   (XGBoost on 11 scalar features)
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ── Sentinel files ────────────────────────────────────────────────────────────
PC_RAW     = ROOT / "data"           / "point_cloud_df.txt"
WF_RAW     = ROOT / "data"           / "waveform_df.txt"
FEAT_RAW   = ROOT / "data_processed" / "features.csv"
GRIDS      = ROOT / "data_processed" / "waveform_grids.npy"
FEAT_CUR   = ROOT / "data_processed" / "features_current.csv"
GRIDS_NORM = ROOT / "data_processed" / "waveform_grids_norm.npy"
FEAT_V9    = ROOT / "data_processed" / "features_v9.csv"
OUT        = ROOT / "pointclouds"    / "labeled_pointcloud_wcn.csv"

# ── Models that must exist ────────────────────────────────────────────────────
REQUIRED_MODELS = [
    ROOT / "models" / "labeling" / "v6_xgb.json",
    ROOT / "models" / "labeling" / "v6_deep.pt",
    ROOT / "models" / "labeling" / "v6_deep_stats.json",
    ROOT / "models" / "current"  / "v8_xgb.json",
    ROOT / "models" / "current"  / "v8_deep.pt",
    ROOT / "models" / "current"  / "v8_deep_stats.json",
    ROOT / "models" / "wcn_v9"   / "wcn_refined.pt",
    ROOT / "models" / "wcn_v9"   / "wcn_xgb.json",
    ROOT / "models" / "wcn_v9"   / "wcn_stats.json",
]


def _header(msg: str) -> None:
    print(f"\n{'='*60}")
    print(msg)
    print("=" * 60)


def _run(cmd: list[str], label: str, cwd: Path | None = None) -> None:
    _header(label)
    result = subprocess.run(cmd, cwd=cwd or ROOT)
    if result.returncode != 0:
        sys.exit(f"\nFailed (exit {result.returncode}): {label}")


def _skip(label: str) -> None:
    print(f"  Skipping {label} — outputs exist.")


def _check_models() -> None:
    missing = [p for p in REQUIRED_MODELS if not p.exists()]
    if missing:
        print("ERROR: trained models not found. Run training pipeline first:")
        print("  python train.py")
        for p in missing:
            print(f"  missing: {p}")
        sys.exit(1)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Apply trained LiDAR water classifier — no retraining")
    p.add_argument("--force", action="store_true",
                   help="Re-run feature extraction and WCN preprocessing "
                        "even if outputs already exist")
    args = p.parse_args()

    _check_models()

    _header("APPLY — LiDAR water/land classifier (no retraining)")
    print(f"  Output will be: {OUT}")

    # ── Stage 1: Feature extraction ───────────────────────────────────────────
    if not args.force and FEAT_RAW.exists() and GRIDS.exists():
        _skip("Feature extraction")
    else:
        for raw in [PC_RAW, WF_RAW]:
            if not raw.exists():
                sys.exit(f"Raw data missing: {raw}")
        _run(
            [sys.executable, "src/features/feature_extractor.py",
             "--pc", str(PC_RAW), "--wf", str(WF_RAW),
             "--out", str(FEAT_RAW), "--grid-out", str(GRIDS)],
            "Stage 1: Feature Extraction",
        )

    # ── Stage 2: Feature augmentation ────────────────────────────────────────
    if not args.force and FEAT_CUR.exists():
        _skip("Feature augmentation")
    else:
        feat_v2 = ROOT / "data_processed" / "features_v2.csv"
        _run(
            [sys.executable, str(ROOT / "src/features/add_features.py")],
            "Stage 2: Feature Augmentation",
        )
        if feat_v2.exists() and not FEAT_CUR.exists():
            feat_v2.rename(FEAT_CUR)

    # ── Stage 3: v6 waveform inference ───────────────────────────────────────
    _run(
        [sys.executable, "src/labeling/auto_labeler.py", "--no-train"],
        "Stage 3: v6 Waveform Inference",
    )

    # ── Stage 4: v8 geometry + surface model inference ───────────────────────
    _run(
        [sys.executable, "src/labeling/water_surface_model.py", "--no-train"],
        "Stage 4: v8 Surface Model Inference",
    )

    # ── Stage 5: WCN preprocessing ───────────────────────────────────────────
    if not args.force and GRIDS_NORM.exists() and FEAT_V9.exists():
        _skip("WCN preprocessing")
    else:
        _run(
            [sys.executable, "src/training/preprocess_wcn.py"],
            "Stage 5: WCN Preprocessing",
        )

    # ── Stage 6: Bootstrap labels ─────────────────────────────────────────────
    _run_bootstrap()

    # ── Stage 7: WCN v9 inference ────────────────────────────────────────────
    _run(
        [sys.executable, "src/training/train_wcn_v9.py", "--no-train"],
        "Stage 7: WCN v9 Inference",
    )

    # ── Stage 8: v10 — geometry refinement using v9 probabilities ────────────
    # Re-run surface + waterbed reconstruction anchored on v9 probs (better than v8).
    # v9 produces more accurate water/land probabilities → tighter footprint,
    # better surface grid, more accurate waterbed reconstruction.
    v10_out     = ROOT / "pointclouds" / "labeled_pointcloud_v10.csv"
    v10_plotdir = ROOT / "models"      / "v10"
    _run(
        [sys.executable, "src/labeling/water_surface_model.py",
         "--label-src", str(OUT),
         "--geometry-only",
         "--out",      str(v10_out),
         "--plot-dir", str(v10_plotdir)],
        "Stage 8: v10 — v9 Geometry Refinement (surface + waterbed on v9 probs)",
    )

    _header("DONE")
    print(f"\n  Final output  : {v10_out}")
    print(f"  Plots         : {v10_plotdir}/")
    print(f"  (v9 ML output : {OUT})")
    print(f"\n  Open labeled_pointcloud_v10.csv in CloudCompare:")
    print(f"  Colour by 'ensemble'         →  0=land  1=water  2=uncertain")
    print(f"  Colour by 'reconstructed_label'  →  3=waterbed-reconstructed water")


def _run_bootstrap() -> None:
    """Map labeled_pointcloud_current.csv → labeled_pointcloud_wcn.csv format."""
    import pandas as pd
    pc_cur = ROOT / "pointclouds" / "labeled_pointcloud_current.csv"
    pc_wcn = ROOT / "pointclouds" / "labeled_pointcloud_wcn.csv"
    _header("Stage 6: Label Bootstrap")
    if not pc_cur.exists():
        sys.exit(f"Stage 6: {pc_cur.name} missing — Stage 4 must have failed.")
    df = pd.read_csv(pc_cur)
    df["ensemble"]  = df["merged_label"]
    df["wcn_proba"] = df["deep_proba"]
    # reconstructed_label already present in pc_cur; ensure it flows through
    if "reconstructed_label" not in df.columns and "merged_label" in df.columns:
        df["reconstructed_label"] = df["merged_label"]
    df.to_csv(pc_wcn, index=False)
    n_w = int((df["ensemble"] == 1).sum())
    n_l = int((df["ensemble"] == 0).sum())
    print(f"  land={n_l:,}  water={n_w:,}  uncertain={int((df['ensemble']==2).sum()):,}")


if __name__ == "__main__":
    main()
