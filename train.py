"""
train.py — End-to-end training pipeline from raw LiDAR data to WCN v9 model.

Stages
------
1. Feature extraction   src/features/feature_extractor.py
                        → data_processed/features.csv
                        → data_processed/waveform_grids.npy

2. Feature augmentation src/features/add_features.py
                        → data_processed/features_current.csv

3. Auto-labeling        src/labeling/auto_labeler.py
                        → data_processed/labels_current.csv
                        → pointclouds/labeled_pointcloud_waveform_only.csv

4. Water surface model  src/labeling/water_surface_model.py
   Phase 1  tight concave-hull footprint from high-conf riverbed anchors
   Phase 2  local 2m adaptive water-surface grid (Gaussian-smoothed)
   Phase 3  geometry-based classification  → merged_label (0=land 1=water 2=uncertain)
   Phase 3b waterbed reconstruction        → reconstructed_label (adds 3=recon-water)
            Uses confirmed deep-water hits (z < 259.6 m) to fit a riverbed
            elevation grid.  Points geometrically inside the water corridor
            [bed − 0.5 m, surface + 0.3 m] but missed by the waveform
            classifier (tree-over-water n_peaks spike) are recovered as
            label 3.  No-penetration areas produce no false positives.
   Phase 4  retrain XGBoost + V8Net on merged_label
                        → pointclouds/labeled_pointcloud_current.csv
                          columns: merged_label, reconstructed_label, …

5. WCN preprocessing    src/training/preprocess_wcn.py
                        → data_processed/waveform_grids_norm.npy
                        → data_processed/features_v9.csv

6. Label bootstrap      (inline — maps surface model labels → WCN format)
                        → pointclouds/labeled_pointcloud_wcn.csv

7. WCN v9 training      src/training/train_wcn_v9.py
                        → models/wcn_v9/wcn_refined.pt  (deploy this)
                        → models/wcn_v9/wcn_xgb.json
                        → pointclouds/labeled_pointcloud_wcn.csv

Usage
-----
    python train.py                # skip stages whose outputs already exist
    python train.py --from 3       # re-run from stage 3 onward
    python train.py --only 7       # run only stage 7
"""

import argparse
import sys
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

# ── Raw inputs ────────────────────────────────────────────────────────────────
PC_RAW  = ROOT / "data"           / "point_cloud_df.txt"
WF_RAW  = ROOT / "data"           / "waveform_df.txt"

# ── Stage outputs (used as skip sentinels) ────────────────────────────────────
FEAT_RAW   = ROOT / "data_processed" / "features.csv"
GRIDS      = ROOT / "data_processed" / "waveform_grids.npy"
FEAT_V2    = ROOT / "data_processed" / "features_v2.csv"       # add_features output
FEAT_CUR   = ROOT / "data_processed" / "features_current.csv"  # renamed from v2
LABELS     = ROOT / "data_processed" / "labels_current.csv"
PC_WFONLY  = ROOT / "pointclouds"    / "labeled_pointcloud_waveform_only.csv"
PC_CUR     = ROOT / "pointclouds"    / "labeled_pointcloud_current.csv"
GRIDS_NORM = ROOT / "data_processed" / "waveform_grids_norm.npy"
FEAT_V9    = ROOT / "data_processed" / "features_v9.csv"
PC_WCN     = ROOT / "pointclouds"    / "labeled_pointcloud_wcn.csv"
MODEL_DIR  = ROOT / "models"         / "wcn_v9"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _header(stage: str) -> None:
    print(f"\n{'='*60}")
    print(stage)
    print("="*60)


def _run(cmd: list[str], stage: str, cwd: Path | None = None) -> None:
    _header(stage)
    result = subprocess.run(cmd, cwd=cwd or ROOT)
    if result.returncode != 0:
        sys.exit(f"\nStage failed (exit {result.returncode}): {stage}")


def _skip(stage: str) -> None:
    print(f"Skipping {stage} — outputs exist.")


# ── Stage implementations ─────────────────────────────────────────────────────

def stage1_extract_features(run: bool) -> None:
    if not run and FEAT_RAW.exists() and GRIDS.exists():
        _skip("Stage 1: Feature Extraction")
        return
    for p in [PC_RAW, WF_RAW]:
        if not p.exists():
            sys.exit(f"Raw data missing: {p}")
    _run(
        [sys.executable, "src/features/feature_extractor.py",
         "--pc",       str(PC_RAW),
         "--wf",       str(WF_RAW),
         "--out",      str(FEAT_RAW),
         "--grid-out", str(GRIDS)],
        "Stage 1: Feature Extraction",
    )


def stage2_add_features(run: bool) -> None:
    if not run and FEAT_CUR.exists():
        _skip("Stage 2: Feature Augmentation")
        return
    if not (FEAT_RAW.exists() and GRIDS.exists()):
        sys.exit("Stage 2: features.csv or waveform_grids.npy missing — run Stage 1 first.")
    # add_features.py uses relative paths; run from data_processed/
    # it reads  features.csv       (= FEAT_RAW)
    # it reads  waveform_grids.npy (= GRIDS)
    # it writes features_v2.csv    (= FEAT_V2)
    _run(
        [sys.executable, str(ROOT / "src/features/add_features.py")],
        "Stage 2: Feature Augmentation",
        cwd=ROOT / "data_processed",
    )
    FEAT_V2.rename(FEAT_CUR)


def stage3_auto_label(run: bool, no_train: bool = False) -> None:
    if not run and LABELS.exists() and PC_WFONLY.exists():
        _skip("Stage 3: Auto-Labeling")
        return
    cmd = [sys.executable, "src/labeling/auto_labeler.py"]
    if no_train:
        cmd.append("--no-train")
    _run(cmd, "Stage 3: Auto-Labeling")


def stage4_surface_model(run: bool, no_train: bool = False) -> None:
    if not run and PC_CUR.exists():
        _skip("Stage 4: Water Surface Model")
        return
    cmd = [sys.executable, "src/labeling/water_surface_model.py"]
    if no_train:
        cmd.append("--no-train")
    _run(cmd, "Stage 4: Water Surface Model")


def stage5_preprocess_wcn(run: bool) -> None:
    if not run and GRIDS_NORM.exists() and FEAT_V9.exists():
        _skip("Stage 5: WCN Preprocessing")
        return
    _run(
        [sys.executable, "src/training/preprocess_wcn.py"],
        "Stage 5: WCN Preprocessing",
    )


def stage6_bootstrap_labels(run: bool) -> None:
    """
    Map labeled_pointcloud_current.csv → labeled_pointcloud_wcn.csv format.

    train_wcn_v9.py reads columns: ensemble, wcn_proba, xgb_proba
    labeled_pointcloud_current.csv has:  merged_label, deep_proba, xgb_proba
    """
    if not run and PC_WCN.exists():
        _skip("Stage 6: Label Bootstrap")
        return
    _header("Stage 6: Label Bootstrap")
    if not PC_CUR.exists():
        sys.exit("Stage 6: labeled_pointcloud_current.csv missing — run Stage 4 first.")
    df = pd.read_csv(PC_CUR)
    df["ensemble"]  = df["merged_label"]
    df["wcn_proba"] = df["deep_proba"]
    # carry reconstructed_label forward so WCN v9 export includes it
    if "reconstructed_label" not in df.columns and "merged_label" in df.columns:
        df["reconstructed_label"] = df["merged_label"]
    df.to_csv(PC_WCN, index=False)
    n_water = int((df["ensemble"] == 1).sum())
    n_land  = int((df["ensemble"] == 0).sum())
    print(f"Bootstrap labels → {PC_WCN}")
    print(f"  land={n_land:,}  water={n_water:,}  "
          f"uncertain={int((df['ensemble']==2).sum()):,}")


def stage7_train_wcn(run: bool, no_train: bool = False) -> None:
    cmd = [sys.executable, "src/training/train_wcn_v9.py"]
    if no_train:
        cmd.append("--no-train")
    label = "Stage 7: WCN v9 Inference (--no-train)" if no_train else "Stage 7: WCN v9 Training"
    _run(cmd, label)


# ── Entry point ───────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full LiDAR water classifier training pipeline")
    group = p.add_mutually_exclusive_group()
    group.add_argument(
        "--from", dest="from_stage", type=int, metavar="N",
        help="Re-run from stage N onward (1-7), ignoring skip checks",
    )
    group.add_argument(
        "--only", dest="only_stage", type=int, metavar="N",
        help="Run only stage N (1-7)",
    )
    p.add_argument(
        "--no-train", action="store_true",
        help="Skip all model training — apply existing models for inference only "
             "(stages 3, 4 use --no-train; stage 7 runs WCN v9 inference)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    nt = args.no_train
    stages = [
        (1, lambda run: stage1_extract_features(run)),
        (2, lambda run: stage2_add_features(run)),
        (3, lambda run: stage3_auto_label(run, no_train=nt)),
        (4, lambda run: stage4_surface_model(run, no_train=nt)),
        (5, lambda run: stage5_preprocess_wcn(run)),
        (6, lambda run: stage6_bootstrap_labels(run)),
        (7, lambda run: stage7_train_wcn(run, no_train=nt)),
    ]

    for n, fn in stages:
        if args.only_stage is not None:
            if n != args.only_stage:
                continue
            fn(run=True)
        elif args.from_stage is not None:
            fn(run=(n >= args.from_stage))
        else:
            fn(run=False)

    _header("Done")
    print(f"  Deploy:  {MODEL_DIR}/wcn_refined.pt")
    print(f"  XGBoost: {MODEL_DIR}/wcn_xgb.json")
    print(f"  Output:  {PC_WCN}")


if __name__ == "__main__":
    main()
