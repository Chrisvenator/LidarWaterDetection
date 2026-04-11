"""
train.py — Full training pipeline, start to finish.

Stages:
  1. auto_labeler        — waveform-only zone labeler (v6)
                           → data_processed/labels_current.csv
                           → pointclouds/labeled_pointcloud_waveform_only.csv
                           → models/labeling/

  2. water_surface_model — tight concave-hull footprint + adaptive surface grid (v8)
                           → pointclouds/labeled_pointcloud_current.csv
                           → models/current/

  3. preprocess_wcn      — per-sample max-normalise waveforms; build features_v9.csv
                           → data_processed/waveform_grids_norm.npy
                           → data_processed/features_v9.csv

  4. train_wcn_v9        — WaveformContextNet v9 (3-phase: MAE pre-train → fine-tune
                           → pseudo-label refinement) + XGBoost on 11 features
                           → models/wcn_v9/
                           → pointclouds/labeled_pointcloud_wcn.csv

Usage:
  source .venv/bin/activate
  python src/train.py
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

V8_CSV  = ROOT / "pointclouds" / "labeled_pointcloud_current.csv"
WCN_CSV = ROOT / "pointclouds" / "labeled_pointcloud_wcn.csv"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _banner(title: str, stage: str = "") -> None:
    line = "═" * 70
    prefix = f"[{stage}] " if stage else ""
    print(f"\n{line}")
    print(f"  {prefix}{title}")
    print(f"{line}\n")


def _elapsed(t0: float) -> str:
    s = time.time() - t0
    return f"{s/60:.1f} min" if s >= 60 else f"{s:.0f}s"


def _bootstrap_wcn_labels() -> None:
    """
    labeled_pointcloud_wcn.csv must exist before train_wcn_v9 loads it.

    train_wcn_v9 reads: ensemble, wcn_proba, xgb_proba
    v8 output provides: ensemble, deep_proba, xgb_proba

    Map deep_proba → wcn_proba as bootstrap.  Always regenerated here so
    v9 trains on the freshly-updated v8 labels, not stale priors.
    """
    import pandas as pd
    print(f"[bootstrap] Creating {WCN_CSV.name} from v8 output …")
    v8 = pd.read_csv(
        V8_CSV,
        usecols=["x", "y", "z", "ensemble", "xgb_proba", "deep_proba"],
    )
    v8 = v8.rename(columns={"deep_proba": "wcn_proba"})
    v8.to_csv(WCN_CSV, index=False)
    print(f"[bootstrap] Saved {len(v8):,} rows → {WCN_CSV}")


# ── Pipeline ───────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()
    stage_times: list[tuple[str, float]] = []

    # ── Stage 1: Waveform-only auto-labeler (v6) ──────────────────────────────
    _banner("Auto-labeler  (v6 waveform-only)", stage="1/4")
    t1 = time.time()
    from labeling.auto_labeler import main as _auto_labeler
    _auto_labeler()
    stage_times.append(("1/4  Auto-labeler", time.time() - t1))
    print(f"\n[train.py] Stage 1 complete  (step {_elapsed(t1)}  |  total {_elapsed(t0)})")

    # ── Stage 2: Water surface model (v8) ────────────────────────────────────
    _banner("Water surface model  (v8 tight footprint + surface grid)", stage="2/4")
    t1 = time.time()
    from labeling.water_surface_model import main as _surface_model
    _surface_model()
    stage_times.append(("2/4  Water surface model", time.time() - t1))
    print(f"\n[train.py] Stage 2 complete  (step {_elapsed(t1)}  |  total {_elapsed(t0)})")

    # ── Bootstrap: v8 labels → wcn label file ────────────────────────────────
    _bootstrap_wcn_labels()

    # ── Stage 3: WCN v9 preprocessing ────────────────────────────────────────
    _banner("WCN v9 preprocessing  (normalise waveforms + build features_v9)", stage="3/4")
    t1 = time.time()
    from training.preprocess_wcn import main as _preprocess
    _preprocess()
    stage_times.append(("3/4  WCN preprocessing", time.time() - t1))
    print(f"\n[train.py] Stage 3 complete  (step {_elapsed(t1)}  |  total {_elapsed(t0)})")

    # ── Stage 4: WCN v9 training ──────────────────────────────────────────────
    _banner("WCN v9 training  (MAE pre-train → fine-tune → pseudo-label refine)", stage="4/4")
    t1 = time.time()
    from training.train_wcn_v9 import main as _train_wcn
    _train_wcn()
    stage_times.append(("4/4  WCN training", time.time() - t1))
    print(f"\n[train.py] Stage 4 complete  (step {_elapsed(t1)}  |  total {_elapsed(t0)})")

    # ── Done ──────────────────────────────────────────────────────────────────
    total = time.time() - t0
    _banner(f"Pipeline complete — {_elapsed(t0)}")
    print("  Final output : pointclouds/labeled_pointcloud_wcn.csv")
    print("  Models       : models/wcn_v9/")
    print("  CloudCompare : colour by 'ensemble'  0=land  1=water  2=uncertain")
    print()
    print("  Stage breakdown:")
    for name, secs in stage_times:
        bar = "█" * int(secs / total * 30)
        dur = f"{secs/60:.1f} min" if secs >= 60 else f"{secs:.0f}s"
        print(f"    {name:<28}  {dur:>7}  {bar}")
    print(f"    {'TOTAL':<28}  {_elapsed(t0):>7}")


if __name__ == "__main__":
    main()
