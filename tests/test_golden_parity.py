"""Slow parity tests against the real Pielach dataset and trained
checkpoints already present in this repository's data/ and models/
(both gitignored — skipped automatically if absent). Run explicitly with
``pytest -m golden``; not part of the default test run or CI.

These compare lidarwater's classify() output to the golden CSVs produced by
the original per-script pipeline (pointclouds/labeled_pointcloud_final.csv),
established once by manually diffing the two outputs — see the tolerances'
docstrings for what's actually been verified and what's expected to drift.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
DATA_OK = (ROOT / "data" / "point_cloud_df.txt").exists() and (ROOT / "data" / "waveform_df.txt").exists()
MODELS_OK = (ROOT / "models" / "wcn_v9" / "wcn_refined.pt").exists()
GOLDEN_CSV = ROOT / "pointclouds" / "labeled_pointcloud_final.csv"

pytestmark = pytest.mark.golden

requires_data = pytest.mark.skipif(
    not (DATA_OK and MODELS_OK and GOLDEN_CSV.exists()),
    reason="needs data/, models/, and pointclouds/labeled_pointcloud_final.csv (gitignored)",
)


@pytest.fixture(scope="module")
def real_classify_state():
    from lidarwater import WaterPipeline
    from lidarwater.io import read_pielach_txt

    cloud = read_pielach_txt(ROOT / "data" / "point_cloud_df.txt", ROOT / "data" / "waveform_df.txt")
    pipeline = WaterPipeline.from_local_models(ROOT / "models")
    state = pipeline.classify(cloud)
    golden = pd.read_csv(GOLDEN_CSV)
    return cloud, state, golden


@requires_data
def test_row_alignment(real_classify_state):
    cloud, _state, golden = real_classify_state
    np.testing.assert_allclose(golden["X"].values, cloud.x, atol=1e-3)
    np.testing.assert_allclose(golden["Y"].values, cloud.y, atol=1e-3)
    np.testing.assert_allclose(golden["Z"].values, cloud.z, atol=1e-3)


@requires_data
def test_wcn_transformer_output_matches_deployed_checkpoint(real_classify_state):
    """WCN v9 inference (features -> normalise -> transformer forward) is
    the part of the port with no algorithmic ambiguity — verified to match
    the deployed checkpoint's output to floating-point rounding."""
    _cloud, state, golden = real_classify_state
    diff = np.abs(state.wcn_proba - golden["deep_proba"].values)
    assert diff.max() < 1e-3, f"WCN proba diverged from golden: max diff={diff.max():.5f}"


@requires_data
def test_geometry_water_footprint_matches_golden(real_classify_state):
    """Phase 1 (concave-hull footprint) is deterministic given the same
    WCN probas — footprint area and point count should match exactly."""
    import json

    _cloud, state, _golden = real_classify_state
    golden_metrics = json.loads((ROOT / "models" / "v10" / "metrics.json").read_text())
    golden_fp = golden_metrics["footprint"]

    assert state.footprint_geom.area == pytest.approx(golden_fp["footprint_area_m2"], abs=1.0)
    assert int(state.in_footprint.sum()) == golden_fp["points_inside"]


@requires_data
def test_geometry_labels_match_golden_closely(real_classify_state):
    """Pre-canopy label counts (land/water/uncertain from Phases 1-3)
    should match the golden v10 output almost exactly — small (<0.1%)
    drift is tolerated for RANSAC/float-precision edge cases at the
    water-surface tolerance boundary."""
    import json

    _cloud, state, _golden = real_classify_state
    golden_counts = json.loads((ROOT / "models" / "v10" / "metrics.json").read_text())["merged_label_counts"]
    n = len(state.merged_label)

    for label, name in [(0, "land"), (1, "water"), (2, "uncertain")]:
        ours = int((state.merged_label == label).sum())
        theirs = golden_counts[name]
        assert abs(ours - theirs) / n < 0.001, f"{name}: ours={ours} golden={theirs} (n={n})"


@requires_data
def test_final_label_agreement_with_golden(real_classify_state):
    """End-to-end (through canopy + merge) label agreement. The canopy
    stage's DTM nearest-fill has tie-breaking sensitivity at flat cells, so
    this is a looser bound than the geometry-only checks above — verified
    at 97.0% agreement on the full Pielach dataset when this test was
    written; regressions below that deserve investigation."""
    _cloud, state, golden = real_classify_state
    agreement = (state.final_label == golden["final_label"].values).mean()
    assert agreement > 0.95, f"final_label agreement dropped to {agreement * 100:.2f}%"


@requires_data
def test_final_label_distribution_matches_golden_within_tolerance(real_classify_state):
    _cloud, state, golden = real_classify_state
    n = len(state.final_label)
    golden_counts = golden["final_label"].value_counts().sort_index()
    for label in range(5):
        ours = int((state.final_label == label).sum())
        theirs = int(golden_counts.get(label, 0))
        assert abs(ours - theirs) / n < 0.02, (
            f"label {label}: ours={ours} golden={theirs} (n={n}) — drifted >2%"
        )
