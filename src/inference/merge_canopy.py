"""Final pipeline stage: merge canopy predictions into the v10 water labels.

The canopy model runs after v10 because it needs the local water surface as
ground reference. Its predictions then refine the final classification:
land and uncertain points that the canopy model flags become canopy, which
resolves most of v10's uncertain class (vegetation over the transition zone).
Water labels (1, 3) are kept — the rare water-vs-canopy conflicts (<0.2%)
sit at the water surface and the water model is the authority there.

Final label: 0=land, 1=water, 2=uncertain, 3=water-under-canopy, 4=canopy.

Output: pointclouds/labeled_pointcloud_final.csv (open in CloudCompare)
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
V10_PATH = ROOT / "pointclouds" / "labeled_pointcloud_v10.csv"
CANOPY_PATH = ROOT / "pointclouds" / "labeled_pointcloud_canopy.csv"
OUT_PATH = ROOT / "pointclouds" / "labeled_pointcloud_final.csv"

LABEL_LAND, LABEL_WATER, LABEL_UNCERTAIN, LABEL_RECON_WATER, LABEL_CANOPY = 0, 1, 2, 3, 4


def main() -> None:
    v10 = pd.read_csv(V10_PATH)
    canopy = pd.read_csv(CANOPY_PATH,
                         usecols=["X", "Y", "Z", "canopy_proba", "canopy_pred"])
    if len(v10) != len(canopy) or not np.allclose(
            v10[["X", "Y", "Z"]].to_numpy(), canopy[["X", "Y", "Z"]].to_numpy()):
        raise ValueError("v10 and canopy clouds are not row-aligned")

    final = v10["reconstructed_label"].copy()
    is_canopy = canopy["canopy_pred"] == 1
    overridable = final.isin([LABEL_LAND, LABEL_UNCERTAIN])
    final[overridable & is_canopy] = LABEL_CANOPY

    for src, name in [(LABEL_LAND, "land"), (LABEL_UNCERTAIN, "uncertain")]:
        n = ((v10["reconstructed_label"] == src) & is_canopy).sum()
        print(f"{name} -> canopy: {n}")
    n_conflict = (v10["reconstructed_label"].isin([LABEL_WATER, LABEL_RECON_WATER])
                  & is_canopy).sum()
    print(f"water kept despite canopy flag (conflicts): {n_conflict}")
    print("final:", final.value_counts().sort_index().to_dict())

    out = v10[["X", "Y", "Z", "reflectance_dB", "reconstructed_label",
               "deep_proba", "local_surface_z", "z_above_surface"]].copy()
    out.insert(4, "final_label", final)
    out["canopy_proba"] = canopy["canopy_proba"]
    out["canopy_pred"] = canopy["canopy_pred"]
    out.to_csv(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
