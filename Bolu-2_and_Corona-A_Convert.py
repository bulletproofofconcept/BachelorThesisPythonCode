"""
Author: Maxim Glyzhev
e-Mail: fn6237@kit.edu
Affiliation: KIT Institut der Stochastik.

Generate gridded height maps (raw + detrended) for:
  - Bolu_2_XYZ.mat
  - Corona_A_XYZ.mat

Prerequisite:
  You already saved the converter in: Point_Cloud_to_Grid.py
  and it defines a function: pointcloud_to_grid(...)

Outputs:
  - Prints basic grid/extent/range stats
  - Saves results to:
      bolu2_grids.npz, coronaA_grids.npz
    (and optionally .mat files if you keep SAVE_MAT=True)
"""

from __future__ import annotations

import os
import numpy as np
import scipy.io as sio

from Point_Cloud_to_Grid import pointcloud_to_grid

"""
This code was used for the study to convert the Bolu-2 and Corona-A point cloud .mat files
to regular grid .npz files

"""


# ----------------------------
# User paths / settings
# ----------------------------
DATA_DIR = "C:\\Users\\Flo\\Desktop\\BA\\"        # path to the .mat files 
OUT_DIR = "./"               # output directory for .npz/.mat
SAVE_MAT = True              # also write Matlab .mat files
FILL_NEAREST = False         # keep False unless you explicitly want gap-filling

# If you want to force Corona-A to a known step (e.g., 5 mm):
FORCE_CORONA_DX = None       # e.g., 0.005
FORCE_CORONA_DY = None       # e.g., 0.005


def _pack_for_save(result: dict) -> dict:
    """
    Convert result dict into a flat dict suitable for np.savez / savemat.
    (Matlab will store tuples as arrays; that's fine.)
    """
    return {
        "xg": result["xg"],
        "yg": result["yg"],
        "Z_grid": result["Z_grid"],
        "Z_detrended": result["Z_detrended"],
        "dx": np.array(result["dx"], dtype=float),
        "dy": np.array(result["dy"], dtype=float),
        "Lx": np.array(result["Lx"], dtype=float),
        "Ly": np.array(result["Ly"], dtype=float),
        "z_range": np.array(result["z_range"], dtype=float),
        "plane_abc": np.array(result["plane_abc"], dtype=float),
        # meta can be saved too, but .npz/.mat handle dicts differently.
        # Keep it simple: store meta as a string representation.
        "meta_str": np.array(str(result.get("meta", {})), dtype=object),
    }


def _print_summary(name: str, result: dict) -> None:
    ny, nx = result["Z_grid"].shape
    nan_frac = np.mean(~np.isfinite(result["Z_grid"]))
    print(f"\n{name}")
    print("-" * len(name))
    print(f"Grid shape (ny, nx): ({ny}, {nx})")
    print(f"dx, dy: {result['dx']:.6g}, {result['dy']:.6g}")
    print(f"Lx, Ly: {result['Lx']:.6g}, {result['Ly']:.6g}")
    print(f"Raw z-range (max-min): {result['z_range']:.6g}")
    print(f"NaN fraction in Z_grid: {nan_frac:.3%}")
    a, b, c = result["plane_abc"]
    print(f"Detrend plane (a, b, c): ({a:.6g}, {b:.6g}, {c:.6g})")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    # ----------------------------
    # Bolu-2
    # ----------------------------
    bolu_path = os.path.join(DATA_DIR, "Bolu_2_XYZ.mat")
    bolu = pointcloud_to_grid(
        bolu_path,
        fill_nearest=FILL_NEAREST,
        unit_scale=1.0,          # keep units unchanged (expected meters)
    )
    _print_summary("Bolu-2", bolu)

    bolu_out = _pack_for_save(bolu)
    np.savez(os.path.join(OUT_DIR, "bolu2_grids.npz"), **bolu_out)
    if SAVE_MAT:
        sio.savemat(os.path.join(OUT_DIR, "bolu2_grids.mat"), bolu_out)

    # ----------------------------
    # Corona-A
    # ----------------------------
    corona_path = os.path.join(DATA_DIR, "Corona_A_XYZ.mat")

    corona_kwargs = dict(
        fill_nearest=FILL_NEAREST,
        unit_scale=1.0,          # keep units unchanged (expected meters)
    )
    if FORCE_CORONA_DX is not None:
        corona_kwargs["grid_dx"] = float(FORCE_CORONA_DX)
    if FORCE_CORONA_DY is not None:
        corona_kwargs["grid_dy"] = float(FORCE_CORONA_DY)

    corona = pointcloud_to_grid(corona_path, **corona_kwargs)
    _print_summary("Corona-A", corona)

    corona_out = _pack_for_save(corona)
    np.savez(os.path.join(OUT_DIR, "coronaA_grids.npz"), **corona_out)
    if SAVE_MAT:
        sio.savemat(os.path.join(OUT_DIR, "coronaA_grids.mat"), corona_out)

    # ----------------------------
    # Convenience: expose arrays if running interactively
    # ----------------------------
    # Raw (non-detrended):
    #   bolu["Z_grid"], corona["Z_grid"]
    # Detrended:
    #   bolu["Z_detrended"], corona["Z_detrended"]

    print("\nDone.")
    print(f"Saved: {os.path.join(OUT_DIR, 'bolu2_grids.npz')}")
    print(f"Saved: {os.path.join(OUT_DIR, 'coronaA_grids.npz')}")
    if SAVE_MAT:
        print(f"Saved: {os.path.join(OUT_DIR, 'bolu2_grids.mat')}")
        print(f"Saved: {os.path.join(OUT_DIR, 'coronaA_grids.mat')}")


if __name__ == "__main__":
    main()
