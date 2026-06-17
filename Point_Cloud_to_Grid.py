from __future__ import annotations
from typing import Optional, Dict, Any, Tuple
import numpy as np
import scipy.io
from scipy.spatial import cKDTree
from scipy.interpolate import LinearNDInterpolator



"""
This files contains function which convert a point cloud to a regular grid.
However, this files does not the cotain the convesrion code for Bolu-2 and Corona-A

"""


def _is_almost_regular_lattice(
    x: np.ndarray,
    y: np.ndarray,
    n: int,
    tol_frac: float = 0.02
) -> Tuple[bool, int, int]:
    """
    Heuristic test for “rectilinear lattice-like” sampling.

    Many lab profilometer datasets are delivered as a (possibly gappy) Cartesian grid
    that has been flattened into 1D vectors X, Y, Z. In that case the number of points
    N is close to N_unique_x * N_unique_y. This helper checks that relationship.

    Parameters
    ----------
    x, y
        1D arrays of point coordinates.
    n
        Total number of points (after filtering finite values).
    tol_frac
        Fractional tolerance for classifying the dataset as lattice-like:
        abs(N_unique_x * N_unique_y - N) / (N_unique_x * N_unique_y) <= tol_frac.

    Returns
    -------
    is_lattice : bool
        True if the dataset is classified as lattice-like.
    n_unique_x : int
        Number of unique x coordinates.
    n_unique_y : int
        Number of unique y coordinates.
    """
    ux = np.unique(x)
    uy = np.unique(y)
    prod = ux.size * uy.size
    if prod == 0:
        return False, ux.size, uy.size
    frac_err = abs(prod - n) / prod
    return frac_err <= tol_frac, ux.size, uy.size


def _estimate_spacing_irregular(
    x: np.ndarray,
    y: np.ndarray,
    sample: int = 50000,
    q: float = 0.10,
    seed: int = 0
) -> float:
    """
    Estimate a characteristic sampling spacing for irregular point clouds.

    For irregular clouds (e.g., field LiDAR), the point density varies spatially and
    there is no intrinsic dx/dy. We estimate an “effective local sampling resolution”
    from nearest-neighbor (NN) distances in the horizontal (x,y) plane.

    The function:
      1) randomly subsamples up to `sample` points,
      2) builds a KD-tree in (x,y),
      3) computes the distance to each point's nearest neighbor,
      4) returns the q-quantile of the NN distance distribution.

    Using a lower quantile (default q=0.10) is robust to holes/occlusions and
    better reflects the densest resolvable sampling.

    Parameters
    ----------
    x, y
        1D arrays of point coordinates.
    sample
        Maximum number of points used to estimate NN distances (for speed).
    q
        Quantile of the NN distance distribution to return (0<q<1).
        Default 0.10 corresponds to the 10th percentile.
    seed
        Random seed for reproducible subsampling.

    Returns
    -------
    spacing : float
        Estimated characteristic point spacing (same units as x and y).
    """
    rng = np.random.default_rng(seed)
    n = x.size
    m = min(sample, n)
    idx = rng.choice(n, size=m, replace=False)
    pts = np.column_stack((x[idx], y[idx]))
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=2)  # dists[:,0]=0, dists[:,1]=NN distance
    nn = dists[:, 1]
    nn = nn[np.isfinite(nn) & (nn > 0)]
    if nn.size == 0:
        raise ValueError("Could not estimate spacing: NN distance array is empty.")
    return float(np.quantile(nn, q))


def _fit_plane_least_squares(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray
) -> Tuple[float, float, float]:
    """
    Fit a best-fit plane to a set of points by least squares.

    The plane model is:
        z = a*x + b*y + c

    Parameters
    ----------
    x, y, z
        1D arrays of coordinates for the samples used in the fit.
        Non-finite values are removed internally.

    Returns
    -------
    a, b, c : float
        Plane coefficients in z = a*x + b*y + c.
        (Units: a and b are "height per horizontal distance", c is height.)
    """
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[m], y[m], z[m]
    A = np.column_stack([x, y, np.ones_like(x)])
    coeff, *_ = np.linalg.lstsq(A, z, rcond=None)
    a, b, c = coeff
    return float(a), float(b), float(c)


def pointcloud_to_grid(
    mat_path: str,
    *,
    grid_dx: Optional[float] = None,
    grid_dy: Optional[float] = None,
    grid_nx: Optional[int] = None,
    grid_ny: Optional[int] = None,
    method: str = "linear",
    fill_nearest: bool = False,
    unit_scale: float = 1.0,
    regular_lattice_tol: float = 0.02,
    nn_sample: int = 60000,
    nn_quantile: float = 0.10,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Convert a 3D point cloud (X,Y,Z) into an equidistant gridded height map, and
    compute a 2D linearly detrended version of that height map.

    This function is designed for fault surface datasets provided as Matlab `.mat`
    files containing arrays X, Y, Z (or x, y, z). It supports two common data layouts:

    1) **Lattice-like (rectilinear) sampling** (often lab profilometers):
       - The dataset is essentially a Cartesian grid that has been flattened into
         1D vectors.
       - The function detects this case when N ≈ N_unique_x * N_unique_y.
       - In this mode, the height matrix is assembled by mapping each point to its
         nearest grid node (no 2D interpolation is required).

    2) **Irregular sampling** (often field LiDAR / SfM point clouds):
       - There is no natural grid.
       - The function builds an equidistant grid and interpolates linearly over a
         Delaunay triangulation (via `LinearNDInterpolator`).

    In both cases, the output includes:
      - `Z_grid`: the raw, non-detrended height map on the equidistant grid
      - `Z_detrended`: `Z_grid` after subtracting a best-fit plane (2D linear detrend)
      - grid spacing and surface extents needed for later profile extraction and scaling

    Coordinate / unit handling
    --------------------------
    The function assumes X, Y, Z are already in consistent physical units.
    If you need to convert (e.g., meters -> millimeters), set `unit_scale=1000`.
    The same scale factor is applied to X, Y, and Z to preserve isotropic units.

    Grid definition and spacing choice
    ----------------------------------
    You can control the target grid in two equivalent ways:

    A) Specify spacing directly:
        - `grid_dx` and/or `grid_dy` in the same units as X,Y (after scaling).

    B) Specify number of nodes:
        - `grid_nx` and/or `grid_ny` (integers) for the number of grid nodes.
        - If `grid_nx` is provided it overrides `grid_dx` in x (same for y).

    If neither spacing nor node count is provided, an **automatic spacing** is chosen:
      - lattice-like: median spacing of unique X and Y coordinate values.
      - irregular: estimate a characteristic point spacing using nearest-neighbor
        distances in (x,y), take the `nn_quantile` (default 10th percentile),
        and set the grid step to approximately `2 * spacing_estimate`
        (a conservative choice intended to avoid introducing artificial high
        frequencies by oversampling the interpolation).

    Handling of gaps / convex hull
    ------------------------------
    Linear interpolation on irregular clouds is only defined inside the convex hull
    of the input points. Grid nodes outside the hull (or inside holes) are returned
    as NaN. If `fill_nearest=True`, remaining NaNs are filled by nearest-neighbor
    assignment using a KD-tree. For spectral/fractal analysis, leaving NaNs and
    masking profiles is often preferable; filling may bias statistics near gaps.

    Parameters
    ----------
    mat_path
        Path to a Matlab `.mat` file containing:
          - `X`, `Y`, `Z` arrays (any shape), OR
          - `x`, `y`, `z` arrays.
        Arrays are flattened internally.

    grid_dx, grid_dy
        Desired grid spacing in x and y directions (float), in the same units
        as X,Y after applying `unit_scale`. Optional.
        If dataset is lattice-like and dx/dy are provided, the function **coarsens**
        the grid by sub-sampling the native coordinate arrays.

    grid_nx, grid_ny
        Number of grid nodes along x and y (int). Optional.
        If provided, overrides `grid_dx`/`grid_dy` for that direction and uses
        `np.linspace(min, max, n)` to define grid nodes.

    method
        Interpolation method for irregular clouds.
        Currently only `"linear"` is implemented (recommended and stable).

    fill_nearest
        If True, fill NaN grid cells after linear interpolation using nearest-neighbor
        assignment. Default False.

    unit_scale
        Multiplicative scale applied to X, Y, Z *together*.
        Examples:
          - meters to millimeters: unit_scale=1000
          - meters to micrometers: unit_scale=1e6
        Default 1.0 (no scaling).

    regular_lattice_tol
        Tolerance used to classify a dataset as lattice-like:
        abs(N_unique_x * N_unique_y - N) / (N_unique_x * N_unique_y) <= tol.
        Default 0.02 (2%).

    nn_sample
        Maximum number of points used to estimate nearest-neighbor distances for
        irregular clouds (speed vs robustness). Default 60000.

    nn_quantile
        Quantile of the nearest-neighbor distance distribution used as the
        characteristic spacing estimate (0<q<1). Default 0.10.

    seed
        Seed for subsampling in NN spacing estimation. Default 0.

    Returns
    -------
    result : dict
        Dictionary containing gridded data and summary scalars:

        **Grid coordinates**
        - `xg` : (nx,) ndarray
            1D array of x grid node coordinates (monotonically increasing).
        - `yg` : (ny,) ndarray
            1D array of y grid node coordinates (monotonically increasing).

        **Height maps**
        - `Z_grid` : (ny, nx) ndarray
            Raw gridded height map (non-detrended). NaN where undefined.
            Indexing convention: `Z_grid[j, i]` corresponds to `(xg[i], yg[j])`.
        - `Z_detrended` : (ny, nx) ndarray
            Detrended height map obtained by subtracting the best-fit plane from
            `Z_grid`. NaNs are preserved in the same locations as `Z_grid`.

        **Grid spacing / extents**
        - `dx` : float
            Grid spacing in x (xg[1]-xg[0]) in the working units.
        - `dy` : float
            Grid spacing in y (yg[1]-yg[0]) in the working units.
        - `Lx` : float
            Total surface length covered by the grid along x, computed as
            `(nx-1) * dx`.
        - `Ly` : float
            Total surface length covered by the grid along y, computed as
            `(ny-1) * dy`.
        - `z_range` : float
            Height range of the **raw** gridded surface, computed as
            `nanmax(Z_grid) - nanmin(Z_grid)` over finite cells.

        **Detrending model**
        - `plane_abc` : tuple(float, float, float)
            Coefficients (a, b, c) of the fitted plane:
                z_plane(x,y) = a*x + b*y + c
            such that:
                Z_detrended = Z_grid - z_plane(xg, yg)

        **Metadata / diagnostics**
        - `meta` : dict
            Diagnostic information such as:
              - lattice detection result
              - point count
              - coordinate bounds
              - interpolation method used
              - auto grid-step estimates (for irregular clouds)

    Notes
    -----
    1) Profile extraction:
       - x-direction profile at row j: `Z_grid[j, :]` (or `Z_detrended[j, :]`)
       - y-direction profile at column i: `Z_grid[:, i]`

    2) NaNs and analysis:
       For fBm/Hurst or spectral analysis, you typically want regularly sampled,
       gap-free profiles. Either:
         - select rows/columns with minimal NaNs, or
         - enable `fill_nearest` and document that gap-filling was applied.

    3) Orientation:
       This function does not rotate the surface; it assumes the provided X and Y
       axes are the intended directions for “along x” and “along y” profiles.
    """
    mat = scipy.io.loadmat(mat_path)
    for keyset in (("X", "Y", "Z"), ("x", "y", "z")):
        if all(k in mat for k in keyset):
            X = mat[keyset[0]]
            Y = mat[keyset[1]]
            Z = mat[keyset[2]]
            break
    else:
        raise KeyError(f"Expected variables X,Y,Z in {mat_path}, found keys: {list(mat.keys())}")

    x = np.asarray(X).ravel().astype(float) * unit_scale
    y = np.asarray(Y).ravel().astype(float) * unit_scale
    z = np.asarray(Z).ravel().astype(float) * unit_scale

    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[m], y[m], z[m]
    n = x.size
    if n < 3:
        raise ValueError("Need at least 3 points to grid and detrend.")

    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())

    is_lattice, ux_n, uy_n = _is_almost_regular_lattice(x, y, n, tol_frac=regular_lattice_tol)

    meta: Dict[str, Any] = {
        "mat_path": mat_path,
        "unit_scale": float(unit_scale),
        "n_points": int(n),
        "xmin_xmax": (xmin, xmax),
        "ymin_ymax": (ymin, ymax),
        "is_almost_regular_lattice": bool(is_lattice),
        "unique_x_count": int(ux_n),
        "unique_y_count": int(uy_n),
        "regular_lattice_tol": float(regular_lattice_tol),
        "fill_nearest": bool(fill_nearest),
    }

    # --- Build grid and raw height map ---
    if is_lattice:
        xg = np.unique(x)
        yg = np.unique(y)

        if grid_dx is not None and xg.size > 1:
            dx0 = float(np.median(np.diff(xg)))
            k = max(1, int(round(float(grid_dx) / dx0)))
            xg = xg[::k]
        if grid_dy is not None and yg.size > 1:
            dy0 = float(np.median(np.diff(yg)))
            k = max(1, int(round(float(grid_dy) / dy0)))
            yg = yg[::k]

        if grid_nx is not None:
            xg = np.linspace(xmin, xmax, int(grid_nx))
        if grid_ny is not None:
            yg = np.linspace(ymin, ymax, int(grid_ny))

        Z_grid = np.full((yg.size, xg.size), np.nan, dtype=float)

        xi = np.clip(np.searchsorted(xg, x, side="left"), 0, xg.size - 1)
        yi = np.clip(np.searchsorted(yg, y, side="left"), 0, yg.size - 1)
        xi = np.where((xi > 0) & (np.abs(xg[xi] - x) > np.abs(xg[xi - 1] - x)), xi - 1, xi)
        yi = np.where((yi > 0) & (np.abs(yg[yi] - y) > np.abs(yg[yi - 1] - y)), yi - 1, yi)

        Z_grid[yi, xi] = z

        dx = float(np.median(np.diff(xg))) if xg.size > 1 else np.nan
        dy = float(np.median(np.diff(yg))) if yg.size > 1 else np.nan

        meta["interpolation"] = "none (assembled from lattice-like data)"

    else:
        if method != "linear":
            raise ValueError("For irregular clouds, only method='linear' is implemented (recommended).")

        auto_d = None
        need_auto_x = (grid_dx is None) and (grid_nx is None)
        need_auto_y = (grid_dy is None) and (grid_ny is None)
        if need_auto_x or need_auto_y:
            s = _estimate_spacing_irregular(x, y, sample=nn_sample, q=nn_quantile, seed=seed)
            auto_d = 2.0 * s
            meta["nn_spacing_quantile"] = float(nn_quantile)
            meta["nn_spacing_estimate"] = float(s)
            meta["auto_grid_step"] = float(auto_d)

        if grid_nx is not None:
            xg = np.linspace(xmin, xmax, int(grid_nx))
        else:
            dx = float(grid_dx) if grid_dx is not None else float(auto_d)
            xg = np.arange(xmin, xmax + 0.5 * dx, dx)

        if grid_ny is not None:
            yg = np.linspace(ymin, ymax, int(grid_ny))
        else:
            dy = float(grid_dy) if grid_dy is not None else float(auto_d)
            yg = np.arange(ymin, ymax + 0.5 * dy, dy)

        dx = float(xg[1] - xg[0]) if xg.size > 1 else np.nan
        dy = float(yg[1] - yg[0]) if yg.size > 1 else np.nan

        XI, YI = np.meshgrid(xg, yg)

        interp = LinearNDInterpolator(list(zip(x, y)), z, fill_value=np.nan)
        Z_grid = interp(XI, YI)

        if fill_nearest:
            nan_mask = ~np.isfinite(Z_grid)
            if np.any(nan_mask):
                tree = cKDTree(np.column_stack((x, y)))
                qpts = np.column_stack((XI[nan_mask], YI[nan_mask]))
                _, idx = tree.query(qpts, k=1)
                Z_grid[nan_mask] = z[idx]

        meta["interpolation"] = "LinearNDInterpolator (linear over Delaunay triangulation)"

    # --- Detrend (remove best-fit plane) ---
    XI, YI = np.meshgrid(xg, yg)
    finite = np.isfinite(Z_grid)
    a, b, c = _fit_plane_least_squares(XI[finite], YI[finite], Z_grid[finite])
    Z_plane = a * XI + b * YI + c
    Z_detrended = Z_grid - Z_plane

    # --- Output scalars ---
    Lx = float((xg.size - 1) * (xg[1] - xg[0])) if xg.size > 1 else 0.0
    Ly = float((yg.size - 1) * (yg[1] - yg[0])) if yg.size > 1 else 0.0

    if np.any(finite):
        zmin = float(np.nanmin(Z_grid))
        zmax = float(np.nanmax(Z_grid))
        z_range = zmax - zmin
    else:
        z_range = np.nan

    meta.update({
        "dx": float(dx),
        "dy": float(dy),
        "grid_shape_ny_nx": (int(yg.size), int(xg.size)),
        "plane_abc": (a, b, c),
    })

    return {
        "xg": xg,
        "yg": yg,
        "Z_grid": Z_grid,
        "Z_detrended": Z_detrended,
        "dx": float(dx),
        "dy": float(dy),
        "Lx": float(Lx),
        "Ly": float(Ly),
        "z_range": float(z_range),
        "plane_abc": (a, b, c),
        "meta": meta,
    }
