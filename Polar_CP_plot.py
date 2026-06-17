import numpy as np
import matplotlib.pyplot as plt




def cp_hurst_polar(
    img_or_height,
    taus,
    n_directions=72,
    phi_range=None,                  # None => full 360° (2π)
    profile_spacing=1,
    interpolation="linear",          # "nearest" | "linear" | "exp"
    exp_sigma=0.75,                  # only for "exp"
    detrend2d="none",                # "none" | "linear" | "quadratic"
    detrend1d="none",                # "none" | "linear" | "quadratic"
    aggregate="mean",                # "mean" | "median"
    title_prefix="",
    show=True,
    return_data=True,
    dpi=600
):
    """
    Author: Maxim Glyzhev
    e-Mail: fn6237@kit.edu
    Affiliation: KIT Institut der Stochastik.
    
    One-shot function:
      - input: image path OR 2D height matrix
      - computes directional CP p_phi(tau) and H_phi(tau)=h(p)
      - outputs: TWO polar plots (CP and H), and returns arrays if requested

    Returns (if return_data=True):
      phis, taus_sorted, P, H
      where P.shape == H.shape == (n_directions, len(taus))
    """
    import numpy as np
    import matplotlib.pyplot as plt

    try:
        from PIL import Image
    except Exception:
        Image = None

    if phi_range is None:
        phi_range = 2.0 * np.pi  # full 360°

    # -------------------------
    # helpers (nested: still a single public function)
    # -------------------------
    def _h_from_p(p):
        p = np.asarray(p, dtype=float)
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return 1.0 + np.log2(np.sin(np.pi * (1.0 - p) / 2.0))

    def _cp_1d(x, tau):
        x = np.asarray(x, dtype=float)
        m = x.size - 1
        if tau <= 0 or (m < 2 * tau):
            return np.nan
        a = x[0 : (m - 2 * tau + 1)]
        b = x[tau : (m - tau + 1)]
        c = x[2 * tau : (m + 1)]
        changes = ((a < b) & (b >= c)) | ((a >= b) & (b < c))
        return float(np.sum(changes) / (m - 2 * tau + 1))

    def _detrend_1d(x, mode):
        x = np.asarray(x, dtype=float)
        if mode == "none":
            return x
        t = np.arange(x.size, dtype=float)
        deg = 1 if mode == "linear" else 2 if mode == "quadratic" else None
        if deg is None:
            raise ValueError("detrend1d must be 'none', 'linear', or 'quadratic'")
        coeff = np.polyfit(t, x, deg=deg)
        return x - np.polyval(coeff, t)

    def _detrend_2d(A, mode):
        A = np.asarray(A, dtype=float)
        if mode == "none":
            return A
        h, w = A.shape
        yy, xx = np.mgrid[0:h, 0:w]
        x = xx.ravel().astype(float)
        y = yy.ravel().astype(float)
        z = A.ravel().astype(float)

        if mode == "linear":
            X = np.column_stack([np.ones_like(x), x, y])
        elif mode == "quadratic":
            X = np.column_stack([np.ones_like(x), x, y, x**2, x*y, y**2])
        else:
            raise ValueError("detrend2d must be 'none', 'linear', or 'quadratic'")

        beta, *_ = np.linalg.lstsq(X, z, rcond=None)
        z_fit = (X @ beta).reshape(h, w)
        return A - z_fit

    def _sample_2d(A, x, y, method, exp_sigma=0.75):
        h, w = A.shape

        if method == "nearest":
            xi = int(round(x))
            yi = int(round(y))
            xi = min(max(xi, 0), w - 1)
            yi = min(max(yi, 0), h - 1)
            return float(A[yi, xi])

        x0 = int(np.floor(x)); y0 = int(np.floor(y))
        x1 = x0 + 1;          y1 = y0 + 1

        x0c = min(max(x0, 0), w - 1)
        x1c = min(max(x1, 0), w - 1)
        y0c = min(max(y0, 0), h - 1)
        y1c = min(max(y1, 0), h - 1)

        q00 = A[y0c, x0c]; q10 = A[y0c, x1c]
        q01 = A[y1c, x0c]; q11 = A[y1c, x1c]

        if method == "linear":
            dx = x - x0
            dy = y - y0
            v0 = (1 - dx) * q00 + dx * q10
            v1 = (1 - dx) * q01 + dx * q11
            return float((1 - dy) * v0 + dy * v1)

        if method == "exp":
            pts = np.array([[x0c, y0c], [x1c, y0c], [x0c, y1c], [x1c, y1c]], dtype=float)
            vals = np.array([q00, q10, q01, q11], dtype=float)
            d2 = (pts[:, 0] - x) ** 2 + (pts[:, 1] - y) ** 2
            wts = np.exp(-d2 / (2.0 * exp_sigma**2))
            s = float(np.sum(wts))
            if s <= 0:
                xi = min(max(int(round(x)), 0), w - 1)
                yi = min(max(int(round(y)), 0), h - 1)
                return float(A[yi, xi])
            return float(np.sum(wts * vals) / s)

        raise ValueError("interpolation must be 'nearest', 'linear', or 'exp'")

    def _line_rect_t_interval(origin, direction, w, h):
        ox, oy = origin
        dx, dy = direction
        xmin, xmax = 0.0, float(w - 1)
        ymin, ymax = 0.0, float(h - 1)

        tmin, tmax = -np.inf, np.inf

        if abs(dx) < 1e-12:
            if not (xmin <= ox <= xmax):
                return None
        else:
            tx1 = (xmin - ox) / dx
            tx2 = (xmax - ox) / dx
            tmin = max(tmin, min(tx1, tx2))
            tmax = min(tmax, max(tx1, tx2))

        if abs(dy) < 1e-12:
            if not (ymin <= oy <= ymax):
                return None
        else:
            ty1 = (ymin - oy) / dy
            ty2 = (ymax - oy) / dy
            tmin = max(tmin, min(ty1, ty2))
            tmax = min(tmax, max(ty1, ty2))

        if tmax < tmin:
            return None
        return (tmin, tmax)

    def _extract_profiles(A, phi, profile_spacing, interpolation, exp_sigma):
        h, w = A.shape
        cx = (w - 1) / 2.0
        cy = (h - 1) / 2.0
        origin = np.array([cx, cy], dtype=float)

        # phi measured from +vertical axis => d = (sin(phi), cos(phi))
        d = np.array([np.sin(phi), np.cos(phi)], dtype=float)
        d /= (np.linalg.norm(d) + 1e-15)

        # normal for parallel offsets
        n = np.array([np.cos(phi), -np.sin(phi)], dtype=float)

        max_shift = 0.5 * np.hypot(w, h)
        shifts = np.arange(-max_shift, max_shift + 1e-9, float(profile_spacing), dtype=float)

        profiles = []
        for s in shifts:
            o2 = origin + s * n
            interval = _line_rect_t_interval(o2, d, w, h)
            if interval is None:
                continue
            tmin, tmax = interval
            t0 = int(np.ceil(tmin))
            t1 = int(np.floor(tmax))
            if t1 - t0 + 1 < 3:
                continue

            xs = []
            for t in range(t0, t1 + 1):
                px, py = (o2 + t * d)
                xs.append(_sample_2d(A, px, py, method=interpolation, exp_sigma=exp_sigma))
            profiles.append(np.asarray(xs, dtype=float))
        return profiles

    def _make_polar_figure(theta, Y, taus, title):
        fig = plt.figure(dpi=dpi)
        ax = fig.add_subplot(111, projection="polar")

        # --- CLOSE THE CIRCLE ---
        theta_closed = np.concatenate([theta, [theta[0] + 2.0 * np.pi]])
        Y_closed = np.vstack([Y, Y[0:1, :]])  # append first direction row

        for j, tau in enumerate(taus):
            ax.plot(theta_closed, Y_closed[:, j], label=f"τ={int(tau)}")

        ax.set_title(title, y=1.18)
        fig.subplots_adjust(top=0.80, bottom=0.22)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=min(len(taus), 6),
            frameon=True
        )
        return fig, ax

    # -------------------------
    # load / validate input
    # -------------------------
    if isinstance(img_or_height, str):
        if Image is None:
            raise ImportError("PIL not available. Install pillow: pip install pillow")
        img = Image.open(img_or_height).convert("L")
        A = np.asarray(img, dtype=float)
    else:
        A = np.asarray(img_or_height, dtype=float)
        if A.ndim != 2:
            raise ValueError("If not passing a path, img_or_height must be a 2D array.")

    taus = [int(t) for t in taus]
    if any(t <= 0 for t in taus):
        raise ValueError("taus must all be positive integers.")
    taus = np.array(sorted(set(taus)), dtype=int)

    if interpolation not in {"nearest", "linear", "exp"}:
        raise ValueError("interpolation must be 'nearest', 'linear', or 'exp'")
    if detrend2d not in {"none", "linear", "quadratic"}:
        raise ValueError("detrend2d must be 'none', 'linear', or 'quadratic'")
    if detrend1d not in {"none", "linear", "quadratic"}:
        raise ValueError("detrend1d must be 'none', 'linear', or 'quadratic'")
    if aggregate not in {"mean", "median"}:
        raise ValueError("aggregate must be 'mean' or 'median'")

    A = _detrend_2d(A, detrend2d)

    # -------------------------
    # compute directional P and H (full 360° by default)
    # -------------------------
    phis = np.linspace(0.0, float(phi_range), int(n_directions), endpoint=False)
    P = np.full((phis.size, taus.size), np.nan, dtype=float)
    H = np.full_like(P, np.nan)

    for k, phi in enumerate(phis):
        profs = _extract_profiles(A, phi, profile_spacing, interpolation, exp_sigma)

        for j, tau in enumerate(taus):
            vals = []
            for x in profs:
                x2 = _detrend_1d(x, detrend1d)
                cpv = _cp_1d(x2, int(tau))
                if np.isfinite(cpv):
                    vals.append(cpv)

            if len(vals) == 0:
                continue

            vals = np.asarray(vals, dtype=float)
            p_dir = float(np.mean(vals)) if aggregate == "mean" else float(np.median(vals))
            P[k, j] = p_dir
            H[k, j] = _h_from_p(p_dir)

    # Matplotlib polar angles are from +x; our phi is from +y -> rotate by +pi/2
    theta = (phis + np.pi / 2.0) % (2.0 * np.pi)

    _make_polar_figure(theta, P, taus, f"{title_prefix}CP p(φ,τ)")
    _make_polar_figure(theta, H, taus, f"{title_prefix}H(φ(τ)) = H(p(φ,τ))")

    if show:
        plt.show()

    if return_data:
        return phis, taus, P, H



#path = '/Users/maximglischev/Documents/testfield/BF7.png'





'''
phis, taus, P, H = cp_hurst_polar(
    path,
    taus=[3, 5, 8, 13],
    n_directions=72,
    profile_spacing=2,
    interpolation="linear",
    detrend2d="linear",
    detrend1d="none",
    aggregate="median",
    title_prefix="Sample: "
)
'''

