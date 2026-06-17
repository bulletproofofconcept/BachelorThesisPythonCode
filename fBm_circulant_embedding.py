import numpy as np


# L-1 should be interpreted as length of the profile here (L profile points yield length m-1). 

def circulant_fBm(
    H: float,
    L: int,
    topothesy: float = 1.0,
    rng: np.random.Generator | None = None,
):
    """
    Generate TWO independent fractional Brownian motion (fBm) profiles of length m
    (including the initial 0) using a circulant-embedding FFT method on fractional Gaussian noise (fGn).

    Parameters
    ----------
    H : float
        Hurst exponent in (0, 1).
    m : int
        Number of sample points in the returned fBm path (so there are m-1 increments).
        Must be >= 2. Output arrays have shape (m,).
    topothesy : float, default 1.0
        Positive scaling factor applied to BOTH resulting fBm profiles. Must be > 0.
        With topothesy=1, the increments at unit step have Var=1 (i.e., c(0)=1).
    rng : np.random.Generator, optional
        NumPy random generator for reproducibility. If None, uses default_rng().

    Returns
    -------
    (fbm1, fbm2)
    fbm1 : np.ndarray, shape (m,)
    fbm2 : np.ndarray, shape (m,)
        Two i.i.d. fBm realizations.
    """
    if not (0.0 < H < 1.0):
        raise ValueError("H must be in (0, 1).")
    if L < 2:
        raise ValueError("m must be >= 2.")
    if topothesy <= 0:
        raise ValueError("topothesy must be positive.")
    if rng is None:
        rng = np.random.default_rng()

    # number of increments
    n = L - 1
    m2 = 2 * n
    H2 = 2.0 * H

    # fGn autocovariance:
    # c(k) = 0.5*(|k+1|^{2H} + |k-1|^{2H} - 2|k|^{2H})
    k1 = np.arange(0, n + 1, dtype=np.float64)   # 0..n
    k2 = np.arange(1, n, dtype=np.float64)       # 1..n-1

    c1 = 0.5 * (np.abs(k1 + 1.0) ** H2 + np.abs(k1 - 1.0) ** H2 - 2.0 * np.abs(k1) ** H2)  # len n+1
    c2 = 0.5 * (
        np.abs((n - k2) + 1.0) ** H2
        + np.abs((n - k2) - 1.0) ** H2
        - 2.0 * np.abs(n - k2) ** H2
    )  # len n-1

    s = np.concatenate((c1, c2))  # length 2n

    # eigenvalues (real up to roundoff); clamp tiny negatives
    lam = np.fft.fft(s).real
    lam[lam < 0.0] = 0.0

    # IMPORTANT normalization:
    # - NumPy ifft includes a 1/m2 factor
    # - Use complex eps with E|eps|^2 = 1 (divide by sqrt(2))
    # - Multiply by sqrt(m2) so the time-domain covariance matches s
    amp = np.sqrt(lam * m2)

    eps = (rng.standard_normal(m2) + 1j * rng.standard_normal(m2)) / np.sqrt(2.0)

    e = np.fft.ifft(amp * eps)  # length 2n, complex

    fgn1 = e.real[:n]
    fgn2 = e.imag[:n]

    # integrate to fBm; include initial 0
    fbm1 = np.empty(L, dtype=np.float64)
    fbm2 = np.empty(L, dtype=np.float64)
    fbm1[0] = 0.0
    fbm2[0] = 0.0
    fbm1[1:] = np.cumsum(fgn1)
    fbm2[1:] = np.cumsum(fgn2)

    fbm1 *= topothesy
    fbm2 *= topothesy
    return fbm1, fbm2


# N should be interpreted as point density per unit interval here
def fBm_on_unit_interval(
    H: float,
    N: int,
    topothesy: float = 1.0,
    rng: np.random.Generator | None = None
) -> np.ndarray:
    """
    Generate one fBm realization sampled on [0, 1] with N points (including endpoints).
    Step size: dt = 1/(N-1). Uses self-similarity: B_H(dt*k) = dt^H B_H(k).

    With topothesy=1, this targets:
        E[(B(t)-B(s))^2] = |t-s|^{2H}
    on the grid {0, 1/(N-1), ..., 1}.
    """
    if N < 2:
        raise ValueError("N must be >= 2")
    if rng is None:
        rng = np.random.default_rng()

    fbm1, _ = circulant_fBm(H=H, m=N, topothesy=topothesy, rng=rng)  # length N
    dt = 1.0 / (N - 1)
    return (dt ** H) * fbm1


