import numpy as np
from CP import estimate_change_probability
from CP import hurst_from_change_probability
from scipy.interpolate import RegularGridInterpolator

def change_probabilities_2D(grid, x_coords, y_coords, phi, taus, delta=1):
    """
    Berechnet die Change-Probability Matrix für ein 2D-Feld und die Hurst-Exponenten.

    grid      : 2D numpy array, z.B. Z_grid aus NPZ
    x_coords  : 1D array für x-Achse, z.B. xg
    y_coords  : 1D array für y-Achse, z.B. yg
    phi       : Liste von Winkeln in Radianten
    taus      : Liste von Delays (stepsize für die 1D change probability)
    delta     : Abstand der parallelen Linien
    """

    n_y, n_x = grid.shape
    d = int(np.ceil(np.sqrt(n_y**2 + n_x**2)))
    d_delta = int(np.ceil(np.sqrt(n_y**2 + n_x**2)/delta))

    # Interpolator auf vorhandenes Grid
    interpolant = RegularGridInterpolator(
        (y_coords, x_coords),
        grid,
        method="linear",
        bounds_error=False,
        fill_value=np.nan
    )

    z = np.arange(-(d+1), d+1) + 0.5
    z_delta = np.arange(-(d_delta+1)*delta, d_delta*delta+1, delta) + 0.5

    prob = np.zeros((len(phi), len(taus)))

    for m, angle in enumerate(phi):
        print(f"Processing parallel lines with angle number {m+1}/{len(phi)}")
        for k in range(len(z_delta)):
            # Linie erzeugen (Rotation + Parallelverschiebung)
            line_x = np.cos(angle)*z - np.sin(angle)*z_delta[k]
            line_y = np.sin(angle)*z + np.cos(angle)*z_delta[k]

            in_image = (line_x > x_coords[0]) & (line_x < x_coords[-1]) & \
                       (line_y > y_coords[0]) & (line_y < y_coords[-1])

            if not np.any(in_image):
                continue

            points = np.vstack((line_y[in_image], line_x[in_image])).T
            data_line = interpolant(points)
            data_line = data_line[~np.isnan(data_line)]

            if len(data_line) < 3:
                continue  # zu kurz für mindestens eine Triple

            # Für alle Delays/taus die Change-Probability berechnen
            for j, tau in enumerate(taus):
                if len(data_line) <= 2*tau:
                    continue
                p = estimate_change_probability(data_line, tau)
                prob[m, j] += p  # summieren über alle parallelen Linien

        # mitteln über alle parallelen Linien
        prob[m, :] /= len(z_delta)

    # Hurst Exponenten berechnen
    H = np.array([hurst_from_change_probability(p) for p in prob.flatten()]).reshape(prob.shape)

    return prob, H