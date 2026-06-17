from Extract_Profiles import change_probabilities_2D
from Polar_CP_plot import cp_hurst_polar
import numpy as np



#data = np.load("C:\\Users\\Flo\\Desktop\\BA\\convert_output\\bolu2_grids.npz")
data = np.load("C:\\Users\\Flo\\Desktop\\BA\\convert_output\\coronaA_grids.npz")
grid = data["Z_grid"]       # 2D Feld
x_coords = data["xg"]       # 1D x-Koordinaten
y_coords = data["yg"]       # 1D y-Koordinaten

x_coords_pix = np.linspace(0, grid.shape[1]-1, grid.shape[1])
y_coords_pix = np.linspace(0, grid.shape[0]-1, grid.shape[0])

phi = np.linspace(0, np.pi, 10)  # 10 Winkel von 0 bis 180°
taus = [1, 2, 4, 8]



def main() -> None:

    
    #print(change_probabilities_2D(grid, x_coords_pix, y_coords_pix, phi, taus))

    print("Grid shape:", grid.shape)
    print("x_coords shape:", x_coords.shape)
    print("y_coords shape:", y_coords.shape)
    print("x_coords[0], x_coords[-1]:", x_coords[0], x_coords[-1])
    print("y_coords[0], y_coords[-1]:", y_coords[0], y_coords[-1])
    




    
    #path = "C:\\Users\\Flo\\Desktop\\BA\\convert_output\\Bolu_2.jpg"
    path = "C:\\Users\\Flo\\Desktop\\BA\\convert_output\\Corona_A.jpg"

    # --- 2. Parameter für die CP/Hurst Analyse ---
    taus = [3, 5, 8, 13]
    n_directions = 72
    profile_spacing = 2
    interpolation = "linear"
    detrend2d = "linear"
    detrend1d = "none"
    aggregate = "median"
    title_prefix = "Sample: "

    # --- 3. Analyse aufrufen ---
    phis, taus, P, H = cp_hurst_polar(
        img_or_height=path,
        taus=taus,
        n_directions=n_directions,
        profile_spacing=profile_spacing,
        interpolation=interpolation,
        detrend2d=detrend2d,
        detrend1d=detrend1d,
        aggregate=aggregate,
        title_prefix=title_prefix,
        show=True,         # zeigt automatisch die Polar-Plots
        return_data=True
    )








    print(data)





if __name__ == "__main__":
    main()