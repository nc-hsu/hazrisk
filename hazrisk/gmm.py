from pathlib import Path 
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / '_data'

def bjf97(T1: float, M: float, r_jb: float, Vs, mechanism: str='all', 
          epsilon: float=0.0) -> float:
    # Boore, Joyner, and Fumal (1997) Ground Motion Model
    # returns median Y and sigma_lnY for given T1, M, r_jb, Vs, and mechanism
    # get the coefficients
    df = pd.read_csv(DATA_DIR / 'BJF97_coefficients.csv', header=0).set_index('Period')
    cs = df.reindex(df.index.union([T1])).sort_index().interpolate(method='index').loc[T1] # coefficients for T1

    if mechanism == "strike-slip": b1 = cs["B_1SS"]; 
    elif mechanism == "reverse-slip": b1 = cs["B_1RV"]; 
    else: b1 = cs["B_1ALL"]

    Y = _bjf97_Y(M, r_jb, Vs, b1, cs["B_2"], cs["B_3"], cs["B_5"], cs["B_V"], cs["V_A"], cs["h"])
    
    sigma_lnY = cs["sigma_lny"]
    Y *= np.exp(epsilon * sigma_lnY)  # adjust for epsilon
    return (Y, sigma_lnY)


def _bjf97_Y(M:float, r_jb:float, Vs:float, b1:float, b2:float, 
             b3:float, b5:float, bV:float, VA:float, h:float) -> float:
    # returns Y and not lnY
    r = np.sqrt(r_jb**2 + h**2)  # in km
    lnY = b1 + b2 * (M - 6) + b3 * (M - 6)**2 + b5 * np.log(r) + bV * np.log(Vs / VA)
    Y = np.exp(lnY)
    return Y


if __name__ == "__main__":
    print(bjf97(0.2, 6, 20, 300, mechanism='strike-slip'))