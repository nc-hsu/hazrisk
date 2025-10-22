import numpy as np

def BJ08(T1: float, T2: float) -> float:
    T_min = min(T1, T2)
    T_max = max(T1, T2)

    if T_max < 0.109:
        return _C2(T_min, T_max)
    elif T_min > 0.109:
        return _C1(T_min, T_max)
    elif T_max < 0.2:
        return min(_C2(T_min, T_max), _C4(T_min, T_max))
    else:
        return _C4(T_min, T_max)
    
def _C1(T_min: float, T_max: float) -> float:
    return 1 - np.cos(np.pi/2 - 0.366 * np.log(T_max / max(T_min, 0.109))) 


def _C2(T_min: float, T_max: float) -> float:
    if T_max < 0.2:
        return 1 - 0.105 * (1 - 1 / (1 + np.exp(100 * T_max - 5))) * (T_max - T_min) / (T_max - 0.0099)
    return 0


def _C3(T_min: float, T_max: float) -> float:
    if T_max < 0.109:
        return _C2(T_max, T_min)
    return _C1(T_max, T_min) 


def _C4(T_min: float, T_max: float) -> float:
    c1 = _C1(T_max, T_min)
    c3 = _C3(T_max, T_min)
    return c1 + 0.5 * (np.sqrt(c3) - c3) * (1 + np.cos(np.pi * T_min / 0.109))

if __name__ == "__main__":
    print(BJ08(0.2, 1.0))