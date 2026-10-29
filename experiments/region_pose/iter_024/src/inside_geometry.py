import numpy as np
def inside(p, q, margin):
    lo = np.array([-0.008, q[1] + 0.0005, 0.0854]) - margin
    hi = np.array([0.008, q[0] - 0.0005, 0.1014]) + margin
    return np.all(p >= lo, axis=-1) & np.all(p <= hi, axis=-1)
