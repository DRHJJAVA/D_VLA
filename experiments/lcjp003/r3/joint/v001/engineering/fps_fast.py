import numpy as np
from numba import njit
@njit(nogil=True,fastmath=False,cache=True)
def _fps_core(x, n, center):
    n = min(n, len(x))
    out = np.zeros(n, dtype=np.int64)
    ds = np.full(len(x), np.inf)
    p = int(np.argmax(((x - center) ** 2).sum(1)))
    for j in range(n):
        out[j] = p
        ds = np.minimum(ds, ((x - x[p]) ** 2).sum(1))
        p = int(ds.argmax())
    return out
def fps(x,n):
    return _fps_core(x,n,x.mean(0))
