import numpy as np
def fps(xyz, count):
    selected = np.empty(count, dtype=np.int32)
    distance = np.full(len(xyz), np.inf, dtype=np.float32)
    current = int(np.argmax(np.sum((xyz - xyz.mean(axis=0)) ** 2, axis=1)))
    for step in range(count):
        selected[step] = current
        diff = xyz - xyz[current]
        distance = np.minimum(distance, np.einsum('ij,ij->i', diff, diff))
        current = int(np.argmax(distance))
    return selected
