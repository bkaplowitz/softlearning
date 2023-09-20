import numpy as np


def spherical(size=None, ndim=2):
    size = np.atleast_1d(size if size is not None else ())
    random_normal = np.random.standard_normal((ndim, *size))
    return random_normal / np.linalg.norm(random_normal, axis=0)
