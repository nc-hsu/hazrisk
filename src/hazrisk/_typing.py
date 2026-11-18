"""Type aliases and input conversion shared by the hazrisk modules (private).

Public functions take ``numpy.typing.ArrayLike`` inputs (a number, a list, a
numpy array, ...) and return ``float | FloatArray``: a scalar for scalar
inputs, otherwise a float array with the broadcast shape of the inputs.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

# A float64 numpy array of any shape.
type FloatArray = NDArray[np.float64]

# What the functions return: a scalar for scalar input, otherwise an array.
type FloatOrArray = float | FloatArray


def as_float(x: ArrayLike) -> FloatOrArray:
    """Convert a numeric input to float, leaving plain Python numbers alone.

    Lists, tuples and numpy arrays (including integer arrays) become float
    arrays. Integer arrays need this because numpy refuses negative integer
    powers (``np.array([1, 2]) ** -2.4`` raises), and lists because they do
    not support arithmetic at all.

    Python ``int`` and ``float`` are returned unchanged, so a call with
    scalar arguments is evaluated with Python's own scalar arithmetic and
    returns exactly what it would without the conversion. (Turning them into
    0-d arrays would switch some operations to numpy's versions, e.g. ``x**2``
    to ``x*x``, which can differ in the last bit.)
    """
    if isinstance(x, (int, float)):
        return x
    return np.asarray(x, dtype=float)
