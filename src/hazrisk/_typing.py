"""Type aliases and input conversion shared by the hazrisk modules (private).

``scalar`` is a single number and ``vector`` a float array. Most public
functions accept ``scalar | vector`` for the intensity measure, which means
they must stay numpy-broadcastable.
"""

from typing import Union

import numpy as np
from numpy.typing import NDArray

scalar = Union[float, int]
vector = NDArray[np.float64]


def as_float(x):
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
