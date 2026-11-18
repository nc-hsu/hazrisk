"""Type aliases shared by the hazrisk modules (private).

``scalar`` is a single number and ``vector`` a float array. Most public
functions accept ``scalar | vector`` for the intensity measure, which means
they must stay numpy-broadcastable.
"""

from typing import Union

import numpy as np
from numpy.typing import NDArray

scalar = Union[float, int]
vector = NDArray[np.float64]
