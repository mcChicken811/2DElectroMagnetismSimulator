"""Compact PIC source filter; never a filter on the evolved/displayed fields."""
import numpy as np
from numba import njit


@njit(cache=True)
def binomial_filter(values, passes):
    """Apply [1, 2, 1]/4 in x and y per pass, with zero exterior extension.

    Filter rho and both orbit currents with exactly this same convolution:
    it commutes with interior Yee differences and thus preserves continuity.
    Filtering a copy of the gathered fields applies the adjoint particle shape.
    Sources and gather locations are separated from the outer edge by padding.
    Input is never mutated. Zero passes returns the original array (read-only
    by convention), avoiding allocation in the unfiltered comparison mode.
    """
    if passes == 0:
        return values
    result = values.copy()
    scratch = np.empty_like(result)
    nx, ny = values.shape
    for _ in range(passes):
        for i in range(nx):
            for j in range(ny):
                value = .5*result[i, j]
                if i > 0:
                    value += .25*result[i-1, j]
                if i+1 < nx:
                    value += .25*result[i+1, j]
                scratch[i, j] = value
        for i in range(nx):
            for j in range(ny):
                value = .5*scratch[i, j]
                if j > 0:
                    value += .25*scratch[i, j-1]
                if j+1 < ny:
                    value += .25*scratch[i, j+1]
                result[i, j] = value
    return result
