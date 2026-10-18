"""Free-space electrostatics and an outgoing termination for the hidden grid."""
from functools import lru_cache
import math
import numpy as np
from numba import njit


@lru_cache(maxsize=4)
def free_space_kernel(nx, ny, dx, dy):
    """FFT of the infinite-lattice Green function for the Yee Laplacian.

    For L = -Dxx-Dyy, integrate one Fourier variable analytically:
      G[n,m]-G[0,0] = integral_0^pi
        (exp(-n*alpha)*cos(m*t)-1)/(2*pi*a*sinh(alpha)) dt,
      a=1/dx**2, cosh(alpha)=1+(dx/dy)**2*(1-cos(t)).
    The subtracted constant removes the 2D logarithmic divergence, not net
    charge. L G = delta, including for a non-neutral scene. This matches the
    discrete divergence exactly, unlike sampling the continuum log kernel.
    See https://arxiv.org/abs/cond-mat/0509002 for the lattice formulation.

    Quadrature is only needed once per grid. Folding the symmetric kernel on
    a doubled mesh makes the subsequent FFT convolution aperiodic in the
    requested domain; it does NOT introduce periodic images of the charges.
    """
    order = max(128, 2*ny+32, math.ceil(12*math.sqrt(nx*dx/dy)))
    nodes, weights = np.polynomial.legendre.leggauss(order)
    theta = (nodes+1)*np.pi/2
    alpha = 2*np.arcsinh((dx/dy)*np.sin(theta/2))
    weights = weights*dx**2/(4*np.sinh(alpha))
    decay = np.exp(-np.arange(nx+1)[:, None]*alpha)
    cosine = np.cos(np.arange(ny+1)[:, None]*theta)
    green = (decay*weights) @ cosine.T - np.sum(weights)
    green[0, 0] = 0.0
    ix, iy = np.arange(2*nx), np.arange(2*ny)
    kernel = green[np.minimum(ix, 2*nx-ix)[:, None],
                   np.minimum(iy, 2*ny-iy)[None, :]]
    spectrum = np.fft.rfft2(kernel)
    spectrum.flags.writeable = False
    return spectrum


def free_space_potential(source, kernel):
    """Solve -Laplacian(phi)=source without grounding the perimeter."""
    nx, ny = source.shape[0]-1, source.shape[1]-1
    shape = (2*nx, 2*ny)
    spectrum = np.fft.rfft2(source, s=shape)*kernel
    return np.fft.irfft2(spectrum, s=shape)[:nx+1, :ny+1]


@njit(cache=True)
def apply_mur_boundary(bz, centered_b, dx, dy, dt, c):
    """First-order outgoing (Mur) condition on the outermost Bz samples.

    Also used for the nodal removal scalar, which has the same exterior wave
    equation and half-step timing. Array dimensions determine its perimeter.
    Bz obeys the scalar wave equation in the source-free exterior. Using Bz
    for the termination leaves the electric curl update and its discrete
    Gauss identity intact. A stationary curl-free Coulomb field is untouched.
    Formula: new_edge = old_inner + (c*dt-h)/(c*dt+h)*(new_inner-old_edge).
    Old B is recovered from the time average produced by advance_b, avoiding
    a full extra field copy. The two one-way predictions are averaged at corners.
    The outer tangential E samples are unused ghosts: their trial B curl is
    replaced here. There is no material wall at the visible canvas boundary.
    """
    nx, ny = bz.shape
    rx, ry = (c*dt-dx)/(c*dt+dx), (c*dt-dy)/(c*dt+dy)
    for edge, inner in ((0, 1), (nx-1, nx-2)):
        for j in range(1, ny-1):
            old = 2*centered_b[edge, j]-bz[edge, j]
            previous_inner = 2*centered_b[inner, j]-bz[inner, j]
            new = previous_inner+rx*(bz[inner, j]-old)
            bz[edge, j], centered_b[edge, j] = new, .5*(old+new)
    for edge, inner in ((0, 1), (ny-1, ny-2)):
        for i in range(1, nx-1):
            old = 2*centered_b[i, edge]-bz[i, edge]
            previous_inner = 2*centered_b[i, inner]-bz[i, inner]
            new = previous_inner+ry*(bz[i, inner]-old)
            bz[i, edge], centered_b[i, edge] = new, .5*(old+new)
    for i, ii in ((0, 1), (nx-1, nx-2)):
        for j, jj in ((0, 1), (ny-1, ny-2)):
            old = 2*centered_b[i, j]-bz[i, j]
            x = 2*centered_b[ii, j]-bz[ii, j]+rx*(bz[ii, j]-old)
            y = 2*centered_b[i, jj]-bz[i, jj]+ry*(bz[i, jj]-old)
            new = .5*(x+y)
            bz[i, j], centered_b[i, j] = new, .5*(old+new)
