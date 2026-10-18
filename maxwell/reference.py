"""Uniform-motion 2D line-charge reference, derived by a Lorentz field boost.

This is an explicit initial-condition tool, never a replacement for a Yee step.
The rest field is q R'/(2*pi*|R'|²); z-invariant charge per unit z is unchanged
by a boost within xy. This differs from the 3D point-charge angular dependence.
"""
import math
import numpy as np
from .filtering import binomial_filter


def moving_line_field(x, y, position, velocity, charge, c, core=0.):
    """Lab E/B at equal lab time for an eternally uniformly moving line charge.

    A small rest-frame core is used only when sampling the singular initial
    condition onto the mesh. core=0 gives the ideal continuum reference.
    """
    vx, vy = velocity
    beta2 = (vx*vx+vy*vy)/c**2
    if not np.isfinite(beta2) or beta2 >= 1:
        raise ValueError('The reference velocity must be finite and below c.')
    gamma = 1/math.sqrt(1-beta2)
    rx, ry = np.asarray(x)-position[0], np.asarray(y)-position[1]
    denominator = rx*rx+ry*ry+gamma**2*((vx*rx+vy*ry)/c)**2+core**2
    factor = charge*gamma/(2*np.pi*np.maximum(denominator, np.finfo(float).tiny))
    ex, ey = factor*rx, factor*ry
    return ex, ey, (vx*ey-vy*ex)/c**2


def prepare_uniform_motion(sim, index):
    """Replace fields with a single-source uniform-motion test initial state.

    Seed a Lorentz-boosted field, project E onto the existing discrete CIC Gauss
    constraint, and center B correctly in the leapfrog representation. This
    approximates a moving singular source on a finite mesh; refine away from
    the core to assess accuracy. Existing radiation/removal waves are cleared
    ONLY by this explicit operation. Time and particle position stay unchanged.
    """
    if len(sim.charge) != 1 or index != 0:
        raise ValueError('Steady-motion fields require exactly one selected charge.')
    velocity, _ = sim.particle_state(index)
    if np.linalg.norm(velocity) >= sim.config.c:
        raise ValueError('Velocity must be below c.')
    p = sim.config.field_padding
    passes = sim.config.source_filter_passes
    core = .25*min(sim.dx, sim.dy)
    arrays = []
    for field, offset, component in ((sim._ex, (.5, 0.), 0),
                                     (sim._ey, (0., .5), 1),
                                     (sim._bz, (.5, .5), 2)):
        # Unlike compact deposited sources, the analytic field does not vanish
        # outside the grid. Sample an extra filter halo before convolving, so
        # zero extension cannot create a spurious perimeter charge layer.
        x = (np.arange(-passes, field.shape[0]+passes)-p+offset[0])[:, None]*sim.dx
        y = (np.arange(-passes, field.shape[1]+passes)-p+offset[1])[None, :]*sim.dy
        sampled = moving_line_field(x, y, sim.pos[index], velocity,
                                    sim.charge[index], sim.config.c, core)[component]
        filtered = binomial_filter(sampled, passes)
        arrays.append(filtered[passes:-passes, passes:-passes] if passes else filtered)
    sim._ex[:], sim._ey[:], sim._bz[:] = arrays
    sim.update_rho()
    sim.correct_gauss()
    # E(t), B(t-dt/2): the first advance_b will gather exactly the seeded B(t).
    curl = np.diff(sim._ey, axis=0)/sim.dx-np.diff(sim._ex, axis=1)/sim.dy
    sim._bz += .5*sim.dt*curl
    sim._centered_b[:] = arrays[2]
    sim._jx[:] = 0
    sim._jy[:] = 0
    sim._removal_psi[:] = 0
    sim._centered_psi[:] = 0
    sim.removal_active = False
    sim.set_motion(index, velocity=velocity, acceleration=np.zeros(2))
