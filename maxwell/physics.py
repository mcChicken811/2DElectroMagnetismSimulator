"""Yee Maxwell solver with filtered CIC sources and conservative orbit currents.

Coordinates: rho[i,j] is nodal; Ex[i,j] is at (i+1/2,j), Ey[i,j]
at (i,j+1/2), Bz[i,j] at (i+1/2,j+1/2). E and positions are at
integer times; momentum and B are staggered by half a step. Prescribed-motion
test particles instead store integer-time momentum and bypass the force push.
"""
from dataclasses import dataclass
import math
import numpy as np
from numba import njit
from .open_boundary import free_space_kernel, free_space_potential, apply_mur_boundary
from .filtering import binomial_filter

# CODATA 2022: https://physics.nist.gov/cgi-bin/cuu/Value?mpsme
# Masses are measured in electron masses. Keep the existing charge/field scale.
PROTON_ELECTRON_MASS_RATIO = 1836.152673426
SPECIES = {'charge': 0, 'proton': 1, 'electron': 2}


class AccelerationLimitError(ValueError):
    """A prescribed coordinate acceleration would reach the speed of light."""


@njit(cache=True)
def drag_force(x, y, ux, uy, c, target):
    gamma = math.sqrt(1+(ux*ux+uy*uy)/(c*c))
    fx, fy = 90*(target[0]-x)-16*ux/gamma, 90*(target[1]-y)-16*uy/gamma
    scale = min(1.0, 180.0/max(math.hypot(fx, fy), 1e-15))
    return fx*scale, fy*scale


@njit(cache=True)
def sample(a, x, y, dx, dy, ox=0.0, oy=0.0):
    """Bilinear gathering at a grid's physical stagger, with edge clamping."""
    u = min(max(x / dx - ox, 0.0), a.shape[0] - 1.000000001)
    v = min(max(y / dy - oy, 0.0), a.shape[1] - 1.000000001)
    i, j = int(u), int(v)
    u, v = u - i, v - j
    return ((1-u)*(1-v)*a[i, j] + u*(1-v)*a[i+1, j]
            + (1-u)*v*a[i, j+1] + u*v*a[i+1, j+1])


@njit(cache=True)
def deposit_charge(pos, charge, rho, dx, dy):
    rho[:] = 0.0
    for p in range(len(charge)):
        u, v = pos[p, 0]/dx, pos[p, 1]/dy
        i, j = int(u), int(v)
        u, v = u-i, v-j
        q = charge[p]/(dx*dy)
        rho[i, j] += q*(1-u)*(1-v)
        rho[i+1, j] += q*u*(1-v)
        rho[i, j+1] += q*(1-u)*v
        rho[i+1, j+1] += q*u*v


@njit(cache=True)
def deposit_segment(x0, y0, x1, y1, q, jx, jy, dx, dy, dt):
    """Split a straight orbit at grid lines; integrate the CIC flux exactly.

    In one cell, delta(Sx*Sy) = delta(Sx)*mean(Sy) + mean(Sx)*delta(Sy).
    Thus each segment satisfies the discrete continuity equation, including
    diagonals and cell crossings. dt is the FULL step, not segment duration.
    """
    ax, ay = x0/dx, y0/dy
    bx, by = x1/dx, y1/dy
    du, dv = bx-ax, by-ay
    t = 0.0
    while t < 1.0-1e-13:
        # Select the cell just ahead, including for a negative-going grid hit.
        u, v = ax+du*t, ay+dv*t
        i = int(math.floor(u + (1e-10 if du >= 0 else -1e-10)))
        j = int(math.floor(v + (1e-10 if dv >= 0 else -1e-10)))
        i = min(max(i, 0), jx.shape[0]-1)
        j = min(max(j, 0), jy.shape[1]-1)
        tx = ((i+1-ax)/du if du > 1e-14 else
              (i-ax)/du if du < -1e-14 else 2.0)
        ty = ((j+1-ay)/dv if dv > 1e-14 else
              (j-ay)/dv if dv < -1e-14 else 2.0)
        end = min(1.0, tx, ty)
        if end <= t + 1e-14:
            end = min(1.0, t+1e-10)
        um = ax + du*(t+end)*0.5 - i
        vm = ay + dv*(t+end)*0.5 - j
        fx = q*du*(end-t)/(dy*dt)
        fy = q*dv*(end-t)/(dx*dt)
        jx[i, j] += fx*(1-vm)
        jx[i, j+1] += fx*vm
        jy[i, j] += fy*(1-um)
        jy[i+1, j] += fy*um
        t = end


@njit(cache=True)
def drift_reflect(x, y, vx, vy, q, jx, jy, dx, dy, dt, width, height, radius):
    """Drift along the actual reflected orbit, preserving tangential velocity."""
    left = dt
    sx, sy = 1.0, 1.0
    while left > dt*1e-13:
        tx = ((width-radius-x)/vx if vx > 1e-14 else
              (radius-x)/vx if vx < -1e-14 else math.inf)
        ty = ((height-radius-y)/vy if vy > 1e-14 else
              (radius-y)/vy if vy < -1e-14 else math.inf)
        duration = max(0.0, min(left, tx, ty))
        xx = min(max(x+vx*duration, radius), width-radius)
        yy = min(max(y+vy*duration, radius), height-radius)
        deposit_segment(x, y, xx, yy, q, jx, jy, dx, dy, dt)
        x, y = xx, yy
        left -= duration
        if tx <= duration + 1e-13:
            vx, sx = -vx, -sx
        if ty <= duration + 1e-13:
            vy, sy = -vy, -sy
    return x, y, sx, sy


@njit(cache=True)
def boris(px, py, ex, ey, bz, q_over_m, dt, c, fx=0.0, fy=0.0):
    """Reference relativistic Boris pusher, retained for numerical comparisons.

    The optional external force is acceleration (force/mass), split into
    the same two kicks as the electric field. No arbitrary speed clipping.
    """
    ux = px + 0.5*dt*(q_over_m*ex + fx)
    uy = py + 0.5*dt*(q_over_m*ey + fy)
    gamma = math.sqrt(1.0 + (ux*ux+uy*uy)/(c*c))
    t = 0.5*dt*q_over_m*bz/gamma
    s = 2*t/(1+t*t)
    prime_x, prime_y = ux+uy*t, uy-ux*t
    return (ux+prime_y*s + 0.5*dt*(q_over_m*ex+fx),
            uy-prime_x*s + 0.5*dt*(q_over_m*ey+fy))


@njit(cache=True)
def higuera_cary(px, py, ex, ey, bz, q_over_m, dt, c, fx=0.0, fy=0.0):
    """Higuera–Cary update of proper velocity u=gamma*v in the xy plane.

    Same kick/rotation/kick as Boris, with the momentum-centered rotation
    factor from Eq. (20) of https://arxiv.org/abs/1701.05605.
    Since u_z=0 and B=(0,0,Bz), u_minus dot tau is identically zero.
    External force/mass is held constant over the two kicks, as before.
    """
    kick_x, kick_y = .5*dt*(q_over_m*ex+fx), .5*dt*(q_over_m*ey+fy)
    ux, uy = px+kick_x, py+kick_y
    tau = .5*dt*q_over_m*bz
    sigma = 1.0+(ux*ux+uy*uy)/(c*c)-tau*tau
    root = math.hypot(sigma, 2*tau)
    # Rationalize the positive quadratic root when sigma<0 to avoid losing
    # all significant digits in strong B / large timestep cases.
    gamma_sq = .5*(sigma+root) if sigma >= 0 else 2*tau*tau/(root-sigma)
    t = tau/math.sqrt(gamma_sq)
    s = 2*t/(1+t*t)
    prime_x, prime_y = ux+uy*t, uy-ux*t
    return ux+prime_y*s+kick_x, uy-prime_x*s+kick_y


@njit(cache=True)
def advance_b(ex, ey, bz, centered_b, dx, dy, dt):
    for i in range(bz.shape[0]):
        for j in range(bz.shape[1]):
            old = bz[i, j]
            bz[i, j] -= dt*((ey[i+1, j]-ey[i, j])/dx
                            - (ex[i, j+1]-ex[i, j])/dy)
            centered_b[i, j] = (old+bz[i, j])*0.5


@njit(cache=True)
def advance_e(ex, ey, bz, jx, jy, dx, dy, dt, c):
    for i in range(ex.shape[0]):
        for j in range(1, ex.shape[1]-1):
            ex[i, j] += dt*(c*c*(bz[i, j]-bz[i, j-1])/dy-jx[i, j])
    for i in range(1, ey.shape[0]-1):
        for j in range(ey.shape[1]):
            ey[i, j] -= dt*(c*c*(bz[i, j]-bz[i-1, j])/dx+jy[i, j])
    # Outermost tangential samples are unused ghosts; B's outgoing condition
    # closes the full grid. The visible perimeter is updated like any interior.


@njit(cache=True)
def advance_removal_wave(psi, centered_psi, ex, ey, rho, dx, dy, dt, c):
    """Hyperbolic correction for the external, nonconservative deletion edit.

    psi_t = -c**2 * (div E - rho), E_t gains -grad psi. The Gauss defect
    therefore obeys a wave equation at speed c, rather than remaining as a
    static ghost charge. psi shares B's half-step time and rho's nodal grid.
    Ordinary charge-conserving runs never enable this additional equation.
    """
    centered_psi[:] = psi
    for i in range(1, psi.shape[0]-1):
        for j in range(1, psi.shape[1]-1):
            defect = ((ex[i, j]-ex[i-1, j])/dx
                      + (ey[i, j]-ey[i, j-1])/dy-rho[i, j])
            psi[i, j] -= dt*c*c*defect
            centered_psi[i, j] = .5*(centered_psi[i, j]+psi[i, j])


@njit(cache=True)
def apply_removal_wave(ex, ey, psi, dx, dy, dt):
    """A discrete gradient preserves curl E and pre-existing radiation."""
    for i in range(ex.shape[0]):
        for j in range(ex.shape[1]):
            ex[i, j] -= dt*(psi[i+1, j]-psi[i, j])/dx
    for i in range(ey.shape[0]):
        for j in range(ey.shape[1]):
            ey[i, j] -= dt*(psi[i, j+1]-psi[i, j])/dy


@njit(cache=True)
def push_particles(pos, momentum, charge, mass, fixed, ex, ey, bz, jx, jy,
                   dx, dy, dt, c, width, height, radius, dragged, target,
                   prescribed=None, acceleration=None):
    jx[:] = 0.0
    jy[:] = 0.0
    for p in range(len(charge)):
        if fixed[p]:
            # The charge remains in rho, but its stationary orbit carries no J.
            momentum[p, 0], momentum[p, 1] = 0.0, 0.0
            continue
        x, y = pos[p, 0], pos[p, 1]
        electric_x = sample(ex, x, y, dx, dy, 0.5, 0.0)
        electric_y = sample(ey, x, y, dx, dy, 0.0, 0.5)
        magnetic = sample(bz, x, y, dx, dy, 0.5, 0.5)
        fx, fy = 0.0, 0.0
        if p == dragged:
            # A damped, force-limited spring lets the user inject real momentum.
            fx, fy = drag_force(x, y, momentum[p, 0], momentum[p, 1], c, target)
        if prescribed is not None and prescribed[p]:
            # Explicit external trajectory constraint for controlled tests.
            # Store integer-time velocity, drift at the midpoint, and preserve
            # the existing reflected orbit and exactly conservative current.
            g = math.sqrt(1+np.sum(momentum[p]**2)/(c*c))
            vx = momentum[p, 0]/g+dt*acceleration[p, 0]
            vy = momentum[p, 1]/g+dt*acceleration[p, 1]
            drift_x, drift_y = vx-.5*dt*acceleration[p, 0], vy-.5*dt*acceleration[p, 1]
            g = 1/math.sqrt(1-(vx*vx+vy*vy)/(c*c))
            ux, uy = g*vx, g*vy
        else:
            ux, uy = higuera_cary(momentum[p, 0], momentum[p, 1], electric_x,
                                 electric_y, magnetic, charge[p]/mass[p], dt, c, fx, fy)
            gamma = math.sqrt(1+(ux*ux+uy*uy)/(c*c))
            drift_x, drift_y = ux/gamma, uy/gamma
        x, y, sx, sy = drift_reflect(x, y, drift_x, drift_y, charge[p],
                                    jx, jy, dx, dy, dt, width, height, radius)
        pos[p, 0], pos[p, 1] = x, y
        momentum[p, 0], momentum[p, 1] = sx*ux, sy*uy


@dataclass(frozen=True)
class Config:
    nx: int = 128
    ny: int = 80
    width: float = 16.0
    height: float = 10.0
    c: float = 6.0
    cfl: float = 0.75
    charge: float = 2.5
    mass: float = 1.0
    field_padding: int = 32
    source_filter_passes: int = 2

    def __post_init__(self):
        if self.nx < 16 or self.ny < 16:
            raise ValueError("Grid dimensions must be at least 16.")
        if min(self.width, self.height, self.c, self.mass) <= 0:
            raise ValueError("Dimensions, c, and mass must be positive.")
        if not 0 < self.cfl < 1:
            raise ValueError("CFL safety factor must be between 0 and 1.")
        if not isinstance(self.field_padding, int) or self.field_padding < 4:
            raise ValueError("Field padding must be an integer of at least 4 cells.")
        if (not isinstance(self.source_filter_passes, int)
                or not 0 <= self.source_filter_passes <= 4):
            raise ValueError('Source filtering must be an integer from 0 to 4 passes.')


class Simulation:
    def __init__(self, config=None, particles=6, seed=7):
        self.config = config or Config()
        cfg = self.config
        self.dx, self.dy = cfg.width/cfg.nx, cfg.height/cfg.ny
        self.dt = cfg.cfl/(cfg.c*math.hypot(1/self.dx, 1/self.dy))
        self.radius = 1.6*max(self.dx, self.dy)
        # Public field arrays are views of the visible region. Particles and
        # renderers keep their original coordinates, shapes, and algorithms.
        p = cfg.field_padding
        nx, ny = cfg.nx+2*p, cfg.ny+2*p
        for name, shape in (('ex', (nx, ny+1)), ('ey', (nx+1, ny)),
                            ('bz', (nx, ny)), ('centered_b', (nx, ny)),
                            ('jx', (nx, ny+1)), ('jy', (nx+1, ny))):
            full = np.zeros(shape)
            setattr(self, '_'+name, full)
            setattr(self, name, full[p:shape[0]-p, p:shape[1]-p])
        self._rho = np.zeros((nx+1, ny+1))
        self.rho = self._rho[p:-p, p:-p]
        self._removal_psi = np.zeros_like(self._rho)
        self._centered_psi = np.zeros_like(self._rho)
        self.removal_active = False
        self.pos = np.empty((0, 2))
        self.momentum = np.empty((0, 2))
        self.charge = np.empty(0)
        self.mass = np.empty(0)
        self.time = 0.0
        self.steps = 0
        self.dragged = -1
        self.target = np.zeros(2)
        self._poisson_kernel = free_space_kernel(nx, ny, self.dx, self.dy)
        self.reset(particles, seed)

    def reset(self, particles=6, seed=7):
        if particles < 0:
            raise ValueError("Particle count must be nonnegative.")
        rng = np.random.default_rng(seed)
        cfg = self.config
        if particles == 6:
            self.pos = np.array([[5.1, 6.6], [10.7, 6.4], [7.8, 3.4],
                                 [4.5, 3.1], [11.9, 3.0], [8.1, 7.6]])
            self.pos *= np.array([cfg.width/16, cfg.height/10])
        else:
            self.pos = rng.uniform([2*self.radius, 2*self.radius],
                                   [cfg.width-2*self.radius, cfg.height-2*self.radius],
                                   size=(particles, 2))
        self.charge = np.where(np.arange(particles) % 2 == 0, cfg.charge, -cfg.charge)
        self.mass = np.full(particles, cfg.mass)
        self.fixed = np.zeros(particles, dtype=np.bool_)
        self.prescribed = np.zeros(particles, dtype=np.bool_)
        self.prescribed_acceleration = np.zeros((particles, 2))
        self.species = np.full(particles, SPECIES['charge'], dtype=np.int8)
        v = rng.normal(0, 0.22, (particles, 2))
        v *= np.minimum(1.0, (0.4*cfg.c)/np.maximum(np.linalg.norm(v, axis=1), 1e-12))[:, None]
        self.momentum = v / np.sqrt(1-np.sum(v*v, axis=1)/cfg.c**2)[:, None]
        for a in (self._ex, self._ey, self._bz, self._centered_b, self._jx, self._jy):
            a[:] = 0
        self._removal_psi[:] = 0
        self._centered_psi[:] = 0
        self.removal_active = False
        self.time, self.steps, self.dragged = 0.0, 0, -1
        self.update_rho()
        self.correct_gauss()
        self._stagger_initial_momentum(range(particles))

    def reset_to_single_positive(self):
        """Clear particles, fields, and time; leave one free positive at center."""
        self.reset(0)
        self.add_charge(self.config.width/2, self.config.height/2)

    def toggle_fixed(self, index):
        """Shortcut for constraint mode with v=a=0, or release that constraint."""
        if not 0 <= index < len(self.charge):
            raise IndexError('Charge index out of range.')
        if self.fixed[index]:
            self.release_motion(index)
        else:
            self.set_motion(index, velocity=np.zeros(2), acceleration=np.zeros(2))
        return bool(self.fixed[index])

    def _stagger_initial_momentum(self, indices):
        """Convert specified initial u(t) to the pusher's u(t-dt/2)."""
        electric_x, electric_y, magnetic = self.force_fields()
        for i in indices:
            x, y = self.pos[i]
            ex = sample(electric_x, x, y, self.dx, self.dy, .5, 0.)
            ey = sample(electric_y, x, y, self.dx, self.dy, 0., .5)
            bz = sample(magnetic, x, y, self.dx, self.dy, .5, .5)
            self.momentum[i] = higuera_cary(*self.momentum[i], ex, ey, bz,
                                           self.charge[i]/self.mass[i], -self.dt/2,
                                           self.config.c)

    def magnetic_now(self, full=False):
        """B at E's integer time, without altering the staggered solver state."""
        ex, ey, bz = (self._ex, self._ey, self._bz) if full else (self.ex, self.ey, self.bz)
        curl = np.diff(ey, axis=0)/self.dx-np.diff(ex, axis=1)/self.dy
        return bz-.5*self.dt*curl

    def force_fields(self, magnetic=None):
        """Gather with the adjoint of the source filter, including hidden cells.

        These are temporary copies, not changes to the evolved Maxwell fields.
        """
        if magnetic is None:
            magnetic = self.magnetic_now(full=True)
        p, passes = self.config.field_padding, self.config.source_filter_passes
        return tuple(binomial_filter(a, passes)[p:-p, p:-p]
                     for a in (self._ex, self._ey, magnetic))

    def particle_state(self, index):
        """Integer-time v and coordinate acceleration dv/dt, not du/dt.

        Wall reflections and user velocity edits are impulses, not finite
        accelerations. Fixed particles have zero actual v and acceleration.
        """
        if not 0 <= index < len(self.charge):
            raise IndexError('Charge index out of range.')
        if self.fixed[index]:
            return np.zeros(2), np.zeros(2)
        c = self.config.c
        if self.prescribed[index]:
            return self.velocity[index].copy(), self.prescribed_acceleration[index].copy()
        x, y = self.pos[index]
        electric_x, electric_y, magnetic = self.force_fields()
        ex = sample(electric_x, x, y, self.dx, self.dy, .5, 0.)
        ey = sample(electric_y, x, y, self.dx, self.dy, 0., .5)
        bz = sample(magnetic, x, y, self.dx, self.dy, .5, .5)
        fx, fy = (drag_force(x, y, *self.momentum[index], c, self.target)
                  if index == self.dragged else (0., 0.))
        u = np.array(higuera_cary(*self.momentum[index], ex, ey, bz,
                                  self.charge[index]/self.mass[index], self.dt/2, c, fx, fy))
        gamma = math.sqrt(1+np.dot(u, u)/c**2)
        v = u/gamma
        external = (np.array(drag_force(x, y, *u, c, self.target))
                    if index == self.dragged else np.zeros(2))
        force_per_mass = self.charge[index]/self.mass[index]*np.array(
            [ex+v[1]*bz, ey-v[0]*bz])+external
        a = (force_per_mass-v*np.dot(v, force_per_mass)/c**2)/gamma
        return v, a

    def set_motion(self, index, velocity=None, acceleration=None):
        """External edit at current time; E, B, position, and time are preserved.

        Editing acceleration enables a prescribed TOTAL lab acceleration.
        None leaves the existing mode unchanged; use release_motion for forces.
        All validation precedes mutation, including the next prescribed step.
        """
        current_v, _ = self.particle_state(index)
        v = current_v if velocity is None else np.asarray(velocity, dtype=float)
        a = (self.prescribed_acceleration[index].copy() if acceleration is None
             else np.asarray(acceleration, dtype=float))
        if v.shape != (2,) or a.shape != (2,) or not np.all(np.isfinite(v)) or not np.all(np.isfinite(a)):
            raise ValueError('Velocity and acceleration must be finite x/y values.')
        controlled = self.prescribed[index] or acceleration is not None
        c = self.config.c
        candidates = (v, v+self.dt*a) if controlled else (v,)
        if any(np.linalg.norm(value) >= c for value in candidates):
            raise ValueError('Speed must stay below c, including the next step. Reduce v or a.')
        self.dragged = -1
        self.prescribed[index] = controlled
        self.prescribed_acceleration[index] = a
        self.fixed[index] = controlled and np.all(v == 0) and np.all(a == 0)
        self.momentum[index] = v/math.sqrt(1-np.dot(v, v)/c**2)
        if not controlled:
            self._stagger_initial_momentum([index])

    def release_motion(self, index):
        """Remove the acceleration constraint while preserving current v."""
        v, _ = self.particle_state(index)
        self.prescribed[index] = False
        self.prescribed_acceleration[index] = 0
        self.fixed[index] = False
        self.set_motion(index, velocity=v)

    def set_constraint(self, index, enabled):
        """Toggle external prescribed motion; enabling starts with a=0."""
        if enabled:
            self.set_motion(index, acceleration=np.zeros(2))
        else:
            self.release_motion(index)

    def update_rho(self):
        self._rho[:] = 0.0  # Clear tails in the hidden cells before redepositing.
        deposit_charge(self.pos, self.charge, self.rho, self.dx, self.dy)
        self._rho[:] = binomial_filter(self._rho, self.config.source_filter_passes)

    def charge_source(self, pos, charge):
        """A filtered contribution on the FULL grid, including source tails."""
        source = np.zeros_like(self._rho)
        p = self.config.field_padding
        deposit_charge(pos, charge, source[p:-p, p:-p], self.dx, self.dy)
        return binomial_filter(source, self.config.source_filter_passes)

    def divergence(self):
        return ((self.ex[1:, 1:-1]-self.ex[:-1, 1:-1])/self.dx
                + (self.ey[1:-1, 1:]-self.ey[1:-1, :-1])/self.dy)

    def correct_gauss(self, added_rho=None):
        """Initialize Gauss, or add ONLY a new charge's electrostatic field.

        An aperiodic FFT convolution with the infinite-lattice Green function
        replaces the grounded-box solve, including for nonzero total charge.
        Adding a charge must not erase a previous deletion's travelling defect.
        Deletion never calls this instantaneous Poisson correction.
        """
        residual = (self._rho if added_rho is None else added_rho).copy()
        if added_rho is None:
            residual[1:-1, 1:-1] -= (
                np.diff(self._ex[:, 1:-1], axis=0)/self.dx
                + np.diff(self._ey[1:-1, :], axis=1)/self.dy)
        phi = free_space_potential(residual, self._poisson_kernel)
        self._ex -= np.diff(phi, axis=0)/self.dx
        self._ey -= np.diff(phi, axis=1)/self.dy

    def add_charge(self, x, y, sign=1.0, species='charge'):
        cfg = self.config
        if species not in SPECIES:
            raise ValueError(f'Unknown particle species: {species}')
        mass = cfg.mass
        if species == 'proton':
            sign, mass = 1.0, PROTON_ELECTRON_MASS_RATIO
        elif species == 'electron':
            sign, mass = -1.0, 1.0
        x = np.clip(x, self.radius, cfg.width-self.radius)
        y = np.clip(y, self.radius, cfg.height-self.radius)
        self.pos = np.vstack((self.pos, [x, y]))
        self.momentum = np.vstack((self.momentum, [0.0, 0.0]))
        self.charge = np.append(self.charge, cfg.charge*sign)
        self.mass = np.append(self.mass, mass)
        self.fixed = np.append(self.fixed, False)
        self.prescribed = np.append(self.prescribed, False)
        self.prescribed_acceleration = np.vstack((self.prescribed_acceleration, [0., 0.]))
        self.species = np.append(self.species, np.int8(SPECIES[species]))
        self.update_rho()
        added_rho = self.charge_source(self.pos[-1:], self.charge[-1:])
        self.correct_gauss(added_rho)
        self._stagger_initial_momentum([len(self.charge)-1])
        return len(self.charge)-1

    def remove_charge(self, index):
        """Remove the particle immediately, leaving E and B exactly unchanged.

        Disappearance violates local charge continuity. Enable an explicit
        hyperbolic removal model so its longitudinal field adjustment travels
        outward, while existing Maxwell waves keep evolving. This is an edit,
        not a physical charge-annihilation process. Fixed particles work too.
        """
        if 0 <= index < len(self.charge):
            removed_rho = self.charge_source(self.pos[index:index+1], self.charge[index:index+1])
            # Change the half-step representation without changing psi(t_now).
            # This gives the first E update its correct half-sized initial kick.
            self._removal_psi += .5*self.dt*self.config.c**2*removed_rho
            self.removal_active = True
            self.pos = np.delete(self.pos, index, axis=0)
            self.momentum = np.delete(self.momentum, index, axis=0)
            self.charge = np.delete(self.charge, index)
            self.mass = np.delete(self.mass, index)
            self.fixed = np.delete(self.fixed, index)
            self.prescribed = np.delete(self.prescribed, index)
            self.prescribed_acceleration = np.delete(self.prescribed_acceleration, index, axis=0)
            self.species = np.delete(self.species, index)
            self.dragged = -1
            self.update_rho()

    def step(self, count=1):
        cfg = self.config
        for _ in range(count):
            active = self.prescribed & ~self.fixed
            if np.any(active):
                end_v = self.velocity[active]+self.dt*self.prescribed_acceleration[active]
                if np.any(np.linalg.norm(end_v, axis=1) >= cfg.c):
                    raise AccelerationLimitError('Prescribed acceleration would reach c. Reduce v or a, or use Lorentz motion.')
            advance_b(self._ex, self._ey, self._bz, self._centered_b, self.dx, self.dy, self.dt)
            apply_mur_boundary(self._bz, self._centered_b, self.dx, self.dy, self.dt, cfg.c)
            if self.removal_active:
                self.update_rho()  # rho and E at the same integer time, before drift.
                advance_removal_wave(self._removal_psi, self._centered_psi,
                                     self._ex, self._ey, self._rho,
                                     self.dx, self.dy, self.dt, cfg.c)
                apply_mur_boundary(self._removal_psi, self._centered_psi,
                                   self.dx, self.dy, self.dt, cfg.c)
            self._jx[:] = 0.0
            self._jy[:] = 0.0
            electric_x, electric_y, magnetic = self.force_fields(self._centered_b)
            push_particles(self.pos, self.momentum, self.charge, self.mass, self.fixed,
                           electric_x, electric_y, magnetic, self.jx, self.jy,
                           self.dx, self.dy, self.dt, cfg.c, cfg.width, cfg.height,
                           self.radius, self.dragged, self.target,
                           self.prescribed, self.prescribed_acceleration)
            self._jx[:] = binomial_filter(self._jx, cfg.source_filter_passes)
            self._jy[:] = binomial_filter(self._jy, cfg.source_filter_passes)
            advance_e(self._ex, self._ey, self._bz, self._jx, self._jy,
                      self.dx, self.dy, self.dt, cfg.c)
            if self.removal_active:
                apply_removal_wave(self._ex, self._ey, self._removal_psi,
                                   self.dx, self.dy, self.dt)
            self.time += self.dt
            self.steps += 1

    def diagnostics(self):
        self.update_rho()
        gauss = float(np.max(np.abs(self.divergence()-self.rho[1:-1, 1:-1])))
        cfg = self.config
        gamma = np.sqrt(1+np.sum(self.momentum**2, axis=1)/cfg.c**2)
        speed = np.linalg.norm(self.momentum/gamma[:, None], axis=1)
        kinetic = float(np.sum(self.mass*cfg.c**2*(gamma-1)))
        field = 0.5*self.dx*self.dy*float(np.sum(self.ex**2)+np.sum(self.ey**2)
                                         + cfg.c**2*np.sum(self.bz**2))
        return {"time": self.time, "particles": len(self.charge), "gauss_linf": gauss,
                "kinetic_energy": kinetic, "field_energy": field,
                "max_speed_over_c": float(np.max(speed, initial=0)/cfg.c)}

    @property
    def velocity(self):
        return self.momentum/np.sqrt(1+np.sum(self.momentum**2, axis=1)
                                    /self.config.c**2)[:, None]
