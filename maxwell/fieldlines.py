"""Adaptive RK4 streamlines of the instantaneous, interpolated electric field."""
import math
import numpy as np
from numba import njit
from .physics import sample


@njit(cache=True)
def direction(ex, ey, x, y, dx, dy, sign):
    a = sample(ex, x, y, dx, dy, 0.5, 0.0)
    b = sample(ey, x, y, dx, dy, 0.0, 0.5)
    strength = math.hypot(a, b)
    factor = sign/max(strength, 1e-20)
    return a*factor, b*factor, strength


@njit(cache=True)
def rk4(ex, ey, x, y, dx, dy, sign, h):
    a, b, _ = direction(ex, ey, x, y, dx, dy, sign)
    c, d, _ = direction(ex, ey, x+a*h/2, y+b*h/2, dx, dy, sign)
    e, f, _ = direction(ex, ey, x+c*h/2, y+d*h/2, dx, dy, sign)
    g, k, _ = direction(ex, ey, x+e*h, y+f*h, dx, dy, sign)
    return x+h*(a+2*c+2*e+g)/6, y+h*(b+2*d+2*f+k)/6


@njit(cache=True)
def trace(ex, ey, seeds, core, dx, dy, width, height, max_segments):
    """Step doubling estimates local error; accepted paths use two half steps.

    A step below the minimum with excessive error ends at a null/separatrix
    instead of drawing an inaccurate bridge. Arrowheads always point with E.
    """
    vertices = np.empty((max_segments*2, 3), dtype=np.float32)
    used = 0
    hmax = 1.2*min(dx, dy)
    hmin = 0.06*min(dx, dy)
    tolerance = 0.004*min(dx, dy)
    for seed in seeds:
        x, y, sign = seed
        start_x, start_y = x, y
        h, length, arrow_at = hmax, 0.0, 0.4
        for step in range(750):
            if used+6 >= len(vertices):
                return vertices[:used]
            ux, uy, strength = direction(ex, ey, x, y, dx, dy, sign)
            if strength < 1e-8:
                break
            accepted = False
            for _ in range(8):
                ax, ay = rk4(ex, ey, x, y, dx, dy, sign, h)
                mx, my = rk4(ex, ey, x, y, dx, dy, sign, h/2)
                bx, by = rk4(ex, ey, mx, my, dx, dy, sign, h/2)
                error = math.hypot(ax-bx, ay-by)/15.0
                if error <= tolerance:
                    accepted = True
                    break
                if h <= hmin:
                    break
                h = max(hmin, h*0.5)
            if not accepted:
                break
            outside = bx <= 0 or bx >= width or by <= 0 or by >= height
            # Intersect the final segment with the first wall it reaches.
            fraction = 1.0
            if bx < 0: fraction = min(fraction, -x/(bx-x))
            if bx > width: fraction = min(fraction, (width-x)/(bx-x))
            if by < 0: fraction = min(fraction, -y/(by-y))
            if by > height: fraction = min(fraction, (height-y)/(by-y))
            bx, by = x+(bx-x)*fraction, y+(by-y)*fraction
            distance = math.hypot(bx-x, by-y)
            if distance < 1e-10:
                break
            vertices[used] = x, y, strength
            vertices[used+1] = bx, by, strength
            used += 2
            length += distance
            if length > arrow_at and not outside:
                arrow_at += 1.1
                vx, vy = (bx-x)/distance*sign, (by-y)/distance*sign
                size = 0.065
                vertices[used] = bx-vx*size-vy*size*0.6, by-vy*size+vx*size*0.6, strength
                vertices[used+1] = bx, by, strength
                vertices[used+2] = bx-vx*size+vy*size*0.6, by-vy*size-vx*size*0.6, strength
                vertices[used+3] = bx, by, strength
                used += 4
            x, y = bx, by
            if outside or length > 2.5*max(width, height):
                break
            if step > 24 and math.hypot(x-start_x, y-start_y) < h*0.7:
                break
            ix = min(int(x/dx+0.5), core.shape[0]-1)
            iy = min(int(y/dy+0.5), core.shape[1]-1)
            if core[ix, iy]:
                break
            if error < tolerance/16:
                h = min(hmax, h*1.5)
    return vertices[:used]


def build_lines(sim, density=1):
    cfg = sim.config
    total_budget = (60, 112, 176)[density]
    seeds = []
    n = len(sim.charge)
    charge_budget = max(0, total_budget-16)
    if n:
        # Allocate a bounded budget across the whole population, even at large N.
        count = min(n, charge_budget)
        indices = np.linspace(0, n-1, count, dtype=int)
        for rank, i in enumerate(indices):
            rays = charge_budget//count + (rank < charge_budget % count)
            x, y = sim.pos[i]
            for k in range(rays):
                angle = 2*math.pi*(k/rays + 0.0382*rank)
                r = max(2.1*min(sim.dx, sim.dy), sim.radius*1.2)
                sx, sy = x+r*math.cos(angle), y+r*math.sin(angle)
                if 0 < sx < cfg.width and 0 < sy < cfg.height:
                    seeds.append((sx, sy, float(np.sign(sim.charge[i]))))
    # Capture wall-connected and radiative field structure without charge seeds.
    for t in np.linspace(0.08, 0.92, 4):
        for x, y, nx, ny in ((t*cfg.width, .02, 0, 1),
                              (t*cfg.width, cfg.height-.02, 0, -1),
                              (.02, t*cfg.height, 1, 0),
                              (cfg.width-.02, t*cfg.height, -1, 0)):
            a, b, _ = direction(sim.ex, sim.ey, x, y, sim.dx, sim.dy, 1)
            seeds.append((x, y, 1 if a*nx+b*ny >= 0 else -1))
    if not n:
        for x in np.linspace(1, cfg.width-1, 6):
            for y in np.linspace(1, cfg.height-1, 4):
                seeds.extend([(x, y, 1), (x, y, -1)])
    core = np.zeros_like(sim.rho, dtype=np.bool_)
    for x, y in sim.pos:
        i, j = int(x/sim.dx+0.5), int(y/sim.dy+0.5)
        core[max(0, i-1):i+2, max(0, j-1):j+2] = True
    return trace(sim.ex, sim.ey, np.array(seeds, dtype=float).reshape(-1, 3),
                 core, sim.dx, sim.dy, cfg.width, cfg.height,
                 (18000, 34000, 52000)[density])
