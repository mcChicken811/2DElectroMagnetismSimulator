# Maxwell · Field Lab

An interactive Python Maxwell–Lorentz simulator for **N moving charges in a
reflecting 2D box**, viewed within an approximately open electromagnetic space.
The electric and magnetic fields evolve across the visible boundary;
electric field lines trace the current field. Built with **Numba, NumPy, and
pyglet's GPU-batched OpenGL rendering**, using native ARM64 dependencies on macOS.

## Run

From this project directory, using your existing environment:

```sh
.venv/bin/python main.py
```

Dependencies are already installed in `.venv`. To reproduce the tested versions:

```sh
mkdir -p .cache/tmp .cache/pip
TMPDIR="$PWD/.cache/tmp" PIP_CACHE_DIR="$PWD/.cache/pip" .venv/bin/python -m pip install -r requirements-lock.txt
```

Python 3.13 is tested. The first launch compiles numerical kernels; subsequent
launches use `.cache/numba`. No global Python packages are needed. Source,
dependencies, application caches, temporary files, and generated screenshots
belong to this project. The Cocoa setup installs **volatile, in-memory defaults**
before pyglet initializes. On macOS, the launcher also re-executes itself under
`/usr/bin/sandbox-exec` with **all filesystem writes outside this project denied**,
because AppKit can attempt asynchronous saved-state writes even with restoration
disabled. This process-local policy does not change shell profiles or system
settings. GUI launch fails if that macOS facility is unavailable.

Examples:

```sh
.venv/bin/python main.py --charges 100
.venv/bin/python main.py --charges 0 --theme light
.venv/bin/python main.py --grid 192x120 --charges 30
.venv/bin/python main.py --grid 96x60 --charges 200
.venv/bin/python main.py --field-padding 64
```

`--charges` accepts any nonnegative count subject to available memory and compute.
The default six-charge scene is fixed; other counts use a reproducible random
scene. Change `--seed` to vary it. New charges can be added throughout a run.
`--field-padding` sets the hidden field cells on **each** side of the canvas
(default 32, minimum 4). Increasing it moves the approximate outgoing boundary
farther away, at additional computational cost; it does not change particle walls.

## Controls

| Input | Action |
| --- | --- |
| Left-drag a free charge | Apply a damped spring force toward the cursor; release to coast |
| Double-click a charge | Fix it in space / release it from rest |
| Space, with cursor over the box | Add the selected charge or particle species at that location |
| Shift + Space | Add a negative charge |
| `1` / `2`, or sign buttons | Select positive / negative injection |
| `3` / `4`, or species buttons | Select proton / electron injection |
| Right-click a charge | Remove it immediately; preserve its fields and let the adjustment travel outward |
| Delete / Backspace | The same field-preserving removal for the charge under the cursor |
| `P`, or Pause button | Pause / resume |
| `N` while paused | Advance one physics timestep |
| `R`, or Reset button | Clear the scene and leave one free positive charge at rest in the center |
| `E` | Toggle electric streamlines |
| `F` | Toggle smooth electric-field magnitude coloring, \(|E|\) |
| `B` | Toggle smooth magnetic-field magnitude coloring, \(|B|\) |
| `U` | Toggle smooth electromagnetic energy-density coloring |
| `M` | Toggle signed magnetic-field coloring, \(B_z\) |
| `V` | Toggle velocity arrows |
| `D` | Cycle low / balanced / high line detail |
| `T`, or top-right switch | Switch dark / white theme |
| Time-scale buttons | 0.5×, 1×, or 2× simulation speed |
| Escape | Close |

Space is debounced: holding it does not flood the scene with particles. Dragging
while paused records the force target; resume or step with `N` to move the charge.
Positions at the wall are clamped to the finite charge radius. Charges may pass
through one another: there is no extra hard-sphere collision force.

Fixed charges have an outline ring and ignore Lorentz forces and dragging. They
still contribute charge density and electric fields, but carry no current.
Double-click again to release from rest. Pinning removes the old velocity; it
does not save and restore it. Hovering shows the particle's species, mass, and
fixed/free state. Proton and electron markers also carry a small `p` or `e`.
Reset clears all old fields, motion, pins, and simulation time, and starts the
single positive charge's electrostatic field. It preserves the pause and view
settings. The six-charge startup scene and `--charges` option remain available.

Right-click works directly on stationary and fixed charges, as well as moving
ones. The charge marker disappears immediately, while E and B remain unchanged
at the instant of deletion. Existing waves continue and the removed charge's
electric-field adjustment spreads outward at the simulation's light speed.
While paused, the fields stay frozen too: press `N` to single-step or `P` to
resume. Adding another charge preserves an ongoing removal adjustment. Reset
clears both the old fields and all removal adjustments.

Electric arrows point along **E**. Lines may continue beyond the visible canvas;
its edge is not a conductor. Coral markers are positive; blue markers are negative.
The optional signed **Bz** overlay uses violet for out of the screen and amber
for into the screen. B is perpendicular to this plane, so drawing in-plane
magnetic streamlines would be misleading. Electric
lines are instantaneous integral curves, not particle paths or wave trajectories.
Line count is chosen for readability and is not a calibrated flux measurement.

Choose **one canvas color mode** independently of lines and velocity arrows:

- **|E|:** electric-field magnitude, from weak to strong.
- **|B|:** magnetic-field magnitude, independent of sign.
- **EM energy:** electromagnetic energy per area per unit z,
  `u = (|E|² + c²|B|²)/2` in the normalized units. Particle kinetic and rest-mass
  energy are not included.
- **Signed Bz:** signed out-of-plane magnetic field.

Colors use a continuous logarithmic ramp with bilinear GPU interpolation. The
legend states the current range and **log / auto**: the upper bound is the
current maximum, so compare numeric legends when comparing different frames or
modes. Normalized values are mapped by `log(1+50*value/max)/log(51)`; signed B uses
the magnitude for intensity and sign for hue. Small range floors avoid amplifying
roundoff in zero fields (`1e-6` for E, `1e-7` for B, `1e-10` for energy).
The display averages staggered electric components to cell centers and advances
B by a temporary half-step to the same time as E. It never modifies the solver's
arrays. Toggling lines off with `E` provides an unobstructed color-only view.

## Physical model and accuracy

The underlying solver is a **2D Maxwell particle-in-cell model** with fields `(Ex, Ey, Bz)` and
particle motion `(x, y)`. Physically it represents a system invariant along z:
the charges correspond to finite-width line charges per unit z, **not isolated
3D point charges restricted to a plane**. The corresponding electrostatic force
has the 2D distance law. All quantities use normalized units:

```text
epsilon0 = 1,  c = 6,  mu0 = 1/c²
default q = ±2.5,  m = 1,  box = 16 × 10
electron mass = 1,  proton mass = 1836.152673426

dEx/dt =  c² dBz/dy − Jx
dEy/dt = −c² dBz/dx − Jy
dBz/dt = −(dEy/dx − dEx/dy)
div E  = rho
div B  = 0  (identically for z-invariant Bz)

du/dt = (q/m)(E + v × B) + Fmanual/m
v = u / sqrt(1 + |u|²/c²)
```

**Particle masses:** the mass unit is one electron mass
(`9.1093837139e-31 kg`), and the proton preset uses the
[CODATA proton/electron mass ratio](https://physics.nist.gov/cgi-bin/cuu/Value?mpsme).
The electron-mass reference is [NIST CODATA](https://physics.nist.gov/cgi-bin/cuu/Value?me).
Protons and electrons have equal and opposite charge weights, `+2.5` and `-2.5`;
one elementary charge is thus 2.5 simulation charge units. Generic charges keep
the previous default mass of 1. These are classical, softened 2D particle presets
with the physical mass ratio; they do not turn this line-charge model into an
atomic or quantum simulation. Spatial and temporal coordinates remain normalized.

Using electron-mass units means the lightest new particle has the same mass as
the original particles. There is no need to shrink the existing timestep or
rescale the field solver. At the same electric field, a free proton's proper
velocity changes about 1836 times more slowly than an electron's (opposite sign).
Dragging retains the existing acceleration-based spring, so its applied force
scales with mass and both species remain easy to manipulate.

- **Fields:** second-order Yee finite differences with a fixed leapfrog timestep
  `dt = 0.75 / (c sqrt(dx⁻² + dy⁻²))`. The CFL limit is respected independently
  of rendering rate or time scale. Fields use float64 throughout.
- **Particles:** relativistic Boris kicks and magnetic rotation, with
  time-centered magnetic gathering and staggered momentum. A magnetic-only
  force preserves momentum magnitude to roundoff, and finite momentum implies
  speed below c without an artificial velocity clamp.
- **Charge and current:** bilinear cloud-in-cell charge shapes. Every trajectory
  is split at crossed grid lines and at wall collisions. Cell-local orbit fluxes
  satisfy the discrete continuity equation to roundoff. This preserves Gauss's
  law during ordinary timesteps **without periodic divergence cleaning**.
  User-triggered deletion enables the separate removal model described below.
- **Initialization and addition:** a free-space Poisson solve initializes E using
  the infinite-lattice Green function for the same discrete Yee Laplacian.
  Its kernel is computed once by quadrature; a zero-padded FFT convolution then
  applies curl-free corrections, preserving existing transverse waves and B.
  Adding a charge adds only its own electrostatic field, so an ongoing deletion
  adjustment is not accidentally erased by a global constraint correction.
  There is no grounded perimeter, periodic charge image, or neutralizing background.
  Nonzero net charge is allowed: the 2D potential has logarithmic behavior at
  infinity, while its electric field decays. Only potential differences are used.
  Addition is an external intervention, not a charge-conserving physical injection
  process; it can instantaneously change energy and the longitudinal field.
- **Deletion:** E and B are retained exactly when the particle is removed; no
  Poisson solve or field reset runs. Since disappearance violates charge
  continuity, this edit explicitly enables a hyperbolic constraint correction:
  `dpsi/dt = -c²(div E - rho)` and an additional `-grad psi` in `dE/dt`.
  This carries the resulting Gauss defect outward at speed c instead of leaving
  a permanent static ghost field. The nodal scalar psi uses the existing
  leapfrog timestep and outgoing Mur termination on the hidden grid. Its
  half-step initialization preserves its time-centered value at each deletion.
  The discrete gradient leaves curl E, and therefore pre-existing transverse
  radiation, intact. The particle pusher and orbit-current algorithms are unchanged.
  This additional equation stays enabled until reset; it does no work before
  the first deletion.
- **Open fields:** the visible canvas is an interior crop of a larger Yee grid,
  with 32 extra cells on each side by default. Its edge imposes no EM condition.
  First-order Mur outgoing conditions terminate Bz at the hidden outer edge.
  This leaves the electric curl update and discrete Gauss identity intact, and
  does not damp stationary Coulomb fields. There is no conducting wall or
  periodic wraparound at the canvas edge. The outermost tangential E samples
  are unused ghosts; their trial B update is replaced by the outgoing condition.
- **Particle walls:** centers still reflect at the original finite-radius inset.
  Only momentum normal to the wall changes sign; both components reverse at a
  corner. These walls are a mechanical constraint, not an EM material.
- **Dragging:** a spring with acceleration `90(target − x) − 16v`, limited to
  magnitude 180, adds external force through the Boris kicks. It injects/removes
  physical momentum and energy; it does not teleport the particle.
- **Fixed particles:** a single early-out in the particle loop keeps position
  constant and momentum zero. Charge deposition still includes them. Pin/release
  is an external constraint that can remove kinetic energy; it leaves Gauss's
  law intact. The Yee curls, orbit-current deposition, Boris rotation, and RK4
  streamline algorithms are unchanged by the pinning/species/color additions.
- **Field lines:** bilinear gathering from the staggered E grid, followed by
  adaptive RK4 with step-doubling error estimates. The local tolerance is
  `0.004 min(dx,dy)`; steps range from `0.06` to `1.2` cell widths. Traces stop
  at the viewing edge, charge cores, unresolved nulls, closed loops, or a finite
  work budget.

The deletion behavior is a **numerical editing model**, not physical annihilation
of an isolated charge. Maxwell's equations require charge continuity; deleting
a source while retaining its field necessarily creates a temporary mismatch in
`div E = rho`. The display labels the raw mismatch **Gauss (removal)** after a
deletion rather than presenting it as roundoff-level constraint preservation.
The removed static field's adjustment is longitudinal; it is not newly generated
transverse electromagnetic radiation. In particular, deleting an otherwise
static isolated charge does not create a magnetic pulse. Finite-grid dispersion
can leave weak ringing as the adjustment exits. This type of hyperbolic
divergence correction is also described in the
[WarpX documentation](https://warpx.readthedocs.io/en/23.08/usage/parameters.html#additional-parameters).

The open-domain treatment approximates infinite space with no imposed incoming
radiation. A finite grid cannot be made infinite simply by deleting boundary
assignments. The Mur termination can reflect oblique, poorly resolved, or
low-frequency fields; accuracy near moving charges and over long runs should
be checked by increasing `--field-padding` and refining the mesh. This small
boundary treatment preserves the existing Yee/Boris/orbit algorithms and avoids
introducing a PML subsystem. For the tested smooth circular pulse, the visible
field-difference energy against a much larger reference domain was below 0.02%
of the initial pulse energy at the sampled times through t=6. That figure is a
specific regression result, not a universal reflection bound. Energy diagnostics
cover only the visible region, so energy can flow out of the displayed canvas.

The discrete free-space formulation is described by
[Hollos & Hollos](https://arxiv.org/abs/cond-mat/0509002). This implementation
integrates one Fourier coordinate analytically and the remaining coordinate
numerically, including for unequal cell widths. The outgoing update follows the
first-order condition in
[Schneider's FDTD notes, Chapter 6](https://eecs.wsu.edu/~schneidj/ufdtd/chap6.pdf).

The finite grid regularizes the singular charge core over roughly a cell. PIC
interpolation has finite resolution, self-force artifacts, numerical dispersion,
and energy error; this is an interactive numerical laboratory, not a validated
precision plasma-research package. Close encounters and dense clouds need a
finer grid and convergence checks. The displayed Gauss residual measures
constraint preservation, not the total physical error. No Coulomb pairwise force
is added on top of the grid field. Binary collisions, quantum physics, and an
explicit radiation-reaction force are not modeled. Electromagnetic waves do
propagate and feed back on particles through the resolved fields.

The standard PIC/Yee/Boris method is described in the
[Smilei algorithms documentation](https://smileipic.github.io/Smilei/Understand/algorithms.html).
The current deposition here uses the explicit 2D CIC identity documented in
`deposit_segment`, rather than importing Smilei or claiming its higher-order
Esirkepov implementation. GPU batching follows the
[pyglet rendering documentation](https://docs.pyglet.org/en/latest/programming_guide/rendering.html).
OpenGL is hardware accelerated on the supported Mac but is not a native Metal backend.

## Performance and verification

The default visible grid is **128 × 80** (the field grid including padding is
**192 × 144**), with a **60 Hz render target** and electric-line
updates at **15 Hz**. Physics uses as many fixed steps as needed for elapsed
simulation time. A bounded catch-up budget slows simulated time under overload
instead of increasing the numerical timestep. The window reports FPS, simulation
time, maximum speed/c, and the interior Gauss residual. If overloaded it reports
`COMPUTE LIMITED`. Window resizing changes presentation, not the mesh or domain.

Work scales approximately as `O(grid cells + N)` per physics step; no `O(N²)`
pair loop is used. Line seeds and vertices have bounded budgets at each detail
level. The GPU draws all streamline segments in one draw call. Physics and line
tracing use single-threaded compiled loops to avoid thread-launch overhead on
this grid. Reduce line detail first for battery use; choose `--grid 96x60` for
lower numerical cost, or `192x120` for finer charge cores.

Benchmark the numerical kernels separately from the GUI:

```sh
.venv/bin/python main.py --benchmark --charges 100 --steps 3000
```

Open-field measurements on the available ARM64 Mac / Python 3.13.1, after JIT warmup,
with the default 32-cell padding:

| Charges | Grid | Physics step | Line refresh (median) |
| --- | --- | --- | --- |
| 6 | 128 × 80 | 0.058 ms | 2.67 ms |
| 100 | 128 × 80 | 0.065 ms | 1.15 ms |

These are CPU kernel timings, not a guaranteed M2 Air frame rate. Wall-clock
rendering depends on window resolution, thermals, background load, and driver.
The 100-charge case kept the Gauss residual below `2e-11` after 3,000 steps.
Line timings depend on the evolved scene and the number of traced segments.
With the optional removal evolution active, a separate 100-charge run measured
about **0.128 ms per physics step**. Its kernels are compiled during GUI startup
to avoid a first-deletion compilation pause.

For a full-window measurement (two-second warmup, then automatic exit):

```sh
.venv/bin/python main.py --charges 100 --profile-seconds 8
```

The available Mac measured about **62 rendered FPS** in a six-second open-field
run with 100 charges at a 2560 × 1578 Retina framebuffer and balanced line detail.
Presentation uses vsync;
the Cocoa event loop polls at 120 Hz to avoid alternate-tick timer throttling.

Run numerical checks and the real-window integration smoke test:

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python main.py --smoke-test
```

The 27-test numerical suite covers discrete Poisson, continuity for random orbits and
cell-boundary endpoints, wall and corner reflection, Boris magnetic invariance,
electric acceleration, the Yee curls against an analytic mode with test-only
fixed endpoints, 1,200-step
constraint preservation, empty scenes, dragging, attraction, and straight
streamlines. It also checks reset-to-one-charge, fixed sources with zero current,
release, species mass response and metadata, scalar magnitude/energy formulas,
magnetic time centering, and color-map zero/sign behavior.
Open-space checks cover stationary edge charges, translation invariance (no
conductor-image force), rectangular cells, a non-neutral free-space field, and
pulse escape compared with a much larger domain after reflected waves could return.
Deletion checks verify exact instantaneous field preservation for both particle
signs and fixed/free states, local outward evolution and energy escape, continued
magnetic-wave propagation, the modified Gauss identity, repeated deletion,
addition during a removal wave, invalid-index handling, and reset cleanup.
The GUI test exercises add/delete, negative injection, key-repeat
suppression, force dragging, double-click pin/release, proton/electron injection,
single-charge reset, all scalar layers, themes, and resizing, then
writes `artifacts/maxwell-dark.png`, `artifacts/maxwell-light.png`, and a compact
window preview `artifacts/maxwell-compact.png`. Additional previews show the reset
scene and electric, magnetic, energy, and signed B canvas modes. The test then exits.
It also exercises right-click deletion of a fixed charge while paused and running,
Delete/Backspace, empty-space clicks, single-step propagation, and saves
`artifacts/maxwell-deletion-preserved.png` and `artifacts/maxwell-deletion-wave.png`.

## Source map

- `main.py`: launch options, benchmark, and GUI smoke test.
- `maxwell/physics.py`: field solver, orbit deposition, pusher, and diagnostics.
- `maxwell/open_boundary.py`: infinite-lattice Poisson kernel and outgoing grid edge.
- `maxwell/fieldlines.py`: adaptive tracing and seed budgets.
- `maxwell/heatmap.py`: cell-centered scalar diagnostics and continuous color ramps.
- `maxwell/app.py`: rendering, themes, controls, and scheduling.
- `maxwell/macos.py`: volatile Cocoa preferences and restoration opt-out.
- `tests/test_physics.py`: numerical regression tests.
