# Can analytical stream-function contours replace electric-field tracing?

**Yes for an ideal electrostatic reference; not as a general analytical
replacement for the evolving PIC field.** For this project's 2D line charges,
an electrostatic stream function has a closed-form angle-sum expression.
Contours can replace RK4 tracing for that reference. The evolving numerical
field contains additional information that cannot be recovered from the
instantaneous particle positions.

This investigation changes documentation only. The field solver, pusher,
streamline generation, and rendering algorithms are unchanged.

## 1. Which scalar's contours are field lines?

Electric-potential contours `phi = constant` are perpendicular to electrostatic
field lines. The proposed method instead uses contours of a **stream function**
`psi`. Choose the convention

```text
Ex = dpsi/dy,     Ey = -dpsi/dx.
```

Then `E · grad(psi) = 0`, so `psi = constant` follows the electric field.
The construction using conjugate harmonic functions for two-dimensional
electrostatics is described in the
[Feynman Lectures, II–7, section 7–2](https://www.feynmanlectures.caltech.edu/II_07.html#Ch7-S2).

There is an integrability condition. These derivatives imply

```text
div(E) = d²psi/dxdy - d²psi/dydx = 0.
```

Thus a regular stream function of this form exists locally in charge-free
regions. A changing magnetic field does NOT by itself rule it out. Curl-free
E is needed for an electrostatic potential, whereas divergence-free E is
needed for this flux stream function. These are different conditions.

## 2. Analytical integration for N ideal line charges

Write `Xi = x-xi`, `Yi = y-yi`, and `ri² = Xi²+Yi²`. The continuum static field
for the model's infinitely long z-directed line charges is

```text
E = sum_i [qi/(2*pi*epsilon0)] (Xi, Yi)/ri².
```

The differential of the stream function is

```text
dpsi = -Ey dx + Ex dy
     = sum_i [qi/(2*pi*epsilon0)] (-Yi dx + Xi dy)/ri²
     = sum_i [qi/(2*pi*epsilon0)] d arg(Xi+i Yi).
```

Consequently, on a consistent choice of angle branches,

```text
psi(x,y) = sum_i [qi/(2*pi*epsilon0)] atan2(y-yi, x-xi) + constant.
```

Here `epsilon0=1` in simulation units. This expression can be evaluated
directly at each display sample; there is no streamline ODE to integrate.
The corresponding potential is `phi = -sum_i qi*log(ri)/(2*pi*epsilon0)`,
up to an arbitrary reference length and constant. Equivalently,
`-phi + i*psi = sum_i qi*log(z-zi)/(2*pi*epsilon0)` locally.

For nonzero E, equal increments `Delta psi` between contours represent equal
electric flux per unit z between neighboring lines. This directly addresses
the present tracer's arbitrary, unweighted seeding. It does not imply uniform
angular intersections with a circle: that count depends on the normal field
component, not just on field magnitude.

## 3. The almost-overlapping dipole has an especially simple solution

Place `+q` at `(-a,0)` and `-q` at `(a,0)`, in coordinates relative to their
midpoint and aligned with their separation. Let `k=q/(2*pi*epsilon0)`.

```text
psi/k = arg[(z+a)/(z-a)]
      = atan2(-2*a*y, x²+y²-a²)     modulo 2*pi.
```

The latter form avoids subtracting two almost equal angles in the far field.
For a contour `psi/k = alpha`, with `sin(alpha) != 0`, its supporting circle is

```text
x² + (y + a*cot(alpha))² = a²*csc(alpha)².
```

The field line is the appropriate **arc** through the two charges, selected
by the angle branch; plotting the whole circle would combine different
contour values. The x-axis supplies the degenerate straight-line cases.
These arcs can be tessellated directly to a screen-space geometric tolerance,
without either RK4 or a sampled contour grid. This special shortcut does not
extend to general N-charge configurations as a family of simple circles.

As `a -> 0` at fixed q, the field and stream function vanish away from the
charges. At exact overlap the opposite sources should be combined to zero,
and no field lines drawn. A finite ideal dipole at zero separation is a
different limit: it holds `2*a*q` fixed by increasing q.

The [previous near-overlap investigation](fieldlines-review.md) measured
substantial grid anisotropy near the default seed circles. Analytical dipole
arcs avoid that grid error **in the reference visualization**, but the
particle forces still come from the existing CIC/Yee field. Presenting those
arcs as the computed field would hide the discrepancy rather than resolve it.

## 4. Practical contouring requirements

- **Angle branches:** a loop around charge qi changes psi by `qi/epsilon0`.
  A globally single-valued regular psi therefore needs cuts or separate
  patches. Net zero total charge does not remove the issue for loops around
  individual charges. Ordinary contouring of raw `atan2` values can draw
  spurious lines across their discontinuities. Cut-aware unwrapping, patch
  stitching, or phase-aware contours are required.
- **Coincidence and cancellation:** combine exactly coincident opposite
  sources, use stable paired expressions for near dipoles, and suppress
  contours where the physical field is zero. `atan2(0,0)` is not a meaningful
  field-line label at a source.
- **Resolution:** an exact scalar formula sampled on a coarse uniform grid
  can still miss tiny loops and saddles or misconnect branches. General N
  requires adaptive sampling near charges, cancellation regions, and contour
  ambiguities. Marching squares needs appropriate saddle connectivity and
  explicit handling of singularities and cuts.
- **Geometry:** analytical integration removes the ODE error, not every
  numerical approximation. General contours still require numerical extraction
  and polygonal rendering. For a two-charge dipole, direct circle arcs bypass
  the grid-extraction step.
- **Meaning of line density:** use a stated Delta psi if equal-flux spacing is
  desired. Automatically changing it between scenes changes the represented
  flux per line. Arrows must be oriented using E; contours alone have no direction.

For the UI's equal-magnitude charges, phase-product methods could avoid many
individual angle evaluations and handle periodic contour labels. They still
need winding, singularity, and near-cancellation handling. The timings below
are for the simpler angle-sum implementation, not an optimized phase or GPU method.

## 5. Why particle positions are insufficient for the Maxwell simulation

The implementation initializes/adds electrostatic fields with a discrete
free-space Poisson solve. Subsequent Yee timesteps evolve E and B through
Maxwell's curl equations and deposited current; they do not reset E to the
instantaneous Coulomb field. Radiation and the surviving field after deletion
therefore cannot be reconstructed by the charge-angle sum.

A simple counterexample is a charge-free plane wave:

```text
Ey = E0*cos(k*x-omega*t),  Ex=0,  Bz=(E0/c)*cos(k*x-omega*t),  omega=c*k.
psi = -(E0/k)*sin(k*x-omega*t).
```

There are no charges to contribute to the angle sum, yet E and its stream
function are nonzero. In a regular source-free patch, combining the stream
definition with Faraday's law gives

```text
curl_z(E) = -Laplacian(psi) = -dBz/dt,
therefore Laplacian(psi) = dBz/dt.
```

So a dynamic vacuum stream function can exist, but it need not be harmonic
or a closed-form function of present particle positions. These conditions
follow from [Maxwell's equations](https://www.feynmanlectures.caltech.edu/II_18.html).
Pausing the app or pinning every charge does not erase pre-existing radiation.

There are also specific numerical constraints:

- **Distributed charge:** CIC spreads charge over nodal weights. A regular
  flux stream function cannot represent E in regions with nonzero divergence;
  charge support must be excluded or treated with patches/cuts. The ideal
  angular formula is not the discrete lattice-Green-function solution.
- **Interpolation:** the current separate bilinear gathers of Ex and Ey do
  not enforce continuous zero divergence between nodes, even when the discrete
  Yee divergence is zero. In the earlier 0.02-separation test, the discrete
  Gauss residual was about `1.1e-12`, but a centered finite-difference check of
  that interpolant at `(9.031,6.021)` gave divergence about `-2.86e-4`, outside
  the charged nodes. An exact scalar representation of that same interpolant
  is therefore generally unavailable. A contour method must specify a
  compatible field reconstruction and quantify the difference.
- **Deletion correction:** the optional removal model carries a nonzero
  `div(E)-rho` defect outward. A stream function assuming zero divergence
  throughout the particle-free region would discard part of that effect.

A more general scalar label chi satisfying `E dot grad(chi)=0` can exist
locally even for a divergent field. However, finding it requires solving a
transport equation whose characteristics are the field lines themselves.
That observation does not supply a general analytical shortcut.

## 6. A viable grid-based contour alternative

If the goal is to display the **evolved numerical field**, a separate approach
is to reconstruct a stream function from the field grid in suitable patches,
instead of from the particles. On this Yee layout, psi can live at Bz cell
centers, with

```text
Ex[i,j] = (psi[i,j]-psi[i,j-1])/dy,
Ey[i,j] = -(psi[i,j]-psi[i-1,j])/dx.
```

These differences have identically zero discrete divergence. Where Gauss's
law is source-free, psi differences can be accumulated on the grid or solved
for consistently, with source masks, cuts, and boundary data. An appropriate
divergence-preserving interpolation then supplies contours. This shares work
across all lines and avoids integrating a separate ODE per line, but it is
**numerical reconstruction**, not the requested closed-form charge solution.
It needs separate treatment where charge or the deletion defect makes
divergence nonzero, and more development than a small tracer replacement.

## 7. Measured cost and recommendation

I timed a straightforward float64, Numba-compiled angle sum on the available
ARM64 Mac, with compilation warmed and 15 repetitions per case. Each case
used a freshly initialized `Simulation(particles=N)` scene. M below counts
the sampling grid; evaluation costs O(N*M).

| Charges | psi on 129×81 samples | psi on 257×161 samples | Existing balanced RK4 lines |
| ---: | ---: | ---: | ---: |
| 2 | 0.21 ms | 0.69 ms | 2.92 ms |
| 6 | 0.72 ms | 2.21 ms | 1.86 ms |
| 100 | 14.36 ms | 47.15 ms | 0.79 ms |

The psi columns **exclude contour extraction, cuts, adaptive refinement, and
rendering**. The RK4 column includes completed line geometry and depends on
the scene, termination rules, and bounded seed/segment budgets. These are
not equivalent finished renderers and do not establish a general speedup.
The small-N analytical approach is promising; scaling to 100+ charges needs
optimization and an end-to-end measurement. As a formula check, finite
differences of the stable dipole psi reproduced its analytical field to about
`7.7e-10` relative error at four nonsingular sample points.

My recommendation is an explicitly labeled **electrostatic analytical
reference** using stream-function contours, with direct arcs for the
two-charge dipole and adaptive contour extraction for general N. It would
provide useful comparison against the evolving PIC field, especially for
the reported near-overlap case. Keep the numerical field visualization for
the full dynamic simulation. If contours of that dynamic field are the goal,
investigate the separate Yee-compatible reconstruction rather than replacing
it with the instantaneous charge-angle sum.
