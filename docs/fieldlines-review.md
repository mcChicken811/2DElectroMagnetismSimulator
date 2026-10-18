# Electric field lines and nearly overlapping dipoles

The lines are integral curves of the **numerical electric field** at one
simulation time. They are not analytical Coulomb curves, particle paths, or
the gradient of a separately solved stream function. The Higuera–Cary change
does not alter any of this code.

## What the renderer does

In [`direction`, `rk4`, and `trace`](../maxwell/fieldlines.py), the traced equation is

```text
dr/ds = ± E(r, t_snapshot) / |E(r, t_snapshot)|.
```

1. **Gather E:** independently interpolate Ex and Ey bilinearly from their
   correct staggered Yee locations. Arithmetic is float64 during tracing.
   The direction is normalized, so step size is measured in spatial distance,
   not electric-field strength.
2. **Integrate:** fourth-order Runge–Kutta (RK4), comparing one full step with
   two half-steps. The local error estimate is their endpoint separation / 15.
   Accept the two-half-step result if it is within `0.004*min(dx,dy)`.
   Step sizes range from `0.06` to `1.2` cell widths, with an initial maximum
   step and up to eight trial refinements. Tight integration tolerances do not
   remove error in the interpolated field itself.
3. **Seed:** low/balanced/high detail allows 60/112/176 nominal seeds, of which
   16 are placed near the canvas edges. The rest are shared across charges,
   approximately uniformly in angle on a circle of radius
   `max(2.1*min(dx,dy), 1.2*particle_radius)` around each charge. Positive seeds
   follow E; negative seeds trace backward. All arrowheads point along E.
   Empty scenes receive additional interior seeds to show residual fields.
4. **Stop:** at the visible edge, a 3×3 nodal mask around any charge, a weak
   field (`|E| < 1e-8`), a failed error tolerance at the minimum step, a detected
   return near the starting point, 750 steps, a path length exceeding 2.5 times
   the larger box dimension, or the global segment budget. These are display
   rules, not electromagnetic boundary conditions.
5. **Draw:** [`refresh_field`](../maxwell/app.py) turns float32 endpoint data
   into approximately 1.1-point-wide triangle ribbons, batched in OpenGL.
   Arrowheads start after 0.4 distance units and recur about every 1.1 units.
   Opacity increases with sampled field strength. Lines refresh at roughly
   15 Hz, while particles are drawn each frame, so moving markers can briefly
   be ahead of the last traced field snapshot.

Line count is **not weighted by charge magnitude or calibrated to flux**.
There is no occupancy grid or minimum-distance rule to avoid duplicate traces.
Starting from both signs and from the boundary can draw neighboring or
overlapping pieces of the same family. Consequently, apparent line crowding
alone is not a quantitative measure of field strength. Short gaps near charge
markers can also come from the core mask and finite seed radius.

## The near-overlap case

This simulation is invariant along z: its charges are line charges. The
continuum electrostatic comparison is therefore

```text
E(r) = sum_i [qi/(2*pi*epsilon0)] (r-ri)/|r-ri|²,
```

with `epsilon0=1`, rather than the 3D point-charge expression with a cubic
distance denominator. Exact overlap of equal and opposite static sources
cancels their field. A small nonzero separation has a nonzero dipole moment:
the far-field magnitude of an ideal 2D dipole is independent of angle at a
fixed radius, even though its vector direction and curved integral lines
depend on angle. The familiar 3D dipole has different angular behavior.

Two separate effects matter when separation is much smaller than a grid cell:

- **The grid cannot resolve the two cores.** Cloud-in-cell deposition and
  the discrete Poisson solution represent the pair as a small difference of
  mesh charge distributions. Close to the cores, this can introduce substantial
  direction-dependent error even when Gauss's law is satisfied to roundoff.
- **The two seed circles nearly coincide.** On the default mesh, each circle
  has radius 0.2625. Balanced detail assigns 48 seeds to each member of a
  two-charge pair. These are not flux-weighted dipole seeds; their traced
  paths can overlap or bunch up, and their core stopping masks also overlap.

I compared the initialized field for `+2.5` at `(7.99,5)` and `−2.5` at
`(8.01,5)` in a 16×10 box, before any motion, against the continuum expression
above. At each radius I sampled 360 equally spaced angles using the same
bilinear gather as the tracer. Both grids used 32 hidden cells per side.

| Visible grid | Radius from midpoint | Maximum direction error | Numerical max/min magnitude | Analytical max/min magnitude |
| --- | ---: | ---: | ---: | ---: |
| 128×80 | 0.2625 | 17.06° | 1.948 | 1.0029 |
| 256×160 | 0.2625 | 4.81° | 1.276 | 1.0029 |
| 128×80 | 1.0 | 1.79° | 1.066 | 1.0002 |
| 256×160 | 1.0 | 0.43° | 1.016 | 1.0002 |

The default-grid maximum at radius 0.2625 lies along the dipole axis. Thus
there is measurable artificial angular concentration in the field near the
default seed radius, before considering the uneven line sampling. Moving
farther from the cores or refining the grid reduces the error. For a separate
exact-overlap initialization at `(8,5)`, both electric components canceled
exactly to zero in this check.

These measurements identify plausible causes of the reported appearance;
they are a controlled stationary example, not a reconstruction of a saved
interactive scene. Existing radiation, recent dragging, or charge motion can
also make the instantaneous field differ from an electrostatic dipole.
Higuera–Cary improves particle integration in crossed fields; it does not fix
this near-core field resolution or the choice of streamline seeds.

For comparison without code changes, use stationary charges, inspect the
`|E|` color layer with lines hidden, and compare the same physical points on
a finer grid (for example `--grid 256x160`). Keep comparing numeric color
legends: each heatmap uses its own automatic maximum. No field-line code was
modified for this investigation.
