# Moving-charge wake investigation and logarithmic time control

## Finding

The reported wake is reproducible with the supplied settings: one positive charge, `vx/c = 0.8`, `vy/c = 0`, default 128 × 80 grid, zero-acceleration constraint, and **Steady-motion fields**, before any bounce. Starting at the reset position `(8, 5)`, a run to `t = 0.39775` produces a closely matching trailing ripple pattern. The screenshot's exact elapsed time was unavailable.

The Yee curl signs, component staggering, light-speed factors, and charge-current continuity are correct. The important error is numerical: a narrow moving CIC source excites grid-scale modes and aliases in the dispersive Yee solver. The sampled continuum initial field is also not an exact discrete traveling solution, so startup transients accompany the continuing wake. Small Gauss residual does not exclude these transverse field errors.

The earlier `t ≈ 0.1`, `r > 1.5` benchmark was insufficient: it largely excluded the region reached by the source-core transient. Its small errors did **not** establish accurate longer-time moving-source fields. The new regression measures that wake directly.

## Checks and diagnosis

The update implements, in normalized units,

```text
Ex(t+dt) = Ex(t) + dt [c² Dy Bz(t+dt/2) - Jx(t+dt/2)]
Ey(t+dt) = Ey(t) - dt [c² Dx Bz(t+dt/2) + Jy(t+dt/2)]
Bz(t+dt/2) = Bz(t-dt/2) - dt [Dx Ey(t) - Dy Ex(t)]
```

The particle orbit supplies the step-averaged current. Its discrete divergence equals `-(rho_new-rho_old)/dt`. An independent traveling-wave test verifies the field update against the **discrete** Yee dispersion relation:

```text
sin²(omega dt/2) = c² dt² [sin²(kx dx/2)/dx² + sin²(ky dy/2)/dy²].
```

This differs from the continuum relation `omega² = c² |k|²`. With the default square mesh, `c dt/dx = 0.75/sqrt(2)`. Axis-aligned near-Nyquist waves have a phase speed of about `0.67c`, so a `0.8c` charge can outrun numerical waves even though it cannot outrun physical light. Cell-scale source structure and aliasing make the discrepancy visible. This interpretation is consistent with the single-particle analysis in [Xu et al., *On numerical errors to the fields surrounding a relativistically moving particle in PIC codes*](https://arxiv.org/abs/1910.13529), especially its Yee-solver and filtering discussion.

Further checks:

- Doubling hidden padding from 32 to 64 cells changes the electric components in the tested wake region by less than `2.3e-8`. The visible wake is already present before boundary disturbances return; it is not caused by a bounce or the visible perimeter.
- Halving the numerical timestep does not remove it: at `0.8c`, E error changes from 57.2% to 52.0%, while B error increases from 171% to 206% in the specified region. Smaller dt alone does not fix spatial dispersion and aliases.
- The error exists in raw E/B arrays, independently of color mapping and streamline tracing.
- The source-free traveling wave retains the expected Yee amplitude and phase. There is no new per-step field damping.

## Minimal mitigation implemented

The default now applies **two separable binomial passes** to deposited charge and current. Each pass convolves each axis with `[1, 2, 1]/4`. Two passes give `[1, 4, 6, 4, 1]/16` along each axis, on top of the existing CIC shape. This is a standard PIC noise-reduction approach; [Smilei's moving-plasma tutorial](https://github.com/SmileiPIC/tutorials/blob/master/advanced_wakefield_AMcylindrical.rst) discusses binomial current filtering for numerical Cherenkov radiation. It mitigates rather than eliminates the solver's dispersion error.

Consistency is essential:

1. Both `rho` and **both** orbit-current components receive the same convolution. Since convolution commutes with interior finite differences, the filtered sources still obey the discrete continuity equation.
2. The filter acts on the full padded grid, including source tails beyond the particle walls. Hidden current/charge tails are cleared before redeposition; they cannot accumulate from old steps.
3. Force gathering uses a filtered **copy** of the fields, the adjoint of this symmetric source filter. Stored E/B and the displayed fields are not smoothed during timesteps.
4. Initialization, new-charge addition, deletion's removed-source scalar, and the steady-motion initializer use the same source shape. The analytic initializer samples extra exterior cells before convolution, avoiding an artificial perimeter charge layer.

The Yee curls, Higuera–Cary pusher, exact reflected-orbit current formula, removal-wave evolution, outgoing boundary, and field-line rendering remain in place. The addition is localized to source/gather regularization. This avoids replacing the field solver with a spectral solver and changing its boundary/current treatment.

**Tradeoff:** two passes add one cell of RMS width in each coordinate to the charge shape. Core fields and close encounters therefore change; real source structure at the grid scale is suppressed too. The method does not turn the model into an exact point-charge calculation, nor guarantee relativistic accuracy near c. There is no claim of exact energy conservation. At fixed pass count, refining the grid also shrinks the physical source core, so a point-source wake need not improve monotonically. A convergence study of a smooth source must keep its physical width fixed while refining.

Use these launch options for comparison:

```sh
.venv/bin/python main.py                    # default: two passes
.venv/bin/python main.py --source-filter 0  # previous unfiltered CIC behavior
.venv/bin/python main.py --source-filter 4  # stronger smoothing, wider core
```

All integer pass counts from 0 to 4 are accepted. Restart, reset to one charge, enter the desired velocity, and click **Steady-motion fields** before comparing. Slowing playback alone does not change numerical accuracy.

## Measured results

All errors below are relative L2 errors of cell-centered vector E or B against the Lorentz-transformed continuum **line-charge** field, in `0.75 < |R| < 3` about the instantaneous source. Source starts at `(8, 5)` and moves horizontally at constant velocity. The singular/softened core is excluded. These are scenario-specific measurements, not universal error bounds.

| Speed | Time | Unfiltered E | Default E | Unfiltered B | Default B |
| --- | --- | --- | --- | --- | --- |
| 0.6c | 0.39775 | 44.5% | 1.40% | 152% | 2.89% |
| **0.8c** | **0.39775** | **57.2%** | **2.17%** | **171%** | **3.00%** |
| 0.8c | 0.69606 | 68.4% | 1.96% | 208% | 2.34% |
| 0.9c | 0.39775 | 70.3% | 4.57% | 178% | 5.92% |
| 0.99c | 0.39775 | 82.5% | 26.7% | 149% | 27.6% |

The `0.99c` case remains poorly resolved; smoothing is not a complete high-gamma field-solver fix. At `0.8c` and an oblique angle of 0.45 radians, default-grid errors decrease from 39.5%/109% to 4.40%/7.14% (E/B). Four passes reduce that case further to 3.23%/3.60%, at the cost of a wider core.

Regenerate the full table and image with:

```sh
.venv/bin/python tests/diagnose_solver.py
```

It writes `artifacts/solver-investigation.json` and `artifacts/solver-comparison.png`. The PNG's three panels are **unfiltered / default two passes / continuum reference**, all at `0.8c`, `t=0.39775`, with a shared color scale. The default panel still has a small residual transient and a softened core.

## Time slider and verification

The Controls tab now has a continuous base-2 logarithmic slider. For horizontal fraction `s` from left to right,

```text
time_scale = 2^(-5 + 6s),    0 <= s <= 1.
```

The range is `1/32×` to `2×`, with ticks at powers of two. The initial `1×` position is exactly `s=5/6`. Clicking or dragging sets the scale, including while paused. This controls accumulated simulation time, **not the CFL timestep**, and preserves the previous pause/reset behavior.

All **52 numerical tests** pass, including filtered-source charge conservation across grid crossings and wall bounces, hidden-cell tails, creation/deletion, independent wave dispersion, the longer `0.8c` wake comparison, oblique motion, and padding comparisons. The existing pusher, force, scalar display, and removal-wave checks still pass. The GUI smoke test verifies slider endpoints, midpoint, dragging outside the track, initial 1× position, paused edits, and actual 1/32-speed time accumulation, alongside all previous controls.

A warmed 100-charge run on the available ARM64 Mac measured approximately **0.35 ms per physics step**, versus approximately **0.07 ms** before source filtering, at 128 × 80 with 32-cell padding. This remains well below the approximately 11 ms simulation timestep; CPU timing is not a GUI frame-rate guarantee. All code and generated artifacts remain inside the project.

A separate six-second GUI measurement after two seconds of warmup rendered
**60.0 FPS** with 100 charges, the default two-pass filter, balanced line detail,
and a 2560 × 1578 framebuffer (95th-percentile frame interval: 20.6 ms). This is
the measured result on the available Mac, not a guarantee for every machine/load.
