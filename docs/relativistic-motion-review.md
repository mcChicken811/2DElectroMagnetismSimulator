# Relativistic motion, moving-source fields, and the Inspector

**Follow-up:** the [solver investigation](solver-investigation.md) reproduces the
reported longer-time 0.8c wake and adds consistent source/gather filtering.
The short-time accuracy figures and 45-test count below describe the earlier
unfiltered version. They do not establish longer-run field accuracy; the current
suite has 52 tests and the default uses two source-filter passes.

## Conclusion of the code and literature review

The existing Higuera–Cary particle pusher already implements the special-relativistic Lorentz equation. Replacing it would not, by itself, fix a moving charge's field. The simulation also has a finite-resolution Yee field solver, a CIC source shape, and an initial condition. Those must be distinguished from the equation of motion.

The small numerical correction made here is to use **B at the same integer time as E** when converting an edited/initial velocity into the pusher's staggered momentum. The new Inspector uses the same centering. The ordinary Higuera–Cary update, Yee curls, orbit-current deposition, field-line tracing, and outer boundary algorithm are retained.

Three primary academic sources informed the review:

- A. V. Higuera and J. R. Cary, *Structure-preserving second-order integration of relativistic charged particle trajectories in electromagnetic fields*, Physics of Plasmas **24**, 052104 (2017). [Paper](https://arxiv.org/abs/1701.05605), [published article](https://doi.org/10.1063/1.4979989). Equations (1–2) specify the relativistic dynamics; Eq. (20) supplies the magnetic rotation factor used here. The method preserves force-balanced E × B drift and phase-space volume.
- B. Ripperda et al., *A comprehensive comparison of relativistic particle integrators*, Astrophysical Journal Supplement Series **235**, 21 (2018). [Paper](https://arxiv.org/abs/1710.09164), [published article](https://doi.org/10.3847/1538-4365/aab114). This comparison tests relativistic pushers in imposed fields. Its distinction between trajectory tests and self-consistent field feedback is relevant here: successful orbit integration does not validate the field solver.
- X. Xu et al., *On numerical errors to the fields surrounding a relativistically moving particle in PIC codes* (2019 preprint). [Paper](https://arxiv.org/abs/1910.13529). It analyzes discretization errors in the fields of moving PIC particles, including unphysical space-charge fields, numerical Cherenkov radiation, and self-forces. These can occur independently of the particle pusher. Its customized field solvers would require a larger change than this Inspector feature.

## Equation of motion and the acceleration displayed

Let `u = gamma v` denote proper velocity, so the code's `momentum` is momentum **per unit rest mass**, not ordinary velocity. With lab-frame three-force F,

```text
gamma = 1 / sqrt(1 - |v|²/c²)
dx/dt = v = u / sqrt(1 + |u|²/c²)
du/dt = f,       f = F/m = (q/m)(E + v × B) + Fexternal/m
```

Differentiating `u = gamma v` gives

```text
f = gamma a + gamma³ v (v · a)/c²
v · f = gamma³ (v · a)
a = dv/dt = [f - v(v · f)/c²] / gamma
```

Consequently, force parallel to the velocity gives `a_parallel = F_parallel/(m gamma³)`; perpendicular force gives `a_perp = F_perp/(m gamma)`. Substituting `F/m` directly into a displayed acceleration would be incorrect at relativistic speeds. The Inspector displays this **coordinate acceleration**, including the existing drag force if active. It does not display `du/dt`, proper acceleration, or a noisy finite difference of adjacent positions.

In the current 2D geometry, `u_z = 0` and `B = (0, 0, Bz)`. The Higuera–Cary rotation specializes to

```text
u_minus = u_old + (dt/2) f_electric
 tau = (q/m) Bz dt/2
 sigma = 1 + |u_minus|²/c² - tau²
 gamma_rotation² = (sigma + sqrt(sigma² + 4 tau²)) / 2
```

The existing implementation matches this expression, including a rationalized positive root for large negative sigma. No extra factor of gamma should be placed in the electric kick: gamma enters when converting u to v and in the magnetic rotation.

Ordinary particles store `u(t-dt/2)`, fields store `E(t), B(t-dt/2)`. For inspection, a temporary Faraday half-step reconstructs `B(t)`, and a temporary Higuera–Cary half-step reconstructs `u(t)`. Edits use the inverse half-step. This is a second-order synchronization using the locally frozen fields, not an exact dense solution between timesteps. It does not mutate the fields. Wall reflections, velocity edits, and pinning are impulses; they are excluded from the finite acceleration readout.

## Why constant velocity can appear to give the wrong field

These causes follow from the inspected implementation; they do not uniquely diagnose an unsaved screenshot:

1. **Initial data:** startup/reset uses a Poisson electric field and zero magnetic field. Startup particles can nevertheless have nonzero velocities. A velocity edit also preserves the existing E/B. These are transient initial conditions, not the fields of an eternally uniformly moving charge. Maxwell evolution cannot instantly replace them everywhere with a boosted static field.
2. **Dimensionality:** this is a z-invariant line-charge model. A Lorentz-transformed **3D point-charge** field has a different angular dependence and distance law.
3. **Discretization:** a Cartesian Yee mesh and its CIC source are not exactly Lorentz covariant at finite resolution. Near-core fields, sub-cell source position, numerical dispersion, self-force, and outgoing-boundary error remain. High gamma compresses the field along the motion and makes resolution more demanding. Small Gauss residual alone does not establish field accuracy.
4. **Display:** streamlines show direction. Their density is not calibrated to flux. For uniform translation the ideal electric field below is still radial about the instantaneous source position, even though its magnitude is anisotropic. Use magnitude coloring and numerical comparisons for that anisotropy.

A field-solver replacement or a full 3D model is outside this minimal update. The explicit initializer below supplies a reproducible comparison without overwriting the evolved fields on every step.

## Lorentz-transformed line-charge reference

This derivation starts with the model's rest-frame electrostatic solution, with `epsilon0 = 1`:

```text
E'(R') = q R' / (2 pi |R'|²),      B' = 0.
```

Here q is charge per unit z. An in-plane boost leaves the total charge of a unit-z slice unchanged. For a source moving at lab velocity v, resolve the displacement from its **instantaneous lab position** as `R_parallel + R_perp`. At equal lab time,

```text
R'_parallel = gamma R_parallel,   R'_perp = R_perp
E_parallel = E'_parallel,         E_perp = gamma E'_perp
B = v × E / c²
```

These are the rest-to-lab electromagnetic tensor transformation with a source moving at **+v**. Combining them gives

```text
D = |R|² + gamma² (v · R / c)²
E = q gamma R / (2 pi D)
Bz = (vx Ey - vy Ex) / c²
```

At equal radius, `E_transverse/E_parallel = gamma²` for this line source; the 3D point-source ratio is `gamma³`. The unit test independently builds the 4×4 antisymmetric field tensor, applies `F_lab = Lambda F_rest Lambda^T` for oblique boosts, and compares its components with this formula. The ideal formula excludes the singular source itself.

**Steady-motion fields** requires exactly one selected particle. It samples this field on the staggered Yee locations with a small rest-frame core of one quarter cell, projects E onto the existing discrete CIC Gauss constraint, and stores B at its required half-time. This projection changes the sampled analytic E; the result is a compatible **approximate mesh initial condition**, not an exact discrete traveling-wave solution. The action clears old radiation/removal waves and enables zero-acceleration constraint motion at the current velocity. It preserves position and simulation time and leaves the simulation paused. Later timesteps use the normal Maxwell solver.

For a reproducible check, start at `(7.173, 4.619)`, set `v=(0.6c, 0)`, initialize, then advance to approximately `t=0.1`. Compare cell-centered fields with the continuum reference in the annulus `1.5 < |R| < 3`, before wall/boundary effects return. Measured relative L2 errors were:

| Visible mesh | Time | E error | B error |
| --- | --- | --- | --- |
| 64 × 40 | 0.11049 | 1.65% | 0.591% |
| 128 × 80 | 0.09944 | 1.46% | 0.126% |
| 256 × 160 | 0.09944 | 0.671% | 0.0315% |

Padding is 4 physical units at every resolution. These are one short-run benchmark, not universal bounds. Source-core projection and CIC sub-cell position can make E error nonmonotonic at other resolutions. No accuracy claim near the singular core or at arbitrarily high gamma follows from this test.

## Inspector and constraint semantics

- Single-click a charge, proton, or electron to select it and open **Inspector**. Selection alone applies no force. A free particle starts dragging only after the cursor moves more than five pixels while held.
- The two velocity inputs are **vx/c and vy/c**. Acceleration inputs are **dvx/dt and dvy/dt in normalized simulation units**, not divided by c. The state, speed/c, and gamma are also shown.
- Clicking an editable value pauses and clears accumulated time. Type a number (scientific notation supported), then **Apply edits / Enter**. **Discard / Escape** abandons the draft. Apply does not advance time or change E/B. Resume is blocked while an edit is pending; after applying/discarding, **P** resumes and **N** takes one timestep. Switching tabs or selecting another particle discards an uncommitted draft.
- With **Constraint Mode OFF**, velocity is editable and acceleration is a live Lorentz-force readout. With **Constraint Mode ON**, acceleration is editable and specifies the **total** coordinate acceleration. Enabling starts at `a=0` while keeping current v. Disabling preserves current v and restores Lorentz dynamics.
- **Double-click / Fix charge** sets `v=0`, `a=0`, and Constraint Mode ON. Double-click a fixed charge again to release it. Editing its velocity or acceleration can make it move while keeping the constraint enabled. Fixed particles retain the outline ring.
- Constrained motion overrides Lorentz and drag forces, while the particle remains a charge/current source. Disable the mode before dragging. Ordinary field evolution and other particles continue normally.
- Deleting a selected particle clears its Inspector selection while retaining the existing field-preserving deletion behavior. Reset clears constraints and selection.

For the prescribed acceleration a, the implicit external constraint force is

```text
Fconstraint = m gamma [a + gamma² v(v · a)/c²] - q(E + v × B).
```

It is implemented by advancing the specified velocity and drifting at its midpoint, rather than by numerically cancelling large forces. Constant lab acceleration is integrated exactly between reflections; the existing straight-segment reflected orbit deposition conserves charge. Reflections preserve the previous collision rule (normal velocity flips), with the same split-step wall approximation. They do not reverse the prescribed lab acceleration vector. This externally driven mode need not conserve particle-plus-field energy.

A nonzero constant coordinate acceleration can eventually demand `|v| >= c`; the required constraint force then diverges. Invalid edits are rejected. During a run, the simulator **pauses before the first inadmissible step**, preserving the last valid state, rather than clamping velocity or accepting a superluminal trajectory. Reduce v/a or disable Constraint Mode to continue.

## Verification

The numerical suite now has **45 tests**. New checks cover coordinate acceleration's parallel/perpendicular relativistic factors, integer-time velocity round trips, field-preserving edits, constraint trajectories and release, reflection and Gauss preservation, fixed-mode equivalence, metadata lifecycle, superluminal rejection, independent tensor boosts, and the moving-field benchmark above. The existing crossed-field drift tests through `0.9999c` and all field/deletion tests remain.

The real-window smoke test covers selection without force, edits, text/shortcut isolation, pause/resume and single-step, constraint toggling, fixed velocity editing, invalid drafts, the speed-limit pause, and field initialization, plus the earlier controls. It saves dark/light/compact Inspector screenshots under `artifacts/`.

A warmed 100-charge, 128 × 80 numerical run on the available ARM64 Mac measured about **0.070 ms per physics step** over 1,000 steps and **1.08 ms** median line refresh. These are CPU timings, not a GUI frame-rate guarantee. The optional analytic initializer runs only on explicit request; it adds no per-step analytic field calculation.
