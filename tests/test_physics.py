import unittest
import numpy as np
from maxwell.physics import (Config, Simulation, boris, deposit_charge,
                             deposit_segment, drift_reflect, advance_b, advance_e,
                             PROTON_ELECTRON_MASS_RATIO, SPECIES)
from maxwell.fieldlines import trace
from maxwell.heatmap import scalar_field, field_scale, colorize


class PhysicsTests(unittest.TestCase):
    def small(self, n=0):
        return Simulation(Config(nx=48, ny=32, width=12, height=8), particles=n)

    def assert_removal_continuity(self, sim):
        """Particle continuity still holds with the explicit scalar correction."""
        sim.update_rho()
        before = sim.divergence()-sim.rho[1:-1, 1:-1]
        sim.step()
        sim.update_rho()
        p = sim.config.field_padding
        psi = sim._removal_psi[p:-p, p:-p]
        laplacian = ((psi[2:, 1:-1]-2*psi[1:-1, 1:-1]+psi[:-2, 1:-1])/sim.dx**2
                     + (psi[1:-1, 2:]-2*psi[1:-1, 1:-1]+psi[1:-1, :-2])/sim.dy**2)
        after = sim.divergence()-sim.rho[1:-1, 1:-1]
        np.testing.assert_allclose(after, before-sim.dt*laplacian, atol=3e-11)

    def test_discrete_poisson_and_addition(self):
        sim = self.small(12)
        self.assertLess(sim.diagnostics()['gauss_linf'], 2e-12)
        # Editing must preserve curl E and the existing magnetic field.
        sim.bz[:] = np.random.default_rng(9).normal(size=sim.bz.shape)
        before = sim.bz.copy()
        curl0 = np.diff(sim.ey, axis=0)/sim.dx-np.diff(sim.ex, axis=1)/sim.dy
        sim.add_charge(3.2, 4.8, -1)
        curl1 = np.diff(sim.ey, axis=0)/sim.dx-np.diff(sim.ex, axis=1)/sim.dy
        np.testing.assert_allclose(curl0, curl1, atol=2e-13)
        np.testing.assert_array_equal(before, sim.bz)
        self.assertLess(sim.diagnostics()['gauss_linf'], 2e-12)

    def test_orbit_current_continuity(self):
        sim = self.small()
        rng = np.random.default_rng(42)
        cases = [(rng.uniform([1, 1], [11, 7]), rng.uniform([1, 1], [11, 7]))
                 for _ in range(70)]
        cases += [(np.array([3., 4.]), np.array([1., 2.])),
                  (np.array([2., 2.]), np.array([2., 6.])),
                  (np.array([3., 3.]), np.array([3., 3.]))]
        for start, end in cases:
            sim.jx[:] = 0
            sim.jy[:] = 0
            old, new = np.zeros_like(sim.rho), np.zeros_like(sim.rho)
            deposit_charge(start[None, :], np.array([2.5]), old, sim.dx, sim.dy)
            deposit_charge(end[None, :], np.array([2.5]), new, sim.dx, sim.dy)
            deposit_segment(*start, *end, 2.5, sim.jx, sim.jy, sim.dx, sim.dy, sim.dt)
            divj = np.diff(sim.jx[:, 1:-1], axis=0)/sim.dx + np.diff(sim.jy[1:-1, :], axis=1)/sim.dy
            residual = (new-old)[1:-1, 1:-1]+sim.dt*divj
            self.assertLess(np.max(np.abs(residual)), 3e-12)

    def test_reflection_changes_only_normal_velocity(self):
        sim = self.small()
        x, y, sx, sy = drift_reflect(11.5, 4, 2, .3, 1, sim.jx, sim.jy,
                                     sim.dx, sim.dy, .5, 12, 8, .4)
        self.assertAlmostEqual(x, 10.7)
        self.assertAlmostEqual(y, 4.15)
        self.assertEqual((sx, sy), (-1, 1))

    def test_corner_reflection_and_current(self):
        sim = self.small()
        old = np.zeros_like(sim.rho)
        start = np.array([[11.5, 7.5]])
        deposit_charge(start, np.array([1.]), old, sim.dx, sim.dy)
        x, y, sx, sy = drift_reflect(11.5, 7.5, 2, 2, 1, sim.jx, sim.jy,
                                     sim.dx, sim.dy, .5, 12, 8, .4)
        self.assertEqual((sx, sy), (-1, -1))
        deposit_charge(np.array([[x, y]]), np.array([1.]), sim.rho, sim.dx, sim.dy)
        divj = np.diff(sim.jx[:, 1:-1], axis=0)/sim.dx + np.diff(sim.jy[1:-1, :], axis=1)/sim.dy
        self.assertLess(np.max(np.abs((sim.rho-old)[1:-1, 1:-1]+.5*divj)), 1e-12)

    def test_boris_magnetic_energy_and_rotation(self):
        ux, uy = 1.7, -.4
        norm = np.hypot(ux, uy)
        px, py = boris(1, 0, 0, 0, 1, 1, .01, 6)
        self.assertLess(py, 0)  # v cross B turns positive x toward negative y.
        for _ in range(20000):
            ux, uy = boris(ux, uy, 0, 0, 3, 1, .02, 6)
        self.assertAlmostEqual(np.hypot(ux, uy), norm, places=11)

    def test_constant_electric_acceleration(self):
        ux, uy = 0., 0.
        for _ in range(100):
            ux, uy = boris(ux, uy, 3, -2, 0, 2, .01, 6)
        np.testing.assert_allclose([ux, uy], [6., -4.], atol=2e-13)

    def test_yee_curls_against_analytic_mode(self):
        # Isolate the unchanged curl kernels with TEST-ONLY fixed endpoints.
        # Simulation.step instead evolves the padded grid with an outgoing edge.
        sim = self.small()
        cfg = sim.config
        initial = np.sin(np.pi*np.arange(cfg.nx+1)/cfg.nx)[:, None]
        sim.ey[:] = initial
        sim.ey[0, :], sim.ey[-1, :] = 0, 0
        sim.bz[:] = .5*sim.dt*np.diff(sim.ey, axis=0)/sim.dx
        kd = 2*np.sin(np.pi/(2*cfg.nx))/sim.dx
        omega = 2/sim.dt*np.arcsin(cfg.c*sim.dt*kd/2)
        for _ in range(200):
            advance_b(sim.ex, sim.ey, sim.bz, sim.centered_b, sim.dx, sim.dy, sim.dt)
            advance_e(sim.ex, sim.ey, sim.bz, sim.jx, sim.jy, sim.dx, sim.dy, sim.dt, cfg.c)
        expected = np.broadcast_to(initial*np.cos(omega*200*sim.dt), sim.ey.shape)
        np.testing.assert_allclose(sim.ey, expected, atol=4e-13)

    def test_long_run_gauss_and_confinement(self):
        sim = self.small(24)
        sim.momentum *= 18  # Deliberately exercise repeated wall collisions.
        sim.step(1200)
        d = sim.diagnostics()
        self.assertLess(d['gauss_linf'], 3e-10)
        self.assertTrue(np.all(np.isfinite(sim.ex)))
        self.assertTrue(np.all(np.isfinite(sim.ey)))
        self.assertTrue(np.all(np.isfinite(sim.bz)))
        self.assertLess(d['max_speed_over_c'], 1)
        self.assertTrue(np.all(sim.pos >= sim.radius-1e-12))
        self.assertTrue(np.all(sim.pos <= np.array([12, 8])-sim.radius+1e-12))
        # Deletion now creates an explicit travelling constraint adjustment.
        # Current deposition must still satisfy the modified Gauss identity.
        sim.add_charge(.5, .5)
        sim.remove_charge(4)
        sim.step(100)
        self.assert_removal_continuity(sim)

    def test_free_space_static_field_and_translation(self):
        patches = []
        for x in (.5, 6.):
            sim = self.small()
            sim.add_charge(x, 4.)
            sim.toggle_fixed(0)
            p = sim.config.field_padding
            i, j = p+round(x/sim.dx), p+round(4/sim.dy)
            patches.append((sim._ex[i-5:i+5, j-5:j+6].copy(),
                            sim._ey[i-5:i+6, j-5:j+5].copy()))
            ex, ey = sim.ex.copy(), sim.ey.copy()
            sim.step(1500)  # Much longer than a round trip to the outer boundary.
            np.testing.assert_allclose(sim.ex, ex, atol=3e-12)
            np.testing.assert_allclose(sim.ey, ey, atol=3e-12)
            np.testing.assert_allclose(sim.bz, 0, atol=3e-12)
            self.assertLess(sim.diagnostics()['gauss_linf'], 3e-12)
            # Tangential E is allowed on every visible edge (no conductor).
            self.assertGreater(np.max(np.abs(sim.ex[:, [0, -1]])), .01)
            self.assertGreater(np.max(np.abs(sim.ey[[0, -1], :])), .01)
        for edge_field, center_field in zip(*patches):
            np.testing.assert_allclose(edge_field, center_field, atol=3e-13)

    def test_free_space_poisson_on_rectangular_cells(self):
        # Unequal dx/dy and non-neutral charge must work without a neutralizing
        # background. Check the continuum 2D line-charge field far from its core.
        sim = Simulation(Config(nx=64, ny=32, width=16, height=4), particles=0)
        sim.add_charge(4, 2)
        self.assertLess(sim.diagnostics()['gauss_linf'], 5e-12)
        x = (48+.5)*sim.dx
        expected = sim.config.charge/(2*np.pi*(x-4))
        self.assertAlmostEqual(sim.ex[48, 16]/expected, 1., delta=.002)
        self.assertAlmostEqual(sim.ey[16, 24]/
                               (sim.config.charge/(2*np.pi*((24+.5)*sim.dy-2))),
                               1., delta=.025)

    def test_outgoing_pulse_against_large_domain(self):
        sims = []
        for padding in (32, 128):
            sim = Simulation(Config(nx=48, ny=32, width=12, height=8,
                                     field_padding=padding), particles=0)
            x = (np.arange(sim._bz.shape[0])+.5-padding)*sim.dx-6
            y = (np.arange(sim._bz.shape[1])+.5-padding)*sim.dy-4
            sim._bz[:] = np.exp(-(x[:, None]**2+y[None, :]**2)/(2*.6**2))
            sims.append(sim)
        initial = sims[0].diagnostics()['field_energy']
        for time in (2., 4., 6.):
            for sim in sims:
                sim.step(round((time-sim.time)/sim.dt))
            actual, reference = sims
            # Includes all four outgoing sides and their corners. The reference
            # edge is far enough away that reflected waves have not returned.
            error = .5*actual.dx*actual.dy*(
                np.sum((actual.ex-reference.ex)**2)
                + np.sum((actual.ey-reference.ey)**2)
                + actual.config.c**2*np.sum((actual.bz-reference.bz)**2))
            self.assertLess(error/initial, 5e-4)
            self.assertLess(actual.diagnostics()['gauss_linf'], 1e-12)
        self.assertLess(actual.diagnostics()['field_energy']/initial, 5e-4)

    def test_drag_and_empty_simulation(self):
        sim = self.small()
        sim.step(10)
        self.assertEqual(sim.diagnostics()['field_energy'], 0)
        sim.add_charge(4, 4)
        sim.dragged = 0
        sim.target[:] = [6, 4]
        sim.step(40)
        self.assertGreater(sim.pos[0, 0], 4.5)
        self.assertLess(sim.diagnostics()['gauss_linf'], 2e-11)
        sim.remove_charge(0)
        sim.step(10)
        self.assert_removal_continuity(sim)

    def test_delete_stationary_charge_preserves_all_fields(self):
        for fixed in (False, True):
            for species in ('proton', 'electron'):
                with self.subTest(fixed=fixed, species=species):
                    sim = self.small()
                    sim.add_charge(6, 4, species=species)
                    if fixed:
                        sim.toggle_fixed(0)
                    sim.bz[:] += .01  # Existing magnetic field also survives.
                    fields = [a.copy() for a in (sim._ex, sim._ey, sim._bz, sim._centered_b)]
                    old_rho = sim.rho.copy()
                    sim.remove_charge(0)
                    self.assertEqual(len(sim.charge), 0)
                    self.assertTrue(sim.removal_active)
                    np.testing.assert_array_equal(sim.rho, 0)
                    for actual, expected in zip((sim._ex, sim._ey, sim._bz, sim._centered_b), fields):
                        np.testing.assert_array_equal(actual, expected)
                    np.testing.assert_allclose(sim.divergence(), old_rho[1:-1, 1:-1], atol=2e-12)

    def test_deletion_adjustment_spreads_then_leaves(self):
        sim = self.small()
        sim.add_charge(6, 4)
        sim.toggle_fixed(0)
        ex, ey = sim.ex.copy(), sim.ey.copy()
        energy = sim.diagnostics()['field_energy']
        sim.remove_charge(0)
        sim.step(5)
        # A remote strip is outside even the discrete stencil's causal reach.
        np.testing.assert_allclose(sim.ex[:8], ex[:8], atol=2e-13)
        np.testing.assert_allclose(sim.ey[:8], ey[:8], atol=2e-13)
        self.assertGreater(np.max(np.abs(sim.ex-ex)), .1)
        # No hidden static ghost source; most field energy eventually escapes.
        sim.step(round(8/sim.dt)-sim.steps)
        self.assertLess(sim.diagnostics()['field_energy']/energy, .005)
        np.testing.assert_allclose(sim.bz, 0, atol=2e-13)
        self.assert_removal_continuity(sim)

    def test_deletion_preserves_propagating_magnetic_wave(self):
        sims = [self.small(), self.small()]
        for sim in sims:
            sim.add_charge(6, 4)
            sim.toggle_fixed(0)
            p = sim.config.field_padding
            x = (np.arange(sim._bz.shape[0])+.5-p)*sim.dx-4
            y = (np.arange(sim._bz.shape[1])+.5-p)*sim.dy-3
            sim._bz[:] = .1*np.exp(-(x[:, None]**2+y[None, :]**2))
        sims[0].remove_charge(0)
        for sim in sims:
            sim.step(150)
        # The longitudinal removal adjustment cannot erase transverse radiation.
        np.testing.assert_allclose(sims[0]._bz, sims[1]._bz, atol=3e-13)
        self.assertGreater(np.max(np.abs(sims[0].bz)), 1e-5)

    def test_addition_and_repeated_deletion_preserve_removal_wave(self):
        sim = self.small()
        sim.add_charge(6, 4)
        sim.remove_charge(0)
        sim.step(20)
        before = sim.divergence()-sim.rho[1:-1, 1:-1]
        psi = sim._removal_psi.copy()
        sim.add_charge(9, 4, species='electron')
        np.testing.assert_allclose(sim.divergence()-sim.rho[1:-1, 1:-1], before, atol=3e-12)
        np.testing.assert_array_equal(sim._removal_psi, psi)
        fields = [a.copy() for a in (sim._ex, sim._ey, sim._bz)]
        sim.remove_charge(0)
        for actual, expected in zip((sim._ex, sim._ey, sim._bz), fields):
            np.testing.assert_array_equal(actual, expected)
        self.assert_removal_continuity(sim)
        sim.reset_to_single_positive()
        self.assertFalse(sim.removal_active)
        np.testing.assert_array_equal(sim._removal_psi, 0)
        np.testing.assert_array_equal(sim._centered_psi, 0)
        self.assertLess(sim.diagnostics()['gauss_linf'], 2e-12)

    def test_invalid_deletion_leaves_state_unchanged(self):
        sim = self.small(1)
        before = sim.ex.copy()
        sim.remove_charge(-1)
        sim.remove_charge(1)
        self.assertEqual(len(sim.charge), 1)
        self.assertFalse(sim.removal_active)
        np.testing.assert_array_equal(sim.ex, before)

    def test_uniform_field_streamlines(self):
        sim = self.small()
        sim.ex[:] = 2
        out = trace(sim.ex, sim.ey, np.array([[2., 3., 1.]]),
                    np.zeros_like(sim.rho, dtype=np.bool_), sim.dx, sim.dy, 12, 8, 10000)
        # Main segments and arrows all remain close to the straight line.
        self.assertGreater(len(out), 20)
        self.assertLess(np.max(np.abs(out[:, 1]-3)), .04)
        self.assertAlmostEqual(float(np.max(out[:, 0])), 12)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_opposite_charges_attract(self):
        sim = self.small()
        sim.add_charge(4, 4, 1)
        sim.add_charge(8, 4, -1)
        sim.step(1)
        self.assertGreater(sim.momentum[0, 0], 0)
        self.assertLess(sim.momentum[1, 0], 0)

    def test_reset_leaves_one_free_positive_at_rest(self):
        sim = self.small(12)
        sim.toggle_fixed(0)
        sim.step(40)
        sim.reset_to_single_positive()
        self.assertEqual(len(sim.charge), 1)
        self.assertGreater(sim.charge[0], 0)
        np.testing.assert_allclose(sim.pos[0], [6., 4.])
        np.testing.assert_allclose(sim.momentum, 0, atol=1e-14)
        self.assertFalse(sim.fixed[0])
        self.assertEqual(sim.species[0], SPECIES['charge'])
        self.assertEqual((sim.steps, sim.time, sim.dragged), (0, 0., -1))
        np.testing.assert_array_equal(sim.bz, 0)
        np.testing.assert_array_equal(sim._bz, 0)  # Clear hidden radiation too.
        np.testing.assert_array_equal(sim._jx, 0)
        np.testing.assert_array_equal(sim._jy, 0)
        self.assertLess(sim.diagnostics()['gauss_linf'], 2e-12)

    def test_fixed_charge_stays_a_source_but_has_no_current(self):
        sim = self.small()
        sim.add_charge(4, 4)
        initial_pos, initial_rho = sim.pos.copy(), sim.rho.copy()
        self.assertTrue(sim.toggle_fixed(0))
        sim.momentum[0] = [3, 4]
        sim.ex += .4
        sim.ey -= .3
        sim.bz += .2
        sim.dragged, sim.target[:] = 0, [9, 7]
        sim.step(80)
        np.testing.assert_array_equal(sim.pos, initial_pos)
        np.testing.assert_array_equal(sim.momentum, 0)
        np.testing.assert_array_equal(sim.jx, 0)
        np.testing.assert_array_equal(sim.jy, 0)
        sim.update_rho()
        np.testing.assert_array_equal(sim.rho, initial_rho)
        self.assertGreater(np.sum(sim.rho), 0)
        self.assertLess(sim.diagnostics()['gauss_linf'], 2e-11)

    def test_release_restores_lorentz_motion_and_gauss(self):
        sim = self.small()
        sim.add_charge(4, 4)
        sim.add_charge(8, 4, -1)
        sim.toggle_fixed(0)
        sim.toggle_fixed(1)
        sim.step(10)
        self.assertFalse(sim.toggle_fixed(0))
        sim.step(40)
        self.assertGreater(sim.pos[0, 0], 4)
        np.testing.assert_array_equal(sim.pos[1], [8, 4])
        self.assertLess(sim.diagnostics()['gauss_linf'], 2e-11)

    def test_species_charge_and_mass_response(self):
        momenta = []
        for species in ('proton', 'electron'):
            sim = self.small()
            sim.add_charge(6, 4, species=species)
            sim.ex[:] = 1
            sim.ey[:] = 0
            sim.momentum[:] = 0
            sim.step()
            momenta.append(sim.momentum[0, 0])
            self.assertEqual(sim.species[0], SPECIES[species])
        self.assertGreater(momenta[0], 0)
        self.assertLess(momenta[1], 0)
        self.assertAlmostEqual(-momenta[1]/momenta[0], PROTON_ELECTRON_MASS_RATIO, places=9)

    def test_species_and_fixed_metadata_survive_deletion(self):
        sim = self.small(3)
        sim.add_charge(4, 4, species='proton')
        sim.add_charge(7, 4, species='electron')
        sim.toggle_fixed(3)
        sim.remove_charge(0)
        self.assertTrue(sim.fixed[2])
        np.testing.assert_array_equal(sim.species[-2:], [SPECIES['proton'], SPECIES['electron']])
        np.testing.assert_allclose(sim.mass[-2:], [PROTON_ELECTRON_MASS_RATIO, 1])
        np.testing.assert_array_equal(sim.charge[-2:], [sim.config.charge, -sim.config.charge])
        sim.step(100)
        self.assert_removal_continuity(sim)

    def test_scalar_field_magnitude_and_energy(self):
        sim = self.small()
        sim.ex[:] = 3
        sim.ey[:] = 4
        sim.bz[:] = -.2
        for mode, expected in [('electric', 5), ('magnetic', .2), ('signed', -.2),
                               ('energy', .5*(25+sim.config.c**2*.04))]:
            np.testing.assert_allclose(scalar_field(sim, mode), expected)
        np.testing.assert_array_equal(sim.bz, -.2)

    def test_scalar_field_uses_time_centered_magnetic_field(self):
        sim = self.small()
        sim.ey[:] = (np.arange(sim.config.nx+1)*sim.dx)[:, None]
        expected_b = -.5*sim.dt
        np.testing.assert_allclose(scalar_field(sim, 'signed'), expected_b)
        np.testing.assert_allclose(scalar_field(sim, 'magnetic'), abs(expected_b))
        np.testing.assert_array_equal(sim.bz, 0)  # Diagnostics cannot advance physics.

    def test_heatmap_zero_and_signed_colors(self):
        for dark, base in [(True, (10, 19, 29)), (False, (247, 250, 250))]:
            for mode in ('electric', 'magnetic', 'energy', 'signed'):
                values = np.array([[0., .5, 1.]])
                rgb = colorize(values, mode, field_scale(values, mode), base, dark)
                np.testing.assert_array_equal(rgb[0, 0], base)
                self.assertFalse(np.array_equal(rgb[0, 0], rgb[0, -1]))
                self.assertTrue(np.isfinite(colorize(values*0, mode,
                    field_scale(values*0, mode), base, dark)).all())
            colors = colorize(np.array([-1., 1.]), 'signed', 1., base, dark)
            self.assertFalse(np.array_equal(colors[0], colors[1]))


if __name__ == '__main__':
    unittest.main()
