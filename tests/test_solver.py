"""Yee propagation and conservative source-filter regressions."""
import unittest
import numpy as np
from maxwell.physics import Config, Simulation
from maxwell.filtering import binomial_filter
from maxwell.reference import moving_line_field, prepare_uniform_motion


def moving_case(passes=2, beta=.8, angle=0., padding=32, nx=128, cfl=.75, duration=.4):
    sim = Simulation(Config(nx=nx, ny=nx*5//8, field_padding=padding,
                            source_filter_passes=passes, cfl=cfl), particles=0)
    sim.add_charge(8, 5)
    velocity = beta*sim.config.c*np.array([np.cos(angle), np.sin(angle)])
    sim.set_motion(0, velocity=velocity, acceleration=[0, 0])
    prepare_uniform_motion(sim, 0)
    sim.step(round(duration/sim.dt))
    x = (np.arange(sim.config.nx)+.5)[:, None]*sim.dx
    y = (np.arange(sim.config.ny)+.5)[None, :]*sim.dy
    exact = moving_line_field(x, y, sim.pos[0], velocity, sim.charge[0], sim.config.c)
    actual = (.5*(sim.ex[:, 1:]+sim.ex[:, :-1]),
              .5*(sim.ey[1:, :]+sim.ey[:-1, :]), sim.magnetic_now())
    r = np.hypot(x-sim.pos[0, 0], y-sim.pos[0, 1])
    mask = (r > .75) & (r < 3.)
    errors = []
    for aa, bb in ((actual[:2], exact[:2]), (actual[2:], exact[2:])):
        errors.append(float(np.sqrt(sum(np.sum((a[mask]-b[mask])**2) for a, b in zip(aa, bb))
                                    /sum(np.sum(b[mask]**2) for b in bb))))
    return sim, actual, exact, errors, mask


class SourceFilterTests(unittest.TestCase):
    def test_filter_normalization_center_variance_and_adjoint(self):
        impulse = np.zeros((25, 21))
        impulse[12, 10] = 1
        for passes in (0, 1, 2, 4):
            filtered = binomial_filter(impulse, passes)
            self.assertAlmostEqual(filtered.sum(), 1)
            self.assertTrue(np.all(filtered >= 0))
            x, y = np.arange(25)[:, None]-12, np.arange(21)[None, :]-10
            self.assertAlmostEqual(np.sum(x*filtered), 0)
            self.assertAlmostEqual(np.sum(y*filtered), 0)
            self.assertAlmostEqual(np.sum(x*x*filtered), passes/2)
            self.assertAlmostEqual(np.sum(y*y*filtered), passes/2)
        field = np.random.default_rng(7).normal(size=impulse.shape)
        before = field.copy()
        np.testing.assert_allclose(np.sum(field*binomial_filter(impulse, 2)),
                                   np.sum(impulse*binomial_filter(field, 2)), atol=1e-16)
        np.testing.assert_array_equal(field, before)
        self.assertEqual(np.count_nonzero(impulse), 1)

    def test_filtered_charge_current_continuity_at_walls_and_grid_crossings(self):
        for passes in (0, 2, 4):
            sim = Simulation(Config(nx=48, ny=32, width=12, height=8,
                                    field_padding=4, source_filter_passes=passes), particles=0)
            for x, y, vx, vy, sign in ((.4, .4, -3., -2., 1),
                                       (11.6, 7.6, 2., 4., -1)):
                i = sim.add_charge(x, y, sign)
                sim.set_motion(i, velocity=[vx, vy], acceleration=[0, 0])
            for _ in range(50):
                sim.update_rho()
                old = sim._rho.copy()
                sim.step()
                sim.update_rho()
                divj = (np.diff(sim._jx[:, 1:-1], axis=0)/sim.dx
                        +np.diff(sim._jy[1:-1, :], axis=1)/sim.dy)
                np.testing.assert_allclose((sim._rho-old)[1:-1, 1:-1]+sim.dt*divj,
                                           0, atol=3e-12)
                self.assertAlmostEqual(np.sum(sim._rho)*sim.dx*sim.dy, sum(sim.charge), places=12)
                self.assertAlmostEqual(np.sum(abs(sim._rho))*sim.dx*sim.dy,
                                       sum(abs(sim.charge)), places=12)
            self.assertLess(sim.diagnostics()['gauss_linf'], 2e-11)

    def test_initialization_addition_deletion_and_gather_preserve_fields(self):
        sim = Simulation(Config(field_padding=4), particles=0)
        i = sim.add_charge(sim.radius, sim.radius)
        sim.set_motion(i, velocity=[.3, .5], acceleration=[0, 0])
        self.assertGreater(np.sum(abs(sim._rho[:4, :])), 0)  # Real hidden-cell tails.
        self.assertAlmostEqual(sim._rho.sum()*sim.dx*sim.dy, sim.charge.sum(), places=12)
        sim.step(20)
        sim.add_charge(12, 7, -1)
        self.assertLess(sim.diagnostics()['gauss_linf'], 2e-11)
        arrays = [a.copy() for a in (sim._ex, sim._ey, sim._bz)]
        sim.force_fields()
        sim.remove_charge(0)
        for a, b in zip((sim._ex, sim._ey, sim._bz), arrays):
            np.testing.assert_array_equal(a, b)
        self.assertTrue(sim.removal_active)
        sim.update_rho()
        self.assertAlmostEqual(sim._rho.sum()*sim.dx*sim.dy, sim.charge.sum(), places=12)

    def test_moving_charge_wake_reduction_at_point_eight_c(self):
        old = moving_case(passes=0)[3]
        sim, _, _, new, _ = moving_case()
        # A longer test that actually samples the numerical wake; t=.1 at
        # r>1.5 cannot see the high-k transient from the initial source core.
        self.assertGreater(old[0], .4)
        self.assertGreater(old[1], 1.)
        self.assertLess(new[0], .03)
        self.assertLess(new[1], .04)
        self.assertLess(new[0], .06*old[0])
        self.assertLess(new[1], .03*old[1])
        self.assertLess(sim.diagnostics()['gauss_linf'], 2e-11)
        np.testing.assert_allclose(sim.particle_state(0)[0], [4.8, 0], atol=1e-13)

    def test_wake_is_present_before_outer_boundary_signals_return(self):
        for passes in (0, 2):
            _, small, _, _, mask = moving_case(passes=passes, padding=32)
            _, large, _, _, _ = moving_case(passes=passes, padding=64)
            for a, b in zip(small, large):
                np.testing.assert_allclose(a[mask], b[mask], atol=2e-5, rtol=0)

    def test_oblique_motion_and_stronger_filter_option(self):
        for nx in (128, 256):
            old = moving_case(passes=0, angle=.45, nx=nx, padding=nx//4)[3]
            filtered = moving_case(angle=.45, nx=nx, padding=nx//4)[3]
            stronger = moving_case(passes=4, angle=.45, nx=nx, padding=nx//4)[3]
            self.assertLess(filtered[0], .07)
            self.assertLess(filtered[1], .11)
            self.assertLess(filtered[0], old[0]*.12)
            self.assertLess(filtered[1], old[1]*.08)
            self.assertLess(stronger[0], filtered[0])
            self.assertLess(stronger[1], filtered[1])
        # Refining at fixed pass count also shrinks the physical source core;
        # it is not a fixed-smooth-source convergence experiment.

    def test_source_free_yee_mode_matches_discrete_dispersion(self):
        # Traveling Ey/Bz plane wave on an interior strip, with the known Yee
        # frequency and B half-time. Filtering sources must not damp a wave.
        sim = Simulation(Config(nx=128, ny=80), particles=0)
        k = 2*np.pi/(8*sim.dx)
        omega = 2/sim.dt*np.arcsin(sim.config.c*sim.dt/sim.dx*np.sin(k*sim.dx/2))
        x_e = np.arange(sim._ey.shape[0])*sim.dx
        x_b = (np.arange(sim._bz.shape[0])+.5)*sim.dx
        sim._ey[:] = np.cos(k*x_e)[:, None]
        sim._bz[:] = np.cos(k*x_b+omega*sim.dt/2)[:, None]/sim.config.c
        sim.step(8)
        expected = np.cos(k*x_e-omega*8*sim.dt)
        np.testing.assert_allclose(sim._ey[20:-20, 30], expected[20:-20], atol=2e-14)


if __name__ == '__main__':
    unittest.main()
