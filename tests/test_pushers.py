"""Physical limiting solutions and accuracy of the active Higuera–Cary pusher."""
import unittest
import numpy as np
from maxwell.physics import Config, Simulation, boris, higuera_cary, push_particles


class HigueraCaryTests(unittest.TestCase):
    def test_relativistic_crossed_field_equilibrium(self):
        # For any charge sign, E=-v cross B makes the Lorentz force zero.
        # Resolve both in-plane components, relativistic speeds, and coarse dt.
        c = 6.
        worst = 0.
        for beta in (.5, .99, .9999):
            for angle in (.2, 1.3):
                v = beta*c*np.array([np.cos(angle), np.sin(angle)])
                initial = v/np.sqrt(1-beta*beta)
                for qm in (-1., 1.):
                    bz, dt = 3., 1.
                    ex, ey = -v[1]*bz, v[0]*bz
                    u = initial.copy()
                    for _ in range(1000):
                        u = np.array(higuera_cary(*u, ex, ey, bz, qm, dt, c))
                        worst = max(worst, np.linalg.norm(u-initial)/np.linalg.norm(initial))
                    np.testing.assert_allclose(u, initial, rtol=2e-11, atol=2e-11)
        self.assertLess(worst, 2e-11)
        # The comparison must actually distinguish HC from the old algorithm.
        v = .99*c
        u = np.array([v/np.sqrt(1-.99**2), 0.])
        old = np.array(boris(*u, 0., v*3, 3., 1., 1., c))
        self.assertGreater(np.linalg.norm(old-u)/np.linalg.norm(u), 1e-3)

    def test_magnetic_energy_and_rotation_sign(self):
        initial = np.array([8., -3.])
        u = initial.copy()
        self.assertLess(higuera_cary(1, 0, 0, 0, 1, 1, .01, 6)[1], 0)
        for _ in range(20000):
            u = higuera_cary(*u, 0, 0, 3, 1, .02, 6)
        self.assertAlmostEqual(np.linalg.norm(u), np.linalg.norm(initial), places=10)

    def test_exact_electric_and_manual_force_kicks(self):
        u = (0., 0.)
        for _ in range(100):
            u = higuera_cary(*u, 3, -2, 0, 2, .01, 6, 1., .5)
        np.testing.assert_allclose(u, [7., -3.5], atol=2e-13)
        # Neutral test particle: the external force is still force / mass.
        np.testing.assert_allclose(higuera_cary(0, 0, 3, 4, 100, 0, 2, 6, 1, 2), [2, 4])

    def test_frozen_field_time_reversal(self):
        rng = np.random.default_rng(3)
        for _ in range(100):
            u = rng.normal(size=2)*20
            ex, ey, bz = rng.normal(size=3)*5
            qm, dt = -.7, .2
            after = higuera_cary(*u, ex, ey, bz, qm, dt, 6)
            back = higuera_cary(*after, ex, ey, bz, qm, -dt, 6)
            np.testing.assert_allclose(back, u, rtol=3e-14, atol=3e-14)

    def test_second_order_against_analytic_magnetic_orbit(self):
        initial, c, bz, duration = np.array([5., 2.]), 6., 4., 1.6
        gamma = np.sqrt(1+np.dot(initial, initial)/c**2)
        phase = bz*duration/gamma
        exact = np.array([[np.cos(phase), np.sin(phase)],
                          [-np.sin(phase), np.cos(phase)]]) @ initial
        errors = []
        for dt in (.1, .05, .025):
            u = initial.copy()
            for _ in range(round(duration/dt)):
                u = higuera_cary(*u, 0, 0, bz, 1, dt, c)
            errors.append(np.linalg.norm(u-exact))
        self.assertTrue(all(3.9 < errors[i]/errors[i+1] < 4.1 for i in (0, 1)))

    def test_strong_magnetic_rotation_avoids_quadratic_cancellation(self):
        # The direct sigma+sqrt(sigma²+4tau²) expression rounds to zero here.
        u = np.array(higuera_cary(8, -3, 0, 0, 2e8, 1, 1, 6))
        self.assertTrue(np.all(np.isfinite(u)))
        np.testing.assert_allclose(np.linalg.norm(u), np.hypot(8, 3), rtol=2e-14)

    def test_active_particle_loop_and_initial_stagger_use_hc(self):
        sim = Simulation(Config(nx=32, ny=24, width=12, height=8), particles=1)
        sim.pos[0] = [6, 4]
        c, bz, vx = sim.config.c, 20., .99*sim.config.c
        initial = np.array([vx/np.sqrt(1-(vx/c)**2), 0.])
        sim.momentum[0] = initial
        sim.ex[:] = 0
        sim.ey[:] = vx*bz
        sim.bz[:] = bz
        sim._stagger_initial_momentum([0])
        np.testing.assert_allclose(sim.momentum[0], initial, atol=2e-13)
        # Isolate one push in imposed uniform fields, without evolving fields
        # or conflating the pusher test with self-fields and outgoing edges.
        dt = .15
        push_particles(sim.pos, sim.momentum, sim.charge, sim.mass, sim.fixed,
            sim.ex, sim.ey, sim.bz, sim.jx, sim.jy, sim.dx, sim.dy, dt, c,
            12, 8, sim.radius, -1, np.zeros(2))
        np.testing.assert_allclose(sim.momentum[0], initial, atol=2e-13)
        np.testing.assert_allclose(sim.pos[0], [6+vx*dt, 4], atol=2e-13)


if __name__ == '__main__':
    unittest.main()
