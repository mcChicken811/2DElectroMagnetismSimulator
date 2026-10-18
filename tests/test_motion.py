"""Inspector motion semantics and Lorentz-boosted line-source benchmarks."""
import unittest
import numpy as np
from maxwell.physics import Config, Simulation, AccelerationLimitError
from maxwell.reference import moving_line_field, prepare_uniform_motion


class MotionTests(unittest.TestCase):
    def small(self):
        sim = Simulation(Config(nx=48, ny=32, width=12, height=8), particles=0)
        sim.add_charge(6, 4)
        return sim

    def test_integer_velocity_and_relativistic_coordinate_acceleration(self):
        sim = self.small()
        sim._ex[:] = 3
        sim._ey[:] = -2
        sim._bz[:] = 0
        v = np.array([.8*sim.config.c, 0])
        sim.set_motion(0, velocity=v)
        actual_v, actual_a = sim.particle_state(0)
        gamma = 1/np.sqrt(1-.8**2)
        qm = sim.charge[0]/sim.mass[0]
        np.testing.assert_allclose(actual_v, v, atol=1e-14)
        np.testing.assert_allclose(actual_a, qm*np.array([3/gamma**3, -2/gamma]), atol=1e-14)
        # Nonzero magnetic force also needs projection perpendicular to v.
        sim._bz[:] = .7
        v = np.array([2., -3.])
        sim.set_motion(0, velocity=v)
        actual_v, actual_a = sim.particle_state(0)
        force = qm*np.array([3+v[1]*.7, -2-v[0]*.7])
        gamma = 1/np.sqrt(1-np.dot(v, v)/sim.config.c**2)
        np.testing.assert_allclose(actual_v, v, atol=1e-14)
        np.testing.assert_allclose(actual_a, (force-v*np.dot(v, force)/sim.config.c**2)/gamma)

    def test_edits_preserve_fields_position_and_time_and_reject_invalid_values(self):
        sim = self.small()
        sim.step(4)
        arrays = (sim._ex, sim._ey, sim._bz, sim.pos, sim._rho)
        before = [a.copy() for a in arrays]
        time = sim.time
        sim.set_motion(0, velocity=[2, 1])
        sim.set_constraint(0, True)
        sim.set_motion(0, acceleration=[.1, -.2])
        for actual, expected in zip(arrays, before):
            np.testing.assert_array_equal(actual, expected)
        self.assertEqual(sim.time, time)
        momentum = sim.momentum.copy()
        acceleration = sim.prescribed_acceleration.copy()
        for kwargs in (dict(velocity=[6, 0]), dict(velocity=[float('nan'), 0]),
                       dict(acceleration=[0, float('inf')]), dict(acceleration=[1e9, 0])):
            with self.assertRaises(ValueError):
                sim.set_motion(0, **kwargs)
            np.testing.assert_array_equal(sim.momentum, momentum)
            np.testing.assert_array_equal(sim.prescribed_acceleration, acceleration)

    def test_constraint_holds_total_acceleration_and_release_keeps_velocity(self):
        sim = self.small()
        sim.add_charge(9, 6, species='proton')  # Nonzero external EM forces.
        initial = sim.pos[0].copy()
        v, a = np.array([.7, -.3]), np.array([.2, .1])
        sim.set_motion(0, velocity=v, acceleration=a)
        sim.step(30)
        t = sim.time
        np.testing.assert_allclose(sim.pos[0], initial+v*t+.5*a*t*t, atol=2e-14)
        np.testing.assert_allclose(sim.particle_state(0)[0], v+a*t, atol=2e-14)
        np.testing.assert_allclose(sim.particle_state(0)[1], a)
        self.assertLess(sim.diagnostics()['gauss_linf'], 2e-11)
        sim.release_motion(0)
        self.assertFalse(sim.prescribed[0])
        np.testing.assert_allclose(sim.particle_state(0)[0], v+a*t, atol=2e-14)
        self.assertGreater(np.linalg.norm(sim.particle_state(0)[1]-a), .01)

    def test_constant_velocity_ignores_forces_and_reflects_only_normal_component(self):
        sim = self.small()
        sim.pos[0] = [12-sim.radius-.01, 4]
        sim.update_rho()
        sim.correct_gauss()
        sim.set_motion(0, velocity=[2, .3], acceleration=[0, 0])
        before = sim.pos[0].copy()
        sim.step()
        np.testing.assert_allclose(sim.particle_state(0)[0], [-2, .3], atol=1e-14)
        self.assertAlmostEqual(sim.pos[0, 1], before[1]+.3*sim.dt)
        self.assertLess(sim.diagnostics()['gauss_linf'], 2e-11)

    def test_fix_is_zero_velocity_zero_acceleration_constraint(self):
        sim = self.small()
        sim.set_motion(0, velocity=[1, 2], acceleration=[.2, -.1])
        self.assertTrue(sim.toggle_fixed(0))
        self.assertTrue(sim.prescribed[0])
        np.testing.assert_array_equal(sim.particle_state(0), np.zeros((2, 2)))
        position = sim.pos.copy()
        sim.step(8)
        np.testing.assert_array_equal(sim.pos, position)
        self.assertFalse(sim.toggle_fixed(0))
        self.assertFalse(sim.prescribed[0])
        np.testing.assert_allclose(sim.particle_state(0)[0], [0, 0], atol=1e-16)
        sim.toggle_fixed(0)
        sim.set_motion(0, velocity=[1, 0])  # Even pinned velocity remains editable.
        self.assertTrue(sim.prescribed[0])
        self.assertFalse(sim.fixed[0])
        sim.step()
        np.testing.assert_allclose(sim.particle_state(0)[0], [1, 0], atol=1e-14)

    def test_speed_limit_stops_before_mutating_a_step(self):
        sim = self.small()
        # The first step fits, the next would exceed c. No velocity clamping.
        sim.set_motion(0, velocity=[5, 0], acceleration=[.75/sim.dt, 0])
        sim.step()
        arrays = (sim._ex, sim._ey, sim._bz, sim.pos, sim.momentum)
        before = [a.copy() for a in arrays]
        time = sim.time
        with self.assertRaises(AccelerationLimitError):
            sim.step()
        for actual, expected in zip(arrays, before):
            np.testing.assert_array_equal(actual, expected)
        self.assertEqual(sim.time, time)

    def test_constraint_metadata_tracks_add_delete_and_reset(self):
        sim = self.small()
        sim.add_charge(8, 5, species='electron')
        sim.set_motion(1, velocity=[1, 0], acceleration=[.2, 0])
        sim.remove_charge(0)
        self.assertTrue(sim.prescribed[0])
        np.testing.assert_array_equal(sim.prescribed_acceleration[0], [.2, 0])
        sim.reset_to_single_positive()
        self.assertEqual(sim.prescribed.shape, (1,))
        self.assertEqual(sim.prescribed_acceleration.shape, (1, 2))
        self.assertFalse(sim.prescribed[0])
        self.assertFalse(sim.fixed[0])


class MovingFieldTests(unittest.TestCase):
    def test_reference_matches_independent_tensor_boost(self):
        # Boost rest-frame F^{mu nu}, with F^{0i}=-Ei/c and F^{ij}=-epsilon_ijk Bk.
        # Lab receives a source moving along an arbitrary in-plane direction.
        c, q = 6., -2.5
        velocity = np.array([3.6, 2.4])
        beta = velocity/c
        gamma = 1/np.sqrt(1-np.dot(beta, beta))
        boost = np.eye(4)
        boost[0, 0] = gamma
        boost[0, 1:3] = boost[1:3, 0] = gamma*beta
        boost[1:3, 1:3] += (gamma-1)*np.outer(beta, beta)/np.dot(beta, beta)
        for r in (np.array([2., .5]), np.array([-1., 3.]), np.array([-.4, -2.])):
            rest_r = r+(gamma-1)*np.dot(r, beta)*beta/np.dot(beta, beta)
            rest_e = q*rest_r/(2*np.pi*np.dot(rest_r, rest_r))
            tensor = np.zeros((4, 4))
            tensor[0, 1:3], tensor[1:3, 0] = -rest_e/c, rest_e/c
            lab = boost @ tensor @ boost.T
            expected = [-c*lab[0, 1], -c*lab[0, 2], -lab[1, 2]]
            actual = moving_line_field(*r, [0, 0], velocity, q, c)
            np.testing.assert_allclose(actual, expected, atol=1e-15)

    def test_2d_angular_ratio_and_nonrelativistic_limit(self):
        q, c = 2.5, 6.
        for beta in (0., .5, .9):
            parallel = moving_line_field(2., 0., [0, 0], [beta*c, 0], q, c)
            transverse = moving_line_field(0., 2., [0, 0], [beta*c, 0], q, c)
            self.assertAlmostEqual(transverse[1]/parallel[0], 1/(1-beta**2))
            self.assertAlmostEqual(transverse[2], beta/c*transverse[1])
        np.testing.assert_allclose(parallel[1:], [0, 0])

    def test_initializer_preserves_time_position_and_discrete_gauss(self):
        sim = Simulation(particles=0)
        sim.add_charge(7, 4)
        sim.step(3)
        sim.set_motion(0, velocity=[3.6, .4])
        position, time = sim.pos.copy(), sim.time
        prepare_uniform_motion(sim, 0)
        np.testing.assert_array_equal(sim.pos, position)
        self.assertEqual(sim.time, time)
        np.testing.assert_allclose(sim.particle_state(0)[0], [3.6, .4], atol=1e-14)
        self.assertTrue(sim.prescribed[0])
        self.assertFalse(sim.removal_active)
        self.assertLess(sim.diagnostics()['gauss_linf'], 2e-11)
        np.testing.assert_allclose(sim.magnetic_now(), sim.centered_b, atol=1e-16)
        sim.step(12)
        self.assertLess(sim.diagnostics()['gauss_linf'], 2e-11)
        sim.add_charge(10, 7)
        before = sim._ex.copy()
        with self.assertRaises(ValueError):
            prepare_uniform_motion(sim, 0)
        np.testing.assert_array_equal(sim._ex, before)

    def test_short_uniform_motion_field_error_away_from_source(self):
        errors = []
        for nx in (128, 256):
            # Retain the original unfiltered, short-time reference regression.
            # The stronger default-filter and longer-run cases live in test_solver.
            sim = Simulation(Config(nx=nx, ny=nx*5//8, field_padding=nx//4,
                                    source_filter_passes=0), particles=0)
            sim.add_charge(7.173, 4.619)
            sim.set_motion(0, velocity=[.6*sim.config.c, 0], acceleration=[0, 0])
            prepare_uniform_motion(sim, 0)
            sim.step(round(.1/sim.dt))
            x = (np.arange(sim.config.nx)+.5)[:, None]*sim.dx
            y = (np.arange(sim.config.ny)+.5)[None, :]*sim.dy
            exact = moving_line_field(x, y, sim.pos[0], sim.particle_state(0)[0],
                                      sim.charge[0], sim.config.c)
            actual = [.5*(sim.ex[:, 1:]+sim.ex[:, :-1]),
                      .5*(sim.ey[1:, :]+sim.ey[:-1, :]), sim.magnetic_now()]
            radius = np.hypot(x-sim.pos[0, 0], y-sim.pos[0, 1])
            mask = (radius > 1.5) & (radius < 3.)
            e_error = np.sqrt(sum(np.sum((a[mask]-b[mask])**2) for a, b in zip(actual[:2], exact[:2]))
                              /sum(np.sum(b[mask]**2) for b in exact[:2]))
            b_error = np.linalg.norm((actual[2]-exact[2])[mask])/np.linalg.norm(exact[2][mask])
            self.assertLess(e_error, .02)
            self.assertLess(b_error, .002)
            errors.append((e_error, b_error))
        self.assertLess(errors[1][0], .6*errors[0][0])
        self.assertLess(errors[1][1], .3*errors[0][1])


if __name__ == '__main__':
    unittest.main()
