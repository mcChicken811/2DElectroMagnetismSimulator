#!/usr/bin/env python3
"""Run the interactive Maxwell field lab, or benchmark the numerical kernels."""
import argparse
import json
import platform
import time
from maxwell import ROOT
from maxwell.physics import Config, Simulation
from maxwell.fieldlines import build_lines


def grid_size(text):
    try:
        x, y = (int(part) for part in text.lower().split('x'))
        if not (16 <= x <= 1024 and 16 <= y <= 1024):
            raise ValueError
        return x, y
    except ValueError:
        raise argparse.ArgumentTypeError('Grid must look like 128x80; each dimension is 16–1024.')


def benchmark(sim, steps):
    import numpy as np
    # Warm every kernel before measuring (Numba compilation is a startup cost).
    sim.step(2)
    build_lines(sim)
    start = time.perf_counter()
    sim.step(steps)
    physics_s = time.perf_counter()-start
    samples, vertices = [], 0
    for _ in range(15):
        start = time.perf_counter()
        lines = build_lines(sim)
        samples.append((time.perf_counter()-start)*1000)
        vertices = len(lines)
    report = {
        'machine': platform.machine(), 'python': platform.python_version(),
        'grid': [sim.config.nx, sim.config.ny], 'charges': len(sim.charge),
        'field_padding': sim.config.field_padding,
        'field_grid': list(sim._bz.shape),
        'steps': steps, 'physics_ms_per_step': physics_s*1000/steps,
        'physics_realtime_capacity': steps*sim.dt/physics_s,
        'streamlines_median_ms': float(np.median(samples)),
        'streamlines_p95_ms': float(np.percentile(samples, 95)),
        'line_segments': vertices//2, 'diagnostics': sim.diagnostics(),
        'note': 'CPU kernel timing only; this does not measure GUI frame rate.'
    }
    print(json.dumps(report, indent=2))


def smoke_test(window):
    """Exercise real window handlers and both rendering themes; save artifacts."""
    import pyglet
    import numpy as np
    from pyglet.window import key, mouse
    from maxwell.physics import PROTON_ELECTRON_MASS_RATIO, SPECIES
    output = ROOT/'artifacts'
    output.mkdir(exist_ok=True)
    failures = []
    def press(symbol, modifiers=0):
        window.on_key_press(symbol, modifiers)
        window.on_key_release(symbol, modifiers)

    def double_click(x, y):
        for _ in range(2):
            window.on_mouse_press(x, y, mouse.LEFT, 0)
            window.on_mouse_release(x, y, mouse.LEFT, 0)

    def screenshot(name):
        window.switch_to()
        window.on_draw()
        pyglet.image.get_buffer_manager().get_color_buffer().save(str(output/name))

    def run(_):
        try:
            sim = window.sim
            window.action('pause')
            count = len(sim.charge)
            x, y = window.to_screen(6, 5)
            window.on_mouse_motion(x, y, 0, 0)
            window.on_key_press(key.SPACE, 0)
            window.on_key_press(key.SPACE, 0)
            assert len(sim.charge) == count+1, 'Space repeat should not add extra charges'
            window.on_key_release(key.SPACE, 0)
            np.testing.assert_allclose(sim.pos[-1], [6, 5])
            window.on_mouse_leave(x, y)
            window.on_key_press(key.SPACE, 0)
            window.on_key_release(key.SPACE, 0)
            assert len(sim.charge) == count+1, 'Space outside the window must not add a charge'
            window.on_mouse_enter(x, y)
            window.on_mouse_press(x, y, mouse.LEFT, 0)
            assert sim.dragged == count
            window.on_mouse_drag(x+65, y, 65, 0, mouse.LEFT, 0)
            for _ in range(30):
                sim.step()
            assert sim.pos[-1, 0] > 6.1, 'Drag spring should move the charge'
            window.on_mouse_release(x+65, y, mouse.LEFT, 0)
            assert sim.dragged == -1
            window.on_key_press(key.SPACE, key.MOD_SHIFT)
            window.on_key_release(key.SPACE, key.MOD_SHIFT)
            assert sim.charge[-1] < 0
            before_delete = [a.copy() for a in (sim._ex, sim._ey, sim._bz)]
            window.on_mouse_press(x+65, y, mouse.RIGHT, 0)
            assert len(sim.charge) == count+1
            for actual, expected in zip((sim._ex, sim._ey, sim._bz), before_delete):
                np.testing.assert_array_equal(actual, expected)
            window.action('proton')
            px, py = window.to_screen(11, 5)
            window.on_mouse_motion(px, py, 0, 0)
            press(key.SPACE)
            proton = len(sim.charge)-1
            assert sim.mass[proton] == PROTON_ELECTRON_MASS_RATIO
            assert sim.species[proton] == SPECIES['proton']
            window.action('electron')
            ex, ey = window.to_screen(8, 5)
            window.on_mouse_motion(ex, ey, 0, 0)
            press(key.SPACE)
            assert sim.mass[-1] == 1 and sim.charge[-1] < 0
            double_click(px, py)
            assert sim.fixed[proton]
            fixed_position = sim.pos[proton].copy()
            window.on_mouse_press(px, py, mouse.LEFT, 0)
            window.on_mouse_drag(px+30, py, 30, 0, mouse.LEFT, 0)
            sim.step(50)
            window.on_mouse_release(px+30, py, mouse.LEFT, 0)
            np.testing.assert_array_equal(sim.pos[proton], fixed_position)
            double_click(px, py)
            assert not sim.fixed[proton]
            sim.step(10)
            assert not np.array_equal(sim.pos[proton], fixed_position)
            window.action('velocity')
            window.action('density')
            window.action('reset')
            assert len(sim.charge) == 1 and sim.charge[0] > 0 and not sim.fixed[0]
            np.testing.assert_allclose(sim.pos[0], [sim.config.width/2, sim.config.height/2])
            assert sim.time == 0 and sim.dragged == -1
            screenshot('maxwell-reset.png')
            # Delete a stationary/fixed charge while paused: the marker goes,
            # but its full field survives until the user resumes or single-steps.
            cx, cy = window.to_screen(*sim.pos[0])
            double_click(cx, cy)
            assert sim.fixed[0]
            before_delete = [a.copy() for a in (sim._ex, sim._ey, sim._bz)]
            window.on_mouse_press(cx, cy, mouse.RIGHT, 0)
            assert len(sim.charge) == 0 and window.paused and sim.time == 0
            assert 'fields preserved' in window.message
            for actual, expected in zip((sim._ex, sim._ey, sim._bz), before_delete):
                np.testing.assert_array_equal(actual, expected)
            window.action('heat_electric')
            screenshot('maxwell-deletion-preserved.png')
            window.on_mouse_press(cx, cy, mouse.RIGHT, 0)  # Empty-space no-op.
            np.testing.assert_array_equal(sim._ex, before_delete[0])
            press(key.N)
            assert sim.steps == 1 and not np.array_equal(sim._ex, before_delete[0])
            sim.step(39)
            window.refresh_field()
            window.update_stats()
            screenshot('maxwell-deletion-wave.png')
            for symbol in (key.DELETE, key.BACKSPACE):
                sim.add_charge(6, 5)
                window.on_mouse_motion(*window.to_screen(6, 5), 0, 0)
                fields = [a.copy() for a in (sim._ex, sim._ey, sim._bz)]
                press(symbol)
                assert len(sim.charge) == 0
                for actual, expected in zip((sim._ex, sim._ey, sim._bz), fields):
                    np.testing.assert_array_equal(actual, expected)
            window.action('reset')
            assert not sim.removal_active and not np.any(sim._removal_psi)
            # Also exercise the running-state notice and scheduled field advance.
            sim.toggle_fixed(0)
            window.action('pause')
            window.on_mouse_press(cx, cy, mouse.RIGHT, 0)
            assert len(sim.charge) == 0 and 'travels outward' in window.message
            window.update(sim.dt*1.1)
            assert sim.steps > 0
            window.action('pause')
            window.action('reset')
            window.action('heat_electric')  # Restore the original color-off state.
            # Use one snapshot for comparing all four scalar views.
            proton = sim.add_charge(11, 5, species='proton')
            sim.add_charge(6, 4, species='electron')
            sim.toggle_fixed(proton)
            sim.step(60)
            window.sync_particles()
            for mode in ('electric', 'magnetic', 'energy', 'signed'):
                window.action('heat_'+mode)
                assert window.heat_mode == mode
                assert 'log / auto' in window.heat_range_label.text
                screenshot('maxwell-'+mode+'.png')
                if mode == 'electric':
                    screenshot('maxwell-dark.png')
            window.action('heat_energy')
            window.action('theme')
            screenshot('maxwell-light.png')
            window.set_size(1000, 760)
            window.dispatch_events()
            screenshot('maxwell-compact.png')
            assert sim.diagnostics()['gauss_linf'] < 1e-9
            print('GUI smoke test passed: add, key repeat, negative charge, drag, release, '
                  'field-preserving mouse/keyboard deletion, paused/running deletion, '
                  'single-charge reset, proton/electron masses, double-click pin/release, '
                  'all scalar layers, theme, resize, and screenshots.', flush=True)
        except Exception as error:
            failures.append(error)
            import traceback
            traceback.print_exc()
        finally:
            window.close()
            pyglet.app.exit()
    pyglet.clock.schedule_once(run, 1.)
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--charges', '-n', type=int, default=6, help='initial number of charges (default: 6)')
    parser.add_argument('--grid', type=grid_size, default=(128, 80), metavar='NXxNY')
    parser.add_argument('--field-padding', type=int, default=32, metavar='CELLS',
                        help='hidden field cells on each side (default: 32; minimum: 4)')
    parser.add_argument('--seed', type=int, default=7, help='repeatable initial scene')
    parser.add_argument('--theme', choices=('dark', 'light'), default='dark')
    parser.add_argument('--benchmark', action='store_true', help='measure kernels without opening a window')
    parser.add_argument('--steps', type=int, default=2000, help='benchmark timesteps')
    parser.add_argument('--smoke-test', action='store_true', help='test GUI interactions and save screenshots inside artifacts/')
    parser.add_argument('--profile-seconds', type=float, default=0,
                        help='measure GUI frame rate for this duration after warmup, then exit')
    args = parser.parse_args()
    if args.charges < 0 or args.steps < 1:
        parser.error('Charges must be nonnegative and steps must be positive.')
    if args.field_padding < 4:
        parser.error('Field padding must be at least 4 cells.')
    if not args.benchmark:
        from maxwell.macos import restrict_file_writes
        restrict_file_writes()
    sim = Simulation(Config(nx=args.grid[0], ny=args.grid[1],
                            field_padding=args.field_padding), args.charges, args.seed)
    if args.benchmark:
        benchmark(sim, args.steps)
        return
    print('Preparing numerical kernels (the first launch compiles a project-local cache)…', flush=True)
    sim.step()
    # Compile the optional deletion path before opening the window, avoiding a
    # first-use JIT pause when the user deletes a charge. Reset discards this step.
    sim.removal_active = True
    sim.step()
    build_lines(sim)
    sim.reset(args.charges, args.seed)
    from maxwell.app import FieldWindow
    import pyglet
    window = FieldWindow(sim, args.charges, args.seed, args.theme)
    print('Ready. Space: add · Shift+Space: negative · drag: force · P: pause · T: theme', flush=True)
    failures = smoke_test(window) if args.smoke_test else []
    if args.profile_seconds > 0:
        def begin_profile(_):
            window.record_frames = True
        def end_profile(_):
            import numpy as np
            intervals = np.diff(window.draw_times)
            if len(intervals):
                print(json.dumps({'rendered_frames': len(intervals)+1,
                    'average_fps': float(1/np.mean(intervals)),
                    'frame_ms_p95': float(np.percentile(intervals, 95)*1000),
                    'charges': len(sim.charge), 'grid': list(args.grid),
                    'window_points': list(window.get_size()),
                    'framebuffer_pixels': list(window.get_framebuffer_size())}, indent=2), flush=True)
            window.close()
            pyglet.app.exit()
        pyglet.clock.schedule_once(begin_profile, 2.)
        pyglet.clock.schedule_once(end_profile, 2.+args.profile_seconds)
    # Cocoa's repeating timer can arrive just BEFORE a 60 Hz clock deadline,
    # otherwise producing alternate-tick (30 Hz) rendering. Poll at 120 Hz;
    # window vsync still limits actual presentation to the display refresh.
    pyglet.app.run(interval=1/120)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
