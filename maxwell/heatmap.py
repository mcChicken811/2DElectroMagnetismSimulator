"""Read-only scalar diagnostics and color mapping; no changes to the solver."""
import numpy as np

HEAT_TITLES = {
    'off': 'ELECTRIC FIELD', 'electric': 'ELECTRIC FIELD MAGNITUDE  |E|',
    'magnetic': 'MAGNETIC FIELD MAGNITUDE  |B|',
    'energy': 'ELECTROMAGNETIC ENERGY DENSITY', 'signed': 'SIGNED MAGNETIC FIELD  Bz'
}
HEAT_UNITS = {'electric': '|E|', 'magnetic': '|B|', 'energy': 'u EM', 'signed': 'Bz'}


def scalar_field(sim, mode):
    """Collocate E and B at cell centers and at E's integer time.

    B is stored at t-dt/2. Faraday's law advances it half a step in a
    temporary array; this never alters the magnetic field used by the solver.
    """
    if mode not in HEAT_TITLES or mode == 'off':
        raise ValueError(f'Not a scalar field: {mode}')
    if mode in ('electric', 'energy'):
        ex = (sim.ex[:, :-1]+sim.ex[:, 1:])*0.5
        ey = (sim.ey[:-1, :]+sim.ey[1:, :])*0.5
        e2 = ex*ex+ey*ey
        if mode == 'electric':
            return np.sqrt(e2)
    curl = np.diff(sim.ey, axis=0)/sim.dx-np.diff(sim.ex, axis=1)/sim.dy
    b = sim.bz-0.5*sim.dt*curl
    if mode == 'signed':
        return b
    if mode == 'magnetic':
        return np.abs(b)
    return 0.5*(e2+sim.config.c**2*b*b)


def field_scale(values, mode):
    # Avoid visually amplifying roundoff in a physically zero magnetic field.
    floors = {'electric': 1e-6, 'magnetic': 1e-7, 'signed': 1e-7, 'energy': 1e-10}
    return max(float(np.max(np.abs(values), initial=0)), floors[mode])


def colorize(values, mode, scale, background, dark=True):
    """Continuous logarithmic color ramp, with an explicit displayed upper bound.

    The mapping uses log(1+50*value/scale)/log(51); each frame's maximum sets
    scale. Signed B uses the same magnitude mapping on either side of zero.
    """
    base = tuple(background)
    dark_ramps = {
        'electric': [base, (20, 67, 92), (25, 142, 145), (126, 207, 151), (255, 222, 114)],
        'magnetic': [base, (51, 48, 108), (111, 76, 173), (186, 138, 223), (242, 222, 255)],
        'energy': [base, (62, 39, 98), (151, 53, 114), (237, 116, 87), (255, 230, 158)]}
    light_ramps = {
        'electric': [base, (181, 224, 228), (75, 170, 179), (24, 114, 125), (16, 63, 91)],
        'magnetic': [base, (218, 207, 241), (172, 145, 214), (116, 73, 174), (58, 28, 103)],
        'energy': [base, (252, 222, 169), (241, 159, 105), (187, 71, 94), (85, 27, 88)]}
    t = np.log1p(50*np.clip(np.abs(values)/scale, 0, 1))/np.log(51)
    if mode == 'signed':
        plus = np.array((193, 147, 241) if dark else (119, 65, 182))
        minus = np.array((249, 182, 90) if dark else (206, 119, 28))
        target = np.where((values >= 0)[..., None], plus, minus)
        rgb = np.asarray(base)+t[..., None]*(target-np.asarray(base))
    else:
        ramp = np.asarray((dark_ramps if dark else light_ramps)[mode])
        stops = np.linspace(0, 1, len(ramp))
        rgb = np.stack([np.interp(t, stops, ramp[:, c]) for c in range(3)], axis=-1)
    return np.round(rgb).astype(np.uint8)
