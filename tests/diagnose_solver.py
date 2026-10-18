"""Reproduce the moving-charge wake: python tests/diagnose_solver.py.

Writes measurements and a three-panel heatmap inside this project's artifacts.
The PNG uses one common color scale: unfiltered / default filter / continuum.
"""
import json
import struct
import sys
import zlib
from pathlib import Path
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from maxwell.heatmap import colorize
from test_solver import moving_case


def save_png(path, rgb):
    """Encode a NumPy diagnostic raster with the standard-library PNG codec."""
    def chunk(name, data):
        return (struct.pack('!I', len(data))+name+data
                +struct.pack('!I', zlib.crc32(name+data) & 0xffffffff))
    header = struct.pack('!2I5B', rgb.shape[1], rgb.shape[0], 8, 2, 0, 0, 0)
    scanlines = b''.join(b'\x00'+row.tobytes() for row in rgb)
    path.write_bytes(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR', header)
                     +chunk(b'IDAT', zlib.compress(scanlines))+chunk(b'IEND', b''))


def main():
    output = PROJECT/'artifacts'
    output.mkdir(exist_ok=True)
    rows, snapshots = [], {}
    cases = [(passes, beta, 0., 128, 32, .75, duration)
             for passes in (0, 2) for beta in (.6, .8, .9, .99) for duration in (.4, .7)]
    cases += [(p, .8, angle, nx, pad, cfl, .4) for p in (0, 2) for angle, nx, pad, cfl in
              ((0., 128, 64, .75), (0., 128, 32, .375), (.45, 128, 32, .75),
               (.45, 256, 64, .75))]
    for passes, beta, angle, nx, padding, cfl, duration in cases:
        sim, actual, exact, errors, _ = moving_case(passes, beta, angle, padding, nx, cfl, duration)
        rows.append(dict(passes=passes, beta=beta, angle=angle, nx=nx, padding=padding,
                         cfl=cfl, time=sim.time, E_relative_L2=errors[0], B_relative_L2=errors[1],
                         gauss_linf=sim.diagnostics()['gauss_linf']))
        if (beta, angle, nx, padding, cfl, duration) == (.8, 0., 128, 32, .75, .4):
            snapshots[passes] = (np.hypot(*actual[:2]), np.hypot(*exact[:2]))
    arrays = [snapshots[0][0], snapshots[2][0], snapshots[2][1]]
    scale = max(float(a.max()) for a in arrays)
    panels = []
    for a in arrays:
        rgb = colorize(a, 'electric', scale, (10, 19, 29))
        panels.append(np.repeat(np.repeat(rgb.transpose(1, 0, 2)[::-1], 4, axis=0), 4, axis=1))
    gap = np.full((panels[0].shape[0], 12, 3), (137, 158, 178), dtype=np.uint8)
    save_png(output/'solver-comparison.png', np.concatenate([panels[0], gap, panels[1], gap, panels[2]], axis=1))
    (output/'solver-investigation.json').write_text(json.dumps(rows, indent=2)+'\n')
    print('PNG panels: unfiltered | two source-filter passes | continuum reference (shared color scale).')
    for row in rows:
        print(f"passes={row['passes']} beta={row['beta']:.2f} angle={row['angle']:.2f} "
              f"nx={row['nx']} padding={row['padding']} cfl={row['cfl']:.3f} t={row['time']:.5f} "
              f"E={row['E_relative_L2']:.3%} B={row['B_relative_L2']:.3%}")


if __name__ == '__main__':
    main()
