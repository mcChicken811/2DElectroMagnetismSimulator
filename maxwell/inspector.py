"""Small numeric inspector, using the window's existing batched UI primitives."""
import math
import numpy as np
from .reference import prepare_uniform_motion


class ParticleInspector:
    names = ('vx', 'vy', 'ax', 'ay')

    def __init__(self, window):
        self.window = window
        self.selected = -1
        self.editor = None
        self.dirty = set()
        self.values = dict.fromkeys(self.names, '')
        self.labels = {}
        self.replace_text = False

    def select(self, index):
        self.selected = index
        self.cancel()

    def deleted(self, index):
        if index == self.selected:
            self.select(-1)
        elif index < self.selected:
            self.selected -= 1

    def cancel(self):
        self.editor = None
        self.dirty.clear()

    def valid(self):
        return 0 <= self.selected < len(self.window.sim.charge)

    def draw(self, sx, top):
        w, p = self.window, self.window.palette
        self.labels = {}
        # The color legend belongs to the Controls tab.
        w.heat_range_label = w.label('', -1000, 0)
        self.title = w.label('Click a particle to inspect', sx+10, top-77, 10, bold=True)
        self.state = w.label('', sx+10, top-101, 9, p['muted'])
        self.speed = w.label('', sx+10, top-122, 9, p['muted'])
        for row, names in enumerate((('vx', 'vy'), ('ax', 'ay'))):
            for col, name in enumerate(names):
                x, y = sx+8+104*col, top-187-67*row
                caption = name+' / c' if row == 0 else name+'  (dv'+name[1]+'/dt)'
                w.label(caption, x+4, y+40, 9, p['muted'])
                w.button('inspect_edit_'+name, '', x, y, 96, self.editor == name)
                self.labels[name] = w.buttons['inspect_edit_'+name][5]
        w.button('inspect_apply', 'Apply edits', sx+8, top-289, 96, bool(self.dirty))
        w.button('inspect_cancel', 'Discard', sx+112, top-289, 96)
        w.button('inspect_constraint', 'Constraint mode: OFF', sx+8, top-323, 200)
        w.button('inspect_fixed', 'Fix / release charge', sx+8, top-357, 200)
        w.button('inspect_prepare', 'Steady-motion fields', sx+8, top-397, 200)
        w.label('1 charge · resets E/B · holds v', sx+10, top-413, 8, p['muted'])
        w.label('Editing pauses. P resumes; N steps.', sx+10, top-436, 8, p['muted'])
        w.label('Enter applies · Esc discards', sx+10, top-452, 8, p['muted'])
        self.refresh()

    def refresh(self):
        if not self.labels or self.window.panel_tab != 'inspector':
            return
        w, sim, i = self.window, self.window.sim, self.selected
        if not self.valid():
            self.selected = -1
            self.title.text = 'Click a particle to inspect'
            self.state.text = self.speed.text = ''
            for label in self.labels.values():
                label.text = '—'
            return
        velocity, acceleration = sim.particle_state(i)
        speed = np.linalg.norm(velocity)/sim.config.c
        gamma = 1/math.sqrt(max(1-speed*speed, 1e-15))
        self.title.text = f'{("Charge", "Proton", "Electron")[sim.species[i]]} #{i+1}  ({"+" if sim.charge[i] > 0 else "−"})'
        self.state.text = ('FIXED · v = a = 0' if sim.fixed[i] else
                           'CONSTRAINED · prescribed a' if sim.prescribed[i] else 'FREE · Lorentz forces')
        self.speed.text = f'|v| = {speed:.5f}c    γ = {gamma:.5g}'
        for name, value in zip(self.names, (*(velocity/sim.config.c), *acceleration)):
            if name not in self.dirty:
                self.values[name] = f'{value:.7g}'
            label = self.labels[name]
            label.text = self.values[name]+('|' if name == self.editor else '')
            label.color = (*(w.palette['accent'] if name == self.editor else w.palette['text']), 255)
        if 'inspect_fixed' in w.buttons:
            w.buttons['inspect_fixed'][5].text = 'Release charge' if sim.fixed[i] else 'Fix charge'
            w.buttons['inspect_constraint'][5].text = f'Constraint mode: {"ON" if sim.prescribed[i] else "OFF"}'

    def focus(self, name):
        if not self.valid():
            self.window.notice('Select a charge, proton, or electron first.')
            return
        if name.startswith('a') and not self.window.sim.prescribed[self.selected]:
            self.window.notice('Enable Constraint Mode to prescribe acceleration; otherwise it follows the Lorentz force.')
            return
        self.refresh()
        self.editor, self.replace_text = name, True
        self.window.paused = True
        self.window.accumulator = 0
        self.window.sim.dragged = -1

    def text(self, text):
        if self.editor is None:
            return
        text = ''.join(c for c in text if c in '0123456789+-.eE')
        if not text:
            return
        old = '' if self.replace_text else self.values[self.editor]
        self.values[self.editor] = (old+text)[:14]
        self.dirty.add(self.editor)
        self.replace_text = False
        self.refresh()

    def backspace(self):
        if self.editor is not None:
            self.values[self.editor] = '' if self.replace_text else self.values[self.editor][:-1]
            self.dirty.add(self.editor)
            self.replace_text = False
            self.refresh()

    def apply(self):
        if not self.valid() or not self.dirty:
            self.editor = None
            return
        sim, i = self.window.sim, self.selected
        velocity, acceleration = sim.particle_state(i)
        edit_v = edit_a = False
        for name in self.dirty:
            try:
                value = float(self.values[name])
            except ValueError:
                raise ValueError('Enter a number, such as 0.5, -0.2, or 1e-3.') from None
            if not math.isfinite(value):
                raise ValueError('Motion values must be finite.')
            component = 0 if name[1] == 'x' else 1
            if name[0] == 'v':
                velocity[component] = value*sim.config.c
                edit_v = True
            else:
                acceleration[component] = value
                edit_a = True
        sim.set_motion(i, velocity if edit_v else None, acceleration if edit_a else None)
        self.cancel()
        self.window.notice('Motion updated; fields preserved. P resumes; N steps.'
            if not edit_a else 'Total acceleration prescribed; external control overrides Lorentz motion.')

    def action(self, name):
        w, sim = self.window, self.window.sim
        try:
            if name.startswith('inspect_edit_'):
                self.focus(name[len('inspect_edit_'):])
            elif name == 'inspect_cancel':
                self.cancel()
            elif not self.valid():
                w.notice('Select a charge, proton, or electron first.')
            elif name == 'inspect_apply':
                self.apply()
            elif name == 'inspect_constraint':
                enabled = not sim.prescribed[self.selected]
                sim.set_constraint(self.selected, enabled)
                self.cancel()
                w.notice('Constraint ON: zero acceleration holds the current velocity.' if enabled else
                         'Constraint OFF: Lorentz forces determine motion.')
            elif name == 'inspect_fixed':
                sim.toggle_fixed(self.selected)
                self.cancel()
            elif name == 'inspect_prepare':
                if self.dirty:
                    raise ValueError('Apply or discard motion edits before preparing fields.')
                prepare_uniform_motion(sim, self.selected)
                w.paused, w.accumulator = True, 0
                self.cancel()
                w.notice('Moving-source fields prepared; holds velocity. P runs; N steps.')
        except ValueError as error:
            w.notice(str(error))
