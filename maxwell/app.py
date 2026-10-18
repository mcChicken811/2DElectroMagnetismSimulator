"""Native pyglet / OpenGL front end. All physics remains in physics.py."""
import math
import sys
import time
import numpy as np
import pyglet
from .macos import prepare_cocoa, disable_restoration
prepare_cocoa()
# Use Cocoa's native event loop on Apple Silicon (pyglet's recommended ARM path).
pyglet.options.osx_alt_loop = True
pyglet.options.dpi_scaling = 'stretch'  # Logical points; keep Retina text/UI legible.
from pyglet import gl, shapes
disable_restoration()
from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet.window import key, mouse
from .fieldlines import build_lines
from .heatmap import HEAT_TITLES, HEAT_UNITS, scalar_field, field_scale, colorize
from .physics import SPECIES, PROTON_ELECTRON_MASS_RATIO


THEMES = {
    'dark': dict(bg=(13, 19, 29), panel=(19, 28, 41), plot=(10, 19, 29),
                 border=(40, 55, 71), text=(233, 241, 247), muted=(137, 158, 178),
                 accent=(99, 224, 195), positive=(255, 157, 117), negative=(117, 177, 255),
                 button=(29, 43, 58), hover=(42, 61, 78), line=(106, 213, 196)),
    'light': dict(bg=(243, 245, 247), panel=(255, 255, 255), plot=(247, 250, 250),
                  border=(208, 218, 226), text=(28, 47, 63), muted=(96, 116, 131),
                  accent=(0, 126, 113), positive=(206, 86, 41), negative=(40, 103, 202),
                  button=(235, 240, 244), hover=(218, 231, 236), line=(19, 132, 122))
}

VERTEX = """#version 150 core
in vec3 position;
out float strength;
uniform vec2 origin;
uniform float scale;
uniform WindowBlock { mat4 projection; mat4 view; } window;
void main() {
    gl_Position = window.projection * window.view * vec4(origin + scale*position.xy, 0., 1.);
    strength = position.z;
}
"""
FRAGMENT = """#version 150 core
in float strength;
out vec4 final_color;
uniform vec4 tint;
void main() {
    float a = 0.50 + 0.50 * (1.0-exp(-strength*0.9));
    final_color = vec4(tint.rgb, tint.a*a);
}
"""


class FieldWindow(pyglet.window.Window):
    def __init__(self, sim, initial_count=6, seed=7, theme='dark', visible=True):
        self.ready = False
        config = gl.Config(double_buffer=True, major_version=3, minor_version=3,
                           sample_buffers=1, samples=4)
        try:
            super().__init__(1280, 860, caption="Maxwell · Field Lab", resizable=True,
                             vsync=True, config=config, visible=visible)
        except pyglet.window.NoSuchConfigException:
            super().__init__(1280, 860, caption="Maxwell · Field Lab", resizable=True,
                             vsync=True, visible=visible)
        self.set_minimum_size(980, 760)
        if sys.platform == 'darwin':
            self._nswindow.setRestorable_(False)
            self._nswindow.disableSnapshotRestoration()
        self.sim, self.initial_count, self.seed = sim, initial_count, seed
        self.theme = theme
        self.paused = False
        self.show_lines, self.show_velocity = True, False
        self.heat_mode = 'off'
        self.injection_species = 'charge'
        self.sign, self.speed, self.density = 1, 1., 1
        self.cursor = (-100., -100.)
        self.hovered = -1
        self.keys_down = set()
        self.last_click = None
        self.accumulator, self.line_age, self.stats_age = 0., 1., 1.
        self.frame_ms, self.physics_ms, self.lines_ms = 16.67, 0., 0.
        self.last_draw = time.perf_counter()
        self.draw_times = []
        self.record_frames = False
        self.slow = False
        self.line_count = 0
        self.message = "Drag to apply force. Double-click a charge to fix or release it."
        self.message_until = 0.
        self.program = ShaderProgram(Shader(VERTEX, 'vertex'), Shader(FRAGMENT, 'fragment'))
        self.line_vertices = None
        self.vector_vertices = None
        self.line_data = np.empty((0, 3), np.float32)
        self.particle_batch = pyglet.graphics.Batch()
        self.particle_drawables = []
        self.particle_groups = [pyglet.graphics.Group(order=i) for i in range(3)]
        self.chrome_items, self.grid_items = [], []
        self.heat_texture = pyglet.image.Texture.create(sim.config.nx, sim.config.ny)
        gl.glBindTexture(self.heat_texture.target, self.heat_texture.id)
        gl.glTexParameteri(self.heat_texture.target, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(self.heat_texture.target, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(self.heat_texture.target, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        self.heat_sprite = pyglet.sprite.Sprite(self.heat_texture)
        self.ready = True
        self.layout()
        self.refresh_field()
        self.sync_particles()
        pyglet.clock.schedule(self.update)

    @property
    def palette(self):
        return THEMES[self.theme]

    def label(self, text, x, y, size=11, color=None, bold=False, anchor='left', batch=None):
        obj = pyglet.text.Label(text, font_name='Helvetica Neue', font_size=size,
                                weight='bold' if bold else 'normal', x=x, y=y,
                                anchor_x=anchor, anchor_y='center',
                                color=(*(color or self.palette['text']), 255),
                                batch=batch or self.chrome, group=self.text_group)
        self.chrome_items.append(obj)
        return obj

    def rect(self, x, y, w, h, color, radius=8, batch=None, group=None):
        obj = shapes.RoundedRectangle(x, y, w, h, radius=radius, segments=8, color=color,
                                      batch=batch or self.chrome, group=group or self.bg_group)
        self.chrome_items.append(obj)
        return obj

    def button(self, name, text, x, y, w, active=False, height=28):
        p = self.palette
        rect = self.rect(x, y, w, height, p['button'])
        label = self.label(text, x+w/2, y+height/2, 10, p['accent'] if active else p['text'],
                           bold=active, anchor='center')
        self.buttons[name] = (x, y, w, height, rect, label)

    def layout(self):
        for obj in self.chrome_items + self.grid_items:
            obj.delete()
        self.chrome_items, self.grid_items, self.buttons = [], [], {}
        self.chrome, self.grid_batch = pyglet.graphics.Batch(), pyglet.graphics.Batch()
        self.bg_group, self.text_group = pyglet.graphics.Group(order=0), pyglet.graphics.Group(order=1)
        p, w, h = self.palette, self.width, self.height
        cfg = self.sim.config
        self.world_scale = min((w-300)/cfg.width, (h-242)/cfg.height)
        self.plot_w, self.plot_h = self.world_scale*cfg.width, self.world_scale*cfg.height
        self.ox, self.oy = 28+(w-300-self.plot_w)/2, 116+(h-242-self.plot_h)/2
        self.heat_sprite.x, self.heat_sprite.y = self.ox, self.oy
        self.heat_sprite.scale_x = self.plot_w/cfg.nx
        self.heat_sprite.scale_y = self.plot_h/cfg.ny
        self.label('M A X W E L L', 28, h-43, 21, bold=True)
        self.label('FIELD LAB  /  2D ELECTRODYNAMICS', 30, h-74, 9, p['muted'])
        self.label('Interact with charges. Watch the field respond.', 310, h-45, 12, p['muted'])
        self.label('APPEARANCE', w-222, h-44, 8, p['muted'])
        self.button('theme', 'LIGHT' if self.theme == 'dark' else 'DARK', w-128, h-61, 98, height=34)
        self.rect(w-116, h-49, 22, 10, p['accent'], radius=5, group=self.text_group)
        knob = shapes.Circle(w-100 if self.theme == 'dark' else w-111, h-44, 3,
                              color=p['panel'], batch=self.chrome, group=pyglet.graphics.Group(order=2))
        self.chrome_items.append(knob)
        # Offset the theme caption to make room for the switch graphic.
        self.buttons['theme'][5].x += 14
        self.label(HEAT_TITLES[self.heat_mode], self.ox, self.oy+self.plot_h+20, 9, p['muted'], bold=True)
        self.run_label = self.label('RUNNING', self.ox+self.plot_w, self.oy+self.plot_h+20,
                                     9, p['accent'], anchor='right')
        self.rect(self.ox-1, self.oy-1, self.plot_w+2, self.plot_h+2, p['border'], radius=1)
        # The interior is covered by the field texture, which is drawn afterward.
        for i in range(17):
            x = self.ox+self.plot_w*i/16
            self.grid_items.append(shapes.Line(x, self.oy, x, self.oy+self.plot_h,
                                    thickness=.6, color=(*p['border'], 90), batch=self.grid_batch))
        for j in range(11):
            y = self.oy+self.plot_h*j/10
            self.grid_items.append(shapes.Line(self.ox, y, self.ox+self.plot_w, y,
                                    thickness=.6, color=(*p['border'], 90), batch=self.grid_batch))
        self.label('OPEN EM SPACE  /  PARTICLES REFLECT', self.ox, self.oy-17, 8, p['muted'])
        self.label(f'{cfg.width:g} × {cfg.height:g}  normalized units', self.ox+self.plot_w,
                   self.oy-17, 8, p['muted'], anchor='right')
        sx, top = w-244, h-126
        self.rect(sx-8, 108, 224, h-224, p['panel'], radius=12)
        self.label('CONTROLS', sx+10, top-13, 9, p['muted'], bold=True)
        self.button('pause', 'Resume  [P]' if self.paused else 'Pause  [P]', sx+8, top-52, 96)
        self.button('reset', 'Reset  [R]', sx+112, top-52, 96)
        self.label('ADD AT CURSOR  ·  SPACE', sx+10, top-72, 9, p['muted'])
        generic = self.injection_species == 'charge'
        self.button('positive', '+ Charge [1]', sx+8, top-108, 96, generic and self.sign > 0)
        self.button('negative', '− Charge [2]', sx+112, top-108, 96, generic and self.sign < 0)
        self.button('proton', 'Proton [3]', sx+8, top-140, 96, self.injection_species == 'proton')
        self.button('electron', 'Electron [4]', sx+112, top-140, 96, self.injection_species == 'electron')
        self.label('MASS UNIT: electron mass', sx+10, top-154, 8, p['muted'])
        self.label('FIELD LAYERS', sx+10, top-177, 9, p['muted'])
        self.button('lines', 'Lines [E]', sx+8, top-214, 96, self.show_lines)
        self.button('velocity', 'Velocity [V]', sx+112, top-214, 96, self.show_velocity)
        self.label('CANVAS COLOR', sx+10, top-235, 9, p['muted'])
        self.button('heat_off', 'Off', sx+8, top-271, 96, self.heat_mode == 'off')
        self.button('heat_electric', '|E| [F]', sx+112, top-271, 96, self.heat_mode == 'electric')
        self.button('heat_magnetic', '|B| [B]', sx+8, top-303, 96, self.heat_mode == 'magnetic')
        self.button('heat_energy', 'EM energy [U]', sx+112, top-303, 96, self.heat_mode == 'energy')
        self.button('heat_signed', 'Signed Bz [M]', sx+8, top-335, 200, self.heat_mode == 'signed')
        if self.heat_mode != 'off':
            ramp = (51**np.linspace(0, 1, 64)-1)/50
            if self.heat_mode == 'signed':
                ramp = np.concatenate((-ramp[::-1], ramp))
            colors = colorize(ramp, self.heat_mode, 1., p['plot'], self.theme == 'dark')
            for i, color in enumerate(colors):
                self.chrome_items.append(shapes.Rectangle(sx+8+i*200/len(colors), top-350,
                    200/len(colors)+.1, 6, color=tuple(int(c) for c in color), batch=self.chrome,
                    group=self.bg_group))
        self.heat_range_label = self.label('', sx+10, top-363, 8, p['muted'])
        self.label('TIME SCALE', sx+10, top-383, 9, p['muted'])
        for i, speed in enumerate((.5, 1., 2.)):
            self.button(f'speed{speed}', f'{speed:g}×', sx+8+i*69, top-419, 62, self.speed == speed)
        self.button('density', f'Detail: {("low", "balanced", "high")[self.density]}   [D]', sx+8, top-451, 200)
        self.stats_label = self.label('', sx+10, top-473, 9, p['muted'])
        self.time_label = self.label('', sx+10, top-493, 9, p['muted'])
        self.error_label = self.label('', sx+10, top-513, 9, p['muted'])
        # Compact bottom bar remains legible at the minimum supported window size.
        self.rect(28, 23, w-56, 62, p['panel'], radius=10)
        self.message_label = self.label(self.message, 44, 62, 11)
        self.label('DRAG  force    DOUBLE CLICK  fix/free    SPACE  add    RIGHT CLICK  remove',
                   44, 39, 9, p['muted'])
        self.fps_label = self.label('', w-44, 62, 11, p['accent'], anchor='right')
        self.label(f'Yee {cfg.nx} × {cfg.ny}  ·  Boris  ·  RK4', w-44, 39, 9, p['muted'], anchor='right')
        self.update_stats()
        self.refresh_heat()

    def refresh_heat(self):
        p, sim = self.palette, self.sim
        rgb = np.broadcast_to(p['plot'], (*sim.bz.shape, 3))
        if self.heat_mode != 'off':
            values = scalar_field(sim, self.heat_mode)
            scale = field_scale(values, self.heat_mode)
            rgb = colorize(values, self.heat_mode, scale, p['plot'], self.theme == 'dark')
            bounds = f'±{scale:.2g}' if self.heat_mode == 'signed' else f'0–{scale:.2g}'
            self.heat_range_label.text = f'{bounds} {HEAT_UNITS[self.heat_mode]}  ·  log / auto'
        else:
            self.heat_range_label.text = 'Color scale: normalized field units'
        rgba = np.empty((*sim.bz.shape, 4), dtype=np.uint8)
        rgba[..., :3], rgba[..., 3] = rgb.astype(np.uint8), 255
        data = np.ascontiguousarray(rgba.transpose(1, 0, 2)).tobytes()
        img = pyglet.image.ImageData(sim.config.nx, sim.config.ny, 'RGBA', data)
        self.heat_texture.blit_into(img, 0, 0, 0)

    def refresh_field(self):
        start = time.perf_counter()
        if self.show_lines:
            self.line_data = build_lines(self.sim, self.density)
            # Triangle ribbons give consistent 1.1-point lines on Retina. Core
            # OpenGL wide GL_LINES are not reliably available on macOS.
            segments = self.line_data.reshape(-1, 2, 3)
            delta = segments[:, 1, :2]-segments[:, 0, :2]
            normal = np.column_stack((-delta[:, 1], delta[:, 0]))
            normal *= (.55/self.world_scale)/np.maximum(np.linalg.norm(delta, axis=1), 1e-12)[:, None]
            a, b = segments[:, 0].copy(), segments[:, 0].copy()
            c, d = segments[:, 1].copy(), segments[:, 1].copy()
            a[:, :2] -= normal
            b[:, :2] += normal
            c[:, :2] -= normal
            d[:, :2] += normal
            triangles = np.stack((a, b, c, b, d, c), axis=1).reshape(-1, 3)
            count = len(triangles)
            if count:
                if self.line_vertices is None:
                    self.line_vertices = self.program.vertex_list(count, gl.GL_TRIANGLES,
                        position=('f', triangles.ravel()))
                else:
                    if self.line_vertices.count != count:
                        self.line_vertices.resize(count)
                    self.line_vertices.position[:] = triangles.ravel()
            self.line_count = count
        self.refresh_heat()
        self.lines_ms = (time.perf_counter()-start)*1000
        self.line_age = 0

    def sync_particles(self):
        sim, p = self.sim, self.palette
        if len(self.particle_drawables) != len(sim.charge):
            for objects in self.particle_drawables:
                for obj in objects:
                    obj.delete()
            self.particle_drawables = []
            for _ in sim.charge:
                self.particle_drawables.append([
                    shapes.Circle(0, 0, 19, segments=32, batch=self.particle_batch, group=self.particle_groups[0]),
                    shapes.Circle(0, 0, 11, segments=32, batch=self.particle_batch, group=self.particle_groups[1]),
                    shapes.Line(0, 0, 1, 1, thickness=1.6, batch=self.particle_batch, group=self.particle_groups[2]),
                    shapes.Line(0, 0, 1, 1, thickness=1.6, batch=self.particle_batch, group=self.particle_groups[2]),
                    shapes.Arc(0, 0, 15, segments=32, thickness=1.5,
                               batch=self.particle_batch, group=self.particle_groups[1]),
                    pyglet.text.Label('', font_name='Helvetica Neue', font_size=8, weight='bold',
                                      batch=self.particle_batch, group=self.particle_groups[2])])
        for i, objects in enumerate(self.particle_drawables):
            x, y = self.to_screen(*sim.pos[i])
            color = p['positive'] if sim.charge[i] > 0 else p['negative']
            halo, dot, horizontal, vertical, ring, species_label = objects
            halo.position = dot.position = x, y
            halo.color = (*color, 65 if i == self.hovered or i == sim.dragged else 23)
            dot.color = color
            dot.radius = max(6., min(10., sim.radius*self.world_scale*.7))
            horizontal.position, horizontal.x2, horizontal.y2 = (x-3, y), x+3, y
            vertical.position, vertical.x2, vertical.y2 = (x, y-3), x, y+3
            horizontal.color = vertical.color = p['plot']
            vertical.visible = sim.charge[i] > 0
            if ring.visible != bool(sim.fixed[i]):
                ring.visible = bool(sim.fixed[i])
            if sim.fixed[i]:
                ring.position, ring.color = (x, y), p['accent']
            species_label.text = ('', 'p', 'e')[sim.species[i]]
            if sim.species[i] != SPECIES['charge']:
                species_label.x, species_label.y = x+12, y+10
                species_label.color = (*p['text'], 255)

    def to_screen(self, x, y):
        return self.ox+x*self.world_scale, self.oy+y*self.world_scale

    def world(self, x, y):
        return (x-self.ox)/self.world_scale, (y-self.oy)/self.world_scale

    def in_plot(self, x, y):
        return self.ox <= x <= self.ox+self.plot_w and self.oy <= y <= self.oy+self.plot_h

    def pick(self, x, y):
        if not self.in_plot(x, y) or not len(self.sim.charge):
            return -1
        distance = np.linalg.norm((self.sim.pos-np.array(self.world(x, y)))*self.world_scale, axis=1)
        i = int(np.argmin(distance))
        return i if distance[i] < 22 else -1

    def notice(self, text):
        self.message, self.message_until = text, time.perf_counter()+5
        self.message_label.text = text

    def delete_at_cursor(self):
        """Share field-preserving deletion across mouse and keyboard controls."""
        i = self.pick(*self.cursor)
        if i < 0:
            return
        self.sim.remove_charge(i)
        self.last_click = None
        self.hovered = -1
        self.refresh_field()
        self.sync_particles()
        self.update_stats()
        self.notice('Charge removed; fields preserved. Press P or N to evolve them.'
                    if self.paused else
                    'Charge removed; its field adjustment travels outward.')

    def action(self, name):
        self.last_click = None
        if name == 'theme':
            self.theme = 'light' if self.theme == 'dark' else 'dark'
        elif name == 'pause':
            self.paused = not self.paused
            self.accumulator = 0
        elif name == 'reset':
            self.sim.reset_to_single_positive()
            self.accumulator = 0
            self.injection_species, self.sign = 'charge', 1
            self.notice('Reset: one free positive charge at the center.')
        elif name in ('positive', 'negative'):
            self.sign = 1 if name == 'positive' else -1
            self.injection_species = 'charge'
        elif name in ('proton', 'electron'):
            self.injection_species = name
            self.sign = 1 if name == 'proton' else -1
            mass = PROTON_ELECTRON_MASS_RATIO if name == 'proton' else 1
            self.notice(f'{name.title()} selected: mass {mass:.5g} electron masses. Press Space to add.')
        elif name == 'lines':
            self.show_lines = not self.show_lines
        elif name.startswith('heat_'):
            selected = name[5:]
            self.heat_mode = 'off' if selected == self.heat_mode else selected
            if self.heat_mode == 'signed':
                self.notice('Signed Bz: violet points out of the screen; amber points into it.')
            elif self.heat_mode == 'energy':
                self.notice('EM energy density = (|E|² + c²|B|²) / 2. The legend shows the color range.')
        elif name == 'velocity':
            self.show_velocity = not self.show_velocity
        elif name == 'density':
            self.density = (self.density+1) % 3
        elif name.startswith('speed'):
            self.speed = float(name[5:])
        self.layout()
        self.refresh_field()
        self.sync_particles()

    def on_resize(self, width, height):
        super().on_resize(width, height)
        if self.ready:
            self.layout()
            self.refresh_field()
            self.sync_particles()

    def on_mouse_motion(self, x, y, dx, dy):
        self.cursor = x, y
        self.hovered = self.pick(x, y)
        for name, (bx, by, w, h, rect, _) in self.buttons.items():
            rect.color = self.palette['hover'] if bx <= x <= bx+w and by <= y <= by+h else self.palette['button']

    def on_mouse_enter(self, x, y):
        if self.ready:
            self.on_mouse_motion(x, y, 0, 0)

    def on_mouse_leave(self, x, y):
        if self.ready:
            self.cursor = (-100., -100.)
            self.hovered = -1
            self.last_click = None

    def on_mouse_press(self, x, y, button, modifiers):
        self.cursor = x, y
        if button == mouse.LEFT:
            for name, (bx, by, w, h, _, _) in self.buttons.items():
                if bx <= x <= bx+w and by <= y <= by+h:
                    self.action(name)
                    return
            i, now = self.pick(x, y), time.perf_counter()
            previous = self.last_click
            if (i >= 0 and previous is not None and previous[0] == i
                    and now-previous[1] <= .35
                    and math.hypot(x-previous[2], y-previous[3]) <= 8):
                fixed = self.sim.toggle_fixed(i)
                self.last_click = None
                self.notice('Charge fixed in space; double-click to release.' if fixed
                            else 'Charge released from rest; Lorentz forces apply again.')
                self.sync_particles()
                self.update_stats()
                return
            self.last_click = (i, now, x, y) if i >= 0 else None
            self.sim.dragged = i if i >= 0 and not self.sim.fixed[i] else -1
            self.sim.target[:] = self.world(x, y)
            if i >= 0 and self.sim.fixed[i]:
                self.notice('This charge is fixed. Double-click it to release.')
            if self.sim.dragged >= 0 and self.paused:
                self.notice('Paused. Press P to resume, or N to advance one step with the drag force.')
        elif button == mouse.RIGHT:
            self.delete_at_cursor()

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        self.cursor = x, y
        if self.last_click is not None and math.hypot(x-self.last_click[2], y-self.last_click[3]) > 5:
            self.last_click = None  # A drag is not the first half of a double-click.
        if self.sim.dragged >= 0:
            cfg, r = self.sim.config, self.sim.radius
            wx, wy = self.world(x, y)
            self.sim.target[:] = np.clip([wx, wy], [r, r], [cfg.width-r, cfg.height-r])

    def on_mouse_release(self, x, y, button, modifiers):
        if button == mouse.LEFT:
            self.sim.dragged = -1

    def on_deactivate(self):
        if self.ready:
            self.sim.dragged = -1
            self.keys_down.clear()
            self.accumulator = 0
            self.last_click = None

    def on_key_press(self, symbol, modifiers):
        if symbol in self.keys_down:
            return  # One particle per actual press, not operating-system key repeat.
        self.keys_down.add(symbol)
        actions = {key.P: 'pause', key.R: 'reset', key.T: 'theme', key.E: 'lines',
                   key.B: 'heat_magnetic', key.F: 'heat_electric', key.U: 'heat_energy',
                   key.M: 'heat_signed', key.V: 'velocity', key.D: 'density',
                   key._1: 'positive', key._2: 'negative', key._3: 'proton', key._4: 'electron'}
        if symbol in actions:
            self.action(actions[symbol])
        elif symbol == key.SPACE and self.in_plot(*self.cursor):
            sign = -1 if modifiers & key.MOD_SHIFT else self.sign
            species = 'charge' if modifiers & key.MOD_SHIFT else self.injection_species
            self.sim.add_charge(*self.world(*self.cursor), sign, species)
            self.last_click = None
            self.refresh_field()
            self.sync_particles()
            self.update_stats()
            description = f'{"Positive" if sign > 0 else "Negative"} charge' if species == 'charge' else species.title()
            self.notice(f'{description} added at the cursor.')
        elif symbol in (key.BACKSPACE, key.DELETE):
            self.delete_at_cursor()
        elif symbol == key.N and self.paused:
            self.sim.step()
            self.refresh_field()
            self.sync_particles()
            self.update_stats()
        elif symbol == key.ESCAPE:
            self.close()

    def on_key_release(self, symbol, modifiers):
        self.keys_down.discard(symbol)

    def update_stats(self):
        d = self.sim.diagnostics()
        self.stats_label.text = f'{d["particles"]} charges · {np.count_nonzero(self.sim.fixed)} fixed · vmax {d["max_speed_over_c"]:.2f}c'
        self.time_label.text = f't = {self.sim.time:7.2f}    ·    dt {self.sim.dt:.4f}'
        gauss_label = 'Gauss (removal)' if self.sim.removal_active else 'Gauss residual'
        self.error_label.text = f'{gauss_label}  {d["gauss_linf"]:.1e}'
        self.fps_label.text = f'{1000/max(self.frame_ms, 1):.0f} FPS'
        self.run_label.text = 'PAUSED' if self.paused else 'COMPUTE LIMITED' if self.slow else 'RUNNING'
        if not self.message_until and 0 <= self.hovered < len(self.sim.charge):
            i = self.hovered
            species = ('Charge', 'Proton', 'Electron')[self.sim.species[i]]
            state = 'fixed' if self.sim.fixed[i] else 'free'
            sign = '+' if self.sim.charge[i] > 0 else '−'
            self.message_label.text = f'{species} · q = {sign}e · mass {self.sim.mass[i]:.5g} electron masses · {state}'
        elif not self.message_until:
            self.message_label.text = self.message

    def update(self, elapsed):
        start = time.perf_counter()
        self.slow = elapsed > .15
        if not self.paused:
            self.accumulator += min(elapsed, .1)*self.speed
            count = min(12, int(self.accumulator/self.sim.dt))
            if count:
                self.sim.step(count)
                self.accumulator -= count*self.sim.dt
            if self.accumulator > 12*self.sim.dt:
                self.slow = True
                self.accumulator = self.sim.dt
        self.physics_ms = .9*self.physics_ms+.1*(time.perf_counter()-start)*1000
        self.line_age += elapsed
        self.stats_age += elapsed
        if not self.paused and self.line_age > 1/15:
            self.refresh_field()
        self.hovered = self.pick(*self.cursor)
        self.sync_particles()
        if self.stats_age > .3:
            self.update_stats()
            self.stats_age = 0
        if self.message_until and time.perf_counter() > self.message_until:
            self.message_until = 0
            self.message = 'Drag to apply force. Double-click a charge to fix or release it.'
            self.message_label.text = self.message

    def draw_vectors(self):
        values = []
        if self.show_velocity:
            for (x, y), (vx, vy) in zip(self.sim.pos, self.sim.velocity):
                length = math.hypot(vx, vy)
                if length < .03:
                    continue
                factor = min(.28, 1.2/length)
                ex, ey = x+vx*factor, y+vy*factor
                ex, ey = np.clip([ex, ey], [0, 0], [self.sim.config.width, self.sim.config.height])
                ux, uy = vx/length, vy/length
                values += [(x, y, 3), (ex, ey, 3),
                           (ex-.09*ux-.04*uy, ey-.09*uy+.04*ux, 3), (ex, ey, 3),
                           (ex-.09*ux+.04*uy, ey-.09*uy-.04*ux, 3), (ex, ey, 3)]
        if self.sim.dragged >= 0:
            x, y = self.sim.pos[self.sim.dragged]
            tx, ty = self.sim.target
            values += [(x, y, 3), (tx, ty, 3)]
        if not values:
            return
        data = np.asarray(values, dtype=np.float32).ravel()
        if self.vector_vertices is None:
            self.vector_vertices = self.program.vertex_list(len(values), gl.GL_LINES, position=('f', data))
        else:
            if self.vector_vertices.count != len(values):
                self.vector_vertices.resize(len(values))
            self.vector_vertices.position[:] = data
        self.program['tint'] = (*[v/255 for v in self.palette['positive']], .85)
        self.vector_vertices.draw(gl.GL_LINES)

    def on_draw(self):
        if not self.ready:
            return
        now = time.perf_counter()
        if self.record_frames:
            self.draw_times.append(now)
        self.frame_ms = .95*self.frame_ms+.05*min((now-self.last_draw)*1000, 200)
        self.last_draw = now
        p = self.palette
        gl.glClearColor(*[v/255 for v in p['bg']], 1.)
        self.clear()
        self.chrome.draw()
        self.heat_sprite.draw()
        self.grid_batch.draw()
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        self.program.use()
        self.program['origin'] = (self.ox, self.oy)
        self.program['scale'] = self.world_scale
        line_color = p['line'] if self.heat_mode == 'off' else p['text']
        self.program['tint'] = (*[v/255 for v in line_color], .82 if self.heat_mode == 'off' else .65)
        if self.show_lines and self.line_count and self.line_vertices is not None:
            self.line_vertices.draw(gl.GL_TRIANGLES)
        self.draw_vectors()
        self.program.stop()
        self.particle_batch.draw()

    def on_close(self):
        pyglet.clock.unschedule(self.update)
        super().on_close()
