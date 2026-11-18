"""Low-poly 3D OpenGL backend for the Fly-Trader League dashboard.
Candlestick charts dynamically represent the fly's real-time net gain/loss performance trajectory,
paired with live order book panels, SEC Edgar crawler terminal, lighting, fog, and isometric camera.
"""
from __future__ import annotations
import math
import random
import pygame
from OpenGL.GL import *
from OpenGL.GLU import *

class OpenGLCanvas:
    def __init__(self, surface, pygame_module):
        self.s = surface
        self.pg = pygame_module
        self._fonts = {}
        self.monitor_texture_id = None
        
        # Initialize performance history series for net gain/loss tracking
        self.perf_series = [100.0, 101.5, 100.8, 102.4, 101.9, 103.5, 104.2]
        self.last_tape_signature = None
        
        w, h = self.s.get_size()
        
        # Setup 3D OpenGL Viewport & Wide Cinematic Perspective Projection
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(58.0, w / max(1, h), 0.1, 100.0)
        
        glMatrixMode(GL_MODELVIEW)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_TEXTURE_2D)
        
        # --- GPU LIGHTING & ATMOSPHERE SETUP ---
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHT1)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.15, 0.18, 0.25, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.85, 0.92, 1.0, 1.0])
        glLightfv(GL_LIGHT0, GL_POSITION, [0.5, 3.0, 1.5, 1.0])
        
        glLightfv(GL_LIGHT1, GL_AMBIENT, [0.05, 0.0, 0.05, 1.0])
        glLightfv(GL_LIGHT1, GL_DIFFUSE, [1.0, 0.1, 0.6, 1.0])
        glLightfv(GL_LIGHT1, GL_POSITION, [-2.0, 1.0, -1.0, 1.0])
        
        glEnable(GL_FOG)
        glFogi(GL_FOG_MODE, GL_EXP2)
        glFogf(GL_FOG_DENSITY, 0.09)
        glFogfv(GL_FOG_COLOR, [0.015, 0.015, 0.03, 1.0])
        
        self.rain_drops = [(random.uniform(-5.0, 5.0), random.uniform(-2.0, 3.0), random.uniform(-4.5, -0.5)) for _ in range(140)]

    def font(self, size):
        if size not in self._fonts:
            self._fonts[size] = self.pg.font.SysFont("consolas,dejavusansmono,couriernew,menlo,monospace", size)
        return self._fonts[size]

    def clear(self, color=(0.015, 0.015, 0.03, 1.0)):
        glClearColor(*color)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    def draw_scene_3d(self, view):
        glLoadIdentity()
        
        gluLookAt(-0.45, 0.55, 2.35,   0.15, -0.2, -1.1,   0.0, 1.0, 0.0)

        self._draw_environment()
        self._draw_workstation(view.pose)
        self._draw_pc_tower()
        self._draw_medium_poly_fly(view.pose)
        self._draw_monitors(view, view.pose, view.tape)
        self._draw_terminal(view)
        self._draw_header(view)
        self._draw_log(view)

    def _draw_environment(self):
        glDisable(GL_FOG)
        glPushMatrix()
        glTranslatef(0.0, 0.2, -4.5)
        building_heights = [2.5, 3.2, 1.9, 3.8, 2.7, 1.7, 3.3, 2.9, 2.1]
        for i, h_val in enumerate(building_heights):
            x_pos = -4.2 + (i * 1.05)
            glPushMatrix()
            glTranslatef(x_pos, h_val / 2.0 - 0.5, 0.0)
            glColor3f(0.025, 0.03, 0.06)
            self._draw_box(0.85, h_val, 0.5)
            
            glDisable(GL_LIGHTING)
            glColor3f(0.3, 0.2, 0.45)
            for wx in [-0.26, 0.26]:
                for wy in range(int(h_val * 2.2)):
                    glPushMatrix()
                    glTranslatef(wx, (wy * 0.22) - (h_val/2) + 0.2, 0.28)
                    self._draw_box(0.05, 0.07, 0.01)
                    glPopMatrix()
            glEnable(GL_LIGHTING)
            glPopMatrix()
        glPopMatrix()
        glEnable(GL_FOG)

        glDisable(GL_LIGHTING)
        glDisable(GL_FOG)
        glLineWidth(1.5)
        glColor4f(0.6, 0.8, 1.0, 0.4)
        glBegin(GL_LINES)
        t_ticks = pygame.time.get_ticks() * 0.0045
        for idx, (rx, ry, rz) in enumerate(self.rain_drops):
            current_y = ry - ((t_ticks + idx) % 4.0)
            glVertex3f(rx, current_y, rz)
            glVertex3f(rx - 0.04, current_y - 0.32, rz)
        glEnd()
        glEnable(GL_FOG)
        glEnable(GL_LIGHTING)

    def _draw_workstation(self, pose):
        glPushMatrix()
        glTranslatef(0.1, -0.68, -1.1)
        glColor3f(0.07, 0.08, 0.12)
        self._draw_box(2.8, 0.12, 1.5)
        
        glDisable(GL_LIGHTING)
        glDisable(GL_FOG)
        glColor3f(0.0, 0.95, 1.0)
        glLineWidth(4.5)
        self._draw_wire_box(2.8, 0.12, 1.5)
        glEnable(GL_FOG)
        glEnable(GL_LIGHTING)
        glPopMatrix()

        glPushMatrix()
        glTranslatef(-0.7, -0.58, -0.8)
        glColor3f(0.11, 0.13, 0.18)
        self._draw_box(0.75, 0.08, 0.75)
        
        glTranslatef(-0.32, 0.5, -0.32)
        self._draw_box(0.08, 0.95, 0.6)
        
        glDisable(GL_LIGHTING)
        glDisable(GL_FOG)
        glColor3f(1.0, 0.0, 0.5)
        self._draw_box(0.09, 0.85, 0.04)
        glEnable(GL_FOG)
        glEnable(GL_LIGHTING)
        glPopMatrix()

        glPushMatrix()
        glTranslatef(0.75, -0.54, -0.9)
        glColor3f(0.7, 0.32, 0.65)
        self._draw_cylinder(0.075, 0.15, 12)
        glPopMatrix()

        self._draw_lever(-0.15, -0.52, -0.95, pose.lever_buy, is_buy=True)
        self._draw_lever(0.42, -0.52, -0.95, pose.lever_sell, is_buy=False)

    def _draw_pc_tower(self):
        glPushMatrix()
        glTranslatef(1.35, -0.15, -0.8)
        glRotatef(-15, 0.0, 1.0, 0.0)
        
        glColor3f(0.04, 0.05, 0.08)
        self._draw_box(0.8, 1.35, 0.9)
        
        glColor3f(0.08, 0.10, 0.15)
        self._draw_box(0.82, 1.28, 0.02)
        
        glDisable(GL_LIGHTING)
        glDisable(GL_FOG)
        for fy_offset in [0.35, -0.35]:
            glPushMatrix()
            glTranslatef(0.0, fy_offset, 0.46)
            glColor3f(0.0, 0.98, 1.0)
            self._draw_concentric_ring(0.2, 0.14, 16)
            glPopMatrix()
            
        glPushMatrix()
        glTranslatef(0.28, 0.58, 0.46)
        glColor3f(1.0, 0.0, 0.55)
        self._draw_box(0.05, 0.05, 0.02)
        glPopMatrix()
        glEnable(GL_FOG)
        glEnable(GL_LIGHTING)
        glPopMatrix()

    def _draw_lever(self, x, y, z, press_val, is_buy):
        glPushMatrix()
        glTranslatef(x, y - (press_val * 0.05), z)
        glColor3f(0.16, 0.18, 0.22)
        self._draw_box(0.35, 0.05, 0.1)
        
        glTranslatef(0.0, 0.08, 0.0)
        if is_buy:
            glColor3f(0.0, 1.0, 0.2)
        else:
            glColor3f(1.0, 0.1, 0.1)
        self._draw_sphere(0.07, 8, 8)
        glPopMatrix()

    def _draw_medium_poly_fly(self, pose):
        glPushMatrix()
        glTranslatef(-0.55, -0.28, -0.9)
        
        if pose.state == "SILENT":
            glRotatef(65, 1.0, 0.0, 0.0)
            glTranslatef(0.0, -0.18, 0.14)
        else:
            glTranslatef(0.0, math.sin(pygame.time.get_ticks() * 0.006) * 0.022, 0.0)

        glPushMatrix()
        glScalef(0.58, 0.44, 0.78)
        glColor3f(0.13, 0.15, 0.21)
        self._draw_sphere(0.38, 12, 12)
        glPopMatrix()

        glPushMatrix()
        glTranslatef(0.34, 0.16, 0.0)
        glColor3f(0.24, 0.28, 0.37)
        self._draw_sphere(0.32, 12, 12)
        glPopMatrix()

        glPushMatrix()
        glTranslatef(0.6, 0.28, 0.0)
        glColor3f(0.30, 0.34, 0.44)
        self._draw_sphere(0.25, 12, 12)
        
        glTranslatef(0.12, 0.05, 0.09)
        if pose.state == "BUY":
            glColor3f(0.0, 1.0, 0.3)
        elif pose.state == "SELL":
            glColor3f(1.0, 0.1, 0.2)
        else:
            glColor3f(0.92, 0.04, 0.08)
        self._draw_sphere(0.14, 10, 10)
        
        glDisable(GL_LIGHTING)
        glDisable(GL_FOG)
        glColor4f(0.3, 0.0, 0.0, 0.4)
        glLineWidth(1.0)
        self._draw_wire_sphere(0.14, 6, 6)
        glEnable(GL_FOG)
        glEnable(GL_LIGHTING)
        glPopMatrix()

        for lx_off in [-0.1, 0.15, 0.35]:
            glPushMatrix()
            glTranslatef(lx_off, -0.15, 0.24)
            glColor3f(0.55, 0.6, 0.68)
            self._draw_cylinder(0.035, 0.38, 8)
            glPopMatrix()

        glPushMatrix()
        glTranslatef(0.05, 0.44, 0.0)
        glRotatef(18, 0.0, 0.0, 1.0)
        wing_flap = math.sin(pygame.time.get_ticks() * 0.075) * 18 if pose.state != "SILENT" else 0
        glRotatef(wing_flap, 1.0, 0.0, 0.0)
        
        glDisable(GL_LIGHTING)
        glDisable(GL_FOG)
        glColor4f(0.85, 0.9, 0.96, 0.55)
        self._draw_box(1.1, 0.01, 0.36)
        glColor4f(0.4, 0.5, 0.6, 0.8)
        self._draw_wire_box(1.1, 0.01, 0.36)
        glEnable(GL_FOG)
        glEnable(GL_LIGHTING)
        glPopMatrix()

        if pose.state == "SILENT":
            self._render_text_billboard(0.6, 0.6, 0.0, "Z  z  z...", (255, 190, 60), 16)

        glPopMatrix()

    def _draw_monitors(self, view, pose, tape):
        glPushMatrix()
        glTranslatef(0.78, 0.48, -1.3)
        glRotatef(-15, 0.0, 1.0, 0.0)
        
        # Monitor Frame
        glColor3f(0.06, 0.08, 0.11)
        self._draw_box(1.5, 0.85, 0.06)
        
        ui_surface = pygame.Surface((512, 288))
        ui_surface.fill((12, 15, 26))
        
        # --- UPDATE PERFORMANCE TRAJECTORY FROM LIVE TAPE / SCORES ---
        current_tape_sig = len(tape) if tape else 0
        if tape and len(tape) > 0:
            score = tape[0][1] if len(tape[0]) > 1 else 0.0
            latest_val = self.perf_series[-1] + (score * 0.4)
        else:
            latest_val = self.perf_series[-1] + ((random.random() - 0.49) * 0.25)

        # Shift performance history series when new trades occur or periodically
        if current_tape_sig != self.last_tape_signature or pygame.time.get_ticks() % 50 == 0:
            self.perf_series.pop(0)
            self.perf_series.append(latest_val)
            self.last_tape_signature = current_tape_sig

        # Render Candlesticks representing Net Gain/Loss Performance Curve
        candles = []
        min_v = min(self.perf_series) - 1.5
        max_v = max(self.perf_series) + 1.5
        val_range = max(1.0, max_v - min_v)

        for i, val in enumerate(self.perf_series):
            cx = 35 + (i * 42)
            norm = (val - min_v) / val_range
            center_y = int(210 - (norm * 130))
            
            prev_val = self.perf_series[max(0, i-1)]
            is_green = val >= prev_val
            
            open_p = center_y + (4 if is_green else -4)
            close_p = center_y + (-4 if is_green else 4)
            high_p = min(open_p, close_p) - 12
            low_p = max(open_p, close_p) + 12
            
            candles.append((cx, high_p, low_p, open_p, close_p, is_green))

        for cx, high, low, open_p, close_p, is_green in candles:
            col = (57, 255, 20) if is_green else (255, 40, 50)
            pygame.draw.line(ui_surface, col, (cx, high), (cx, low), 2)
            pygame.draw.rect(ui_surface, col, (cx - 8, min(open_p, close_p), 16, max(6, abs(open_p - close_p))))

        # Order Book & Net PnL Data Panel on the Right
        pygame.draw.rect(ui_surface, (20, 24, 34), (320, 10, 182, 268))
        font_sm = self.font(12)
        
        state_col = (57, 255, 20) if pose.state == "BUY" else ((255, 40, 50) if pose.state == "SELL" else (80, 220, 255))
        lbl_img = font_sm.render(f"ACTION: {pose.state}", True, state_col)
        ui_surface.blit(lbl_img, (332, 20))
        
        display_ticker = pose.ticker if pose.ticker else (tape[0][0] if tape and len(tape) > 0 else "N/A")
        tk_img = font_sm.render(f"TICKER: {display_ticker}", True, (210, 220, 235))
        ui_surface.blit(tk_img, (332, 42))

        net_gain = self.perf_series[-1] - self.perf_series[0]
        net_col = (57, 255, 160) if net_gain >= 0 else (255, 40, 50)
        
        rows = [
            ("NET PnL", f"{net_gain:+.2f}%"),
            ("MBON Hz", f"{(tape[0][1] if tape and len(tape)>0 and len(tape[0])>1 else 0.0):+.1f}"),
            ("STATUS", "PERF TRACK"),
            ("VOL", "1.4M")
        ]
        for idx, (k_txt, v_txt) in enumerate(rows):
            k_img = font_sm.render(k_txt, True, (150, 160, 180))
            v_img = font_sm.render(v_txt, True, net_col if k_txt == "NET PnL" else (210, 220, 235))
            ui_surface.blit(k_img, (332, 90 + (idx * 32)))
            ui_surface.blit(v_img, (415, 90 + (idx * 32)))

        w, h = ui_surface.get_width(), ui_surface.get_height()
        data = pygame.image.tostring(ui_surface, "RGBA", True)
        
        if self.monitor_texture_id is None:
            self.monitor_texture_id = glGenTextures(1)
        
        glBindTexture(GL_TEXTURE_2D, self.monitor_texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
        
        glPushMatrix()
        glTranslatef(0.0, 0.0, 0.038)
        glDisable(GL_LIGHTING)
        glDisable(GL_FOG)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.monitor_texture_id)
        glColor4f(1, 1, 1, 1)
        
        qw, qh = 0.7, 0.38
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex3f(-qw, -qh, 0)
        glTexCoord2f(1, 0); glVertex3f(qw, -qh, 0)
        glTexCoord2f(1, 1); glVertex3f(qw, qh, 0)
        glTexCoord2f(0, 1); glVertex3f(-qw, qh, 0)
        glEnd()
        
        glDisable(GL_TEXTURE_2D)
        glEnable(GL_FOG)
        glEnable(GL_LIGHTING)
        glPopMatrix()

        glPopMatrix()

    def _draw_terminal(self, view):
        glPushMatrix()
        glTranslatef(-0.82, 0.38, -1.3)
        glRotatef(15, 0.0, 1.0, 0.0)
        
        glColor3f(0.03, 0.11, 0.06)
        self._draw_box(0.9, 0.76, 0.06)
        
        research_active = view.pose.state == "RESEARCH"
        term_title = "CRAWLING SEC EDGAR..." if research_active else "SEC EDGAR CRAWLER"
        self._render_text_billboard(0.0, 0.3, 0.045, term_title, (255, 190, 60), 14)
        
        lines = view.research_lines[-5:] if view.research_lines else ["(standby - scanning cache)"]
        for i, ln in enumerate(lines):
            short_ln = ln[:32] if len(ln) > 32 else ln
            self._render_text_billboard(0.0, 0.14 - (i * 0.08), 0.045, short_ln, (210, 220, 235), 11)
            
        glPopMatrix()

    def _draw_header(self, view):
        h = view.header
        mode = h.get("mode", "idle").upper()
        round_num = h.get("round", "-")
        universe = h.get("universe", 499)
        top_fly = h.get("top_id") or "-"
        ic_val = h.get("top_ic")
        ic_str = f"{ic_val:+.3f}" if ic_val is not None and ic_val == ic_val else "n/a"
        
        header_text = f"ROUND: {round_num} | UNIVERSE: {universe} stocks | TOP FLY: {top_fly} | 3-RD AVG IC: {ic_str} | MODE: {mode}"
        self._render_overlay_text(20, 20, header_text, (57, 255, 160), 15)

    def _draw_log(self, view):
        start_y = 690
        self._render_overlay_text(20, start_y - 30, "LOG STREAM:", (110, 122, 145), 14)
        logs = list(view.log)[-5:]
        for i, (col, text) in enumerate(logs):
            self._render_overlay_text(20, start_y + (i * 20), text, col, 13)

    # --- Geometry & Rendering Helpers ---
    def _render_text_billboard(self, x, y, z, text_str, color, size):
        if not text_str:
            return
        font_obj = self.font(size)
        surface = font_obj.render(text_str, True, color)
        w, h = surface.get_width(), surface.get_height()
        data = pygame.image.tostring(surface, "RGBA", True)
        
        texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
        
        glPushMatrix()
        glTranslatef(x, y, z)
        qw, qh = (w / 400.0), (h / 400.0)
        
        glDisable(GL_LIGHTING)
        glDisable(GL_FOG)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glColor4f(1, 1, 1, 1)
        
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex3f(-qw, -qh, 0)
        glTexCoord2f(1, 0); glVertex3f(qw, -qh, 0)
        glTexCoord2f(1, 1); glVertex3f(qw, qh, 0)
        glTexCoord2f(0, 1); glVertex3f(-qw, qh, 0)
        glEnd()
        
        glDisable(GL_TEXTURE_2D)
        glEnable(GL_FOG)
        glEnable(GL_LIGHTING)
        glDeleteTextures([texture_id])
        glPopMatrix()

    def _render_overlay_text(self, x, y, text_str, color, size):
        if not text_str:
            return
        font_obj = self.font(size)
        surface = font_obj.render(text_str, True, color)
        self.s.blit(surface, (x, y))

    def _draw_box(self, w, h, d):
        hw, hh, hd = w / 2, h / 2, d / 2
        glBegin(GL_QUADS)
        glNormal3f(0.0, 0.0, 1.0); glVertex3f(-hw, -hh, hd); glVertex3f(hw, -hh, hd); glVertex3f(hw, hh, hd); glVertex3f(-hw, hh, hd)
        glNormal3f(0.0, 0.0, -1.0); glVertex3f(hw, -hh, -hd); glVertex3f(-hw, -hh, -hd); glVertex3f(-hw, hh, -hd); glVertex3f(hw, hh, -hd)
        glNormal3f(0.0, 1.0, 0.0); glVertex3f(-hw, hh, hd); glVertex3f(hw, hh, hd); glVertex3f(hw, hh, -hd); glVertex3f(-hw, hh, -hd)
        glNormal3f(0.0, -1.0, 0.0); glVertex3f(-hw, -hh, -hd); glVertex3f(hw, -hh, -hd); glVertex3f(hw, -hh, hd); glVertex3f(-hw, -hh, hd)
        glNormal3f(1.0, 0.0, 0.0); glVertex3f(hw, -hh, hd); glVertex3f(hw, -hh, -hd); glVertex3f(hw, hh, -hd); glVertex3f(hw, hh, hd)
        glNormal3f(-1.0, 0.0, 0.0); glVertex3f(-hw, -hh, -hd); glVertex3f(-hw, -hh, hd); glVertex3f(-hw, hh, hd); glVertex3f(-hw, hh, -hd)
        glEnd()

    def _draw_wire_box(self, w, h, d):
        hw, hh, hd = w / 2, h / 2, d / 2
        glBegin(GL_LINE_LOOP)
        glVertex3f(-hw, -hh, hd); glVertex3f(hw, -hh, hd); glVertex3f(hw, hh, hd); glVertex3f(-hw, hh, hd)
        glEnd()

    def _draw_concentric_ring(self, outer_r, inner_r, segments):
        glBegin(GL_QUAD_STRIP)
        for i in range(segments + 1):
            theta = (i / segments) * 2.0 * math.pi
            cos_t, sin_t = math.cos(theta), math.sin(theta)
            glNormal3f(cos_t, sin_t, 0.0)
            glVertex3f(cos_t * inner_r, sin_t * inner_r, 0.0)
            glVertex3f(cos_t * outer_r, sin_t * outer_r, 0.0)
        glEnd()

    def _draw_cylinder(self, radius, height, slices):
        hh = height / 2.0
        glBegin(GL_QUAD_STRIP)
        for i in range(slices + 1):
            theta = (i / slices) * 2.0 * math.pi
            x, z = math.cos(theta) * radius, math.sin(theta) * radius
            glNormal3f(math.cos(theta), 0.0, math.sin(theta))
            glVertex3f(x, -hh, z)
            glVertex3f(x, hh, z)
        glEnd()

    def _draw_sphere(self, radius, slices, stacks):
        quadric = gluNewQuadric()
        gluQuadricNormals(quadric, GLU_SMOOTH)
        gluSphere(quadric, radius, slices, stacks)
        gluDeleteQuadric(quadric)

    def _draw_wire_sphere(self, radius, slices, stacks):
        quadric = gluNewQuadric()
        gluQuadricDrawStyle(quadric, GLU_LINE)
        gluSphere(quadric, radius, slices, stacks)
        gluDeleteQuadric(quadric)