# ── Imports ───────────────────────────────────────────────────────────────────
import math
import random
import sys
import time
import threading

import numpy as np
import pygame

COLOR_FILE = ",/somab_color.txt"

# Face Color

def get_face_color():
    try:
        with open(COLOR_FILE, "r") as f:
            vals = f.read().strip().split(",")
            return (int(vals[0]), int(vals[1]), int(vals[2]))
    except Exception:
        return (77, 217, 224)  # default teal

# ── Init ──────────────────────────────────────────────────────────────────────
pygame.init()
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN | pygame.NOFRAME)
W, H   = screen.get_size()
clock  = pygame.time.Clock()
pygame.mouse.set_visible(False)

# ── Colors ────────────────────────────────────────────────────────────────────
BLACK = (0, 0, 0)
TEAL  = get_face_color()

# ── Layout ────────────────────────────────────────────────────────────────────
cx     = W // 2
cy     = H // 2
eLx    = cx - 340      # left eye center x
eRx    = cx + 340      # right eye center x
eyeY   = cy - 160      # eye center y
mouthY = eyeY + 270    # mouth center y
EW     = 130           # eye half-width
EH     = 90            # eye half-height
MW     = 120           # mouth half-width
MH     = 80            # mouth max open height
LINE_W = 10            # stroke width

# Eyebrow settings
BROW_OFFSET_Y = -145   # how far above eye center brows sit by default
BROW_W        = 100    # half-width of brow
BROW_THICK    = 8      # brow stroke thickness

# ── File paths ────────────────────────────────────────────────────────────────
STATE_FILE          = ",/somab_state.txt"
VISEME_FILE         = ",/somab_viseme.txt"
INFO_FILE           = ",/somab_info.txt"
DEVMODE_INPUT_FILE  = ",/somab_devmode_input.txt"
DEVMODE_RESULT_FILE = ",/somab_devmode_result.txt"

# ══════════════════════════════════════════════════════════════════════════════
# POSE SYSTEM
# Eyes: bezier control points relative to eye center (0,0).
# x: -EW to +EW. y: negative = up.
# Blending interpolates control point positions directly — true shape morphing.
# ══════════════════════════════════════════════════════════════════════════════

EYE_POSES = {
    "open": {
        "top":    [(-EW,0), (-EW*0.5,-EH*1.0), (0,-EH*1.1), (EW*0.5,-EH*1.0), (EW,0)],
        "bot":    [(-EW,0), (-EW*0.5, EH*1.0), (0, EH*1.1), (EW*0.5, EH*1.0), (EW,0)],
        "filled": True,
        "circle": True,
    },
    "half_closed": {
        "top":    [(-EW,0), (-EW*0.5,-EH*0.42), (0,-EH*0.5),  (EW*0.5,-EH*0.42), (EW,0)],
        "bot":    [(-EW,0), (-EW*0.5, EH*1.0),  (0, EH*1.1),  (EW*0.5, EH*1.0),  (EW,0)],
        "filled": True,
    },
    "squint": {
        "top":    [(-EW,0), (-EW*0.5,-EH*0.5),  (0,-EH*0.6),  (EW*0.5,-EH*0.5),  (EW,0)],
        "bot":    [(-EW,0), (-EW*0.5, EH*0.5),  (0, EH*0.6),  (EW*0.5, EH*0.5),  (EW,0)],
        "filled": True,
    },
    "wide": {
        "top":    [(-EW,0), (-EW*0.5,-EH*1.35), (0,-EH*1.5),  (EW*0.5,-EH*1.35), (EW,0)],
        "bot":    [(-EW,0), (-EW*0.5, EH*1.35), (0, EH*1.5),  (EW*0.5, EH*1.35), (EW,0)],
        "filled": True,
        "circle": True,
    },
    "closed": {
        "top":    [(-EW,0), (-EW*0.5,-EH*0.08), (0,-EH*0.1),  (EW*0.5,-EH*0.08), (EW,0)],
        "bot":    [(-EW,0), (-EW*0.5, EH*0.08), (0, EH*0.1),  (EW*0.5, EH*0.08), (EW,0)],
        "filled": False,
    },
    "wink": {
        "top":    [(-EW,0), (-EW*0.5,-EH*0.05), (0,-EH*0.05), (EW*0.5,-EH*0.05), (EW,0)],
        "bot":    [(-EW,0), (-EW*0.5, EH*0.05), (0, EH*0.05), (EW*0.5, EH*0.05), (EW,0)],
        "filled": True,
    },
    "x_squint": {
        # > < chevron shape — lines point inward
        "top": [(-EW, -EH*0.5), (-EW*0.1, 0), (-EW, EH*0.5)],   # left eye: >
        "bot": [( EW, -EH*0.5), ( EW*0.1, 0), ( EW, EH*0.5)],   # right eye: 
        "filled": False,
        "x_mode": True,
    },
}

MOUTH_POSES = {
    "smile": {
        "type":   "curve",
        "points": [(-MW,-10), (-MW*0.3,MH*0.7), (0,MH*0.85), (MW*0.3,MH*0.7), (MW,-10)],
    },
    "flat": {
        "type":   "curve",
        "points": [(-MW*0.8,8), (-MW*0.3,6), (0,6), (MW*0.3,6), (MW*0.8,8)],
    },
    "frown": {
        "type":   "curve",
        "points": [(-MW,MH*0.3), (-MW*0.3,-MH*0.4), (0,-MH*0.55), (MW*0.3,-MH*0.4), (MW,MH*0.3)],
    },
    "smirk": {
        "type":   "curve",
        "points": [(-MW*0.7,15), (-MW*0.1,10), (MW*0.2,MH*0.5), (MW*0.6,MH*0.7), (MW*0.9,MH*0.3)],
    },
    "open": {
        "type": "open",
        "top":  [(-MW*0.65,0), (0,-MH*0.15), (MW*0.65,0)],
        "bot":  [(-MW*0.65,0), (-MW*0.3,MH*1.5), (0,MH*1.7), (MW*0.3,MH*1.5), (MW*0.65,0)],
    },
}

# ── State definitions ─────────────────────────────────────────────────────────
STATES = {
    "idle": {
        "left_eye":  {"open": 1.0},
        "right_eye": {"open": 1.0},
        "mouth":     {"smile": 1.0},
        "brow_l": 0, "brow_r": 0,
    },
    "listening": {
        "left_eye":  {"open": 1.0},
        "right_eye": {"open": 1.0},
        "mouth":     {"smirk": 1.0},
        "brow_l": 0, "brow_r": -28,
    },
    "thinking": {
        "left_eye":  {"x_squint": 1.0},
        "right_eye": {"x_squint": 1.0},
        "mouth":     {"flat": 1.0},
        "brow_l": 0, "brow_r": 0,
    },
    "speaking": {
        "left_eye":  {"squint": 1.0},
        "right_eye": {"squint": 1.0},
        "mouth":     {"open": 1.0},
        "brow_l": 0, "brow_r": 0,
    },
    "sleeping": {
        "left_eye":  {"half_closed": 0.2, "wink": 0.8},
        "right_eye": {"half_closed": 0.2, "wink": 0.8},
        "mouth":     {"smile": 1.0},
        "brow_l": 12, "brow_r": 12,
    },
    "devmode": {
        "left_eye":  {"open": 1.0},
        "right_eye": {"open": 1.0},
        "mouth":     {"flat": 1.0},
        "brow_l": 0, "brow_r": 0,
    },
}

# ── Viseme map ────────────────────────────────────────────────────────────────
VISEME_MAP = {
    'rest': {"smile": 0.2, "flat": 0.8},
    'm':    {"flat": 1.0},
    'b':    {"flat": 1.0},
    'p':    {"flat": 1.0},
    'oʊ':  {"open": 0.7,  "smile": 0.3},
    'uː':  {"open": 0.55},
    'ʊ':   {"open": 0.45},
    'æ':   {"open": 1.0},
    'ɑː':  {"open": 1.0},
    'aɪ':  {"open": 0.85},
    'aʊ':  {"open": 0.85},
    'iː':  {"smile": 0.6, "open": 0.25},
    'ɪ':   {"smile": 0.5, "open": 0.2},
    'eɪ':  {"smile": 0.5, "open": 0.35},
    'ɛ':   {"smile": 0.4, "open": 0.35},
    'ɜː':  {"open": 0.45},
    'ə':   {"open": 0.25},
    'ʌ':   {"open": 0.45},
    'f':   {"open": 0.18},
    'v':   {"open": 0.18},
    'θ':   {"open": 0.25},
    'ð':   {"open": 0.25},
    's':   {"open": 0.12},
    'z':   {"open": 0.12},
    't':   {"open": 0.12},
    'd':   {"open": 0.12},
    'n':   {"open": 0.08},
    'l':   {"open": 0.18},
    'ɹ':   {"open": 0.28},
    'k':   {"open": 0.18},
    'ɡ':   {"open": 0.18},
    'ŋ':   {"open": 0.08},
    'h':   {"open": 0.38},
    'w':   {"open": 0.45},
    'j':   {"smile": 0.3, "open": 0.18},
}

# ══════════════════════════════════════════════════════════════════════════════
# BLEND + RENDER HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _bezier(controls, steps=32):
    pts = []
    n   = len(controls) - 1
    for i in range(steps + 1):
        t  = i / steps
        pt = [0.0, 0.0]
        for j, cp in enumerate(controls):
            b     = math.comb(n, j) * (1 - t) ** (n - j) * t ** j
            pt[0] += b * cp[0]
            pt[1] += b * cp[1]
        pts.append(tuple(pt))
    return pts

def _lerp(a, b, t):
    return a + (b - a) * t

def _lerp_weights(w_a, w_b, t):
    all_keys = set(w_a) | set(w_b)
    return {k: _lerp(w_a.get(k, 0.0), w_b.get(k, 0.0), t) for k in all_keys}

def _blend_eye_curve(pose_weights, curve_key):
    total  = sum(v for k, v in pose_weights.items() if "x_mode" not in EYE_POSES.get(k, {}))
    if total < 0.001:
        return None
    result = None
    for pose_name, w in pose_weights.items():
        pose = EYE_POSES.get(pose_name, {})
        if pose.get("x_mode"):
            continue
        pts = pose.get(curve_key)
        if pts is None:
            continue
        nw = w / total
        if result is None:
            result = [(p[0] * nw, p[1] * nw) for p in pts]
        else:
            result = [(result[i][0] + pts[i][0] * nw,
                       result[i][1] + pts[i][1] * nw) for i in range(len(pts))]
    return result

# ── Face state ────────────────────────────────────────────────────────────────
class FaceState:
    def __init__(self):
        s              = STATES["idle"]
        self.left_eye  = dict(s["left_eye"])
        self.right_eye = dict(s["right_eye"])
        self.mouth     = dict(s["mouth"])
        self.brow_l    = float(s["brow_l"])
        self.brow_r    = float(s["brow_r"])

    def lerp_toward_state(self, state_name, speed, dt, mouth_override=None):
        tgt = STATES[state_name]
        t   = min(1.0, speed * dt)
        self.left_eye  = _lerp_weights(self.left_eye,  tgt["left_eye"],  t)
        self.right_eye = _lerp_weights(self.right_eye, tgt["right_eye"], t)
        self.mouth     = _lerp_weights(self.mouth, mouth_override if mouth_override else tgt["mouth"], min(1.0, t * (1.8 if mouth_override else 1.0)))
        self.brow_l    = _lerp(self.brow_l, float(tgt["brow_l"]), t)
        self.brow_r    = _lerp(self.brow_r, float(tgt["brow_r"]), t)

    def lerp_toward_expr(self, expr, speed, dt):
        t              = min(1.0, speed * dt)
        self.left_eye  = _lerp_weights(self.left_eye,  expr["left_eye"],  t)
        self.right_eye = _lerp_weights(self.right_eye, expr["right_eye"], t)
        self.mouth     = _lerp_weights(self.mouth,     expr["mouth"],     t)
        self.brow_l    = _lerp(self.brow_l, float(expr["brow_l"]), t)
        self.brow_r    = _lerp(self.brow_r, float(expr["brow_r"]), t)

# ══════════════════════════════════════════════════════════════════════════════
# DRAW FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def draw_eye(surface, cx_e, cy_e, pose_weights, blink_frac=0.0, is_left=True):
    # Fast path — if dominant pose is circle, just draw a circle
    circle_w = sum(v for k, v in pose_weights.items() if EYE_POSES.get(k, {}).get("circle"))
    other_w  = sum(v for k, v in pose_weights.items() if not EYE_POSES.get(k, {}).get("circle") and k != "x_squint")

    if circle_w > 0.05:
        radius  = int(EH * (1.0 + 0.5 * pose_weights.get("wide", 0.0)) * (1.0 - blink_frac * 0.9))
        radius  = max(2, radius)
        alpha   = int(255 * min(1.0, circle_w))
        surf    = pygame.Surface((W, H), pygame.SRCALPHA)
        pygame.draw.circle(surf, (*TEAL, alpha), (int(cx_e), int(cy_e)), radius)
        surface.blit(surf, (0, 0))
        return
    
    x_w          = pose_weights.get("x_squint", 0.0)
    other_weights = {k: v for k, v in pose_weights.items() if k != "x_squint"}
    other_total  = sum(other_weights.values())

    surf = pygame.Surface((W, H), pygame.SRCALPHA)

    # X squint lines
    if x_w > 0.02:
        x_alpha  = int(255 * min(1.0, x_w))
        surf     = pygame.Surface((W, H), pygame.SRCALPHA)
        chevron  = EYE_POSES["x_squint"]["top"] if is_left else EYE_POSES["x_squint"]["bot"]
        pts      = [(int(cx_e + p[0]), int(cy_e + p[1])) for p in chevron]
        pygame.draw.lines(surf, (*TEAL, x_alpha), False, pts, LINE_W)
        surface.blit(surf, (0, 0))

    # Blended open/closed/squint/etc eyes
    if other_total > 0.02:
        top_ctrl = _blend_eye_curve(other_weights, "top")
        bot_ctrl = _blend_eye_curve(other_weights, "bot")
        if top_ctrl and bot_ctrl:
            # Apply blink
            if blink_frac > 0.001:
                top_ctrl = [(p[0], p[1] * (1.0 - blink_frac)) for p in top_ctrl]
                bot_ctrl = [(p[0], p[1] * (1.0 - blink_frac)) for p in bot_ctrl]

            top_pts    = _bezier(top_ctrl)
            bot_pts    = _bezier(bot_ctrl)
            top_screen = [(int(cx_e + p[0]), int(cy_e + p[1])) for p in top_pts]
            bot_screen = [(int(cx_e + p[0]), int(cy_e + p[1])) for p in bot_pts]
            other_alpha = int(255 * min(1.0, other_total))

            should_fill = any(
                EYE_POSES.get(k, {}).get("filled", False) and v > 0.2
                for k, v in other_weights.items()
            )

            if should_fill:
                poly = top_screen + list(reversed(bot_screen))
                if len(poly) >= 3:
                    pygame.draw.polygon(surf, (*TEAL, other_alpha), poly)
            else:
                if len(top_screen) >= 2:
                    pygame.draw.lines(surf, (*TEAL, other_alpha), False, top_screen, LINE_W)
                if len(bot_screen) >= 2:
                    pygame.draw.lines(surf, (*TEAL, other_alpha), False, bot_screen, LINE_W)

    surface.blit(surf, (0, 0))

def draw_eyebrow(surface, cx_e, cy_e, brow_offset):
    by  = cy_e + BROW_OFFSET_Y + brow_offset
    pts = [
        (cx_e - BROW_W,       by + 14),
        (cx_e - BROW_W * 0.3, by),
        (cx_e + BROW_W * 0.3, by),
        (cx_e + BROW_W,       by + 14),
    ]
    bezier_pts = _bezier(pts, steps=20)
    screen_pts = [(int(p[0]), int(p[1])) for p in bezier_pts]
    surf = pygame.Surface((W, H), pygame.SRCALPHA)
    if len(screen_pts) >= 2:
        pygame.draw.lines(surf, (*TEAL, 255), False, screen_pts, BROW_THICK)
    surface.blit(surf, (0, 0))

def draw_mouth(surface, cx_m, cy_m, pose_weights):
    open_w        = pose_weights.get("open", 0.0)
    curve_weights = {k: v for k, v in pose_weights.items()
                     if k != "open" and MOUTH_POSES.get(k, {}).get("type") == "curve"}
    curve_total   = sum(curve_weights.values())

    surf = pygame.Surface((W, H), pygame.SRCALPHA)

    # Filled open mouth
    if open_w > 0.02:
        top_ctrl       = MOUTH_POSES["open"]["top"]
        bot_ctrl_base  = MOUTH_POSES["open"]["bot"]
        bot_ctrl_scaled = [(p[0], p[1] * open_w) for p in bot_ctrl_base]
        top_pts = _bezier(top_ctrl, steps=24)
        bot_pts = _bezier(bot_ctrl_scaled, steps=24)
        top_s   = [(int(cx_m + p[0]), int(cy_m + p[1])) for p in top_pts]
        bot_s   = [(int(cx_m + p[0]), int(cy_m + p[1])) for p in bot_pts]
        poly    = top_s + list(reversed(bot_s))
        if len(poly) >= 3:
            pygame.draw.polygon(surf, (*TEAL, 255), poly)

    # Blended curve mouths
    if curve_total > 0.02:
        blended = None
        for pose_name, w in curve_weights.items():
            pts = MOUTH_POSES[pose_name]["points"]
            nw  = w / curve_total
            if blended is None:
                blended = [(p[0] * nw, p[1] * nw) for p in pts]
            else:
                blended = [(blended[i][0] + pts[i][0] * nw,
                            blended[i][1] + pts[i][1] * nw) for i in range(len(pts))]
        if blended:
            curve_alpha = int(255 * min(1.0, curve_total) * (1.0 - open_w * 0.8))
            screen_pts  = [(int(cx_m + p[0]), int(cy_m + p[1])) for p in _bezier(blended)]
            if len(screen_pts) >= 2 and curve_alpha > 4:
                pygame.draw.lines(surf, (*TEAL, curve_alpha), False, screen_pts, LINE_W)

    surface.blit(surf, (0, 0))

# ══════════════════════════════════════════════════════════════════════════════
# IPC READERS
# ══════════════════════════════════════════════════════════════════════════════

def get_state():
    try:
        with open(STATE_FILE, "r") as f:
            s = f.read().strip()
            return s if s in STATES else "idle"
    except Exception:
        return "idle"

def get_viseme():
    try:
        with open(VISEME_FILE, "r") as f:
            content = f.read().strip()
            return None if not content or content == "none" else content
    except Exception:
        return None

def get_info():
    try:
        with open(INFO_FILE, "r") as f:
            return f.read().strip()
    except Exception:
        return ""

# ══════════════════════════════════════════════════════════════════════════════
# DEV MODE OVERLAY
# ══════════════════════════════════════════════════════════════════════════════

def draw_devmode(surface, password_text, show_password, error=False):
    overlay = pygame.Surface((W, H), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 200))
    surface.blit(overlay, (0, 0))

    font_title = pygame.font.SysFont("monospace", 48, bold=True)
    font_input = pygame.font.SysFont("monospace", 36)
    font_hint  = pygame.font.SysFont("monospace", 24)

    title = font_title.render("DEV MODE", True, TEAL)
    surface.blit(title, title.get_rect(center=(W // 2, H // 2 - 160)))

    sub_col  = (255, 80, 80) if error else (180, 180, 180)
    sub_text = "Incorrect password" if error else "Enter password to continue"
    sub      = font_hint.render(sub_text, True, sub_col)
    surface.blit(sub, sub.get_rect(center=(W // 2, H // 2 - 100)))

    box_w, box_h = 600, 70
    box_x = W // 2 - box_w // 2
    box_y = H // 2 - box_h // 2
    pygame.draw.rect(surface, (20, 20, 30), (box_x, box_y, box_w, box_h), border_radius=12)
    pygame.draw.rect(surface, TEAL,         (box_x, box_y, box_w, box_h), 2, border_radius=12)

    display_text = password_text if show_password else "•" * len(password_text)
    pw_surf      = font_input.render(display_text, True, (255, 255, 255))
    surface.blit(pw_surf, (box_x + 20, box_y + 18))

    if int(time.time() * 2) % 2 == 0:
        cursor_x = box_x + 20 + font_input.size(display_text)[0] + 2
        pygame.draw.line(surface, TEAL, (cursor_x, box_y + 14), (cursor_x, box_y + 54), 2)

    toggle_text = "HIDE" if show_password else "SHOW"
    toggle_surf = font_hint.render(toggle_text, True, TEAL)
    toggle_rect = toggle_surf.get_rect(center=(W // 2, H // 2 + 70))
    surface.blit(toggle_surf, toggle_rect)

    hint = font_hint.render("ENTER to confirm   ESC to cancel", True, (100, 100, 100))
    surface.blit(hint, hint.get_rect(center=(W // 2, H // 2 + 130)))
    return toggle_rect

# ══════════════════════════════════════════════════════════════════════════════
# INFO PANEL
# ══════════════════════════════════════════════════════════════════════════════

def draw_info_panel(surface, info, panel_alpha):
    if not info or panel_alpha < 0.01:
        return
    lines   = [l for l in info.split("\n") if l.strip()]
    panel_w = W // 2
    max_y   = H - 80

    panel_surf = pygame.Surface((panel_w, H), pygame.SRCALPHA)
    panel_surf.fill((0, 0, 0, 0))
    pygame.draw.line(panel_surf, (*TEAL, int(180 * panel_alpha)), (0, 80), (0, H - 80), 2)

    y = 120
    for i, line in enumerate(lines):
        if y > max_y:
            break
        if i == 0:
            font = pygame.font.SysFont("monospace", 48, bold=True)
            text = font.render(line, True, TEAL)
            if text.get_width() > panel_w - 120:
                scale = (panel_w - 120) / text.get_width()
                font  = pygame.font.SysFont("monospace", int(48 * scale), bold=True)
                text  = font.render(line, True, TEAL)
            panel_surf.blit(text, (60, y))
            y += text.get_height() + 16
            pygame.draw.line(panel_surf, (*TEAL, int(100 * panel_alpha)),
                             (60, y - 8), (panel_w - 60, y - 8), 1)
            y += 10
        else:
            remaining = len(lines) - i
            available = max_y - y
            line_h    = min(44, max(22, available // max(remaining, 1)))
            font_size = max(18, line_h - 8)
            font      = pygame.font.SysFont("monospace", font_size)
            words     = line.split()
            cur       = ""
            for word in words:
                test = f"{cur} {word}".strip()
                if font.size(test)[0] < panel_w - 120:
                    cur = test
                else:
                    if cur:
                        if y + line_h > max_y:
                            panel_surf.blit(font.render("...", True, (150,150,150)), (60, y))
                            surface.blit(panel_surf, (W // 2, 0))
                            return
                        panel_surf.blit(font.render(cur, True, (220,220,220)), (60, y))
                        y += line_h
                    cur = word
            if cur and y + line_h <= max_y:
                panel_surf.blit(font.render(cur, True, (220,220,220)), (60, y))

    surface.blit(panel_surf, (W // 2, 0))

# ══════════════════════════════════════════════════════════════════════════════
# BOOT ANIMATION
# ══════════════════════════════════════════════════════════════════════════════

def draw_boot(surface, t):
    surface.fill(BLACK)
    face_surf = pygame.Surface((W, H), pygame.SRCALPHA)

    if t < 2.0:
        frac    = min(1.0, t / 2.0)
        weights = {"open": frac, "closed": 1.0 - frac}
    elif t < 2.3:
        b       = math.sin((t - 2.0) / 0.3 * math.pi)
        weights = {"closed": b, "open": 1.0 - b}
    elif t < 2.6:
        weights = {"open": 1.0}
    elif t < 2.9:
        b       = math.sin((t - 2.6) / 0.3 * math.pi)
        weights = {"closed": b, "open": 1.0 - b}
    else:
        weights = {"open": 1.0}

    draw_eye(face_surf, eLx, eyeY, weights, is_left=True)
    draw_eye(face_surf, eRx, eyeY, weights, is_left=False)
    draw_eyebrow(face_surf, eLx, eyeY, 0)
    draw_eyebrow(face_surf, eRx, eyeY, 0)
    if t >= 3.0:
        draw_mouth(face_surf, cx, mouthY, {"smile": 1.0})

    surface.blit(face_surf, (0, 0))

# ══════════════════════════════════════════════════════════════════════════════
# IDLE EXPRESSIONS
# ══════════════════════════════════════════════════════════════════════════════

class IdleExpression:
    def __init__(self):
        self.active       = False
        self.target       = None
        self.duration     = 0.0
        self.elapsed      = 0.0
        self.next_trigger = random.uniform(4.0, 9.0)
        self.timer        = 0.0

    def tick(self, dt, current_state):
        if current_state != "idle":
            self.active = False
            self.target = None
            self.timer  = 0.0
            return
        self.timer += dt
        if not self.active:
            if self.timer >= self.next_trigger:
                self._trigger()
        else:
            self.elapsed += dt
            if self.elapsed >= self.duration:
                self.active       = False
                self.target       = None
                self.next_trigger = random.uniform(3.0, 8.0)
                self.timer        = 0.0

    def _trigger(self):
        expressions = [
            {"left_eye": {"squint": 0.7, "open": 0.3}, "right_eye": {"squint": 0.7, "open": 0.3},
             "mouth": {"smile": 1.0}, "brow_l": 0, "brow_r": 0,
             "duration": random.uniform(0.8, 1.4)},
            {"left_eye": {"open": 1.0}, "right_eye": {"wide": 0.4, "open": 0.6},
             "mouth": {"smile": 1.0}, "brow_l": 0, "brow_r": -28,
             "duration": random.uniform(0.7, 1.2)},
            {"left_eye": {"squint": 0.5, "open": 0.5}, "right_eye": {"squint": 0.5, "open": 0.5},
             "mouth": {"smile": 1.0}, "brow_l": 0, "brow_r": 0,
             "duration": random.uniform(1.0, 1.8)},
            {"left_eye": {"wide": 1.0}, "right_eye": {"wide": 1.0},
             "mouth": {"smile": 0.6, "flat": 0.4}, "brow_l": -20, "brow_r": -20,
             "duration": random.uniform(0.5, 0.9)},
            {"left_eye": {"closed": 1.0}, "right_eye": {"closed": 1.0},
             "mouth": {"smile": 1.0}, "brow_l": 0, "brow_r": 0,
             "duration": random.uniform(0.25, 0.4)},
            {"left_eye": {"open": 1.0}, "right_eye": {"squint": 0.6, "open": 0.4},
             "mouth": {"smirk": 0.5, "flat": 0.5}, "brow_l": -15, "brow_r": 8,
             "duration": random.uniform(0.8, 1.4)},
            {"left_eye": {"half_closed": 1.0}, "right_eye": {"half_closed": 1.0},
             "mouth": {"smile": 0.8, "flat": 0.2}, "brow_l": 8, "brow_r": 8,
             "duration": random.uniform(0.8, 1.5)},
        ]
        chosen        = random.choice(expressions)
        self.target   = chosen
        self.duration = chosen["duration"]
        self.elapsed  = 0.0
        self.active   = True

    def get_target(self):
        return self.target if self.active else None

# ══════════════════════════════════════════════════════════════════════════════
# SLEEPING Z ANIMATION
# ══════════════════════════════════════════════════════════════════════════════

class SleepingZ:
    def __init__(self):
        self.zs          = []
        self.spawn_timer = 0.0

    def tick(self, dt, current_state):
        if current_state != "sleeping":
            self.zs          = []
            self.spawn_timer = 0.0
            return
        self.spawn_timer += dt
        if self.spawn_timer > 2.5:
            self.spawn_timer = 0.0
            self.zs.append({
                "x":     eRx + random.randint(80, 160),
                "y":     float(eyeY),
                "size":  random.randint(60, 110),
                "alpha": 0.0,
                "phase": "in",
                "age":   0.0,
                "drift": random.uniform(-15, 15),
            })
        for z in self.zs:
            z["age"]  += dt
            z["y"]    -= 40 * dt
            z["x"]    += z["drift"] * dt
            if z["phase"] == "in":
                z["alpha"] = min(1.0, z["alpha"] + dt * 2.5)
                if z["alpha"] >= 1.0:
                    z["phase"] = "hold"
            elif z["phase"] == "hold":
                if z["age"] > 2.0:
                    z["phase"] = "out"
            elif z["phase"] == "out":
                z["alpha"] = max(0.0, z["alpha"] - dt * 1.5)
        self.zs = [z for z in self.zs if not (z["phase"] == "out" and z["alpha"] <= 0.0)]

    def draw(self, surface):
        for z in self.zs:
            if z["alpha"] <= 0.01:
                continue
            font = pygame.font.SysFont("monospace", z["size"], bold=True)
            text = font.render("Z", True, TEAL)
            text.set_alpha(int(z["alpha"] * 210))
            surface.blit(text, (int(z["x"]), int(z["y"])))

# ══════════════════════════════════════════════════════════════════════════════
# MAIN FACE LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_face():
    # Boot screen
    boot_start = time.time()
    while True:
        t = time.time() - boot_start
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit(); sys.exit()
        draw_boot(screen, t)
        pygame.display.flip()
        clock.tick(60)
        if t > 3.5:
            break

    # Main loop setup
    face               = FaceState()
    idle_expr          = IdleExpression()
    sleep_zs           = SleepingZ()
    current_state      = "idle"
    devmode_password   = ""
    devmode_show_pw    = False
    devmode_error      = False
    devmode_toggle_rect = None
    blink_t            = 0.0
    next_blink         = random.uniform(3.0, 6.0)
    blinking           = False
    blink_start        = 0.0
    last_t             = time.time()
    face_x_offset      = 0.0
    target_x_offset    = 0.0
    panel_alpha        = 0.0

    while True:
        t      = time.time()
        dt     = min(t - last_t, 0.05)
        last_t = t
        global TEAL
        TEAL = get_face_color()

        # Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if current_state == "devmode":
                    if event.key == pygame.K_RETURN:
                        with open(DEVMODE_INPUT_FILE, "w") as f:
                            f.write(devmode_password)
                        devmode_password = ""
                        devmode_error    = False
                    elif event.key == pygame.K_ESCAPE:
                        with open(DEVMODE_RESULT_FILE, "w") as f:
                            f.write("cancelled")
                        devmode_password = ""
                        devmode_error    = False
                    elif event.key == pygame.K_BACKSPACE:
                        devmode_password = devmode_password[:-1]
                    else:
                        if event.unicode and len(devmode_password) < 32:
                            devmode_password += event.unicode
                else:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit(); sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN and current_state == "devmode":
                if devmode_toggle_rect and devmode_toggle_rect.collidepoint(event.pos):
                    devmode_show_pw = not devmode_show_pw

        # Devmode error check
        if current_state == "devmode":
            try:
                with open(DEVMODE_RESULT_FILE, "r") as f:
                    result = f.read().strip()
                if result == "error":
                    devmode_error = True
                    open(DEVMODE_RESULT_FILE, "w").close()
            except Exception:
                pass

        # State update
        new_state = get_state()
        if new_state != current_state:
            current_state = new_state
            pygame.mouse.set_visible(current_state == "devmode")

        # Viseme / idle expression / morph
        viseme_key = get_viseme()
        idle_expr.tick(dt, current_state)
        sleep_zs.tick(dt, current_state)

        mouth_override = None
        if viseme_key is not None and viseme_key in VISEME_MAP:
            mouth_override = VISEME_MAP[viseme_key]
            morph_speed    = 14.0
        elif idle_expr.get_target() is not None:
            morph_speed = 4.0
        else:
            morph_speed = 6.0

        if idle_expr.get_target() is not None and current_state == "idle":
            face.lerp_toward_expr(idle_expr.get_target(), morph_speed, dt)
        else:
            face.lerp_toward_state(current_state, morph_speed, dt, mouth_override)

        # Blink
        blink_t += dt
        if not blinking and blink_t > next_blink and current_state not in ("sleeping",):
            blinking    = True
            blink_start = t
            blink_t     = 0
            next_blink  = random.uniform(3.0, 6.0)

        blink_frac = 0.0
        if blinking and current_state != "speaking":
            elapsed = t - blink_start
            if elapsed < 0.13:
                blink_frac = math.sin(elapsed / 0.13 * math.pi)
            else:
                blinking = False

        # Info panel
        info            = get_info()
        has_info        = bool(info)
        target_x_offset = -W // 4 if has_info else 0
        face_x_offset  += (target_x_offset - face_x_offset) * min(1.0, 6.0 * dt)
        panel_alpha    += ((1.0 if has_info else 0.0) - panel_alpha) * min(1.0, 6.0 * dt)

        # Draw
        screen.fill(BLACK)
        face_surf = pygame.Surface((W, H), pygame.SRCALPHA)

        draw_eye(face_surf, eLx, eyeY, face.left_eye,  blink_frac, is_left=True)
        draw_eye(face_surf, eRx, eyeY, face.right_eye, blink_frac, is_left=False)
        draw_eyebrow(face_surf, eLx, eyeY, face.brow_l)
        draw_eyebrow(face_surf, eRx, eyeY, face.brow_r)
        draw_mouth(face_surf, cx, mouthY, face.mouth)

        screen.blit(face_surf, (int(face_x_offset), 0))

        if has_info:
            draw_info_panel(screen, info, panel_alpha)

        sleep_zs.draw(screen)

        if current_state == "devmode":
            devmode_toggle_rect = draw_devmode(screen, devmode_password, devmode_show_pw, devmode_error)

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    run_face()
