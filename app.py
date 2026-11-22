"""
N-D Physics Simulation & Interactions (Advanced GUI v3)
=====================================================================
4D / 5D / N-D rigid geometric body dynamics with full angular collisions.
Controlled via a dynamic, interactive GUI sidebar with thick glowing wireframes.

Controls
--------
Left-Click (Viewport) Spawn selected shape
Right-Drag         Grab & drag nearest shape
Mouse Wheel        Zoom (or scroll GUI)
Left-Drag          Rotate active plane
Arrow Keys         Pan camera (X/Y)
Shift + Up/Down    Pan camera (Z)
9 / 0              Camera Roll (Left / Right)
BACKSPACE          Toggle Time-Reversal
SPACE              Trigger N-D Shockwave at mouse
P pause  R reduced-motion  H hide GUI  ESC quit
"""
import pygame
import numpy as np
import math
import random
import sys
import argparse
from dataclasses import dataclass, field
from typing import List, Tuple, Any
from itertools import product

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

W, H = 1280, 800
FPS  = 60
GUI_W = 280
PLAY_W = W - GUI_W
GUI_TOP = 50
GUI_BOTTOM = H - 100

THEMES = [
    { # 0: Abyss
        "bg": (8, 10, 18), "fg": (220, 225, 235), "dim": (90, 100, 120),
        "accent": (255, 180, 60), "grid": (28, 32, 50), "card": (16, 20, 32),
        "btn": (24, 28, 42), "btn_hover": (32, 38, 56), "btn_active": (45, 55, 80)
    },
    { # 1: Matrix
        "bg": (0, 0, 0), "fg": (0, 255, 0), "dim": (0, 100, 0),
        "accent": (0, 255, 150), "grid": (0, 40, 0), "card": (0, 10, 0),
        "btn": (0, 20, 0), "btn_hover": (0, 40, 0), "btn_active": (0, 60, 0)
    },
    { # 2: Solar
        "bg": (240, 235, 220), "fg": (20, 20, 20), "dim": (100, 100, 100),
        "accent": (200, 50, 50), "grid": (200, 195, 180), "card": (230, 225, 210),
        "btn": (220, 215, 200), "btn_hover": (210, 205, 190), "btn_active": (200, 195, 180)
    }
]

def clamp(v, lo, hi): return max(lo, min(hi, v))
def lerp(a, b, t):    return a + (b - a) * t

def hsv_to_rgb(h, s, v):
    c = pygame.Color(0)
    c.hsva = (h % 360, max(0, min(100, s * 100)),
              max(0, min(100, v * 100)), 100)
    return (c.r, c.g, c.b)

# ─── N-D math ──────────────────────────────────────────────────────────────
def givens(i, j, theta, n):
    G = np.eye(n)
    c, s = math.cos(theta), math.sin(theta)
    G[i, i] = c; G[j, j] = c
    G[i, j] = -s; G[j, i] = s
    return G

def rotation_matrix(angles, n):
    R = np.eye(n)
    for (i, j), t in angles.items():
        if abs(t) < 1e-9: continue
        R = R @ givens(i, j, t, n)
    return R

# ─── N-D Geometry: Polytopes ───────────────────────────────────────────────
def generate_duoprism(p, q, half):
    verts = []
    for i in range(p):
        for j in range(q):
            theta1 = 2 * math.pi * i / p
            theta2 = 2 * math.pi * j / q
            verts.append([math.cos(theta1), math.sin(theta1), math.cos(theta2), math.sin(theta2)])
    verts = np.array(verts) * half * 0.8
    edges = []
    for i in range(p):
        for j in range(q):
            idx = i * q + j
            idx_p = ((i + 1) % p) * q + j
            idx_q = i * q + ((j + 1) % q)
            edges.append((idx, idx_p))
            edges.append((idx, idx_q))
    return verts, edges

def generate_polytope(shape_type, dim, half):
    if shape_type in [3, 5, 6] and dim != 4:
        shape_type = 0
        
    if shape_type == 0: # Hypercube
        idx = np.arange(2 ** dim)
        bits = ((idx[:, None] >> np.arange(dim)[None, :]) & 1).astype(float)
        verts = (bits * 2 - 1) * half * 0.8
        edges = [(i, i ^ (1 << k)) for i in range(len(verts)) for k in range(dim) if i < (i ^ (1 << k))]
        return verts, edges
    elif shape_type == 1: # Cross-polytope
        verts = []
        for i in range(dim):
            v1 = np.zeros(dim); v1[i] = 1.0; verts.append(v1)
            v2 = np.zeros(dim); v2[i] = -1.0; verts.append(v2)
        verts = np.array(verts) * half
        edges = [(i, j) for i in range(len(verts)) for j in range(i + 1, len(verts)) if i // 2 != j // 2]
        return verts, edges
    elif shape_type == 2: # Simplex
        verts = []
        for i in range(dim + 1):
            v = np.zeros(dim + 1)
            v[i] = 1.0
            v = v - np.ones(dim + 1) / (dim + 1)
            verts.append(v[:-1])
        verts = np.array(verts)
        max_r = np.max(np.linalg.norm(verts, axis=1))
        if max_r > 0: verts = verts / max_r * half
        edges = [(i, j) for i in range(len(verts)) for j in range(i + 1, len(verts))]
        return verts, edges
    elif shape_type == 3 and dim == 4: # 24-cell
        verts = []
        for i in range(4):
            for j in range(i + 1, 4):
                for s1 in [1, -1]:
                    for s2 in [1, -1]:
                        v = np.zeros(4); v[i] = s1; v[j] = s2
                        verts.append(v)
        verts = np.array(verts) * half * 0.8
        edges = []
        target_dist = half * 0.8 * math.sqrt(2)
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                if np.linalg.norm(verts[i] - verts[j]) < target_dist * 1.1:
                    edges.append((i, j))
        return verts, edges
    elif shape_type == 4: # Tesseractic Honeycomb (Small grid)
        grid_size = 2
        coords = np.linspace(-half*0.6, half*0.6, grid_size)
        verts = np.array(list(product(coords, repeat=dim)))
        step = abs(coords[1] - coords[0])
        edges = []
        for i in range(len(verts)):
            for j in range(i+1, len(verts)):
                if abs(np.linalg.norm(verts[i]-verts[j]) - step) < 1e-4:
                    edges.append((i, j))
        return verts, edges
    elif shape_type == 5 and dim == 4: return generate_duoprism(3, 3, half)
    elif shape_type == 6 and dim == 4: return generate_duoprism(3, 4, half)
    return np.array([]), []

SHAPE_NAMES = ["Hypercube", "Cross-polytope", "Simplex", "24-cell", "Honeycomb", "3-3 Duoprism", "3-4 Duoprism"]
BOUND_NAMES = ["Off", "Box", "Sphere"]

# ─── Data ──────────────────────────────────────────────────────────────────
@dataclass
class RigidPolytope:
    shape_type: int
    pos: np.ndarray
    vel: np.ndarray
    rot_angles: dict = field(default_factory=dict)
    ang_vel: dict = field(default_factory=dict)
    mass: float = 5.0
    radius: float = 2.0
    hue: float = 200.0
    local_verts: np.ndarray = field(default_factory=lambda: np.array([]))
    edges: list = field(default_factory=list)

@dataclass
class World:
    dim: int = 4
    shapes: List[RigidPolytope] = field(default_factory=list)
    rot_angles: dict = field(default_factory=dict)
    active_pair_idx: int = 0
    spawn_shape_type: int = 0
    theme_idx: int = 0
    cam_offset: np.ndarray = field(default_factory=lambda: np.zeros(4))
    cam_roll: float = 0.0
    cam_dist: float = 12.0
    focal: float = 600.0
    bounds: float = 8.0
    
    gravity_g: float = 4.0
    central_k: float = 0.5
    slice_pos: float = 0.0
    slice_range: float = 1.5
    time_scale: float = 1.0
    
    paused: bool = False
    time_reversing: bool = False
    history: list = field(default_factory=list)
    shockwaves: list = field(default_factory=list)
    
    gravity_enabled: bool = True
    well_enabled: bool = True
    collisions_enabled: bool = True
    bounds_mode: int = 1
    auto_rotate: bool = False
    slice_enabled: bool = False
    bloom_enabled: bool = True
    
    mouse_force_mode: int = 0
    drag_idx: int = -1
    drag_active: bool = False
    drag_last: tuple = (0, 0)
    drag_dist: float = 0.0
    mouse_pos: tuple = (0, 0)
    dragging_scrollbar: bool = False
    
    time: float = 0.0
    shake: float = 0.0
    flash: float = 0.0
    show_gui: bool = True
    pair_change_t: float = 0.0
    gui_scroll: int = 0
    gui_max_scroll: int = 0
    
    gui_elements: list = field(default_factory=list)
    gui_dirty: bool = True


def init_polytope(shape_type, dim, pos):
    half = 1.5
    local_verts, edges = generate_polytope(shape_type, dim, half)
    
    rot_angles = {(i, j): 0.0 for i in range(dim) for j in range(i + 1, dim)}
    ang_vel = {(i, j): 0.0 for i in range(dim) for j in range(i + 1, dim)}
    
    if len(ang_vel) > 0:
        keys = list(ang_vel.keys())
        for k in random.sample(keys, min(2, len(keys))):
            ang_vel[k] = random.uniform(-0.5, 0.5)
            
    return RigidPolytope(
        shape_type=shape_type,
        pos=pos,
        vel=np.zeros(dim),
        rot_angles=rot_angles,
        ang_vel=ang_vel,
        mass=5.0,
        radius=half * 1.2,
        hue=random.uniform(0, 360),
        local_verts=local_verts,
        edges=edges
    )

def make_world(dim, n_shapes=3, prev=None):
    dim = clamp(dim, 2, 8)
    w = World(dim=dim)
    w.cam_offset = np.zeros(dim)
    w.rot_angles = {(i, j): 0.0 for i in range(dim) for j in range(i + 1, dim)}
    
    if dim >= 4: w.rot_angles[(0, 3)] = 0.6; w.rot_angles[(1, 3)] = 0.4
    if dim >= 5: w.rot_angles[(0, 4)] = 0.5; w.rot_angles[(2, 4)] = 0.3

    if prev is not None:
        w.theme_idx = prev.theme_idx
        w.cam_dist = prev.cam_dist
        w.cam_roll = prev.cam_roll
        w.gravity_g = prev.gravity_g
        w.central_k = prev.central_k
        w.bounds = prev.bounds
        w.time_scale = prev.time_scale
        w.gravity_enabled = prev.gravity_enabled
        w.well_enabled = prev.well_enabled
        w.collisions_enabled = prev.collisions_enabled
        w.bounds_mode = prev.bounds_mode
        w.auto_rotate = prev.auto_rotate
        w.slice_enabled = prev.slice_enabled
        w.slice_pos = prev.slice_pos
        w.bloom_enabled = prev.bloom_enabled
        w.mouse_force_mode = prev.mouse_force_mode
        w.spawn_shape_type = prev.spawn_shape_type
        w.show_gui = prev.show_gui
        w.active_pair_idx = prev.active_pair_idx % len(w.rot_angles) if w.rot_angles else 0

    for _ in range(n_shapes):
        pos = np.random.uniform(-3.0, 3.0, dim)
        st = random.randint(0, 2) if dim != 4 else random.randint(0, 6)
        w.shapes.append(init_polytope(st, dim, pos))

    return w

def add_shockwave(world: World, pos=None, strength=100.0):
    if pos is None:
        R_cam = rotation_matrix(world.rot_angles, world.dim)
        p3 = np.array([ (world.mouse_pos[0] - (PLAY_W//2)) * world.cam_dist / world.focal,
                        -(world.mouse_pos[1] - (H//2)) * world.cam_dist / world.focal,
                        0.0 ])
        p_nd = np.zeros(world.dim)
        p_nd[:3] = p3
        target_pos = p_nd @ R_cam.T
    else:
        target_pos = pos
        
    world.shockwaves.append({'pos': target_pos, 'radius': 0.1, 'strength': strength})
    world.shake = min(world.shake + 0.5, 1.0)
    world.flash = 0.6


# ─── Physics ───────────────────────────────────────────────────────────────
def step_physics(world: World, dt: float):
    if world.paused: return
    dt *= world.time_scale
    if dt == 0: return
    
    n, M = world.dim, len(world.shapes)
    if M == 0: return

    pos  = np.array([s.pos  for s in world.shapes])
    vel  = np.array([s.vel  for s in world.shapes])
    mass = np.array([s.mass for s in world.shapes])
    radii = np.array([s.radius for s in world.shapes])

    if world.time_reversing:
        if world.history:
            old_pos, old_vel, old_rot, old_ang_vel = world.history.pop()
            for i, s in enumerate(world.shapes):
                if i < len(old_pos):
                    s.pos = old_pos[i].copy()
                    s.vel = old_vel[i].copy()
                    s.rot_angles = old_rot[i].copy()
                    s.ang_vel = old_ang_vel[i].copy()
        else:
            world.time_reversing = False
        return

    world.history.append((pos.copy(), vel.copy(), [s.rot_angles.copy() for s in world.shapes], [s.ang_vel.copy() for s in world.shapes]))
    if len(world.history) > 300: world.history.pop(0)

    acc = np.zeros_like(pos)
    
    if world.well_enabled:
        acc += -world.central_k * pos

    if M > 1:
        diff = pos[:, None, :] - pos[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2)) + 0.1
        inv_rN = 1.0 / (dist ** (n - 1))
        np.fill_diagonal(inv_rN, 0.0)

        if world.gravity_enabled and world.gravity_g > 0:
            grav_coeff = world.gravity_g * mass[None, :] * inv_rN
            acc += -np.einsum('ij,ijk->ik', grav_coeff, diff / dist[..., None])

    if world.mouse_force_mode != 0:
        R_cam = rotation_matrix(world.rot_angles, n)
        p3 = np.array([ (world.mouse_pos[0] - (PLAY_W//2)) * world.cam_dist / world.focal,
                        -(world.mouse_pos[1] - (H//2)) * world.cam_dist / world.focal,
                        0.0 ])
        p_nd = np.zeros(n)
        p_nd[:3] = p3
        target_pos = p_nd @ R_cam.T
        
        diff_m = target_pos - pos
        dist_m = np.linalg.norm(diff_m, axis=1) + 0.1
        force_mag = 150.0 / (dist_m ** 2)
        if world.mouse_force_mode == 2: force_mag *= -1
        acc += (force_mag[:, None] * diff_m / dist_m[:, None]) / mass[:, None]

    if world.drag_idx != -1 and world.drag_idx < M:
        R_cam = rotation_matrix(world.rot_angles, n)
        s = world.shapes[world.drag_idx]
        rotated = s.pos @ R_cam.T
        depth = max(world.cam_dist - rotated[2], 0.4)
        rotated[0] = (world.mouse_pos[0] - (PLAY_W // 2)) * depth / world.focal
        rotated[1] = -(world.mouse_pos[1] - (H // 2)) * depth / world.focal
        target_pos = rotated @ R_cam
        acc[world.drag_idx] += 300.0 * (target_pos - s.pos)
        vel[world.drag_idx] *= 0.75

    if world.shockwaves:
        for sw in world.shockwaves:
            sw['radius'] += dt * 12.0
            diff_sw = pos - sw['pos']
            dist_sw = np.linalg.norm(diff_sw, axis=1) + 0.1
            mask = np.abs(dist_sw - sw['radius']) < 1.5
            if np.any(mask):
                force_mag = sw['strength'] / (dist_sw[mask]**2 + 0.1)
                acc[mask] += force_mag[:, None] * (diff_sw[mask] / dist_sw[mask, None])
        world.shockwaves = [sw for sw in world.shockwaves if sw['radius'] < 25.0]

    vel += acc * dt
    pos += vel * dt

    # N-D Collisions with Angular Impulse
    if world.collisions_enabled and M > 1:
        diff = pos[:, None, :] - pos[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        r_sum = radii[:, None] + radii[None, :]
        colliding = (dist2 < r_sum**2) & (dist2 > 1e-6)
        idx_i, idx_j = np.where(np.triu(colliding, k=1))
        
        for i, j in zip(idx_i, idx_j):
            d = pos[j] - pos[i]
            dist = np.linalg.norm(d)
            if dist < 1e-4: continue
            n_hat = d / dist
            v_rel = vel[i] - vel[j]
            v_dot_n = np.dot(v_rel, n_hat)
            
            if v_dot_n > 0:
                e = 0.8
                inv_m_sum = (1/mass[i]) + (1/mass[j])
                impulse = (1 + e) * v_dot_n / inv_m_sum
                vel[i] -= (impulse / mass[i]) * n_hat
                vel[j] += (impulse / mass[j]) * n_hat
                
                # N-D Angular Impulse: find the 2D plane most aligned with the normal
                abs_n = np.abs(n_hat)
                idx1 = np.argmax(abs_n)
                abs_n[idx1] = -1
                idx2 = np.argmax(abs_n)
                
                if idx1 != idx2:
                    pair = (min(idx1, idx2), max(idx1, idx2))
                    s_i = world.shapes[i]
                    s_j = world.shapes[j]
                    if pair in s_i.ang_vel:
                        s_i.ang_vel[pair] -= v_dot_n * 0.15 / mass[i]
                    if pair in s_j.ang_vel:
                        s_j.ang_vel[pair] += v_dot_n * 0.15 / mass[j]
                
            overlap = (radii[i] + radii[j]) - dist
            if overlap > 0:
                inv_m_sum = (1/mass[i]) + (1/mass[j])
                corr = overlap * 0.8
                pos[i] -= corr * n_hat * (1/mass[i]) / inv_m_sum
                pos[j] += corr * n_hat * (1/mass[j]) / inv_m_sum

    bound = world.bounds
    if world.bounds_mode == 1:
        for d in range(n):
            mask = pos[:, d] > bound
            pos[mask, d] = bound
            vel[mask, d] *= -0.8
            mask = pos[:, d] < -bound
            pos[mask, d] = -bound
            vel[mask, d] *= -0.8
    elif world.bounds_mode == 2:
        r = np.linalg.norm(pos, axis=1)
        mask = r > bound
        if np.any(mask):
            n_hat = pos[mask] / r[mask, None]
            pos[mask] = n_hat * bound
            v_dot_n = np.einsum('ij,ij->i', vel[mask], n_hat)
            v_dot_n = np.minimum(v_dot_n, 0)
            vel[mask] -= 1.6 * v_dot_n[:, None] * n_hat

    for i, s in enumerate(world.shapes):
        s.pos = pos[i].copy()
        s.vel = vel[i].copy()
        
        for pair, speed in s.ang_vel.items():
            s.rot_angles[pair] += speed * dt
            
        s.vel *= 0.999 if (world.well_enabled or world.mouse_force_mode != 0) else 0.9999

    world.time += dt
    world.shake = max(0, world.shake - dt * 2.5)
    world.flash = max(0, world.flash - dt * 3)
    world.pair_change_t = max(0, world.pair_change_t - dt)


# ─── Projection ────────────────────────────────────────────────────────────
def project_points(points, R, focal, cam_dist, cx, cy, offset, cam_roll=0.0):
    if len(points) == 0:
        return (np.array([]),) * 3 + (np.zeros((0, R.shape[0])),)
    offset_pts = points - offset
    rotated = offset_pts @ R.T
    M, N = rotated.shape
    x = rotated[:, 0]
    y = rotated[:, 1]
    
    if cam_roll != 0:
        cos_r, sin_r = math.cos(cam_roll), math.sin(cam_roll)
        x_r = x * cos_r - y * sin_r
        y_r = x * sin_r + y * cos_r
    else:
        x_r = x
        y_r = y
        
    z = rotated[:, 2] if N >= 3 else np.zeros_like(x)
    depth = np.maximum(cam_dist - z, 0.4)
    sx = cx + x_r * focal / depth
    sy = cy - y_r * focal / depth
    return sx, sy, depth, rotated


# ─── Pre-rendered Assets & Cache ───────────────────────────────────────────
BG_SURFS = []
GLOW_CACHE = {}

def init_backgrounds():
    BG_SURFS.clear()
    for theme in THEMES:
        surf = pygame.Surface((PLAY_W, H))
        for i in range(H):
            t = i / H
            col = (int(lerp(theme["bg"][0], theme["bg"][0]+6, t)), 
                   int(lerp(theme["bg"][1], theme["bg"][1]+6, t)), 
                   int(lerp(theme["bg"][2], theme["bg"][2]+12, t)))
            pygame.draw.line(surf, col, (0, i), (PLAY_W, i))
        BG_SURFS.append(surf)

def get_glow_surf(color, size):
    key = (color, size)
    if key not in GLOW_CACHE:
        size = max(2, size)
        surf = pygame.Surface((size*4, size*4), pygame.SRCALPHA)
        for r in range(size*2, 0, -1):
            alpha = int(40 * (1 - r/(size*2))**2)
            pygame.draw.circle(surf, (*color, alpha), (size*2, size*2), r)
        GLOW_CACHE[key] = surf
    return GLOW_CACHE[key]


# ─── Drawing ───────────────────────────────────────────────────────────────
def draw_background(screen, world, theme):
    screen.blit(BG_SURFS[world.theme_idx], (0, 0))

def draw_shockwaves(screen, world, R, cx, cy, theme):
    for sw in world.shockwaves:
        pts = np.array([sw['pos']])
        sx, sy, depth, _ = project_points(pts, R, world.focal, world.cam_dist, cx, cy, world.cam_offset, world.cam_roll)
        if depth[0] > 0.4:
            r_2d = sw['radius'] * world.focal / depth[0]
            alpha = int(150 * (1 - sw['radius'] / 25.0))
            gs = pygame.Surface((int(r_2d*2), int(r_2d*2)), pygame.SRCALPHA)
            pygame.draw.circle(gs, (255, 150, 50, alpha), (int(r_2d), int(r_2d)), int(r_2d), 3)
            screen.blit(gs, (sx[0] - r_2d, sy[0] - r_2d), special_flags=pygame.BLEND_ADD)

def draw_shapes(screen, world, R_cam, cx, cy, theme):
    n = world.dim
    all_verts = []
    shape_ranges = []
    for s in world.shapes:
        R_shape = rotation_matrix(s.rot_angles, n)
        world_verts = s.local_verts @ R_shape.T + s.pos
        start_idx = len(all_verts)
        all_verts.extend(world_verts)
        shape_ranges.append((s, start_idx, len(world_verts)))
        
    if not all_verts: return
    arr = np.array(all_verts)
    sx, sy, depth, rotated = project_points(arr, R_cam, world.focal, world.cam_dist, cx, cy, world.cam_offset, world.cam_roll)
    
    for s, start, _ in shape_ranges:
        for a, b in s.edges:
            if depth[start+a] < 0.5 or depth[start+b] < 0.5: continue
            
            alpha = 1.0
            if world.slice_enabled and n > 3:
                wa = rotated[start+a, 3]; wb = rotated[start+b, 3]
                slice_min = world.slice_pos - world.slice_range
                slice_max = world.slice_pos + world.slice_range
                if (wa < slice_min and wb < slice_min) or (wa > slice_max and wb > slice_max):
                    continue
                avg_w = (wa + wb) / 2
                dist_from_slice = abs(avg_w - world.slice_pos)
                alpha = clamp(1.0 - (dist_from_slice / world.slice_range)**2, 0, 1)
                if alpha < 0.05: continue
                
            avg_d = (depth[start+a] + depth[start+b]) / 2
            t_depth = clamp(avg_d / (world.cam_dist + world.bounds), 0, 1)
            
            col = hsv_to_rgb(s.hue, 0.8, 1.0)
            v = int(lerp(30, 200, 1 - t_depth) * alpha)
            col_dim = (int(col[0]*v/255), int(col[1]*v/255), int(col[2]*v/255))
            
            # Thick line rendering (Glow effect)
            try: 
                pygame.draw.line(screen, col_dim, (sx[start+a], sy[start+a]), (sx[start+b], sy[start+b]), 2)
                bright_col = tuple(min(255, c + 40) for c in col_dim)
                pygame.draw.aaline(screen, bright_col, (sx[start+a], sy[start+a]), (sx[start+b], sy[start+b]))
            except: pass

        if world.slice_enabled and n > 3:
            slice_intersections = {}
            for a, b in s.edges:
                wa = rotated[start+a, 3]; wb = rotated[start+b, 3]
                if (wa - world.slice_pos) * (wb - world.slice_pos) < 0:
                    t = (world.slice_pos - wa) / (wb - wa)
                    p_int = rotated[start+a] * (1-t) + rotated[start+b] * t
                    depth_int = max(world.cam_dist - p_int[2], 0.4)
                    ix = cx + p_int[0] * world.focal / depth_int
                    iy = cy - p_int[1] * world.focal / depth_int
                    slice_intersections.setdefault(start+a, []).append((ix, iy))
                    slice_intersections.setdefault(start+b, []).append((ix, iy))
                    
            cs_col = (150, 220, 255)
            for v_idx, pts in slice_intersections.items():
                for i in range(len(pts)):
                    for j in range(i + 1, len(pts)):
                        pygame.draw.line(screen, cs_col, pts[i], pts[j], 2)
                    pygame.draw.circle(screen, (255, 255, 255), (int(pts[i][0]), int(pts[i][1])), 4)
                    pygame.draw.circle(screen, cs_col, (int(pts[i][0]), int(pts[i][1])), 2)

    for i in range(len(arr)):
        if depth[i] < 0.5: continue
        if world.slice_enabled and n > 3:
            w = rotated[i, 3]
            if abs(w - world.slice_pos) > world.slice_range: continue
        
        owner_shape = None
        for s, start, end in shape_ranges:
            if start <= i < end:
                owner_shape = s
                break
                
        if owner_shape is None: continue
        
        col = hsv_to_rgb(owner_shape.hue, 0.8, 1.0)
        r = max(3, int(5 * world.cam_dist / depth[i]))
        
        gs = get_glow_surf(col, r)
        screen.blit(gs, (sx[i] - r*2, sy[i] - r*2), special_flags=pygame.BLEND_ADD)
        
        pygame.draw.circle(screen, col, (int(sx[i]), int(sy[i])), r)
        pygame.draw.circle(screen, (255, 255, 255), (int(sx[i]), int(sy[i])), max(1, r//2))

def draw_gimbal(screen, world, R, cx, cy, theme):
    pairs = list(world.rot_angles.keys())
    if not pairs: return
    ap = pairs[world.active_pair_idx % len(pairs)]
    pts_4d = np.zeros((32, world.dim))
    pts_4d[:, ap[0]] = np.cos(np.linspace(0, 2*math.pi, 32)) * 3.0
    pts_4d[:, ap[1]] = np.sin(np.linspace(0, 2*math.pi, 32)) * 3.0
    sx, sy, depth, _ = project_points(pts_4d, R, world.focal, world.cam_dist, cx, cy, world.cam_offset, world.cam_roll)
    if len(sx) > 0:
        col = hsv_to_rgb(world.time * 50 % 360, 0.5, 1.0) if world.pair_change_t > 0 else theme["accent"]
        for i in range(len(sx)-1):
            if depth[i] > 0.4 and depth[i+1] > 0.4:
                pygame.draw.aaline(screen, col, (sx[i], sy[i]), (sx[i+1], sy[i+1]))

def draw_mouse_force(screen, world, theme):
    if world.mouse_force_mode == 0: return
    mx, my = world.mouse_pos
    if my < 0 or my > H or mx < 0 or mx > PLAY_W: return
    
    if world.mouse_force_mode == 1:
        col = (100, 255, 100)
        for r in range(10, 50, 10):
            a = int(100 * (1 - r/50))
            gs = pygame.Surface((r*2, r*2), pygame.SRCALPHA)
            pygame.draw.circle(gs, (*col, a), (r, r), r, 1)
            screen.blit(gs, (mx - r, my - r), special_flags=pygame.BLEND_ADD)
    else:
        col = (255, 100, 100)
        for angle in range(0, 360, 45):
            rad = math.radians(angle)
            ex = mx + math.cos(rad) * 30
            ey = my + math.sin(rad) * 30
            pygame.draw.aaline(screen, col, (mx, my), (ex, ey))
            ang2 = math.atan2(ey-my, ex-mx)
            p1 = (ex - 6*math.cos(ang2-math.pi/6), ey - 6*math.sin(ang2-math.pi/6))
            p2 = (ex - 6*math.cos(ang2+math.pi/6), ey - 6*math.sin(ang2+math.pi/6))
            pygame.draw.aaline(screen, col, (ex, ey), p1)
            pygame.draw.aaline(screen, col, (ex, ey), p2)

def draw_bounds(screen, world, R, cx, cy, theme):
    if world.bounds_mode == 0: return
    b = world.bounds
    if world.bounds_mode == 1:
        verts, edges = generate_polytope(0, world.dim, b)
    elif world.bounds_mode == 2:
        verts = []
        for i in range(world.dim):
            v1 = np.zeros(world.dim); v1[i] = 1.0; verts.append(v1)
            v2 = np.zeros(world.dim); v2[i] = -1.0; verts.append(v2)
        verts = np.array(verts) * b
        edges = [(i, j) for i in range(len(verts)) for j in range(i + 1, len(verts)) if i // 2 != j // 2]
        
    sx, sy, depth, _ = project_points(verts, R, world.focal, world.cam_dist, cx, cy, world.cam_offset, world.cam_roll)
    for a, c in edges:
        if depth[a] > 0.4 and depth[c] > 0.4:
            try: pygame.draw.aaline(screen, theme["dim"], (sx[a], sy[a]), (sx[c], sy[c]))
            except: pass


# ─── Advanced GUI System ───────────────────────────────────────────────────
class Slider:
    def __init__(self, x, y, w, h, min_val, max_val, init_val, label, world_attr, decimals=1):
        self.rect = pygame.Rect(x, y, w, h)
        self.min_val = min_val
        self.max_val = max_val
        self.value = init_val
        self.label = label
        self.world_attr = world_attr
        self.decimals = decimals
        self.dragging = False
        self.hover = False
        self.target_hover = 0.0
        self.current_hover = 0.0

    def get_hit_rect(self):
        return pygame.Rect(self.rect.x, self.rect.y - 16, self.rect.w, self.rect.h + 20)

    def handle_event(self, event, world):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.get_hit_rect().collidepoint(event.pos) or self.dragging:
                self.dragging = True
                self.set_value_from_pos(event.pos[0])
                return True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.dragging:
                self.dragging = False
                return True
        elif event.type == pygame.MOUSEMOTION:
            if self.dragging:
                self.set_value_from_pos(event.pos[0])
                return True
        return False

    def set_value_from_pos(self, x):
        ratio = (x - self.rect.x) / self.rect.w
        ratio = max(0, min(1, ratio))
        self.value = self.min_val + (self.max_val - self.min_val) * ratio
        self.value = round(self.value, self.decimals)

    def update(self, mouse_pos, dt, world):
        # Sync from world (e.g. if changed via keyboard)
        if not self.dragging:
            self.value = getattr(world, self.world_attr, self.value)
            
        self.hover = self.get_hit_rect().collidepoint(mouse_pos) or self.dragging
        self.target_hover = 1.0 if self.hover else 0.0
        self.current_hover += (self.target_hover - self.current_hover) * min(1, dt * 15)
        
        setattr(world, self.world_attr, self.value)

    def draw(self, screen, fonts, theme, scroll_y):
        r = self.rect.move(0, -scroll_y)
        val_str = f"{self.value:.{self.decimals}f}"
        lbl = fonts['small'].render(f"{self.label}: {val_str}", True, theme["dim"])
        screen.blit(lbl, (r.x, r.y - 16))
        
        track_rect = pygame.Rect(r.x, r.y + r.h//2 - 2, r.w, 4)
        pygame.draw.rect(screen, theme["grid"], track_rect, border_radius=2)
        
        ratio = (self.value - self.min_val) / (self.max_val - self.min_val)
        fill_rect = pygame.Rect(track_rect.x, track_rect.y, int(track_rect.w * ratio), 4)
        pygame.draw.rect(screen, theme["accent"], fill_rect, border_radius=2)
        
        hx = r.x + int(r.w * ratio)
        hy = r.y + r.h//2
        rad = 6 + int(2 * self.current_hover)
        pygame.draw.circle(screen, theme["bg"], (hx, hy), rad)
        pygame.draw.circle(screen, theme["accent"], (hx, hy), rad, 2)


class Button:
    def __init__(self, rect, label, action, is_toggle=False, state_func=None, disabled_func=None):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.action = action
        self.is_toggle = is_toggle
        self.state_func = state_func
        self.disabled_func = disabled_func
        self.hover = False
        self.target_hover = 0.0
        self.current_hover = 0.0

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos) and not (self.disabled_func and self.disabled_func()):
                return True
        return False

    def update(self, mouse_pos, dt):
        self.hover = self.rect.collidepoint(mouse_pos) and not (self.disabled_func and self.disabled_func())
        self.target_hover = 1.0 if self.hover else 0.0
        self.current_hover += (self.target_hover - self.current_hover) * min(1, dt * 15)

    def draw(self, screen, fonts, theme, scroll_y):
        r = self.rect.move(0, -scroll_y)
        disabled = self.disabled_func and self.disabled_func()
        
        bg_col = theme["btn"]
        if disabled:
            bg_col = theme["btn"]
        elif self.is_toggle and self.state_func and self.state_func():
            bg_col = theme["btn_active"]
        else:
            bg_col = (
                int(lerp(theme["btn"][0], theme["btn_hover"][0], self.current_hover)),
                int(lerp(theme["btn"][1], theme["btn_hover"][1], self.current_hover)),
                int(lerp(theme["btn"][2], theme["btn_hover"][2], self.current_hover))
            )
            
        pygame.draw.rect(screen, bg_col, r, border_radius=4)
        pygame.draw.rect(screen, theme["grid"], r, 1, border_radius=4)
        
        text_col = theme["dim"] if disabled else theme["fg"]
        lbl = fonts['small'].render(self.label, True, text_col)
        screen.blit(lbl, (r.x + 8, r.y + (r.height - lbl.get_height())//2))
        
        if self.is_toggle and self.state_func and self.state_func() and not disabled:
            pygame.draw.line(screen, theme["accent"], (r.x+4, r.bottom-2), (r.right-4, r.bottom-2), 3)

def build_gui_layout(world):
    elements = []
    x = PLAY_W + 20
    y = 60
    w = GUI_W - 40
    h = 30
    gap = 8

    # Section: Camera & Dimensions
    elements.append({"type": "HEADER", "text": "DIMENSIONS & CAMERA", "y": y})
    y += 30 + gap
    elements.append(Button((x, y, w//2 - gap//2, h), "[-] Dec Dim", 'dim_dec'))
    elements.append(Button((x + w//2 + gap//2, y, w//2 - gap//2, h), "[+] Inc Dim", 'dim_inc'))
    y += h + gap
    
    pairs = list(world.rot_angles.keys())
    ap = pairs[world.active_pair_idx % len(pairs)] if pairs else (0,0)
    elements.append(Button((x, y, w, h), f"Active Plane: ({ap[0]},{ap[1]})", 'cycle_plane'))
    y += h + gap
    
    elements.append(Slider(x, y + 5, w, h, 5.0, 30.0, world.cam_dist, "Zoom", 'cam_dist', 1))
    y += h + 25
    
    # Section: Spawning
    elements.append({"type": "HEADER", "text": "SPAWNING", "y": y})
    y += 30 + gap
    elements.append(Button((x, y, w, h), f"Shape: {SHAPE_NAMES[world.spawn_shape_type]}", 'cycle_shape'))
    y += h + gap
    elements.append(Button((x, y, w, h), "Spawn Selected Shape", 'spawn_shape'))
    y += h + gap
    elements.append(Button((x, y, w, h), "Clear All Shapes", 'clear_shapes'))
    y += h + gap * 2

    # Section: Forces
    elements.append({"type": "HEADER", "text": "FORCES", "y": y})
    y += 30 + gap
    elements.append(Button((x, y, w, h), "Gravity", 'toggle_gravity', True, lambda: world.gravity_enabled))
    y += h + gap
    elements.append(Slider(x, y + 5, w, h, 0.0, 20.0, world.gravity_g, "G Strength", 'gravity_g', 1))
    y += h + 25
    elements.append(Button((x, y, w, h), "Harmonic Well", 'toggle_well', True, lambda: world.well_enabled))
    y += h + gap
    elements.append(Slider(x, y + 5, w, h, 0.0, 2.0, world.central_k, "Well Strength", 'central_k', 2))
    y += h + 25
    elements.append(Button((x, y, w, h), f"Mouse: {['Off', 'Attract', 'Repel'][world.mouse_force_mode]}", 'cycle_mouse_force'))
    y += h + gap * 2

    # Section: Environment
    elements.append({"type": "HEADER", "text": "ENVIRONMENT", "y": y})
    y += 30 + gap
    elements.append(Button((x, y, w, h), "Collisions", 'toggle_collisions', True, lambda: world.collisions_enabled))
    y += h + gap
    elements.append(Button((x, y, w, h), f"Bounds: {BOUND_NAMES[world.bounds_mode]}", 'cycle_bounds'))
    y += h + gap
    elements.append(Slider(x, y + 5, w, h, 4.0, 16.0, world.bounds, "Bounds Size", 'bounds', 1))
    y += h + 25
    elements.append(Button((x, y, w, h), "Auto-Rotate (Clifford)", 'toggle_auto_rotate', True, lambda: world.auto_rotate))
    y += h + gap * 2

    # Section: Visuals & Time
    elements.append({"type": "HEADER", "text": "VISUALS & TIME", "y": y})
    y += 30 + gap
    elements.append(Button((x, y, w, h), "Time Reversal", 'toggle_time_rev', True, lambda: world.time_reversing))
    y += h + gap
    elements.append(Slider(x, y + 5, w, h, 0.0, 3.0, world.time_scale, "Time Scale", 'time_scale', 2))
    y += h + 25
    elements.append(Button((x, y, w, h), "Slice Mode (MRI)", 'toggle_slice', True, lambda: world.slice_enabled, disabled_func=lambda: world.dim < 4))
    y += h + gap
    
    if world.dim >= 4:
        elements.append(Slider(x, y + 5, w, h, -8.0, 8.0, world.slice_pos, "Slice Position", 'slice_pos', 1))
        y += h + 25
        
    elements.append(Button((x, y, w, h), "Bloom & FX", 'toggle_bloom', True, lambda: world.bloom_enabled))
    y += h + gap
    elements.append(Button((x, y, w, h), f"Theme: {['Abyss', 'Matrix', 'Solar'][world.theme_idx]}", 'cycle_themes'))
    y += h + gap
    elements.append(Button((x, y, w, h), "Reset Scene", 'reset_scene'))

    total_height = y + 50
    return elements, total_height

TEXT_CHANGING_ACTIONS = {'dim_dec', 'dim_inc', 'cycle_shape', 'cycle_mouse_force', 'cycle_bounds', 'cycle_themes', 'cycle_plane'}

def handle_gui_action(world, action):
    new_world = None
    if action == 'dim_dec': new_world = make_world(max(2, world.dim - 1), 3, prev=world)
    elif action == 'dim_inc': new_world = make_world(min(8, world.dim + 1), 3, prev=world)
    elif action == 'cycle_shape':
        world.spawn_shape_type = (world.spawn_shape_type + 1) % 7
        while world.spawn_shape_type in [3, 5, 6] and world.dim != 4:
            world.spawn_shape_type = (world.spawn_shape_type + 1) % 7
    elif action == 'cycle_plane':
        pairs = list(world.rot_angles.keys())
        if pairs:
            world.active_pair_idx = (world.active_pair_idx + 1) % len(pairs)
            world.pair_change_t = 0.4
    elif action == 'spawn_shape':
        st = world.spawn_shape_type
        if st in [3, 5, 6] and world.dim != 4: st = 0
        pos = np.random.uniform(-2.0, 2.0, world.dim)
        world.shapes.append(init_polytope(st, world.dim, pos))
        world.flash = 0.4
        world.history.clear()
        world.time_reversing = False
    elif action == 'clear_shapes':
        world.shapes.clear()
        world.history.clear()
        world.time_reversing = False
    elif action == 'toggle_auto_rotate': world.auto_rotate = not world.auto_rotate
    elif action == 'toggle_gravity': world.gravity_enabled = not world.gravity_enabled
    elif action == 'toggle_well': world.well_enabled = not world.well_enabled
    elif action == 'cycle_mouse_force': world.mouse_force_mode = (world.mouse_force_mode + 1) % 3
    elif action == 'toggle_collisions': world.collisions_enabled = not world.collisions_enabled
    elif action == 'cycle_bounds': world.bounds_mode = (world.bounds_mode + 1) % 3
    elif action == 'toggle_time_rev': world.time_reversing = not world.time_reversing
    elif action == 'toggle_slice': world.slice_enabled = not world.slice_enabled
    elif action == 'toggle_bloom': world.bloom_enabled = not world.bloom_enabled
    elif action == 'cycle_themes': world.theme_idx = (world.theme_idx + 1) % 3
    elif action == 'reset_scene': 
        new_world = make_world(world.dim, 3, prev=world)
        new_world.history.clear()
        new_world.time_reversing = False

    if new_world:
        new_world.gui_dirty = True
    elif action in TEXT_CHANGING_ACTIONS:
        world.gui_dirty = True
        
    return new_world

def get_gui_elements(world):
    if world.gui_dirty or not world.gui_elements:
        elements, total_h = build_gui_layout(world)
        world.gui_elements = elements
        world.gui_max_scroll = max(0, total_h - (GUI_BOTTOM - GUI_TOP))
        world.gui_dirty = False
    return world.gui_elements

def get_scrollbar_rect(world):
    if world.gui_max_scroll <= 0: return None
    track_h = GUI_BOTTOM - GUI_TOP
    handle_h = max(30, int(track_h * track_h / (world.gui_max_scroll + track_h)))
    handle_y = GUI_TOP + int((track_h - handle_h) * (world.gui_scroll / world.gui_max_scroll))
    return pygame.Rect(W - 12, handle_y, 8, handle_h)

def handle_gui_click(world, event):
    # Scrollbar check first
    sb_rect = get_scrollbar_rect(world)
    if sb_rect and sb_rect.collidepoint(event.pos):
        world.dragging_scrollbar = True
        return None

    elements = get_gui_elements(world)
    for el in elements:
        if isinstance(el, Slider):
            if el.handle_event(event, world):
                return None
        elif isinstance(el, Button):
            if el.handle_event(event):
                nw = handle_gui_action(world, el.action)
                if nw is not None:
                    return nw
                return None
    return None

def draw_gui(screen, world, fonts, theme, mouse_pos, dt):
    px = PLAY_W
    
    shadow = pygame.Surface((8, H), pygame.SRCALPHA)
    for i in range(8):
        alpha = int(60 * (1 - i/8))
        pygame.draw.line(shadow, (0, 0, 0, alpha), (i, 0), (i, H))
    screen.blit(shadow, (px - 8, 0))
    
    pygame.draw.rect(screen, theme["card"], (px, 0, GUI_W, H))
    pygame.draw.line(screen, theme["grid"], (px, 0), (px, H), 1)
    
    title = fonts['title'].render("N-D Physics", True, theme["fg"])
    screen.blit(title, (px + 20, 20))
    
    elements = get_gui_elements(world)
    world.gui_scroll = clamp(world.gui_scroll, 0, world.gui_max_scroll)
    
    clip_rect = pygame.Rect(px, GUI_TOP, GUI_W, GUI_BOTTOM - GUI_TOP)
    screen.set_clip(clip_rect)
    
    for el in elements:
        if isinstance(el, dict) and el["type"] == "HEADER":
            y_offset = el["y"] - world.gui_scroll
            lbl = fonts['small'].render(el["text"], True, theme["dim"])
            screen.blit(lbl, (px + 20, y_offset))
            pygame.draw.line(screen, theme["grid"], (px + 20, y_offset + 20), (px + GUI_W - 20, y_offset + 20), 1)
        elif isinstance(el, Slider):
            el.update(mouse_pos, dt, world)
            el.draw(screen, fonts, theme, world.gui_scroll)
        elif isinstance(el, Button):
            el.update(mouse_pos, dt)
            el.draw(screen, fonts, theme, world.gui_scroll)
            
    screen.set_clip(None)
    
    # Scrollbar
    track_h = GUI_BOTTOM - GUI_TOP
    pygame.draw.rect(screen, theme["grid"], (W - 12, GUI_TOP, 8, track_h), border_radius=4)
    sb_rect = get_scrollbar_rect(world)
    if sb_rect:
        pygame.draw.rect(screen, theme["btn_hover"], sb_rect, border_radius=4)
        if world.dragging_scrollbar or sb_rect.collidepoint(mouse_pos):
            pygame.draw.rect(screen, theme["accent"], sb_rect, 2, border_radius=4)
            
    # Instructions at bottom
    y = GUI_BOTTOM + 10
    txts = [
        "[Left-Click] Spawn Shape",
        "[Right-Drag] Grab Shape",
        "[Left-Drag] Rotate Plane",
        "[Arrows] Pan Camera",
        "[9/0] Roll Camera",
        "[SPACE] Shockwave",
        "[H] Hide GUI  [P] Pause"
    ]
    for t in txts:
        lbl = fonts['small'].render(t, True, theme["dim"])
        screen.blit(lbl, (px + 20, y))
        y += 14


def draw_world(screen, world, fonts, dt, mouse_pos, fps):
    theme = THEMES[world.theme_idx]
    
    sx = sy = 0
    if world.shake > 0:
        amp = world.shake * 8
        sx = random.uniform(-amp, amp)
        sy = random.uniform(-amp, amp)

    draw_background(screen, world, theme)

    cx, cy = PLAY_W // 2 + sx, H // 2 + sy
    R_cam = rotation_matrix(world.rot_angles, world.dim)
    
    draw_bounds(screen, world, R_cam, cx, cy, theme)
    draw_shapes(screen, world, R_cam, cx, cy, theme)
    draw_shockwaves(screen, world, R_cam, cx, cy, theme)
    draw_gimbal(screen, world, R_cam, cx, cy, theme)
    draw_mouse_force(screen, world, theme)

    if world.bloom_enabled:
        bloom_surf = pygame.transform.smoothscale(screen, (PLAY_W//4, H//4))
        bloom_surf = pygame.transform.smoothscale(bloom_surf, (PLAY_W, H))
        screen.blit(bloom_surf, (0, 0), special_flags=pygame.BLEND_ADD)

    if world.flash > 0:
        fs = pygame.Surface((PLAY_W, H), pygame.SRCALPHA)
        fs.fill((255, 255, 255, int(world.flash * 80)))
        screen.blit(fs, (0, 0))

    if world.paused:
        lbl = fonts['title'].render("PAUSED", True, theme["accent"])
        screen.blit(lbl, (20, 20))
    if world.time_reversing:
        lbl = fonts['title'].render("REWINDING", True, theme["accent"])
        screen.blit(lbl, (20, 60))
        
    # FPS Counter
    fps_lbl = fonts['small'].render(f"{int(fps)} FPS", True, theme["dim"])
    screen.blit(fps_lbl, (PLAY_W - 60, 10))

    if world.show_gui:
        draw_gui(screen, world, fonts, theme, mouse_pos, dt)


# ─── Main ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="N-D Physics Simulation")
    parser.add_argument('--dim', type=int, default=4, help='Starting dimension (2-8)')
    args = parser.parse_args()

    world = make_world(clamp(args.dim, 2, 8), 3)

    pygame.init()
    pygame.display.set_caption(f"{world.dim}D Physics Simulation")
    screen = pygame.display.set_mode((W, H), pygame.SCALED, vsync=1)
    clock = pygame.time.Clock()
    
    init_backgrounds()

    fonts = {
        'title': pygame.font.SysFont(["DejaVu Sans", "Arial", "Helvetica"], 22, bold=True),
        'small': pygame.font.SysFont(["DejaVu Sans", "Arial", "Helvetica"], 14),
    }

    running = True
    while running:
        dt = min(clock.tick(FPS) / 1000.0, 1 / 30)
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_h:
                    world.show_gui = not world.show_gui
                elif event.key == pygame.K_p:
                    world.paused = not world.paused
                elif event.key == pygame.K_BACKSPACE:
                    world.time_reversing = not world.time_reversing
                elif event.key == pygame.K_SPACE and mouse_pos[0] < PLAY_W:
                    add_shockwave(world)
                elif event.key == pygame.K_9:
                    world.cam_roll -= 0.1
                elif event.key == pygame.K_0:
                    world.cam_roll += 0.1
                elif event.key == pygame.K_r:
                    world.bloom_enabled = not world.bloom_enabled
                elif event.key == pygame.K_TAB:
                    pairs = list(world.rot_angles.keys())
                    if pairs:
                        if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                            world.active_pair_idx = (world.active_pair_idx - 1) % len(pairs)
                        else:
                            world.active_pair_idx = (world.active_pair_idx + 1) % len(pairs)
                        world.pair_change_t = 0.4

            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Release scrollbar if clicking elsewhere
                if event.button == 1:
                    world.dragging_scrollbar = False
                    
                if world.show_gui and event.pos[0] > PLAY_W:
                    if event.button == 1:
                        nw = handle_gui_click(world, event)
                        if nw is not None:
                            world = nw
                    elif event.button == 4:
                        world.gui_scroll = max(0, world.gui_scroll - 30)
                    elif event.button == 5:
                        world.gui_scroll = min(world.gui_max_scroll, world.gui_scroll + 30)
                else:
                    if event.button == 1:
                        world.drag_active = True
                        world.drag_last = event.pos
                        world.drag_dist = 0
                    elif event.button == 3:
                        if world.shapes:
                            R_cam = rotation_matrix(world.rot_angles, world.dim)
                            pos = np.array([s.pos for s in world.shapes])
                            sx, sy, depth, _ = project_points(pos, R_cam, world.focal, world.cam_dist, PLAY_W//2, H//2, world.cam_offset, world.cam_roll)
                            best_idx = -1
                            best_dist = 80.0
                            for i in range(len(world.shapes)):
                                if depth[i] < 0.5: continue
                                d = math.hypot(sx[i] - event.pos[0], sy[i] - event.pos[1])
                                if d < best_dist:
                                    best_dist = d
                                    best_idx = i
                            world.drag_idx = best_idx
                            world.mouse_pos = event.pos
                    elif event.button == 4:
                        world.cam_dist = clamp(world.cam_dist - 0.5, 5, 30)
                    elif event.button == 5:
                        world.cam_dist = clamp(world.cam_dist + 0.5, 5, 30)
                        
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    if world.dragging_scrollbar:
                        world.dragging_scrollbar = False
                    else:
                        elements = get_gui_elements(world)
                        slider_swallowed = False
                        for el in elements:
                            if isinstance(el, Slider) and el.dragging:
                                el.handle_event(event, world)
                                slider_swallowed = True
                                break
                                
                        if not slider_swallowed:
                            world.drag_active = False
                            if world.drag_dist < 6 and event.pos[0] < PLAY_W:
                                if len(world.shapes) < 15:
                                    R_cam = rotation_matrix(world.rot_angles, world.dim)
                                    p3 = np.array([ (event.pos[0] - (PLAY_W//2)) * world.cam_dist / world.focal,
                                                    -(event.pos[1] - (H//2)) * world.cam_dist / world.focal,
                                                    0.0 ])
                                    p_nd = np.zeros(world.dim)
                                    p_nd[:3] = p3
                                    target_pos = p_nd @ R_cam.T
                                    
                                    st = world.spawn_shape_type
                                    if st in [3, 5, 6] and world.dim != 4: st = 0
                                    world.shapes.append(init_polytope(st, world.dim, target_pos))
                                    world.flash = 0.4
                                    world.history.clear()
                                    world.time_reversing = False
                elif event.button == 3:
                    world.drag_idx = -1
                    
            elif event.type == pygame.MOUSEMOTION:
                world.mouse_pos = event.pos
                
                if world.dragging_scrollbar:
                    track_h = GUI_BOTTOM - GUI_TOP
                    handle_h = max(30, int(track_h * track_h / (world.gui_max_scroll + track_h)))
                    rel_y = event.pos[1] - GUI_TOP - handle_h / 2
                    ratio = clamp(rel_y / (track_h - handle_h), 0, 1)
                    world.gui_scroll = int(ratio * world.gui_max_scroll)
                    continue

                elements = get_gui_elements(world)
                slider_swallowed = False
                for el in elements:
                    if isinstance(el, Slider) and el.dragging:
                        el.handle_event(event, world)
                        slider_swallowed = True
                        break
                        
                if not slider_swallowed and world.drag_active and event.pos[0] < PLAY_W:
                    dx = event.pos[0] - world.drag_last[0]
                    world.drag_last = event.pos
                    world.drag_dist += abs(dx)
                    pairs = list(world.rot_angles.keys())
                    if pairs:
                        ap = pairs[world.active_pair_idx % len(pairs)]
                        world.rot_angles[ap] += dx * 0.008

        keys = pygame.key.get_pressed()
        move_speed = 8.0 * dt
        if keys[pygame.K_LEFT] and world.dim > 0: world.cam_offset[0] -= move_speed
        if keys[pygame.K_RIGHT] and world.dim > 0: world.cam_offset[0] += move_speed
        if keys[pygame.K_UP]:
            if (keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]) and world.dim > 2:
                world.cam_offset[2] += move_speed
            elif world.dim > 1:
                world.cam_offset[1] += move_speed
        if keys[pygame.K_DOWN]:
            if (keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]) and world.dim > 2:
                world.cam_offset[2] -= move_speed
            elif world.dim > 1:
                world.cam_offset[1] -= move_speed

        if world.auto_rotate and not world.paused:
            pairs = list(world.rot_angles.keys())
            if pairs:
                ap = pairs[world.active_pair_idx % len(pairs)]
                world.rot_angles[ap] += dt * 0.5

        step_physics(world, dt)
        draw_world(screen, world, fonts, dt, mouse_pos, clock.get_fps())
        
        if world.show_gui and mouse_pos[0] > PLAY_W:
            hover = False
            elements = get_gui_elements(world)
            for el in elements:
                if isinstance(el, Slider):
                    if el.get_hit_rect().collidepoint(mouse_pos):
                        hover = True
                        break
                elif isinstance(el, Button):
                    if el.rect.collidepoint(mouse_pos) and not (el.disabled_func and el.disabled_func()):
                        hover = True
                        break
            sb_rect = get_scrollbar_rect(world)
            if sb_rect and sb_rect.collidepoint(mouse_pos):
                hover = True
            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND if hover else pygame.SYSTEM_CURSOR_ARROW)
        else:
            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)

        pygame.display.flip()

    pygame.quit()
    sys.exit(0)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pygame.quit()
        sys.exit(0)