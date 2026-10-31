"""Convert Prism sections into a Thrusections tree node."""

import argparse
import copy
import hashlib
import json
import math
import os
import random
from typing import Any, Dict, List, Optional, Tuple

EPS_ZERO = 1e-5

TILT2_PROB = 0.5
THIRD_SECTION_PROB = 0.8
TILT3_PROB = 2.0 / 3.0
SECOND_CIRCLE_PROB = 0.25
IS_SOLID_TRUE_PROB = 1.0

TILT_MIN_DEG = 0.0
TILT_MAX_DEG = 15.0

SECTION_SIZE_MIN_RATIO = 0.5
SECTION_SIZE_MAX_RATIO = 1.2

def v_sub(a, b):
    return [float(a[0]) - float(b[0]), float(a[1]) - float(b[1]), float(a[2]) - float(b[2])]


def v_add(a, b):
    return [float(a[0]) + float(b[0]), float(a[1]) + float(b[1]), float(a[2]) + float(b[2])]


def v_mul(a, s):
    return [float(a[0]) * s, float(a[1]) * s, float(a[2]) * s]


def v_dot(a, b):
    return float(a[0]) * float(b[0]) + float(a[1]) * float(b[1]) + float(a[2]) * float(b[2])


def v_cross(a, b):
    ax, ay, az = float(a[0]), float(a[1]), float(a[2])
    bx, by, bz = float(b[0]), float(b[1]), float(b[2])
    return [ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx]


def v_len(v):
    return math.sqrt(float(v[0]) ** 2 + float(v[1]) ** 2 + float(v[2]) ** 2)


def v_norm(v):
    L = v_len(v)
    return [0.0, 0.0, 0.0] if L == 0 else [float(v[0]) / L, float(v[1]) / L, float(v[2]) / L]


def axis_from_normal(n: List[float]) -> Tuple[str, int]:
    """Return the dominant coordinate axis of a normal and its direction."""
    nn = v_norm(n)
    ax = [abs(nn[0]), abs(nn[1]), abs(nn[2])]
    i = ax.index(max(ax))
    axis = ["x", "y", "z"][i]
    sign = 1 if nn[i] >= 0 else -1
    return axis, sign


def clamp_small_numbers(obj: Any, eps: float = EPS_ZERO) -> Any:
    if isinstance(obj, bool) or obj is None:
        return obj
    if isinstance(obj, (int, float)):
        x = float(obj)
        if abs(x) < eps:
            return 0
        if x == 0.0:
            return 0
        return obj
    if isinstance(obj, list):
        return [clamp_small_numbers(v, eps) for v in obj]
    if isinstance(obj, dict):
        return {k: clamp_small_numbers(v, eps) for k, v in obj.items()}
    return obj


def canonicalize_normal(n: List[float]) -> List[float]:
    nn = v_norm(n)
    if v_len(nn) < 1e-12:
        return [0.0, 0.0, 1.0]
    ax = [abs(nn[0]), abs(nn[1]), abs(nn[2])]
    i = ax.index(max(ax))
    if nn[i] < 0:
        nn = v_mul(nn, -1.0)
    return nn


def plane_signature(center: List[float], normal: List[float]) -> Tuple[Tuple[float, float, float], float]:
    """Identify a section plane by its canonical normal and plane constant."""
    n = canonicalize_normal(normal)
    d = v_dot(n, center)
    return (n[0], n[1], n[2]), float(d)

def is_wire_circle_node(node: Dict[str, Any]) -> bool:
    if node.get("type") != "Circle":
        return False
    params = node.get("parameters")
    return isinstance(params, dict) and params.get("wire_type") is True and ("r" in params)


def is_wire_node(node: Dict[str, Any]) -> bool:
    if node.get("type") != "Wire":
        return False
    feats = node.get("features")
    if not isinstance(feats, list):
        return False
    return any(
        isinstance(x, dict) and x.get("type") in {"Edge", "Arc"}
        for x in feats
    )

def get_center_and_normal_from_lite_circle(circle_node: Dict[str, Any]) -> Tuple[List[float], List[float]]:
    coords = circle_node.get("coordinates", [])
    if not isinstance(coords, list) or len(coords) < 3:
        raise ValueError("Circle.coordinates must contain at least three points")
    p1, p2, p3 = coords[0], coords[1], coords[2]
    center = [float(p1[0]), float(p1[1]), float(p1[2])]
    normal = v_sub(
        [float(p3[0]), float(p3[1]), float(p3[2])],
        [float(p2[0]), float(p2[1]), float(p2[2])]
    )
    return center, normal

def pt_close(a: List[float], b: List[float], tol: float = 1e-7) -> bool:
    return (abs(a[0] - b[0]) <= tol) and (abs(a[1] - b[1]) <= tol) and (abs(a[2] - b[2]) <= tol)


def edge_feature(p0: List[float], p1: List[float]) -> Dict[str, Any]:
    return {
        "type": "Edge",
        "features": [],
        "group": "creator",
        "coordinates": [p0, p1],
        "parameters": {"pline": False, "combin": False},
    }


def wire_feature_from_vertices(verts: List[List[float]]) -> Dict[str, Any]:
    edges: List[Dict[str, Any]] = []
    if len(verts) >= 2:
        for i in range(len(verts) - 1):
            edges.append(edge_feature(verts[i], verts[i + 1]))
        edges.append(edge_feature(verts[-1], verts[0]))

    return {
        "type": "Wire",
        "features": edges,
        "group": "operator",
        "coordinates": [[0, 0, 0], [0, 0, 1], [0, 1, 1]],
        "parameters": {
            "vertex_input": False,
            "edge_sort": False,
            "linear_to_lerance": 1e-06
        }
    }

def polygon_center_and_normal(verts: List[List[float]]) -> Tuple[List[float], List[float]]:
    c = [0.0, 0.0, 0.0]
    for p in verts:
        c = v_add(c, p)
    c = v_mul(c, 1.0 / float(len(verts)))

    n = None
    for i in range(len(verts) - 2):
        v1 = v_sub(verts[i + 1], verts[i])
        for j in range(i + 2, len(verts)):
            v2 = v_sub(verts[j], verts[i])
            cr = v_cross(v1, v2)
            if v_len(cr) > 1e-12:
                n = v_norm(cr)
                break
        if n is not None:
            break
    if n is None:
        n = [0.0, 0.0, 1.0]
    return c, n


def polygon_max_radius(verts: List[List[float]], center: List[float]) -> float:
    return float(max((v_len(v_sub(p, center)) for p in verts), default=0.0))

def sample_section_size(first_max_radius: float, rng: random.Random) -> float:
    return rng.uniform(
        SECTION_SIZE_MIN_RATIO * float(first_max_radius),
        SECTION_SIZE_MAX_RATIO * float(first_max_radius),
    )


def make_polygon_2d(
    shape: str,
    target_max_edge: float,
    rng: random.Random,
    tilt_m: Optional[float] = None,
) -> List[List[float]]:
    if target_max_edge <= 0:
        raise ValueError(f"target_max_edge must be positive, got {target_max_edge}")

    if shape == "square":
        raw = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]
    elif shape == "rectangle":
        half_h = rng.uniform(0.45, 1.0)
        raw = [[-1.0, -half_h], [1.0, -half_h], [1.0, half_h], [-1.0, half_h]]
    elif shape == "trapezoid":
        top = rng.uniform(0.40, 0.95)
        half_h = rng.uniform(0.45, 1.0)
        raw = [[-top, half_h], [top, half_h], [1.0, -half_h], [-1.0, -half_h]]
    else:
        raise ValueError(shape)

    m = float(tilt_m) if tilt_m is not None else 0.0
    raw_max_edge_3d = 0.0
    for i in range(len(raw)):
        a = raw[i]
        b = raw[(i + 1) % len(raw)]
        du = float(b[0]) - float(a[0])
        dv = float(b[1]) - float(a[1])
        raw_max_edge_3d = max(raw_max_edge_3d, math.sqrt(du * du + dv * dv + (m * dv) ** 2))

    if raw_max_edge_3d <= 1e-12:
        raise ValueError("Generated polygon has zero maximum 3D edge length")

    scale = float(target_max_edge) / raw_max_edge_3d
    return [[float(u) * scale, float(v) * scale] for u, v in raw]


def sample_tilt_m(rng):
    theta = rng.uniform(TILT_MIN_DEG, TILT_MAX_DEG)
    s = 1.0 if rng.random() < 0.5 else -1.0
    return math.tan(math.radians(theta)) * s


def map_2d_to_3d(uv, center, axis, rh, tilt_m):
    """Map 2D section points to an axis-aligned plane at a given offset."""
    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
    out = []
    if axis == "z":
        for u, v in uv:
            z = cz + rh if tilt_m is None else cz + (rh + tilt_m * v)
            out.append([cx + u, cy + v, z])
        return out
    if axis == "y":
        for u, v in uv:
            y = cy + rh if tilt_m is None else cy + (rh + tilt_m * v)
            out.append([cx + u, y, cz + v])
        return out
    if axis == "x":
        for u, v in uv:
            x = cx + rh if tilt_m is None else cx + (rh + tilt_m * v)
            out.append([x, cy + u, cz + v])
        return out
    raise ValueError(axis)


def build_lite_circle_feature(center, normal, r):
    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
    nx, ny, nz = float(normal[0]), float(normal[1]), float(normal[2])
    return {
        "type": "Circle",
        "features": [],
        "group": "creator",
        "coordinates": [
            [cx, cy, cz],
            [cx, cy, cz],
            [cx + nx, cy + ny, cz + nz],
            [0.0, 0.0, 0.0],
        ],
        "parameters": {"r": float(r), "wire_type": True},
    }


def base_normal_and_u_axis(axis: str, sign: int):
    if axis == "z":
        return [0.0, 0.0, float(sign)], [1.0, 0.0, 0.0]
    if axis == "y":
        return [0.0, float(sign), 0.0], [1.0, 0.0, 0.0]
    if axis == "x":
        return [float(sign), 0.0, 0.0], [0.0, 1.0, 0.0]
    raise ValueError(axis)


def rodrigues_rotate(v, k, theta):
    vx, vy, vz = v
    kx, ky, kz = k
    ct = math.cos(theta)
    st = math.sin(theta)
    cx = ky * vz - kz * vy
    cy = kz * vx - kx * vz
    cz = kx * vy - ky * vx
    dot = kx * vx + ky * vy + kz * vz
    return [
        vx * ct + cx * st + kx * dot * (1 - ct),
        vy * ct + cy * st + ky * dot * (1 - ct),
        vz * ct + cz * st + kz * dot * (1 - ct),
    ]


def center_shifted(center: List[float], axis: str, rh: float) -> List[float]:
    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
    if axis == "z":
        return [cx, cy, cz + rh]
    if axis == "y":
        return [cx, cy + rh, cz]
    if axis == "x":
        return [cx + rh, cy, cz]
    raise ValueError(axis)


def choose_circle_tilt_angle_deg(rng: random.Random) -> float:
    return rng.uniform(TILT_MIN_DEG, TILT_MAX_DEG)


def gen_second_section_single_profile(
    r1: float,
    center: List[float],
    source_normal: List[float],
    extrusion_normal: List[float],
    rng: random.Random,
    first_feature: Dict[str, Any],
    first_is_circle: bool,
    extrusion_height: float,
):
    """Generate the second section and optionally apply a tilt."""
    axis, sign = axis_from_normal(extrusion_normal)
    rh2 = sign * float(extrusion_height)
    do_tilt = (rng.random() < TILT2_PROB)
    target_size = sample_section_size(r1, rng)

    if first_is_circle and (rng.random() < SECOND_CIRCLE_PROB):
        r2 = target_size
        c2 = center_shifted(center, axis, rh2)
        N, U = base_normal_and_u_axis(axis, sign)
        theta_deg = 0.0
        if do_tilt:
            theta_deg = choose_circle_tilt_angle_deg(rng)
            s = 1.0 if rng.random() < 0.5 else -1.0
            N = rodrigues_rotate(N, U, math.radians(theta_deg) * s)
        feature2 = build_lite_circle_feature(c2, N, r2)
        meta2 = {
            "is_circle": True,
            "r2": float(r2),
            "c2": c2,
            "n2": N,
            "target_size": float(target_size),
            "tilt_deg": float(theta_deg),
            "axis": axis,
            "sign": sign,
            "rh2": rh2,
        }
        return feature2, do_tilt, rh2, meta2

    shape2 = rng.choice(["square", "rectangle", "trapezoid"])
    tilt_m = sample_tilt_m(rng) if do_tilt else None
    uv = make_polygon_2d(shape2, target_size, rng, tilt_m=tilt_m)
    coords = map_2d_to_3d(uv, center, axis, rh2, tilt_m)

    feature2 = wire_feature_from_vertices(coords)
    meta2 = {
        "is_circle": False,
        "target_size": float(target_size),
        "tilt_deg": float(math.degrees(math.atan(abs(tilt_m)))) if tilt_m is not None else 0.0,
        "axis": axis,
        "sign": sign,
        "rh2": rh2,
    }
    return feature2, do_tilt, rh2, meta2


def gen_third_circle_follow_second(
    r1: float,
    rng: random.Random,
    c2: List[float],
    n2: List[float],
    r2: float,
    extrusion_height: float,
):
    r3 = sample_section_size(r1, rng)

    n2u = v_norm(n2)
    if v_len(n2u) < 1e-10:
        n2u = [0.0, 0.0, 1.0]

    c3 = v_add(c2, v_mul(n2u, float(extrusion_height)))
    return build_lite_circle_feature(c3, n2u, r3)


def gen_third_section_from_axis(r1, center, axis, sign, rh2, rng, extrusion_height: float):
    """Generate a third circular or polygon section along the extrusion axis."""
    rh3 = rh2 + sign * float(extrusion_height)
    do_tilt = rng.random() < TILT3_PROB
    tilt_m = sample_tilt_m(rng) if do_tilt else None
    kind = rng.choice(["circle", "rectangle", "trapezoid"])
    c3 = center_shifted(center, axis, rh3)

    if kind == "circle":
        r3 = sample_section_size(r1, rng)
        N, U = base_normal_and_u_axis(axis, sign)
        if do_tilt:
            theta = rng.uniform(TILT_MIN_DEG, TILT_MAX_DEG)
            s = 1.0 if rng.random() < 0.5 else -1.0
            N = rodrigues_rotate(N, U, math.radians(theta) * s)
        return build_lite_circle_feature(c3, N, r3)

    uv = make_polygon_2d(kind, sample_section_size(r1, rng), rng, tilt_m=tilt_m)
    coords = map_2d_to_3d(uv, center, axis, rh3, tilt_m)
    return wire_feature_from_vertices(coords)


def build_thrusections(features_list, is_solid: bool):
    """Build a Thrusections node from a section list."""
    return {
        "type": "Thrusections",
        "features": features_list,
        "group": "operator",
        "coordinates": [],
        "parameters": {"is_solid": bool(is_solid), "ruled": False},
        "assist_ids": [],
        "model_global_param": {},
    }


def _as_point3(p: Any) -> Optional[List[float]]:
    if not isinstance(p, list) or len(p) < 3:
        return None
    try:
        return [float(p[0]), float(p[1]), float(p[2])]
    except (TypeError, ValueError):
        return None


def collect_wire_curve_points(wire_node: Dict[str, Any]) -> List[List[float]]:
    """Collect 3D Edge and Arc points from a Wire in curve order."""
    points: List[List[float]] = []
    for curve in wire_node.get("features", []):
        if not isinstance(curve, dict) or curve.get("type") not in {"Edge", "Arc"}:
            continue
        coords = curve.get("coordinates", [])
        if not isinstance(coords, list):
            continue
        for p in coords:
            p3 = _as_point3(p)
            if p3 is None:
                continue
            if not points or not pt_close(points[-1], p3):
                points.append(p3)

    if len(points) >= 2 and pt_close(points[-1], points[0]):
        points.pop()
    return points

def profile_node_to_feature_and_plane(
    node: Dict[str, Any]
) -> Tuple[Dict[str, Any], float, List[float], List[float], bool, Tuple[Tuple[float, float, float], float]]:
    """Return (profile, max_radius, center, normal, is_circle, plane_signature)."""
    if is_wire_circle_node(node):
        feat = copy.deepcopy(node)
        r = float(node["parameters"]["r"])
        center, normal = get_center_and_normal_from_lite_circle(node)
        sig = plane_signature(center, normal)
        return feat, r, center, normal, True, sig

    if is_wire_node(node):
        points = collect_wire_curve_points(node)
        if len(points) < 3:
            raise ValueError("Wire has too few 3D Edge/Arc points to estimate its plane")

        center, normal = polygon_center_and_normal(points)
        r = polygon_max_radius(points, center)
        if r <= 1e-12:
            raise ValueError("Wire has zero maximum radius and cannot form a loft section")

        feat = copy.deepcopy(node)
        sig = plane_signature(center, normal)
        return feat, r, center, normal, False, sig

    raise ValueError("Unknown profile type")


def collect_prism_slots(data: Any):
    """Collect Prism slots as (parent, key, node, path)."""
    slots = []

    def rec(node: Any, parent: Any = None, key: Any = None, path: str = "$"):
        if isinstance(node, dict):
            if node.get("type") == "Prism":
                slots.append((parent, key, node, path))
            feats = node.get("features")
            if isinstance(feats, list):
                for i, child in enumerate(feats):
                    rec(child, feats, i, f"{path}.features[{i}]")
        elif isinstance(node, list):
            for i, child in enumerate(node):
                rec(child, node, i, f"{path}[{i}]")

    rec(data)
    return slots


def collect_outermost_prism_slots(data: Any):
    """Collect Prisms outside nested Revol or Thrusections nodes."""
    slots = []

    def rec(node: Any, parent: Any = None, key: Any = None, path: str = "$", depth: int = 0):
        if isinstance(node, dict):
            node_type = node.get("type")

            if node_type == "Prism":
                slots.append((parent, key, node, path, depth))
                return

            if node_type in {"Revol", "Thrusections"}:
                return

            feats = node.get("features")
            if isinstance(feats, list):
                for i, child in enumerate(feats):
                    rec(child, feats, i, f"{path}.features[{i}]", depth + 1)

        elif isinstance(node, list):
            for i, child in enumerate(node):
                rec(child, node, i, f"{path}[{i}]", depth)

    rec(data)
    return slots


def direct_face_of_prism(prism: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    feats = prism.get("features")
    if not isinstance(feats, list):
        return None
    for child in feats:
        if isinstance(child, dict) and child.get("type") == "Face":
            return child
    return None


def collect_profiles_under_face(face: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Collect Wire and Circle profiles below a Face."""
    profiles: List[Dict[str, Any]] = []

    def rec(node: Any):
        if not isinstance(node, dict):
            return
        if is_wire_circle_node(node) or is_wire_node(node):
            profiles.append(node)
            return
        feats = node.get("features")
        if isinstance(feats, list):
            for child in feats:
                rec(child)

    rec(face)
    return profiles


def choose_primary_profile_for_prism(prism: Dict[str, Any]):
    """Return parsed geometry only when the Face contains one profile."""
    face = direct_face_of_prism(prism)
    if face is None:
        return None

    profiles = collect_profiles_under_face(face)
    if len(profiles) != 1:
        return None

    try:
        return profile_node_to_feature_and_plane(profiles[0])
    except Exception:
        return None


def get_prism_height(prism: Dict[str, Any]) -> float:
    params = prism.get("parameters", {})
    if not isinstance(params, dict) or "height" not in params:
        raise ValueError("Prism is missing parameters.height")
    try:
        height = float(params["height"])
    except (TypeError, ValueError):
        raise ValueError("Prism.parameters.height is not numeric")
    if not math.isfinite(height) or abs(height) <= EPS_ZERO:
        raise ValueError(f"Prism height is invalid or too small: {height}")
    return height


def get_prism_face_normal(prism: Dict[str, Any], fallback_normal: List[float]) -> List[float]:
    """Return the Prism Face normal, with a fallback vector."""
    def rec(node: Any) -> Optional[List[float]]:
        if not isinstance(node, dict):
            return None
        if node.get("type") == "Face":
            coords = node.get("coordinates", [])
            if isinstance(coords, list) and len(coords) >= 2:
                n = _as_point3(coords[1])
                if n is not None and v_len(n) > 1e-12:
                    return v_norm(n)
        feats = node.get("features", [])
        if isinstance(feats, list):
            for child in feats:
                n = rec(child)
                if n is not None:
                    return n
        return None

    n = rec(prism)
    if n is not None:
        return n
    n = v_norm(fallback_normal)
    return n if v_len(n) > 1e-12 else [0.0, 0.0, 1.0]


def build_augmented_thrusections_from_prism(
    prism: Dict[str, Any],
    rng: random.Random,
    primary_info=None,
    allow_third_section: bool = True,
) -> Dict[str, Any]:
    """Return a Thrusections node generated from one Prism."""
    info = primary_info or choose_primary_profile_for_prism(prism)
    if info is None:
        raise ValueError("Prism has no usable Wire or Circle profile")

    first_feature, r1, center, normal, first_is_circle, _ = info
    extrusion_height = get_prism_height(prism)
    extrusion_normal = get_prism_face_normal(prism, normal)

    second_feature, second_tilted, _offset2, meta2 = gen_second_section_single_profile(
        float(r1),
        center,
        normal,
        extrusion_normal,
        rng,
        first_feature=first_feature,
        first_is_circle=first_is_circle,
        extrusion_height=extrusion_height,
    )
    features_list = [copy.deepcopy(first_feature), clamp_small_numbers(second_feature, EPS_ZERO)]

    if allow_third_section:
        # Samples with multiple Prisms use two sections; single-Prism samples may add a third.
        if meta2.get("is_circle", False):
            third_feature = gen_third_circle_follow_second(
                r1=float(r1),
                rng=rng,
                c2=meta2["c2"],
                n2=meta2["n2"],
                r2=meta2["r2"],
                extrusion_height=extrusion_height,
            )
            features_list.append(clamp_small_numbers(third_feature, EPS_ZERO))
        elif (not second_tilted) and (rng.random() < THIRD_SECTION_PROB):
            third_feature = gen_third_section_from_axis(
                float(r1),
                center,
                meta2["axis"],
                meta2["sign"],
                meta2["rh2"],
                rng,
                extrusion_height=extrusion_height,
            )
            features_list.append(clamp_small_numbers(third_feature, EPS_ZERO))

    loft = build_thrusections(
        features_list,
        is_solid=(rng.random() < IS_SOLID_TRUE_PROB),
    )

    for k in ("name", "properties", "color"):
        if k in prism:
            loft[k] = copy.deepcopy(prism[k])

    return loft


def derive_file_seed(seed: Optional[int], rel_path: str) -> Optional[int]:
    if seed is None:
        return None
    digest = hashlib.sha256(f"{seed}:{rel_path}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def convert_one(
    input_path: str,
    output_path: str,
    seed: Optional[int] = None,
    prism_index: Optional[int] = None,
) -> Dict[str, Any]:
    """Convert one Prism in a JSON tree and write the result."""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rng = random.Random(seed)

    all_prism_total = len(collect_prism_slots(data))
    prism_slots = collect_outermost_prism_slots(data)
    source_prism_total = len(prism_slots)
    is_multi_prism = source_prism_total > 1

    candidates = []
    for parent, key, prism, path, depth in prism_slots:
        primary_info = choose_primary_profile_for_prism(prism)
        if primary_info is None:
            continue
        try:
            height = get_prism_height(prism)
        except ValueError:
            continue
        candidates.append((parent, key, prism, path, depth, primary_info, height))

    if not candidates:
        raise KeyError("No outermost Prism has both a valid profile and a valid height")

    if prism_index is None:
        # Rank by depth, then profile radius, then absolute height.
        selected_i = min(
            range(len(candidates)),
            key=lambda i: (
                candidates[i][4],
                -float(candidates[i][5][1]),
                -abs(float(candidates[i][6])),
            )
        )
    else:
        if prism_index < 0 or prism_index >= len(candidates):
            raise IndexError(
                f"prism_index={prism_index} is out of range; "
                f"the number of eligible outermost Prisms is {len(candidates)}"
            )
        selected_i = prism_index

    parent, key, prism, selected_path, selected_depth, primary_info, selected_height = candidates[selected_i]
    new_feature = build_augmented_thrusections_from_prism(
        prism,
        rng,
        primary_info=primary_info,
        allow_third_section=(not is_multi_prism),
    )

    if parent is None:
        data = new_feature
    else:
        parent[key] = new_feature

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return {
        "all_prism_total": all_prism_total,
        "outermost_prism_total": source_prism_total,
        "eligible_prisms": len(candidates),
        "selected_index": selected_i,
        "selected_path": selected_path,
        "selected_depth": selected_depth,
        "selected_height": selected_height,
        "multi_prism": is_multi_prism,
        "generated_sections": len(new_feature.get("features", [])),
        "first_max_radius": float(primary_info[1]),
        "size_ratio_range": [SECTION_SIZE_MIN_RATIO, SECTION_SIZE_MAX_RATIO],
        "tilt_deg_range": [TILT_MIN_DEG, TILT_MAX_DEG],
    }


def get_file_size_bytes(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return -1


def size_in_range(path: str, min_kb: float, max_kb: float) -> bool:
    size = get_file_size_bytes(path)
    if size < 0:
        return False

    min_bytes = int(min_kb * 1024)
    max_bytes = int(max_kb * 1024)

    return size > min_bytes and size < max_bytes


def default_output_dir_for_input(input_dir: str) -> str:
    input_dir_abs = os.path.abspath(input_dir)
    parent = os.path.dirname(input_dir_abs)
    base_name = os.path.basename(os.path.normpath(input_dir_abs))
    return os.path.join(parent, base_name + "_thrusections")



def normalize_folder_id(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = str(value).strip()
    return value if value else None


def folder_id_sort_key(value: str):
    value = str(value)
    if value.isdigit():
        return (0, int(value))
    return (1, value)


def folder_in_range(
    folder_name: str,
    start_folder: Optional[str],
    end_folder: Optional[str],
) -> bool:
    start_folder = normalize_folder_id(start_folder)
    end_folder = normalize_folder_id(end_folder)

    if start_folder is None and end_folder is None:
        return True

    folder_key = folder_id_sort_key(folder_name)

    if start_folder is not None:
        start_key = folder_id_sort_key(start_folder)
        if folder_key[0] != start_key[0] or folder_key < start_key:
            return False

    if end_folder is not None:
        end_key = folder_id_sort_key(end_folder)
        if folder_key[0] != end_key[0] or folder_key > end_key:
            return False

    return True


def get_selected_top_folder(rel_path: str) -> str:
    normalized = os.path.normpath(rel_path)
    parts = normalized.split(os.sep)
    return parts[0] if len(parts) >= 2 else ""


def batch_convert(
    input_dir: str,
    output_dir: Optional[str] = None,
    seed: Optional[int] = None,
    min_kb: float = 1.0,
    max_kb: float = 4.0,
    prism_index: Optional[int] = None,
    no_size_filter: bool = False,
    start_folder: Optional[str] = None,
    end_folder: Optional[str] = None,
) -> None:
    input_dir = os.path.abspath(input_dir)

    if output_dir is None:
        output_dir = default_output_dir_for_input(input_dir)
    else:
        output_dir = os.path.abspath(output_dir)

    if not os.path.exists(input_dir) or not os.path.isdir(input_dir):
        raise SystemExit("Input directory does not exist: {}".format(input_dir))

    start_folder = normalize_folder_id(start_folder)
    end_folder = normalize_folder_id(end_folder)

    if start_folder is not None and end_folder is not None:
        start_key = folder_id_sort_key(start_folder)
        end_key = folder_id_sort_key(end_folder)
        if start_key[0] != end_key[0]:
            raise SystemExit(
                "--start-folder and --end-folder must both be numeric or both be text"
            )
        if start_key > end_key:
            raise SystemExit("--start-folder must not be greater than --end-folder")

    if min_kb >= max_kb:
        raise SystemExit("--min-kb must be less than --max-kb")

    os.makedirs(output_dir, exist_ok=True)

    total_json = 0
    selected_json = 0
    converted = 0
    modified_multi_prism = 0
    skipped_by_folder = 0
    skipped_by_size = 0
    failed = 0
    failed_examples = []

    print("\nInput directory: {}".format(input_dir))
    print("Output directory: {}".format(output_dir))
    if start_folder is None and end_folder is None:
        print("Folder filter: disabled; scan all subdirectories")
    else:
        print("Folder filter: {} through {}".format(
            start_folder if start_folder is not None else "first",
            end_folder if end_folder is not None else "last",
        ))
    if no_size_filter:
        print("File-size filter: disabled")
    else:
        print("File-size range: {} KB < size < {} KB".format(min_kb, max_kb))
    print("Output layout: preserve relative paths from the input directory")
    print("Prism selection: outermost valid Prism, then largest primary profile\n")

    for root, dirs, files in os.walk(input_dir):
        for name in files:
            if not name.lower().endswith(".json"):
                continue

            total_json += 1

            in_path = os.path.join(root, name)
            rel_path = os.path.relpath(in_path, input_dir)

            top_folder = get_selected_top_folder(rel_path)
            if start_folder is not None or end_folder is not None:
                # Apply the range filter only to top-level folders.
                if not top_folder or not folder_in_range(top_folder, start_folder, end_folder):
                    skipped_by_folder += 1
                    continue

            if (not no_size_filter) and (not size_in_range(in_path, min_kb=min_kb, max_kb=max_kb)):
                skipped_by_size += 1
                continue

            selected_json += 1

            out_path = os.path.join(output_dir, rel_path)

            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            try:
                file_seed = derive_file_seed(seed, rel_path)
                meta = convert_one(
                    in_path,
                    out_path,
                    seed=file_seed,
                    prism_index=prism_index,
                )
                converted += 1
                if meta["multi_prism"]:
                    modified_multi_prism += 1
            except Exception as e:
                failed += 1
                if len(failed_examples) < 20:
                    failed_examples.append((rel_path, str(e)))

    print("Scanned JSON files: {}".format(total_json))
    print("Skipped by folder range: {}".format(skipped_by_folder))
    print("Entered conversion: {}".format(selected_json))
    print("Skipped by file size: {}".format(skipped_by_size))
    print("Converted: {}".format(converted))
    print("Converted with multiple outermost Prisms: {}".format(modified_multi_prism))
    print("Failed: {}".format(failed))

    if failed_examples:
        print("\n[Failure examples, up to 20]")
        for rel_path, err in failed_examples:
            print("  FAIL  {}  reason: {}".format(rel_path, err))

    print("\nDone.")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "-i",
        "--input",
        default="raw_json",
        help="Input JSON file or directory.",
    )

    ap.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output JSON file or directory; batch output defaults to a sibling directory.",
    )

    mode = ap.add_mutually_exclusive_group()
    mode.add_argument(
        "--batch",
        dest="batch",
        action="store_true",
        default=True,
        help="Recursively convert JSON files under the input directory (default).",
    )
    mode.add_argument(
        "--single",
        dest="batch",
        action="store_false",
        help="Process one input JSON file.",
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible section generation.",
    )

    ap.add_argument(
        "--prism-index",
        type=int,
        default=None,
        help="Zero-based outermost Prism index; choose the best candidate by default.",
    )

    ap.add_argument(
        "--size-filter",
        dest="no_size_filter",
        action="store_false",
        default=True,
        help="Enable file-size filtering; disabled by default.",
    )

    ap.add_argument(
        "--min-kb",
        type=float,
        default=1.0,
        help="Minimum input size in KB, exclusive.",
    )

    ap.add_argument(
        "--max-kb",
        type=float,
        default=4.0,
        help="Maximum input size in KB, exclusive.",
    )

    ap.add_argument(
        "--start-folder",
        type=str,
        required=True,
        help="First top-level folder ID to process, inclusive.",
    )

    ap.add_argument(
        "--end-folder",
        type=str,
        required=True,
        help="Last top-level folder ID to process, inclusive.",
    )

    args = ap.parse_args()

    if args.batch:
        batch_convert(
            input_dir=args.input,
            output_dir=args.output,
            seed=args.seed,
            min_kb=args.min_kb,
            max_kb=args.max_kb,
            prism_index=args.prism_index,
            no_size_filter=args.no_size_filter,
            start_folder=args.start_folder,
            end_folder=args.end_folder,
        )
    else:
        if args.output is None:
            raise SystemExit("Single-file mode requires -o / --output")

        meta = convert_one(
            args.input,
            args.output,
            seed=args.seed,
            prism_index=args.prism_index,
        )
        print(
            "Replaced outermost Prism: {selected_path}; "
            "all Prisms={all_prism_total}, outermost Prisms={outermost_prism_total}, "
            "candidates={eligible_prisms}, selected_index={selected_index}, "
            "depth={selected_depth}, height={selected_height}, "
            "generated_sections={generated_sections}".format(**meta)
        )


if __name__ == "__main__":
    main()



