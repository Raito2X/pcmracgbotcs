from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np


PAD = -1.0
ZERO_EPS = 1e-5
MAX_TOKENS = 100

OPEN_TYPES = {
    "Group": "GROUP",
    "Fuse": "FUSE",
    "Cut": "CUT",
    "Common": "COMMON",
    "Prism": "PRISM",
    "Revol": "REVOL",
    "Thrusections": "THRUSECTIONS",
    "Face": "FACE",
    "Wire": "WIRE",
}
COMMANDS = {
    "ARC": 1,
    "CIRCLE": 2,
    "EDGE": 3,
    "WIRE": 4,
    "FACE": 5,
    "PRISM": 6,
    "REVOL": 7,
    "THRUSECTIONS": 8,
    "CUT": 9,
    "FUSE": 10,
    "COMMON": 11,
    "GROUP": 12,
    "END_CHILD": 13,
    "END_ROOT": 14,
}


def point3(value) -> tuple[float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return (
            float(value[0]),
            float(value[1]),
            float(value[2]) if len(value) > 2 else 0.0,
        )
    except (TypeError, ValueError):
        return None


def quantize(value: float, scale: float) -> int:
    if scale <= 0:
        raise ValueError("scale must be positive")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 128
    if abs(value) < ZERO_EPS:
        return 128
    return int(np.clip(round(((value / scale) + 1.0) * 128.0), 0, 255))


def quantize_direction(value: float) -> int:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 128
    if abs(value) < ZERO_EPS:
        return 128
    return int(np.clip(round((value + 1.0) * 128.0), 0, 255))


def pack_point(point, scale):
    return [quantize(value, scale) for value in point]


def pack_direction(direction):
    return [quantize_direction(value) for value in direction]


def empty_args(n_args):
    return [PAD] * n_args


def emit(name, args, commands, arguments):
    commands.append(COMMANDS[name])
    arguments.append(args)


def emit_edge(node, n_args, scale, commands, arguments):
    if n_args < 6:
        raise ValueError("EDGE requires at least 6 parameter slots")
    args = empty_args(n_args)
    points = [
        point
        for point in (point3(value) for value in node.get("coordinates", []))
        if point is not None
    ][:2]
    flat = [value for point in points for value in pack_point(point, scale)]
    args[: len(flat)] = flat
    emit("EDGE", args, commands, arguments)


def emit_wire_children(children, n_args, scale, commands, arguments):
    for child in children:
        if isinstance(child, dict) and child.get("type") in {"Edge", "Polygon"}:
            emit_edge(child, n_args, scale, commands, arguments)
        else:
            vectorize_node(child, n_args, scale, commands, arguments)


def vectorize_node(node: Any, n_args: int, scale: float, commands, arguments):
    """Emit commands in preorder and close each open node with END_CHILD."""
    if not isinstance(node, dict) or not isinstance(node.get("type"), str):
        return
    node_type = node["type"]
    children = node.get("features", [])
    if not isinstance(children, list):
        children = []

    if node_type in OPEN_TYPES:
        name = OPEN_TYPES[node_type]
        args = empty_args(n_args)
        coordinates = node.get("coordinates", [])

        if name == "FACE" and isinstance(coordinates, list) and len(coordinates) >= 3:
            origin = point3(coordinates[0]) or (0.0, 0.0, 0.0)
            normal = point3(coordinates[1]) or (0.0, 0.0, 1.0)
            x_axis = point3(coordinates[2]) or (1.0, 0.0, 0.0)
            args[:9] = (
                pack_point(origin, scale)
                + pack_direction(normal)
                + pack_direction(x_axis)
            )
        elif name == "REVOL" and isinstance(coordinates, list) and len(coordinates) >= 2:
            first = point3(coordinates[0])
            second = point3(coordinates[1])
            if first is not None and second is not None:
                args[:6] = pack_point(first, scale) + pack_point(second, scale)
        elif name == "PRISM" and isinstance(coordinates, list) and len(coordinates) >= 2:
            origin = point3(coordinates[0])
            direction = point3(coordinates[1])
            if origin is not None and direction is not None:
                args[:6] = pack_point(origin, scale) + pack_direction(direction)
            height = node.get("parameters", {}).get("height")
            if height is not None and n_args >= 7:
                args[6] = quantize(height, scale)

        emit(name, args, commands, arguments)
        if name == "WIRE":
            emit_wire_children(children, n_args, scale, commands, arguments)
        else:
            for child in children:
                vectorize_node(child, n_args, scale, commands, arguments)

        # END_CHILD marks the end of the current open node and its subtree.
        emit("END_CHILD", empty_args(n_args), commands, arguments)
        return

    if node_type == "Arc":
        args = empty_args(n_args)
        points = [
            point
            for point in (point3(value) for value in node.get("coordinates", [])[:3])
            if point is not None
        ]
        flat = [value for point in points for value in pack_point(point, scale)]
        args[: len(flat)] = flat
        emit("ARC", args, commands, arguments)
        return

    if node_type == "Circle":
        args = empty_args(n_args)
        points = [
            point
            for point in (
                point3(value) for value in node.get("coordinates", [])[:3]
            )
            if point is not None
        ]
        flat = []
        for index, point in enumerate(points):
            flat.extend(
                pack_direction(point) if index == 2 else pack_point(point, scale)
            )
        args[: len(flat)] = flat
        radius = node.get("parameters", {}).get("r")
        if radius is not None and n_args > 9:
            args[9] = quantize(radius, scale)
        emit("CIRCLE", args, commands, arguments)
        return

    for child in children:
        vectorize_node(child, n_args, scale, commands, arguments)


def vectorize_file(path: Path, n_args: int, scale: float, max_tokens: int):
    data = json.loads(path.read_text(encoding="utf-8"))
    commands, arguments = [], []
    vectorize_node(data, n_args, scale, commands, arguments)

    # The first END_ROOT terminates the program; later copies are padding.
    emit("END_ROOT", empty_args(n_args), commands, arguments)
    raw_length = len(commands)
    if raw_length > max_tokens:
        return None, raw_length
    if raw_length < max_tokens:
        padding = max_tokens - raw_length
        commands.extend([COMMANDS["END_ROOT"]] * padding)
        arguments.extend(empty_args(n_args) for _ in range(padding))
    return (commands, arguments), raw_length


def save_h5(path: Path, commands, arguments, scale, n_args, max_tokens):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as fp:
        fp.create_dataset(
            "commands",
            data=np.asarray(commands, dtype=np.int32),
            compression="gzip",
        )
        fp.create_dataset(
            "args",
            data=np.asarray(arguments, dtype=np.int16),
            compression="gzip",
        )
        fp.attrs["scale"] = float(scale)
        fp.attrs["n_args"] = int(n_args)
        fp.attrs["max_tokens"] = int(max_tokens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Input JSON file or directories; later directories override earlier ones.",
    )
    parser.add_argument("--output", default="data/cad_vec")
    parser.add_argument("--n_args", type=int, default=17)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--drop_manifest", default="dropped_overlong.tsv")
    args = parser.parse_args()

    input_paths = [Path(value).resolve() for value in args.input]
    single_input_file = len(input_paths) == 1 and input_paths[0].is_file()
    source_map = {}
    if single_input_file:
        source_map[Path(input_paths[0].name)] = input_paths[0]
    else:
        for input_path in input_paths:
            if not input_path.is_dir():
                raise SystemExit(f"input is not a directory: {input_path}")
            for source in input_path.rglob("*.json"):
                relative = source.relative_to(input_path)
                # Overlay directories by relative path; the last input wins.
                source_map[relative] = source

    sources = sorted(source_map.items())
    if not sources:
        raise SystemExit("no JSON files under the input paths")

    output_root = Path(args.output).resolve()
    dropped = []
    failed = []
    succeeded = 0
    for relative, source in sources:
        target = (
            output_root
            if single_input_file and output_root.suffix == ".h5"
            else output_root / relative.with_suffix(".h5")
        )
        try:
            result, raw_length = vectorize_file(
                source,
                args.n_args,
                args.scale,
                args.max_tokens,
            )
            if result is None:
                dropped.append((relative, raw_length))
                continue
            save_h5(target, *result, args.scale, args.n_args, args.max_tokens)
            succeeded += 1
        except Exception as exc:
            failed.append((relative, str(exc)))

    if dropped:
        manifest = (
            output_root.parent / args.drop_manifest
            if output_root.suffix == ".h5"
            else output_root / args.drop_manifest
        )
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(
            "path\traw_tokens\tmax_tokens\n"
            + "\n".join(
                f"{path}\t{length}\t{args.max_tokens}"
                for path, length in dropped
            ),
            encoding="utf-8",
        )
    for path, error in failed:
        print(f"[FAIL] {path}: {error}", file=sys.stderr)
    print(
        f"converted={succeeded} overlong={len(dropped)} failed={len(failed)}"
    )


if __name__ == "__main__":
    main()
