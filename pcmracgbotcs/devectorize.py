from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import h5py


PAD = -1.0
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
ID_TO_COMMAND = {value: name for name, value in COMMANDS.items()}
OPEN_COMMANDS = {
    "GROUP",
    "FUSE",
    "CUT",
    "COMMON",
    "PRISM",
    "REVOL",
    "THRUSECTIONS",
    "FACE",
    "WIRE",
}
WIRE_COORDINATES = [[0, 0, 0], [0, 0, 1], [0, 1, 1]]


def is_pad(value):
    try:
        return float(value) == PAD
    except (TypeError, ValueError):
        return True


def dequantize(value, scale):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if value == PAD:
        return None
    return (value / 128.0 - 1.0) * scale


def dequantize_direction(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if value == PAD:
        return None
    return float(value) / 128.0 - 1.0


def normalize(vector):
    length = sum(value * value for value in vector) ** 0.5
    return vector if length < 1e-9 else [value / length for value in vector]


def read_point(row, start, scale):
    if start + 2 >= len(row) or all(is_pad(value) for value in row[start:start + 3]):
        return None
    point = []
    for value in row[start:start + 3]:
        point.append(0.0 if is_pad(value) else dequantize(value, scale))
    return point


def read_direction(row, start):
    if start + 2 >= len(row) or all(is_pad(value) for value in row[start:start + 3]):
        return None
    return normalize(
        [
            0.0 if is_pad(value) else dequantize_direction(value)
            for value in row[start:start + 3]
        ]
    )


def read_edge(row, scale):
    first = read_point(row, 0, scale)
    second = read_point(row, 3, scale)
    if first is None and second is None:
        return None
    first = first or [0.0, 0.0, 0.0]
    return [first, second or first]


def group_node():
    return {
        "type": "Group",
        "features": [],
        "group": "operator",
        "coordinates": [],
        "parameters": {
            "use_result_filter": False,
            "exclude": False,
            "return_index_list": [],
        },
        "model_global_param": {},
    }


def face_node(row, scale):
    return {
        "type": "Face",
        "features": [],
        "group": "operator",
        "coordinates": [
            read_point(row, 0, scale) or [0.0, 0.0, 0.0],
            read_direction(row, 3) or [0.0, 0.0, 1.0],
            read_direction(row, 6) or [1.0, 0.0, 0.0],
        ],
        "parameters": {"combin": False, "mode_flag": 3, "uv_list": []},
    }


def wire_node():
    return {
        "type": "Wire",
        "features": [],
        "group": "operator",
        "coordinates": [point[:] for point in WIRE_COORDINATES],
        "parameters": {
            "vertex_input": False,
            "edge_sort": False,
            "linear_to_lerance": 1e-6,
        },
    }


def prism_node(row, scale):
    return {
        "type": "Prism",
        "features": [],
        "group": "operator",
        "coordinates": [
            read_point(row, 0, scale) or [0.0, 0.0, 0.0],
            read_direction(row, 3) or [0.0, 0.0, 1.0],
        ],
        "parameters": {
            "height": dequantize(row[6], scale) if len(row) > 6 else None,
            "use_height": True,
            "auto_normal_vector": True,
            "combin": False,
        },
    }


def revol_node(row, scale):
    return {
        "type": "Revol",
        "features": [],
        "group": "operator",
        "coordinates": [
            read_point(row, 0, scale) or [0.0, 0.0, 0.0],
            read_point(row, 3, scale) or [0.0, 0.0, 1.0],
        ],
        "parameters": {"d": None, "copy": False, "combin": False},
    }


def circle_node(row, scale):
    return {
        "type": "Circle",
        "features": [],
        "group": "creator",
        "coordinates": [
            read_point(row, 0, scale) or [0.0, 0.0, 0.0],
            read_point(row, 3, scale) or [0.0, 0.0, 0.0],
            read_direction(row, 6) or [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ],
        "parameters": {
            "r": dequantize(row[9], scale) if len(row) > 9 else None,
            "wire_type": True,
        },
    }


def arc_node(row, scale):
    return {
        "type": "Arc",
        "features": [],
        "group": "creator",
        "coordinates": [
            read_point(row, start, scale) or [0.0, 0.0, 0.0]
            for start in (0, 3, 6)
        ],
        "parameters": {
            "r": 0.0,
            "alpha1": 0.0,
            "alpha2": 0.0,
            "circle_mode": "0p",
            "arc_mode": "3p",
            "sense": True,
        },
    }


def edge_node(points):
    return {
        "type": "Edge",
        "features": [],
        "group": "creator",
        "coordinates": points,
        "parameters": {"pline": False, "combin": False},
    }


def operator_node(name):
    if name == "THRUSECTIONS":
        return {
            "type": "Thrusections",
            "features": [],
            "group": "operator",
            "coordinates": [],
            "parameters": {"is_solid": True, "ruled": False},
            "assist_ids": [],
            "model_global_param": {},
        }
    return {
        "type": name.title(),
        "features": [],
        "group": "operator",
        "coordinates": [],
        "parameters": {},
    }


def append_node(node, root, stack, is_open):
    if root is None:
        if is_open:
            stack.append(node)
            return node, stack
        root = group_node()
        root["features"].append(node)
        stack.append(root)
        return root, stack

    if not stack:
        wrapper = group_node()
        wrapper["features"].extend([root, node])
        stack.append(node if is_open else wrapper)
        return wrapper, stack

    stack[-1]["features"].append(node)
    if is_open:
        stack.append(node)
    return root, stack


def devectorize(commands, arguments, scale):
    """Restore the tree; END_CHILD pops a node and the first END_ROOT stops parsing."""
    root = None
    stack = []
    width = len(arguments[0]) if arguments else 17
    for index, raw_command in enumerate(commands):
        command = int(raw_command)
        name = ID_TO_COMMAND.get(command)
        row = arguments[index] if index < len(arguments) else [PAD] * width
        if name == "END_ROOT":
            # Later END_ROOT tokens are fixed-length padding.
            if stack:
                open_names = [node.get("type") for node in stack]
                print(f"unclosed nodes at row {index}: {open_names}", file=sys.stderr)
            break
        if name == "END_CHILD":
            if stack:
                stack.pop()
            else:
                print(f"extra END_CHILD at row {index}", file=sys.stderr)
            continue
        if name in OPEN_COMMANDS:
            if name == "GROUP":
                node = group_node()
            elif name == "FACE":
                node = face_node(row, scale)
            elif name == "WIRE":
                node = wire_node()
            elif name == "PRISM":
                node = prism_node(row, scale)
            elif name == "REVOL":
                node = revol_node(row, scale)
            else:
                node = operator_node(name)
            root, stack = append_node(node, root, stack, True)
            continue
        if name == "ARC":
            root, stack = append_node(arc_node(row, scale), root, stack, False)
            continue
        if name == "CIRCLE":
            root, stack = append_node(circle_node(row, scale), root, stack, False)
            continue
        if name == "EDGE":
            points = read_edge(row, scale)
            if points is not None:
                root, stack = append_node(edge_node(points), root, stack, False)
            continue
        print(f"unknown command {command} at row {index}", file=sys.stderr)

    root = root or group_node()
    if root.get("type") != "Group":
        wrapper = group_node()
        wrapper["features"].append(root)
        root = wrapper
    return root


def read_h5(path):
    with h5py.File(path, "r") as fp:
        if "commands" in fp and "args" in fp:
            commands = fp["commands"][:].tolist()
            arguments = fp["args"][:].tolist()
        else:
            raise KeyError(f"{path} must contain 'commands' and 'args'")
        scale = float(fp.attrs["scale"]) if "scale" in fp.attrs else None
    return commands, arguments, scale


def default_output(input_path):
    if input_path.is_file():
        return input_path.with_suffix(".json")
    return input_path.with_name(input_path.name + "_json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output")
    parser.add_argument("--scale", type=float)
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    files = [input_path] if input_path.is_file() else sorted(input_path.rglob("*.h5"))
    if not files:
        raise SystemExit(f"no HDF5 files under {input_path}")
    output_root = Path(args.output).resolve() if args.output else default_output(input_path)

    succeeded = 0
    failed = 0
    for source in files:
        target = (
            output_root
            if len(files) == 1 and output_root.suffix == ".json"
            else output_root / source.relative_to(input_path).with_suffix(".json")
        )
        try:
            commands, arguments, stored_scale = read_h5(source)
            scale = stored_scale if stored_scale is not None else (args.scale or 1.0)
            tree = devectorize(commands, arguments, scale)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(tree, ensure_ascii=False, indent=args.indent),
                encoding="utf-8",
            )
            succeeded += 1
        except Exception as exc:
            failed += 1
            print(f"[FAIL] {source}: {exc}", file=sys.stderr)
    print(f"converted={succeeded} failed={failed}")


if __name__ == "__main__":
    main()
