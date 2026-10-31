"""Convert one Prism node in an MCL-lite tree to Revol."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple


Vec3 = List[float]
EPS = 1e-12


def as_vec3(value: Any) -> Optional[Vec3]:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        return [float(value[0]), float(value[1]), float(value[2])]
    except (TypeError, ValueError):
        return None


def v_add(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return [float(a[0]) + float(b[0]), float(a[1]) + float(b[1]), float(a[2]) + float(b[2])]


def v_sub(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return [float(a[0]) - float(b[0]), float(a[1]) - float(b[1]), float(a[2]) - float(b[2])]


def v_mul(a: Sequence[float], s: float) -> Vec3:
    return [float(a[0]) * s, float(a[1]) * s, float(a[2]) * s]


def v_dot(a: Sequence[float], b: Sequence[float]) -> float:
    return float(a[0]) * float(b[0]) + float(a[1]) * float(b[1]) + float(a[2]) * float(b[2])


def v_cross(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return [
        float(a[1]) * float(b[2]) - float(a[2]) * float(b[1]),
        float(a[2]) * float(b[0]) - float(a[0]) * float(b[2]),
        float(a[0]) * float(b[1]) - float(a[1]) * float(b[0]),
    ]


def v_len(a: Sequence[float]) -> float:
    return math.sqrt(v_dot(a, a))


def v_norm(a: Sequence[float]) -> Optional[Vec3]:
    length = v_len(a)
    if length <= EPS:
        return None
    return v_mul(a, 1.0 / length)


def project_to_plane(direction: Sequence[float], normal: Sequence[float]) -> Vec3:
    return v_sub(direction, v_mul(normal, v_dot(direction, normal)))


def fallback_in_plane_axis(normal: Sequence[float]) -> Vec3:
    n = v_norm(normal) or [0.0, 0.0, 1.0]
    basis = [1.0, 0.0, 0.0] if abs(n[0]) < 0.9 else [0.0, 1.0, 0.0]
    axis = v_norm(v_cross(n, basis))
    return axis or [1.0, 0.0, 0.0]


def node_type(node: Any) -> Optional[str]:
    return node.get("type") if isinstance(node, dict) else None


def child_features(node: Dict[str, Any]) -> List[Dict[str, Any]]:
    features = node.get("features", [])
    if not isinstance(features, list):
        return []
    return [f for f in features if isinstance(f, dict)]


def iter_descendants(node: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    for child in child_features(node):
        yield child
        yield from iter_descendants(child)


def direct_face_of_prism(prism: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for child in child_features(prism):
        if node_type(child) == "Face":
            return child
    return None


def face_frame(face: Dict[str, Any]) -> Tuple[Vec3, Vec3, Vec3]:
    """Return the Face origin, normal, and in-plane X axis."""
    coords = face.get("coordinates", [])
    origin = as_vec3(coords[0]) if isinstance(coords, list) and len(coords) >= 1 else None
    normal = as_vec3(coords[1]) if isinstance(coords, list) and len(coords) >= 2 else None
    x_axis = as_vec3(coords[2]) if isinstance(coords, list) and len(coords) >= 3 else None

    origin = origin or [0.0, 0.0, 0.0]
    normal = v_norm(normal or [0.0, 0.0, 1.0]) or [0.0, 0.0, 1.0]

    if x_axis is not None:
        x_axis = v_norm(project_to_plane(x_axis, normal))

    x_axis = x_axis or fallback_in_plane_axis(normal)
    return origin, normal, x_axis


@dataclass
class RotationAxis:
    point1: Vec3
    point2: Vec3
    source: str


def find_longest_edge_axis(face: Dict[str, Any]) -> Optional[RotationAxis]:
    best_axis: Optional[RotationAxis] = None
    best_length = -1.0

    for node in iter_descendants(face):
        if node_type(node) != "Edge":
            continue

        coords = node.get("coordinates", [])
        if not isinstance(coords, list) or len(coords) < 2:
            continue

        p0 = as_vec3(coords[0])
        p1 = as_vec3(coords[1])
        if p0 is None or p1 is None:
            continue

        length = v_len(v_sub(p1, p0))
        if length <= EPS:
            continue

        if length > best_length:
            best_length = length
            best_axis = RotationAxis(
                point1=p0,
                point2=p1,
                source="longest_edge_on_face",
            )

    return best_axis


def find_first_circle_center(face: Dict[str, Any]) -> Optional[Vec3]:
    for node in iter_descendants(face):
        if node_type(node) != "Circle":
            continue
        coords = node.get("coordinates", [])
        if isinstance(coords, list) and coords:
            center = as_vec3(coords[0])
            if center is not None:
                return center
    return None


def calculate_rotation_axis_from_face(face: Dict[str, Any]) -> RotationAxis:
    """Try the longest Edge, Circle center, then Face frame as the rotation axis."""
    origin, _normal, x_axis = face_frame(face)

    edge_axis = find_longest_edge_axis(face)
    if edge_axis is not None:
        return edge_axis

    circle_center = find_first_circle_center(face)
    if circle_center is not None:
        return RotationAxis(
            point1=circle_center,
            point2=v_add(circle_center, x_axis),
            source="circle_center_with_face_x_axis",
        )

    return RotationAxis(
        point1=origin,
        point2=v_add(origin, x_axis),
        source="face_origin_with_face_x_axis",
    )


@dataclass
class PrismCandidate:
    node: Dict[str, Any]
    face: Dict[str, Any]
    path: str
    axis: RotationAxis


def collect_prism_candidates(data: Dict[str, Any]) -> List[PrismCandidate]:
    candidates: List[PrismCandidate] = []

    def visit(node: Any, path: str) -> None:
        if not isinstance(node, dict):
            return

        if node_type(node) == "Prism":
            face = direct_face_of_prism(node)
            if face is not None:
                axis = calculate_rotation_axis_from_face(face)
                candidates.append(
                    PrismCandidate(
                        node=node,
                        face=face,
                        path=path,
                        axis=axis,
                    )
                )

        features = node.get("features", [])
        if isinstance(features, list):
            for index, child in enumerate(features):
                visit(child, f"{path}.features[{index}]")

    visit(data, "$")
    return candidates


@dataclass
class ConversionResult:
    data: Dict[str, Any]
    candidate_count: int
    selected_index: int
    selected_path: str
    axis_source: str
    rotation_axis: List[Vec3]


def convert_one_prism_to_revol(
    data: Dict[str, Any],
    rng: random.Random,
    prism_index: Optional[int] = None,
) -> ConversionResult:
    """Replace one selected Prism and preserve its Face subtree."""
    if not isinstance(data, dict):
        raise ValueError("The JSON root must be an object.")

    if "geometry" in data:
        raise ValueError("Feature/geometry trees are not supported; expected MCL-lite JSON.")

    out = copy.deepcopy(data)
    candidates = collect_prism_candidates(out)

    if not candidates:
        raise ValueError("No Prism with a direct Face child was found.")

    if prism_index is None:
        selected_index = rng.randrange(len(candidates))
    else:
        if prism_index < 0 or prism_index >= len(candidates):
            raise IndexError(
                f"prism_index={prism_index} is out of range; "
                f"the number of available Prism candidates is {len(candidates)}."
            )
        selected_index = prism_index

    selected = candidates[selected_index]
    prism = selected.node
    axis = selected.axis

    prism["type"] = "Revol"
    prism["coordinates"] = [
        [float(v) for v in axis.point1],
        [float(v) for v in axis.point2],
    ]
    prism["parameters"] = {
        "d": None,
        "copy": False,
        "combin": False,
    }

    return ConversionResult(
        data=out,
        candidate_count=len(candidates),
        selected_index=selected_index,
        selected_path=selected.path,
        axis_source=axis.source,
        rotation_axis=prism["coordinates"],
    )


def derive_rng(seed: Optional[int], key: str) -> random.Random:
    """Create a deterministic RNG for one file and seed."""
    if seed is None:
        return random.Random()
    digest = hashlib.sha256(f"{seed}|{key}".encode("utf-8")).digest()
    derived_seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return random.Random(derived_seed)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        obj = json.load(fp)
    if not isinstance(obj, dict):
        raise ValueError(f"{path}: the JSON root is not an object.")
    return obj


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)


def default_output_dir_for_input(input_dir: Path) -> Path:
    return input_dir.resolve().parent / (input_dir.resolve().name + "_Revol")


def make_batch_output_path(input_file: Path, input_root: Path, output_root: Path) -> Path:
    rel = input_file.resolve().relative_to(input_root.resolve())
    return output_root / rel


def process_single_file(
    input_file: Path,
    output_file: Path,
    seed: Optional[int],
    prism_index: Optional[int],
) -> ConversionResult:
    data = read_json(input_file)
    rng = derive_rng(seed, str(input_file.resolve()))
    result = convert_one_prism_to_revol(data, rng=rng, prism_index=prism_index)
    write_json(output_file, result.data)
    return result


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


def get_top_level_folder(input_file: Path, input_root: Path) -> str:
    rel = input_file.resolve().relative_to(input_root.resolve())
    return rel.parts[0] if len(rel.parts) >= 2 else ""


def batch_process(
    input_dir: Path,
    output_dir: Optional[Path],
    seed: Optional[int],
    prism_index: Optional[int],
    start_folder: Optional[str] = None,
    end_folder: Optional[str] = None,
) -> None:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve() if output_dir is not None else default_output_dir_for_input(input_dir)

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

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

    output_dir.mkdir(parents=True, exist_ok=True)

    total_json = 0
    selected_json = 0
    converted = 0
    multi_prism_converted = 0
    skipped_by_folder = 0
    skipped_no_prism = 0
    failed = 0
    failure_examples: List[Tuple[str, str]] = []

    print(f"\nInput directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(
        "Conversion: replace one Prism per sample with Revol, "
        "keep its Face and other features, and preserve the filename."
    )
    if start_folder is None and end_folder is None:
        print("Folder filter: disabled; scan all subdirectories")
    else:
        print(
            "Folder filter: {} through {}".format(
                start_folder if start_folder is not None else "first",
                end_folder if end_folder is not None else "last",
            )
        )
    print()

    for input_file in sorted(input_dir.rglob("*.json")):
        total_json += 1

        top_folder = get_top_level_folder(input_file, input_dir)
        if start_folder is not None or end_folder is not None:
            # Apply the range filter only to top-level folders.
            if not top_folder or not folder_in_range(top_folder, start_folder, end_folder):
                skipped_by_folder += 1
                continue

        selected_json += 1
        output_file = make_batch_output_path(input_file, input_dir, output_dir)

        try:
            result = process_single_file(
                input_file=input_file,
                output_file=output_file,
                seed=seed,
                prism_index=prism_index,
            )
            converted += 1
            if result.candidate_count > 1:
                multi_prism_converted += 1
            print(
                f"[OK] {input_file.relative_to(input_dir)} -> "
                f"{output_file.relative_to(output_dir)} | "
                f"selected={result.selected_path}, "
                f"candidates={result.candidate_count}, "
                f"axis={result.axis_source}"
            )
        except ValueError as exc:
            if "No Prism with a direct Face child was found." in str(exc):
                skipped_no_prism += 1
                continue
            failed += 1
            if len(failure_examples) < 20:
                failure_examples.append((str(input_file.relative_to(input_dir)), str(exc)))
        except Exception as exc:
            failed += 1
            if len(failure_examples) < 20:
                failure_examples.append((str(input_file.relative_to(input_dir)), str(exc)))

    print("\n===== REVOL AUGMENT SUMMARY =====")
    print(f"Scanned JSON files: {total_json}")
    print(f"Skipped by folder range: {skipped_by_folder}")
    print(f"Entered conversion: {selected_json}")
    print(f"Converted: {converted}")
    print(f"Converted with multiple Prism candidates: {multi_prism_converted}")
    print(f"Skipped without a convertible Prism: {skipped_no_prism}")
    print(f"Failed: {failed}")

    if failure_examples:
        print("\n[Failure examples, up to 20]")
        for rel, err in failure_examples:
            print(f"  FAIL  {rel}  reason: {err}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replace one Prism in an MCL-lite JSON tree with Revol."
    )
    parser.add_argument(
        "-i",
        "--input",
        default="raw_json",
        help="Input MCL-lite JSON file or directory.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output JSON file or directory; defaults to <input>_Revol in batch mode.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--batch",
        dest="batch",
        action="store_true",
        default=True,
        help="Recursively process JSON files under the input directory (default).",
    )
    mode.add_argument(
        "--single",
        dest="batch",
        action="store_false",
        help="Process one input JSON file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used to select a Prism when --prism-index is omitted.",
    )
    parser.add_argument(
        "--prism-index",
        type=int,
        default=None,
        help="Zero-based Prism candidate index; a candidate is selected randomly by default.",
    )
    parser.add_argument(
        "--start-folder",
        type=str,
        required=True,
        help="First top-level folder ID to process, inclusive.",
    )
    parser.add_argument(
        "--end-folder",
        type=str,
        required=True,
        help="Last top-level folder ID to process, inclusive.",
    )

    args = parser.parse_args()
    input_path = Path(args.input)

    if args.batch:
        batch_process(
            input_dir=input_path,
            output_dir=Path(args.output) if args.output else None,
            seed=args.seed,
            prism_index=args.prism_index,
            start_folder=args.start_folder,
            end_folder=args.end_folder,
        )
        return

    if not input_path.is_file():
        raise SystemExit(f"Single-file input does not exist: {input_path}")
    if not args.output:
        raise SystemExit("Single-file mode requires -o / --output.")

    output_path = Path(args.output)
    if (output_path.exists() and output_path.is_dir()) or output_path.suffix.lower() != ".json":
        output_path = output_path / input_path.name

    result = process_single_file(
        input_file=input_path,
        output_file=output_path,
        seed=args.seed,
        prism_index=args.prism_index,
    )
    print(
        f"[OK] Converted: {input_path} -> {output_path}\n"
        f"     Output filename: {output_path.name}\n"
        f"     Prism candidates: {result.candidate_count}\n"
        f"     Selected index: {result.selected_index}\n"
        f"     Selected path: {result.selected_path}\n"
        f"     Axis source: {result.axis_source}\n"
        f"     Revol.coordinates: {result.rotation_axis}"
    )


if __name__ == "__main__":
    main()
