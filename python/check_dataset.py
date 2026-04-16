#!/usr/bin/env python3
"""Validate YOLO dataset yaml and label files.

Checks:
1) nc matches names length.
2) class ids in labels are within [0, nc-1].
3) per-split, per-class instance statistics.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate YOLO dataset labels and config")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("dataset/data.yaml"),
        help="Path to YOLO data.yaml (default: dataset/data.yaml)",
    )
    return parser.parse_args()


def load_data_yaml(data_yaml: Path) -> Dict:
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")
    with data_yaml.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def labels_dir_from_images_field(root: Path, field: str) -> Path:
    p = Path(field)
    if not p.is_absolute():
        p = (root / p).resolve()

    parts = list(p.parts)
    if "images" in parts:
        idx = parts.index("images")
        return Path(*parts[:idx], "labels", *parts[idx + 1 :])

    # Fallback: use sibling labels folder if path does not include /images.
    return p.parent / "labels"


def scan_split_labels(
    split_name: str,
    labels_dir: Path,
    nc: int,
    names: List[str],
) -> Tuple[Dict[int, int], List[str], int, int]:
    counts = defaultdict(int)
    errors: List[str] = []
    files_count = 0
    lines_count = 0

    if not labels_dir.exists():
        errors.append(f"[{split_name}] labels dir not found: {labels_dir}")
        return counts, errors, files_count, lines_count

    txt_files = sorted(labels_dir.rglob("*.txt"))
    files_count = len(txt_files)

    for txt in txt_files:
        with txt.open("r", encoding="utf-8") as f:
            for ln, raw in enumerate(f, start=1):
                line = raw.strip()
                if not line:
                    continue
                lines_count += 1
                items = line.split()
                if len(items) < 5:
                    errors.append(f"[{split_name}] malformed line ({txt}:{ln}): {line}")
                    continue

                try:
                    cls_id = int(float(items[0]))
                except ValueError:
                    errors.append(f"[{split_name}] invalid class id ({txt}:{ln}): {items[0]}")
                    continue

                if cls_id < 0 or cls_id >= nc:
                    errors.append(
                        f"[{split_name}] out-of-range class id ({txt}:{ln}): {cls_id}, expected [0, {nc - 1}]"
                    )
                    continue

                # Check xywh range for normalized labels.
                try:
                    x, y, w, h = map(float, items[1:5])
                except ValueError:
                    errors.append(f"[{split_name}] invalid xywh ({txt}:{ln}): {' '.join(items[1:5])}")
                    continue

                if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                    errors.append(f"[{split_name}] xywh out of [0,1] ({txt}:{ln}): {x} {y} {w} {h}")
                    continue

                counts[cls_id] += 1

    # Ensure all classes appear in output table.
    for i in range(nc):
        counts[i] += 0

    return counts, errors, files_count, lines_count


def print_config_summary(data_yaml: Path, data: Dict, nc: int, names: List[str]) -> bool:
    ok = True
    print("=" * 72)
    print(f"Config: {data_yaml}")
    print(f"nc: {nc}")
    print(f"names_count: {len(names)}")
    print("names:")
    for i, n in enumerate(names):
        print(f"  {i}: {n}")

    if nc != len(names):
        ok = False
        print("[ERROR] nc != len(names)")
    else:
        print("[OK] nc matches len(names)")
    print("=" * 72)
    return ok


def main() -> int:
    args = parse_args()
    data_yaml = args.data.resolve()
    data = load_data_yaml(data_yaml)

    nc = int(data.get("nc", -1))
    names = data.get("names", [])
    if not isinstance(names, list):
        print("[ERROR] names must be a list")
        return 2
    if nc <= 0:
        print("[ERROR] nc must be > 0")
        return 2

    ok_config = print_config_summary(data_yaml, data, nc, names)

    root = data_yaml.parent
    split_fields = ["train", "val", "test"]
    split_stats: Dict[str, Dict[int, int]] = {}
    all_errors: List[str] = []

    for split in split_fields:
        if split not in data:
            continue

        labels_dir = labels_dir_from_images_field(root, str(data[split]))
        counts, errors, file_n, line_n = scan_split_labels(split, labels_dir, nc, names)
        split_stats[split] = counts
        all_errors.extend(errors)

        print(f"[{split}] labels_dir: {labels_dir}")
        print(f"[{split}] label_files: {file_n}, objects: {line_n}")

    print("=" * 72)
    print("Per-split class counts")
    print("(rows=class, cols=splits)")

    available_splits = [s for s in split_fields if s in split_stats]
    header = "class_id class_name".ljust(36) + " ".join(f"{s:>10}" for s in available_splits)
    print(header)
    print("-" * len(header))

    for cid in range(nc):
        row = f"{cid:>7} {names[cid]:<27}" + " ".join(
            f"{split_stats[s].get(cid, 0):>10}" for s in available_splits
        )
        print(row)

    print("=" * 72)
    if all_errors:
        print(f"Found {len(all_errors)} errors:")
        for e in all_errors[:200]:
            print(f"  - {e}")
        if len(all_errors) > 200:
            print(f"  ... truncated, {len(all_errors) - 200} more")
    else:
        print("No label format/range/class-id errors found.")

    no_val_classes = []
    if "val" in split_stats:
        for cid in range(nc):
            if split_stats["val"].get(cid, 0) == 0:
                no_val_classes.append(f"{cid}:{names[cid]}")
    if no_val_classes:
        print("[WARN] classes with 0 instances in val:")
        print("  " + ", ".join(no_val_classes))

    final_ok = ok_config and not all_errors
    print("=" * 72)
    print("PASS" if final_ok else "FAIL")
    return 0 if final_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
