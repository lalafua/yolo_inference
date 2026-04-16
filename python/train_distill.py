from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class DistillConfig:
    data: Path = ROOT / "dataset" / "data.yaml"
    teacher: Path = ROOT / "python" / "runs" / "baseline_yolo11s" / "weights" / "best.pt"
    student_cfg: Path = ROOT / "models" / "yolo11s_ghost.yaml"
    student_init: Path = ROOT / "python" / "runs" / "ghost_yolo11s" / "weights" / "best.pt"
    epochs: int = 40
    imgsz: int = 640
    batch: int = 8
    workers: int = 4
    device: str = "cpu"
    pseudo_conf: float = 0.35
    pseudo_iou: float = 0.7
    max_det: int = 300
    teacher_only: bool = False
    cache_dir: Path = ROOT / "dataset" / "distill_cache"
    project: Path = ROOT / "python" / "runs"
    name: str = "ghost_yolo11s_distill"
    seed: int = 42
    deterministic: bool = True


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid yaml structure: {path}")
    return data


def save_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def resolve_dataset_root(data_yaml: Path, data_cfg: dict[str, Any]) -> Path:
    root = data_cfg.get("path")
    if root is None:
        return data_yaml.parent
    root_path = Path(str(root))
    if not root_path.is_absolute():
        root_path = (data_yaml.parent / root_path).resolve()
    return root_path


def resolve_split_path(dataset_root: Path, split: Any) -> Path:
    split_path = Path(str(split))
    if split_path.is_absolute():
        return split_path
    return (dataset_root / split_path).resolve()


def image_files_from_split(split_path: Path) -> list[Path]:
    if split_path.is_file() and split_path.suffix.lower() == ".txt":
        images: list[Path] = []
        for line in split_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            p = Path(line)
            if not p.is_absolute():
                p = (split_path.parent / p).resolve()
            if p.exists() and p.suffix.lower() in IMAGE_SUFFIXES:
                images.append(p)
        return sorted(images)

    if split_path.is_dir():
        images = [p for p in split_path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
        return sorted(images)

    raise FileNotFoundError(f"Split path not found or unsupported: {split_path}")


def infer_label_path_from_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def parse_yolo_labels(label_path: Path) -> list[list[float]]:
    if not label_path.exists():
        return []
    rows: list[list[float]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        cols = line.split()
        if len(cols) < 5:
            continue
        try:
            cls = float(cols[0])
            x = float(cols[1])
            y = float(cols[2])
            w = float(cols[3])
            h = float(cols[4])
        except ValueError:
            continue
        rows.append([cls, x, y, w, h])
    return rows


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def xywhn_to_xyxy(box: list[float]) -> tuple[float, float, float, float]:
    x, y, w, h = box
    x1 = clamp01(x - w / 2.0)
    y1 = clamp01(y - h / 2.0)
    x2 = clamp01(x + w / 2.0)
    y2 = clamp01(y + h / 2.0)
    return x1, y1, x2, y2


def xyxy_to_xywhn(x1: float, y1: float, x2: float, y2: float, w: float, h: float) -> tuple[float, float, float, float]:
    bw = max(1e-6, x2 - x1)
    bh = max(1e-6, y2 - y1)
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    return clamp01(cx / w), clamp01(cy / h), clamp01(bw / w), clamp01(bh / h)


def iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 1e-12:
        return 0.0
    return inter / union


def merge_gt_and_teacher(
    gt_labels: list[list[float]],
    teacher_labels: list[list[float]],
    iou_dedup: float,
    use_gt: bool,
) -> list[list[float]]:
    merged = [r[:] for r in gt_labels] if use_gt else []

    existing_boxes: list[tuple[int, tuple[float, float, float, float]]] = []
    for row in merged:
        cls_id = int(row[0])
        existing_boxes.append((cls_id, xywhn_to_xyxy([row[1], row[2], row[3], row[4]])))

    for row in teacher_labels:
        cls_id = int(row[0])
        box = xywhn_to_xyxy([row[1], row[2], row[3], row[4]])
        duplicate = False
        for e_cls, e_box in existing_boxes:
            if e_cls != cls_id:
                continue
            if iou_xyxy(box, e_box) >= iou_dedup:
                duplicate = True
                break
        if not duplicate:
            merged.append(row)
            existing_boxes.append((cls_id, box))
    return merged


def write_labels(path: Path, labels: list[list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not labels:
        path.write_text("", encoding="utf-8")
        return
    lines = [f"{int(r[0])} {r[1]:.6f} {r[2]:.6f} {r[3]:.6f} {r[4]:.6f}" for r in labels]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def symlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def build_distill_dataset(cfg: DistillConfig) -> Path:
    data_cfg = load_yaml(cfg.data)
    if "train" not in data_cfg:
        raise KeyError("Dataset yaml must contain train split")

    dataset_root = resolve_dataset_root(cfg.data, data_cfg)
    train_path = resolve_split_path(dataset_root, data_cfg["train"])
    train_images = image_files_from_split(train_path)
    if not train_images:
        raise RuntimeError("No train images found for distillation.")

    teacher = YOLO(str(cfg.teacher))

    distill_root = cfg.cache_dir.resolve()
    if distill_root.exists():
        shutil.rmtree(distill_root)
    distill_images_root = distill_root / "images" / "train"
    distill_labels_root = distill_root / "labels" / "train"

    use_gt = not cfg.teacher_only
    print(f"Distillation dataset => {distill_root}")
    print(f"Train images: {len(train_images)} | Use GT labels: {use_gt}")

    for idx, src_img in enumerate(train_images, 1):
        rel = src_img.relative_to(train_path)
        dst_img = distill_images_root / rel
        dst_label = (distill_labels_root / rel).with_suffix(".txt")

        symlink_or_copy(src_img, dst_img)

        gt_rows = parse_yolo_labels(infer_label_path_from_image(src_img))

        pred = teacher.predict(
            source=str(src_img),
            imgsz=cfg.imgsz,
            conf=cfg.pseudo_conf,
            iou=cfg.pseudo_iou,
            max_det=cfg.max_det,
            device=cfg.device,
            verbose=False,
        )[0]

        teacher_rows: list[list[float]] = []
        boxes = getattr(pred, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.detach().cpu().numpy()
            clses = boxes.cls.detach().cpu().numpy()
            confs = boxes.conf.detach().cpu().numpy()
            h, w = pred.orig_shape
            for j in range(len(xyxy)):
                if float(confs[j]) < cfg.pseudo_conf:
                    continue
                x1, y1, x2, y2 = [float(v) for v in xyxy[j]]
                cx, cy, bw, bh = xyxy_to_xywhn(x1, y1, x2, y2, float(w), float(h))
                teacher_rows.append([int(clses[j]), cx, cy, bw, bh])

        merged = merge_gt_and_teacher(
            gt_labels=gt_rows,
            teacher_labels=teacher_rows,
            iou_dedup=0.6,
            use_gt=use_gt,
        )
        write_labels(dst_label, merged)

        if idx % 100 == 0 or idx == len(train_images):
            print(f"[{idx}/{len(train_images)}] pseudo labels generated")

    distill_yaml = {
        "path": str(distill_root),
        "train": "images/train",
        "val": str(resolve_split_path(dataset_root, data_cfg["val"])),
        "test": str(resolve_split_path(dataset_root, data_cfg["test"])) if "test" in data_cfg else None,
        "nc": data_cfg.get("nc"),
        "names": data_cfg.get("names"),
    }
    if distill_yaml["test"] is None:
        distill_yaml.pop("test")

    distill_yaml_path = distill_root / "data_distill.yaml"
    save_yaml(distill_yaml_path, distill_yaml)
    return distill_yaml_path


def train_student(cfg: DistillConfig, data_yaml: Path) -> None:
    student_model_path = cfg.student_init if cfg.student_init.exists() else cfg.student_cfg
    print(f"Student init: {student_model_path}")

    student = YOLO(str(student_model_path))
    student.train(
        data=str(data_yaml),
        epochs=cfg.epochs,
        patience=max(10, cfg.epochs // 4),
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        device=cfg.device,
        workers=cfg.workers,
        project=str(cfg.project),
        name=cfg.name,
        seed=cfg.seed,
        deterministic=cfg.deterministic,
        pretrained=False,
        exist_ok=True,
    )

    best_weight = cfg.project / cfg.name / "weights" / "best.pt"
    if best_weight.exists():
        best_model = YOLO(str(best_weight))
        metrics = best_model.val(data=str(data_yaml), split="val")
        print(f"Distilled Student Metrics(mAP@50-95): {metrics.box.map:.4f}")
    else:
        print("Warning: best.pt not found after training.")


def main() -> None:
    cfg = DistillConfig()

    if not cfg.data.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {cfg.data}")
    if not cfg.teacher.exists():
        raise FileNotFoundError(f"Teacher checkpoint not found: {cfg.teacher}")
    if not cfg.student_cfg.exists() and not cfg.student_init.exists():
        raise FileNotFoundError("Neither student cfg nor student init checkpoint exists.")

    print("Step1/2: build distillation dataset from teacher pseudo labels")
    data_yaml = build_distill_dataset(cfg)
    print(f"Generated distillation yaml: {data_yaml}")

    print("Step2/2: train student model on distilled dataset")
    train_student(cfg, data_yaml)


if __name__ == "__main__":
    main()
