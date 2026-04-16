from pathlib import Path
from typing import Any

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def _class_name(names: Any, idx: int) -> str:
    if isinstance(names, dict):
        return str(names.get(idx, idx))
    if isinstance(names, (list, tuple)) and idx < len(names):
        return str(names[idx])
    return str(idx)


def log_per_class_map(trainer: Any) -> None:
    validator = getattr(trainer, "validator", None)
    box_metrics = getattr(getattr(validator, "metrics", None), "box", None)
    maps = getattr(box_metrics, "maps", None)      # per-class mAP50-95
    ap50s = getattr(box_metrics, "ap50", None)     # per-class mAP50
    if maps is None or len(maps) == 0 or ap50s is None or len(ap50s) == 0:
        return

    names = trainer.data.get("names", {})
    epoch = int(getattr(trainer, "epoch", 0)) + 1
    print(f"\n[Per-class AP] Epoch {epoch}")
    print("  id class                     mAP50   mAP50-95")
    for idx, ap in enumerate(maps):
        ap50 = float(ap50s[idx]) if idx < len(ap50s) else 0.0
        print(f"  {idx:02d} {_class_name(names, idx):<24} {ap50:.4f}  {float(ap):.4f}")


def train_ghost() -> None:
    model = YOLO(str(ROOT / "models" / "yolo11s_ghost.yaml"))
    model.add_callback("on_fit_epoch_end", log_per_class_map)

    model.train(
        data=str(ROOT / "dataset" / "data.yaml"),
        epochs=40,
        patience=12,
        imgsz=640,
        batch=8,
        device="cpu",
        workers=4,
        project=str(ROOT / "python" / "runs"),
        name="ghost_yolo11s",
        seed=42,
        deterministic=True,
        pretrained=False,
        exist_ok=True,
    )

    best_weight = ROOT / "python" / "runs" / "ghost_yolo11s" / "weights" / "best.pt"
    best_model = YOLO(str(best_weight))
    metrics = best_model.val(data=str(ROOT / "dataset" / "data.yaml"), split="val")
    print(f"Scheme D Ghost Metrics(mAP@50-95): {metrics.box.map:.4f}")


def main() -> None:
    print("Ghost training...")
    train_ghost()
    print("Ghost training finished.")


if __name__ == "__main__":
    main()
