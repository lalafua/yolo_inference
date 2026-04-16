from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def train_model() -> None:
    # Baseline should be a stronger reference model; keep this configurable.
    model = YOLO(str(ROOT / "models" / "yolo11s.pt"))

    model.train(
        data=str(ROOT / "dataset" / "data.yaml"),
        epochs=40,
        patience=12,
        imgsz=640,
        batch=8,
        device="cpu",
        workers=2,
        project=str(ROOT / "python" / "runs"),
        name="baseline_yolo11s",
        seed=42,
        deterministic=True,
        pretrained=True,
        exist_ok=True,
    )

    best_weight = ROOT / "python" / "runs" / "baseline_yolo11s" / "weights" / "best.pt"
    best_model = YOLO(str(best_weight))
    metrics = best_model.val(data=str(ROOT / "dataset" / "data.yaml"), split="val")
    print(f"Baseline Metrics(mAP@50-95): {metrics.box.map:.4f}")


def main() -> None:
    print("Start baseline training...")
    best_weight = ROOT / "python" / "runs" / "baseline_yolo11s" / "weights" / "best.pt"
    best_model = YOLO(str(best_weight))
    metrics = best_model.val(data=str(ROOT / "dataset" / "data.yaml"), split="val")
    print(f"Baseline Metrics(mAP@50-95): {metrics.box.map:.4f}")
    # train_model()
    print("Baseline training finished.")


if __name__ == "__main__":
    main()
