from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class QuantConfig:
    source_pt: Path = ROOT / "python" / "runs" / "ghost_yolo11s_distill" / "weights" / "best.pt"
    data_yaml: Path = ROOT / "dataset" / "data.yaml"
    output_dir: Path = ROOT / "python" / "runs" / "onnx_quant"
    fp32_name: str = "ghost_distill_fp32.onnx"
    int8_name: str = "ghost_distill_int8.onnx"
    imgsz: int = 640
    device: str = "cpu"
    val_batch: int = 8
    latency_warmup: int = 10
    latency_runs: int = 80
    export_simplify: bool = True
    export_dynamic: bool = False
    export_opset: int = 12


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def export_fp32_onnx(cfg: QuantConfig) -> Path:
    model = YOLO(str(cfg.source_pt))
    exported = model.export(
        format="onnx",
        imgsz=cfg.imgsz,
        simplify=cfg.export_simplify,
        dynamic=cfg.export_dynamic,
        opset=cfg.export_opset,
    )

    src_path = Path(str(exported)).resolve()
    dst_path = cfg.output_dir / cfg.fp32_name
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if src_path != dst_path:
        shutil.copy2(src_path, dst_path)
    return dst_path


def quantize_to_int8(fp32_onnx: Path, int8_onnx: Path) -> None:
    int8_onnx.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        model_input=str(fp32_onnx),
        model_output=str(int8_onnx),
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
    )


def val_metrics(model_path: Path, cfg: QuantConfig) -> dict[str, Optional[float]]:
    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(cfg.data_yaml),
        split="val",
        batch=cfg.val_batch,
        imgsz=cfg.imgsz,
        device=cfg.device,
        verbose=False,
    )
    return {
        "best_precision": float(getattr(metrics.box, "mp", 0.0)),
        "best_recall": float(getattr(metrics.box, "mr", 0.0)),
        "best_map50": float(getattr(metrics.box, "map50", 0.0)),
        "best_map50_95": float(getattr(metrics.box, "map", 0.0)),
    }


def benchmark_onnx_latency(model_path: Path, cfg: QuantConfig) -> tuple[float, float]:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name

    # Use fixed NCHW input for fair comparison across models.
    x = np.random.rand(1, 3, cfg.imgsz, cfg.imgsz).astype(np.float32)

    for _ in range(max(cfg.latency_warmup, 0)):
        session.run(None, {input_name: x})

    timings: list[float] = []
    for _ in range(max(cfg.latency_runs, 1)):
        t0 = time.perf_counter()
        session.run(None, {input_name: x})
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000.0)

    mean_ms = float(sum(timings) / len(timings))
    variance = float(sum((v - mean_ms) ** 2 for v in timings) / len(timings))
    std_ms = variance ** 0.5
    return mean_ms, std_ms


def summary_row(run_name: str, model_path: Path, cfg: QuantConfig) -> dict[str, Any]:
    metrics = val_metrics(model_path, cfg)
    latency_ms, latency_std_ms = benchmark_onnx_latency(model_path, cfg)
    size_mb = file_size_mb(model_path)
    best_map50_95 = float(metrics["best_map50_95"] or 0.0)
    return {
        "run": run_name,
        "run_dir": str(model_path.parent),
        "model": str(model_path),
        "best_precision": metrics["best_precision"],
        "best_recall": metrics["best_recall"],
        "best_map50": metrics["best_map50"],
        "best_map50_95": metrics["best_map50_95"],
        "model_size_mb": size_mb,
        "best_pt_size_mb": size_mb,
        "latency_ms": latency_ms,
        "latency_std_ms": latency_std_ms,
        "benchmark_device": "cpu",
        "map50_95_per_mb": (best_map50_95 / size_mb) if size_mb > 0 else None,
        "source": "onnx",
    }


def main() -> None:
    cfg = QuantConfig()

    if not cfg.source_pt.exists():
        raise FileNotFoundError(f"Source checkpoint not found: {cfg.source_pt}")
    if not cfg.data_yaml.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {cfg.data_yaml}")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    print("Step1/3: export FP32 ONNX")
    fp32_onnx = export_fp32_onnx(cfg)
    print(f"FP32 ONNX: {fp32_onnx}")

    print("Step2/3: quantize INT8 ONNX")
    int8_onnx = cfg.output_dir / cfg.int8_name
    quantize_to_int8(fp32_onnx, int8_onnx)
    print(f"INT8 ONNX: {int8_onnx}")

    print("Step3/3: evaluate and benchmark ONNX models")
    rows = [
        summary_row("ghost_distill_fp32_onnx", fp32_onnx, cfg),
        summary_row("ghost_distill_int8_onnx", int8_onnx, cfg),
    ]

    summary_path = cfg.output_dir / "summary.json"
    summary_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved ONNX summary: {summary_path}")


if __name__ == "__main__":
    main()
