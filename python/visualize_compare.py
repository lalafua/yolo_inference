from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, cast

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

import torch
from ultralytics import YOLO

try:
    from thop import profile as thop_profile
except Exception:  # pragma: no cover
    thop_profile = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS: List[Tuple[str, Path]] = [
    ("baseline", ROOT / "python" / "runs" / "baseline_yolo11s"),
    ("ghost", ROOT / "python" / "runs" / "ghost_yolo11s"),
    ("ghost_distill", ROOT / "python" / "runs" / "ghost_yolo11s_distill"),
]


@dataclass
class CompareConfig:
    runs: List[Tuple[str, Path]]
    output: Path
    extra_metrics: Optional[Path]
    external_rows: Optional[Path]
    benchmark_device: str
    benchmark_imgsz: int
    benchmark_warmup: int
    benchmark_runs: int
    skip_benchmark: bool


DEFAULT_CONFIG = CompareConfig(
    runs=DEFAULT_RUNS,
    output=ROOT / "python" / "runs" / "comparison",
    extra_metrics=None,
    external_rows=ROOT / "python" / "runs" / "onnx_quant" / "summary.json",
    benchmark_device="cpu",
    benchmark_imgsz=640,
    benchmark_warmup=10,
    benchmark_runs=50,
    skip_benchmark=False,
)


@dataclass
class RunSummary:
    name: str
    run_dir: Path
    model: str
    epochs: int
    best_epoch: int
    best_precision: float
    best_recall: float
    best_map50: float
    best_map50_95: float
    final_train_box_loss: float
    final_val_box_loss: float
    train_time_hours: float
    best_pt_size_mb: float
    map50_95_per_mb: float
    params_m: Optional[float]
    flops_g: Optional[float]
    latency_ms: Optional[float]
    latency_std_ms: Optional[float]
    benchmark_device: str
    map50_95_per_mparam: Optional[float]
    map50_95_per_gflop: Optional[float]
    extra: Dict[str, Any]


def parse_key_value_yaml(yaml_path: Path) -> Dict[str, str]:
    data: Dict[str, str] = {}
    if not yaml_path.exists():
        return data

    for raw in yaml_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip()
    return data


def read_results_rows(results_csv: Path) -> List[Dict[str, str]]:
    if not results_csv.exists():
        raise FileNotFoundError(f"Missing results file: {results_csv}")

    with results_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]

    if not rows:
        raise ValueError(f"No rows found in results file: {results_csv}")
    return rows


def as_float(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(row: Dict[str, str], key: str, default: int = 0) -> int:
    value = row.get(key, "")
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def as_optional_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def resolve_device(device: str) -> str:
    dev = device.strip().lower()
    if dev == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if dev.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable, fallback to cpu.")
        return "cpu"
    return dev


def benchmark_latency_ms(model: torch.nn.Module, imgsz: int, device: str, warmup: int, runs: int) -> Tuple[float, float, str]:
    resolved = resolve_device(device)
    torch_device = torch.device(resolved)

    model = model.to(torch_device)
    model.eval()

    x = torch.randn(1, 3, imgsz, imgsz, device=torch_device)

    with torch.inference_mode():
        for _ in range(max(warmup, 0)):
            _ = model(x)
            if torch_device.type == "cuda":
                torch.cuda.synchronize(torch_device)

        timings_ms: List[float] = []
        for _ in range(max(runs, 1)):
            t0 = time.perf_counter()
            _ = model(x)
            if torch_device.type == "cuda":
                torch.cuda.synchronize(torch_device)
            t1 = time.perf_counter()
            timings_ms.append((t1 - t0) * 1000.0)

    mean_ms = sum(timings_ms) / len(timings_ms)
    variance = sum((v - mean_ms) ** 2 for v in timings_ms) / len(timings_ms)
    std_ms = variance ** 0.5
    return mean_ms, std_ms, resolved


def compute_params_and_flops(model: torch.nn.Module, imgsz: int) -> Tuple[float, Optional[float]]:
    model = model.cpu()
    model.eval()

    params = float(sum(p.numel() for p in model.parameters()))

    gflops: Optional[float] = None
    if thop_profile is not None:
        x = torch.randn(1, 3, imgsz, imgsz)
        profile_result = thop_profile(model, inputs=(x,), verbose=False)
        flops = profile_result[0]
        gflops = float(flops) / 1e9
    return params / 1e6, gflops


def lightweight_metrics(best_pt: Path, imgsz: int, device: str, warmup: int, runs: int, skip_benchmark: bool) -> Dict[str, Any]:
    if not best_pt.exists():
        return {
            "params_m": None,
            "flops_g": None,
            "latency_ms": None,
            "latency_std_ms": None,
            "benchmark_device": "n/a",
        }

    yolo = YOLO(str(best_pt))
    raw_model = yolo.model
    if not isinstance(raw_model, torch.nn.Module):
        return {
            "params_m": None,
            "flops_g": None,
            "latency_ms": None,
            "latency_std_ms": None,
            "benchmark_device": "invalid-model",
        }
    model = cast(torch.nn.Module, raw_model)

    params_m, flops_g = compute_params_and_flops(model, imgsz)

    latency_ms: Optional[float] = None
    latency_std_ms: Optional[float] = None
    benchmark_device = "skipped"
    if not skip_benchmark:
        latency_ms, latency_std_ms, benchmark_device = benchmark_latency_ms(model, imgsz, device, warmup, runs)

    return {
        "params_m": params_m,
        "flops_g": flops_g,
        "latency_ms": latency_ms,
        "latency_std_ms": latency_std_ms,
        "benchmark_device": benchmark_device,
    }


def summarize_run(
    name: str,
    run_dir: Path,
    extra_metrics: Dict[str, Dict[str, Any]],
    benchmark_imgsz: int,
    benchmark_device: str,
    benchmark_warmup: int,
    benchmark_runs: int,
    skip_benchmark: bool,
) -> Tuple[RunSummary, List[Dict[str, str]]]:
    rows = read_results_rows(run_dir / "results.csv")
    args_yaml = parse_key_value_yaml(run_dir / "args.yaml")

    best_row = max(rows, key=lambda r: as_float(r, "metrics/mAP50-95(B)", -1.0))
    final_row = rows[-1]

    best_pt = run_dir / "weights" / "best.pt"
    size_mb = best_pt.stat().st_size / (1024 * 1024) if best_pt.exists() else 0.0
    light = lightweight_metrics(best_pt, benchmark_imgsz, benchmark_device, benchmark_warmup, benchmark_runs, skip_benchmark)

    best_map = as_float(best_row, "metrics/mAP50-95(B)")
    per_mb = (best_map / size_mb) if size_mb > 0 else 0.0
    per_param = (best_map / light["params_m"]) if light["params_m"] and light["params_m"] > 0 else None
    per_gflop = (best_map / light["flops_g"]) if light["flops_g"] and light["flops_g"] > 0 else None

    summary = RunSummary(
        name=name,
        run_dir=run_dir,
        model=args_yaml.get("model", "unknown"),
        epochs=len(rows),
        best_epoch=as_int(best_row, "epoch"),
        best_precision=as_float(best_row, "metrics/precision(B)"),
        best_recall=as_float(best_row, "metrics/recall(B)"),
        best_map50=as_float(best_row, "metrics/mAP50(B)"),
        best_map50_95=best_map,
        final_train_box_loss=as_float(final_row, "train/box_loss"),
        final_val_box_loss=as_float(final_row, "val/box_loss"),
        train_time_hours=as_float(final_row, "time") / 3600.0,
        best_pt_size_mb=size_mb,
        map50_95_per_mb=per_mb,
        params_m=light["params_m"],
        flops_g=light["flops_g"],
        latency_ms=light["latency_ms"],
        latency_std_ms=light["latency_std_ms"],
        benchmark_device=light["benchmark_device"],
        map50_95_per_mparam=per_param,
        map50_95_per_gflop=per_gflop,
        extra=extra_metrics.get(name, {}),
    )
    return summary, rows


def load_extra_metrics(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Extra metrics file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Extra metrics JSON must be a dict: {run_name: {...}}")
    return {str(k): (v if isinstance(v, dict) else {"value": v}) for k, v in payload.items()}


def load_external_summaries(path: Optional[Path]) -> List[RunSummary]:
    if path is None or not path.exists():
        return []

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("External rows JSON must be a list of summary rows.")

    rows: List[RunSummary] = []
    for item in payload:
        if not isinstance(item, dict):
            continue

        name = str(item.get("run", "external"))
        best_map50_95 = as_optional_float(item.get("best_map50_95")) or 0.0
        best_pt_size_mb = as_optional_float(item.get("best_pt_size_mb"))
        if best_pt_size_mb is None:
            best_pt_size_mb = as_optional_float(item.get("model_size_mb")) or 0.0
        map_per_mb = as_optional_float(item.get("map50_95_per_mb"))
        if map_per_mb is None and best_pt_size_mb > 0:
            map_per_mb = best_map50_95 / best_pt_size_mb

        rows.append(
            RunSummary(
                name=name,
                run_dir=Path(str(item.get("run_dir", "external"))),
                model=str(item.get("model", "onnx")),
                epochs=int(as_optional_float(item.get("epochs")) or 0),
                best_epoch=int(as_optional_float(item.get("best_epoch")) or 0),
                best_precision=as_optional_float(item.get("best_precision")) or 0.0,
                best_recall=as_optional_float(item.get("best_recall")) or 0.0,
                best_map50=as_optional_float(item.get("best_map50")) or 0.0,
                best_map50_95=best_map50_95,
                final_train_box_loss=as_optional_float(item.get("final_train_box_loss")) or 0.0,
                final_val_box_loss=as_optional_float(item.get("final_val_box_loss")) or 0.0,
                train_time_hours=as_optional_float(item.get("train_time_hours")) or 0.0,
                best_pt_size_mb=best_pt_size_mb,
                map50_95_per_mb=map_per_mb or 0.0,
                params_m=as_optional_float(item.get("params_m")),
                flops_g=as_optional_float(item.get("flops_g")),
                latency_ms=as_optional_float(item.get("latency_ms")),
                latency_std_ms=as_optional_float(item.get("latency_std_ms")),
                benchmark_device=str(item.get("benchmark_device", "cpu")),
                map50_95_per_mparam=as_optional_float(item.get("map50_95_per_mparam")),
                map50_95_per_gflop=as_optional_float(item.get("map50_95_per_gflop")),
                extra={k: v for k, v in item.items() if k not in {
                    "run", "run_dir", "model", "epochs", "best_epoch", "best_precision", "best_recall",
                    "best_map50", "best_map50_95", "final_train_box_loss", "final_val_box_loss", "train_time_hours",
                    "best_pt_size_mb", "model_size_mb", "map50_95_per_mb", "params_m", "flops_g", "latency_ms",
                    "latency_std_ms", "benchmark_device", "map50_95_per_mparam", "map50_95_per_gflop"
                }},
            )
        )
    return rows


def write_summary_files(output_dir: Path, summaries: List[RunSummary]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []
    for s in summaries:
        row: Dict[str, Any] = {
            "run": s.name,
            "run_dir": str(s.run_dir),
            "model": s.model,
            "epochs": s.epochs,
            "best_epoch": s.best_epoch,
            "best_precision": round(s.best_precision, 6),
            "best_recall": round(s.best_recall, 6),
            "best_map50": round(s.best_map50, 6),
            "best_map50_95": round(s.best_map50_95, 6),
            "final_train_box_loss": round(s.final_train_box_loss, 6),
            "final_val_box_loss": round(s.final_val_box_loss, 6),
            "train_time_hours": round(s.train_time_hours, 4),
            "best_pt_size_mb": round(s.best_pt_size_mb, 4),
            "map50_95_per_mb": round(s.map50_95_per_mb, 6),
            "params_m": round(s.params_m, 4) if s.params_m is not None else None,
            "flops_g": round(s.flops_g, 4) if s.flops_g is not None else None,
            "latency_ms": round(s.latency_ms, 4) if s.latency_ms is not None else None,
            "latency_std_ms": round(s.latency_std_ms, 4) if s.latency_std_ms is not None else None,
            "benchmark_device": s.benchmark_device,
            "map50_95_per_mparam": round(s.map50_95_per_mparam, 6) if s.map50_95_per_mparam is not None else None,
            "map50_95_per_gflop": round(s.map50_95_per_gflop, 6) if s.map50_95_per_gflop is not None else None,
        }
        row.update(s.extra)
        summary_rows.append(row)

    json_path = output_dir / "summary.json"
    json_path.write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = output_dir / "summary.csv"
    fieldnames: List[str] = sorted({k for row in summary_rows for k in row.keys()})
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    report_path = output_dir / "report.md"
    report_lines = [
        "# Model Comparison Report",
        "",
        "## Focus Metrics",
        "- Accuracy: best_map50_95, best_map50, best_precision, best_recall",
        "- Lightweight: latency_ms, params_m, flops_g, best_pt_size_mb",
        "- Efficiency: map50_95_per_mb, map50_95_per_mparam, map50_95_per_gflop",
        "- Training cost: train_time_hours",
        "",
        "## Table",
        "",
        "| run | mAP50-95 | latency(ms) | params(M) | FLOPs(G) | size(MB) | mAP50-95/MB | train_time(h) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        latency = f"{s.latency_ms:.2f}" if s.latency_ms is not None else "n/a"
        params_m = f"{s.params_m:.2f}" if s.params_m is not None else "n/a"
        flops_g = f"{s.flops_g:.2f}" if s.flops_g is not None else "n/a"
        report_lines.append(
            f"| {s.name} | {s.best_map50_95:.4f} | {latency} | {params_m} | {flops_g} | {s.best_pt_size_mb:.2f} | {s.map50_95_per_mb:.4f} | {s.train_time_hours:.2f} |"
        )

    report_lines.extend(
        [
            "",
            "## Extensibility",
            "- Edit DEFAULT_RUNS and DEFAULT_CONFIG in code to control compared runs.",
            "- Use DEFAULT_CONFIG.extra_metrics for custom per-run metrics injection.",
            "- Use DEFAULT_CONFIG.external_rows to merge non-training artifacts (e.g., FP32/INT8 ONNX).",
        ]
    )
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def plot_comparison(output_dir: Path, summaries: List[RunSummary], run_curves: Dict[str, List[Dict[str, str]]]) -> None:
    if plt is None:
        print("[WARN] matplotlib is not available, skip plot generation.")
        return

    names = [s.name for s in summaries]

    fig1, axes = plt.subplots(2, 3, figsize=(16, 9.2))
    axes = axes.flatten()
    axes[0].bar(names, [s.best_map50_95 for s in summaries], color="#4C72B0")
    axes[0].set_title("Best mAP50-95")
    axes[0].set_ylim(0, 1)

    axes[1].bar(names, [s.best_pt_size_mb for s in summaries], color="#55A868")
    axes[1].set_title("best.pt Size (MB)")

    axes[2].bar(names, [s.map50_95_per_mb for s in summaries], color="#C44E52")
    axes[2].set_title("mAP50-95 per MB")

    axes[3].bar(names, [s.params_m if s.params_m is not None else 0.0 for s in summaries], color="#8172B2")
    axes[3].set_title("Parameters (Million)")

    axes[4].bar(names, [s.flops_g if s.flops_g is not None else 0.0 for s in summaries], color="#CCB974")
    axes[4].set_title("FLOPs (G)")

    axes[5].bar(names, [s.latency_ms if s.latency_ms is not None else 0.0 for s in summaries], color="#64B5CD")
    axes[5].set_title("Inference Latency (ms)")

    for ax in axes:
        ax.grid(axis="y", alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(output_dir / "comparison_bars.png", dpi=180)
    plt.close(fig1)

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 4.8))
    for name, rows in run_curves.items():
        epochs = [as_int(r, "epoch") for r in rows]
        map5095 = [as_float(r, "metrics/mAP50-95(B)") for r in rows]
        val_loss = [as_float(r, "val/box_loss") for r in rows]
        axes2[0].plot(epochs, map5095, label=name)
        axes2[1].plot(epochs, val_loss, label=name)

    axes2[0].set_title("mAP50-95 vs Epoch")
    axes2[0].set_xlabel("Epoch")
    axes2[0].set_ylabel("mAP50-95")
    axes2[0].grid(alpha=0.3)

    axes2[1].set_title("Val Box Loss vs Epoch")
    axes2[1].set_xlabel("Epoch")
    axes2[1].set_ylabel("val/box_loss")
    axes2[1].grid(alpha=0.3)

    for ax in axes2:
        ax.legend()

    fig2.tight_layout()
    fig2.savefig(output_dir / "training_curves.png", dpi=180)
    plt.close(fig2)


def main() -> None:
    cfg = DEFAULT_CONFIG

    extra_metrics = load_extra_metrics(cfg.extra_metrics.resolve() if cfg.extra_metrics else None)

    summaries: List[RunSummary] = []
    run_curves: Dict[str, List[Dict[str, str]]] = {}

    for name, run_dir in cfg.runs:
        summary, rows = summarize_run(
            name,
            run_dir,
            extra_metrics,
            benchmark_imgsz=cfg.benchmark_imgsz,
            benchmark_device=cfg.benchmark_device,
            benchmark_warmup=cfg.benchmark_warmup,
            benchmark_runs=cfg.benchmark_runs,
            skip_benchmark=cfg.skip_benchmark,
        )
        summaries.append(summary)
        run_curves[name] = rows

    summaries.extend(load_external_summaries(cfg.external_rows.resolve() if cfg.external_rows else None))

    write_summary_files(cfg.output, summaries)
    plot_comparison(cfg.output, summaries, run_curves)

    print("Saved comparison artifacts:")
    print(f"- {cfg.output / 'summary.csv'}")
    print(f"- {cfg.output / 'summary.json'}")
    print(f"- {cfg.output / 'report.md'}")
    if plt is not None:
        print(f"- {cfg.output / 'comparison_bars.png'}")
        print(f"- {cfg.output / 'training_curves.png'}")


if __name__ == "__main__":
    main()
