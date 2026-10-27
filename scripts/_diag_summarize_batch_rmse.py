"""Resume, leyendo summary.csv de la corrida más reciente de cada video/modelo/condicion, MAE y RMSE. No es parte del pipeline principal."""

import csv
from pathlib import Path

BASE = Path("data/gold_standard/TOMA FRONTAL")
VIDEOS = ["2EJ1", "2EJ3", "6EJ3", "3EJ9", "3EJ10", "5EJ6", "5EJ9", "5EJ10"]
MODELS = ["yolo26n", "mediapipe_heavy", "mediapipe_lite"]
CONDITIONS = ["no_occlusion", "occlusion", "illumination"]


def _latest(video: str, model: str, condition: str) -> Path | None:
    d = BASE / video / model / condition
    if not d.is_dir():
        return None
    ts = sorted(p for p in d.iterdir() if p.is_dir())
    return ts[-1] if ts else None


def _read(run_dir: Path, model: str) -> tuple[float, float]:
    with (run_dir / "metrics" / "summary.csv").open(newline="", encoding="utf-8") as f:
        row = next(r for r in csv.DictReader(f) if r["variante"] == model)
    return float(row["error_medio_deg"]), float(row["rmse_deg"])


def main() -> None:
    print(f"{'video':7s} {'modelo':16s} {'condicion':13s} {'MAE':>7s} {'RMSE':>7s} {'brecha':>7s}")
    print("-" * 65)
    for video in VIDEOS:
        for model in MODELS:
            for condition in CONDITIONS:
                run_dir = _latest(video, model, condition)
                if run_dir is None:
                    continue
                mae, rmse = _read(run_dir, model)
                print(f"{video:7s} {model:16s} {condition:13s} {mae:7.2f} {rmse:7.2f} {rmse - mae:7.2f}")


if __name__ == "__main__":
    main()
