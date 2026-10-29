import json
from pathlib import Path


METRICS_DIR = Path("outputs/metrics")


def save_metrics(name, metrics):
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    path = METRICS_DIR / f"{name}.json"
    path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    return path

