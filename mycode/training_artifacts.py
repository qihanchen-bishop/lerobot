from __future__ import annotations

import json
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import IO, Any


class _TeeStream:
    def __init__(self, *streams: IO[str]):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


@contextmanager
def tee_output(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", buffering=1) as log_file:
        stdout = _TeeStream(sys.stdout, log_file)
        stderr = _TeeStream(sys.stderr, log_file)
        with redirect_stdout(stdout), redirect_stderr(stderr):
            yield


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def plot_training_curves(metrics_jsonl: Path, output_png: Path, title: str) -> None:
    records = read_jsonl(metrics_jsonl)
    if not records:
        return

    loss_keys = [
        key
        for key in ("loss", "action_loss", "seg_loss", "semantic_loss", "metric_loss")
        if any(isinstance(record.get(key), (int, float)) for record in records)
    ]
    if not loss_keys:
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        _plot_training_curves_with_pil(records, loss_keys, output_png, title)
        return

    steps = [record.get("step", record.get("steps")) for record in records]
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    for key in loss_keys:
        xs = []
        ys = []
        for step, record in zip(steps, records, strict=False):
            value = record.get(key)
            if isinstance(step, (int, float)) and isinstance(value, (int, float)):
                xs.append(step)
                ys.append(value)
        if xs:
            plt.plot(xs, ys, label=key)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png, dpi=160)
    plt.close()


def _plot_training_curves_with_pil(
    records: list[dict[str, Any]],
    loss_keys: list[str],
    output_png: Path,
    title: str,
) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return

    series = {}
    for key in loss_keys:
        points = []
        for record in records:
            step = record.get("step", record.get("steps"))
            value = record.get(key)
            if isinstance(step, (int, float)) and isinstance(value, (int, float)):
                points.append((float(step), float(value)))
        if points:
            series[key] = points
    if not series:
        return

    width, height = 1200, 720
    left, right, top, bottom = 90, 40, 70, 90
    plot_w = width - left - right
    plot_h = height - top - bottom
    all_x = [x for points in series.values() for x, _ in points]
    all_y = [y for points in series.values() for _, y in points]
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    if x_min == x_max:
        x_max = x_min + 1
    if y_min == y_max:
        y_max = y_min + 1
    y_pad = (y_max - y_min) * 0.05
    y_min -= y_pad
    y_max += y_pad

    def xy(point: tuple[float, float]) -> tuple[int, int]:
        x, y = point
        px = left + int((x - x_min) / (x_max - x_min) * plot_w)
        py = top + plot_h - int((y - y_min) / (y_max - y_min) * plot_h)
        return px, py

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((left, 20), title, fill=(20, 20, 20))
    draw.line((left, top, left, top + plot_h), fill=(40, 40, 40), width=2)
    draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill=(40, 40, 40), width=2)
    for i in range(6):
        y = top + int(i / 5 * plot_h)
        value = y_max - i / 5 * (y_max - y_min)
        draw.line((left, y, left + plot_w, y), fill=(225, 225, 225), width=1)
        draw.text((10, y - 8), f"{value:.3g}", fill=(70, 70, 70))
    draw.text((left + plot_w // 2 - 20, height - 35), "step", fill=(40, 40, 40))
    draw.text((10, 45), "loss", fill=(40, 40, 40))
    draw.text((left, top + plot_h + 12), f"{x_min:.0f}", fill=(70, 70, 70))
    draw.text((left + plot_w - 50, top + plot_h + 12), f"{x_max:.0f}", fill=(70, 70, 70))

    colors = [
        (31, 119, 180),
        (214, 39, 40),
        (44, 160, 44),
        (148, 103, 189),
        (255, 127, 14),
    ]
    for idx, (key, points) in enumerate(series.items()):
        color = colors[idx % len(colors)]
        pixels = [xy(point) for point in points]
        if len(pixels) == 1:
            x, y = pixels[0]
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
        else:
            draw.line(pixels, fill=color, width=3)
        legend_x = left + 20 + idx * 180
        legend_y = height - 65
        draw.line((legend_x, legend_y, legend_x + 30, legend_y), fill=color, width=4)
        draw.text((legend_x + 38, legend_y - 8), key, fill=(30, 30, 30))

    output_png.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_png)
