#!/usr/bin/env python3
"""
Plot length scaling results and fit cubic trends.
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _load_results(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lengths: List[int] = []
    elapsed_s: List[float] = []
    mem_alloc: List[float] = []
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            lengths.append(int(row["seq_len"]))
            elapsed_s.append(float(row["elapsed_s"]))
            mem_alloc.append(float(row["peak_mem_allocated_bytes"]) / (1024.0 ** 3))
    order = np.argsort(np.array(lengths))
    lengths_np = np.array(lengths, dtype=np.float64)[order]
    elapsed_np = np.array(elapsed_s, dtype=np.float64)[order]
    mem_np = np.array(mem_alloc, dtype=np.float64)[order]
    return lengths_np, elapsed_np, mem_np


def _fit_cubic(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.polyfit(x, y, deg=3)


def _plot_metric(
    datasets: List[Dict],
    metric_key: str,
    ylabel: str,
    output_path: Path,
    title_suffix: str = None,
    hline_y: float = None,
    extrapolate_to: float = None,
):
    plt.figure(figsize=(7, 5))
    for data in datasets:
        x = data["lengths"]
        y = data[metric_key]
        coeffs = _fit_cubic(x, y)
        x_fit = np.linspace(x.min(), x.max(), 200)
        y_fit = np.polyval(coeffs, x_fit)
        plt.scatter(x, y, label=data["label"])
        line = plt.plot(x_fit, y_fit, label="_nolegend_")[0]

        if extrapolate_to is not None and extrapolate_to > x.max():
            x_ext = np.linspace(x.max(), extrapolate_to, 80)
            y_ext = np.polyval(coeffs, x_ext)
            plt.plot(x_ext, y_ext, linestyle=":", color=line.get_color(), label="_nolegend_")
            y_ext_val = float(np.polyval(coeffs, extrapolate_to))
            plt.scatter(
                [extrapolate_to],
                [y_ext_val],
                facecolors="none",
                edgecolors=line.get_color(),
                linewidths=1.5,
                label="_nolegend_",
            )
    # if hline_y is not None:
    #     plt.axhline(hline_y, linestyle="--", color="black", linewidth=1.0, label="H200 (141GB)")
    plt.xlabel("Sequence length")
    plt.ylabel(ylabel)
    title = f"{ylabel} vs. length (cubic fit)"
    if title_suffix:
        title = f"{title} ({title_suffix})"
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot length scaling results with cubic fits")
    parser.add_argument(
        "--csv_paths",
        type=Path,
        nargs="+",
        required=True,
        help="CSV result files from test_length_scaling.py",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        required=True,
        help="Labels corresponding to csv_paths",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to save plots",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="length_scaling_compare",
        help="Prefix for output plot filenames",
    )
    parser.add_argument(
        "--title_suffix",
        type=str,
        default=None,
        help="Optional suffix appended to plot titles.",
    )
    parser.add_argument(
        "--extrapolate_to",
        type=float,
        default=None,
        help="Optional length to extrapolate fit with dotted line.",
    )
    args = parser.parse_args()

    if len(args.csv_paths) != len(args.labels):
        raise ValueError("csv_paths and labels must have the same length")

    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets: List[Dict] = []
    coeffs_rows: List[Dict[str, float]] = []
    for path, label in zip(args.csv_paths, args.labels):
        lengths, elapsed_s, mem_alloc = _load_results(path)
        datasets.append(
            {
                "label": label,
                "lengths": lengths,
                "elapsed_s": elapsed_s,
                "mem_alloc": mem_alloc,
            }
        )
        coeffs_time = _fit_cubic(lengths, elapsed_s)
        coeffs_mem = _fit_cubic(lengths, mem_alloc)
        coeffs_rows.append(
            {
                "label": label,
                "metric": "elapsed_s",
                "c3": float(coeffs_time[0]),
                "c2": float(coeffs_time[1]),
                "c1": float(coeffs_time[2]),
                "c0": float(coeffs_time[3]),
            }
        )
        coeffs_rows.append(
            {
                "label": label,
                "metric": "peak_mem_allocated_gb",
                "c3": float(coeffs_mem[0]),
                "c2": float(coeffs_mem[1]),
                "c1": float(coeffs_mem[2]),
                "c0": float(coeffs_mem[3]),
            }
        )

    runtime_path = output_dir / f"{args.output_prefix}_runtime.png"
    memory_path = output_dir / f"{args.output_prefix}_memory.png"
    coeffs_path = output_dir / f"{args.output_prefix}_coeffs.csv"

    _plot_metric(
        datasets=datasets,
        metric_key="elapsed_s",
        ylabel="Runtime (s)",
        output_path=runtime_path,
        title_suffix=args.title_suffix,
        extrapolate_to=args.extrapolate_to,
    )
    _plot_metric(
        datasets=datasets,
        metric_key="mem_alloc",
        ylabel="Peak allocated memory (GB)",
        output_path=memory_path,
        title_suffix=args.title_suffix,
        hline_y=141.0,
        extrapolate_to=args.extrapolate_to,
    )

    with open(coeffs_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["label", "metric", "c3", "c2", "c1", "c0"],
        )
        writer.writeheader()
        for row in coeffs_rows:
            writer.writerow(row)

    print(f"Saved plots to {runtime_path} and {memory_path}", flush=True)
    print(f"Saved fit coefficients to {coeffs_path}", flush=True)


if __name__ == "__main__":
    main()
