"""Plot cumulative Fourier evaluation losses for one saved run, entirely offline.

    env/bin/python scripts/plot_fourier_loss.py --input unzipped_results
    env/bin/python scripts/plot_fourier_loss.py --input results/123613.0.tar.gz
    env/bin/python scripts/plot_fourier_loss.py --input unzipped_results --list-runs

Accepts a metrics.jsonl, an extracted results folder, or a .tar.gz archive.
Defaults: raw losses for degrees 1..7, steps 0..15000, one graph. The input
results are read without modification. Plotting needs requirements-plot.txt.
Output filenames include the run name; repeated plots receive numbered suffixes.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from dataclasses import dataclass
import io
import json
import math
import os
from pathlib import Path
import re
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[1]
COLORS = ("#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#4D4D4D")


@dataclass
class RunSource:
    path: Path
    member: str | None
    config: dict

    @property
    def label(self):
        return self.config.get("run_name") or (self.member or self.path.parent.name)

    @property
    def description(self):
        return f"{self.path}::{self.member}" if self.member else str(self.path)


@contextmanager
def open_history(source):
    if source.member is None:
        with source.path.open() as stream:
            yield stream
    else:
        with tarfile.open(source.path, "r:gz") as archive:
            with io.TextIOWrapper(archive.extractfile(source.member), encoding="utf-8") as stream:
                yield stream


def is_fourier_history(source):
    with open_history(source) as stream:
        for line in stream:
            if line.strip():
                return "eval/loss_degree_1" in json.loads(line)
    return False


def discover_runs(path):
    path = path.resolve()
    if path.is_dir():
        sources = []
        for metrics in sorted(path.rglob("metrics.jsonl")):
            sources.extend(discover_runs(metrics))
        for archive in sorted(path.rglob("*.tar.gz")):
            sources.extend(discover_runs(archive))
        return sources
    if not path.is_file():
        raise ValueError(f"Input does not exist: {path}")
    if path.name.endswith(".tar.gz"):
        sources = []
        with tarfile.open(path, "r:gz") as archive:
            members = {member.name: member for member in archive.getmembers() if member.isfile()}
            for name in sorted(members):
                if not name.endswith("/metrics.jsonl"):
                    continue
                config_name = name.rsplit("/", 1)[0] + "/config.json"
                config = json.load(archive.extractfile(config_name)) if config_name in members else {}
                sources.append(RunSource(path, name, config))
        return [source for source in sources if is_fourier_history(source)]
    if path.name != "metrics.jsonl":
        raise ValueError("Input must be a results folder, metrics.jsonl, or .tar.gz archive.")
    config_path = path.with_name("config.json")
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    source = RunSource(path, None, config)
    return [source] if is_fourier_history(source) else []


def load_series(source, max_degree=7, max_step=15_000, axis="degree", min_cutoff=1):
    if axis not in ("degree", "input") or not 1 <= min_cutoff <= max_degree:
        raise ValueError("Choose axis degree/input and a minimum cutoff no greater than the maximum.")
    cutoffs = list(range(min_cutoff, max_degree + 1))
    keys = [f"eval/loss_{axis}_{cutoff}" for cutoff in cutoffs]
    # Merge repeated step records by metric, retaining the latest value.
    by_step = {}
    with open_history(source) as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            step = row.get("step", row.get("_step"))
            if not isinstance(step, (int, float)) or isinstance(step, bool) or not math.isfinite(step) or int(step) != step:
                raise ValueError(f"Invalid optimizer step at history line {line_number}.")
            if 0 <= step <= max_step:
                by_step.setdefault(int(step), {}).update({key: row[key] for key in keys if key in row})
    steps = sorted(by_step)
    if not steps:
        raise ValueError(f"No measurements between steps 0 and {max_step}.")
    series = {}
    for degree, key in zip(cutoffs, keys):
        values = [by_step[step].get(key) for step in steps]
        if any(not isinstance(value, (int, float)) or isinstance(value, bool)
               or not math.isfinite(value) or value < 0 for value in values):
            raise ValueError(f"{key} is missing or invalid; choose a smaller cutoff or a run recording this evaluation axis.")
        series[degree] = values
    return steps, series


def plot_output_paths(source, output_dir, max_degree, max_step, axis="degree", min_cutoff=1):
    """Choose one shared filename stem without replacing any existing export."""
    run_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(source.label)).strip("-_.")[:140] or "fourier-run"
    kind = "degrees" if axis == "degree" else "inputs"
    stem = f"{run_name}__eval_loss_{kind}_{min_cutoff}-{max_degree}_steps_0-{max_step}"
    extensions = (".png", ".svg", ".csv", ".json")
    copy_number = 1
    while True:
        suffix = "" if copy_number == 1 else f"__{copy_number}"
        paths = tuple(output_dir / (stem + suffix + extension) for extension in extensions)
        if not any(path.exists() for path in paths):
            return paths
        copy_number += 1


def create_plot(source, steps, series, output_dir, max_step, axis="degree"):
    # Keep plotting usable in sandboxed sessions with an unwritable home cache.
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "sharpness-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    output_dir.mkdir(parents=True, exist_ok=True)
    max_degree = max(series)
    png, svg, csv_path, metadata_path = plot_output_paths(source, output_dir, max_degree, max_step,
                                                        axis=axis, min_cutoff=min(series))
    function = source.config.get("function", {})
    function_name = function.get("name", "Fourier target") if isinstance(function, dict) else str(function)
    context = [function_name]
    for key, label in (("model_max_len", "input bits"), ("model_d_model", "width"),
                       ("model_num_layers", "layers"), ("model_num_attn_heads", "heads"),
                       ("train_batch_size", "batch")):
        if key in source.config:
            context.append(f"{label}: {source.config[key]}")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(12, 6.6), layout="constrained")
    fig.set_facecolor("white")
    for index, (degree, values) in enumerate(series.items()):
        label = f"Degree ≤ {degree}" if axis == "degree" else f"First {degree} inputs"
        ax.plot(steps, values, color=COLORS[index % len(COLORS)], linewidth=1.15,
                alpha=0.9, label=label, rasterized=False)
    title = "Cumulative Fourier evaluation loss" if axis == "degree" else "Input-tower evaluation loss"
    ax.set_title(title, loc="left", fontsize=18, fontweight="bold", pad=38)
    ax.text(0, 1.025, "  ·  ".join(context), transform=ax.transAxes, fontsize=10.5, color="#52606D")
    ax.set_xlabel("Optimizer step", labelpad=10)
    ax.set_ylabel("Held-out mean squared error", labelpad=10)
    ax.set_xlim(0, max_step)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.grid(True, color="#E7EBEF", linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color("#B8C2CC")
    ax.legend(title="Cumulative target" if axis == "degree" else "Conditional input target",
              loc="upper right", framealpha=0.96, edgecolor="#E7EBEF")
    fig.savefig(png, dpi=180)
    fig.savefig(svg)
    plt.close(fig)
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["step", *[f"eval/loss_{axis}_{degree}" for degree in series]])
        writer.writerows([step, *[series[degree][index] for degree in series]] for index, step in enumerate(steps))
    metadata_path.write_text(json.dumps({
        "source": source.description, "run_name": source.label, "function": function_name,
        "metric_definition": ("MSE(model(x), f_through_degree_k(x))" if axis == "degree" else
                              "MSE(model(x), E[f(X) | first k positions of input_order])"),
        "axis": axis, "degrees" if axis == "degree" else "input_counts": list(series), "requested_max_step": max_step,
        "first_step": steps[0], "last_step": steps[-1], "num_measurements": len(steps),
        "smoothing": "none", "config": source.config,
    }, indent=2) + "\n")
    return png, svg, csv_path, metadata_path


def positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return parsed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, help="Folder, metrics.jsonl, or archive; default unzipped_results with archive fallback in results/.")
    parser.add_argument("--run", help="Select one run by a unique name or path substring.")
    parser.add_argument("--list-runs", action="store_true")
    parser.add_argument("--axis", choices=("degree", "input"), default="degree")
    parser.add_argument("--min-cutoff", type=positive_int, default=1)
    parser.add_argument("--max-degree", type=positive_int, default=7)
    parser.add_argument("--max-inputs", type=positive_int, help="Maximum input cutoff for --axis input; defaults to --max-degree.")
    parser.add_argument("--max-step", type=positive_int, default=15_000)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/fourier_loss")
    args = parser.parse_args(argv)
    try:
        if args.input is not None:
            sources = discover_runs(args.input)
        else:
            extracted = ROOT / "unzipped_results"
            sources = discover_runs(extracted) if extracted.exists() else []
            if not sources:
                sources = [source for archive in sorted((ROOT / "results").glob("*.tar.gz")) for source in discover_runs(archive)]
        if not sources:
            raise ValueError("No Fourier metrics.jsonl found. Pass --input with a Fourier results folder or archive.")
        if args.list_runs:
            print(json.dumps([{"run_name": source.label, "source": source.description} for source in sources], indent=2))
            return
        if args.run:
            sources = [source for source in sources if args.run in source.label or args.run in source.description]
            if len(sources) != 1:
                raise ValueError(f"--run matched {len(sources)} runs; use --list-runs and choose a unique substring.")
        source = sources[0]
        max_cutoff = args.max_inputs if args.axis == "input" and args.max_inputs is not None else args.max_degree
        steps, series = load_series(source, max_cutoff, args.max_step, axis=args.axis, min_cutoff=args.min_cutoff)
        outputs = create_plot(source, steps, series, args.output_dir, args.max_step, axis=args.axis)
    except (ValueError, OSError, tarfile.TarError) as error:
        parser.error(str(error))
    print(f"Run: {source.label}\nSource: {source.description}")
    print(f"Plotted {len(steps):,} raw measurements for each {args.axis} cutoff, steps {steps[0]:,}–{steps[-1]:,}.")
    for output in outputs:
        print(output.resolve())
    return outputs


if __name__ == "__main__":
    main()
