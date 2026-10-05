#!/usr/bin/env python3
"""Turn saved W&B loss-space summaries into tables, HTML, Markdown and CSV.

Usage: env/bin/python scripts/summarize_spaces.py unzipped_results
Reads local summaries only; never starts training or contacts W&B.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


# category, explanation. These descriptions also accompany the raw CSV.
METRICS = {
    "parameter_count": ("Settings", "Number of trainable parameter coordinates."),
    "search_dim": ("Settings", "Full parameter count; duplicates parameter_count."),
    "direction_cap": ("Settings", "Maximum accepted directions per task and kind."),
    "hvp_budget": ("Settings", "Hessian-vector product budget per task, shared by both kinds."),
    "krylov_max_dim": ("Settings", "Maximum active basis dimension; storage bound, not the total HVP budget."),
    "tangent_dim": ("Task directions", "Independent accepted safe basis directions; not a certified full dimension."),
    "sharpness_dim": ("Task directions", "Independent accepted harmful basis directions; not a certified full dimension."),
    "interference_dim": ("Interference", "Verified overlap: harmful to the source task and safe for the target task."),
    "min_harm": ("Loss changes", "Actual worst-sign loss increase of the first tangent candidate; not a global minimum."),
    "max_harm": ("Loss changes", "Actual worst-sign loss increase of the first sharpness candidate; not a global maximum."),
    "tangent_stop_harm": ("Loss changes", "Actual worst-sign loss increase of the rejected tangent candidate."),
    "sharpness_stop_harm": ("Loss changes", "Actual worst-sign loss increase of the rejected sharpness candidate."),
    "tangent_cap_reached": ("Search flags", "1 means the tangent direction cap was reached."),
    "sharpness_cap_reached": ("Search flags", "1 means the sharpness direction cap was reached."),
    "tangent_candidate_exhausted": ("Search flags", "1 means available tangent Krylov candidates ran out before the cap."),
    "sharpness_candidate_exhausted": ("Search flags", "1 means available sharpness Krylov candidates ran out before the cap."),
    "hvp_budget_exhausted": ("Search flags", "1 means all allowed HVPs were used; this is expected at the configured budget."),
    "full_space_covered": ("Search flags", "1 means a complete parameter basis was processed; 0 means the full dimension is not certified."),
    "hvp_count": ("Computation", "Actual HVP operator calls for this task."),
    "krylov_dim": ("Computation", "Final active Hessian approximation dimension; can be smaller than HVP count after restarts."),
    "lanczos_restarts": ("Computation", "Thick restart cycles retaining extreme eigenvector estimates and gradient seeds."),
    "ritz_max_relative_residual": ("Numerical quality", "Maximum relative eigen-equation error of checked Hessian Ritz modes."),
    "tangent_max_residual": ("Numerical quality", "Maximum absolute projected eigen-equation error of accepted tangent modes."),
    "sharpness_max_residual": ("Numerical quality", "Maximum absolute quadratic stationarity error of accepted sharpness modes."),
}
GLOBALS = {"parameter_count", "search_dim", "direction_cap", "hvp_budget", "krylov_max_dim"}


def number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def display(value: Any) -> str:
    if value is None:
        return "Not logged"
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            return str(value)
        if value == 0:
            return "0"
        if float(value).is_integer():
            return f"{int(value):,}"
        if abs(value) < 0.001:
            return f"{value:.3e}"
        return f"{value:.6g}"
    return json.dumps(value, ensure_ascii=False)


def ratio(value: Any, epsilon: Any) -> float | None:
    value, epsilon = number(value), number(epsilon)
    return value / epsilon if value is not None and epsilon is not None and epsilon > 0 else None


def read_config(files_directory: Path) -> tuple[dict[str, Any], list[str]]:
    json_path = files_directory / "config.json"
    yaml_path = files_directory / "config.yaml"
    if json_path.exists():
        raw = json.loads(json_path.read_text())
    elif yaml_path.exists():
        try:
            import yaml
        except ImportError:
            return {}, ["Configuration could not be read because PyYAML is unavailable. Use the project environment, or supply --epsilon and --delta."]
        raw = yaml.safe_load(yaml_path.read_text()) or {}
    else:
        return {}, ["No saved configuration was found. Missing settings are marked Not logged."]
    if not isinstance(raw, dict):
        raise ValueError("Saved configuration must be a mapping.")
    return {
        key: value.get("value") if isinstance(value, dict) and "value" in value else value
        for key, value in raw.items()
    }, []


def load_report(summary_path: Path, *, epsilon: float | None = None, delta: float | None = None) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text())
    if not isinstance(summary, dict):
        raise ValueError("Summary JSON must be an object.")
    metrics = {key: value for key, value in summary.items() if key.startswith("spaces/")}
    discovered = set()
    for key in metrics:
        parts = key.split("/", 2)
        if len(parts) == 3 and parts[1] in METRICS and parts[1] != "interference_dim":
            discovered.add(parts[2])
    if not discovered:
        raise ValueError("No per-task spaces measurements were found.")
    config, notes = read_config(summary_path.parent)
    for field, supplied in (("epsilon", epsilon), ("delta", delta)):
        recorded = config.get(f"train_space_{field}")
        if supplied is not None and number(recorded) is not None and supplied != recorded:
            raise ValueError(f"Supplied {field}={supplied} differs from the saved measurement setting {recorded}.")
    configured_tasks = config.get("selector_functions", [])
    if not isinstance(configured_tasks, list):
        configured_tasks = []
    tasks = list(dict.fromkeys([task for task in configured_tasks if isinstance(task, str)]))
    tasks += sorted(discovered - set(tasks))
    folder = summary_path.parent.parent.name if summary_path.parent.name == "files" else summary_path.parent.name
    run_id = folder.rsplit("-", 1)[-1] if folder.startswith(("run-", "offline-run-")) else folder
    run_id = re.sub(r"[^a-zA-Z0-9_.-]", "_", run_id) or "run"
    settings = {
        "run_id": run_id,
        "step": summary.get("_step"),
        "parameter_count": metrics.get("spaces/parameter_count", metrics.get("spaces/search_dim")),
        "direction_cap": metrics.get("spaces/direction_cap", config.get("train_space_num_directions")),
        "hvp_budget": metrics.get("spaces/hvp_budget", config.get("train_space_hvp_budget")),
        "krylov_max_dim": metrics.get("spaces/krylov_max_dim"),
        "delta": delta if delta is not None else config.get("train_space_delta"),
        "epsilon": epsilon if epsilon is not None else config.get("train_space_epsilon"),
        "probes_per_task": config.get("train_space_num_inputs"),
        "intersection_cosine": config.get("train_space_intersection_cosine"),
    }
    for field, lower in (("epsilon", 0), ("delta", 0)):
        value = settings[field]
        if value is not None and (number(value) is None or value < lower or (field == "delta" and value == 0)):
            raise ValueError(f"{field} must be finite and {'positive' if field == 'delta' else 'nonnegative'}.")
    if epsilon is not None or delta is not None:
        notes.append("Threshold/radius values supplied on the command line must match the original measurement settings.")

    def get(family: str, task: str) -> Any:
        return metrics.get(f"spaces/{family}/{task}")

    def stop_reason(kind: str, task: str) -> str:
        if get(f"{kind}_cap_reached", task) == 1:
            return "Direction cap reached"
        if number(get(f"{kind}_stop_harm", task)) is not None:
            return "Next candidate within epsilon" if kind == "sharpness" else "Next candidate above epsilon"
        if get(f"{kind}_candidate_exhausted", task) == 1:
            return "Approximation candidates exhausted"
        return "Not logged"

    task_rows = []
    quality_rows = []
    for index, task in enumerate(tasks, 1):
        task_rows.append({
            "id": f"T{index}", "task": task,
            "safe_directions": get("tangent_dim", task),
            "safe_cap_reached": get("tangent_cap_reached", task),
            "safe_stop_reason": stop_reason("tangent", task),
            "sharp_directions": get("sharpness_dim", task),
            "sharp_cap_reached": get("sharpness_cap_reached", task),
            "sharp_stop_reason": stop_reason("sharpness", task),
            "first_safe_loss_increase": get("min_harm", task),
            "first_sharp_loss_increase": get("max_harm", task),
            "first_sharp_over_epsilon": ratio(get("max_harm", task), settings["epsilon"]),
            "next_sharp_loss_increase": get("sharpness_stop_harm", task),
            "next_safe_loss_increase": get("tangent_stop_harm", task),
        })
        quality_rows.append({
            "task": task, "hvp_count": get("hvp_count", task),
            "krylov_dim": get("krylov_dim", task),
            "lanczos_restarts": get("lanczos_restarts", task),
            "full_space_covered": get("full_space_covered", task),
            "hvp_budget_used": get("hvp_budget_exhausted", task),
            "ritz_relative_error": get("ritz_max_relative_residual", task),
            "tangent_absolute_error": get("tangent_max_residual", task),
            "sharp_absolute_error": get("sharpness_max_residual", task),
        })
    pairs = []
    for safe in tasks:
        for harmed in tasks:
            if harmed == safe:
                continue
            key = f"spaces/interference_dim/{harmed}_to_{safe}"
            pairs.append({
                "safe_task": safe, "harmed_task": harmed,
                "directions": metrics.get(key), "metric": key,
            })
    catalog = []
    for key, value in sorted(metrics.items()):
        parts = key.split("/", 2)
        family = parts[1]
        category, description = METRICS.get(family, ("Other", "Unrecognized metric; retained without interpretation."))
        catalog.append({
            "group": category, "metric": key, "value": value,
            "scope": parts[2] if len(parts) == 3 else "All tasks",
            "meaning": description,
        })
    return {
        "source": str(summary_path.resolve()), "settings": settings, "tasks": task_rows,
        "quality": quality_rows, "pairs": pairs, "catalog": catalog, "notes": notes,
    }


def observations(report: dict[str, Any]) -> list[str]:
    tasks, pairs, settings = report["tasks"], report["pairs"], report["settings"]
    notes = []
    capped = sum(row["safe_cap_reached"] == 1 for row in tasks)
    notes.append(
        f"{capped}/{len(tasks)} tasks reached the tangent direction cap. "
        "Capped counts cannot rank the full tangent dimensions; each reported basis direction is individually safe."
    )
    sharp_counts = [number(row["sharp_directions"]) for row in tasks]
    sharp_counts = [count for count in sharp_counts if count is not None]
    if sharp_counts:
        notes.append(f"The search found {display(min(sharp_counts))}–{display(max(sharp_counts))} harmful directions per measured task at this radius and loss threshold.")
    candidates = [row for row in tasks if number(row["first_sharp_loss_increase"]) is not None]
    if candidates:
        largest = max(candidates, key=lambda row: row["first_sharp_loss_increase"])
        multiple = largest["first_sharp_over_epsilon"]
        suffix = f" ({multiple:.2f} times epsilon)" if multiple is not None else ""
        notes.append(f"Among the first sharpness candidates, {largest['task']} has the largest measured loss increase: {display(largest['first_sharp_loss_increase'])}{suffix}.")
    recorded = [row for row in pairs if number(row["directions"]) is not None]
    positive = [row for row in recorded if row["directions"] > 0]
    notes.append(
        f"{len(positive)}/{len(recorded)} recorded ordered pairs have a verified overlap "
        f"({len(pairs) - len(recorded)} pairs missing). "
        "Zero means no overlap was found between the selected bases; other safe-for-one/harmful-for-another directions may exist."
    )
    residuals = [number(row["ritz_relative_error"]) for row in report["quality"]]
    residuals = [value for value in residuals if value is not None]
    if residuals:
        notes.append(
            f"Checked relative eigen-equation errors range from {display(min(residuals))} to {display(max(residuals))}. "
            "These describe the checked approximate modes and do not certify global extrema."
        )
    if any(number(row["lanczos_restarts"]) is not None for row in report["quality"]):
        notes.append(
            "HVP counts measure total computation across restart cycles. The Krylov dimension is the final active basis, "
            f"bounded by {display(settings['krylov_max_dim'])} vectors; it is not the number of parameter coordinates explored. "
            "Restarts retain extreme eigenvector estimates and gradient seeds."
        )
    notes.append("Loss changes use the fixed space-measurement probes. The summaries do not describe held-out generalization, or the effects of arbitrary combinations of basis directions.")
    notes.append("The tables use scalar summaries. Full vectors and individual +/- loss changes are in the W&B loss-spaces artifact.")
    return notes + report["notes"]


def tables(report: dict[str, Any]) -> list[tuple[str, str, list[str], list[list[Any]]]]:
    tasks, quality, settings = report["tasks"], report["quality"], report["settings"]
    lookup = {(pair["safe_task"], pair["harmed_task"]): pair["directions"] for pair in report["pairs"]}
    matrix = [
        [f"{row['id']} · {row['task']}"] + [
            "—" if row["task"] == other["task"] else lookup[(row["task"], other["task"])]
            for other in tasks
        ] for row in tasks
    ]
    counts = [
        [f"{row['id']} · {row['task']}",
         f"{display(row['safe_directions'])} (cap)" if row["safe_cap_reached"] == 1 else row["safe_directions"],
         f"{display(row['sharp_directions'])} (cap)" if row["sharp_cap_reached"] == 1 else row["sharp_directions"],
         row["first_safe_loss_increase"], row["first_sharp_loss_increase"],
         f"{row['first_sharp_over_epsilon']:.2f}×" if row["first_sharp_over_epsilon"] is not None else None]
        for row in tasks
    ]
    setting_labels = {
        "run_id": "Run ID", "step": "Final training step",
        "parameter_count": "Trainable parameters", "direction_cap": "Maximum directions per kind",
        "hvp_budget": "HVP budget per task", "delta": "Perturbation norm δ",
        "krylov_max_dim": "Maximum active basis dimension",
        "epsilon": "Allowed loss increase ε", "probes_per_task": "Probe inputs per task",
        "intersection_cosine": "Minimum overlap cosine",
    }
    settings_rows = [[setting_labels[key], value] for key, value in settings.items()]

    def next_loss(row: dict[str, Any], kind: str) -> Any:
        value = row[f"next_{kind}_loss_increase"]
        if value is None and row[f"{kind}_cap_reached"] == 1:
            return "Not evaluated (cap)"
        return value
    groups = [[group, count] for group, count in sorted(Counter(row["group"] for row in report["catalog"]).items())]
    quality_columns = [
        ("task", "Task"), ("hvp_count", "HVPs used"), ("krylov_dim", "Final active dimension"),
        ("hvp_budget_used", "Budget used (1=yes)"),
    ]
    for field, label in (("lanczos_restarts", "Thick restarts"), ("full_space_covered", "Full parameter basis (1=yes)")):
        if any(row[field] is not None for row in quality):
            quality_columns.append((field, label))
    quality_columns += [
        ("ritz_relative_error", "Ritz relative error"), ("tangent_absolute_error", "Tangent absolute error"),
        ("sharp_absolute_error", "Sharpness absolute error"),
    ]
    return [
        ("Task directions", "Each count concerns independent basis directions found at the recorded delta and epsilon. The loss columns show actual worst-sign changes for the first candidate, not certified global extrema. A negative worst-sign change means both signs improved the measured loss.",
         ["Task", "Safe directions", "Harmful directions", "First safe Δloss", "First sharp Δloss", "First sharp / ε"], counts),
        ("Search stopping", "The next candidate's measured loss explains threshold-based stopping. A reached cap leaves the number of additional directions unknown.",
         ["Task", "Tangent stopping", "Sharpness stopping", "Next tangent Δloss", "Next sharp Δloss"],
         [[row["task"], row["safe_stop_reason"], row["sharp_stop_reason"], next_loss(row, "safe"), next_loss(row, "sharp")] for row in tasks]),
        ("Interference matrix", "Rows = task kept safe. Columns = task harmed. Row A / column B reads the logged B_to_A metric: source B sharpness intersected with target A tangent. Diagonal cells are not measured. Zero denotes no verified overlap in the selected bases.",
         ["Safe task ↓ / Harmed task →"] + [row["id"] for row in tasks], matrix),
        ("Numerical quality", "Smaller residuals mean smaller equation errors for the checked directions. Tangent and sharpness absolute errors have different scales; do not compare them directly with epsilon. Using the full HVP budget is expected.",
         [label for _, label in quality_columns],
         [[row[field] for field, _ in quality_columns] for row in quality]),
        ("Run settings", "Missing settings are marked Not logged. The direction cap and the HVP budget control different quantities.",
         ["Setting", "Value"], settings_rows),
        ("Metric groups", "Every spaces/ summary value is retained in metrics.csv with its original name, group and explanation.",
         ["Group", "Number of metrics"], groups),
    ]


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(value: Any) -> str:
        return display(value).replace("|", r"\|").replace("\n", " ")
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(cell(value) for value in row) + " |" for row in rows]
    return "\n".join(lines)


def html_table(headers: list[str], rows: list[list[Any]], *, table_id: str = "") -> str:
    identifier = f' id="{html.escape(table_id, quote=True)}"' if table_id else ""
    header = "".join(f"<th scope='col'>{html.escape(label)}</th>" for label in headers)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(display(value))}</td>" for value in row) + "</tr>" for row in rows)
    return f"<div class='table-wrap'><table{identifier}><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>"


STYLE = """
*{box-sizing:border-box}body{margin:0;background:#f5f7fb;color:#172638;font:15px/1.6 system-ui,sans-serif}
main{max-width:1320px;margin:auto;padding:38px 28px}h1{font-size:34px;line-height:1.2;margin:0 0 12px}
h2{font-size:22px;margin:0 0 10px}.subtitle,.caption{color:#53647a}section{background:white;border:1px solid #e1e7f0;border-radius:14px;padding:24px;margin:22px 0}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:24px 0}
.card{padding:18px;background:#eaf1f9;border:1px solid #dbe5f2;border-radius:12px}.card strong{display:block;font-size:27px;color:#21466a}.card span{color:#455f7a}
.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:12px 14px;border-bottom:1px solid #e4eaf1;text-align:left;vertical-align:top}
th{background:#edf2f8;color:#263f59;white-space:nowrap}td:not(:first-child){font-variant-numeric:tabular-nums;white-space:nowrap}tbody tr:nth-child(even){background:#fafbfd}
td:first-child{overflow-wrap:anywhere;min-width:200px}a{color:#175d97}li{margin:10px 0}input{width:100%;padding:12px;border:1px solid #bdccdf;border-radius:8px;margin:10px 0 16px}
#raw-metrics td:nth-child(2),#raw-metrics td:nth-child(4){white-space:normal;overflow-wrap:anywhere;min-width:250px}
footer{color:#61738a;font-size:13px;overflow-wrap:anywhere}code{font-size:13px}@media(max-width:800px){main{padding:22px 12px}.cards{grid-template-columns:repeat(2,1fr)}section{padding:16px}}
@media print{body{background:white}main{max-width:none;padding:0}section{break-inside:avoid}.table-wrap{overflow:visible}input{display:none}table{font-size:10px}th,td{padding:5px}.cards{grid-template-columns:repeat(4,1fr)}}
"""


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def write_report(report: dict[str, Any], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    run_id = report["settings"]["run_id"]
    title = f"Loss-space report · {run_id}"
    readings = observations(report)
    grouped_tables = tables(report)
    markdown = [f"# {title}", "", f"Saved step: {display(report['settings']['step'])}. {len(report['catalog'])} space metrics grouped below.", "", "## Interpretation", ""]
    markdown += [f"- {line}" for line in readings]
    sections = []
    for heading, caption, headers, rows in grouped_tables:
        markdown += ["", f"## {heading}", "", caption, "", markdown_table(headers, rows)]
        sections.append(f"<section><h2>{html.escape(heading)}</h2><p class='caption'>{html.escape(caption)}</p>{html_table(headers, rows)}</section>")
    markdown += ["", "## Files and provenance", "", "CSV tables: task_summary.csv, interference_pairs.csv, interference_matrix.csv, numerical_quality.csv, metrics.csv.", "", f"Source summary: {report['source']}", ""]
    (directory / "report.md").write_text("\n".join(markdown), encoding="utf-8")
    write_csv(directory / "task_summary.csv", report["tasks"])
    write_csv(directory / "interference_pairs.csv", report["pairs"])
    write_csv(directory / "numerical_quality.csv", report["quality"])
    write_csv(directory / "metrics.csv", report["catalog"])
    pair_values = {(row["safe_task"], row["harmed_task"]): row["directions"] for row in report["pairs"]}
    write_csv(directory / "interference_matrix.csv", [
        {"safe_task": row["task"], **{
            other["task"]: "—" if row["task"] == other["task"] else pair_values[(row["task"], other["task"])]
            for other in report["tasks"]
        }} for row in report["tasks"]
    ])
    # Machine-readable data retains original numbers, including missing values.
    (directory / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    recorded = [row for row in report["pairs"] if number(row["directions"]) is not None]
    positives = sum(row["directions"] > 0 for row in recorded)
    capped = sum(row["safe_cap_reached"] == 1 for row in report["tasks"])
    sharp = [number(row["sharp_directions"]) for row in report["tasks"]]
    sharp = [value for value in sharp if value is not None]
    total_hvps = sum(number(row["hvp_count"]) or 0 for row in report["quality"])
    cards = [
        (f"{capped}/{len(report['tasks'])}", "Tangent caps reached"),
        (f"{display(min(sharp))}–{display(max(sharp))}" if sharp else "Not logged", "Harmful directions per task"),
        (f"{positives}/{len(recorded)}", "Pairs with verified overlap"),
        (display(total_hvps), "HVPs recorded across tasks"),
    ]
    card_html = "".join(f"<div class='card'><strong>{html.escape(value)}</strong><span>{html.escape(label)}</span></div>" for value, label in cards)
    interpretation = "<section><h2>Interpretation</h2><ul>" + "".join(f"<li>{html.escape(line)}</li>" for line in readings) + "</ul></section>"
    raw_rows = [[row[field] for field in ("group", "metric", "value", "meaning")] for row in report["catalog"]]
    raw = f"<section><details><summary>All {len(raw_rows)} original space metrics</summary><label for='metric-filter'>Filter by task, metric or group</label><input id='metric-filter' type='search' placeholder='Type a task or metric name'>{html_table(['Group', 'Original metric', 'Value', 'Meaning'], raw_rows, table_id='raw-metrics')}</details></section>"
    exports = "<section><h2>Downloads</h2><p>" + " · ".join(f"<a href='{name}'>{name}</a>" for name in ("report.md", "task_summary.csv", "interference_matrix.csv", "interference_pairs.csv", "numerical_quality.csv", "metrics.csv", "report.json")) + "</p></section>"
    script = """<script>document.getElementById('metric-filter').addEventListener('input',function(){const query=this.value.toLowerCase();document.querySelectorAll('#raw-metrics tbody tr').forEach(function(row){row.hidden=!row.textContent.toLowerCase().includes(query);});});</script>"""
    page = f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>{html.escape(title)}</title><style>{STYLE}</style></head><body><main><h1>{html.escape(title)}</h1><p class='subtitle'>Saved step {html.escape(display(report['settings']['step']))} · {len(report['tasks'])} tasks · {len(raw_rows)} space metrics · δ={html.escape(display(report['settings']['delta']))} · ε={html.escape(display(report['settings']['epsilon']))}</p><div class='cards'>{card_html}</div>{interpretation}{''.join(sections)}{exports}{raw}<footer>Source: {html.escape(report['source'])}</footer></main>{script}</body></html>"
    (directory / "report.html").write_text(page, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=Path("unzipped_results"), help="Run/archive directory, or a wandb-summary.json file.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/spaces_report"))
    parser.add_argument("--epsilon", type=float, help="Original measurement threshold if configuration is unavailable.")
    parser.add_argument("--delta", type=float, help="Original measurement radius if configuration is unavailable.")
    args = parser.parse_args(argv)
    if not args.input.exists():
        parser.error(f"Input does not exist: {args.input}")
    summaries = [args.input] if args.input.is_file() else sorted(args.input.rglob("wandb-summary.json"))
    if not summaries:
        parser.error(f"No wandb-summary.json files found under {args.input}")
    generated = []
    failures = []
    for path in summaries:
        try:
            report = load_report(path, epsilon=args.epsilon, delta=args.delta)
            run_id = report["settings"]["run_id"]
            if run_id in {item["run_id"] for item in generated}:
                raise ValueError(f"Duplicate run id {run_id}; analyze these runs into separate output directories.")
            destination = args.output_dir / run_id
            if destination.resolve() == path.parent.resolve() or destination.resolve() in path.resolve().parents:
                raise ValueError("Output directory must be separate from the input summary directory.")
            write_report(report, destination)
            generated.append({"run_id": run_id, "directory": destination})
            print(f"{run_id}: {len(report['tasks'])} tasks, {len(report['pairs'])} pairs, {len(report['catalog'])} metrics -> {destination / 'report.html'}")
        except (OSError, ValueError, TypeError) as error:
            failures.append(f"{path}: {error}")
    if not generated:
        parser.error("\n".join(failures))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    links = "".join(f"<li><a href='{html.escape(item['run_id'], quote=True)}/report.html'>{html.escape(item['run_id'])}</a></li>" for item in generated)
    skipped = "<ul>" + "".join(f"<li>{html.escape(message)}</li>" for message in failures) + "</ul>" if failures else ""
    (args.output_dir / "index.html").write_text(f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>Loss-space reports</title><style>{STYLE}</style></head><body><main><h1>Loss-space reports</h1><ul>{links}</ul>{skipped}</main></body></html>", encoding="utf-8")
    for message in failures:
        print(f"Skipped {message}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
