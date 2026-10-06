"""Reproduce the local affinity analysis without contacting W&B.

Run with the repository environment: env/bin/python scripts/summarize_affinity.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/affinity_report"


def history(path):
    store = DataStore()
    store.open_for_scan(str(path))
    rows = []
    while True:
        data = store.scan_data()
        if data is None:
            break
        record = wandb_internal_pb2.Record()
        record.ParseFromString(data)
        if record.HasField("history"):
            rows.append({"/".join(x.nested_key) if x.nested_key else x.key:
                         json.loads(x.value_json) for x in record.history.item})
    return pd.DataFrame(rows).sort_values("_step")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs = []
    for path in sorted((ROOT / "unzipped_results").rglob("*.wandb")):
        df = history(path)
        config = {k: v["value"] for k, v in
                  yaml.safe_load((path.parent / "files/config.yaml").read_text()).items()}
        run = {"id": path.stem.removeprefix("run-"), "source": str(path),
               "config": {k: v for k, v in config.items() if k != "_wandb"},
               "final": json.loads((path.parent / "files/wandb-summary.json").read_text())}
        run["eval_series"] = df.loc[df["eval/loss"].notna(),
                                     ["_step", "eval/loss", "eval/accuracy"]].values.tolist()
        runs.append(run)
        if "affinity/radius" not in df:
            continue
        names = config["selector_functions"]
        measurements = df[df["affinity/radius"].notna()].copy()
        assert measurements["_step"].tolist() == list(range(100, 30001, 100))
        assert (measurements.filter(regex=r"^affinity/source_valid/") == 1).all().all()
        radius = measurements["affinity/radius"].unique().tolist()
        assert radius == [0.02]
        learned = (measurements[[f"eval/accuracy/{n}" for n in names]] >= .99).all(axis=1)
        measurements["included_primary"] = (measurements["_step"] >= 10000) & learned
        measurements[["_step", "included_primary"]].to_csv(OUT / "checkpoint_selection.csv", index=False)

        def matrix(data):
            return np.stack([data[[f"affinity/{i}_to_{j}" for j in names]].to_numpy()
                             for i in names], axis=1)

        selected = measurements[measurements["included_primary"]]
        values = matrix(selected)
        magnitude = np.median(np.abs(values), axis=0)
        signed = np.median(values, axis=0)
        q1, q3 = np.quantile(np.abs(values), [.25, .75], axis=0)
        ordered = np.sort(np.abs(values), axis=0)
        trim = int(len(ordered) * .1)
        trimmed = ordered[trim:-trim].mean(axis=0)
        off = ~np.eye(len(names), dtype=bool)
        source_scores = (magnitude * off).sum(axis=1) / 7
        target_scores = (magnitude * off).sum(axis=0) / 7
        relative = magnitude / np.diag(magnitude)[None, :]
        for filename, data in [("median_magnitude_matrix", magnitude), ("median_signed_matrix", signed),
                               ("magnitude_q25_matrix", q1), ("magnitude_q75_matrix", q3),
                               ("trimmed_mean_magnitude_matrix", trimmed), ("target_self_ratio_matrix", relative)]:
            pd.DataFrame(data, index=names, columns=names).rename_axis("source").to_csv(OUT / (filename + ".csv"))
        pairs = sorted([{"source": names[i], "target": names[j], "magnitude": magnitude[i,j],
                         "q25": q1[i,j], "q75": q3[i,j], "reverse": magnitude[j,i],
                         "target_self_ratio": relative[i,j]} for i in range(8) for j in range(8) if i != j],
                       key=lambda p: p["magnitude"], reverse=True)
        pd.DataFrame(pairs).to_csv(OUT / "directed_pairs.csv", index=False)
        task_table = pd.DataFrame({"task": names, "source_score": source_scores, "target_score": target_scores,
                                   "self_magnitude": np.diag(magnitude),
                                   "probe_baseline": [selected[f"affinity/baseline_loss/{n}"].median() for n in names],
                                   "source_gradient_norm": [selected[f"affinity/source_gradient_norm/{n}"].median() for n in names]})
        task_table.to_csv(OUT / "task_summary.csv", index=False)
        windows = []
        for lo, gate in [(5500, True), (10000, False), (10000, True), (20000, True), (25000, True)]:
            sel = measurements[(measurements["_step"] >= lo) & (learned if gate else True)]
            v = matrix(sel)
            m = np.median(np.abs(v), axis=0)
            flat_rank = pd.Series(m[off]).rank().to_numpy()
            base_rank = pd.Series(magnitude[off]).rank().to_numpy()
            windows.append({"start": lo, "gate": gate, "n": len(sel), "median_magnitude": m.tolist(),
                            "source_scores": ((m * off).sum(axis=1)/7).tolist(),
                            "target_scores": ((m * off).sum(axis=0)/7).tolist(),
                            "rank_correlation": float(np.corrcoef(flat_rank, base_rank)[0,1]),
                            "max_cell_relative_change": float(np.max(np.abs(m-magnitude)/magnitude))})
        alignment = df[(df["_step"] >= 10000)].filter(regex=r"^gradient_alignment/")
        alignment_med = alignment.median().sort_values()
        alignment_med.rename("median_cosine").to_csv(OUT / "gradient_alignment_summary.csv")
        curve = []
        for lo in range(100, 30001, 1000):
            sel = measurements[(measurements["_step"] >= lo) & (measurements["_step"] < lo+1000)]
            v = matrix(sel)
            curve.append([float(sel["_step"].median()), float(np.median(np.abs(v[:, off]))),
                          float(np.median(np.abs(v[:, np.eye(8,dtype=bool)])))])
        run["affinity"] = {"names": names, "n_total": len(measurements), "n_primary": len(selected),
                           "excluded_steps": measurements[(measurements["_step"] >= 10000) & ~learned]["_step"].tolist(),
                           "magnitude": magnitude.tolist(), "signed": signed.tolist(),
                           "q25": q1.tolist(), "q75": q3.tolist(), "trimmed": trimmed.tolist(),
                           "relative": relative.tolist(), "task_summary": task_table.to_dict("records"),
                           "pairs": pairs, "windows": windows, "curve": curve,
                           "negative_fraction": float(np.mean(values[:,off] < 0)),
                           "diagonal_negative_fraction": float(np.mean(values[:,np.eye(8,dtype=bool)] < 0)),
                           "pooled_quantiles": np.quantile(np.abs(values[:,off]), [.25,.5,.75]).tolist(),
                           "alignment_positive_medians": int((alignment_med > 0).sum()),
                           "alignment_medians": alignment_med.to_dict(),
                           "late_eval_loss_median": float(df[df["_step"] >= 10000]["eval/loss"].median()),
                           "first_all99": float(df[(df[[f"eval/accuracy/{n}" for n in names]] >= .99).all(axis=1)]["_step"].min())}
        # A scalar history and table agree at every saved checkpoint.
        for row in measurements.to_dict("records"):
            table_file = next((path.parent / "files/media/table/affinity").glob(f"matrix_{int(row['_step'])-1}_*.table.json"))
            table = json.loads(table_file.read_text())
            assert table["columns"] == ["source / target", *names]
            assert [r[0] for r in table["data"]] == names
            assert np.array_equal(np.array([r[1:] for r in table["data"]]),
                                  np.array([[row[f"affinity/{i}_to_{j}"] for j in names] for i in names]))
    output = {"runs": runs, "method": "Median absolute affinity; primary steps 10000-30000, all task eval accuracies >=0.99; 10% trimmed-mean and window sensitivity."}
    (OUT / "analysis.json").write_text(json.dumps(output, indent=2))
    latest = next(x for x in runs if "affinity" in x)["affinity"]
    print(json.dumps({k: latest[k] for k in ["n_total", "n_primary", "excluded_steps", "negative_fraction", "alignment_positive_medians"]}, indent=2))
    print("Cross-checked all 300 scalar matrices against saved W&B tables.")


if __name__ == "__main__":
    main()
