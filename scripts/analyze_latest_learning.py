"""Offline, acquisition-focused analysis of the latest extracted W&B run.

Run with env/bin/python scripts/analyze_latest_learning.py. Does not train,
contact W&B, modify source results, or use the current metric implementation.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

from summarize_affinity import history

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/latest_learning_report"
DATA = OUT / "data"
NAMES = ["tribe_5_5", "tribe_5_4", "majority", "parity", "parity_10",
         "Majority_10", "weighted_voting_29", "hamming_weight_square_wave_29_4"]
LABELS = ["Tribe (5,5)", "Tribe (5,4)", "Majority-29", "Parity-29", "Parity-10",
          "Majority-10", "Weighted vote", "Square wave"]


def weighted_sensitivity(weights):
    threshold = sum(weights) // 2 + 1
    influences = []
    for i, weight in enumerate(weights):
        counts = [1]
        for j, other in enumerate(weights):
            if j == i:
                continue
            nxt = [0] * (len(counts) + other)
            for k, count in enumerate(counts):
                nxt[k] += count
                nxt[k + other] += count
            counts = nxt
        influences.append(sum(counts[max(0, threshold-weight):threshold]) / 2**(len(weights)-1))
    return sum(influences), influences


def gates(ev, diag, names, window=1000, threshold=.01, baseline=None):
    """Backward-looking filter; no future measurements choose a checkpoint.

    At least 20 evaluation readings (200 steps); first and last thirds of
    up to `window/10` readings have a positive gap >= threshold. OLS slope
    is positive, current observed accuracy < 1. Baseline is optional.
    """
    output, records = {}, []
    for name in names:
        a = ev[f"eval/accuracy/{name}"]
        mask = []
        for step in diag.index:
            values = a.loc[(a.index > step-window) & (a.index <= step)].to_numpy()
            third = len(values) // 3
            gain = float(values[-third:].mean() - values[:third].mean()) if third else np.nan
            slope = float(np.polyfit(np.arange(len(values)), values, 1)[0]) if len(values)>1 else np.nan
            increasing = len(values)>=20 and gain>=threshold and slope>0 and a.loc[step]<1
            above = baseline is None or a.loc[step] >= baseline[name] + .02
            active = increasing and above
            mask.append(active)
            records.append(dict(task=name, step=int(step), accuracy=a.loc[step], gain=gain,
                                slope_per_1000_steps=slope*100, above_constant_baseline=above,
                                increasing_below_one=increasing, selected=active))
        output[name] = np.array(mask)
    return output, records


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    run_paths = sorted((ROOT / "unzipped_results").rglob("run-*.wandb"), key=lambda p:p.parent.name)
    path = run_paths[-1]
    config = {k:v["value"] for k,v in yaml.safe_load((path.parent/"files/config.yaml").read_text()).items() if k!="_wandb"}
    assert config["selector_functions"] == NAMES, "Review analysis for a changed task mix."
    frame = history(path)
    assert frame["_step"].tolist() == list(range(1,30001))
    ev = frame.loc[frame["eval/accuracy"].notna()].set_index("_step")
    diag = frame.loc[frame["affinity/radius"].notna()].set_index("_step")
    assert ev.index.tolist() == list(range(10,30001,10))
    assert diag.index.tolist() == list(range(100,30001,100))
    assert (diag[[f"affinity/source_valid/{n}" for n in NAMES]] == 1).all().all()
    assert diag["affinity/radius"].unique().tolist() == [.02]
    assert diag.filter(regex=r"^(affinity/.*_to_|sharpness/)").notna().all().all()
    # Tables have W&B table artifact counters one less than completed steps.
    for step, row in diag.iterrows():
        table_path = next((path.parent/"files/media/table/affinity").glob(f"matrix_{step-1}_*.table.json"))
        table = json.loads(table_path.read_text())
        assert table["columns"] == ["source / target", *NAMES]
        assert [r[0] for r in table["data"]] == NAMES
        assert np.array_equal(np.array([r[1:] for r in table["data"]]),
                              [[row[f"affinity/{i}_to_{j}"] for j in NAMES] for i in NAMES])
    # Recreate exactly the run's fixed held-out set using archived source.
    import sys
    sys.path.insert(0, str(ROOT/"unzipped_results/results/code"))
    from src.functions import SelectorFunction, Tribe_ws, Majority, Parity, Parity_n, Majority_n, WeightedVoting, HammingWeightSquareWave
    from src.evaluation import make_evaluation_set
    fs = [Tribe_ws(5,5),Tribe_ws(5,4),Majority(),Parity(),Parity_n(10),Majority_n(10),
          WeightedVoting(29,list(range(1,30))),HammingWeightSquareWave(29,4)]
    eval_set = make_evaluation_set(SelectorFunction(fs),32,320,7)
    positive = eval_set.labels.reshape(8,320).sum(dim=1).tolist()
    baseline = {n:max(k,320-k)/320 for n,k in zip(NAMES,positive)}
    sensitivity = [25/16*(31/32)**4,20/16*(31/32)**3,29*math.comb(28,14)/2**28,
                   29.,10.,10*math.comb(9,5)/2**9,weighted_sensitivity(list(range(1,30)))[0],
                   29*sum(math.comb(28,k) for k in range(3,29,4))/2**28]
    pop_positive = [1-(31/32)**5,1-(31/32)**4,.5,.5,.5,
                    sum(math.comb(10,k) for k in range(6,11))/2**10,.5,
                    sum(math.comb(29,k) for k in range(30) if (k//4)%2)/2**29]
    active, selections = gates(ev,diag,NAMES,baseline=baseline)
    pd.DataFrame(selections).to_csv(DATA/"checkpoint_selection.csv",index=False)
    variants = {"primary_1000_1pp":active}
    for window,threshold in [(500,.01),(1000,.02),(2000,.01),(2000,.02)]:
        variants[f"window_{window}_gain_{100*threshold:g}pp"] = gates(ev,diag,NAMES,window,threshold,baseline)[0]
    variants["1000_1pp_without_baseline_gate"] = gates(ev,diag,NAMES)[0]
    rows=[]
    for n,label,s,pp,k in zip(NAMES,LABELS,sensitivity,pop_positive,positive):
        a=ev[f"eval/accuracy/{n}"]
        sustained=a.rolling(100,min_periods=100).median()
        first99=ev.index[sustained>=.99]
        first95=ev.index[sustained>=.95]
        sel=active[n]
        sh=diag[f"sharpness/{n}"]
        late=sh.loc[20000:30000]
        early=sh.loc[:1000]
        acquiring=sh[sel]
        record=dict(task=n,label=label,sensitivity=s,population_positive_rate=pp,
                    eval_positive_count=k,constant_baseline=baseline[n],
                    final_accuracy=a.iloc[-1],late_accuracy_median=a.loc[20000:30000].median(),
                    final_loss=ev[f"eval/loss/{n}"].iloc[-1],
                    first95=int(ev.index[a>=.95][0]),first99=int(ev.index[a>=.99][0]),
                    sustained95=int(first95[0]) if len(first95) else None,
                    sustained99=int(first99[0]) if len(first99) else None,
                    active_n=int(sel.sum()),active_first=int(diag.index[sel][0]),active_last=int(diag.index[sel][-1]),
                    sharpness_early=float(early.median()),sharpness_active=float(acquiring.median()),
                    sharpness_late=float(late.median()),sharpness_late_q25=float(late.quantile(.25)),
                    sharpness_late_q75=float(late.quantile(.75)),
                    self_active_median=float(diag.loc[sel,f"affinity/{n}_to_{n}"].median()),
                    self_active_positive_fraction=float((diag.loc[sel,f"affinity/{n}_to_{n}"]>0).mean()),
                    self_late_positive_fraction=float((diag.loc[20000:30000,f"affinity/{n}_to_{n}"]>0).mean()),
                    affinity_baseline_active=float(diag.loc[sel,f"affinity/baseline_loss/{n}"].median()))
        rows.append(record)
    tasks=pd.DataFrame(rows)
    tasks.to_csv(DATA/"task_summary.csv",index=False)
    pair_rows=[]
    for variant,masks in variants.items():
        for mode in ["target_active","both_active"]:
            for i in NAMES:
                for j in NAMES:
                    mask=masks[j] & (masks[i] if mode=="both_active" else True)
                    a=diag.loc[mask,f"affinity/{i}_to_{j}"]
                    b=diag.loc[mask,f"affinity/baseline_loss/{j}"]
                    ia,ja=NAMES.index(i),NAMES.index(j)
                    key=f"gradient_alignment/{NAMES[min(ia,ja)]}_vs_{NAMES[max(ia,ja)]}"
                    alignment=diag.loc[mask,key] if i!=j else pd.Series([1.])
                    pair_rows.append(dict(variant=variant,mode=mode,source=i,target=j,n=len(a),
                        median=float(a.median()) if len(a) else None,
                        q25=float(a.quantile(.25)) if len(a) else None,q75=float(a.quantile(.75)) if len(a) else None,
                        mean=float(a.mean()) if len(a) else None,positive_fraction=float((a>0).mean()) if len(a) else None,
                        median_fraction_target_loss=float((a/b).median()) if len(a) else None,
                        gradient_alignment_median=float(alignment.median()) if len(a) else None))
    pairs=pd.DataFrame(pair_rows)
    pairs.to_csv(DATA/"affinity_pairs.csv",index=False)
    # Fixed-step, matched-time comparisons supplement the task-specific masks.
    phases=[]
    for lo,hi in [(100,1000),(1100,3000),(3100,7000),(7100,12000),(12100,20000),(20100,30000)]:
        d=diag.loc[lo:hi]
        for i in NAMES:
            for j in NAMES:
                a=d[f"affinity/{i}_to_{j}"]
                phases.append(dict(start=lo,end=hi,source=i,target=j,n=len(a),median=a.median(),
                                   positive_fraction=(a>0).mean()))
    pd.DataFrame(phases).to_csv(DATA/"affinity_phases.csv",index=False)
    # All diagnostic points are exportable without a W&B dependency.
    ev[[c for c in ev if c.startswith("eval/")]].to_csv(DATA/"evaluation_history.csv")
    diag[[c for c in diag if c.startswith(("affinity/","sharpness","gradient_","curvature/"))
          and not c.startswith("affinity/matrix/")]].to_csv(DATA/"diagnostics.csv")
    for n in NAMES:
        ev[f"smooth/{n}"]=ev[f"eval/accuracy/{n}"].rolling(50,min_periods=1).median()
    plotdata=dict(steps=ev.index.tolist(),accuracies={n:ev[f"eval/accuracy/{n}"].tolist() for n in NAMES},
                  smooth={n:ev[f"smooth/{n}"].tolist() for n in NAMES},diagnostic_steps=diag.index.tolist(),
                  sharpness={n:diag[f"sharpness/{n}"].tolist() for n in NAMES},
                  affinities={f"{i}_to_{j}":diag[f"affinity/{i}_to_{j}"].tolist() for i in NAMES for j in NAMES},
                  selected={n:active[n].tolist() for n in NAMES})
    (DATA/"plot_data.json").write_text(json.dumps(plotdata))
    late=diag.loc[20000:30000]
    offkeys=[f"affinity/{i}_to_{j}" for i in NAMES for j in NAMES if i!=j]
    p=pairs[(pairs.variant=="primary_1000_1pp") & (pairs['mode']=="target_active") & (pairs.source!=pairs.target)]
    all_active=np.concatenate([diag.loc[active[j],f"affinity/{i}_to_{j}"].to_numpy() for j in NAMES for i in NAMES if i!=j])
    corr={k:float(spearmanr(tasks.sensitivity,tasks[k]).statistic) for k in
          ["sharpness_early","sharpness_active","sharpness_late","first95","sustained95","sustained99"]}
    late_windows={}
    for lo,hi in [(10000,20000),(15000,25000),(20000,30000),(25000,30000)]:
        sh=[diag.loc[lo:hi,f"sharpness/{n}"].median() for n in NAMES]
        late_windows[f"{lo}-{hi}"]={"sharpness":sh,"spearman_sensitivity":float(spearmanr(sensitivity,sh).statistic)}
    inventory=[]
    for pth in run_paths:
        c={k:v['value'] for k,v in yaml.safe_load((pth.parent/'files/config.yaml').read_text()).items() if k!='_wandb'}
        inventory.append(dict(run=pth.stem[4:],folder=pth.parent.name,steps=c['train_num_steps'],
                              length=c['train_max_len'],tasks=c['selector_functions']))
    summary=dict(run=path.stem[4:],source=str(path.relative_to(ROOT)),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        config=config,inventory=inventory,tasks=rows,correlations=corr,late_windows=late_windows,
        primary_pairs=p.to_dict('records'),
        late_negative_fraction=float((late[offkeys]<0).to_numpy().mean()),
        active_positive_fraction=float((all_active>0).mean()),
        active_pairs_with_positive_median=int((p['median']>0).sum()),
        gate_counts={v:{n:int(m[n].sum()) for n in NAMES} for v,m in variants.items()},
        verification={'training_rows':len(frame),'evaluation_rows':len(ev),'affinity_rows':len(diag),
                      'table_scalar_matrices_exactly_equal':300,'all_source_directions_valid':True},
        function_parameter_influences={'weighted_voting':weighted_sensitivity(list(range(1,30)))[1]})
    (DATA/'analysis.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    print(tasks.to_string(index=False))
    print('CORRELATIONS',corr,'POSITIVE fraction',summary['active_positive_fraction'],'LATE NEGATIVE',summary['late_negative_fraction'])
    print('TOP PRIMARY\n',p.sort_values('median',ascending=False)[['source','target','n','median','positive_fraction','gradient_alignment_median']].head(16).to_string(index=False))
    print('BOTTOM PRIMARY\n',p.sort_values('median')[['source','target','n','median','positive_fraction','gradient_alignment_median']].head(10).to_string(index=False))


if __name__ == '__main__':
    main()
