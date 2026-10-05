# Loss-space report · wgrcwba5

Saved step: 10,000. 180 space metrics grouped below.

## Interpretation

- 8/8 tasks reached the tangent direction cap. Capped counts cannot rank the full tangent dimensions; each reported basis direction is individually safe.
- The search found 1–5 harmful directions per measured task at this radius and loss threshold.
- Among the first sharpness candidates, mean_middle_majority_parity_10 has the largest measured loss increase: 0.136632 (13.66 times epsilon).
- 0/56 recorded ordered pairs have a verified overlap (0 pairs missing). Zero means no overlap was found between the selected bases; other safe-for-one/harmful-for-another directions may exist.
- Checked relative eigen-equation errors range from 3.529e-06 to 4.386e-05. These describe the checked approximate modes and do not certify global extrema.
- Loss changes use the fixed space-measurement probes. The summaries do not describe held-out generalization, or the effects of arbitrary combinations of basis directions.
- The tables use scalar summaries. Full vectors and individual +/- loss changes are in the W&B loss-spaces artifact.

## Task directions

Each count concerns independent basis directions found at the recorded delta and epsilon. The loss columns show actual worst-sign changes for the first candidate, not certified global extrema. A negative worst-sign change means both signs improved the measured loss.

| Task | Safe directions | Harmful directions | First safe Δloss | First sharp Δloss | First sharp / ε |
| --- | --- | --- | --- | --- | --- |
| T1 · isordered_0_10 | 16 (cap) | 1 | -6.700e-05 | 0.0853389 | 8.53× |
| T2 · isrepeating_0_10 | 16 (cap) | 1 | -3.087e-05 | 0.0807542 | 8.08× |
| T3 · Majority_0_10 | 16 (cap) | 3 | -6.862e-05 | 0.0523887 | 5.24× |
| T4 · tribe_3_4 | 16 (cap) | 2 | -2.661e-06 | 0.0503611 | 5.04× |
| T5 · mean_majority_parity_5 | 16 (cap) | 5 | -1.268e-04 | 0.0894971 | 8.95× |
| T6 · mean_tribe_majority_tail | 16 (cap) | 4 | 3.847e-04 | 0.043693 | 4.37× |
| T7 · mean_middle_majority_parity_10 | 16 (cap) | 5 | -2.021e-04 | 0.136632 | 13.66× |
| T8 · mean_tribe_2_5_majority_tail | 16 (cap) | 3 | -1.834e-04 | 0.11606 | 11.61× |

## Search stopping

The next candidate's measured loss explains threshold-based stopping. A reached cap leaves the number of additional directions unknown.

| Task | Tangent stopping | Sharpness stopping | Next tangent Δloss | Next sharp Δloss |
| --- | --- | --- | --- | --- |
| isordered_0_10 | Direction cap reached | Next candidate within epsilon | Not evaluated (cap) | 0.00868701 |
| isrepeating_0_10 | Direction cap reached | Next candidate within epsilon | Not evaluated (cap) | 0.00621483 |
| Majority_0_10 | Direction cap reached | Next candidate within epsilon | Not evaluated (cap) | 0.0085652 |
| tribe_3_4 | Direction cap reached | Next candidate within epsilon | Not evaluated (cap) | 0.00457713 |
| mean_majority_parity_5 | Direction cap reached | Next candidate within epsilon | Not evaluated (cap) | 0.00478182 |
| mean_tribe_majority_tail | Direction cap reached | Next candidate within epsilon | Not evaluated (cap) | 0.00437311 |
| mean_middle_majority_parity_10 | Direction cap reached | Next candidate within epsilon | Not evaluated (cap) | 0.00910733 |
| mean_tribe_2_5_majority_tail | Direction cap reached | Next candidate within epsilon | Not evaluated (cap) | 0.00996995 |

## Interference matrix

Rows = task kept safe. Columns = task harmed. Row A / column B reads the logged B_to_A metric: source B sharpness intersected with target A tangent. Diagonal cells are not measured. Zero denotes no verified overlap in the selected bases.

| Safe task ↓ / Harmed task → | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T1 · isordered_0_10 | — | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| T2 · isrepeating_0_10 | 0 | — | 0 | 0 | 0 | 0 | 0 | 0 |
| T3 · Majority_0_10 | 0 | 0 | — | 0 | 0 | 0 | 0 | 0 |
| T4 · tribe_3_4 | 0 | 0 | 0 | — | 0 | 0 | 0 | 0 |
| T5 · mean_majority_parity_5 | 0 | 0 | 0 | 0 | — | 0 | 0 | 0 |
| T6 · mean_tribe_majority_tail | 0 | 0 | 0 | 0 | 0 | — | 0 | 0 |
| T7 · mean_middle_majority_parity_10 | 0 | 0 | 0 | 0 | 0 | 0 | — | 0 |
| T8 · mean_tribe_2_5_majority_tail | 0 | 0 | 0 | 0 | 0 | 0 | 0 | — |

## Numerical quality

Smaller residuals mean smaller equation errors for the checked directions. Tangent and sharpness absolute errors have different scales; do not compare them directly with epsilon. Using the full HVP budget is expected.

| Task | HVPs used | Krylov dimension | Budget used (1=yes) | Ritz relative error | Tangent absolute error | Sharpness absolute error |
| --- | --- | --- | --- | --- | --- | --- |
| isordered_0_10 | 500 | 500 | 1 | 3.278e-05 | 5.824e-06 | 2.523e-07 |
| isrepeating_0_10 | 500 | 500 | 1 | 4.386e-05 | 5.007e-06 | 2.469e-07 |
| Majority_0_10 | 500 | 500 | 1 | 4.491e-06 | 3.224e-06 | 1.393e-07 |
| tribe_3_4 | 500 | 500 | 1 | 1.046e-05 | 1.717e-06 | 1.262e-07 |
| mean_majority_parity_5 | 500 | 500 | 1 | 3.893e-06 | 6.026e-06 | 2.833e-07 |
| mean_tribe_majority_tail | 500 | 500 | 1 | 4.715e-06 | 4.083e-06 | 1.218e-07 |
| mean_middle_majority_parity_10 | 500 | 500 | 1 | 3.529e-06 | 1.032e-05 | 4.425e-07 |
| mean_tribe_2_5_majority_tail | 500 | 500 | 1 | 8.458e-06 | 5.114e-06 | 1.431e-07 |

## Run settings

Missing settings are marked Not logged. The direction cap and the HVP budget control different quantities.

| Setting | Value |
| --- | --- |
| Run ID | wgrcwba5 |
| Final training step | 10,000 |
| Trainable parameters | 101,761 |
| Maximum directions per kind | 16 |
| HVP budget per task | 500 |
| Perturbation norm δ | 0.02 |
| Allowed loss increase ε | 0.01 |
| Probe inputs per task | 64 |
| Minimum overlap cosine | 0.999 |

## Metric groups

Every spaces/ summary value is retained in metrics.csv with its original name, group and explanation.

| Group | Number of metrics |
| --- | --- |
| Computation | 16 |
| Interference | 56 |
| Loss changes | 24 |
| Numerical quality | 24 |
| Search flags | 40 |
| Settings | 4 |
| Task directions | 16 |

## Files and provenance

CSV tables: task_summary.csv, interference_pairs.csv, interference_matrix.csv, numerical_quality.csv, metrics.csv.

Source summary: /Users/gaspard/Documents/Sharpness_experiments/unzipped_results/results/wandb/wandb/run-20261005_100714-wgrcwba5/files/wandb-summary.json
