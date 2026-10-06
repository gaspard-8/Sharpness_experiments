# Loss-space report · wgrcwba5

Saved step: 10,000. 180 space metrics grouped below.

## Interpretation

- 8/8 tasks reached the tangent direction cap. Capped counts cannot rank the full tangent dimensions; each reported basis direction is individually safe.
- The search found 1–5 harmful directions per measured task at this radius and loss threshold.
- Among the first sharpness candidates, mean_middle_majority_parity_10 has the largest measured loss increase: 0.136632 (13.66 times epsilon).
- 0/56 recorded ordered pairs have a verified overlap (0 pairs missing). Zero means no overlap was found between the selected bases; other safe-for-one/harmful-for-another directions may exist.
- Checked relative eigen-equation errors range from 3.529e-06 to 4.386e-05. These describe the checked approximate modes and do not certify global extrema.
- Loss changes use the fixed space-measurement probes. The summaries do not describe held-out generalization, or the effects of arbitrary combinations of basis directions.
- Task and interference tables use scalar summaries. Full vectors and individual +/- loss changes are in the W&B loss-spaces artifact.
- Sharpness–sharpness overlap: 0/28 measured unordered pairs have shared directions. The symmetric counts use principal-angle cosines at or above the recorded overlap threshold.
- Tangent–tangent overlap: 0/28 measured unordered pairs have shared directions. The symmetric counts use principal-angle cosines at or above the recorded overlap threshold.
- Shared unit vectors are exported in overlap_directions.pt; overlap_directions.csv identifies each tensor column. At cosine below 1 a vector is a symmetric midpoint near both spans, rather than an exact intersection. Multiply it by delta to obtain the parameter perturbation.
- These overlaps were calculated from saved vectors. New joint +/- loss checks require the trained model; they are unavailable for this artifact and remain marked Not logged. Geometric overlap alone does not certify joint loss behavior.

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

| Task | HVPs used | Final active dimension | Budget used (1=yes) | Ritz relative error | Tangent absolute error | Sharpness absolute error |
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
| Maximum active basis dimension | Not logged |
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

## Sharpness–sharpness overlap matrix

Symmetric approximate geometric intersection of the discovered spans. Each unordered pair is computed once. Zero means no principal-angle cosine reached the overlap threshold; missing values mean no measurement is available. The diagonal is omitted. Counts do not establish the full parameter-space intersection dimension.

| Task ↓ / Task → | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T1 · isordered_0_10 | — | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| T2 · isrepeating_0_10 | 0 | — | 0 | 0 | 0 | 0 | 0 | 0 |
| T3 · Majority_0_10 | 0 | 0 | — | 0 | 0 | 0 | 0 | 0 |
| T4 · tribe_3_4 | 0 | 0 | 0 | — | 0 | 0 | 0 | 0 |
| T5 · mean_majority_parity_5 | 0 | 0 | 0 | 0 | — | 0 | 0 | 0 |
| T6 · mean_tribe_majority_tail | 0 | 0 | 0 | 0 | 0 | — | 0 | 0 |
| T7 · mean_middle_majority_parity_10 | 0 | 0 | 0 | 0 | 0 | 0 | — | 0 |
| T8 · mean_tribe_2_5_majority_tail | 0 | 0 | 0 | 0 | 0 | 0 | 0 | — |

## Sharpness overlap details

Maximum cosine shows the closest alignment even when no direction passes the threshold. Joint loss checks count shared directions satisfying this kind on both tasks: worst-sign loss increase > epsilon for sharpness, or <= epsilon for tangent. They require the model and can be fewer than geometric directions.

| Task A | Task B | Geometric directions | Joint loss checks passed | Maximum cosine | Source |
| --- | --- | --- | --- | --- | --- |
| isordered_0_10 | isrepeating_0_10 | 0 | Not logged | 0.85127 | Direction artifact |
| isordered_0_10 | Majority_0_10 | 0 | Not logged | 0.315855 | Direction artifact |
| isordered_0_10 | tribe_3_4 | 0 | Not logged | 0.488751 | Direction artifact |
| isordered_0_10 | mean_majority_parity_5 | 0 | Not logged | 0.36499 | Direction artifact |
| isordered_0_10 | mean_tribe_majority_tail | 0 | Not logged | 0.544943 | Direction artifact |
| isordered_0_10 | mean_middle_majority_parity_10 | 0 | Not logged | 0.0578262 | Direction artifact |
| isordered_0_10 | mean_tribe_2_5_majority_tail | 0 | Not logged | 0.301023 | Direction artifact |
| isrepeating_0_10 | Majority_0_10 | 0 | Not logged | 0.163953 | Direction artifact |
| isrepeating_0_10 | tribe_3_4 | 0 | Not logged | 0.620288 | Direction artifact |
| isrepeating_0_10 | mean_majority_parity_5 | 0 | Not logged | 0.241576 | Direction artifact |
| isrepeating_0_10 | mean_tribe_majority_tail | 0 | Not logged | 0.677647 | Direction artifact |
| isrepeating_0_10 | mean_middle_majority_parity_10 | 0 | Not logged | 0.0957353 | Direction artifact |
| isrepeating_0_10 | mean_tribe_2_5_majority_tail | 0 | Not logged | 0.359418 | Direction artifact |
| Majority_0_10 | tribe_3_4 | 0 | Not logged | 0.657379 | Direction artifact |
| Majority_0_10 | mean_majority_parity_5 | 0 | Not logged | 0.766167 | Direction artifact |
| Majority_0_10 | mean_tribe_majority_tail | 0 | Not logged | 0.554371 | Direction artifact |
| Majority_0_10 | mean_middle_majority_parity_10 | 0 | Not logged | 0.769007 | Direction artifact |
| Majority_0_10 | mean_tribe_2_5_majority_tail | 0 | Not logged | 0.293919 | Direction artifact |
| tribe_3_4 | mean_majority_parity_5 | 0 | Not logged | 0.464449 | Direction artifact |
| tribe_3_4 | mean_tribe_majority_tail | 0 | Not logged | 0.80885 | Direction artifact |
| tribe_3_4 | mean_middle_majority_parity_10 | 0 | Not logged | 0.285403 | Direction artifact |
| tribe_3_4 | mean_tribe_2_5_majority_tail | 0 | Not logged | 0.578767 | Direction artifact |
| mean_majority_parity_5 | mean_tribe_majority_tail | 0 | Not logged | 0.674958 | Direction artifact |
| mean_majority_parity_5 | mean_middle_majority_parity_10 | 0 | Not logged | 0.694682 | Direction artifact |
| mean_majority_parity_5 | mean_tribe_2_5_majority_tail | 0 | Not logged | 0.44138 | Direction artifact |
| mean_tribe_majority_tail | mean_middle_majority_parity_10 | 0 | Not logged | 0.474055 | Direction artifact |
| mean_tribe_majority_tail | mean_tribe_2_5_majority_tail | 0 | Not logged | 0.8629 | Direction artifact |
| mean_middle_majority_parity_10 | mean_tribe_2_5_majority_tail | 0 | Not logged | 0.431206 | Direction artifact |

## Tangent–tangent overlap matrix

Symmetric approximate geometric intersection of the discovered spans. Each unordered pair is computed once. Zero means no principal-angle cosine reached the overlap threshold; missing values mean no measurement is available. The diagonal is omitted. Counts do not establish the full parameter-space intersection dimension.

| Task ↓ / Task → | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T1 · isordered_0_10 | — | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| T2 · isrepeating_0_10 | 0 | — | 0 | 0 | 0 | 0 | 0 | 0 |
| T3 · Majority_0_10 | 0 | 0 | — | 0 | 0 | 0 | 0 | 0 |
| T4 · tribe_3_4 | 0 | 0 | 0 | — | 0 | 0 | 0 | 0 |
| T5 · mean_majority_parity_5 | 0 | 0 | 0 | 0 | — | 0 | 0 | 0 |
| T6 · mean_tribe_majority_tail | 0 | 0 | 0 | 0 | 0 | — | 0 | 0 |
| T7 · mean_middle_majority_parity_10 | 0 | 0 | 0 | 0 | 0 | 0 | — | 0 |
| T8 · mean_tribe_2_5_majority_tail | 0 | 0 | 0 | 0 | 0 | 0 | 0 | — |

## Tangent overlap details

Maximum cosine shows the closest alignment even when no direction passes the threshold. Joint loss checks count shared directions satisfying this kind on both tasks: worst-sign loss increase > epsilon for sharpness, or <= epsilon for tangent. They require the model and can be fewer than geometric directions.

| Task A | Task B | Geometric directions | Joint loss checks passed | Maximum cosine | Source |
| --- | --- | --- | --- | --- | --- |
| isordered_0_10 | isrepeating_0_10 | 0 | Not logged | 0.594952 | Direction artifact |
| isordered_0_10 | Majority_0_10 | 0 | Not logged | 0.402325 | Direction artifact |
| isordered_0_10 | tribe_3_4 | 0 | Not logged | 0.543293 | Direction artifact |
| isordered_0_10 | mean_majority_parity_5 | 0 | Not logged | 0.599718 | Direction artifact |
| isordered_0_10 | mean_tribe_majority_tail | 0 | Not logged | 0.587162 | Direction artifact |
| isordered_0_10 | mean_middle_majority_parity_10 | 0 | Not logged | 0.356556 | Direction artifact |
| isordered_0_10 | mean_tribe_2_5_majority_tail | 0 | Not logged | 0.411976 | Direction artifact |
| isrepeating_0_10 | Majority_0_10 | 0 | Not logged | 0.226062 | Direction artifact |
| isrepeating_0_10 | tribe_3_4 | 0 | Not logged | 0.383105 | Direction artifact |
| isrepeating_0_10 | mean_majority_parity_5 | 0 | Not logged | 0.39729 | Direction artifact |
| isrepeating_0_10 | mean_tribe_majority_tail | 0 | Not logged | 0.442834 | Direction artifact |
| isrepeating_0_10 | mean_middle_majority_parity_10 | 0 | Not logged | 0.210234 | Direction artifact |
| isrepeating_0_10 | mean_tribe_2_5_majority_tail | 0 | Not logged | 0.303686 | Direction artifact |
| Majority_0_10 | tribe_3_4 | 0 | Not logged | 0.214494 | Direction artifact |
| Majority_0_10 | mean_majority_parity_5 | 0 | Not logged | 0.242566 | Direction artifact |
| Majority_0_10 | mean_tribe_majority_tail | 0 | Not logged | 0.23828 | Direction artifact |
| Majority_0_10 | mean_middle_majority_parity_10 | 0 | Not logged | 0.824344 | Direction artifact |
| Majority_0_10 | mean_tribe_2_5_majority_tail | 0 | Not logged | 0.275082 | Direction artifact |
| tribe_3_4 | mean_majority_parity_5 | 0 | Not logged | 0.666211 | Direction artifact |
| tribe_3_4 | mean_tribe_majority_tail | 0 | Not logged | 0.850613 | Direction artifact |
| tribe_3_4 | mean_middle_majority_parity_10 | 0 | Not logged | 0.21559 | Direction artifact |
| tribe_3_4 | mean_tribe_2_5_majority_tail | 0 | Not logged | 0.732019 | Direction artifact |
| mean_majority_parity_5 | mean_tribe_majority_tail | 0 | Not logged | 0.77806 | Direction artifact |
| mean_majority_parity_5 | mean_middle_majority_parity_10 | 0 | Not logged | 0.235519 | Direction artifact |
| mean_majority_parity_5 | mean_tribe_2_5_majority_tail | 0 | Not logged | 0.577354 | Direction artifact |
| mean_tribe_majority_tail | mean_middle_majority_parity_10 | 0 | Not logged | 0.209161 | Direction artifact |
| mean_tribe_majority_tail | mean_tribe_2_5_majority_tail | 0 | Not logged | 0.727996 | Direction artifact |
| mean_middle_majority_parity_10 | mean_tribe_2_5_majority_tail | 0 | Not logged | 0.203494 | Direction artifact |

## Files and provenance

Downloads: report.md, task_summary.csv, interference_matrix.csv, interference_pairs.csv, numerical_quality.csv, metrics.csv, report.json, sharpness_overlap_pairs.csv, sharpness_overlap_matrix.csv, tangent_overlap_pairs.csv, tangent_overlap_matrix.csv, overlap_directions.pt, overlap_directions.csv.

Source summary: /Users/gaspard/Documents/Sharpness_experiments/unzipped_results/results/wandb/wandb/run-20261005_100714-wgrcwba5/files/wandb-summary.json

Source directions: /Users/gaspard/Documents/Sharpness_experiments/results/space_artifacts/wgrcwba5/spaces_step_10000.pt
