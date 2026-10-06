# Loss-space report · y9er89ad

Saved step: 10,000. 197 space metrics grouped below.

## Interpretation

- 6/8 tasks reached the tangent direction cap. Capped counts cannot rank the full tangent dimensions; each reported basis direction is individually safe.
- The search found 1–5 harmful directions per measured task at this radius and loss threshold.
- Among the first sharpness candidates, tribe_7_3 has the largest measured loss increase: 0.0326262 (16.31 times epsilon).
- 0/56 recorded ordered pairs have a verified overlap (0 pairs missing). Zero means no overlap was found between the selected bases; other safe-for-one/harmful-for-another directions may exist.
- Checked relative eigen-equation errors range from 5.751e-05 to 0.387852. These describe the checked approximate modes and do not certify global extrema.
- HVP counts measure total computation across restart cycles. The Krylov dimension is the final active basis, bounded by 512 vectors; it is not the number of parameter coordinates explored. Restarts retain extreme eigenvector estimates and gradient seeds.
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
| T1 · tribe_21_1 | 100 (cap) | 1 | -4.768e-06 | 0.0283809 | 14.19× |
| T2 · tribe_10_2 | 100 (cap) | 2 | -7.364e-05 | 0.0288717 | 14.44× |
| T3 · tribe_7_3 | 100 (cap) | 2 | -8.028e-04 | 0.0326262 | 16.31× |
| T4 · tribe_5_4 | 100 (cap) | 3 | -0.00122667 | 0.0289902 | 14.50× |
| T5 · tribe_4_5 | 0 | 5 | 0.00951939 | 0.0229319 | 11.47× |
| T6 · tribe_3_6 | 100 (cap) | 4 | -1.022e-04 | 0.0278345 | 13.92× |
| T7 · tribe_3_7 | 100 (cap) | 3 | 4.179e-04 | 0.0212988 | 10.65× |
| T8 · tribe_2_8 | 0 | 4 | 0.00908267 | 0.0270632 | 13.53× |

## Search stopping

The next candidate's measured loss explains threshold-based stopping. A reached cap leaves the number of additional directions unknown.

| Task | Tangent stopping | Sharpness stopping | Next tangent Δloss | Next sharp Δloss |
| --- | --- | --- | --- | --- |
| tribe_21_1 | Direction cap reached | Next candidate within epsilon | Not evaluated (cap) | 3.435e-04 |
| tribe_10_2 | Direction cap reached | Next candidate within epsilon | Not evaluated (cap) | 2.459e-04 |
| tribe_7_3 | Direction cap reached | Next candidate within epsilon | Not evaluated (cap) | 0.00154122 |
| tribe_5_4 | Direction cap reached | Next candidate within epsilon | Not evaluated (cap) | 7.561e-04 |
| tribe_4_5 | Next candidate above epsilon | Next candidate within epsilon | 0.00951939 | 9.259e-04 |
| tribe_3_6 | Direction cap reached | Next candidate within epsilon | Not evaluated (cap) | 0.00174916 |
| tribe_3_7 | Direction cap reached | Next candidate within epsilon | Not evaluated (cap) | 8.779e-04 |
| tribe_2_8 | Next candidate above epsilon | Next candidate within epsilon | 0.00908267 | 0.00134665 |

## Interference matrix

Rows = task kept safe. Columns = task harmed. Row A / column B reads the logged B_to_A metric: source B sharpness intersected with target A tangent. Diagonal cells are not measured. Zero denotes no verified overlap in the selected bases.

| Safe task ↓ / Harmed task → | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T1 · tribe_21_1 | — | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| T2 · tribe_10_2 | 0 | — | 0 | 0 | 0 | 0 | 0 | 0 |
| T3 · tribe_7_3 | 0 | 0 | — | 0 | 0 | 0 | 0 | 0 |
| T4 · tribe_5_4 | 0 | 0 | 0 | — | 0 | 0 | 0 | 0 |
| T5 · tribe_4_5 | 0 | 0 | 0 | 0 | — | 0 | 0 | 0 |
| T6 · tribe_3_6 | 0 | 0 | 0 | 0 | 0 | — | 0 | 0 |
| T7 · tribe_3_7 | 0 | 0 | 0 | 0 | 0 | 0 | — | 0 |
| T8 · tribe_2_8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | — |

## Numerical quality

Smaller residuals mean smaller equation errors for the checked directions. Tangent and sharpness absolute errors have different scales; do not compare them directly with epsilon. Using the full HVP budget is expected.

| Task | HVPs used | Final active dimension | Budget used (1=yes) | Thick restarts | Full parameter basis (1=yes) | Ritz relative error | Tangent absolute error | Sharpness absolute error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tribe_21_1 | 10,000 | 272 | 1 | 32 | 0 | 0.00237547 | 2.535e-05 | 4.838e-08 |
| tribe_10_2 | 10,000 | 272 | 1 | 32 | 0 | 4.976e-04 | 1.103e-05 | 4.301e-08 |
| tribe_7_3 | 10,000 | 272 | 1 | 32 | 0 | 0.387852 | 0.0641785 | 4.269e-07 |
| tribe_5_4 | 10,000 | 272 | 1 | 32 | 0 | 0.061602 | 0.0334686 | 2.177e-05 |
| tribe_4_5 | 10,000 | 272 | 1 | 32 | 0 | 5.751e-05 | Not logged | 8.493e-08 |
| tribe_3_6 | 10,000 | 272 | 1 | 32 | 0 | 0.112645 | 0.017016 | 1.271e-06 |
| tribe_3_7 | 10,000 | 272 | 1 | 32 | 0 | 2.314e-04 | 3.191e-05 | 3.340e-08 |
| tribe_2_8 | 10,000 | 272 | 1 | 32 | 0 | 2.368e-04 | Not logged | 5.725e-08 |

## Run settings

Missing settings are marked Not logged. The direction cap and the HVP budget control different quantities.

| Setting | Value |
| --- | --- |
| Run ID | y9er89ad |
| Final training step | 10,000 |
| Trainable parameters | 101,761 |
| Maximum directions per kind | 100 |
| HVP budget per task | 10,000 |
| Maximum active basis dimension | 512 |
| Perturbation norm δ | 0.02 |
| Allowed loss increase ε | 0.002 |
| Probe inputs per task | 64 |
| Minimum overlap cosine | 0.999 |

## Metric groups

Every spaces/ summary value is retained in metrics.csv with its original name, group and explanation.

| Group | Number of metrics |
| --- | --- |
| Computation | 24 |
| Interference | 56 |
| Loss changes | 26 |
| Numerical quality | 22 |
| Search flags | 48 |
| Settings | 5 |
| Task directions | 16 |

## Sharpness–sharpness overlap matrix

Symmetric approximate geometric intersection of the discovered spans. Each unordered pair is computed once. Zero means no principal-angle cosine reached the overlap threshold; missing values mean no measurement is available. The diagonal is omitted. Counts do not establish the full parameter-space intersection dimension.

| Task ↓ / Task → | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T1 · tribe_21_1 | — | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| T2 · tribe_10_2 | 0 | — | 0 | 0 | 0 | 0 | 0 | 0 |
| T3 · tribe_7_3 | 0 | 0 | — | 0 | 0 | 0 | 0 | 0 |
| T4 · tribe_5_4 | 0 | 0 | 0 | — | 0 | 0 | 0 | 0 |
| T5 · tribe_4_5 | 0 | 0 | 0 | 0 | — | 0 | 0 | 0 |
| T6 · tribe_3_6 | 0 | 0 | 0 | 0 | 0 | — | 0 | 0 |
| T7 · tribe_3_7 | 0 | 0 | 0 | 0 | 0 | 0 | — | 0 |
| T8 · tribe_2_8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | — |

## Sharpness overlap details

Maximum cosine shows the closest alignment even when no direction passes the threshold. Joint loss checks count shared directions satisfying this kind on both tasks: worst-sign loss increase > epsilon for sharpness, or <= epsilon for tangent. They require the model and can be fewer than geometric directions.

| Task A | Task B | Geometric directions | Joint loss checks passed | Maximum cosine | Source |
| --- | --- | --- | --- | --- | --- |
| tribe_21_1 | tribe_10_2 | 0 | Not logged | 0.989442 | Direction artifact |
| tribe_21_1 | tribe_7_3 | 0 | Not logged | 0.989851 | Direction artifact |
| tribe_21_1 | tribe_5_4 | 0 | Not logged | 0.976727 | Direction artifact |
| tribe_21_1 | tribe_4_5 | 0 | Not logged | 0.968405 | Direction artifact |
| tribe_21_1 | tribe_3_6 | 0 | Not logged | 0.932368 | Direction artifact |
| tribe_21_1 | tribe_3_7 | 0 | Not logged | 0.915752 | Direction artifact |
| tribe_21_1 | tribe_2_8 | 0 | Not logged | 0.773311 | Direction artifact |
| tribe_10_2 | tribe_7_3 | 0 | Not logged | 0.972276 | Direction artifact |
| tribe_10_2 | tribe_5_4 | 0 | Not logged | 0.984394 | Direction artifact |
| tribe_10_2 | tribe_4_5 | 0 | Not logged | 0.965967 | Direction artifact |
| tribe_10_2 | tribe_3_6 | 0 | Not logged | 0.956507 | Direction artifact |
| tribe_10_2 | tribe_3_7 | 0 | Not logged | 0.923607 | Direction artifact |
| tribe_10_2 | tribe_2_8 | 0 | Not logged | 0.788504 | Direction artifact |
| tribe_7_3 | tribe_5_4 | 0 | Not logged | 0.978735 | Direction artifact |
| tribe_7_3 | tribe_4_5 | 0 | Not logged | 0.948967 | Direction artifact |
| tribe_7_3 | tribe_3_6 | 0 | Not logged | 0.909377 | Direction artifact |
| tribe_7_3 | tribe_3_7 | 0 | Not logged | 0.903707 | Direction artifact |
| tribe_7_3 | tribe_2_8 | 0 | Not logged | 0.787155 | Direction artifact |
| tribe_5_4 | tribe_4_5 | 0 | Not logged | 0.962279 | Direction artifact |
| tribe_5_4 | tribe_3_6 | 0 | Not logged | 0.958834 | Direction artifact |
| tribe_5_4 | tribe_3_7 | 0 | Not logged | 0.939369 | Direction artifact |
| tribe_5_4 | tribe_2_8 | 0 | Not logged | 0.850424 | Direction artifact |
| tribe_4_5 | tribe_3_6 | 0 | Not logged | 0.978598 | Direction artifact |
| tribe_4_5 | tribe_3_7 | 0 | Not logged | 0.976621 | Direction artifact |
| tribe_4_5 | tribe_2_8 | 0 | Not logged | 0.932436 | Direction artifact |
| tribe_3_6 | tribe_3_7 | 0 | Not logged | 0.987915 | Direction artifact |
| tribe_3_6 | tribe_2_8 | 0 | Not logged | 0.982788 | Direction artifact |
| tribe_3_7 | tribe_2_8 | 0 | Not logged | 0.979711 | Direction artifact |

## Tangent–tangent overlap matrix

Symmetric approximate geometric intersection of the discovered spans. Each unordered pair is computed once. Zero means no principal-angle cosine reached the overlap threshold; missing values mean no measurement is available. The diagonal is omitted. Counts do not establish the full parameter-space intersection dimension.

| Task ↓ / Task → | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T1 · tribe_21_1 | — | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| T2 · tribe_10_2 | 0 | — | 0 | 0 | 0 | 0 | 0 | 0 |
| T3 · tribe_7_3 | 0 | 0 | — | 0 | 0 | 0 | 0 | 0 |
| T4 · tribe_5_4 | 0 | 0 | 0 | — | 0 | 0 | 0 | 0 |
| T5 · tribe_4_5 | 0 | 0 | 0 | 0 | — | 0 | 0 | 0 |
| T6 · tribe_3_6 | 0 | 0 | 0 | 0 | 0 | — | 0 | 0 |
| T7 · tribe_3_7 | 0 | 0 | 0 | 0 | 0 | 0 | — | 0 |
| T8 · tribe_2_8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | — |

## Tangent overlap details

Maximum cosine shows the closest alignment even when no direction passes the threshold. Joint loss checks count shared directions satisfying this kind on both tasks: worst-sign loss increase > epsilon for sharpness, or <= epsilon for tangent. They require the model and can be fewer than geometric directions.

| Task A | Task B | Geometric directions | Joint loss checks passed | Maximum cosine | Source |
| --- | --- | --- | --- | --- | --- |
| tribe_21_1 | tribe_10_2 | 0 | Not logged | 0.948128 | Direction artifact |
| tribe_21_1 | tribe_7_3 | 0 | Not logged | 0.667052 | Direction artifact |
| tribe_21_1 | tribe_5_4 | 0 | Not logged | 0.71733 | Direction artifact |
| tribe_21_1 | tribe_4_5 | 0 | Not logged | Not logged | Direction artifact |
| tribe_21_1 | tribe_3_6 | 0 | Not logged | 0.541104 | Direction artifact |
| tribe_21_1 | tribe_3_7 | 0 | Not logged | 0.879023 | Direction artifact |
| tribe_21_1 | tribe_2_8 | 0 | Not logged | Not logged | Direction artifact |
| tribe_10_2 | tribe_7_3 | 0 | Not logged | 0.843516 | Direction artifact |
| tribe_10_2 | tribe_5_4 | 0 | Not logged | 0.743403 | Direction artifact |
| tribe_10_2 | tribe_4_5 | 0 | Not logged | Not logged | Direction artifact |
| tribe_10_2 | tribe_3_6 | 0 | Not logged | 0.69417 | Direction artifact |
| tribe_10_2 | tribe_3_7 | 0 | Not logged | 0.773472 | Direction artifact |
| tribe_10_2 | tribe_2_8 | 0 | Not logged | Not logged | Direction artifact |
| tribe_7_3 | tribe_5_4 | 0 | Not logged | 0.747075 | Direction artifact |
| tribe_7_3 | tribe_4_5 | 0 | Not logged | Not logged | Direction artifact |
| tribe_7_3 | tribe_3_6 | 0 | Not logged | 0.696252 | Direction artifact |
| tribe_7_3 | tribe_3_7 | 0 | Not logged | 0.663402 | Direction artifact |
| tribe_7_3 | tribe_2_8 | 0 | Not logged | Not logged | Direction artifact |
| tribe_5_4 | tribe_4_5 | 0 | Not logged | Not logged | Direction artifact |
| tribe_5_4 | tribe_3_6 | 0 | Not logged | 0.538552 | Direction artifact |
| tribe_5_4 | tribe_3_7 | 0 | Not logged | 0.549434 | Direction artifact |
| tribe_5_4 | tribe_2_8 | 0 | Not logged | Not logged | Direction artifact |
| tribe_4_5 | tribe_3_6 | 0 | Not logged | Not logged | Direction artifact |
| tribe_4_5 | tribe_3_7 | 0 | Not logged | Not logged | Direction artifact |
| tribe_4_5 | tribe_2_8 | 0 | Not logged | Not logged | Direction artifact |
| tribe_3_6 | tribe_3_7 | 0 | Not logged | 0.474293 | Direction artifact |
| tribe_3_6 | tribe_2_8 | 0 | Not logged | Not logged | Direction artifact |
| tribe_3_7 | tribe_2_8 | 0 | Not logged | Not logged | Direction artifact |

## Files and provenance

Downloads: report.md, task_summary.csv, interference_matrix.csv, interference_pairs.csv, numerical_quality.csv, metrics.csv, report.json, sharpness_overlap_pairs.csv, sharpness_overlap_matrix.csv, tangent_overlap_pairs.csv, tangent_overlap_matrix.csv, overlap_directions.pt, overlap_directions.csv.

Source summary: /Users/gaspard/Documents/Sharpness_experiments/unzipped_results/results/wandb/wandb/run-20261005_151051-y9er89ad/files/wandb-summary.json

Source directions: /Users/gaspard/Documents/Sharpness_experiments/results/space_artifacts/y9er89ad/spaces_step_10000.pt
