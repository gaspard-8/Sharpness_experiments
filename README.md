# Sharpness experiments

`Voting_Function(function)` evaluates the wrapped function on the same input
and converts its score into a binary label. It uses a strict `> 0.5` decision
for 0/1 scores and `> 0` for signed scores, inferred from the function's
`negative_value`. Ties return the negative label, so signed outputs remain
-1/1 even when the score is zero. Output encoding defaults to the wrapped
function's encoding; `negative_value=0` or `-1` can override it without
changing the score threshold. Plain callables default to a threshold of 0.5;
use `threshold=0` for signed callable scores or another explicit boundary.
The wrapper reports binary accuracy with the same strict decision rule.

`WeightedVoting(n, weights)` uses exactly `n` finite weights on the first `n`
payload bits, ignoring extra bits. For 0/1 inputs it computes
`sum(a_i * (2*x_i - 1)) > 0`; equal positive weights reproduce strict majority.
For -1/1 inputs, set `input_negative_value=-1` to decide directly from
`sum(a_i*x_i) > 0`. Negative and zero weights are allowed. Output labels
default to 0/1; set `negative_value=-1` for -1/1. Zero scores return the
negative label.

```python
from src.functions import Isordered, Mean, Voting_Function, WeightedVoting

majority = Voting_Function(Mean())
ordered_vote = Voting_Function(Isordered(n=0, m=10))
weighted = WeightedVoting(3, [2.0, 1.0, 1.0])
signed_weighted = WeightedVoting(3, [2.0, 1.0, 1.0],
                                 input_negative_value=-1, negative_value=-1)
```

`test.py` wraps all eight selector tasks with `Voting_Function`, so every
task has binary targets and binary accuracy. Metric names gain a `voting_`
prefix, unless a custom wrapper `name` is supplied. A mean of entirely signed
functions uses the signed threshold; for means mixing output encodings,
choose the intended boundary with `threshold` explicitly.

`Majority_nm(n, m)` computes a strict majority over `x[..., n:m]`: indices are
zero-based, `n` is included, and `m` is excluded. Ties return the negative
label, and inputs must contain at least `m` bits.

`Tribe_ws(w, s)` uses the first `w * s` payload bits as `s` consecutive,
disjoint groups of width `w`. It returns `1` if at least one group contains
only ones, and `0` otherwise; extra bits are ignored. Both parameters must be
positive integers, and inputs must contain at least `w * s` bits.

```python
from src.functions import Tribe_ws

tribes = Tribe_ws(w=3, s=4)  # Four groups of three bits; needs 12 payload bits.
```

Set `negative_value=-1` for -1/1 labels. Accuracy uses a threshold of `0.5`
for 0/1 labels or `0` for -1/1 labels; predictions at the threshold count as
positive. Within a `SelectorFunction`, the selector prefix is excluded and
accuracy is logged as `accuracy/tribe_3_4` (or a custom `name`) and included
in overall accuracy.

`Isordered(n=0, m=None)` and `Isrepeating(n=0, m=None)` inspect the window
`x[..., n:m]`: zero-based `n` is included and `m` is excluded, just like
`Majority_nm`. Omitting `m` uses the input end; omitting both bounds preserves
full-payload behavior. Bounds must be integers, with `n >= 0` and a window
of at least two bits; an explicit `m` must not exceed the input length.
Only pairs fully inside the window count. Selector prefixes are removed
before applying these bounds. `test.py` explicitly uses `[0, 10)` for both
functions, covering the first 10 bits of its 21-bit payload.

`Isordered(n, m)` assigns a penalty of `2` to each overlapping `10` pair and `0`
to `00`, `01`, and `11`. It returns one minus the mean penalty:
`1 - 2 * count(10) / (L - 1)`, where `L` is the window length.
Fully ordered windows (no `10` pair), including constant
sequences, return `1`. For example, `0011` returns `1`, `1100` returns `1/3`,
and `010` returns `0`.

Alternating sequences approach zero with increasing length. The exact value
depends on the endpoints: `101010101010101010101010` has 12 unordered pairs
among 23 pairs, so it returns `-1/23`, approximately `-0.0435`. Outputs are
not clamped to zero. Accuracy rounds predictions to the nearest attainable
score `1 - 2*k/(L-1)`, where `k` is an integer from `0` to `floor(L/2)`.
Out-of-range predictions are clamped for accuracy, and exact midpoint ties
round the inferred count `k` to the nearest even integer. Within a
`SelectorFunction`, only window pairs are scored; accuracy is logged as
`accuracy/isordered_0_21` for the example below (or a custom `name`) and
included in overall accuracy. The default full-payload constructor retains
the metric name `isordered`.

```python
from src.functions import Isordered

ordered = Isordered(n=0, m=21)
```

`Isrepeating(n, m)` gives `1` to each overlapping `00` or `11` pair and `0` to
each `01` or `10` pair, then averages over the window's `L - 1` pairs.
Constant windows score `1`, alternating windows score
`0`, and `0011` scores `2/3`.

Accuracy rounds predictions to the nearest repeating-pair count, clamping
predictions to `[0, 1]` and rounding midpoint ties to the nearest even count.
Within a `SelectorFunction`, pairs exclude the selector prefix; accuracy is
logged as `accuracy/isrepeating_0_21` for the example below (or a custom
`name`) and included in overall accuracy. The default full-payload constructor
retains the metric name `isrepeating`. Both functions use the window length
for accuracy quantization, regardless of the number of bits outside it.

```python
from src.functions import Isrepeating

repeating = Isrepeating(n=0, m=21)
```

`MeanOfFunctions([...])` evaluates every supplied function on the same input
and returns the arithmetic mean of their outputs, preserving each function's
output encoding. It is one continuous-valued task and can be selected normally:

```python
from src.functions import Majority_nm, MeanOfFunctions, Parity, SelectorFunction

average = MeanOfFunctions([Majority_nm(0, 4), Majority_nm(4, 8)], name="average")
task = SelectorFunction([Parity(), average])
```

Here the selector uses one leading bit; both majority ranges refer to the
remaining payload. For binary child functions, the average logs both loss and
accuracy. Predictions are rounded to the nearest discrete mean level: for two
0/1 functions, these levels are `0`, `0.5`, and `1`. Predictions outside the
output range are clamped, and exact midpoint ties use ties-to-even rounding,
as with `Mean`. Signed and mixed 0/1 and -1/1 encodings are supported. Accuracy
is logged per selector task and included in the overall accuracy. If any child
has nonbinary outputs (such as `Mean`), only loss is reported for that average.

The experiment in `test.py` uses AdamW. Set `lr` and `weight_decay`
in `optimizer_config`; defaults are `1e-4` and `1e-2`, respectively.
These settings are recorded in the W&B run configuration.

The experiment also reserves a fixed random evaluation set: 320 unique inputs
per function (2,560 total for its eight tasks), at the configured `max_len` of
24. Every 10 optimizer updates it logs `eval/loss`, `eval/accuracy`, and
`eval/loss/<task>` / `eval/accuracy/<task>`. Existing `loss` and `accuracy`
metrics remain training-batch measurements. Evaluation uses the post-update
model with dropout disabled and `torch.no_grad()` (not inference mode), then
restores all model training modes. It does not change gradients or parameters.

Configure `eval_interval`, `eval_num_inputs_per_function`, `eval_batch_size`,
and `eval_seed` in `training_config`. Evaluation forwards are split into
batches of 320 to limit memory use. A separate seeded generator builds the
same set for repeated runs without consuming training randomness. Exact
held-out sequences, including their selectors, are rejected during training;
only their payloads are resampled, preserving task frequencies. The set is
balanced across valid tasks even if training uses unequal task probabilities,
so overall evaluation accuracy reflects that balanced mixture. With variable
training lengths, evaluation measures only `max_len`; other lengths are not
held out. The set must leave at least one possible input per task for training.

The reusable `train()` helper defaults to `eval_interval=None` (disabled);
`test.py` enables it explicitly. When enabled without W&B, the inputs are
still reserved but evaluation forwards are skipped. The independent
`evaluate_fixed_set()` helper can measure the set without W&B. The older
`evaluate()` helper still reports loss on freshly sampled inputs.

Training logs `gradient_norm/<task>` and
`gradient_alignment/<task_i>_vs_<task_j>` every `gradient_interval` completed
optimizer steps (default `20`). Each gradient is the gradient of that task's
mean loss with respect to all trainable parameters, including positional
embeddings. Norms are Euclidean; alignment is cosine similarity of the two
mean-loss gradients. Each distinct pair is logged once. Positive cosine
indicates locally compatible descent directions, negative cosine indicates
conflict, and zero indicates first-order orthogonality. Cosine is recorded as
NaN when either gradient norm is at most `1e-12`.

The measurement uses `gradient_num_inputs` uniform binary payloads (default
`256`), shared across selectors and fixed throughout each `train` call, at
total input length `max_len`. `gradient_seed` controls a separate probe RNG
(default `0`; the example sets it to the experiment seed), so these samples do
not consume training randomness. Each task is evaluated with its own selector
and labels, without weighting by training probabilities. Duplicate functions
remain distinct tasks, e.g. `first_0`, `first_1`, and `first_2`; unused selector
codes are excluded. The model runs in evaluation mode for these measurements,
and its training modes and existing gradients are preserved. The gradient
diagnostics report per-task norms and pairwise alignment, with no aggregate
alignment score.
These metrics describe raw loss gradients. AdamW's actual updates also depend
on its running gradient moments, adaptive scaling, and weight decay.

Training measures a **directed task-affinity matrix every 100 completed
optimizer steps**, at the post-update parameters. For source task `i`, let
`g_i` be its mean-loss gradient across all trainable parameters, including
positional embeddings. The hypothetical step is
`theta_i = theta - affinity_radius * g_i / ||g_i||`, so every valid source
moves the same global Euclidean parameter distance (`affinity_radius=0.02`).
Each matrix entry is `A[i,j] = L_j(theta) - L_j(theta_i)`:
**positive means help, negative means harm**, in raw mean-loss units. Rows
are source tasks and columns are target tasks. The diagonal measures the
effect on the source itself; both directions of each pair are recorded.
These are normalized raw-gradient steps, independent of AdamW's state.

W&B receives `affinity/matrix` as a labeled matrix table, and
`affinity/<source>_to_<target>` as scalar histories. Duplicate function names
get the same distinct suffixes as gradient metrics; unused selector codes
are excluded. Tasks are measured without training-probability weighting.
`affinity/baseline_loss/<task>`, `affinity/source_gradient_norm/<task>`,
`affinity/source_valid/<task>`, and `affinity/radius` provide context. If a
source gradient norm is nonfinite or at most `1e-12`, its normalized direction
is undefined: its entire matrix row is NaN and `source_valid` is zero.

Configure `affinity_interval` (default `100`, or `None` to disable),
`affinity_radius`, `affinity_num_inputs` (default `64` for each probe set),
`affinity_eval_batch_size` (default `256`), and `affinity_seed` in
`training_config`. Two independent uniform payload batches are sampled with
a private seeded generator, then fixed across checkpoints and shared across
selectors. One computes source gradients; the other measures target losses,
using identical target inputs before and after each step. These diagnostic
probes are not excluded from training, so the matrix measures probe-loss
transfer and does not certify held-out generalization.

Target forwards are batched across tasks to bound memory. At the eight-task
defaults, a measurement uses eight source forward/backward passes and 18
forward-only target batches of 256 examples, with no Hessian or space search.
`affinity/compute_seconds` logs the measurement's runtime. The measure uses
evaluation mode and functional parameter replacements; it preserves model
parameters, buffers, existing gradients, optimizer state and module modes,
and does not consume training randomness. It runs only when W&B logging is
enabled; `task_affinity_metrics()` also works independently of W&B. Smoke
tests measure it on each of their two steps with four inputs per probe set.

Training also logs `curvature/multi_task`, the multi-task curvature from
[PCGrad Definition 3](https://papers.neurips.cc/paper_files/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf):

\[
\mathcal H(L;\theta,\theta') = \int_0^1
g^\top \nabla^2 L\bigl(\theta+a(\theta'-\theta)\bigr)g\,da,
\qquad g=\nabla L(\theta),\quad L=\sum_i L_i.
\]

Here each `L_i` is a task's mean probe loss. The tasks are summed, not averaged
or weighted by training probabilities, matching the paper's convention.
Only valid selector tasks participate, with repeated functions kept separate.
`theta` and `theta'` are the parameters immediately before and after the actual
optimizer update at the logged step. The gradient `g` is evaluated before that
step and held fixed throughout the integral; it is not replaced by AdamW's
update direction. This uses the same fixed payloads as the gradient metrics,
controlled by `gradient_num_inputs` and `gradient_seed`.

`curvature_interval` defaults to `100`. `curvature_num_points` defaults to `3`
and controls Gauss-Legendre quadrature over the step segment; increasing it
improves the integral approximation at additional cost. The integrand uses
autograd Hessian-vector products, including all trainable parameters and
positional embeddings, without constructing the full Hessian. The measurement
disables dropout and temporarily uses math attention for second derivatives,
restoring the original training modes and attention backend settings afterward.
It does not change parameters, existing gradients, or optimizer state.

The reported value is signed and **not normalized by `||g||^2`**, as in
Definition 3. Its scale depends on gradient magnitude and the number of tasks;
it is zero when the total probe gradient is zero, even if the loss surface is
curved. It measures total-loss curvature along the observed step, not a
per-task forgetting score. Gradient norms, alignment, and sharpness describe
the post-update model; curvature describes the transition into that model.
Training-batch loss and accuracy use the pre-update predictions.

## Task-loss tangent and sharpness directions

These expensive diagnostics are **disabled by default**, including in cluster
runs and smoke tests. To enable them, run `python test.py --loss-spaces`;
`--no-loss-spaces` explicitly disables them. The setting is recorded in W&B as
`train_space_enabled`. It controls the final tangent, sharpness, and pairwise
overlap computations and their direction artifact.

When enabled, the experiment measures these directions **once, after the final
optimizer update**. In `train()`, set `space_enabled=True` and `space_delta` to a
positive global parameter-vector norm (`0.02` in `test.py`).
`space_epsilon` is the allowed increase in a task's mean loss (`0.002`, reduced
from `0.01` by a factor of five to let more directions qualify as sharp).
The mean loss uses the configured criterion on `space_num_inputs` fixed uniform
payloads per task (`64` in `test.py`), with a private `space_seed`. All trainable
parameters, including positional embeddings, participate. The measure preserves
model parameters, existing gradients, optimizer state, module modes and training RNG.

For a unit parameter direction `v`, define its harm to task `i` as
`max(L_i(theta + delta*v), L_i(theta - delta*v)) - L_i(theta)`.
Both perturbed losses are evaluated on the actual model. A tangent direction
has harm at most `space_epsilon`; a sharpness direction has harm above it.
Loss decreases are allowed. The check uses both signs so changing the sign of
a reported direction does not change its classification.

The implementation uses **matrix-free thick-restart block Lanczos**.
`space_hvp_budget` is **10,000 Hessian-vector products
per task**, shared by the tangent and sharpness calculations. For the eight
tasks in `test.py`, this allows 80,000 HVPs in total for this final measure.
If the complete parameter basis fits in the active window, the search stops
once it has been covered. An enabled normal experiment uses 10,000;
`--smoke-test --loss-spaces` uses 4. Other periodic curvature diagnostics have
their own existing HVP cost.

Each HVP differentiates the task's mean loss at the final model parameters.
The gradient graph is reused within a task. Lanczos starts with task gradients
and independent full-parameter random vectors, then repeatedly applies the
Hessian to grow an adaptive Krylov space. The active window is bounded by
`min(parameter_count, hvp_budget, max(512, 4 * block_size + task_count))`, where
`block_size = max(direction_cap, task_count)`. At each thick restart, the lowest
and highest `block_size` Ritz modes and independent task-gradient components
are retained. Their cached Hessian images are transformed with them, and their
residuals seed the next expansion. Each new HVP therefore improves the existing
estimates. Two-pass reorthogonalization and random starts at breakdown handle
numerical error and repeated eigenvalues. This follows the
[thick-restart Lanczos approach](https://doi.org/10.1137/S0895479898334605).
`space_direction_seed` controls a private RNG. A small projected Hessian is
diagonalized; the full parameter-by-parameter Hessian is never constructed.
The returned-direction cap (`space_num_directions=100`) is independent of the
HVP budget. In the normal experiment, up to 512 Krylov vectors and their Hessian
images are stored on the solver device, costing approximately 417 MB at 101,761 float32
parameters, plus model, differentiation, retained directions and temporary
workspace. On CUDA, this storage and the task gradients, orthogonalization,
restarts, projected eigensystems, direction checks and interference algebra
stay on the GPU. Hessian-vector products do not transfer full parameter vectors
between CPU and GPU. Finished artifact tensors are copied to CPU for portable
serialization; logged scalars and adaptive stopping checks still synchronize
with the host. CPU and MPS models use CPU solver storage because the solver
requires float64 algebra, which MPS does not support. The 10,000 products are
spread across restart cycles; they do not require storing 10,000 vectors.
Tasks reuse the solver storage. The artifact records `solver_device`.

The private direction generator lives on the solver device and preserves the
training RNG. A fixed seed is repeatable on the same backend; CPU and CUDA
random starts need not match. Gradient norms, pairwise alignments, and periodic
curvature reductions also stay on CUDA, exporting only their final metrics.

For small perturbations the both-sign harm is approximated by
`delta * abs(g_i.T @ v) + delta**2 / 2 * (v.T @ H_i @ v)`.
Sharpness uses a sphere-constrained quadratic maximization in the Krylov space,
including the gradient term. Each accepted vector is removed orthogonally
before finding the next. When the gradient is zero, this selects the largest
Hessian Ritz eigenvalues in descending order.

Tangent candidates are the lowest-curvature eigenvectors in the portion of
the Krylov space **orthogonal to the task loss gradient**, in ascending order.
This enforces first-order loss tangency. At zero gradient the entire Krylov
space is available. Negative curvature is allowed because loss decreases are
allowed; the search uses the smallest eigenvalues, not the smallest absolute
eigenvalues. Tangent and sharpness bases are independently orthonormal.

Every candidate is validated with both actual perturbed losses. Each search
stops at its first rejecting candidate, its direction cap, or exhaustion of
its available Krylov directions. The eigenvalue/quadratic calculation proposes
directions; it does not find exact minima or maxima of the nonlinear loss at
finite delta. A failed candidate does not certify that no other safe or harmful
direction exists, and 10,000 products do not guarantee eigenpair convergence.

W&B logs `spaces/search_dim` and `spaces/parameter_count` (both the number of
trainable parameters), `spaces/direction_cap`, and per-task
`spaces/tangent_dim/<task>` / `spaces/sharpness_dim/<task>`. The dimensions count
the independent accepted directions found, each capped at 100 by default.
`spaces/tangent_cap_reached/<task>` / `spaces/sharpness_cap_reached/<task>` are
1 when the relevant cap is reached; additional directions may exist. These
counts do not certify the full spaces' dimensions. The searches are independent,
so their counts need not add to the parameter count.
`spaces/min_harm/<task>` and `spaces/max_harm/<task>` report the first search
candidate's actual harm. When a search stops on a failed loss check,
`spaces/tangent_stop_harm/<task>` or `spaces/sharpness_stop_harm/<task>` records
the rejecting direction's harm. `spaces/<kind>_candidate_exhausted/<task>`
records exhaustion of the available Krylov directions before reaching the cap.
W&B also logs `spaces/hvp_budget`, `spaces/hvp_count/<task>`,
`spaces/krylov_dim/<task>`, and `spaces/hvp_budget_exhausted/<task>`.
The HVP count is the total number of operator calls across all cycles; the
Krylov dimension is the final active basis size. `spaces/krylov_max_dim` records
the storage bound, `spaces/lanczos_restarts/<task>` counts thick restart cycles,
and `spaces/full_space_covered/<task>` is 1 only when a complete parameter basis
was processed. These measures distinguish computation from active storage.
`spaces/ritz_max_relative_residual/<task>` checks the first/last up to 100
Hessian Ritz modes using cached Hessian images without extra HVPs. A small
residual measures the eigen-equation error; it does not certify that the
estimated eigenvalue is globally smallest or largest. Per-kind maximum
residuals describe tangent projected eigenvectors and sharpness quadratic
stationarity, respectively.

For each ordered pair `source_to_target`, interference is the approximate
intersection of source sharpness and target tangent directions. Principal-angle
cosines at least `space_intersection_cosine` (`0.999`) propose directions;
actual finite-radius loss checks retain only those that hurt the source above
epsilon and leave the target within epsilon in both signs.
`spaces/interference_dim/<source>_to_<target>` counts these verified directions
in the discovered spans. A zero count does not rule out interference involving
directions not discovered. Both task orders are reported.

For each unordered pair `A_and_B`, same-kind overlaps also compare **sharpness
with sharpness** and **tangent with tangent**. Orthonormalized bases and an SVD
give the principal-angle cosines; values at least `space_intersection_cosine`
count as approximate shared directions. Each exported unit direction is the
normalized midpoint of the two aligned principal vectors, equally close to
both spans. At cosine 1 this is an exact common direction. These calculations
use no additional HVPs and run once, at the end of training.

W&B logs `spaces/<kind>_overlap_dim/<A>_and_<B>` for this geometric dimension,
`spaces/<kind>_overlap_max_cosine/<A>_and_<B>` for the closest alignment when
both spans are nonempty, and `spaces/<kind>_overlap_verified_dim/<A>_and_<B>`
for the shared directions passing actual finite-radius loss checks on both
tasks. Tangent checks require both signs to stay within epsilon for both tasks;
sharpness checks require worst-sign harm above epsilon for each task. The
harmful sign can differ between tasks. Since rotating a basis can change
nonlinear loss behavior, geometric and loss-verified counts are kept separate.

The final W&B `loss-spaces` artifact, aliased by its training step, contains
parameter names and shapes, full task and interference direction vectors,
baselines, loss changes for both signs, task gradients, probe payloads, cap
flags, HVP counts, Ritz eigenvalues and residual diagnostics. Each direction
matrix has one row per trainable
parameter coordinate and one column per found unit direction. Multiply a
column by delta, then split and reshape in `parameter_names` / `parameter_shapes`
order to obtain the parameter perturbation. Artifact formats 4 and 5 record
full vectors directly; no reduced search basis is required. The stored
loss-change columns correspond to `+delta` and `-delta` in that order.
`<kind>_curvatures` stores `v.T @ H_i @ v`; `<kind>_predicted_harms` stores the
second-order approximation. Tangent residuals use the gradient-projected
Hessian; sharpness residuals use the quadratic objective projected off earlier
sharpness directions. Individual directions are checked, but nonlinear loss can behave
differently along combinations of them.
Format 5 also stores `overlaps["sharpness"]` and `overlaps["tangent"]`, with one
entry per unordered pair. Entries contain the task names, shared direction
matrix, retained cosines, all principal cosines, `loss_changes_a` /
`loss_changes_b` (columns +delta then -delta), and a per-direction
`loss_verified` boolean mask. `directions` contains the geometric basis;
select its columns using that mask to get the directions passing joint loss
checks. Training artifacts include their W&B run ID.

### Readable reports from a saved run

Generate a task summary, a pairwise interference matrix, and numerical-quality
tables from the local W&B summaries:

```bash
env/bin/python scripts/summarize_spaces.py unzipped_results \
  --output-dir results/spaces_report
```

Pass either an extracted results directory or a specific `wandb-summary.json`.
Directories are searched recursively, with one report per run. The script
reads the saved `config.yaml` using the project's existing environment and
preserves the original measurements. No training or W&B connection is needed.

Reports also include symmetric sharpness–sharpness and tangent–tangent matrices,
pair details, and maximum alignment cosines. Older runs need their full
`loss-spaces` artifact to calculate these overlaps. Put its `spaces_step_*.pt`
file in `results/space_artifacts/<run-id>/` and rerun the same command, or select
it explicitly for one run:

```bash
env/bin/python scripts/summarize_spaces.py path/to/wandb-summary.json \
  --directions-artifact path/to/spaces_step_10000.pt
```

The artifacts downloaded for the existing `y9er89ad` and `wgrcwba5` reports are
cached under `results/space_artifacts/` (excluded from Git). Backfilling uses only
saved parameter directions; it does not rerun Hessian computations. New joint
loss checks cannot be reconstructed without the trained model and remain
missing for older artifacts.

Open `results/spaces_report/index.html`, or a run's `report.html` / `report.md`.
Each run also gets `task_summary.csv`, `interference_matrix.csv`,
`interference_pairs.csv`, `numerical_quality.csv`, and `metrics.csv`.
The last CSV preserves every original `spaces/` value with a group and an
explanation. `report.json` retains the grouped data with full numeric precision.
Same-kind overlaps also get `<kind>_overlap_pairs.csv` and
`<kind>_overlap_matrix.csv`. When an artifact is supplied, `overlap_directions.pt`
exports the actual vectors and parameter layout; `overlap_directions.csv`
maps direction IDs to `overlaps[vector_kind][vector_pair]["directions"][:, vector_column]`.
Columns are zero-based. An empty direction table means no principal cosine
passed the threshold. Original `metrics.csv` values are preserved exactly;
backfilled counts live in the separate overlap tables.

The interference matrix uses **safe tasks as rows and harmed tasks as columns**:
row A / column B reads the original `B_to_A` metric. Capped counts are marked
explicitly. Missing measurements stay missing, and zero overlap is interpreted
only within the selected direction bases.

Training logs Hahn–Rofin average direction sharpness every 100 completed
optimizer steps. The estimate is the mean squared change in model predictions
after adding independent zero-mean Gaussian noise with standard deviation
`sharpness_rho` (default `0.02`) to each non-positional model parameter.
Gaussian noise here replaces the fixed-radius sphere used in the paper.

`sharpness_num_inputs` (default `32`) controls the number of uniformly sampled
payloads per selector code; `sharpness_num_perturbations` (default `4`)
controls the number of noise draws. Sharpness uses the fixed input length
`max_len`. Each function's value conditions on its selector code, while the
overall value averages equally over all selector codes. Unused codes are
included in `sharpness/invalid_selectors`, so the overall value represents
uniform sampling over the entire binary input space. Training selector
probabilities do not affect this measurement.

## Run on the HTCondor cluster

The runtime dependencies are `torch==2.8.0`, `numpy==2.0.2`, and
`wandb==0.26.1`, matching the installed local environment. Tests use Python's
built-in `unittest`. Inputs are generated by the experiment: there is no dataset
to download. The Dockerfile uses Python 3.11 and the official PyTorch CUDA 12.6
wheels. The older `TestCluster` NGC image is too old for the `torch.func` and
`torch.nn.attention` APIs used by this experiment.

`test.py` prefers CUDA, then Mac MPS, then CPU. Set `SHARPNESS_DEVICE` to select
a device explicitly. The cluster runner requires CUDA and fails if it is
unavailable, so an allocated GPU will not silently go unused.

### Build and publish the dependency image

Run from this project directory on a machine with Docker and registry access:

```bash
cd /Users/gaspard/Documents/Sharpness_experiments
image=docker.coli.uni-saarland.de/gtomas/sharpness-experiments:v1
docker login docker.coli.uni-saarland.de
docker buildx build --platform linux/amd64 --load -t "$image" .
docker run --rm --platform linux/amd64 "$image" python -c \
  'import torch, numpy, wandb; from torch.func import functional_call; from torch.nn.attention import SDPBackend, sdpa_kernel; print(torch.__version__, torch.version.cuda, numpy.__version__, wandb.__version__)'
docker push "$image"
```

The Linux AMD64 target is for the cluster, including when building on an Apple
Silicon Mac. The import check inside the build does not require a GPU. CUDA
execution must be tested on a cluster worker. Check with the cluster operator
that its NVIDIA driver supports CUDA 12.6; NVIDIA documents driver branch 525
or newer as the CUDA 12.x minor-compatibility baseline, with limitations for
some features. The cluster's NVIDIA container runtime must expose the allocated
GPU and driver libraries. See
[PyTorch's versioned installation commands](https://pytorch.org/get-started/previous-versions/),
[Docker platform builds](https://docs.docker.com/build/building/multi-platform/), and
[NVIDIA compatibility requirements](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).

The image contains dependencies. `test.py` and `src/` are transferred separately
by HTCondor, so the image contains neither the experiment source nor W&B keys,
the local virtual environment, or earlier run data. `.dockerignore` permits
only the Dockerfile and dependency manifest into the build context.

### Copy the project to the submission host

These paths follow the `gtomas` setup in `TestCluster`. Replace the submission
host below; change `test.sub` and `submit-wandb.sh` if using another account or
project location.

```bash
export CLUSTER_LOGIN=gtomas@YOUR_SUBMISSION_HOST
ssh "$CLUSTER_LOGIN" 'mkdir -p /nethome/gtomas/projects/sharpness_experiments'
rsync -av dockerfile .dockerignore requirements.txt test.py src tests \
  run-test.sh test.sub submit-wandb.sh README.md \
  "$CLUSTER_LOGIN:/nethome/gtomas/projects/sharpness_experiments/"
ssh "$CLUSTER_LOGIN"
cd /nethome/gtomas/projects/sharpness_experiments
```

This copies the current working files, including changes that are not committed.
Keep those files in place while jobs are queued; changing a file before a worker
receives it can change what that job runs. Each finished job's archive contains
a copy of the Python sources it executed.

### W&B credentials and destination

Reuse `$HOME/.config/wandb/cluster.env` on the submission host if it already
exists from `TestCluster`. Otherwise create it once, in Bash:

```bash
(
  set +x
  umask 077
  mkdir -p "$HOME/.config/wandb"
  chmod 700 "$HOME/.config/wandb"
  read -rsp 'W&B API key: ' wandb_key
  printf '\n'
  printf 'export WANDB_API_KEY=%q\n' "$wandb_key" \
    > "$HOME/.config/wandb/cluster.env"
  chmod 600 "$HOME/.config/wandb/cluster.env"
)
```

The default project is `multiple tasks sharpness` and the default entity is
`gaspardtomas-universit-t-des-saarlandes-saarland-university`. Settings already
present in `cluster.env` are respected. An explicit `WANDB_PROJECT` or
`WANDB_ENTITY` passed to the wrapper takes precedence, as shown below. Confirm
that the key has permission to write runs and artifacts to that destination.
Workers need outbound HTTPS access to W&B, including artifact storage.

The key stays out of the image and transferred input files. Only the three
required W&B environment variables are forwarded. HTCondor puts forwarded
environment variables in job metadata, so full job dumps can contain the key;
keep them private. The runner sets W&B cache, configuration, staging, and log
paths before import so it can run with an unwritable container home directory.

### Submit a short GPU check, then the full experiment

On the submission host:

```bash
WANDB_PROJECT='multiple tasks sharpness' bash submit-wandb.sh \
  'experiment_args=--smoke-test'
```

The wrapper creates log/result directories and loads credentials. The smoke
test performs two optimizer updates, evaluation, gradient, curvature, and
sharpness diagnostics. Add `--loss-spaces` to also check a small final loss-space
search and W&B artifact:

```bash
WANDB_PROJECT='multiple tasks sharpness' bash submit-wandb.sh \
  'experiment_args=--smoke-test --loss-spaces'
```

It retains the full model and eight tasks but reduces sample counts and search
settings. Its W&B config records `smoke_test=true`; its measurements are only
for validating the pipeline.

Use the cluster ID printed by submission:

```bash
condor_q JOB_ID -nobatch
cat /scratch/gtomas/logs/sharpness_experiments/test.JOB_ID.0.out
cat /scratch/gtomas/logs/sharpness_experiments/test.JOB_ID.0.err
cat /scratch/gtomas/logs/sharpness_experiments/test.JOB_ID.log
```

Verify the CUDA device and GPU name in stdout, the W&B run URL, exit code 0,
and the metrics in W&B (also the `loss-spaces` artifact if enabled). W&B emits normal status
messages to stderr, so a nonempty `.err` file is not itself a failure. If held,
inspect `condor_q JOB_ID -hold`. If idle for a long time, use
`condor_q JOB_ID -better-analyze` to inspect resource matching.

Once that succeeds, submit the full 10,000-step experiment:

```bash
WANDB_PROJECT='multiple tasks sharpness' bash submit-wandb.sh
```

`test.sub` starts one job with one GPU, two CPUs, 8 GB of host RAM, and 4 GB of
job disk. Host RAM does not reserve GPU memory. These are starting allocations;
adjust after observing the smoke/full run. To enable the final direction searches
in a full run, pass `'experiment_args=--loss-spaces'` to the submission wrapper.
These searches can take substantial time after training. Check cluster runtime limits before the
full run. The default experiment has no model checkpoints or resume mechanism;
an evicted job starts training again.

The runner sets numerical-library thread limits from `request_cpus`, including
OpenBLAS and PyTorch's independent interop pool. Startup prints the effective
thread counts. During the final loss-space measurement, stdout and
`results/diagnostic-progress.jsonl` record task/stage, HVP counts every 500
products, restart counts, Python-process RSS/CPU usage, and CUDA allocated,
reserved and peak memory. These records distinguish a slow computation from a
failure in eigendecomposition, direction checks, serialization or W&B shutdown.
The process memory fields describe Python, not every process in the Condor job.
Native Python fault tracebacks are enabled; Python exceptions are printed before
attempting W&B shutdown. An uncatchable kill can still prevent these tracebacks.

### Diagnosing an apparent crash or runtime limit

A W&B `crashed` status means its server stopped receiving heartbeats; it does
not establish that Condor killed Python. The SDK sends heartbeats independently
of metric logging, so a long measurement alone should not cause this state
with a healthy service and connection. See the
[W&B run states](https://docs.wandb.ai/ref/python/experiments/run/).
Check the Condor job state and reason on the cluster:

```bash
condor_q JOB_ID -long -attributes JobStatus,NumJobStarts,NumVacates,NumVacatesByReason,HoldReason,HoldReasonCode,HoldReasonSubCode,AllowedExecuteDuration,AllowedJobDuration,MaxRuntime
condor_history JOB_ID -limit 1 -long -attributes NumJobStarts,NumVacates,NumVacatesByReason,HoldReason,HoldReasonCode,HoldReasonSubCode,RemoveReason,ExitCode,ExitBySignal,ExitSignal,AllowedExecuteDuration,AllowedJobDuration,MaxRuntime
```

Use `condor_q` while the job is still queued, and history after it leaves the
queue. Manual removal records your removal and may obscure earlier evidence;
also retain the complete event log and W&B `logs/debug-internal.log`.
Running (R) can follow an earlier eviction and restart. `NumJobStarts` greater
than 1 and the vacate counters/event log help identify that case; the current
state alone does not rule out preemption. See the
[Condor job attributes](https://htcondor.readthedocs.io/en/25.0/classad-attributes/job-classad-attributes.html#NumJobStarts).
The submit file and runner impose no execution deadline. Standard Condor
`allowed_execute_duration` and `allowed_job_duration` limit the job, not a
single Hessian operation; their hold codes are 47 and 46 respectively.
See the [Condor policy commands](https://htcondor.readthedocs.io/en/25.0/man-pages/condor_submit.html#allowed_execute_duration)
and [hold codes](https://htcondor.readthedocs.io/en/lts/codes-other-values/hold-reason-codes.html).
If the job ad confirms an overridable execution-duration limit, an explicit
24-hour ceiling for a new submission can be supplied with:

```bash
bash submit-wandb.sh -append 'allowed_execute_duration = 86400'
```

This sets the job's execution-duration attribute; it does not bypass a site's
administrative ceiling, preemption policy, or site-specific runtime attribute.
Verify the effective job ad after submission. No 300-second operation timer
exists in the experiment code, and no such timer is inferred from W&B's state.

### Results and image updates

Metrics and the final loss-space direction artifact are uploaded to W&B.
`run.finish()` waits for pending uploads. The runner also archives local W&B
run files and artifact staging data, executed sources, a timestamp/hostname,
and the exit code, including
when Python exits with an error. HTCondor returns the archive to:

```text
/scratch/gtomas/sharpness_experiments/results/JOB_ID.0.tar.gz
```

To inspect it after completion:

```bash
mkdir -p /scratch/gtomas/sharpness_experiments/inspect/JOB_ID.0
tar -xzf /scratch/gtomas/sharpness_experiments/results/JOB_ID.0.tar.gz \
  -C /scratch/gtomas/sharpness_experiments/inspect/JOB_ID.0
```

The archive contains local tracking files, not a trained-model checkpoint.
An interrupted worker, eviction, or forced kill may prevent archiving or final
artifact upload; `ON_EXIT` transfers outputs when the process exits normally,
including a nonzero Python exit. See
[HTCondor submission and transfer options](https://htcondor.readthedocs.io/en/24.0/man-pages/condor_submit.html).

For Python or experiment-setting changes, edit `test.py`/`src/`, copy the changed
files to the submission host, and submit a new job. No Docker rebuild is needed.

For dependency or system-library changes, edit `requirements.txt`/`Dockerfile`,
build and push a new version tag, and use that tag for new submissions:

```bash
# On the Docker build machine, from this project directory:
image=docker.coli.uni-saarland.de/gtomas/sharpness-experiments:v2
docker buildx build --platform linux/amd64 --load -t "$image" .
docker push "$image"

# On the submission host:
WANDB_PROJECT='multiple tasks sharpness' bash submit-wandb.sh \
  'image=docker.coli.uni-saarland.de/gtomas/sharpness-experiments:v2' \
  'experiment_args=--smoke-test'
```

If changing the PyTorch version, update both its explicit CUDA installation in
the Dockerfile and its pin in `requirements.txt`, plus the CUDA assertion if
changing the runtime. Verify the worker driver supports the chosen runtime.
Use a fresh tag rather than overwriting an old one: workers may cache images.
After the new smoke test passes, use that tag for the full submission or update
the default `image` in `test.sub`. Editing the Dockerfile, pushing a tag, or
editing a submit file does not modify jobs already submitted.

### Build error: no matching PyTorch distribution

If pip lists `2.6.0+cu126` and `2.9.0+cu126` but omits `2.8.0`, the build is
likely targeting ARM64. The CUDA 12.6 PyTorch 2.8.0 wheels support Linux AMD64,
not ARM64. On an Apple Silicon Mac, a build without `--platform linux/amd64`
defaults to ARM64. Keep the pinned version and rebuild for the cluster:

```bash
docker buildx build --platform linux/amd64 --load \
  -t docker.coli.uni-saarland.de/gtomas/sharpness-experiments:v1 .
```

The Dockerfile checks the target architecture before installing packages and
prints this instruction when the wrong target is selected. See the
[official CUDA wheel index](https://download.pytorch.org/whl/cu126/torch/).
