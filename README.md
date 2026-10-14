# Sharpness experiments

## Input-dependence tower sweep

`sweep_input_tower.py` prepares **three independent runs**, one function per
model, with the same one-layer transformer, width 256, 8 heads, batch size 64,
AdamW settings, and 30,000 updates as the Fourier experiment. The new targets
use **32 input bits and degrees 1–8**, starting from an **8-input core** and
adding one relevant input at each subsequent stage up to 32 inputs.

All three targets have zero mean and squared L2 norm one. The input-stage
mass is identical across targets: stage `r=1` is the whole core, `r=2` adds
input 9, and so on; `v_r = r^(-2) / sum(q=1..25, q^(-2))`. The core carries
62.3% of the variance, adding input 9 contributes 15.6%, and adding input 10
contributes 6.92%. Thus the first 10 inputs recover 84.8% of each target's
mass, independent of its degree profile. Further contributions strictly
decrease. This controls **approximate dependence on a small set of inputs**;
the complete default targets still depend on all 32 inputs.

| Function | Global Fourier mass by degree |
| --- | --- |
| `input_tower_increasing` | `w_d = d^2 / sum(q=1..8, q^2)` |
| `input_tower_random` | Reproducible positive random weights, normalized to sum to one |
| `input_tower_top_heavy` | 90% of mass on degrees 7 and 8; 10% shared over degrees 1–6 |

For each input stage and degree, the construction chooses up to 16 distinct
subsets and divides that stage/degree's mass equally across them, with random
signs. The core uses subsets of its first eight inputs. Later stages use only
available inputs and require every term to contain the newly introduced input.
Coefficient magnitudes are `sqrt(v_r * w_d / number_of_terms_at_stage_and_degree)`.
Supports and signs are **shared across all three profiles**, as are model
initialization and held-out inputs. Only the degree weights vary.

The core is needed to separate the two controls. If decreasing input-stage
mass starts with a single input, degree-1 mass must be at least that first
stage's mass, whereas degree-32 mass cannot exceed the last stage's mass.
Consequently the full degree spectrum cannot be strictly increasing. Starting
with a core at least as large as the maximum degree permits independent global
degree weights at every stage. The matched input-tail guarantee applies from
the core onward; approximations within the core need not have matched errors.

Every step, including step zero, logs both:

- `eval/loss_degree_1` through `eval/loss_degree_8`: MSE against cumulative
  Fourier approximations through each degree.
- `eval/loss_input_1` through `eval/loss_input_32`: MSE against
  `E[f(X) | the first k positions in input_order]`. Under uniform independent
  bits, this retains exactly the Fourier terms fully contained in those inputs.

All targets are precomputed on the same held-out set, so both axes use one
bank of model predictions. `eval/loss_degree_8`, `eval/loss_input_32`, and
`eval/loss` are identical. Training always receives the full function output.
These losses compare a full model output with a truncated target; they do not
project the model itself. Lower-cutoff losses can rise as other parts are learned.

The input projections use a specified nested ordering; they are not a search
for the best possible k-input subset. `target.json` records that ordering,
all coefficients, actual degree mass, input-stage mass, prefix tail errors,
and individual input influences. `evaluation.pt` stores both target banks.
Results use new directories under `results/input_tower/`, or the cluster's
`SHARPNESS_RESULTS_DIR`. The original four-run Fourier sweep remains available.

```bash
env/bin/python sweep_input_tower.py --list
env/bin/python sweep_input_tower.py --run-index 0 --smoke-test --no-wandb --device cpu
# On the submission host, queue the three full runs:
bash submit-input-tower-sweep.sh
```

`--input-order permuted` renames the input positions with a reproducible
permutation while retaining the same structure and coefficients in rank order.
This tests physical-position preferences. Change `--tower-core-size`,
`--max-degree`, `--input-power`, `--input-spectrum`, `--input-decay`,
`--terms-per-stage-degree`, `--degree-power`, `--top-mass`, or `--top-band`
to customize the family; all three runs receive the same options.

A one-input core is also supported with `--tower-core-size 1 --max-degree 8`.
In that mode, each stage renormalizes its requested degree weights over the
degrees it can actually represent. The global degree profile is consequently
different from the requested weights, and metadata explicitly identifies the
profile as conditional and saves the actual spectrum.

For independent global spectra through degree 32 with a nontrivial input
tower, use `--input-len 64 --tower-core-size 32 --max-degree 32`.

```bash
bash submit-input-tower-sweep.sh -append 'experiment_args=--input-order permuted'
# Plot input cutoffs 8–14 from one extracted run:
env/bin/python scripts/plot_fourier_loss.py --input unzipped_results --run input_tower_increasing \
  --axis input --min-cutoff 8 --max-inputs 14 --max-step 15000
# Plot degree cutoffs from the same run:
env/bin/python scripts/plot_fourier_loss.py --input unzipped_results --run input_tower_increasing \
  --axis degree --max-degree 8 --max-step 15000
```

The input-tower launcher defaults to the W&B project `input tower Fourier
sharpness`; existing project/entity overrides and credentials are respected.
Source transfers and result archives use the existing cluster runner.
Plot filenames distinguish input/degree axes and preserve earlier exports.

## Single-task Fourier sweep

The separate experiment in `test_fourier.py` trains a **one-layer transformer**
on **one real-valued Fourier function per model**, with identical model settings
for all four targets. Inputs are independent uniform
0/1 bits, with no task selector. The default input length and highest degree
are both **32**; input length must exceed 24 even in smoke tests. Training uses
the existing token/position embedding concatenation, scalar regression head,
MSE, and AdamW (`lr=1e-4`, `weight_decay=1e-2`). The model uses the largest
embedding size and head count from the arithmetic sweep: token and position
embeddings of size 128 each (`d_model=256`), 8 heads, and one layer. Defaults are 30,000 updates,
batch size 64, and 1,024 fixed unique held-out inputs excluded from training.

The Boolean-cube Fourier convention is
`chi_S(x) = product(i in S, 1 - 2*x_i)`, with
`f(x) = sum(S, a_S * chi_S(x))`. There is no constant term. All four functions
have mean zero and squared L2 norm one under the uniform input distribution.
**Mass at degree d** is `w_d = sum(|S|=d, a_S^2)`, with strictly decreasing,
positive mass at every degree from 1 through the configured `max_degree=K`:

| Function | Subsets at each degree | Degree mass before normalization |
| --- | --- | --- |
| `concentrated_powerlaw` | One nested subset `{0, ..., d-1}` | `d^(-2)` |
| `concentrated_geometric` | One nested subset `{0, ..., d-1}` | `0.9^(d-1)` |
| `sparse_powerlaw` | Up to 16 randomly selected distinct subsets | `d^(-2)` |
| `sparse_geometric` | Up to 16 randomly selected distinct subsets | `0.9^(d-1)` |

Sparse targets split each degree's mass equally over their subsets, using
random coefficient signs. They use `min(16, binomial(input_len, d))` terms;
at degree 32 there is necessarily just one subset. Within each spectrum,
concentrated and sparse targets have exactly the same per-degree mass.
For concentrated targets, coefficient magnitudes are proportional to `1/d`
(power-law) or `0.9^((d-1)/2)` (geometric). For sparse targets, each selected
coefficient has magnitude `sqrt(w_d / terms_at_degree_d)`. Strict decrease is
guaranteed for the **total squared mass per degree**. Individual sparse
coefficient magnitudes can increase when fewer subsets are available, as at
degree 32, even though the total mass still decreases.
Sparse support/signs use `--function-seed 17`, independently of `--seed 7`
for model initialization/training and `--eval-seed 23` for evaluation inputs.
These defaults share initialization and evaluation inputs across the four runs
and keep each target's coefficients reproducible.
Exact coefficients and all seeds are saved with each run.

The cumulative approximation is `f_{<=k} = sum(|S|<=k, a_S*chi_S)`, retaining
the original coefficients without renormalizing. Its squared L2 error is
`epsilon_k = sum(d>k, w_d)`. Thus every cutoff k has its explicit epsilon;
this is concentration **up to degree k**, rather than all mass at exactly k.
`FourierFunction.approximation_degree(epsilon)` returns the smallest suitable
cutoff. `target.json` records the mass and tail above every cutoff, including
degree zero. `--power`, `--geometric-decay`, `--sparse-terms`, `--max-degree`,
and `--input-len` customize this construction. A degree cap below input length
uses zero coefficients above the cap.

**Every completed optimizer update**, evaluation logs
`eval/loss_degree_1`, ..., `eval/loss_degree_K`, defined as
`mean((model(x) - f_{<=k}(x))^2)`. `eval/loss` equals the degree-K loss.
Step zero also records the untrained baseline. Only the full function supplies
training labels; the approximations are evaluation targets. All target
approximations are precomputed once, and one set of batched model predictions
supplies all K losses. This does not estimate the model's Fourier coefficients.
When the model learns the full target perfectly, lower-degree losses approach
their omitted tail masses in expectation, so they need not approach zero or
decrease monotonically with degree during training.

Evaluation runs with dropout disabled and no gradients, then restores model
modes. Results work with or without W&B: each run saves `config.json`,
`target.json`, `evaluation.pt`, `metrics.jsonl`, and `checkpoint.pt` in a new
directory under `results/fourier/` (or under `SHARPNESS_RESULTS_DIR` on the
cluster). Output sharpness retains the Gaussian parameter-noise diagnostic
every 100 updates, excluding positional embeddings. It uses independent
randomness; `--sharpness-interval 0` disables it.

Inspect the sweep or check one configuration locally:

```bash
env/bin/python sweep_fourier.py --list
env/bin/python sweep_fourier.py --run-index 0 --smoke-test --no-wandb --device cpu
env/bin/python test_fourier.py --function sparse_powerlaw --max-degree 16 --no-wandb
```

`sweep_fourier.py` contains **four independent runs**, one per target. All use
embedding size 128 per token/position embedding (`d_model=256`), 8 attention
heads, batch size 64, and one transformer layer. Only the target function
varies. Smoke tests shorten
training to two updates and reduce evaluation/probe counts, preserving the
chosen input length, degree range, architecture, and sweep batch size.

Copy the new launchers together with `src/`, `run-test.sh`, and the existing
`submit-wandb.sh` to the submission host. Using its existing dependency image
and credentials, queue the sweep from the project directory with:

```bash
bash submit-fourier-sweep.sh
# Override settings shared by all four jobs:
bash submit-fourier-sweep.sh -append 'experiment_args=--steps 10000 --max-degree 16'
```

The default W&B project is `single task Fourier sharpness`; `WANDB_PROJECT`
and `WANDB_ENTITY` overrides are honored. Jobs share a W&B group based on the
Condor cluster ID and archive local metrics, checkpoints, target coefficients,
and executed sources in the existing `results.tar.gz` output.

Plot cumulative evaluation losses from one run on one graph:

```bash
env/bin/python -m pip install -r requirements-plot.txt
env/bin/python scripts/plot_fourier_loss.py --input unzipped_results --max-degree 7 --max-step 15000
# An archive can also be read directly without extraction:
env/bin/python scripts/plot_fourier_loss.py --input 123613.0.tar.gz
```

The script uses raw recorded values, including step zero, and writes PNG, SVG,
CSV, and a source metadata JSON to `results/fourier_loss/`.

Output filenames include the run name, degree range, and step limit. Repeating
the same plot adds `__2`, `__3`, etc. to all four exports, preserving previous
plots and data, even when only one of the previous exports is present.

Use `--list-runs` to see available runs and `--run <name-or-path-substring>` to select one when
the input folder contains several. Defaults are degrees 1 through 7 and steps
0 through 15,000. With no input specified, it first looks in `unzipped_results`
and falls back to Fourier archives in the project directory if needed.

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

`HammingWeightSquareWave(n, L)` returns `floor(count / L) % 2`, where `count`
is the number of ones in the first `n` input bits. It alternates zero and one
bands of width `L`: for `L=3`, the count pattern is `0, 0, 0, 1, 1, 1, ...`.
Both parameters must be positive integers; extra bits are ignored and
`negative_value=-1` selects signed output labels. `L=1` gives parity, and
`L > n` gives constant zero (or -1 with signed labels).
`avg_sensitivity(seq_len)` computes the exact uniform average sensitivity:
`n * P(Binomial(n-1, 1/2) % L == L-1)` for `seq_len >= n`.
When `L` grows more slowly than `sqrt(n)`, this approaches `n / L`.
Metric names include both parameters: `hamming_weight_square_wave_<n>_<L>`.

```python
from src.functions import HammingWeightSquareWave

square_wave = HammingWeightSquareWave(n=24, L=3)
```

`Mod4SquareWave(n)` is the `L=2` specialization. It counts ones in the first
`n` input bits and returns `1`
when that count modulo 4 is 2 or 3, giving the repeating pattern
`0, 0, 1, 1` as the count increases. Extra bits are ignored, and inputs must
contain at least `n` bits. Any positive integer `n` is supported. Set
`negative_value=-1` for -1/1 output labels; inputs remain 0/1 bits.
Uniform average sensitivity is `n / 2` for `n >= 2`, and zero for `n = 1`.
Within a `SelectorFunction`, `n` counts payload bits after the selector
prefix, and accuracy is logged as `accuracy/mod4_square_wave_<n>`.

```python
from src.functions import Mod4SquareWave

mod4 = Mod4SquareWave(n=10)
```

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

## Affinity and local curvature

Training retains loss, accuracy, held-out evaluations, output sharpness, and
paired affinity/local-curvature matrices. Gradient norms/cosines, PCGrad's
aggregate curvature, sensitivity, and final loss-space searches no longer run.
Their training options and `--loss-spaces` switches have been removed. Existing
saved results and standalone analysis helpers remain available.

Every `affinity_interval` completed updates (default `100`), both matrices are
measured at the post-update model on the same fixed task probes. Rows are
**source tasks** and columns are **target tasks**, including the diagonal.
For source task `i`, the diagnostic applies a hypothetical SGD step
`delta_i = -lr * grad L_i(theta)` to all trainable parameters, including
positional embeddings. The learning rates are read from the optimizer's
current parameter groups; trainable parameters outside the optimizer remain
fixed. The diagnostic does not use AdamW moments or weight decay.

Affinity retains the relative TAG definition:

```text
A[i,j] = (L_j(theta) - L_j(theta + delta_i)) / L_j(theta)
```

Positive affinity means help and negative affinity means harm. For example,
`+0.1` means the target loss decreased by 10%. This is the lookahead definition
from [TAG Equation (1)](https://proceedings.neurips.cc/paper/2021/file/e77910ebb93b511588557806310f78f1-Paper.pdf).

The new matrix is the **forward-step curvature contribution**, in loss units:

```text
R[i,j] = L_j(theta + delta_i) - L_j(theta) - grad L_j(theta).T @ delta_i
R[i,j] ≈ 0.5 * delta_i.T @ Hessian(L_j)(theta) @ delta_i
```

Positive `R` means the step performed worse than its first-order prediction;
negative `R` means it performed better. Positive affinity and positive `R`
can coexist when the linear benefit exceeds the curvature penalty. This is a
finite-step residual, including higher-order effects, not a Hessian eigenvalue
or curvature per unit distance. It scales approximately quadratically with
step length locally. It is **not divided by the target loss**; to compare its
contribution in affinity units use `-R[i,j] / L_j(theta)`. The probe baseline is
used internally and is not logged; training/evaluation losses use other batches.

Only these matrix entries and tables are logged:

- `affinity/<source>_to_<target>` and `affinity/matrix`
- `local_curvature/<source>_to_<target>` and `local_curvature/matrix`

No affinity timing, learning-rate, baseline-loss, gradient-norm, or validity
series are emitted. Duplicate task names receive distinct suffixes. Zero source
gradients/rates give zero displacements and zero residuals. A zero target loss
makes relative affinity undefined, but its curvature residual can still be
measured. Undefined values remain NaN (JSON null in local arithmetic results).

Set `affinity_interval=None` to disable both matrices. `affinity_num_inputs`
(default `64`), `affinity_eval_batch_size` (default `256`), and `affinity_seed`
control their shared probe. The same inputs compute all gradients and losses;
these probes are not excluded from training and do not certify generalization.
The existing task gradients also supply every linear prediction, so there are
no additional backward passes, opposite-direction evaluations, or Hessian
computations for `R`. Functional probe parameters and floating buffers use
float64 to reduce cancellation in small residuals (CPU fallback on MPS).
The training model's dtype, parameters, buffers, gradients, modes, optimizer
state, and RNG are preserved. All task gradients are held during the measurement.

Boolean diagnostics run when a W&B logger is supplied. Arithmetic diagnostics
also work without W&B and write pairwise scalar histories to `metrics.jsonl`.
Smoke tests measure both matrices at each of two training steps.

## Output sharpness

Training logs average direction sharpness every 100 completed optimizer steps.
It estimates the mean squared change in predictions after independent Gaussian
noise with standard deviation `sharpness_rho` (default `0.02`) is added to each
non-positional parameter. This is output sensitivity, not a loss Hessian
measurement. In particular, `rho` is a per-coordinate standard deviation,
not the total parameter displacement length.

`sharpness_num_inputs` (default `32`) sets the uniform payload sample count;
`sharpness_num_perturbations` (default `4`) sets the number of noise draws.
Boolean sharpness uses `max_len` and averages over all selector codes, including
unused codes (also reported as `sharpness/invalid_selectors`). Arithmetic
sharpness averages squared token-probability changes over vocabulary, valid
answer positions, examples and tasks. Training selector probabilities do not
weight these diagnostics. The Boolean and arithmetic output scales differ.

Historical loss-space results can still be read without training or a W&B
connection using `env/bin/python scripts/summarize_spaces.py --help`. Those
standalone helpers do not participate in the current training diagnostics.

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
rsync -av dockerfile .dockerignore requirements.txt test.py test_arithmetic.py src tests \
  run-test.sh test.sub submit-wandb.sh submit.sh sweep_arithmetic.py \
  sweep_arithmetic.sub submit-sweep.sh test_fourier.py sweep_fourier.py \
  sweep_fourier.sub submit-fourier-sweep.sh sweep_input_tower.py \
  sweep_input_tower.sub submit-input-tower-sweep.sh README.md \
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

### Submit the arithmetic experiment

On the submission host, with the W&B credentials configured above:

```bash
bash submit.sh
```

This runs `test_arithmetic.py --steps 30000 --payload-len 4 --grouped-selectors`
with all seven arithmetic tasks. It performs 30,000 optimizer updates with the
full model (width 64, four layers) and batches of 320, with periodic held-out
evaluation, affinity, local curvature, and scalar sharpness diagnostics.
Tangent and sharpness loss-space searches are disabled. It uses
the existing container and GPU allocation from `test.sub`; no image rebuild is
needed. W&B defaults to the project
`arithmetic tasks sharpness`; set `WANDB_PROJECT` to override it.

The mix includes addition, subtraction, multiplication, integer division,
remainder, `addition_mod_10`, and `majority`. The last two take a single digit
argument: `addition_mod_10` returns the sum of its digits modulo 10 (`987` gives
`4`), and `majority` returns the most frequent digit (`0012` gives `0`). Leading
zeros count, and majority ties select the smallest digit. Grouped selectors use
`20` and `21` for these two tasks in the default mix. Both return one digit plus EOS.

The default four-digit payload is split into two two-digit operands for the
binary tasks. Majority and addition modulo 10 each use **all four payload
digits** as their single argument. For example, payload `9911` gives majority
`1` (smallest digit wins the tie) and digit sum modulo 10 of `0`.
`--digits-per-argument N` changes the arithmetic operand width; the unary digit
tasks always consume the resulting full payload. `--payload-len N` instead
splits N digits by each task's arity. Use `--operations` to choose a subset.

Model options are `--embedding-size`, `--num-layers`, `--num-attn-heads`, and
`--batch-size`. The token and positional embeddings each have `embedding_size`
dimensions and are concatenated, so the transformer width is twice that size.
Alternatively, `--d-model` sets the total width directly; it must be even and
divisible by the head count. Defaults remain embedding size 32, total width 64,
four layers, four heads, and batch size 320. Explicit model and batch options
are preserved in smoke tests.

Logs are saved under `/scratch/gtomas/logs/sharpness_experiments/` as
`arithmetic.JOB_ID.0.out`, `arithmetic.JOB_ID.0.err`, and `arithmetic.JOB_ID.log`.
The job archive is `/scratch/gtomas/sharpness_experiments/results/JOB_ID.0.tar.gz`.
It includes `results/exit-code.txt`, executed sources, and the arithmetic run's
configuration, metrics, and checkpoint under `results/arithmetic/`.

The script accepts the same Condor overrides as `submit-wandb.sh`. Use `-append`
for overrides: bare `name=value` arguments are read before `test.sub`, whose
defaults can overwrite them. To opt into a two-step smoke test with the small
model instead of the full experiment:

```bash
bash submit.sh -append 'experiment_args=--smoke-test --grouped-selectors'
```

For a local check without submitting or using W&B:

```bash
env/bin/python test_arithmetic.py --smoke-test --no-wandb --device cpu \
  --grouped-selectors
```

### Sweep arithmetic model and batch hyperparameters

`sweep_arithmetic.py` defines a Cartesian grid of **36 runs**:

| Parameter | Values |
| --- | --- |
| Token embedding size (and positional embedding size) | 16, 32, 64 |
| Total transformer width | 32, 64, 128, respectively |
| Transformer layers | 2, 4 |
| Attention heads | 2, 4, 8 |
| Batch size | 64, 320 |

All combinations have valid head dimensions. Each run performs 30,000 optimizer
updates with seed 7, AdamW learning rate `1e-4` and weight decay `1e-2`. All seven
tasks are sampled equally. Every prompt has two selector digits and four payload
digits; both unary digit tasks consume all four payload digits. The held-out
evaluation set uses the same seed and 320 inputs per task for every configuration.
Larger batches see more training examples at the fixed step count. Each run
records the retained loss, accuracy, evaluation, sharpness and paired matrices.

Copy the new sweep files and the updated `test_arithmetic.py`, `src/`,
`submit-wandb.sh`, and `run-test.sh` to the submission host using the copy command
above. Preview the grid without training or submitting anything:

```bash
python sweep_arithmetic.py --list
```

Submit all 36 independent GPU jobs on the cluster:

```bash
bash submit-sweep.sh
```

The wrapper selects `sweep_arithmetic.sub` and loads the existing W&B credentials.
Its 36 processes use `--run-index 0` through `35`. Each has a separate W&B run in
`arithmetic tasks sharpness`, with a name such as
`arithmetic-e16-l2-h2-b64-seed7`. Runs from one submission share the W&B group
`arithmetic-sweep-CLUSTER_ID`. Model dimensions, batch size, operand widths, and
training settings are recorded in W&B and local `config.json` files.

The resource requests match the existing single-job setup: one GPU with at least
64 GB GPU memory, two CPUs, 8 GB host RAM, and 4 GB disk per job. The scheduler
determines how many can run concurrently. Logs use the prefix
`arithmetic-sweep.CLUSTER_ID.PROCESS_ID`; result archives are saved to the existing
result directory as `CLUSTER_ID.PROCESS_ID.tar.gz`. Each archive includes both
Python entrypoints, `src/`, the run config, metrics, and checkpoint.

For a local two-step check of one configuration without uploading to W&B:

```bash
env/bin/python sweep_arithmetic.py --run-index 0 --smoke-test --no-wandb --device cpu
```

The smoke flag reduces the number of updates and diagnostic samples; it preserves
the selected architecture and batch size. To shorten the entire cluster sweep:

```bash
bash submit-sweep.sh -append 'experiment_args=--steps 1000'
```

Run the arithmetic and submission regression checks locally:

```bash
env/bin/python -m unittest discover -s tests -p 'test_arithmetic*.py' -v
env/bin/python -m unittest discover -s tests -p 'test_submission.py' -v
```

### Submit a short Boolean GPU check, then the full experiment

On the submission host:

```bash
WANDB_PROJECT='multiple tasks sharpness' bash submit-wandb.sh \
  -append 'experiment_args=--smoke-test'
```

The wrapper creates log/result directories and loads credentials. The smoke
test performs two optimizer updates, evaluation, affinity, local curvature,
and sharpness diagnostics. It retains the full Boolean model and eight tasks
but reduces probe sizes. Its W&B config records `smoke_test=true`; these
measurements validate the pipeline.

Use the cluster ID printed by submission:

```bash
condor_q JOB_ID -nobatch
cat /scratch/gtomas/logs/sharpness_experiments/test.JOB_ID.0.out
cat /scratch/gtomas/logs/sharpness_experiments/test.JOB_ID.0.err
cat /scratch/gtomas/logs/sharpness_experiments/test.JOB_ID.log
```

Verify the CUDA device and GPU name in stdout, the W&B run URL, exit code 0,
and the retained metrics in W&B. W&B emits normal status
messages to stderr, so a nonempty `.err` file is not itself a failure. If held,
inspect `condor_q JOB_ID -hold`. If idle for a long time, use
`condor_q JOB_ID -better-analyze` to inspect resource matching.

Once that succeeds, submit the full 10,000-step experiment:

```bash
WANDB_PROJECT='multiple tasks sharpness' bash submit-wandb.sh
```

`test.sub` starts one job with one GPU, two CPUs, 8 GB of host RAM, and 4 GB of
job disk. Host RAM does not reserve GPU memory. These are starting allocations;
adjust after observing the smoke/full run. Check cluster runtime limits before
starting a full run. The default Boolean experiment has no model checkpoints
or resume mechanism; an evicted job starts training again.

The runner sets numerical-library thread limits from `request_cpus`, including
OpenBLAS and PyTorch's independent interop pool. Startup prints the effective
thread counts. Run completion and failure events are written to
`results/diagnostic-progress.jsonl` when a results directory is configured.
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
  -append 'image=docker.coli.uni-saarland.de/gtomas/sharpness-experiments:v2' \
  -append 'experiment_args=--smoke-test'
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
