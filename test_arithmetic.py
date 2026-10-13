"""Run the arithmetic experiment without changing test.py or the Boolean modules.

Quick local check (all periodic diagnostics, no network):
    env/bin/python test_arithmetic.py --smoke-test --no-wandb
    env/bin/python test_arithmetic.py --smoke-test --no-wandb --grouped-selectors

Full experiment (same default transformer dimensions and AdamW as test.py):
    python test_arithmetic.py --grouped-selectors --digits-per-argument 2

The default seven tasks are a+b, a-b, a*b, a//b, a%b (b != 0 for division and
remainder), addition_mod_10 and majority. The last two take one argument:
addition_mod_10 sums its digits modulo 10; majority returns its most frequent
digit, choosing the smallest digit on ties. Leading zeros count as digits.
Operands are nonnegative, fixed-width, big-endian decimal digit strings;
answers are canonical signed integers followed by EOS. A payload ``1203``
means arguments 12 and 3 at width 2. With grouped selectors, prefixes 00/01
select addition/subtraction, 10/11/12 select multiplication/division/remainder,
and 20/21 select addition_mod_10/majority.
The operand width is configurable; --payload-len instead splits n digits into
k equal arguments. Unary digit tasks always use the entire payload, including
all four digits in the default mix. --arity affects
only addition/multiplication; other tasks retain their unary/binary arity.

Output limit includes EOS (and a possible minus sign). The runner refuses a
limit that would truncate a valid target; generation stops at EOS or the limit.
There is no loss on the prompt or padding. Training loss is mean answer-token
cross entropy per example, then mean over examples. Evaluation accuracy is
exact autoregressive sequence accuracy including EOS; teacher-forced metrics
have separate names. Fixed unique evaluation inputs are excluded from training.

Affinity and local curvature share the task diagnostics in src.metrics. Both use
fixed teacher-forced contexts and the same cross entropy as training. Output sharpness
uses Gaussian parameter noise (excluding positions) and squared changes of token
probabilities, averaged over vocabulary, valid positions, examples, then tasks.
It is a sequence-output extension, not numerically comparable to Boolean MSE.
Probes share digit payloads; invalid operands are resampled per task. Diagnostic
probes are fixed but are not held out. Every diagnostic works without W&B.

Each run saves config.json, metrics.jsonl and checkpoint.pt in a fresh directory.
Existing Boolean cluster launchers still invoke test.py; invoke this file
explicitly to run arithmetic with the same installed dependencies.
"""

import argparse
from dataclasses import asdict
from datetime import datetime
import faulthandler
import json
import os
from pathlib import Path
from uuid import uuid4

from src.runtime import DiagnosticProgress, configure_cpu_environment


OPERATIONS = ("addition", "subtraction", "multiplication", "division", "remainder",
              "addition_mod_10", "majority")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grouped-selectors", action=argparse.BooleanOptionalAction, default=False)
    layout = parser.add_mutually_exclusive_group()
    layout.add_argument("--digits-per-argument", type=int, help="Digits per arithmetic operand (default: 2); digit tasks use the full payload.")
    layout.add_argument("--payload-len", type=int, help="Alternatively, divide n payload digits equally by each task's arity.")
    parser.add_argument("--arity", type=int, default=2, help="Arity for addition and multiplication only.")
    parser.add_argument("--operations", nargs="+", choices=OPERATIONS, default=list(OPERATIONS))
    parser.add_argument("--selector-probabilities", nargs="+", type=float,
                        help="One probability per flattened selector task, summing to 1.")
    parser.add_argument("--balanced-selectors", action="store_true")
    parser.add_argument("--max-output-tokens", type=int, help="Generation cap, including EOS and sign; default fits all tasks.")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    embedding = parser.add_mutually_exclusive_group()
    embedding.add_argument("--embedding-size", type=positive_int,
                           help="Dimension of each token/position embedding; d_model is twice this value.")
    embedding.add_argument("--d-model", type=positive_int, help="Total concatenated model width (default: 64).")
    parser.add_argument("--num-layers", type=positive_int, help="Transformer layers (default: 4).")
    parser.add_argument("--num-attn-heads", type=positive_int, help="Attention heads (default: 4).")
    parser.add_argument("--eval-inputs", type=int, help="Unique evaluation prompts per task (default 320; smoke test 4).")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", help="Defaults to SHARPNESS_DEVICE, then CUDA/MPS/CPU.")
    parser.add_argument("--output-dir", type=Path, help="Empty/new run directory; defaults to results/arithmetic/<run-id>.")
    parser.add_argument("--run-name", help="Optional descriptive W&B run name.")
    parser.add_argument("--wandb-group", help="Optional W&B group for comparing sweep runs.")
    return parser


def positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return parsed


def build_model_config(args, selector, max_output_tokens):
    d_model = args.d_model if args.d_model is not None else (
        2 * args.embedding_size if args.embedding_size is not None else (16 if args.smoke_test else 64))
    heads = args.num_attn_heads if args.num_attn_heads is not None else 4
    layers = args.num_layers if args.num_layers is not None else (1 if args.smoke_test else 4)
    if d_model % 2 or d_model % heads:
        raise ValueError("d_model must be even and divisible by num_attn_heads.")
    return dict(d_model=d_model, num_attn_heads=heads, num_layers=layers, dropout=0.0,
                max_len=selector.prompt_len + max_output_tokens)


def build_selector(args):
    from src.arithmetic import (Addition, Subtraction, Multiplication, IntegerDivision,
                                Remainder, AdditionMod10, Majority, ArithmeticSelector, FunctionGroup)
    classes = dict(zip(OPERATIONS, (Addition, Subtraction, Multiplication, IntegerDivision,
                                   Remainder, AdditionMod10, Majority)))
    width = None if args.payload_len is not None else (
        args.digits_per_argument if args.digits_per_argument is not None else 2)
    arities = {"addition": args.arity, "multiplication": args.arity,
               "addition_mod_10": 1, "majority": 1}
    tasks = [classes[operation](arity=arities.get(operation, 2),
                                digits_per_argument=None if operation in ("addition_mod_10", "majority")
                                else width) for operation in args.operations]
    payload_len = args.payload_len if args.payload_len is not None else max(2, max(task.arity for task in tasks)) * width
    if not args.grouped_selectors:
        return ArithmeticSelector(tasks, payload_len=payload_len)
    groups = []
    for name, members in (("additive", ("addition", "subtraction")),
                          ("multiplicative", ("multiplication", "integer_division", "remainder")),
                          ("digit", ("addition_mod_10", "majority"))):
        grouped = [task for task in tasks if task.operation in members]
        if grouped:
            groups.append(FunctionGroup(name, grouped))
    return ArithmeticSelector(groups=groups, payload_len=payload_len)


def main(argv=None):
    args = build_parser().parse_args(argv)
    cluster_cpus = configure_cpu_environment()
    import torch
    from src.arithmetic import ArithmeticTransformer, decode_answer, sample_batch
    from src.arithmetic_training import TrainingConfig, train

    if cluster_cpus is not None:
        torch.set_num_threads(cluster_cpus)
        torch.set_num_interop_threads(1)
    elif args.smoke_test:
        torch.set_num_threads(1)
    faulthandler.enable()
    default_device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(args.device or os.getenv("SHARPNESS_DEVICE", default_device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    selector = build_selector(args)
    config = TrainingConfig(seed=args.seed, eval_seed=args.seed,
                            affinity_seed=args.seed, sharpness_seed=args.seed,
                            max_output_tokens=args.max_output_tokens if args.max_output_tokens is not None
                            else selector.required_output_tokens)
    if args.smoke_test:
        config.num_steps = 2
        config.batch_size = max(16, len(selector.functions))
        config.eval_interval = config.affinity_interval = 1
        config.sharpness_interval = 1
        config.eval_num_inputs_per_function = 4
        config.eval_batch_size = 16
        config.affinity_num_inputs = config.sharpness_num_inputs = 4
        config.affinity_eval_batch_size = 16
        config.sharpness_num_perturbations = 1
    for argument, field in ((args.steps, "num_steps"), (args.batch_size, "batch_size"),
                            (args.eval_inputs, "eval_num_inputs_per_function")):
        if argument is not None:
            setattr(config, field, argument)
    torch.manual_seed(args.seed)
    model_config = build_model_config(args, selector, config.max_output_tokens)
    model = ArithmeticTransformer(**model_config).to(device)
    config.validate(selector, model)
    optimizer_config = {"lr": 1e-4, "weight_decay": 1e-2}
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_config)
    output_dir = args.output_dir or Path("results/arithmetic") / (
        datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8])
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Choose a new or empty output directory to preserve previous results.")
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "experiment": "arithmetic", "seed": args.seed, "device": str(device),
        "run_name": args.run_name, "wandb_group": args.wandb_group,
        "model_embedding_size": model_config["d_model"] // 2,
        "smoke_test": args.smoke_test, "loss_name": "SequenceCrossEntropy",
        "loss_reduction": "mean_per_example_mean_valid_token", "optimizer_name": "AdamW",
        "selector_size": selector.selector_size, "payload_len": selector.payload_len,
        "selector_functions": selector.names, "selector_codes": selector.codes,
        "selector_groups": selector.group_names, "selector_group_ids": selector.group_ids,
        "selector_probabilities": args.selector_probabilities, "balanced_selectors": args.balanced_selectors,
        "functions": [{"operation": task.operation, "arity": task.arity,
                       "digits_per_argument": task.digits_per_argument,
                       "argument_width": task.argument_width(selector.payload_len)}
                      for task in selector.functions],
        "tokens": {"eos": 10, "minus": 11, "pad": 12, "answer": 13},
        **{f"model_{key}": value for key, value in model_config.items()},
        **{f"optimizer_{key}": value for key, value in optimizer_config.items()},
        **{f"train_{key}": value for key, value in asdict(config).items()},
    }
    (output_dir / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Device: {device}; results: {output_dir.resolve()}", flush=True)
    print(f"Model: {model_config}; batch size: {config.batch_size}; steps: {config.num_steps}", flush=True)
    print("Selectors: " + ", ".join(f"{''.join(map(str, code))}={name}"
                                   for code, name in zip(selector.codes, selector.names)), flush=True)
    progress = DiagnosticProgress(output_dir, device)
    run = None
    if args.wandb:
        import wandb
        run = wandb.init(entity=os.getenv("WANDB_ENTITY", "gaspardtomas-universit-t-des-saarlandes-saarland-university"),
                         project=os.getenv("WANDB_PROJECT", "arithmetic tasks sharpness"), config=metadata,
                         name=args.run_name, group=args.wandb_group,
                         settings=wandb.Settings(_disable_stats=True))
    try:
        result = train(model, selector, optimizer, config,
                       selector_probabilities=args.selector_probabilities, balanced_selectors=args.balanced_selectors,
                       wandb_run=run, output_dir=output_dir)
        torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                    "model_config": model_config, "config": metadata, "completed_steps": config.num_steps},
                   output_dir / "checkpoint.pt")
        print(f"Completed {len(result.losses)} steps; final training loss: {result.losses[-1]:.6f}", flush=True)
        if "eval/accuracy" in result.last_metrics:
            print(f"Held-out exact-answer accuracy: {result.last_metrics['eval/accuracy']:.3f}", flush=True)
        inputs, targets = sample_batch(selector, min(5, config.batch_size), config.max_output_tokens,
                                      device=device, generator=torch.Generator().manual_seed(args.seed + 100))
        predictions = model.generate(inputs, config.max_output_tokens)
        for prompt, target, prediction in zip(inputs.cpu().tolist(), targets.cpu().tolist(), predictions.cpu().tolist()):
            print(f"{prompt}: target={decode_answer(target)}, prediction={decode_answer(prediction)}", flush=True)
        progress({"stage": "run_complete"})
    except BaseException as error:
        progress({"stage": "failed", "error_type": type(error).__name__, "error": str(error)})
        if run is not None:
            run.finish(exit_code=1)
        raise
    else:
        if run is not None:
            run.finish()
    return output_dir


if __name__ == "__main__":
    main()
