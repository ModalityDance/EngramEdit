"""Sequential additive EngramEdit on LongCat, with editing and capability tests."""

import argparse
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import torch
from tqdm import tqdm

from EngramEdit import (
    EngramEditHyperParams,
    apply_engramedit_to_model,
    prepare_engramedit_model,
)
from EngramEdit.data import request_id
from EngramEdit.engramedit_main import generate_context_templates
from dsets import (
    AttributeSnippets,
    CounterFactDataset,
    MultiCounterFactDataset,
    ZsREDataset,
    MQUAKEDataset,
    get_tfidf_vectorizer,
)
from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact
from experiments.py.eval_utils_zsre import compute_rewrite_quality_zsre
from experiments.py.eval_utils_mquake import compute_rewrite_quality_mquake
from experiments.summarize import summarize
from experiments.utils import flatten_requests, load_model, read_json, save_json
from glue_eval.glue_eval import evaluate as evaluate_capabilities
from util.globals import DATA_DIR, HPARAMS_DIR

DATASETS = {
    "mcf": (MultiCounterFactDataset, compute_rewrite_quality_counterfact),
    "cf": (CounterFactDataset, compute_rewrite_quality_counterfact),
    "zsre": (ZsREDataset, compute_rewrite_quality_zsre),
    "mquake": (MQUAKEDataset, compute_rewrite_quality_mquake),
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default=str(DATA_DIR / "LongCat-Flash-Lite"))
    parser.add_argument("--ds_name", choices=DATASETS, default="mcf")
    parser.add_argument("--data_dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--hparams",
        type=Path,
        default=HPARAMS_DIR / "EngramEdit/longcat-flash-lite_lenfreq.json",
    )
    parser.add_argument("--dataset_size_limit", type=int, default=2000)
    parser.add_argument(
        "--num_edits",
        type=int,
        default=100,
        help="Cases per sequential batch; MQuAKE cases contain multiple facts.",
    )
    parser.add_argument("--paraphrase_path", type=Path, required=True)
    parser.add_argument("--paraphrase_append_count", type=int, default=4)
    parser.add_argument("--ngram_frequency_cache", type=Path, required=True)
    parser.add_argument(
        "--run_dir",
        type=Path,
        required=True,
        help="Exact output directory; must be empty for a new run.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="With --resume, evaluate the saved state without editing.",
    )
    parser.add_argument("--pre_eval", action="store_true")
    parser.add_argument(
        "--skip_generation_tests",
        action="store_true",
        help="Skip CounterFact fluency and consistency only.",
    )
    parser.add_argument("--generation_test_interval", type=int, default=1)
    parser.add_argument(
        "--edit_eval_interval",
        type=int,
        default=0,
        help="Evaluate accumulated cases every N cases (0 disables).",
    )
    parser.add_argument(
        "--downstream_eval_steps",
        type=int,
        default=0,
        help="Run all six capability tasks every N edit batches (0 disables).",
    )
    parser.add_argument("--glue_num_examples", type=int, default=100)
    parser.add_argument("--mquake_shuffle_seed", type=int, default=0)
    parser.add_argument(
        "--torch_dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16"
    )
    args = parser.parse_args(argv)
    for name in (
        "dataset_size_limit",
        "num_edits",
        "generation_test_interval",
        "glue_num_examples",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive.")
    for name in (
        "paraphrase_append_count",
        "edit_eval_interval",
        "downstream_eval_steps",
    ):
        if getattr(args, name) < 0:
            parser.error(f"--{name} must be non-negative.")
    if args.paraphrase_append_count == 0:
        parser.error("--paraphrase_append_count must be positive for the main method.")
    if args.eval_only and not args.resume:
        parser.error("--eval_only requires --resume.")
    if args.ds_name == "cf" and args.num_edits != 1:
        parser.error(
            "Use mcf (MultiCounterFact) for sequential batches, or cf with --num_edits=1."
        )
    if args.edit_eval_interval % args.num_edits:
        parser.error("--edit_eval_interval must be a multiple of --num_edits.")
    for path in (args.hparams, args.paraphrase_path, args.ngram_frequency_cache):
        if not path.is_file():
            parser.error(f"Required file does not exist: {path}")
    return args


def run_config(args, hparams, records):
    return {
        "method": "EngramEdit",
        "memory_update_mode": "additive",
        "model_name": args.model_name,
        "torch_dtype": args.torch_dtype,
        "ds_name": args.ds_name,
        "dataset_size_limit": len(records),
        "num_edits": args.num_edits,
        "mquake_shuffle_seed": args.mquake_shuffle_seed
        if args.ds_name == "mquake"
        else None,
        "paraphrase_append_count": args.paraphrase_append_count,
        "hparams": asdict(hparams),
    }


def validate_run_directory(run_dir, resume):
    if resume:
        for name in ("0_run_metadata.json", "0_params.json", "engramedit_state.pt"):
            if not (run_dir / name).is_file():
                raise FileNotFoundError(f"Cannot resume without {run_dir / name}")
    elif run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Nonempty run directory: {run_dir}. Use --resume or a new directory."
        )


def save_or_check_config(run_dir, config, resume):
    path = run_dir / "0_run_metadata.json"
    if resume:
        previous = read_json(path)
        if previous != config:
            keys = sorted(
                key
                for key in config.keys() | previous.keys()
                if config.get(key) != previous.get(key)
            )
            raise ValueError(
                f"Resume configuration differs: {', '.join(keys)}. Start a new run."
            )
    else:
        save_json(path, config)
        save_json(run_dir / "0_params.json", config["hparams"])


def completed_case_count(wrapper, records, batch_size):
    """Reject partial batches or a state from a different sequence."""
    completed = wrapper.processed_request_ids | wrapper.skipped_request_ids
    count, expected = 0, set()
    seen_pending = False
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        ids = {request_id(request) for request in flatten_requests(batch)}
        expected.update(ids)
        overlap = ids & completed
        if overlap and overlap != ids:
            raise ValueError(
                "Saved state contains a partial edit batch; use its original batching."
            )
        if overlap:
            if seen_pending:
                raise ValueError(
                    "Saved state is not a contiguous prefix of this edit sequence."
                )
            count += len(batch)
        else:
            seen_pending = True
    if completed - expected:
        raise ValueError("Saved state contains requests outside this dataset.")
    return count


def evaluate_records(
    model, tok, records, args, output_dir, phase, snips=None, vec=None, edit_count=0
):
    output_dir = Path(output_dir)
    evaluator = DATASETS[args.ds_name][1]
    model.eval()
    with torch.no_grad():
        for record in tqdm(records, desc=f"{phase} evaluation"):
            path = (
                output_dir
                / "case_results"
                / f"{args.num_edits}_edits-case_{record['case_id']}.json"
            )
            metrics = read_json(path) if path.exists() else {}
            generation_helpers = (
                (snips, vec)
                if record["case_id"] % args.generation_test_interval == 0
                else (None, None)
            )
            # Re-evaluation writes a complete metric block for the current state.
            result = evaluator(
                model,
                tok,
                record,
                *generation_helpers,
                **({"evaluation_mode": phase} if args.ds_name == "mquake" else {}),
            )
            metrics.update(
                {
                    "case_id": record["case_id"],
                    "num_edits": args.num_edits,
                    "requested_rewrite": record["requested_rewrite"],
                    phase: result,
                }
            )
            if phase == "post":
                metrics["edit_eval_count"] = edit_count
            save_json(path, metrics)
    summarize(output_dir, args.ds_name)


def run_capabilities(model, tok, args, count):
    name = "base" if count == 0 else f"edit_{count}"
    path = args.run_dir / "glue_eval" / f"{name}_glue.json"
    if path.exists():
        previous = read_json(path)
        if previous.get("number_of_tests") == args.glue_num_examples:
            return
    with (args.run_dir / "capability_eval.log").open("a") as log, redirect_stdout(log):
        evaluate_capabilities(model, tok, path, args.glue_num_examples, count)
    print(f"[Capabilities] Saved {path}")


def main(argv=None):
    args = parse_args(argv)
    validate_run_directory(args.run_dir, args.resume)
    hparams = EngramEditHyperParams.from_json(args.hparams)
    model, tok = load_model(args.model_name, args.torch_dtype)
    dataset_class = DATASETS[args.ds_name][0]
    dataset = dataset_class(
        args.data_dir,
        tok=tok,
        size=args.dataset_size_limit,
        **(
            {"shuffle_seed": args.mquake_shuffle_seed}
            if args.ds_name == "mquake"
            else {}
        ),
    )
    records = list(dataset)
    if len(records) != args.dataset_size_limit:
        raise ValueError(
            f"Requested {args.dataset_size_limit} cases, but found {len(records)}."
        )
    save_or_check_config(args.run_dir, run_config(args, hparams, records), args.resume)
    save_json(
        args.run_dir / "evaluation_config.json",
        {
            name: getattr(args, name)
            for name in (
                "pre_eval",
                "eval_only",
                "skip_generation_tests",
                "generation_test_interval",
                "edit_eval_interval",
                "downstream_eval_steps",
                "glue_num_examples",
            )
        },
    )
    snips, vec = None, None
    if args.ds_name in {"mcf", "cf"} and not args.skip_generation_tests:
        snips = AttributeSnippets(args.data_dir)
        vec = get_tfidf_vectorizer(args.data_dir)

    # These evaluations happen before loading any saved edits, including on resume.
    if args.pre_eval:
        evaluate_records(model, tok, records, args, args.run_dir, "pre", snips, vec)
    if args.downstream_eval_steps:
        run_capabilities(model, tok, args, 0)
    model = prepare_engramedit_model(
        model,
        state_dir=args.run_dir if args.resume else None,
    )
    wrapper = model.model.ngram_embeddings
    completed = completed_case_count(wrapper, records, args.num_edits)
    print(f"[EngramEdit] {completed}/{len(records)} cases already processed")

    context_path = args.run_dir / "context_templates.json"
    context_templates = None
    if context_path.exists():
        context_templates = read_json(context_path)
    elif completed and not args.eval_only:
        raise FileNotFoundError(
            "Missing context_templates.json; cannot resume target computation."
        )
    elif not args.eval_only:
        context_templates = generate_context_templates(model, tok)
        save_json(context_path, context_templates)

    if not args.eval_only:
        # Finish evaluations at the saved boundary before proceeding to new edits.
        if completed and completed < len(records):
            if args.edit_eval_interval and completed % args.edit_eval_interval == 0:
                evaluate_records(
                    model,
                    tok,
                    records[:completed],
                    args,
                    args.run_dir / "edit_eval" / f"edit_{completed}",
                    "post",
                    edit_count=completed,
                )
            if (
                args.downstream_eval_steps
                and (completed // args.num_edits) % args.downstream_eval_steps == 0
            ):
                run_capabilities(model, tok, args, completed)
        for start in tqdm(
            range(completed, len(records), args.num_edits), desc="Editing"
        ):
            batch = records[start : start + args.num_edits]
            started = perf_counter()
            with (args.run_dir / "editing.log").open("a") as log, redirect_stdout(log):
                model = apply_engramedit_to_model(
                    model,
                    tok,
                    flatten_requests(batch),
                    hparams,
                    context_templates=context_templates,
                    run_dir=args.run_dir,
                    paraphrase_path=str(args.paraphrase_path),
                    paraphrase_append_count=args.paraphrase_append_count,
                    ngram_frequency_cache=str(args.ngram_frequency_cache),
                )
            completed = start + len(batch)
            print(
                f"[EngramEdit] {completed}/{len(records)} cases, {perf_counter() - started:.1f}s"
            )
            if (
                args.edit_eval_interval
                and completed < len(records)
                and completed % args.edit_eval_interval == 0
            ):
                evaluate_records(
                    model,
                    tok,
                    records[:completed],
                    args,
                    args.run_dir / "edit_eval" / f"edit_{completed}",
                    "post",
                    edit_count=completed,
                )
            batch_number = (completed + args.num_edits - 1) // args.num_edits
            if (
                args.downstream_eval_steps
                and batch_number % args.downstream_eval_steps == 0
            ):
                run_capabilities(model, tok, args, completed)
    if completed == 0:
        raise ValueError("The saved state contains no processed edits.")
    # In eval-only mode, never label a partial state as the complete run.
    output_dir = (
        args.run_dir
        if completed == len(records)
        else args.run_dir / "edit_eval" / f"edit_{completed}"
    )
    evaluate_records(
        model,
        tok,
        records[:completed],
        args,
        output_dir,
        "post",
        snips,
        vec,
        edit_count=completed,
    )
    if args.downstream_eval_steps:
        run_capabilities(model, tok, args, completed)


if __name__ == "__main__":
    main()
