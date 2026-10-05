"""Evaluate a saved EngramEdit checkpoint without running any editing steps."""

import argparse
from pathlib import Path

from EngramEdit import prepare_engramedit_model
from dsets import AttributeSnippets, get_tfidf_vectorizer
from experiments.evaluate import DATASETS, evaluate_records
from experiments.utils import load_model, save_json
from util.globals import DATA_DIR


def evaluate(args):
    state_path = args.state_dir / "engramedit_state.pt"
    if not state_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {state_path}")
    if args.output_dir.exists() and (
        not args.output_dir.is_dir() or any(args.output_dir.iterdir())
    ):
        raise FileExistsError(f"Choose an empty output directory: {args.output_dir}")
    if args.generation_tests and args.ds_name not in {"cf", "mcf"}:
        raise ValueError("--generation_tests is for CounterFact only; MQuAKE always generates.")

    model, tok = load_model(args.model_name, args.torch_dtype)
    model = prepare_engramedit_model(model, state_dir=args.state_dir)
    records = list(DATASETS[args.ds_name][0](
        args.data_dir, tok=tok, size=args.dataset_size_limit,
        **({"shuffle_seed": args.mquake_shuffle_seed} if args.ds_name == "mquake" else {}),
    ))
    if len(records) != args.dataset_size_limit:
        raise ValueError(f"Requested {args.dataset_size_limit} cases, but found {len(records)}.")
    snips, vec = None, None
    if args.generation_tests:
        snips = AttributeSnippets(args.data_dir)
        vec = get_tfidf_vectorizer(args.data_dir)
    save_json(args.output_dir / "evaluation_config.json", {
        name: str(value) if isinstance(value, Path) else value
        for name, value in vars(args).items()
    })
    memory = model.model.ngram_embeddings
    request_ids = memory.processed_request_ids | memory.skipped_request_ids
    edit_count = len({request_id.split(":")[0] for request_id in request_ids})
    evaluate_records(model, tok, records, args, args.output_dir, "post", snips, vec,
                     edit_count=edit_count)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", required=True, help="Path to the matching LongCat base model.")
    parser.add_argument("--state_dir", type=Path, required=True,
                        help="Directory containing engramedit_state.pt.")
    parser.add_argument("--ds_name", choices=DATASETS, required=True)
    parser.add_argument("--data_dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--dataset_size_limit", type=int, default=2000)
    parser.add_argument("--num_edits", type=int, default=100,
                        help="Original edit batch size, used to name per-case result files.")
    parser.add_argument("--mquake_shuffle_seed", type=int, default=0)
    parser.add_argument("--torch_dtype", choices=["bfloat16", "float16", "float32"],
                        default="bfloat16")
    parser.add_argument("--generation_tests", action="store_true",
                        help="Also evaluate CounterFact Consistency and Fluency.")
    parser.set_defaults(generation_test_interval=1)
    args = parser.parse_args(argv)
    if args.dataset_size_limit <= 0 or args.num_edits <= 0:
        parser.error("--dataset_size_limit and --num_edits must be positive.")
    evaluate(args)


if __name__ == "__main__":
    main()
