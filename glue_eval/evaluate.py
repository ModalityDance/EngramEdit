"""Standalone base-model or saved-additive-state capability evaluation."""

import argparse
from pathlib import Path

from EngramEdit import prepare_engramedit_model
from experiments.utils import load_model, read_json
from glue_eval.glue_eval import evaluate
from util.globals import DATA_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_name", help="Defaults to the model path recorded in the run."
    )
    parser.add_argument(
        "--state_dir",
        type=Path,
        help="Directory containing engramedit_state.pt.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--number_of_tests", type=int, default=100)
    parser.add_argument("--torch_dtype", choices=["bfloat16", "float16", "float32"])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Choose a new output path: {args.output}")
    if (
        args.state_dir is not None
        and not (args.state_dir / "engramedit_state.pt").is_file()
    ):
        raise FileNotFoundError(f"No additive state in {args.state_dir}")
    metadata = (
        read_json(args.state_dir / "0_run_metadata.json")
        if args.state_dir is not None and (args.state_dir / "0_run_metadata.json").is_file()
        else {}
    )
    model_name = args.model_name or metadata.get(
        "model_name", str(DATA_DIR / "LongCat-Flash-Lite")
    )
    torch_dtype = args.torch_dtype or metadata.get("torch_dtype", "bfloat16")
    model, tok = load_model(model_name, torch_dtype)
    edit_count = 0
    if args.state_dir is not None:
        model = prepare_engramedit_model(
            model,
            state_dir=args.state_dir,
        )
        # A MQuAKE case can contain several atomic requests.
        wrapper = model.model.ngram_embeddings
        request_ids = wrapper.processed_request_ids | wrapper.skipped_request_ids
        edit_count = len({request_id.split(":")[0] for request_id in request_ids})
    evaluate(model, tok, args.output, args.number_of_tests, edit_count)


if __name__ == "__main__":
    main()
