"""Evaluate the six fixed tasks without changing their prompts or scoring."""

from pathlib import Path

import torch

from experiments.utils import save_json
from .sst_eval import SSTEval
from .mrpc_eval import MRPCEval
from .cola_eval import COLAEval
from .rte_eval import RTEEval
from .nli_eval import NLIEval
from .mmlu_eval import MMLUEval

TASKS = {
    "sst": SSTEval,
    "mrpc": MRPCEval,
    "cola": COLAEval,
    "rte": RTEEval,
    "nli": NLIEval,
    "mmlu": MMLUEval,
}


def evaluate(model, tokenizer, output_path, number_of_tests=100, edit_count=0):
    """Save task-level F1 and its arithmetic mean, both on a 0--1 scale."""
    output_path = Path(output_path)
    results = {"edit_num": edit_count, "number_of_tests": number_of_tests}
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for task, evaluator_class in TASKS.items():
                print(f"[Capabilities] {task}")
                evaluator = evaluator_class(
                    model,
                    tokenizer,
                    number_of_tests=number_of_tests,
                    number_of_few_shots=0,
                )
                metrics, generations = evaluator.evaluate(gen_len=5)
                results[task] = metrics
                save_json(
                    output_path.with_name(f"{output_path.stem}_{task}_gen.json"),
                    generations,
                )
    finally:
        model.train(was_training)
    results["mean_f1"] = sum(results[task]["f1_new"] for task in TASKS) / len(TASKS)
    save_json(output_path, results)
    return results
