"""Summarize per-case metrics; Utility is the arithmetic mean of E/G/S."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def summarize(path, ds_name=None):
    path = Path(path)
    if ds_name is None:
        with (path / "0_run_metadata.json").open() as stream:
            ds_name = json.load(stream)["ds_name"]
    values = defaultdict(list)
    files = sorted((path / "case_results").glob("*_edits-case_*.json"))
    if not files:
        raise ValueError(f"No case results in {path}")
    for case_file in files:
        with case_file.open() as stream:
            case = json.load(stream)
        for phase in ("pre", "post"):
            metrics = case.get(phase, {})
            if ds_name in {"cf", "mcf"}:
                for name, field in (
                    ("Efficacy", "rewrite"),
                    ("Generalization", "paraphrase"),
                    ("Specificity", "neighborhood"),
                ):
                    probabilities = metrics.get(f"{field}_prompts_probs", [])
                    if probabilities:
                        scores = [
                            p["target_true"] < p["target_new"]
                            if field == "neighborhood"
                            else p["target_new"] < p["target_true"]
                            for p in probabilities
                        ]
                        values[f"{phase}/{name}"].append(float(np.mean(scores)) * 100)
                for field, name in (
                    ("ngram_entropy", "Fluency"),
                    ("reference_score", "Consistency"),
                ):
                    if field in metrics:
                        values[f"{phase}/{name}"].append(metrics[field] * 100)
            else:
                fields = (
                    [("Multi-hop", "rewrite"), ("Multi-hop (CoT)", "multihop_cot")]
                    if ds_name == "mquake"
                    else [
                        ("Efficacy", "rewrite"),
                        ("Generalization", "paraphrase"),
                        ("Specificity", "neighborhood"),
                    ]
                )
                for name, field in fields:
                    scores = metrics.get(f"{field}_prompts_correct", [])
                    if scores:
                        values[f"{phase}/{name}"].append(float(np.mean(scores)) * 100)
            if ds_name == "mquake" and "hop_count" in metrics:
                hops = metrics["hop_count"]
                for name, field in (
                    ("Multi-hop", "rewrite"),
                    ("Multi-hop (CoT)", "multihop_cot"),
                ):
                    scores = metrics.get(f"{field}_prompts_correct", [])
                    if scores:
                        values[f"{phase}/{name}/{hops}-hop"].append(
                            float(np.mean(scores)) * 100
                        )
    summary = {"ds_name": ds_name, "num_case_files": len(files), "metrics": {}}
    for key, scores in values.items():
        # These are between-case standard deviations, not confidence intervals.
        summary["metrics"][key] = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "n": len(scores),
        }
    for phase in ("pre", "post"):
        keys = [
            f"{phase}/{name}" for name in ("Efficacy", "Generalization", "Specificity")
        ]
        if all(key in summary["metrics"] for key in keys):
            summary["metrics"][f"{phase}/Utility"] = {
                "mean": sum(summary["metrics"][key]["mean"] for key in keys) / 3
            }
    with (path / "summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    for key, metric in summary["metrics"].items():
        print(f"{key}: {metric['mean']:.2f}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--ds_name", choices=["cf", "mcf", "zsre", "mquake"])
    args = parser.parse_args()
    summarize(args.path, args.ds_name)


if __name__ == "__main__":
    main()
