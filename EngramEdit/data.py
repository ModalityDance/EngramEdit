"""Inputs and exact n-gram mapping for the LongCat editor."""

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoTokenizer

from util import repr_tools

NGRAM_LENGTHS = (2, 3, 4)


def request_id(request: Dict) -> str:
    case_id = int(request["case_id"])
    rewrite_idx = request.get("rewrite_idx")
    if rewrite_idx is None:
        return str(case_id)
    rewrite_idx = int(rewrite_idx)
    if rewrite_idx < 0:
        raise ValueError("rewrite_idx must be non-negative.")
    return f"{case_id}:{rewrite_idx}"


@lru_cache(maxsize=4)
def load_expressions(path: Path) -> Dict[str, List[str]]:
    """Read the expression file once for sequential edit batches."""
    result = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            key = request_id(record)
            expressions = record["paraphrase_prompts"]
            if not isinstance(expressions, list) or not all(
                isinstance(x, str) for x in expressions
            ):
                raise ValueError(f"Expected a list of expressions at {path}:{line_no}.")
            if key in result:
                raise ValueError(f"Duplicate request ID {key} in {path}.")
            result[key] = expressions
    return result


def get_edit_expressions(request, expression_lookup, count):
    key = request_id(request)
    expressions = expression_lookup[key][:count]
    if len(expressions) < count:
        raise ValueError(f"Request {key} requires {count} generated expressions.")
    return dedupe_prompts([base_prompt(request), *expressions])[1:]


def normalize_prompt(prompt_template: str) -> str:
    return " ".join(prompt_template.replace("\u2019", "'").replace("\r", "\n").split())


def dedupe_prompts(prompt_templates: List[str]) -> List[str]:
    deduped = []
    seen_exact = set()
    seen_normalized = set()
    for prompt_template in prompt_templates:
        normalized = normalize_prompt(prompt_template)
        if not normalized:
            continue
        if prompt_template in seen_exact or normalized in seen_normalized:
            continue
        seen_exact.add(prompt_template)
        seen_normalized.add(normalized)
        deduped.append(prompt_template)
    return deduped


def base_prompt(request: Dict) -> str:
    prompt = request["prompt"]
    if "{}" not in prompt:
        if request["subject"] not in prompt:
            raise ValueError(
                f"Subject '{request['subject']}' not found in prompt '{prompt}'"
            )
        prompt = prompt.replace(request["subject"], "{}")
    return prompt


def subject_positions(
    tok: AutoTokenizer,
    prompt_templates: List[str],
    subject: str,
) -> List[int]:
    return repr_tools.get_last_subject_token_indices(tok, prompt_templates, subject)


def get_ngram_keys(
    tok: AutoTokenizer,
    prompt_template: str,
    subject: str,
) -> Dict[int, Tuple[int, ...]]:
    formatted_prompt = prompt_template.format(subject)
    input_ids = tok(formatted_prompt, return_tensors="pt")["input_ids"][0].tolist()
    fact_lookup_idx = subject_positions(tok, [prompt_template], subject)[0]
    eos_token_id = tok.eos_token_id

    keys = {}
    for length in NGRAM_LENGTHS:
        start = fact_lookup_idx - length + 1
        if start < 0:
            continue
        window = input_ids[start : fact_lookup_idx + 1]
        if len(window) != length:
            continue
        if eos_token_id is not None and eos_token_id in window[:-1]:
            continue
        keys[length] = tuple(window)

    return keys


@lru_cache(maxsize=4)
def load_frequency_cache(path: Path) -> Dict:
    """Load the cache produced by scripts/prepare/frequency.py."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    return {
        "counts_by_length": {
            length: {
                tuple(key): int(count)
                for key, count in payload["counts_by_length"][length].items()
            }
            for length in NGRAM_LENGTHS
        },
        "count_percentiles_by_length": {
            length: {
                int(count): float(rank)
                for count, rank in payload["count_percentiles_by_length"][
                    length
                ].items()
            }
            for length in NGRAM_LENGTHS
        },
    }
