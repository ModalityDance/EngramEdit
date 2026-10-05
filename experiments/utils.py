"""Small shared helpers for the editing and evaluation entrypoints."""

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
    temporary.replace(path)


def load_model(model_name, torch_dtype="bfloat16"):
    """Load LongCat on the available GPUs; its repository supplies model code."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=getattr(torch, torch_dtype),
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, fix_mistral_regex=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def flatten_requests(records):
    requests = []
    for record in records:
        rewrites = record["requested_rewrite"]
        if isinstance(rewrites, list):
            for index, rewrite in enumerate(rewrites):
                requests.append(
                    {**rewrite, "case_id": record["case_id"], "rewrite_idx": index}
                )
        else:
            requests.append({**rewrites, "case_id": record["case_id"]})
    return requests
