#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from EngramEdit.data import NGRAM_LENGTHS, get_ngram_keys

SUPPORTED_LENGTHS = NGRAM_LENGTHS
TARGET_KEYS_BY_LENGTH: Dict[int, Set[Tuple[int, ...]]] = {}


def _encode(tok, text: str) -> List[int]:
    return tok(text, add_special_tokens=True)["input_ids"]


def _target_prefix(tok, target_new: Dict) -> str:
    target = target_new["str"]
    if not target.startswith(" "):
        target = " " + target
    target_ids = _encode(tok, target)
    if target_ids and target_ids[0] in {tok.bos_token_id, tok.unk_token_id}:
        target_ids = target_ids[1:]
    return tok.decode(target_ids[:-1])


def _load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def _request_id(record: Dict) -> str:
    case_id = int(record["case_id"])
    rewrite_idx = record.get("rewrite_idx")
    return str(case_id) if rewrite_idx is None else f"{case_id}:{int(rewrite_idx)}"


def _load_paraphrase_sidecar(
    path: Path,
    append_count: int,
) -> Dict[str, List[str]]:
    lookup = {}
    with path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            payload = json.loads(line)
            request_id = _request_id(payload)
            if request_id in lookup:
                raise ValueError(f"Duplicate request_id={request_id} in {path}.")
            prompts = payload.get("paraphrase_prompts", [])
            if not isinstance(prompts, list):
                raise ValueError(
                    f"request_id={request_id} in {path} has invalid paraphrase_prompts."
                )
            if len(prompts) < append_count:
                raise ValueError(
                    f"request_id={request_id} in {path} has fewer than "
                    f"{append_count} paraphrases."
                )
            lookup[request_id] = prompts[:append_count]
    return lookup


def _iter_records(ds_name: str, data_dir: Path, tok, size: Optional[int]):
    if ds_name in {"cf", "mcf"}:
        filename = "counterfact.json" if ds_name == "cf" else "multi_counterfact.json"
        records = _load_json(data_dir / filename)
        for record in records[:size]:
            yield record
        return

    if ds_name == "mquake":
        records = _load_json(data_dir / "MQuAKE-CF-3k-v2.json")
        for case_id, record in enumerate(records[:size]):
            rewrites = record.get("requested_rewrite")
            if not isinstance(rewrites, list) or not rewrites:
                raise ValueError(
                    f"MQuAKE case_id={case_id} has no requested_rewrite list."
                )
            for rewrite_idx, rewrite in enumerate(rewrites):
                yield {
                    "case_id": case_id,
                    "rewrite_idx": rewrite_idx,
                    "requested_rewrite": rewrite,
                }
        return

    if ds_name != "zsre":
        raise ValueError(
            f"Unsupported dataset {ds_name!r}; expected cf, mcf, zsre, or mquake."
        )

    raw_records = _load_json(data_dir / "zsre_mend_eval.json")
    for case_id, record in enumerate(raw_records[:size]):
        yield {
            "case_id": case_id,
            "requested_rewrite": {
                "prompt": record["src"].replace(record["subject"], "{}"),
                "subject": record["subject"],
                "target_new": {"str": record["answers"][0]},
                "target_true": {"str": "<|endoftext|>"},
            },
        }


def _dedupe_prompt_templates(prompt_templates: List[str]) -> List[str]:
    deduped = []
    seen = set()
    for prompt_template in prompt_templates:
        normalized = " ".join(
            prompt_template.replace("\u2019", "'").replace("\r", "\n").split()
        )
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(prompt_template)
    return deduped


def _target_prompt_templates_for_record(
    record: Dict,
    target_prefix: str,
    sidecar: Dict[str, List[str]],
    append_count: int,
) -> List[str]:
    rewrite = record["requested_rewrite"]
    prompts = [rewrite["prompt"]]
    request_id = _request_id(record)
    if request_id not in sidecar:
        raise ValueError(f"Missing request_id={request_id} in paraphrase sidecar.")
    prompts.extend(sidecar[request_id][:append_count])
    return _dedupe_prompt_templates([prompt + target_prefix for prompt in prompts])


def _collect_target_keys(
    records: Iterable[Dict],
    tok,
    sidecar: Dict[str, List[str]],
    append_count: int,
) -> Dict[int, Set[Tuple[int, ...]]]:
    target_keys = {length: set() for length in SUPPORTED_LENGTHS}
    for record in records:
        rewrite = record["requested_rewrite"]
        subject = rewrite["subject"]
        target_prefix = _target_prefix(tok, rewrite["target_new"])
        for prompt_template in _target_prompt_templates_for_record(
            record,
            target_prefix,
            sidecar,
            append_count,
        ):
            keys = get_ngram_keys(
                tok,
                prompt_template,
                subject,
            )
            for length, key in keys.items():
                target_keys[length].add(tuple(key))
    return target_keys


def _iter_wikipedia_texts(
    wikipedia_dir: Path, max_docs: Optional[int]
) -> Iterable[str]:
    parquet_files = sorted(wikipedia_dir.glob("train-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No train-*.parquet shards found under {wikipedia_dir}."
        )

    yielded = 0
    for parquet_file in parquet_files:
        dataset = load_dataset("parquet", data_files=str(parquet_file), split="train")
        for row in dataset:
            text = row.get("text")
            if isinstance(text, str) and text:
                yield text
                yielded += 1
                if max_docs is not None and yielded >= max_docs:
                    return


def _iter_batches(items: Iterable[str], batch_size: int) -> Iterable[List[str]]:
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _tokenize_batch(
    tok, texts: List[str], max_tokens: Optional[int]
) -> List[List[int]]:
    encoded = tok(
        texts,
        add_special_tokens=False,
        truncation=max_tokens is not None,
        max_length=max_tokens,
    )
    return encoded["input_ids"]


def _count_target_key_hits(
    token_batches: Iterable[List[int]],
    target_keys_by_length: Dict[int, Set[Tuple[int, ...]]],
) -> Dict[int, Counter]:
    counts = {length: Counter() for length in SUPPORTED_LENGTHS}
    active_lengths = [
        length for length in SUPPORTED_LENGTHS if target_keys_by_length.get(length)
    ]
    for token_ids in token_batches:
        for length in active_lengths:
            if len(token_ids) < length:
                continue
            target_keys = target_keys_by_length[length]
            counter = counts[length]
            for start in range(0, len(token_ids) - length + 1):
                key = tuple(token_ids[start : start + length])
                if key in target_keys:
                    counter[key] += 1
    return counts


def _merge_counts(left: Dict[int, Counter], right: Dict[int, Counter]) -> None:
    for length, counter in right.items():
        left[length].update(counter)


def _init_count_worker(target_keys_by_length: Dict[int, Set[Tuple[int, ...]]]) -> None:
    global TARGET_KEYS_BY_LENGTH
    TARGET_KEYS_BY_LENGTH = target_keys_by_length


def _count_worker(token_batch: List[List[int]]) -> Dict[int, Counter]:
    return _count_target_key_hits(token_batch, TARGET_KEYS_BY_LENGTH)


def _count_wikipedia_target_hits(
    tok,
    wikipedia_dir: Path,
    max_docs: Optional[int],
    max_tokens: Optional[int],
    batch_size: int,
    num_workers: int,
    target_keys_by_length: Dict[int, Set[Tuple[int, ...]]],
) -> Dict[int, Counter]:
    counts = {length: Counter() for length in SUPPORTED_LENGTHS}
    text_batches = _iter_batches(
        _iter_wikipedia_texts(wikipedia_dir, max_docs=max_docs),
        batch_size,
    )
    token_batch_iter = (
        _tokenize_batch(tok, text_batch, max_tokens) for text_batch in text_batches
    )

    total_batches = (
        None if max_docs is None else (max_docs + batch_size - 1) // batch_size
    )
    if num_workers <= 1:
        for token_batch in tqdm(
            token_batch_iter, total=total_batches, desc="Counting target ngrams"
        ):
            _merge_counts(
                counts, _count_target_key_hits(token_batch, target_keys_by_length)
            )
        return counts

    with mp.Pool(
        processes=num_workers,
        initializer=_init_count_worker,
        initargs=(target_keys_by_length,),
    ) as pool:
        for partial_counts in tqdm(
            pool.imap_unordered(_count_worker, token_batch_iter, chunksize=4),
            total=total_batches,
            desc="Counting target ngrams",
        ):
            _merge_counts(counts, partial_counts)
    return counts


def _count_percentiles(counts: Dict[Tuple[int, ...], int]) -> Dict[int, float]:
    unique_counts = sorted(set(counts.values()))
    if len(unique_counts) <= 1:
        return {count: 0.0 for count in unique_counts}
    denominator = len(unique_counts) - 1
    return {count: rank / denominator for rank, count in enumerate(unique_counts)}


def build_frequency_cache(
    model_name: str,
    data_dir: Path,
    datasets: List[str],
    dataset_size_limit: Optional[int],
    paraphrase_path: Path,
    paraphrase_append_count: int,
    wikipedia_dir: Path,
    max_docs: Optional[int],
    max_tokens: Optional[int],
    batch_size: int,
    num_workers: int,
) -> Dict:
    tok = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )
    sidecar = _load_paraphrase_sidecar(
        paraphrase_path,
        paraphrase_append_count,
    )
    records = []
    for ds_name in datasets:
        records.extend(list(_iter_records(ds_name, data_dir, tok, dataset_size_limit)))
    target_keys_by_length = _collect_target_keys(
        records,
        tok,
        sidecar,
        paraphrase_append_count,
    )
    counts = _count_wikipedia_target_hits(
        tok,
        wikipedia_dir,
        max_docs,
        max_tokens,
        batch_size,
        num_workers,
        target_keys_by_length,
    )
    compact_counts = {
        length: {
            key: int(counts[length].get(key, 0))
            for key in sorted(target_keys_by_length[length])
        }
        for length in SUPPORTED_LENGTHS
    }
    count_percentiles = {
        length: _count_percentiles(compact_counts[length])
        for length in SUPPORTED_LENGTHS
    }
    return {
        "cache_type": "target_key_only",
        "model_name": model_name,
        "data_dir": str(data_dir),
        "datasets": datasets,
        "dataset_size_limit": dataset_size_limit,
        "request_count": len(records),
        "ngram_key_lengths": list(NGRAM_LENGTHS),
        "paraphrase_path": str(paraphrase_path),
        "paraphrase_append_count": paraphrase_append_count,
        "wikipedia_dir": str(wikipedia_dir),
        "max_docs": max_docs,
        "max_tokens": max_tokens,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "lengths": list(SUPPORTED_LENGTHS),
        "target_key_counts_by_length": {
            length: len(target_keys_by_length[length]) for length in SUPPORTED_LENGTHS
        },
        "counts_by_length": compact_counts,
        "count_percentiles_by_length": count_percentiles,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build the n-gram frequency cache for EngramEdit."
    )
    parser.add_argument(
        "--dataset", choices=["cf", "mcf", "zsre", "mquake"], required=True
    )
    parser.add_argument("--model_name", default="data/LongCat-Flash-Lite")
    parser.add_argument("--data_dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--dataset_size_limit", type=int)
    parser.add_argument("--paraphrase_path", type=Path)
    parser.add_argument("--paraphrase_append_count", type=int, default=4)
    parser.add_argument("--wikipedia_dir", type=Path)
    parser.add_argument("--max_docs", type=int, default=3000000)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.dataset_size_limit is None:
        args.dataset_size_limit = 3000 if args.dataset == "mquake" else 2000
    for name in (
        "dataset_size_limit",
        "max_docs",
        "max_tokens",
        "batch_size",
        "num_workers",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive.")
    if args.paraphrase_append_count <= 0:
        parser.error("--paraphrase_append_count must be positive.")
    expressions = Path(f"{args.dataset}_longcat_before_subject.jsonl")
    if args.dataset == "mquake":
        expressions = Path("paraphrases") / expressions
    args.paraphrase_path = args.paraphrase_path or args.data_dir / expressions
    args.wikipedia_dir = args.wikipedia_dir or args.data_dir / "wikipedia/20231101.en"
    args.output = args.output or args.data_dir / "ngram_frequency" / (
        f"LongCat-Flash-Lite_{args.dataset}{args.dataset_size_limit}_para{args.paraphrase_append_count}.pt"
    )
    return args


def main() -> None:
    args = parse_args()
    payload = build_frequency_cache(
        model_name=args.model_name,
        data_dir=args.data_dir,
        datasets=[args.dataset],
        dataset_size_limit=args.dataset_size_limit,
        paraphrase_path=args.paraphrase_path,
        paraphrase_append_count=args.paraphrase_append_count,
        wikipedia_dir=args.wikipedia_dir,
        max_docs=args.max_docs,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"Saved frequency cache to {args.output}")
    print(f"Target keys by length: {payload['target_key_counts_by_length']}")


if __name__ == "__main__":
    main()
