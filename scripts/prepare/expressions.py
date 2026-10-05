"""Generate additional expressions for CounterFact, ZsRE, and MQuAKE."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PLACEHOLDER = "[SUBJ]"
NOISE_PATTERNS = ("...", "___", "____")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate expressions for EngramEdit with LongCat."
    )
    parser.add_argument(
        "--dataset", choices=["cf", "mcf", "zsre", "mquake"], required=True
    )
    parser.add_argument("--model_name", default="data/LongCat-Flash-Lite")
    parser.add_argument("--input_path", type=Path)
    parser.add_argument("--output_path", type=Path)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument(
        "--limit",
        type=int,
        help="Number of cases; defaults to 2,000 for CF/ZsRE and 3,000 for MQuAKE.",
    )
    parser.add_argument("--target_count", type=int, default=4)
    parser.add_argument("--candidate_batch_size", type=int, default=16)
    parser.add_argument("--max_rounds", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=4)
    parser.add_argument("--repetition_penalty", type=float, default=1.06)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument(
        "--torch_dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16"
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    inputs = {
        "cf": "counterfact.json",
        "mcf": "multi_counterfact.json",
        "zsre": "zsre_mend_eval.json",
        "mquake": "MQuAKE-CF-3k-v2.json",
    }
    args.input_path = args.input_path or PROJECT_ROOT / "data" / inputs[args.dataset]
    output = Path(f"{args.dataset}_longcat_before_subject.jsonl")
    if args.dataset == "mquake":
        output = Path("paraphrases") / output
    args.output_path = args.output_path or PROJECT_ROOT / "data" / output
    if args.limit is None:
        args.limit = 3000 if args.dataset == "mquake" else 2000
    if args.start_idx < 0 or args.limit < 0:
        parser.error("--start_idx and --limit must be non-negative.")
    for name in (
        "target_count",
        "candidate_batch_size",
        "max_rounds",
        "max_new_tokens",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive.")
    return args


@dataclass
class CandidateDecision:
    accepted: bool
    normalized_prompt: Optional[str]
    reason: Optional[str]


def load_counterfact_records(input_path: Path) -> List[Dict]:
    with input_path.open("r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {input_path}, got {type(data).__name__}")
    return data


def load_completed_case_ids(output_path: Path) -> set[int]:
    completed = set()
    if not output_path.exists():
        return completed

    with output_path.open("r") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if "case_id" not in payload:
                raise ValueError(
                    f"Missing case_id in existing output at line {line_no}"
                )
            completed.add(int(payload["case_id"]))
    return completed


def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dtype_from_name(dtype_name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[dtype_name]


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def prompt_to_placeholder(prompt: str) -> str:
    if prompt.count("{}") != 1:
        raise ValueError(f"Expected exactly one '{{}}' in prompt {prompt!r}")
    return prompt.replace("{}", PLACEHOLDER)


def placeholder_to_prompt(prompt: str) -> str:
    return prompt.replace(PLACEHOLDER, "{}")


def dedupe_key(text: str) -> str:
    text = text.lower()
    text = text.replace(PLACEHOLDER.lower(), " subj ")
    text = text.replace("{}", " subj ")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_candidate_text(text: str) -> str:
    text = normalize_text(text)
    text = text.replace("\r", "\n").strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    text = lines[0] if lines else ""

    while len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"', "`"}:
        text = text[1:-1].strip()

    text = re.sub(r"^\s*(?:[-*]|\d+[\)\].:-])\s*", "", text)
    text = re.sub(
        r"^\s*(?:paraphrase|rewrite|answer)\s*[:\-]\s*", "", text, flags=re.IGNORECASE
    )
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+\.", ".", text)
    text = text.rstrip(" .")
    return text


def evaluate_candidate(
    raw_candidate: str,
    base_prompt_with_placeholder: str,
    target_true: str,
    target_new: str,
    existing_prompts: Sequence[str],
) -> CandidateDecision:
    cleaned = clean_candidate_text(raw_candidate)
    if not cleaned:
        return CandidateDecision(False, None, "empty")

    placeholder_count = cleaned.count(PLACEHOLDER)
    if placeholder_count == 0:
        return CandidateDecision(False, None, "missing_placeholder")
    if placeholder_count > 1:
        return CandidateDecision(False, None, "multiple_placeholders")

    cleaned_lower = cleaned.lower()
    true_lower = target_true.strip().lower()
    new_lower = target_new.strip().lower()
    if true_lower and true_lower in cleaned_lower:
        return CandidateDecision(False, None, "contains_target_true")
    if new_lower and new_lower in cleaned_lower:
        return CandidateDecision(False, None, "contains_target_new")

    if any(pattern in cleaned for pattern in NOISE_PATTERNS):
        return CandidateDecision(False, None, "contains_noise_placeholder")

    if len(dedupe_key(cleaned)) < 6:
        return CandidateDecision(False, None, "too_short")

    if dedupe_key(cleaned) == dedupe_key(base_prompt_with_placeholder):
        return CandidateDecision(False, None, "matches_base_prompt")

    normalized_prompt = placeholder_to_prompt(cleaned)
    if normalized_prompt in existing_prompts:
        return CandidateDecision(False, None, "duplicate_exact")

    existing_keys = {
        dedupe_key(prompt.replace("{}", PLACEHOLDER)) for prompt in existing_prompts
    }
    if dedupe_key(cleaned) in existing_keys:
        return CandidateDecision(False, None, "duplicate_near")

    return CandidateDecision(True, normalized_prompt, None)


def build_user_instruction(
    base_prompt_with_placeholder: str,
    target_true: str,
    target_new: str,
    existing_prompts: Sequence[str],
) -> str:
    local_rewrite_rules = (
        f"- Prioritize changing the local phrasing immediately before {PLACEHOLDER} when natural.\n"
        f"- Good ways to vary it include changing the relation phrase or preposition before {PLACEHOLDER}, using a possessive form like {PLACEHOLDER}'s ..., or rewriting as a question.\n"
        f"- Do not rely only on changing words after {PLACEHOLDER}; try to make the left-side local frame meaningfully different.\n"
    )

    instruction = (
        "Rewrite the factual prompt stem below into one alternative stem with the same meaning.\n"
        "Rules:\n"
        f"- The output must contain {PLACEHOLDER} exactly once.\n"
        "- Keep it as an incomplete factual sentence stem that naturally expects a completion.\n"
        "- Keep the meaning aligned with the original stem.\n"
        f"{local_rewrite_rules}"
        "- Natural paraphrases are more important than forcing an unnatural token pattern.\n"
        f"- Do not use the answer words '{target_true.strip()}' or '{target_new.strip()}'.\n"
        "- Do not use placeholders like ... or ____.\n"
        "- Do not add numbering, explanation, quotation marks, or multiple options.\n"
        f"- Output exactly one rewritten stem and nothing else.\n\n"
        f"Original stem: {base_prompt_with_placeholder}"
    )
    if existing_prompts:
        prior_stems = "\n".join(
            f"- {prompt.replace('{}', PLACEHOLDER)}" for prompt in existing_prompts
        )
        instruction += (
            "\n\nAlready accepted stems for this case. Try not to closely mirror the phrasing immediately before "
            f"{PLACEHOLDER} in these stems:\n"
            f"{prior_stems}"
        )
    return instruction


def build_generation_prompt(tokenizer: AutoTokenizer, user_instruction: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_instruction}],
            tokenize=False,
            add_generation_prompt=True,
        )

    return f"You rewrite factual prompt stems.\n{user_instruction}\nRewritten stem:"


def generate_candidate_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    generation_prompt: str,
    candidate_batch_size: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    max_new_tokens: int,
) -> List[str]:
    model_device = next(model.parameters()).device
    inputs = tokenizer([generation_prompt], return_tensors="pt").to(model_device)

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            num_return_sequences=candidate_batch_size,
            pad_token_id=pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_length = inputs["input_ids"].shape[1]
    continuations = outputs[:, prompt_length:]
    return tokenizer.batch_decode(continuations, skip_special_tokens=True)


def create_record(
    record: Dict,
    paraphrase_prompts: List[str],
    audit: Dict,
    args: argparse.Namespace,
    generated_at: str,
) -> Dict:
    rewrite = record["requested_rewrite"]
    return {
        "case_id": int(record["case_id"]),
        "subject": rewrite["subject"],
        "base_prompt": rewrite["prompt"],
        "target_true": rewrite["target_true"]["str"],
        "target_new": rewrite["target_new"]["str"],
        "paraphrase_prompts": paraphrase_prompts,
        "generator": {
            "model_name": args.model_name,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
            "max_new_tokens": args.max_new_tokens,
            "candidate_batch_size": args.candidate_batch_size,
            "max_rounds": args.max_rounds,
            "target_count": args.target_count,
            "seed": args.seed,
            "generated_at": generated_at,
        },
        "audit": audit,
    }


def append_jsonl(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def generate_for_record(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    record: Dict,
    args: argparse.Namespace,
) -> Tuple[List[str], Dict]:
    rewrite = record["requested_rewrite"]
    base_prompt = rewrite["prompt"]
    base_prompt_with_placeholder = prompt_to_placeholder(base_prompt)
    target_true = rewrite["target_true"]["str"]
    target_new = rewrite["target_new"]["str"]

    accepted_prompts: List[str] = []
    raw_candidates_sample: List[str] = []
    rejection_counts: Counter = Counter()

    rounds_used = 0
    for round_idx in range(args.max_rounds):
        if len(accepted_prompts) >= args.target_count:
            break
        rounds_used = round_idx + 1
        user_instruction = build_user_instruction(
            base_prompt_with_placeholder=base_prompt_with_placeholder,
            target_true=target_true,
            target_new=target_new,
            existing_prompts=accepted_prompts,
        )
        generation_prompt = build_generation_prompt(tokenizer, user_instruction)

        raw_candidates = generate_candidate_batch(
            model=model,
            tokenizer=tokenizer,
            generation_prompt=generation_prompt,
            candidate_batch_size=args.candidate_batch_size,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            max_new_tokens=args.max_new_tokens,
        )

        for raw_candidate in raw_candidates:
            if len(raw_candidates_sample) < 12:
                raw_candidates_sample.append(raw_candidate)

            decision = evaluate_candidate(
                raw_candidate=raw_candidate,
                base_prompt_with_placeholder=base_prompt_with_placeholder,
                target_true=target_true,
                target_new=target_new,
                existing_prompts=accepted_prompts,
            )
            if decision.accepted:
                accepted_prompts.append(decision.normalized_prompt)
                if len(accepted_prompts) >= args.target_count:
                    break
            else:
                rejection_counts[decision.reason] += 1

    stopped_reason = (
        "target_reached" if len(accepted_prompts) >= args.target_count else "max_rounds"
    )
    audit = {
        "rounds_used": rounds_used,
        "accepted_count": len(accepted_prompts),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "raw_candidates_sample": raw_candidates_sample,
        "stopped_reason": stopped_reason,
    }
    return accepted_prompts, audit


@dataclass
class ZsRERecord:
    case_id: int
    subject: str
    base_prompt: str
    target_new: str
    target_true: str
    blocked_answers: List[str]
    original_record: Dict


def load_zsre_records(input_path: Path) -> List[ZsRERecord]:
    with input_path.open("r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {input_path}, got {type(data).__name__}")

    records: List[ZsRERecord] = []
    for case_id, record in enumerate(data):
        subject = record["subject"]
        src = record["src"]
        if subject not in src:
            raise ValueError(
                f"Subject {subject!r} not found in ZsRE src at case_id={case_id}: {src!r}"
            )
        answers = record.get("answers") or []
        if not answers:
            raise ValueError(f"Missing answers for ZsRE case_id={case_id}")

        blocked_answers = [answers[0]]
        for key in ("pred", "alt"):
            value = record.get(key)
            if isinstance(value, str) and value.strip():
                blocked_answers.append(value)

        records.append(
            ZsRERecord(
                case_id=case_id,
                subject=subject,
                base_prompt=src.replace(subject, "{}", 1),
                target_new=answers[0],
                target_true="<|endoftext|>",
                blocked_answers=blocked_answers,
                original_record=record,
            )
        )
    return records


def _answer_words_for_instruction(blocked_answers: Sequence[str]) -> str:
    unique_answers = []
    seen = set()
    for answer in blocked_answers:
        stripped = answer.strip()
        if not stripped:
            continue
        key = stripped.lower()
        if key in seen:
            continue
        seen.add(key)
        unique_answers.append(stripped)
    return "', '".join(unique_answers)


def build_zsre_user_instruction(
    base_prompt_with_placeholder: str,
    blocked_answers: Sequence[str],
    existing_prompts: Sequence[str],
) -> str:
    local_rewrite_rules = (
        f"- Prioritize changing the local phrasing immediately before {PLACEHOLDER} when natural.\n"
        f"- Good ways to vary it include changing the question frame before {PLACEHOLDER}, using a possessive form like {PLACEHOLDER}'s ..., or changing the wh-phrase.\n"
        f"- Do not rely only on changing words after {PLACEHOLDER}; try to make the left-side local frame meaningfully different.\n"
    )

    answer_words = _answer_words_for_instruction(blocked_answers)
    instruction = (
        "Rewrite the factual question below into one alternative question with the same meaning.\n"
        "Rules:\n"
        f"- The output must contain {PLACEHOLDER} exactly once.\n"
        "- Keep it as a natural question or question-like prompt that expects the same answer type.\n"
        "- Keep the meaning aligned with the original question.\n"
        f"{local_rewrite_rules}"
        "- Natural paraphrases are more important than forcing an unnatural token pattern.\n"
        f"- Do not use these answer words: '{answer_words}'.\n"
        "- Do not use placeholders like ... or ____.\n"
        "- Do not add numbering, explanation, quotation marks, or multiple options.\n"
        f"- Output exactly one rewritten question or prompt and nothing else.\n\n"
        f"Original question: {base_prompt_with_placeholder}"
    )
    if existing_prompts:
        prior_prompts = "\n".join(
            f"- {prompt.replace('{}', PLACEHOLDER)}" for prompt in existing_prompts
        )
        instruction += (
            "\n\nAlready accepted questions for this case. Try not to closely mirror the phrasing immediately before "
            f"{PLACEHOLDER} in these questions:\n"
            f"{prior_prompts}"
        )
    return instruction


def evaluate_zsre_candidate(
    raw_candidate: str,
    base_prompt_with_placeholder: str,
    blocked_answers: Sequence[str],
    existing_prompts: Sequence[str],
) -> CandidateDecision:
    cleaned = clean_candidate_text(raw_candidate)
    if not cleaned:
        return CandidateDecision(False, None, "empty")

    placeholder_count = cleaned.count(PLACEHOLDER)
    if placeholder_count == 0:
        return CandidateDecision(False, None, "missing_placeholder")
    if placeholder_count > 1:
        return CandidateDecision(False, None, "multiple_placeholders")

    cleaned_lower = cleaned.lower()
    for answer in blocked_answers:
        answer_lower = answer.strip().lower()
        if answer_lower and answer_lower in cleaned_lower:
            return CandidateDecision(False, None, "contains_answer")

    if len(dedupe_key(cleaned)) < 6:
        return CandidateDecision(False, None, "too_short")

    if dedupe_key(cleaned) == dedupe_key(base_prompt_with_placeholder):
        return CandidateDecision(False, None, "matches_base_prompt")

    normalized_prompt = placeholder_to_prompt(cleaned)
    if normalized_prompt in existing_prompts:
        return CandidateDecision(False, None, "duplicate_exact")

    existing_keys = {
        dedupe_key(prompt.replace("{}", PLACEHOLDER)) for prompt in existing_prompts
    }
    if dedupe_key(cleaned) in existing_keys:
        return CandidateDecision(False, None, "duplicate_near")

    return CandidateDecision(True, normalized_prompt, None)


def create_zsre_record(
    record: ZsRERecord,
    paraphrase_prompts: List[str],
    audit: Dict,
    args: argparse.Namespace,
    generated_at: str,
) -> Dict:
    return {
        "case_id": record.case_id,
        "subject": record.subject,
        "base_prompt": record.base_prompt,
        "target_true": record.target_true,
        "target_new": record.target_new,
        "paraphrase_prompts": paraphrase_prompts,
        "generator": {
            "model_name": args.model_name,
            "dataset": "zsre",
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
            "max_new_tokens": args.max_new_tokens,
            "candidate_batch_size": args.candidate_batch_size,
            "max_rounds": args.max_rounds,
            "target_count": args.target_count,
            "seed": args.seed,
            "generated_at": generated_at,
        },
        "audit": audit,
    }


def generate_zsre_for_record(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    record: ZsRERecord,
    args: argparse.Namespace,
) -> Tuple[List[str], Dict]:
    base_prompt_with_placeholder = prompt_to_placeholder(record.base_prompt)
    accepted_prompts: List[str] = []
    raw_candidates_sample: List[str] = []
    rejection_counts: Counter = Counter()

    rounds_used = 0
    for round_idx in range(args.max_rounds):
        if len(accepted_prompts) >= args.target_count:
            break
        rounds_used = round_idx + 1
        user_instruction = build_zsre_user_instruction(
            base_prompt_with_placeholder=base_prompt_with_placeholder,
            blocked_answers=record.blocked_answers,
            existing_prompts=accepted_prompts,
        )
        generation_prompt = build_generation_prompt(tokenizer, user_instruction)

        raw_candidates = generate_candidate_batch(
            model=model,
            tokenizer=tokenizer,
            generation_prompt=generation_prompt,
            candidate_batch_size=args.candidate_batch_size,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            max_new_tokens=args.max_new_tokens,
        )

        for raw_candidate in raw_candidates:
            if len(raw_candidates_sample) < 12:
                raw_candidates_sample.append(raw_candidate)

            decision = evaluate_zsre_candidate(
                raw_candidate=raw_candidate,
                base_prompt_with_placeholder=base_prompt_with_placeholder,
                blocked_answers=record.blocked_answers,
                existing_prompts=accepted_prompts,
            )
            if decision.accepted:
                accepted_prompts.append(decision.normalized_prompt)
                if len(accepted_prompts) >= args.target_count:
                    break
            else:
                rejection_counts[decision.reason] += 1

    stopped_reason = (
        "target_reached" if len(accepted_prompts) >= args.target_count else "max_rounds"
    )
    audit = {
        "rounds_used": rounds_used,
        "accepted_count": len(accepted_prompts),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "raw_candidates_sample": raw_candidates_sample,
        "stopped_reason": stopped_reason,
    }
    return accepted_prompts, audit


def iter_atomic_rewrites(
    cases: List[Dict], start_idx: int, limit: int
) -> Iterable[Dict]:
    end_idx = min(len(cases), start_idx + limit)
    for case_id in range(start_idx, end_idx):
        rewrites = cases[case_id].get("requested_rewrite")
        if not isinstance(rewrites, list) or not rewrites:
            raise ValueError(f"MQuAKE case_id={case_id} has no requested_rewrite list.")
        for rewrite_idx, rewrite in enumerate(rewrites):
            yield {
                "case_id": case_id,
                "rewrite_idx": rewrite_idx,
                "requested_rewrite": rewrite,
            }


def load_completed_request_ids(path: Path, target_count: int) -> Set[Tuple[int, int]]:
    completed: Set[Tuple[int, int]] = set()
    if not path.exists():
        return completed
    with path.open("r") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            request_id = (int(payload["case_id"]), int(payload["rewrite_idx"]))
            if request_id in completed:
                raise ValueError(
                    f"Duplicate request identity {request_id} at line {line_no}."
                )
            prompts = payload.get("paraphrase_prompts")
            if not isinstance(prompts, list) or len(prompts) != target_count:
                raise ValueError(
                    f"Incomplete request identity {request_id} at line {line_no}: "
                    f"expected {target_count} paraphrases."
                )
            completed.add(request_id)
    return completed


def create_payload(
    record: Dict,
    paraphrase_prompts: List[str],
    audit: Dict,
    args: argparse.Namespace,
    generated_at: str,
) -> Dict:
    rewrite = record["requested_rewrite"]
    return {
        "case_id": int(record["case_id"]),
        "rewrite_idx": int(record["rewrite_idx"]),
        "subject": rewrite["subject"],
        "base_prompt": rewrite["prompt"],
        "target_true": rewrite["target_true"]["str"],
        "target_new": rewrite["target_new"]["str"],
        "paraphrase_prompts": paraphrase_prompts,
        "generator": {
            "generator_backend": "longcat_local",
            "model_name": args.model_name,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
            "max_new_tokens": args.max_new_tokens,
            "candidate_batch_size": args.candidate_batch_size,
            "max_rounds": args.max_rounds,
            "target_count": args.target_count,
            "seed": args.seed,
            "generated_at": generated_at,
        },
        "audit": audit,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    is_zsre = args.dataset == "zsre"
    is_mquake = args.dataset == "mquake"
    cases = (
        load_zsre_records(args.input_path)
        if is_zsre
        else load_counterfact_records(args.input_path)
    )
    if args.start_idx > len(cases):
        raise ValueError(f"start_idx={args.start_idx} exceeds {len(cases)} cases.")
    records = (
        list(iter_atomic_rewrites(cases, args.start_idx, args.limit))
        if is_mquake
        else cases[args.start_idx : args.start_idx + args.limit]
    )
    completed = set()
    if args.resume:
        completed = (
            load_completed_request_ids(args.output_path, args.target_count)
            if is_mquake
            else load_completed_case_ids(args.output_path)
        )
    elif args.output_path.exists():
        raise FileExistsError(
            f"{args.output_path} exists; enable --resume or choose another output."
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype_from_name(args.torch_dtype),
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    generated_at = datetime.now(timezone.utc).isoformat()

    progress = tqdm(records, desc=f"Generating {args.dataset} expressions")
    for record in progress:
        case_id = record.case_id if is_zsre else int(record["case_id"])
        request_id = (case_id, record["rewrite_idx"]) if is_mquake else case_id
        progress.set_postfix(request_id=request_id)
        if request_id in completed:
            continue
        generate = generate_zsre_for_record if is_zsre else generate_for_record
        prompts, audit = generate(model, tokenizer, record, args)
        if is_mquake and len(prompts) != args.target_count:
            raise RuntimeError(
                f"Generation stopped for request {request_id} with {len(prompts)}/"
                f"{args.target_count} accepted expressions; no incomplete row was saved."
            )
        create = (
            create_payload
            if is_mquake
            else create_zsre_record
            if is_zsre
            else create_record
        )
        append_jsonl(
            args.output_path, create(record, prompts, audit, args, generated_at)
        )
        progress.set_postfix(request_id=request_id, accepted=len(prompts))


if __name__ == "__main__":
    main()
