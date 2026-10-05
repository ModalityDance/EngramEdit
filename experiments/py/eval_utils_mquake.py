"""Official MQuAKE multi-hop generation evaluation."""

import re
import typing
from functools import lru_cache
from pathlib import Path

import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForCausalLM, AutoTokenizer

from dsets import AttributeSnippets


PROMPT_DIR = Path(__file__).resolve().parents[2] / "dsets" / "mquake_prompts"


def _get_input_device(model: AutoModelForCausalLM) -> torch.device:
    if hasattr(model, "get_input_embeddings"):
        embeddings = model.get_input_embeddings()
        if embeddings is not None and hasattr(embeddings, "weight"):
            return embeddings.weight.device
    return next(model.parameters()).device


@lru_cache(maxsize=2)
def _load_prompt(filename: str) -> str:
    with open(PROMPT_DIR / filename, "r") as f:
        return f.read().rstrip()


def _normalize_answer(text: str) -> str:
    text = " ".join(text.strip().casefold().split())
    return text.strip(" \t\r\n.,;:!?\"'")


def _extract_standard_answer(generation: str) -> str:
    first_line = next(
        (line.strip() for line in generation.splitlines() if line.strip()),
        "",
    )
    return re.sub(r"^(?:a|answer)\s*:\s*", "", first_line, flags=re.IGNORECASE)


def _extract_cot_answer(generation: str) -> str:
    matches = re.findall(
        r"(?:^|\n)\s*Answer\s*:\s*([^\n]+)",
        generation,
        flags=re.IGNORECASE,
    )
    return matches[-1].strip() if matches else ""


def _matches_answer(prediction: str, answers: typing.Iterable[str]) -> bool:
    normalized_prediction = _normalize_answer(prediction)
    return bool(normalized_prediction) and any(
        normalized_prediction == _normalize_answer(answer) for answer in answers
    )


def _build_prompts(questions: typing.List[str], use_cot: bool) -> typing.List[str]:
    if use_cot:
        demonstrations = _load_prompt("multihop-cot-prompts.txt")
        return [
            f"{demonstrations}\n\nQuestion: {question}\nThoughts:"
            for question in questions
        ]

    demonstrations = _load_prompt("multihop-prompts.txt")
    return [f"{demonstrations}\nQ: {question} A:" for question in questions]


def _generate_answers(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompts: typing.List[str],
    *,
    use_cot: bool,
) -> typing.Tuple[typing.List[str], typing.List[str]]:
    input_device = _get_input_device(model)
    original_padding_side = tok.padding_side
    tok.padding_side = "left"
    try:
        inputs = tok(prompts, padding=True, return_tensors="pt").to(input_device)
    finally:
        tok.padding_side = original_padding_side

    generation_kwargs = {
        "max_new_tokens": 128 if use_cot else 32,
        "do_sample": False,
        "pad_token_id": tok.pad_token_id,
    }
    if tok.eos_token_id is not None:
        generation_kwargs["eos_token_id"] = tok.eos_token_id

    with torch.no_grad():
        output_ids = model.generate(**inputs, **generation_kwargs)

    continuations = output_ids[:, inputs["input_ids"].shape[1] :]
    generations = tok.batch_decode(continuations, skip_special_tokens=True)
    extractor = _extract_cot_answer if use_cot else _extract_standard_answer
    answers = [extractor(generation) for generation in generations]
    return generations, answers


def compute_rewrite_quality_mquake(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    record: typing.Dict,
    snips: AttributeSnippets,
    vec: TfidfVectorizer,
    evaluation_mode: str = "post",
) -> typing.Dict:
    """Evaluate standard and CoT MQuAKE multi-hop accuracy."""
    if evaluation_mode not in {"pre", "post"}:
        raise ValueError(
            f"evaluation_mode must be 'pre' or 'post', got {evaluation_mode!r}."
        )

    questions = record["paraphrase_prompts"]
    if evaluation_mode == "pre":
        answers = [record["answer"], *record.get("answer_alias", [])]
    else:
        answers = [record["new_answer"], *record.get("new_answer_alias", [])]

    standard_generations, standard_answers = _generate_answers(
        model,
        tok,
        _build_prompts(questions, use_cot=False),
        use_cot=False,
    )
    cot_generations, cot_answers = _generate_answers(
        model,
        tok,
        _build_prompts(questions, use_cot=True),
        use_cot=True,
    )

    standard_question_correct = [
        _matches_answer(answer, answers) for answer in standard_answers
    ]
    cot_question_correct = [_matches_answer(answer, answers) for answer in cot_answers]

    return {
        "rewrite_prompts_correct": [any(standard_question_correct)],
        "multihop_cot_prompts_correct": [any(cot_question_correct)],
        "multihop_question_correct": standard_question_correct,
        "multihop_cot_question_correct": cot_question_correct,
        "multihop_answers": standard_answers,
        "multihop_cot_answers": cot_answers,
        "multihop_generations": standard_generations,
        "multihop_cot_generations": cot_generations,
        "evaluation_mode": evaluation_mode,
        "hop_count": len(record["new_single_hops"]),
        "num_atomic_edits": len(record["requested_rewrite"]),
    }
