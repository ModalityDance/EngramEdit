"""EngramEdit: target computation, memory mapping, and joint updating."""

from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from util import nethook
from util.generate import generate_fast

from .data import (
    NGRAM_LENGTHS,
    base_prompt,
    get_edit_expressions,
    get_ngram_keys,
    load_expressions,
    load_frequency_cache,
    normalize_prompt,
    request_id,
    subject_positions,
)
from .engramedit_hparams import EngramEditHyperParams
from .memory import prepare_engramedit_model

LOSS_CHECK_INTERVAL = 2


def _get_input_device(model: AutoModelForCausalLM) -> torch.device:
    if hasattr(model, "get_input_embeddings"):
        embeddings = model.get_input_embeddings()
        if embeddings is not None and hasattr(embeddings, "weight"):
            return embeddings.weight.device
    return next(model.parameters()).device


def _compute_delta_kl_loss(
    init_log_probs: torch.Tensor,
    current_log_probs: torch.Tensor,
    kl_factor: float,
) -> torch.Tensor:
    """Measure the KL change induced by the shared edit delta."""
    return kl_factor * torch.nn.functional.kl_div(
        init_log_probs,
        current_log_probs,
        log_target=True,
        reduction="batchmean",
    )


def _format_output_tensor(output):
    if isinstance(output, tuple):
        output = output[0]
    return output


def generate_context_templates(model, tok):
    """Generate the five prefix contexts used for target computation."""
    prefixes = generate_fast(
        model,
        tok,
        ["The", "Therefore", "Because", "I", "You"],
        n_gen_per_prompt=1,
        max_out_len=10,
    )
    return [
        ["{}"],
        [text.replace("{", " ").replace("}", " ") + ". {}" for text in prefixes],
    ]


def _build_target_prompts(
    context_templates: List[List[str]],
    base_prompt: str,
    target_prefix: str,
    expressions: List[str],
) -> Tuple[List[str], List[str]]:
    source_prompts = [base_prompt]
    if expressions:
        source_prompts.extend(expressions)

    rewriting_prompts = []
    ngram_key_prompts = []
    seen_exact = set()
    seen_normalized = set()
    for source_prompt in source_prompts:
        prompt_template_group = [
            context.format(source_prompt)
            for group in context_templates
            for context in group
        ]
        for prompt_template in prompt_template_group:
            normalized = normalize_prompt(prompt_template)
            if not normalized:
                continue
            if prompt_template in seen_exact or normalized in seen_normalized:
                continue
            seen_exact.add(prompt_template)
            seen_normalized.add(normalized)
            rewriting_prompts.append(prompt_template + target_prefix)
            ngram_key_prompts.append(source_prompt + target_prefix)

    return rewriting_prompts, ngram_key_prompts


def compute_targets(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: EngramEditHyperParams,
    expressions: List[str],
    context_templates: List[List[str]],
) -> Dict[str, object]:
    if request["target_new"]["str"][0] != " ":
        request = deepcopy(request)
        request["target_new"]["str"] = " " + request["target_new"]["str"]

    prompt = base_prompt(request)

    lm_w, ln_f = (
        nethook.get_module(model, "lm_head").weight.T,
        nethook.get_module(model, "model.norm"),
    )
    try:
        lm_b = nethook.get_parameter(model, "lm_head.bias")
    except LookupError:
        lm_b = next(model.parameters()).new_zeros(model.config.vocab_size)

    input_device = _get_input_device(model)
    target_ids = tok(request["target_new"]["str"], return_tensors="pt")["input_ids"][
        0
    ].to(input_device)
    if target_ids[0] in {tok.bos_token_id, tok.unk_token_id}:
        target_ids = target_ids[1:]

    rewriting_prompts, ngram_key_prompts = _build_target_prompts(
        context_templates,
        prompt,
        tok.decode(target_ids[:-1].tolist()),
        expressions=expressions,
    )
    if not rewriting_prompts:
        raise ValueError(
            f"No EngramEdit rewriting prompts built for subject={request['subject']!r}."
        )
    kl_prompts = ["{} is a"]
    all_prompts = rewriting_prompts + kl_prompts

    formatted_prompts = [prompt.format(request["subject"]) for prompt in all_prompts]
    input_tok = tok(formatted_prompts, return_tensors="pt", padding=True).to(
        input_device
    )
    rewriting_targets = torch.tensor(-100, device=input_device).repeat(
        len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
    )
    for i in range(len(rewriting_prompts)):
        ex_len = input_tok["attention_mask"][i].sum()
        rewriting_targets[i, ex_len - len(target_ids) : ex_len] = target_ids

    lookup_idxs = subject_positions(
        tok,
        all_prompts,
        request["subject"],
    )
    delta = torch.zeros(
        (model.config.hidden_size,), requires_grad=True, device=input_device
    )
    target_inits = None
    kl_distr_init = None

    def target_norm() -> torch.Tensor:
        return target_inits[0].norm()

    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_inits
        if cur_layer != "model.ngram_embeddings":
            return cur_out

        cur_out = _format_output_tensor(cur_out)
        if target_inits is None:
            target_inits = torch.stack(
                [
                    cur_out[batch_idx, lookup_idxs[batch_idx]].detach().clone()
                    for batch_idx in range(len(rewriting_prompts))
                ],
                dim=0,
            )

        for batch_idx, idx in enumerate(lookup_idxs):
            cur_out[batch_idx, idx, :] = cur_out[batch_idx, idx, :] + delta.to(
                cur_out.device, dtype=cur_out.dtype
            )
        return cur_out

    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)

    for step in range(hparams.v_num_grad_steps):
        opt.zero_grad()

        with nethook.TraceDict(
            module=model,
            layers=[f"model.layers.{hparams.v_loss_layer}", "model.ngram_embeddings"],
            edit_output=edit_output_fn,
        ) as tr:
            model_out = model(**input_tok)

        output = _format_output_tensor(
            tr[f"model.layers.{hparams.v_loss_layer}"].output
        )
        if output.shape[1] != rewriting_targets.shape[1]:
            output = output.transpose(0, 1)
        full_repr = output[: len(rewriting_prompts)]

        log_probs = torch.log_softmax(
            ln_f(full_repr) @ lm_w.to(full_repr.device) + lm_b.to(full_repr.device),
            dim=2,
        )
        loss = torch.gather(
            log_probs,
            2,
            torch.where(rewriting_targets != -100, rewriting_targets, 0)
            .unsqueeze(2)
            .to(log_probs.device),
        ).squeeze(2)
        mask = (rewriting_targets != -100).float().to(loss.device)
        nll_loss_each = -(loss * mask).sum(1) / target_ids.size(0)
        nll_loss = nll_loss_each.mean()

        final_logits = model_out.logits
        kl_start = len(rewriting_prompts)
        kl_logits = torch.stack(
            [
                final_logits[kl_start + i, idx, :]
                for i, idx in enumerate(lookup_idxs[kl_start:])
            ],
            dim=0,
        )
        kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
        if kl_distr_init is None:
            kl_distr_init = kl_log_probs.detach().clone()
        kl_loss = _compute_delta_kl_loss(
            kl_distr_init,
            kl_log_probs,
            hparams.kl_factor,
        )
        weight_decay = hparams.lambda_norm * (torch.norm(delta) / target_norm() ** 2)
        total_loss = (
            nll_loss + kl_loss.to(nll_loss.device) + weight_decay.to(nll_loss.device)
        )

        should_check_loss = (
            step == hparams.v_num_grad_steps - 1 or step % LOSS_CHECK_INTERVAL == 0
        )
        if should_check_loss and total_loss < 5e-2:
            break
        if step == hparams.v_num_grad_steps - 1:
            break

        total_loss.backward()
        opt.step()

        if should_check_loss:
            with torch.no_grad():
                max_norm = hparams.clamp_norm_factor * target_norm()
                if delta.norm() > max_norm:
                    delta[...] = delta * max_norm / delta.norm()

    delta_out = delta.detach()
    target_outputs = target_inits + delta_out.to(
        target_inits.device, dtype=target_inits.dtype
    )

    result = {
        "current_outputs": target_inits.detach(),
        "target_outputs": target_outputs.detach(),
        "ngram_key_prompts": ngram_key_prompts,
    }
    return result


def build_memory_mapping(prompt_key_sets, device):
    """One row per target and one column per distinct n-gram in the batch."""
    unique_keys = list(
        dict.fromkeys(
            (length, keys[length])
            for keys in prompt_key_sets
            for length in NGRAM_LENGTHS
            if length in keys
        )
    )
    if not unique_keys:
        raise ValueError("No n-gram keys selected for this batch.")
    columns = {key: index for index, key in enumerate(unique_keys)}
    mapping = torch.zeros(
        (len(prompt_key_sets), len(unique_keys)), device=device, dtype=torch.float32
    )
    for row, keys in enumerate(prompt_key_sets):
        for length in NGRAM_LENGTHS:
            if length in keys:
                mapping[row, columns[(length, keys[length])]] = 1.0
    return unique_keys, mapping


def _frequency_weight(
    memory_key: Tuple[int, Tuple[int, ...]],
    hparams: EngramEditHyperParams,
    cache: Dict[str, Dict[int, Dict]],
) -> float:
    length, key = memory_key
    counts = cache["counts_by_length"].get(length, {})
    count = counts.get(tuple(key))
    if count is None:
        return float(hparams.frequency_missing_weight)

    percentile = (
        cache["count_percentiles_by_length"].get(length, {}).get(int(count), 0.0)
    )
    raw_weight = 1.0 + float(hparams.frequency_gamma) * (
        float(percentile) ** float(hparams.frequency_power)
    )
    return min(raw_weight, float(hparams.frequency_weight_cap))


def compute_regularization_coefficients(keys, hparams, frequency_cache, device):
    weights = []
    for key in keys:
        length_weight = float(hparams.length_weights[str(key[0])])
        frequency_weight = _frequency_weight(key, hparams, frequency_cache)
        weight = min(
            length_weight + frequency_weight - 1.0, float(hparams.reuse_weight_cap)
        )
        weights.append(weight)
    weights = torch.tensor(weights, device=device, dtype=torch.float32)
    diagonal = hparams.lambda_ridge + hparams.lambda_reuse * weights
    if not torch.isfinite(diagonal).all() or not torch.all(diagonal > 0):
        raise ValueError(
            "Every embedding must have a finite, positive regularization weight."
        )
    return diagonal


def solve_joint_updates(mapping, current, targets, diagonal):
    """Solve (A.T A + Lambda) U = A.T (target - current) in FP32."""
    if (
        current.ndim != 2
        or targets.shape != current.shape
        or current.shape[0] != mapping.shape[0]
    ):
        raise ValueError(
            "Each mapping row must have one current and target representation."
        )
    # Current representations already contain updates from preceding edit batches.
    difference = targets.detach().to(dtype=torch.float32) - current.detach().to(
        dtype=torch.float32
    )
    system = mapping.T @ mapping + torch.diag(diagonal)
    return torch.linalg.solve(system, mapping.T @ difference)


def apply_engramedit_to_model(
    model,
    tok,
    requests,
    hparams,
    *,
    paraphrase_path,
    ngram_frequency_cache,
    context_templates,
    paraphrase_append_count=4,
    run_dir=None,
):
    """Compute targets, map their n-grams, and jointly apply one edit batch."""
    if paraphrase_append_count <= 0:
        raise ValueError("The main method requires generated expressions.")
    prepare_engramedit_model(model)
    memory = model.model.ngram_embeddings
    expressions = load_expressions(Path(paraphrase_path).resolve())
    frequency = load_frequency_cache(Path(ngram_frequency_cache).resolve())
    current_rows, target_rows, key_sets, edited_ids = [], [], [], []

    # 1. Compute targets across each fact's expressions and map their n-grams.
    for request in requests:
        key = request_id(request)
        if key in memory.processed_request_ids or key in memory.skipped_request_ids:
            continue
        generated = get_edit_expressions(request, expressions, paraphrase_append_count)
        result = compute_targets(
            model, tok, request, hparams, generated, context_templates
        )
        keys = [
            get_ngram_keys(tok, prompt, request["subject"])
            for prompt in result["ngram_key_prompts"]
        ]
        if not any(keys):
            memory.skipped_request_ids.add(key)
            print(
                f"[EngramEdit] Skipping request {key}: no complete n-gram at the subject position."
            )
            continue
        current_rows.append(result["current_outputs"])
        target_rows.append(result["target_outputs"])
        key_sets.extend(keys)
        edited_ids.append(key)

    # 2. Construct the batch-level mapping and reuse-based regularization.
    if edited_ids:
        current, targets = torch.cat(current_rows), torch.cat(target_rows)
        keys, mapping = build_memory_mapping(key_sets, current.device)
        diagonal = compute_regularization_coefficients(
            keys, hparams, frequency, current.device
        )

        # 3. Jointly solve and accumulate updates to the selected memory embeddings.
        updates = solve_joint_updates(mapping, current, targets, diagonal)
        memory.add_updates(
            dict(zip(keys, updates)), storage_dtype=next(model.parameters()).dtype
        )
        memory.processed_request_ids.update(edited_ids)
        print(
            f"[EngramEdit] {len(edited_ids)} edits, {len(key_sets)} target rows, {len(keys)} n-grams"
        )

    if run_dir is not None:
        memory.save(run_dir, model.config._name_or_path)
    return model
