"""Locate the last subject token in each expression."""

from transformers import AutoTokenizer


def get_last_subject_token_indices(
    tok: AutoTokenizer, prompt_templates: list[str], subject: str
) -> list[int]:
    positions = []
    for template in prompt_templates:
        if template.count("{}") != 1:
            raise ValueError(
                "Each expression must contain one subject placeholder '{}'."
            )
        prefix = template.split("{}")[0]
        positions.append(len(tok.encode(prefix + subject)) - 1)
    return positions
