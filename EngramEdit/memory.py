"""Exact-key additive memory and saved editing states."""

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch import nn

from .data import NGRAM_LENGTHS


class EngramEditEmbedding(nn.Module):
    """Add cumulative updates whenever their exact n-grams are activated."""

    def __init__(self, base_embedding):
        super().__init__()
        self.base_embedding = base_embedding
        self.key_lengths = NGRAM_LENGTHS
        self.updates = {length: {} for length in NGRAM_LENGTHS}
        self.processed_request_ids = set()
        self.skipped_request_ids = set()

    @property
    def config(self):
        return self.base_embedding.config

    def _context_with_history(
        self,
        input_ids: torch.Tensor,
        ngram_context: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if ngram_context is not None:
            return torch.cat(
                [ngram_context[..., -(self.config.emb_neighbor_num - 1) :], input_ids],
                dim=-1,
            )
        return input_ids

    def _iter_position_keys(
        self,
        context: torch.Tensor,
        seq_len: int,
        eos_token_id: Optional[int],
    ):
        full_len = context.size(-1)
        start_offset = full_len - seq_len
        for batch_idx in range(context.size(0)):
            row = context[batch_idx].tolist()
            for pos in range(seq_len):
                ctx_idx = start_offset + pos
                available = {}
                for length in self.key_lengths:
                    key_start = ctx_idx - length + 1
                    if key_start < 0:
                        continue
                    key_tokens = row[key_start : ctx_idx + 1]
                    if eos_token_id is not None and eos_token_id in key_tokens[:-1]:
                        continue
                    available[length] = tuple(key_tokens)
                yield batch_idx, pos, available

    def _activated_updates(
        self,
        input_ids: torch.Tensor,
        ngram_context: Optional[torch.Tensor],
        output: torch.Tensor,
    ) -> torch.Tensor:
        context = self._context_with_history(input_ids, ngram_context)
        residual = torch.zeros_like(output)
        for batch_idx, pos, keys in self._iter_position_keys(
            context, input_ids.size(-1), self.config.eos_token_id
        ):
            for length, key in keys.items():
                stored = self.updates[length].get(key)
                if stored is not None:
                    residual[batch_idx, pos, :] += stored.to(
                        output.device, dtype=output.dtype
                    )
        return residual

    def forward(
        self, input_ids: torch.Tensor, ngram_context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        base_output = self.base_embedding(input_ids, ngram_context=ngram_context)
        if not any(self.updates[length] for length in NGRAM_LENGTHS):
            return base_output

        residual = self._activated_updates(input_ids, ngram_context, base_output)
        return base_output + residual

    def add_updates(
        self,
        residuals: Dict[Tuple[int, Tuple[int, ...]], torch.Tensor],
        storage_dtype: torch.dtype,
    ) -> None:
        for (length, key), residual in residuals.items():
            stored_residual = residual.detach().to(dtype=storage_dtype).cpu()
            current = self.updates[length].get(key)
            if current is None:
                self.updates[length][key] = stored_residual.clone()
            else:
                self.updates[length][key] = current + stored_residual

    def snapshot(self, model_name):
        return {
            "model_name": model_name,
            "ngram_key_lengths": list(NGRAM_LENGTHS),
            "processed_request_ids": sorted(self.processed_request_ids),
            "skipped_request_ids": sorted(self.skipped_request_ids),
            "memory": self.updates,
        }

    def load_snapshot(self, payload):
        if not isinstance(payload, dict) or payload.get("ngram_key_lengths") != list(NGRAM_LENGTHS):
            raise ValueError("Expected an EngramEdit checkpoint with n-gram lengths [2, 3, 4].")
        memory = payload.get("memory")
        if not isinstance(memory, dict) or set(memory) != set(NGRAM_LENGTHS):
            raise ValueError("Checkpoint must contain memory tables for lengths 2, 3, and 4.")
        for length, table in memory.items():
            if not isinstance(table, dict):
                raise ValueError(f"Invalid memory table for n-gram length {length}.")
            for key, update in table.items():
                if (not isinstance(key, tuple) or len(key) != length
                        or any(type(token) is not int or not 0 <= token < self.config.vocab_size
                               for token in key)):
                    raise ValueError(f"Invalid n-gram key: {key}.")
                if (not isinstance(update, torch.Tensor) or not update.is_floating_point()
                        or update.shape != (self.config.hidden_size,)
                        or not torch.isfinite(update).all()):
                    raise ValueError(f"Invalid memory update for {key}; check the base-model dimensions.")
        self.updates = {
            length: {key: value.detach().cpu() for key, value in table.items()}
            for length, table in memory.items()
        }
        self.processed_request_ids = set(payload.get("processed_request_ids", []))
        self.skipped_request_ids = set(payload.get("skipped_request_ids", []))

    def save(self, directory, model_name):
        path = Path(directory) / "engramedit_state.pt"
        temporary = path.with_suffix(".pt.tmp")
        torch.save(self.snapshot(model_name), temporary)
        temporary.replace(path)


def prepare_engramedit_model(model, state_dir=None):
    """Install additive memory, optionally loading a saved editing state."""
    if not hasattr(model, "model") or not hasattr(model.model, "ngram_embeddings"):
        raise ValueError("Expected a LongCat model with model.ngram_embeddings.")
    embedding = model.model.ngram_embeddings
    if not isinstance(embedding, EngramEditEmbedding):
        embedding = EngramEditEmbedding(embedding)
    if state_dir is not None:
        path = Path(state_dir) / "engramedit_state.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        embedding.load_snapshot(payload)
    model.model.ngram_embeddings = embedding
    return model
