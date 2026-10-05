"""Hyperparameters for the full EngramEdit method."""

import json
from dataclasses import dataclass, field
from typing import Dict

from util.hparams import HyperParams


@dataclass
class EngramEditHyperParams(HyperParams):
    v_num_grad_steps: int
    v_lr: float
    v_loss_layer: int
    lambda_norm: float
    clamp_norm_factor: float
    kl_factor: float
    lambda_ridge: float = 0.01
    lambda_reuse: float = 0.05
    length_weights: Dict[str, float] = field(
        default_factory=lambda: {"2": 8.0, "3": 2.0, "4": 1.0}
    )
    frequency_gamma: float = 9.0
    frequency_power: float = 1.0
    frequency_weight_cap: float = 10.0
    reuse_weight_cap: float = 64.0
    frequency_missing_weight: float = 1.0

    def __post_init__(self):
        if self.lambda_ridge < 0 or self.lambda_reuse < 0:
            raise ValueError("Regularization coefficients must be non-negative.")

    @classmethod
    def from_json(cls, path):
        with open(path, encoding="utf-8") as stream:
            return cls(**json.load(stream))
