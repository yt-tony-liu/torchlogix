"""Layer-wise adaptive discretization utilities for CLGN training."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torchlogix


class EntropyEMA:
    """Tracks an exponential moving average of scalar entropy values."""

    def __init__(self, beta: float):
        if not (0.0 <= beta < 1.0):
            raise ValueError(f"ema_beta must be in [0, 1), got {beta}.")
        self.beta = beta
        self.value: Optional[float] = None
        self.prev_value: Optional[float] = None

    def update(self, entropy: float) -> Tuple[float, Optional[float]]:
        self.prev_value = self.value
        if self.value is None:
            self.value = entropy
        else:
            self.value = self.beta * self.value + (1.0 - self.beta) * entropy
        return self.value, self.prev_value


@dataclass
class LayerDiscretizationState:
    layer_name: str
    ema: Optional[float] = None
    prev_ema: Optional[float] = None
    is_discretized: bool = False
    step_discretized: Optional[int] = None


class AdaptiveDiscretizer:
    """Discretize converged logic conv layers from shallow to deep."""

    def __init__(
        self,
        model: torch.nn.Module,
        check_interval: int,
        warmup_steps: int,
        ema_beta: float,
        entropy_threshold: float,
        delta_threshold: float,
        target: str = "conv",
    ):
        self.model = model
        self.check_interval = check_interval
        self.warmup_steps = warmup_steps
        self.entropy_threshold = entropy_threshold
        self.delta_threshold = delta_threshold
        self.target = target

        self._target_layers = self._collect_target_layers(model, target=target)
        self.layer_states = [
            LayerDiscretizationState(layer_name=name)
            for name, _ in self._target_layers
        ]
        self._ema_trackers = [EntropyEMA(beta=ema_beta) for _ in self._target_layers]

    @staticmethod
    def _missing_layer_api(layer: torch.nn.Module) -> List[str]:
        required = (
            "get_selection_entropy",
            "discretize_inplace",
            "freeze_discrete_params",
            "is_discretized",
        )
        return [name for name in required if not callable(getattr(layer, name, None))]

    @staticmethod
    def _layer_is_discretized(layer: torch.nn.Module) -> bool:
        fn = getattr(layer, "is_discretized", None)
        if callable(fn):
            return bool(fn())
        return bool(getattr(layer, "_force_discrete", False))

    @staticmethod
    def _collect_target_layers(
        model: torch.nn.Module,
        target: str,
    ) -> List[Tuple[str, torch.nn.Module]]:
        if target != "conv":
            raise ValueError(f"Unsupported target '{target}'. Only 'conv' is supported in MVP.")

        layers: List[Tuple[str, torch.nn.Module]] = []
        for name, module in model.named_modules():
            if isinstance(module, torchlogix.layers.LogicConv2d):
                layers.append((name or "<root>", module))
        return layers

    def maybe_discretize(self, global_step: int) -> Optional[str]:
        if not self._target_layers:
            return None

        if self.check_interval > 0 and (global_step % self.check_interval != 0):
            return None

        target_idx: Optional[int] = None
        for idx, ((layer_name, layer), state) in enumerate(zip(self._target_layers, self.layer_states)):
            if state.is_discretized or self._layer_is_discretized(layer):
                state.is_discretized = True
                continue

            missing_api = self._missing_layer_api(layer)
            if missing_api:
                print(
                    f"[AD] Warning: layer {layer_name} is missing AD API {missing_api}. "
                    "Skipping this layer."
                )
                state.is_discretized = True
                continue

            target_idx = idx
            break

        if target_idx is None:
            return None

        layer_name, layer = self._target_layers[target_idx]
        state = self.layer_states[target_idx]

        entropy_tensor = layer.get_selection_entropy()
        entropy = float(entropy_tensor.detach().item())

        ema, prev_ema = self._ema_trackers[target_idx].update(entropy)
        state.prev_ema = prev_ema
        state.ema = ema
        delta_ema = abs(ema - prev_ema) if prev_ema is not None else float("nan")

        print(
            f"[AD] Check layer {layer_name}: entropy={entropy:.6f}, "
            f"ema={ema:.6f}, delta_ema={delta_ema:.6f}"
        )

        converged = (
            global_step > self.warmup_steps
            and prev_ema is not None
            and ema < self.entropy_threshold
            and abs(ema - prev_ema) < self.delta_threshold
        )
        if not converged:
            return None

        layer.discretize_inplace()
        layer.freeze_discrete_params()
        state.is_discretized = True
        state.step_discretized = global_step
        return f"[AD] Discretized layer {layer_name} at step {global_step}, ema={ema:.6f}"
