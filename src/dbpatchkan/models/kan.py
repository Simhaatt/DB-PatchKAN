"""Compact cubic B-spline Kolmogorov-Arnold Network layers.

The KAN is deliberately restricted to the named concept bottleneck.  Every
spline calculation is performed in float32, even when the surrounding patch
Transformer runs under FP16/BF16 autocast.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class BSplineKANLayer(nn.Module):
    """A KAN layer with one learnable cubic spline per input/output pair.

    Parameters
    ----------
    in_features, out_features:
        Input and output widths.
    grid_size:
        Number of equally spaced intervals in ``[-1, 1]``.
    spline_order:
        B-spline degree.  The project default is cubic (3).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        grid_size: int = 5,
        spline_order: int = 3,
        spline_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if in_features < 1 or out_features < 1:
            raise ValueError("KAN dimensions must be positive")
        if grid_size < 2:
            raise ValueError("grid_size must be at least 2")
        if spline_order < 1:
            raise ValueError("spline_order must be positive")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.grid_size = int(grid_size)
        self.spline_order = int(spline_order)
        self.n_basis = self.grid_size + self.spline_order

        step = 2.0 / self.grid_size
        knot_ids = torch.arange(
            -self.spline_order,
            self.grid_size + self.spline_order + 1,
            dtype=torch.float32,
        )
        self.register_buffer("knots", -1.0 + step * knot_ids, persistent=True)

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, self.n_basis)
        )
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer(
            "edge_mask", torch.ones(out_features, in_features), persistent=True
        )
        self.reset_parameters(spline_scale)

    def reset_parameters(self, spline_scale: float = 0.1) -> None:
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.normal_(
            self.spline_weight,
            mean=0.0,
            std=spline_scale / math.sqrt(max(1, self.in_features * self.n_basis)),
        )
        fan_in = self.in_features
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def b_splines(self, x: Tensor) -> Tensor:
        """Return spline bases with shape ``[..., in_features, n_basis]``."""

        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected {self.in_features} KAN features, got {x.shape[-1]}"
            )
        # tanh makes the layer well-defined for unnormalised concept estimates
        # while preserving the residual SiLU path outside the spline branch.
        x_grid = torch.tanh(x).unsqueeze(-1)
        knots = self.knots
        bases = ((x_grid >= knots[:-1]) & (x_grid < knots[1:])).to(x.dtype)

        # Cox-de Boor recursion.  Every iteration removes one trailing basis.
        for degree in range(1, self.spline_order + 1):
            new_count = bases.shape[-1] - 1
            left_den = knots[degree : degree + new_count] - knots[:new_count]
            right_den = (
                knots[degree + 1 : degree + 1 + new_count]
                - knots[1 : 1 + new_count]
            )
            left = (x_grid - knots[:new_count]) / left_den
            right = (knots[degree + 1 : degree + 1 + new_count] - x_grid) / right_den
            bases = left * bases[..., :new_count] + right * bases[..., 1 : new_count + 1]

        if bases.shape[-1] != self.n_basis:
            raise RuntimeError("Internal B-spline basis size mismatch")
        return bases.contiguous()

    def forward(self, x: Tensor) -> Tensor:
        # Explicitly disable mixed precision around KAN splines.  The returned
        # tensor intentionally remains float32 for numerical stability.
        device_type = x.device.type if x.device.type in {"cpu", "cuda"} else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            x32 = x.float()
            base_weight = self.base_weight.float() * self.edge_mask
            spline_weight = self.spline_weight.float() * self.edge_mask.unsqueeze(-1)
            base = F.linear(F.silu(x32), base_weight)
            basis = self.b_splines(x32)
            spline = torch.einsum(
                "...ib,oib->...o", basis, spline_weight
            )
            return base + spline + self.bias.float()

    def regularization_loss(
        self, *, sparsity_weight: float = 1.0, smoothness_weight: float = 1.0
    ) -> Tensor:
        """L1 sparsity plus second-difference spline smoothness."""

        masked_weights = self.spline_weight * self.edge_mask.unsqueeze(-1)
        sparsity = masked_weights.abs().mean()
        if self.n_basis >= 3:
            second_difference = (
                masked_weights[..., 2:]
                - 2 * masked_weights[..., 1:-1]
                + masked_weights[..., :-2]
            )
            smoothness = second_difference.square().mean()
        else:  # pragma: no cover - guarded by constructor defaults
            smoothness = self.spline_weight.new_zeros(())
        return sparsity_weight * sparsity + smoothness_weight * smoothness

    @torch.no_grad()
    def prune_edges(self, threshold: float) -> int:
        """Permanently mask weak edges while retaining one input per output."""

        if threshold < 0:
            raise ValueError("threshold must be non-negative")
        strength = self.base_weight.abs() + self.spline_weight.abs().mean(dim=-1)
        keep = strength >= threshold
        strongest = strength.argmax(dim=1)
        keep[torch.arange(self.out_features, device=keep.device), strongest] = True
        keep &= self.edge_mask.bool()
        removed = int(self.edge_mask.sum().item() - keep.sum().item())
        self.edge_mask.copy_(keep.to(self.edge_mask.dtype))
        return removed


class ConceptKAN(nn.Module):
    """Small KAN fusion head operating on at most twenty named concepts."""

    def __init__(
        self,
        dimensions: Sequence[int],
        *,
        grid_size: int = 5,
        spline_order: int = 3,
    ) -> None:
        super().__init__()
        dims = tuple(int(value) for value in dimensions)
        if len(dims) < 2:
            raise ValueError("ConceptKAN needs at least input and output dimensions")
        if dims[0] > 20:
            raise ValueError("The explainable concept bottleneck is capped at 20 inputs")
        self.dimensions = dims
        self.layers = nn.ModuleList(
            BSplineKANLayer(
                dims[i],
                dims[i + 1],
                grid_size=grid_size,
                spline_order=spline_order,
            )
            for i in range(len(dims) - 1)
        )

    def forward(self, concepts: Tensor) -> Tensor:
        output = concepts.float()
        for layer in self.layers:
            output = layer(output)
        return output

    def regularization_loss(self) -> Tensor:
        losses = [layer.regularization_loss() for layer in self.layers]
        return torch.stack(losses).mean() if losses else torch.zeros(())

    @torch.no_grad()
    def edge_strengths(self) -> list[Tensor]:
        """Return auditable mean absolute spline magnitudes per graph edge."""

        return [layer.spline_weight.abs().mean(dim=-1).detach().cpu() for layer in self.layers]

    @torch.no_grad()
    def prune_edges(self, threshold: float) -> int:
        """Mask weak spline graph edges before post-pruning fine-tuning."""

        return sum(layer.prune_edges(threshold) for layer in self.layers)


class ConceptMLP(nn.Module):
    """Parameter-comparable MLP challenger for the concept fusion head."""

    def __init__(
        self,
        dimensions: Sequence[int],
        *,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        dims = tuple(int(value) for value in dimensions)
        if len(dims) < 2:
            raise ValueError("ConceptMLP needs at least two dimensions")
        layers: list[nn.Module] = []
        for index, (width_in, width_out) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(width_in, width_out))
            if index < len(dims) - 2:
                layers.extend((nn.SiLU(), nn.Dropout(dropout)))
        self.dimensions = dims
        self.network = nn.Sequential(*layers)

    def forward(self, concepts: Tensor) -> Tensor:
        return self.network(concepts.float())

    def regularization_loss(self) -> Tensor:
        first = next(self.parameters())
        return first.new_zeros(())


def make_fusion_head(
    kind: str,
    n_concepts: int,
    n_outputs: int = 3,
    *,
    hidden_dimensions: Iterable[int] | None = None,
    grid_size: int = 5,
    spline_order: int = 3,
    dropout: float = 0.0,
) -> nn.Module:
    """Construct the primary KAN or the mandatory MLP challenger."""

    kind_normalized = kind.lower().replace("-", "_")
    if hidden_dimensions is None:
        hidden_dimensions = (16, 8) if kind_normalized == "kan" else (32, 16)
    dimensions = (n_concepts, *tuple(hidden_dimensions), n_outputs)
    if kind_normalized == "kan":
        return ConceptKAN(
            dimensions, grid_size=grid_size, spline_order=spline_order
        )
    if kind_normalized in {"mlp", "concept_mlp"}:
        return ConceptMLP(dimensions, dropout=dropout)
    raise ValueError(f"Unknown fusion head: {kind}")


__all__ = [
    "BSplineKANLayer",
    "ConceptKAN",
    "ConceptMLP",
    "make_fusion_head",
]
