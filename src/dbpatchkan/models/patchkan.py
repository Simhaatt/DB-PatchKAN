"""DB-PatchKAN: dual-stream patch Transformer with concept-KAN fusion."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .kan import make_fusion_head


DEFAULT_CONCEPT_NAMES = (
    "bsi_rate",
    "bsi_thermal",
    "bsi_soc",
    "bsi_cycle",
    "bsi_transient",
    "bsi_interaction",
    "behavior_share_eco",
    "behavior_share_balanced",
    "behavior_share_aggressive",
    "calendar_age_days",
    "odometer_km",
    "cumulative_stress_dose",
    "last_known_soh",
    "accel_pos_rms",
    "jerk_rms",
    "energy_kwh_per_100km",
    "pack_temp_max_c",
    "ah_throughput",
    "regen_fraction",
    "stop_fraction",
)


@dataclass(slots=True)
class DBPatchKANConfig:
    """Serializable architecture configuration.

    Defaults reproduce the planned high-resolution model.  Smaller values can
    be used for CPU smoke tests without changing the model interfaces.
    """

    driver_channels: int = 13
    battery_channels: int = 11
    segment_length: int = 256
    patch_length: int = 16
    patch_stride: int = 8
    d_model: int = 128
    encoder_layers: int = 3
    n_heads: int = 4
    ff_dim: int = 256
    dropout: float = 0.10
    behavior_classes: int = 3
    num_drivers: int = 7
    concept_names: tuple[str, ...] = DEFAULT_CONCEPT_NAMES
    fusion_head: str = "kan"
    input_modality: str = "both"
    cross_attention_mode: str = "bidirectional"
    prediction_input: str = "concepts"
    kan_hidden: tuple[int, ...] = (16, 8)
    mlp_hidden: tuple[int, ...] = (32, 16)
    kan_grid_size: int = 5
    kan_spline_order: int = 3
    gate_noise_std: float = 0.5
    gate_initial_value: float = 1.0
    protected_driver_channels: tuple[int, ...] = (0, 9, 10, 11, 12)
    protected_battery_channels: tuple[int, ...] = (0, 1, 2, 3, 7, 8, 9, 10)
    calibration_domains: tuple[str, ...] = ("nissan", "capacity")

    def __post_init__(self) -> None:
        if self.driver_channels < 1 or self.battery_channels < 1:
            raise ValueError("Both streams need at least one channel")
        if self.segment_length < self.patch_length:
            raise ValueError("segment_length must be at least patch_length")
        if self.patch_length < 1 or self.patch_stride < 1:
            raise ValueError("Patch length and stride must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if not 1 <= len(self.concept_names) <= 20:
            raise ValueError("Concept bottleneck must contain between 1 and 20 concepts")
        if len(set(self.concept_names)) != len(self.concept_names):
            raise ValueError("Concept names must be unique")
        for index in self.protected_driver_channels:
            if not 0 <= index < self.driver_channels:
                raise ValueError(f"Invalid protected driver channel index: {index}")
        for index in self.protected_battery_channels:
            if not 0 <= index < self.battery_channels:
                raise ValueError(f"Invalid protected battery channel index: {index}")
        if not self.calibration_domains:
            raise ValueError("At least one calibration domain is required")
        if self.input_modality not in {"both", "driver", "battery"}:
            raise ValueError("input_modality must be both, driver, or battery")
        if self.cross_attention_mode not in {"concatenation", "one_way", "bidirectional"}:
            raise ValueError(
                "cross_attention_mode must be concatenation, one_way, or bidirectional"
            )
        if self.prediction_input not in {"concepts", "latent"}:
            raise ValueError("prediction_input must be concepts or latent")
        if self.prediction_input == "latent" and self.fusion_head.lower() == "kan":
            raise ValueError("A latent KAN is intentionally disallowed; use the concept bottleneck")

    @property
    def n_concepts(self) -> int:
        return len(self.concept_names)

    @property
    def n_patches(self) -> int:
        return 1 + (self.segment_length - self.patch_length) // self.patch_stride

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Public alias requested by the training and notebook interfaces.
ModelConfig = DBPatchKANConfig
model_config = DBPatchKANConfig


class StochasticChannelGate(nn.Module):
    """Stochastic feature gates with an expected L0 penalty.

    Protected physical/mask channels are hard-set to one and excluded from the
    sparsity penalty.  Evaluation uses deterministic clipped gate means.
    """

    def __init__(
        self,
        channels: int,
        *,
        protected_indices: tuple[int, ...] = (),
        noise_std: float = 0.5,
        initial_value: float = 1.0,
    ) -> None:
        super().__init__()
        if noise_std <= 0:
            raise ValueError("noise_std must be positive")
        self.channels = int(channels)
        self.noise_std = float(noise_std)
        self.mu = nn.Parameter(torch.full((channels,), float(initial_value)))
        protected = torch.zeros(channels, dtype=torch.bool)
        if protected_indices:
            protected[list(protected_indices)] = True
        self.register_buffer("protected_mask", protected, persistent=True)
        self.register_buffer(
            "pruned_mask", torch.zeros(channels, dtype=torch.bool), persistent=True
        )

    def probabilities(self) -> Tensor:
        normal_cdf = 0.5 * (
            1.0 + torch.erf(self.mu / (self.noise_std * math.sqrt(2.0)))
        )
        probabilities = torch.where(
            self.protected_mask, torch.ones_like(normal_cdf), normal_cdf
        )
        return torch.where(self.pruned_mask, torch.zeros_like(probabilities), probabilities)

    def deterministic_gates(self) -> Tensor:
        gates = torch.where(
            self.protected_mask, torch.ones_like(self.mu), self.mu.clamp(0.0, 1.0)
        )
        return torch.where(self.pruned_mask, torch.zeros_like(gates), gates)

    def sample_gates(self) -> Tensor:
        if self.training:
            values = (self.mu + self.noise_std * torch.randn_like(self.mu)).clamp(0.0, 1.0)
        else:
            values = self.deterministic_gates()
        values = torch.where(self.protected_mask, torch.ones_like(values), values)
        return torch.where(self.pruned_mask, torch.zeros_like(values), values)

    def forward(self, values: Tensor) -> tuple[Tensor, Tensor]:
        if values.shape[-1] != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {values.shape[-1]}")
        gates = self.sample_gates()
        return values * gates.view(1, 1, -1), gates

    def l0_penalty(self) -> Tensor:
        optional = ~(self.protected_mask | self.pruned_mask)
        if not torch.any(optional):
            return self.mu.new_zeros(())
        return self.probabilities()[optional].mean()

    @torch.no_grad()
    def prune(self, indices: tuple[int, ...] | list[int]) -> None:
        """Apply externally validated channel pruning; protected indices fail."""

        if not indices:
            return
        selected = torch.as_tensor(indices, device=self.pruned_mask.device, dtype=torch.long)
        if torch.any(selected < 0) or torch.any(selected >= self.channels):
            raise ValueError("Channel prune index is out of range")
        if torch.any(self.protected_mask[selected]):
            raise ValueError("Protected channels cannot be pruned")
        self.pruned_mask[selected] = True


def _normalise_mask(values: Tensor, mask: Tensor | None) -> tuple[Tensor, Tensor]:
    finite = torch.isfinite(values)
    if mask is None:
        valid = finite
    else:
        mask = mask.to(device=values.device, dtype=torch.bool)
        if mask.ndim == 2:
            mask = mask.unsqueeze(-1).expand_as(values)
        if mask.shape != values.shape:
            raise ValueError(
                f"Mask shape {tuple(mask.shape)} does not match input {tuple(values.shape)}"
            )
        valid = finite & mask
    return torch.where(valid, values, torch.zeros_like(values)), valid


def _safe_padding_mask(valid_tokens: Tensor) -> Tensor:
    """Create an MHA padding mask without an all-masked attention row."""

    padding = ~valid_tokens
    all_invalid = padding.all(dim=1)
    if torch.any(all_invalid):
        padding = padding.clone()
        padding[all_invalid, 0] = False
    return padding


class TemporalPatchEmbedding(nn.Module):
    def __init__(
        self,
        channels: int,
        patch_length: int,
        patch_stride: int,
        d_model: int,
        max_patches: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.patch_length = patch_length
        self.patch_stride = patch_stride
        self.projection = nn.Linear(channels * patch_length, d_model)
        self.position = nn.Parameter(torch.zeros(1, max_patches, d_model))
        self.dropout = nn.Dropout(dropout)
        nn.init.trunc_normal_(self.position, std=0.02)

    def patchify(self, values: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        if values.ndim != 3:
            raise ValueError("Sequence input must have shape [batch, time, channels]")
        if values.shape[-1] != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {values.shape[-1]}")
        if values.shape[1] < self.patch_length:
            padding = self.patch_length - values.shape[1]
            values = F.pad(values, (0, 0, 0, padding))
            valid = F.pad(valid, (0, 0, 0, padding), value=False)
        # unfold returns [B, P, C, L]; transpose to the public [B, P, L, C].
        patches = values.unfold(1, self.patch_length, self.patch_stride).permute(0, 1, 3, 2)
        patch_valid = valid.unfold(1, self.patch_length, self.patch_stride).permute(0, 1, 3, 2)
        return patches.contiguous(), patch_valid.contiguous()

    def _positions(self, n_patches: int) -> Tensor:
        if n_patches <= self.position.shape[1]:
            return self.position[:, :n_patches]
        return F.interpolate(
            self.position.transpose(1, 2),
            size=n_patches,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

    def forward(self, patches: Tensor) -> Tensor:
        batch, n_patches, length, channels = patches.shape
        if length != self.patch_length or channels != self.channels:
            raise ValueError("Unexpected patch shape")
        flat = patches.reshape(batch, n_patches, length * channels)
        return self.dropout(self.projection(flat) + self._positions(n_patches))


class PatchStreamEncoder(nn.Module):
    def __init__(
        self,
        *,
        channels: int,
        protected_channels: tuple[int, ...],
        config: DBPatchKANConfig,
    ) -> None:
        super().__init__()
        self.gate = StochasticChannelGate(
            channels,
            protected_indices=protected_channels,
            noise_std=config.gate_noise_std,
            initial_value=config.gate_initial_value,
        )
        self.patch_embedding = TemporalPatchEmbedding(
            channels,
            config.patch_length,
            config.patch_stride,
            config.d_model,
            config.n_patches,
            config.dropout,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=config.encoder_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(config.d_model)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(
        self,
        values: Tensor,
        mask: Tensor | None = None,
        *,
        masked_patches: Tensor | None = None,
    ) -> dict[str, Tensor]:
        values, valid = _normalise_mask(values, mask)
        gated, sampled_gates = self.gate(values)
        patches, element_valid = self.patch_embedding.patchify(gated, valid)
        target_patches, _ = self.patch_embedding.patchify(values, valid)
        tokens = self.patch_embedding(patches)
        token_valid = element_valid.any(dim=(-1, -2))
        if masked_patches is not None:
            if masked_patches.shape != token_valid.shape:
                raise ValueError("masked_patches does not match patch layout")
            tokens = torch.where(
                masked_patches.unsqueeze(-1), self.mask_token.expand_as(tokens), tokens
            )
        encoded = self.encoder(
            tokens, src_key_padding_mask=_safe_padding_mask(token_valid)
        )
        encoded = self.norm(encoded)
        encoded = encoded * token_valid.unsqueeze(-1).to(encoded.dtype)
        return {
            "tokens": encoded,
            "token_valid": token_valid,
            "target_patches": target_patches,
            "element_valid": element_valid,
            "sampled_gates": sampled_gates,
            "gate_probabilities": self.gate.probabilities(),
        }

    def token_validity(self, values: Tensor, mask: Tensor | None = None) -> Tensor:
        """Return patch validity without running the Transformer encoder."""

        values, valid = _normalise_mask(values, mask)
        _, element_valid = self.patch_embedding.patchify(values, valid)
        return element_valid.any(dim=(-1, -2))


class AttentionPool(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.Tanh(), nn.Linear(d_model // 2, 1)
        )

    def forward(self, tokens: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        logits = self.score(tokens).squeeze(-1).masked_fill(~valid, -1.0e4)
        weights = torch.softmax(logits, dim=-1) * valid.to(logits.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        return torch.einsum("bp,bpd->bd", weights, tokens), weights


class DomainCalibration(nn.Module):
    """Dataset-specific output scale/bias without pooling label definitions."""

    def __init__(self, current_soh_bias: float) -> None:
        super().__init__()
        inverse_softplus_one = math.log(math.e - 1.0)
        self.scale_unconstrained = nn.Parameter(
            torch.full((3,), inverse_softplus_one, dtype=torch.float32)
        )
        self.bias = nn.Parameter(
            torch.tensor([current_soh_bias, -3.0, -3.0], dtype=torch.float32)
        )

    def forward(self, raw: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        calibrated = raw.float() * F.softplus(self.scale_unconstrained) + self.bias
        return calibrated[:, 0], F.softplus(calibrated[:, 1]), F.softplus(calibrated[:, 2])


class DBPatchKAN(nn.Module):
    """Driver–Battery dual-stream Patch Transformer with concept fusion."""

    def __init__(self, config: DBPatchKANConfig | None = None) -> None:
        super().__init__()
        self.config = config or DBPatchKANConfig()
        cfg = self.config
        self.driver_encoder = PatchStreamEncoder(
            channels=cfg.driver_channels,
            protected_channels=cfg.protected_driver_channels,
            config=cfg,
        )
        self.battery_encoder = PatchStreamEncoder(
            channels=cfg.battery_channels,
            protected_channels=cfg.protected_battery_channels,
            config=cfg,
        )
        self.driver_to_battery = nn.MultiheadAttention(
            cfg.d_model, cfg.n_heads, cfg.dropout, batch_first=True
        )
        self.battery_to_driver = nn.MultiheadAttention(
            cfg.d_model, cfg.n_heads, cfg.dropout, batch_first=True
        )
        self.driver_cross_norm = nn.LayerNorm(cfg.d_model)
        self.battery_cross_norm = nn.LayerNorm(cfg.d_model)
        self.driver_pool = AttentionPool(cfg.d_model)
        self.battery_pool = AttentionPool(cfg.d_model)
        # Hierarchical runs first pool patches within each fixed segment, then
        # pool all overlapping segments from the same trip.  Segment-level
        # training remains available through ``forward`` unchanged.
        self.trip_pool = AttentionPool(cfg.d_model)
        self.trip_driver_pool = AttentionPool(cfg.d_model)
        self.fusion_projection = nn.Sequential(
            nn.Linear(2 * cfg.d_model, cfg.d_model), nn.GELU(), nn.LayerNorm(cfg.d_model)
        )
        self.concept_predictor = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.n_concepts),
        )
        hidden = cfg.kan_hidden if cfg.fusion_head.lower() == "kan" else cfg.mlp_hidden
        fusion_inputs = cfg.n_concepts if cfg.prediction_input == "concepts" else cfg.d_model
        self.fusion_head = make_fusion_head(
            cfg.fusion_head,
            fusion_inputs,
            3,
            hidden_dimensions=hidden,
            grid_size=cfg.kan_grid_size,
            spline_order=cfg.kan_spline_order,
            dropout=cfg.dropout,
        )
        self.behavior_head = nn.Sequential(
            nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.behavior_classes)
        )
        self.driver_id_head = nn.Sequential(
            nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.num_drivers)
        )
        self.calibration = nn.ModuleDict(
            {
                name: DomainCalibration(95.0 if name == "nissan" else 100.0)
                for name in cfg.calibration_domains
            }
        )
        self.driver_reconstruction = nn.Linear(
            cfg.d_model, cfg.patch_length * cfg.driver_channels
        )
        self.battery_reconstruction = nn.Linear(
            cfg.d_model, cfg.patch_length * cfg.battery_channels
        )
        self.register_buffer("concept_center", torch.zeros(cfg.n_concepts), persistent=True)
        self.register_buffer("concept_scale", torch.ones(cfg.n_concepts), persistent=True)
        # These statistics are fitted on the training fold only.  The shared
        # validity/time/position/missingness channels retain center=0, scale=1;
        # physical sequence channels are standardised before patch projection.
        self.register_buffer("driver_input_center", torch.zeros(cfg.driver_channels), persistent=True)
        self.register_buffer("driver_input_scale", torch.ones(cfg.driver_channels), persistent=True)
        self.register_buffer("battery_input_center", torch.zeros(cfg.battery_channels), persistent=True)
        self.register_buffer("battery_input_scale", torch.ones(cfg.battery_channels), persistent=True)

    def set_concept_normalization(self, center: Tensor, scale: Tensor) -> None:
        """Set training-fold-only concept statistics used by the KAN."""

        center = torch.as_tensor(center, dtype=torch.float32, device=self.concept_center.device)
        scale = torch.as_tensor(scale, dtype=torch.float32, device=self.concept_scale.device)
        if center.shape != self.concept_center.shape or scale.shape != self.concept_scale.shape:
            raise ValueError("Concept normalisation vectors have the wrong shape")
        self.concept_center.copy_(center)
        self.concept_scale.copy_(scale.clamp_min(1.0e-6))

    def set_input_normalization(
        self,
        driver_center: Tensor,
        driver_scale: Tensor,
        battery_center: Tensor,
        battery_scale: Tensor,
    ) -> None:
        """Persist training-fold-only, mask-aware raw-channel statistics."""

        values = (
            (driver_center, self.driver_input_center, "driver center"),
            (driver_scale, self.driver_input_scale, "driver scale"),
            (battery_center, self.battery_input_center, "battery center"),
            (battery_scale, self.battery_input_scale, "battery scale"),
        )
        for source, destination, label in values:
            source = torch.as_tensor(source, dtype=torch.float32, device=destination.device)
            if source.shape != destination.shape:
                raise ValueError(f"Input normalisation {label} has the wrong shape")
            if "scale" in label:
                source = source.clamp_min(1.0e-6)
            destination.copy_(source)

    def _normalize_inputs(self, driver: Tensor, battery: Tensor) -> tuple[Tensor, Tensor]:
        driver = (driver.float() - self.driver_input_center) / self.driver_input_scale
        battery = (battery.float() - self.battery_input_center) / self.battery_input_scale
        return driver, battery

    def _cross_attention(
        self, driver: dict[str, Tensor], battery: dict[str, Tensor]
    ) -> tuple[Tensor, Tensor]:
        driver_tokens = driver["tokens"]
        battery_tokens = battery["tokens"]
        if self.config.cross_attention_mode == "concatenation":
            return driver_tokens, battery_tokens
        d_cross, _ = self.driver_to_battery(
            driver_tokens,
            battery_tokens,
            battery_tokens,
            key_padding_mask=_safe_padding_mask(battery["token_valid"]),
            need_weights=False,
        )
        if self.config.cross_attention_mode == "one_way":
            b_cross = torch.zeros_like(battery_tokens)
        else:
            b_cross, _ = self.battery_to_driver(
                battery_tokens,
                driver_tokens,
                driver_tokens,
                key_padding_mask=_safe_padding_mask(driver["token_valid"]),
                need_weights=False,
            )
        driver_cross = self.driver_cross_norm(driver_tokens + d_cross)
        battery_cross = self.battery_cross_norm(battery_tokens + b_cross)
        driver_cross = driver_cross * driver["token_valid"].unsqueeze(-1)
        battery_cross = battery_cross * battery["token_valid"].unsqueeze(-1)
        return driver_cross, battery_cross

    def _merge_concepts(
        self,
        predicted: Tensor,
        observed: Tensor | None,
        observed_mask: Tensor | None,
    ) -> Tensor:
        if observed is None:
            return predicted
        observed = observed.to(device=predicted.device, dtype=predicted.dtype)
        if observed.shape != predicted.shape:
            raise ValueError("Observed concepts must match predicted concept shape")
        finite = torch.isfinite(observed)
        if observed_mask is not None:
            observed_mask = observed_mask.to(device=predicted.device, dtype=torch.bool)
            if observed_mask.shape != predicted.shape:
                raise ValueError("Observed concept mask has the wrong shape")
            finite &= observed_mask
        return torch.where(finite, observed, predicted)

    def _encode_segments(
        self,
        driver: Tensor,
        battery: Tensor,
        driver_mask: Tensor | None,
        battery_mask: Tensor | None,
    ) -> dict[str, Tensor]:
        """Encode an ordinary flat batch of telemetry segments."""

        driver, battery = self._normalize_inputs(driver, battery)
        if self.config.input_modality == "driver":
            battery = torch.zeros_like(battery)
            battery_mask = torch.ones_like(battery, dtype=torch.bool)
        elif self.config.input_modality == "battery":
            driver = torch.zeros_like(driver)
            driver_mask = torch.ones_like(driver, dtype=torch.bool)
        d_state = self.driver_encoder(driver, driver_mask)
        b_state = self.battery_encoder(battery, battery_mask)
        driver_cross, battery_cross = self._cross_attention(d_state, b_state)
        d_pool, d_attention = self.driver_pool(driver_cross, d_state["token_valid"])
        b_pool, b_attention = self.battery_pool(battery_cross, b_state["token_valid"])
        fused = self.fusion_projection(torch.cat((d_pool, b_pool), dim=-1))
        return {
            "embedding": fused,
            "driver_embedding": d_pool,
            "battery_embedding": b_pool,
            "driver_patch_attention": d_attention,
            "battery_patch_attention": b_attention,
            "driver_gate_probabilities": d_state["gate_probabilities"],
            "battery_gate_probabilities": b_state["gate_probabilities"],
        }

    def encode_segments(
        self,
        driver: Tensor,
        battery: Tensor,
        *,
        driver_mask: Tensor | None = None,
        battery_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Return dual-stream segment embeddings without applying task heads.

        This is the public transfer-learning interface used by external-fleet
        concept pretraining.  Input normalization and modality/cross-attention
        settings remain identical to the ordinary prediction path.
        """

        return self._encode_segments(driver, battery, driver_mask, battery_mask)

    def _predict_from_embedding(
        self,
        fused: Tensor,
        driver_embedding: Tensor,
        *,
        observed_concepts: Tensor | None,
        observed_concept_mask: Tensor | None,
        domain: str,
    ) -> dict[str, Tensor]:
        """Apply the named concept bottleneck and task heads once per row."""

        concept_prediction_normalized = self.concept_predictor(fused)
        concept_prediction = (
            concept_prediction_normalized * self.concept_scale + self.concept_center
        )
        concept_input = self._merge_concepts(
            concept_prediction, observed_concepts, observed_concept_mask
        )
        normalized_concepts = (
            concept_input.float() - self.concept_center
        ) / self.concept_scale
        head_input = normalized_concepts if self.config.prediction_input == "concepts" else fused
        raw_outputs = self.fusion_head(head_input)
        # A historical SOH value is legitimate for future forecasting but
        # would leak the target into contemporaneous SOH estimation.  Evaluate
        # the current-SOH branch at the training-fold mean for that concept.
        if self.config.prediction_input == "concepts" and "last_known_soh" in self.config.concept_names:
            last_soh_index = self.config.concept_names.index("last_known_soh")
            current_concepts = normalized_concepts.clone()
            current_concepts[:, last_soh_index] = 0.0
            current_raw = self.fusion_head(current_concepts)[:, 0]
            raw_outputs = torch.cat((current_raw.unsqueeze(-1), raw_outputs[:, 1:]), dim=-1)
        current_soh, loss_30d, loss_500km = self.calibration[domain](raw_outputs)
        return {
            "current_soh": current_soh,
            "loss_30d": loss_30d,
            "loss_500km": loss_500km,
            "future_soh_30d": current_soh - loss_30d,
            "future_soh_500km": current_soh - loss_500km,
            "behavior_logits": self.behavior_head(driver_embedding),
            "driver_logits": self.driver_id_head(driver_embedding),
            "concepts": concept_prediction,
            "concepts_normalized": concept_prediction_normalized,
            "concept_input": concept_input,
            "normalized_concepts": normalized_concepts,
            "concept_center": self.concept_center,
            "concept_scale": self.concept_scale,
            "embedding": fused,
            "driver_embedding": driver_embedding,
            "l0_penalty": self.channel_l0_penalty(),
            "kan_regularization": self.fusion_regularization(),
        }

    def forward(
        self,
        driver: Tensor,
        battery: Tensor,
        *,
        driver_mask: Tensor | None = None,
        battery_mask: Tensor | None = None,
        observed_concepts: Tensor | None = None,
        observed_concept_mask: Tensor | None = None,
        domain: str = "nissan",
    ) -> dict[str, Tensor]:
        if domain not in self.calibration:
            raise ValueError(
                f"Unknown calibration domain {domain!r}; choose {tuple(self.calibration)}"
            )
        encoded = self._encode_segments(driver, battery, driver_mask, battery_mask)
        outputs = self._predict_from_embedding(
            encoded["embedding"],
            encoded["driver_embedding"],
            observed_concepts=observed_concepts,
            observed_concept_mask=observed_concept_mask,
            domain=domain,
        )
        outputs.update(
            {
                "battery_embedding": encoded["battery_embedding"],
                "driver_patch_attention": encoded["driver_patch_attention"],
                "battery_patch_attention": encoded["battery_patch_attention"],
                "driver_gate_probabilities": encoded["driver_gate_probabilities"],
                "battery_gate_probabilities": encoded["battery_gate_probabilities"],
            }
        )
        return outputs

    def forward_trip_bag(
        self,
        driver: Tensor,
        battery: Tensor,
        *,
        segment_mask: Tensor,
        driver_mask: Tensor | None = None,
        battery_mask: Tensor | None = None,
        observed_concepts: Tensor | None = None,
        observed_concept_mask: Tensor | None = None,
        domain: str = "nissan",
    ) -> dict[str, Tensor]:
        """Predict once per trip after attention-pooling its segment embeddings."""

        if domain not in self.calibration:
            raise ValueError(
                f"Unknown calibration domain {domain!r}; choose {tuple(self.calibration)}"
            )
        encoded_trip = self.encode_trip_bag(
            driver,
            battery,
            segment_mask=segment_mask,
            driver_mask=driver_mask,
            battery_mask=battery_mask,
        )
        outputs = self._predict_from_embedding(
            encoded_trip["embedding"],
            encoded_trip["driver_embedding"],
            observed_concepts=observed_concepts,
            observed_concept_mask=observed_concept_mask,
            domain=domain,
        )
        outputs.update(encoded_trip)
        return outputs

    def encode_trip_bag(
        self,
        driver: Tensor,
        battery: Tensor,
        *,
        segment_mask: Tensor,
        driver_mask: Tensor | None = None,
        battery_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Return hierarchical trip embeddings without applying any SOH head.

        This public interface lets the physics-informed model reuse pretrained
        dual-stream encoders while replacing the unrestricted fusion head.
        """

        if driver.ndim != 4 or battery.ndim != 4:
            raise ValueError("Trip-bag streams must have shape [B,S,T,C]")
        if driver.shape[:3] != battery.shape[:3]:
            raise ValueError("Trip-bag driver and battery streams must share B,S,T")
        batch, segments, time, driver_channels = driver.shape
        if driver_channels != self.config.driver_channels:
            raise ValueError("Trip-bag driver channel width does not match the model")
        if battery.shape[-1] != self.config.battery_channels:
            raise ValueError("Trip-bag battery channel width does not match the model")
        segment_mask = segment_mask.to(device=driver.device, dtype=torch.bool)
        if segment_mask.shape != (batch, segments):
            raise ValueError("segment_mask must have shape [B,S]")
        if torch.any(~segment_mask.any(dim=1)):
            raise ValueError("Every trip bag must contain at least one valid segment")

        def flatten_mask(mask: Tensor | None) -> Tensor | None:
            if mask is None:
                return None
            if mask.shape[:3] != (batch, segments, time):
                raise ValueError("Trip-bag sequence mask has the wrong shape")
            return mask.reshape(batch * segments, *mask.shape[2:])

        encoded = self._encode_segments(
            driver.reshape(batch * segments, time, driver_channels),
            battery.reshape(batch * segments, time, battery.shape[-1]),
            flatten_mask(driver_mask),
            flatten_mask(battery_mask),
        )
        segment_embeddings = encoded["embedding"].reshape(batch, segments, -1)
        segment_driver = encoded["driver_embedding"].reshape(batch, segments, -1)
        segment_battery = encoded["battery_embedding"].reshape(batch, segments, -1)
        trip_embedding, trip_attention = self.trip_pool(segment_embeddings, segment_mask)
        trip_driver, driver_trip_attention = self.trip_driver_pool(
            segment_driver, segment_mask
        )
        return {
                "embedding": trip_embedding,
                "driver_embedding": trip_driver,
                "battery_embedding": torch.einsum(
                    "bs,bsd->bd", trip_attention, segment_battery
                ),
                "segment_embeddings": segment_embeddings,
                "segment_driver_embeddings": segment_driver,
                "segment_battery_embeddings": segment_battery,
                "segment_concepts": (
                    self.concept_predictor(segment_embeddings)
                    * self.concept_scale
                    + self.concept_center
                ),
                "trip_segment_attention": trip_attention,
                "driver_trip_segment_attention": driver_trip_attention,
                "driver_patch_attention": encoded["driver_patch_attention"].reshape(
                    batch, segments, -1
                ),
                "battery_patch_attention": encoded["battery_patch_attention"].reshape(
                    batch, segments, -1
                ),
                "driver_gate_probabilities": encoded["driver_gate_probabilities"],
                "battery_gate_probabilities": encoded["battery_gate_probabilities"],
            }

    @staticmethod
    def contiguous_patch_mask(valid: Tensor, mask_ratio: float = 0.4) -> Tensor:
        """Select one contiguous valid patch block per sequence."""

        if not 0.0 < mask_ratio < 1.0:
            raise ValueError("mask_ratio must lie strictly between zero and one")
        result = torch.zeros_like(valid)
        for row in range(valid.shape[0]):
            valid_indices = torch.nonzero(valid[row], as_tuple=False).flatten()
            if valid_indices.numel() == 0:
                continue
            count = max(1, int(round(valid_indices.numel() * mask_ratio)))
            count = min(count, int(valid_indices.numel()))
            max_start = max(0, valid.shape[1] - count)
            start = int(torch.randint(max_start + 1, (1,), device=valid.device).item())
            candidate = torch.arange(start, start + count, device=valid.device)
            candidate = candidate[valid[row, candidate]]
            if candidate.numel() == 0:
                candidate = valid_indices[:count]
            result[row, candidate] = True
        return result

    def forward_ssl(
        self,
        driver: Tensor,
        battery: Tensor,
        *,
        driver_mask: Tensor | None = None,
        battery_mask: Tensor | None = None,
        mask_ratio: float = 0.4,
    ) -> dict[str, Tensor]:
        # Derive validity without an unmasked encoder pass, then encode once.
        driver, battery = self._normalize_inputs(driver, battery)
        d_valid = self.driver_encoder.token_validity(driver, driver_mask)
        b_valid = self.battery_encoder.token_validity(battery, battery_mask)
        d_masked = self.contiguous_patch_mask(d_valid, mask_ratio)
        b_masked = self.contiguous_patch_mask(b_valid, mask_ratio)
        d_state = self.driver_encoder(driver, driver_mask, masked_patches=d_masked)
        b_state = self.battery_encoder(battery, battery_mask, masked_patches=b_masked)
        d_pool, _ = self.driver_pool(d_state["tokens"], d_state["token_valid"])
        b_pool, _ = self.battery_pool(b_state["tokens"], b_state["token_valid"])
        batch, d_patches, _ = d_state["tokens"].shape
        _, b_patches, _ = b_state["tokens"].shape
        d_reconstruction = self.driver_reconstruction(d_state["tokens"]).reshape(
            batch,
            d_patches,
            self.config.patch_length,
            self.config.driver_channels,
        )
        b_reconstruction = self.battery_reconstruction(b_state["tokens"]).reshape(
            batch,
            b_patches,
            self.config.patch_length,
            self.config.battery_channels,
        )
        return {
            "driver_reconstruction": d_reconstruction,
            "battery_reconstruction": b_reconstruction,
            "driver_target_patches": d_state["target_patches"],
            "battery_target_patches": b_state["target_patches"],
            "driver_element_valid": d_state["element_valid"],
            "battery_element_valid": b_state["element_valid"],
            "driver_masked_patches": d_masked,
            "battery_masked_patches": b_masked,
            "driver_embedding": d_pool,
            "battery_embedding": b_pool,
            "l0_penalty": self.channel_l0_penalty(),
        }

    def channel_l0_penalty(self) -> Tensor:
        return 0.5 * (
            self.driver_encoder.gate.l0_penalty()
            + self.battery_encoder.gate.l0_penalty()
        )

    def fusion_regularization(self) -> Tensor:
        regularizer = getattr(self.fusion_head, "regularization_loss", None)
        if regularizer is None:
            return self.concept_center.new_zeros(())
        return regularizer()

    def freeze_patch_encoders(self, freeze: bool = True) -> None:
        """Freeze both encoders for low-label Nissan fine-tuning."""

        for module in (self.driver_encoder, self.battery_encoder):
            for parameter in module.parameters():
                parameter.requires_grad = not freeze

    @torch.no_grad()
    def prune_channels(
        self,
        *,
        driver_indices: tuple[int, ...] = (),
        battery_indices: tuple[int, ...] = (),
    ) -> None:
        """Apply channel decisions made from fold stability plus occlusion tests."""

        self.driver_encoder.gate.prune(driver_indices)
        self.battery_encoder.gate.prune(battery_indices)

    @torch.no_grad()
    def prune_kan_edges(self, threshold: float) -> int:
        """Prune the KAN after training; caller should then fine-tune."""

        prune = getattr(self.fusion_head, "prune_edges", None)
        return 0 if prune is None else int(prune(threshold))


__all__ = [
    "DEFAULT_CONCEPT_NAMES",
    "DBPatchKANConfig",
    "ModelConfig",
    "model_config",
    "StochasticChannelGate",
    "TemporalPatchEmbedding",
    "PatchStreamEncoder",
    "AttentionPool",
    "DBPatchKAN",
]
