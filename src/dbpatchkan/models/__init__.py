"""Public model, dataset, and loss interfaces."""

from .datasets import (
    BATTERY_CHANNELS,
    DRIVER_CHANNELS,
    EVSegmentDataset,
    EVTripBagDataset,
    assert_disjoint_trip_bags,
    collate_trip_bags,
    make_synthetic_dataset,
)
from .kan import BSplineKANLayer, ConceptKAN, ConceptMLP, make_fusion_head
from .physics_kan import PhysicsMediatedKAN
from .physics_patchkan import (
    NISSAN_PHYSICS_CONCEPT_NAMES,
    NISSAN_RESPONSE_TARGET_NAMES,
    PhysicsInformedDBPatchKAN,
)
from .physics_corrected import (
    CURRENT_INTERACTION_FEATURES,
    CURRENT_MONOTONIC_FEATURES,
    FORECAST_30D_FEATURES,
    FORECAST_500KM_FEATURES,
    BoundedMonotonicResidualHead,
    CorrectedPhysicsHybrid,
    ScaledForecastHead,
)
from .losses import (
    DBPatchKANMultiTaskLoss,
    LossWeights,
    MaskedSelfSupervisedLoss,
    masked_cross_entropy,
    masked_huber_loss,
    masked_patch_reconstruction_loss,
    nt_xent_loss,
    physics_consistency_loss,
)
from .patchkan import (
    DEFAULT_CONCEPT_NAMES,
    DBPatchKAN,
    DBPatchKANConfig,
    ModelConfig,
    StochasticChannelGate,
    model_config,
)

__all__ = [
    "BATTERY_CHANNELS",
    "DRIVER_CHANNELS",
    "EVSegmentDataset",
    "EVTripBagDataset",
    "assert_disjoint_trip_bags",
    "collate_trip_bags",
    "make_synthetic_dataset",
    "BSplineKANLayer",
    "ConceptKAN",
    "ConceptMLP",
    "make_fusion_head",
    "PhysicsMediatedKAN",
    "NISSAN_PHYSICS_CONCEPT_NAMES",
    "NISSAN_RESPONSE_TARGET_NAMES",
    "PhysicsInformedDBPatchKAN",
    "CURRENT_INTERACTION_FEATURES", "CURRENT_MONOTONIC_FEATURES",
    "FORECAST_30D_FEATURES", "FORECAST_500KM_FEATURES",
    "BoundedMonotonicResidualHead", "CorrectedPhysicsHybrid", "ScaledForecastHead",
    "DBPatchKANMultiTaskLoss",
    "LossWeights",
    "MaskedSelfSupervisedLoss",
    "masked_cross_entropy",
    "masked_huber_loss",
    "masked_patch_reconstruction_loss",
    "nt_xent_loss",
    "physics_consistency_loss",
    "DEFAULT_CONCEPT_NAMES",
    "DBPatchKAN",
    "DBPatchKANConfig",
    "ModelConfig",
    "model_config",
    "StochasticChannelGate",
]
