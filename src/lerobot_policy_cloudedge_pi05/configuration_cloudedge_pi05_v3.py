from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config


@PreTrainedConfig.register_subclass("cloudedge_pi05_v3")
@dataclass
class CloudEdgePI05V3Config(PI05Config):
    """Cloud-plan plus per-step lightweight edge-correction pi0.5."""

    freeze_vision_encoder: bool = True
    history_window: int = 21
    stale_loss_weight_max: float = 0.5
    stale_loss_warmup_steps: int = 2_500
    flow_loss_weight: float = 1.0
    correction_loss_weight: float = 1.0
    edge_model_name: str = "google/siglip-base-patch16-224"
    edge_model_revision: str = "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"
    use_edge_vision: bool = True
    eval_delay_max: int = 0
    cloudedge_train_step: int = 0

    def __post_init__(self):
        super().__post_init__()
        if self.history_window < 2:
            raise ValueError("history_window must be at least 2")
        if not 0.0 <= self.stale_loss_weight_max <= 1.0:
            raise ValueError("stale_loss_weight_max must lie in [0, 1]")
        if self.stale_loss_warmup_steps < 0:
            raise ValueError("stale_loss_warmup_steps must be non-negative")
        if self.flow_loss_weight < 0 or self.correction_loss_weight < 0:
            raise ValueError("loss weights must be non-negative")
        if self.eval_delay_max < 0 or self.cloudedge_train_step < 0:
            raise ValueError("delay and train step must be non-negative")
        if self.n_action_steps != 1:
            raise ValueError("V3 requires n_action_steps=1 for per-environment-step correction")

    @property
    def observation_delta_indices(self) -> list[int]:
        return list(range(-(self.history_window - 1), 1))
