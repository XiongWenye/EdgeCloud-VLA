from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config


@PreTrainedConfig.register_subclass("cloudedge_pi05")
@dataclass
class CloudEdgePI05Config(PI05Config):
    """Configuration for paired-frame CloudEdge training on the pi0.5 flow field.

    The paper specifies a 21-frame training window but does not report lambda_max
    or the curriculum length. Those values are therefore explicit experimental
    assumptions rather than hidden constants.
    """

    architecture_version: int = 2

    # Both visual towers stay frozen. Cloud language/planning weights are adapted
    # with LoRA and the pi0.5 action/time projections plus edge head are trained.
    freeze_vision_encoder: bool = True

    history_window: int = 21
    stale_loss_weight_max: float = 0.5
    stale_loss_warmup_steps: int = 10_000

    edge_model_name: str = "google/siglip-base-patch16-224"
    edge_model_revision: str = "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"
    use_edge_vision: bool = True

    # Inference-only observation-age simulation. Zero is the synchronous case.
    eval_delay_max: int = 0

    # Persisted in config.json so the lambda curriculum survives PEFT checkpoints.
    cloudedge_train_step: int = 0

    def __post_init__(self):
        super().__post_init__()
        if self.architecture_version != 2:
            raise ValueError("Only CloudEdge pi0.5 architecture_version=2 is supported")
        if self.history_window < 2:
            raise ValueError("history_window must be at least 2")
        if not 0.0 <= self.stale_loss_weight_max <= 1.0:
            raise ValueError("stale_loss_weight_max must lie in [0, 1]")
        if self.stale_loss_warmup_steps < 0:
            raise ValueError("stale_loss_warmup_steps must be non-negative")
        if self.eval_delay_max < 0:
            raise ValueError("eval_delay_max must be non-negative")
        if self.cloudedge_train_step < 0:
            raise ValueError("cloudedge_train_step must be non-negative")

    @property
    def observation_delta_indices(self) -> list[int]:
        # Episode-boundary padding is handled by LeRobotDataset. The policy samples
        # one episode-safe stale offset from this window on every training update.
        return list(range(-(self.history_window - 1), 1))
