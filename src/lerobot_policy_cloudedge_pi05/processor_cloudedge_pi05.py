from dataclasses import dataclass
from typing import Any

import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PaliGemmaTokenizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from .configuration_cloudedge_pi05 import CloudEdgePI05Config


@ProcessorStepRegistry.register(name="cloudedge_select_current_state_processor")
@dataclass
class SelectCurrentStateProcessorStep(ProcessorStep):
    """Keep current proprioception while retaining visual history tensors.

    LeRobot applies observation deltas to every observation key. CloudEdge delays
    only cloud images, so the state used to build the pi0.5 prompt must remain at t.
    """

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        observations = transition.get(TransitionKey.OBSERVATION, {})
        state = observations.get(OBS_STATE)
        if isinstance(state, torch.Tensor):
            if state.ndim >= 3:
                observations = observations.copy()
                observations[OBS_STATE] = state[:, -1]
                transition[TransitionKey.OBSERVATION] = observations
            elif state.ndim == 2 and state.shape[0] > 1:
                observations = observations.copy()
                observations[OBS_STATE] = state[-1]
                transition[TransitionKey.OBSERVATION] = observations
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_cloudedge_pi05_pre_post_processors(
    config: CloudEdgePI05Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        SelectCurrentStateProcessorStep(),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        PaliGemmaTokenizerProcessorStep(
            tokenizer_path="paligemma_tokenizer.model",
            max_length=config.tokenizer_max_length,
            task_key="task",
            clean_text=True,
        ),
        DeviceProcessorStep(device=config.device),
    ]
    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        DeviceProcessorStep(device="cpu"),
    ]
    return (
        PolicyProcessorPipeline(
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline(
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
