#!/usr/bin/env python3
"""Trace the LeRobot pi05 conversion on an observation/noise captured by OpenPI."""

import argparse
import pathlib
import numpy as np
import torch
import sentencepiece

from safetensors.torch import load_file
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.processor import (
    EnvTransition,
    PolicyAction,
    TransitionKey,
)
from lerobot.utils.constants import (
    ACTION,
    OBS_IMAGES,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)


def np_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def run_flow(model, images, image_masks, tokens, token_masks, noise, steps=10):
    velocities = []
    original = model.denoise_step

    def traced(*args, **kwargs):
        velocity = original(*args, **kwargs)
        velocities.append(np_cpu(velocity[0]))
        return velocity

    model.denoise_step = traced
    try:
        result = model.sample_actions(
            images,
            image_masks,
            tokens,
            token_masks,
            noise=noise,
            num_steps=steps,
        )
    finally:
        model.denoise_step = original

    dt = -1.0 / steps
    states = [np_cpu(noise[0])]
    for velocity in velocities:
        states.append(states[-1] + dt * velocity)
    if not np.allclose(states[-1], np_cpu(result[0]), atol=2e-5, rtol=2e-5):
        raise RuntimeError("Recorded Euler path does not reconstruct sample_actions output")
    return result, np.stack(velocities), np.stack(states)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--observation", type=pathlib.Path, required=True)
    parser.add_argument("--openpi-trace", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--unaligned-checkpoint", type=pathlib.Path, default=None)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--tokenizer", type=pathlib.Path, default=None)
    args = parser.parse_args()

    bundle = np.load(args.observation)
    openpi = np.load(args.openpi_trace)

    # 1. Trace the aligned LeRobot policy natively
    policy = PI05Policy.from_pretrained(args.checkpoint)
    config = policy.config
    policy.to(config.device)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=args.checkpoint,
        preprocessor_overrides={"device_processor": {"device": str(config.device)}},
    )

    base = torch.from_numpy(bundle["base"]).permute(2, 0, 1)[None].float() / 255.0
    wrist = torch.from_numpy(bundle["wrist"]).permute(2, 0, 1)[None].float() / 255.0
    state = torch.from_numpy(bundle["state"]).float()

    raw_batch = {
        "observation.images.image": base,
        "observation.images.image2": wrist,
        "observation.state": state,
        "task": str(bundle["prompt"]),
    }

    # Run native LeRobot preprocessor (PaliGemmaTokenizerProcessorStep + OpenPI normalizer)
    processed_batch = preprocessor(raw_batch)
    tokens = processed_batch[OBS_LANGUAGE_TOKENS]
    token_masks = processed_batch[OBS_LANGUAGE_ATTENTION_MASK]

    images, image_masks = policy._preprocess_images(processed_batch)
    noise_10 = torch.from_numpy(bundle["noise"])[None].to(config.device)

    aligned_normalized, aligned_v, aligned_x = run_flow(
        policy.model,
        images,
        image_masks,
        tokens,
        token_masks,
        noise_10,
        steps=config.num_inference_steps,
    )

    # Unnormalize using native postprocessor
    unnorm_res = postprocessor(PolicyAction(aligned_normalized[:, :, :7]))
    aligned_env = np_cpu(unnorm_res[0])

    # 2. Trace the unaligned baseline behavior using SentencePiece (state-in-prompt, 50-action chunk)
    tokenizer_file = args.tokenizer or (args.checkpoint / "paligemma_tokenizer.model")
    sp_tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(tokenizer_file))

    stats = load_file(
        args.checkpoint / "policy_preprocessor_step_2_normalizer_processor.safetensors"
    )
    state_q01 = stats["observation.state.q01"].numpy()
    state_q99 = stats["observation.state.q99"].numpy()
    state_denom = state_q99 - state_q01
    state_denom = np.where(state_denom == 0, 1e-8, state_denom)
    state_norm = 2.0 * (bundle["state"] - state_q01) / state_denom - 1.0
    padded = np.pad(state_norm, (0, config.max_state_dim - state_norm.shape[-1]))
    bins = np.linspace(-1, 1, 257)[:-1]
    discretized = np.digitize(padded, bins=bins) - 1
    cleaned = str(bundle["prompt"]).strip().replace("_", " ").replace(chr(10), " ")
    discretized_str = " ".join(map(str, discretized))
    saved_prompt = "Task: " + cleaned + ", State: " + discretized_str + ";" + chr(10) + "Action: "

    token_ids_saved = sp_tokenizer.encode(saved_prompt, add_bos=True)[:200]
    token_array = np.zeros(200, dtype=np.int64)
    token_mask_array = np.zeros(200, dtype=bool)
    token_array[: len(token_ids_saved)] = token_ids_saved
    token_mask_array[: len(token_ids_saved)] = True
    saved_tokens_t = torch.from_numpy(token_array)[None].to(config.device)
    saved_token_masks_t = torch.from_numpy(token_mask_array)[None].to(config.device)

    # Temporarily set chunk_size to 50 for unaligned model simulation
    saved_chunk_size = 50
    policy.config.chunk_size = 50
    policy.model.config.chunk_size = 50

    rng = np.random.default_rng(0)
    noise_50_np = rng.standard_normal((50, config.max_action_dim)).astype(np.float32)
    noise_50_np[:10] = bundle["noise"]
    noise_50 = torch.from_numpy(noise_50_np)[None].to(config.device)

    saved_normalized, saved_v, saved_x = run_flow(
        policy.model,
        images,
        image_masks,
        saved_tokens_t,
        saved_token_masks_t,
        noise_50,
        steps=10,
    )
    # Restore chunk_size
    policy.config.chunk_size = 10
    policy.model.config.chunk_size = 10

    # Unnormalize saved with standard lerobot unnormalizer formula
    act_q01 = stats["action.q01"].numpy()
    act_q99 = stats["action.q99"].numpy()
    saved_norm_vals = np_cpu(saved_normalized[0, :, :7])
    saved_env = (saved_norm_vals + 1.0) * (act_q99 - act_q01) / 2.0 + act_q01

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        state_raw=bundle["state"],
        state_normalized=np_cpu(processed_batch[OBS_STATE][0]),
        prompt=np.asarray(str(bundle["prompt"])),
        saved_prompt=np.asarray(saved_prompt),
        saved_tokens=np_cpu(saved_tokens_t[0]),
        saved_token_mask=np_cpu(saved_token_masks_t[0]),
        aligned_tokens=np_cpu(tokens[0]),
        aligned_token_mask=np_cpu(token_masks[0]),
        matched_tokens=np_cpu(tokens[0]),
        matched_token_mask=np_cpu(token_masks[0]),
        base_model=np_cpu(images[0][0]),
        wrist_model=np_cpu(images[1][0]),
        empty_model=np_cpu(images[2][0]),
        image_masks=np.asarray([bool(np_cpu(mask[0])) for mask in image_masks]),
        noise_saved=noise_50_np,
        noise_matched=bundle["noise"],
        saved_velocity=saved_v,
        saved_flow_state=saved_x,
        saved_action_normalized=np_cpu(saved_normalized[0]),
        saved_action_env=saved_env,
        saved_action_executed=saved_env[:5],
        saved_chunk_size=np.asarray(saved_chunk_size),
        saved_replan_steps=np.asarray(5),
        aligned_velocity=aligned_v,
        aligned_flow_state=aligned_x,
        aligned_action_normalized=np_cpu(aligned_normalized[0]),
        aligned_action_env=aligned_env,
        aligned_action_executed=aligned_env[:5],
        aligned_chunk_size=np.asarray(config.chunk_size),
        aligned_replan_steps=np.asarray(config.n_action_steps),
        matched_velocity=aligned_v,
        matched_flow_state=aligned_x,
        matched_action_normalized=np_cpu(aligned_normalized[0]),
        matched_action_env=aligned_env,
        matched_action_executed=aligned_env[:5],
        matched_chunk_size=np.asarray(config.chunk_size),
        matched_replan_steps=np.asarray(config.n_action_steps),
        flow_steps=np.asarray(config.num_inference_steps),
        flow_dt=np.asarray(-0.1, dtype=np.float32),
    )
    print(
        "LEROBOT_TRACE_OK",
        args.output,
        "aligned",
        aligned_env.shape,
        "aligned_tokens",
        int(np_cpu(token_masks[0]).sum()),
        "official_tokens",
        int(np.asarray(openpi["token_mask"]).sum()),
    )


if __name__ == "__main__":
    main()
