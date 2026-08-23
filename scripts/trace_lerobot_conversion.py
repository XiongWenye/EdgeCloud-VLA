#!/usr/bin/env python3
"""Trace the LeRobot pi05 conversion on an observation/noise captured by OpenPI."""

import argparse
import pathlib

import numpy as np
import torch

import sentencepiece
from safetensors.torch import load_file
from lerobot.policies.pi05.modeling_pi05 import PI05Policy


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


def unnormalize(stats, normalized):
    q01 = stats["action.q01"].numpy()
    q99 = stats["action.q99"].numpy()
    values = np_cpu(normalized[0])
    return (values + 1.0) * (q99 - q01) / 2.0 + q01


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--observation", type=pathlib.Path, required=True)
    parser.add_argument("--openpi-trace", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--tokenizer", type=pathlib.Path, required=True)
    args = parser.parse_args()

    bundle = np.load(args.observation)
    openpi = np.load(args.openpi_trace)
    policy = PI05Policy.from_pretrained(args.checkpoint)
    config = policy.config
    policy.to(config.device)
    policy.eval()
    stats = load_file(
        args.checkpoint
        / "policy_preprocessor_step_2_normalizer_processor.safetensors"
    )
    state_q01 = stats["observation.state.q01"].numpy()
    state_q99 = stats["observation.state.q99"].numpy()
    state_denom = state_q99 - state_q01
    state_denom = np.where(state_denom == 0, 1e-8, state_denom)
    state_norm = 2.0 * (bundle["state"] - state_q01) / state_denom - 1.0
    padded = np.pad(state_norm, (0, config.max_state_dim - state_norm.shape[-1]))
    bins = np.linspace(-1, 1, 257)[:-1]
    discretized = np.digitize(padded, bins=bins) - 1
    cleaned = str(bundle["prompt"]).strip().replace("_", " ").replace("\n", " ")
    saved_prompt = f"Task: {cleaned}, State: {' '.join(map(str, discretized))};\nAction: "

    tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(args.tokenizer))
    token_ids = tokenizer.encode(saved_prompt, add_bos=True)[:200]
    token_array = np.zeros(200, dtype=np.int64)
    token_mask_array = np.zeros(200, dtype=bool)
    token_array[: len(token_ids)] = token_ids
    token_mask_array[: len(token_ids)] = True
    tokens = torch.from_numpy(token_array)[None].to(config.device)
    token_masks = torch.from_numpy(token_mask_array)[None].to(config.device)

    base = torch.from_numpy(bundle["base"]).permute(2, 0, 1)[None].float() / 255.0
    wrist = torch.from_numpy(bundle["wrist"]).permute(2, 0, 1)[None].float() / 255.0
    batch = {
        "observation.images.image": base.to(config.device),
        "observation.images.image2": wrist.to(config.device),
    }
    images, image_masks = policy._preprocess_images(batch)

    rng = np.random.default_rng(0)
    noise_50_np = rng.standard_normal((50, config.max_action_dim)).astype(np.float32)
    noise_50_np[:10] = bundle["noise"]
    noise_50 = torch.from_numpy(noise_50_np)[None].to(config.device)
    saved_normalized, saved_v, saved_x = run_flow(
        policy.model, images, image_masks, tokens, token_masks, noise_50
    )
    saved_env = unnormalize(stats, saved_normalized[:, :, :7])

    saved_chunk_size = config.chunk_size
    policy.config.chunk_size = 10
    policy.model.config.chunk_size = 10
    official_tokens = torch.from_numpy(openpi["tokens"])[None].to(
        device=config.device, dtype=tokens.dtype
    )
    official_masks = torch.from_numpy(openpi["token_mask"])[None].to(
        device=config.device, dtype=token_masks.dtype
    )
    noise_10 = torch.from_numpy(bundle["noise"])[None].to(config.device)
    matched_normalized, matched_v, matched_x = run_flow(
        policy.model,
        images,
        image_masks,
        official_tokens,
        official_masks,
        noise_10,
    )
    matched_env = unnormalize(stats, matched_normalized[:, :, :7])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        state_raw=bundle["state"],
        state_normalized=state_norm,
        saved_prompt=np.asarray(saved_prompt),
        saved_tokens=np_cpu(tokens[0]),
        saved_token_mask=np_cpu(token_masks[0]),
        matched_tokens=np_cpu(official_tokens[0]),
        matched_token_mask=np_cpu(official_masks[0]),
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
        matched_velocity=matched_v,
        matched_flow_state=matched_x,
        matched_action_normalized=np_cpu(matched_normalized[0]),
        matched_action_env=matched_env,
        matched_action_executed=matched_env[:5],
        matched_chunk_size=np.asarray(10),
        matched_replan_steps=np.asarray(5),
        flow_steps=np.asarray(10),
        flow_dt=np.asarray(-0.1, dtype=np.float32),
    )
    print(
        "LEROBOT_TRACE_OK",
        args.output,
        "saved",
        saved_env.shape,
        "matched",
        matched_env.shape,
        "saved_tokens",
        int(np_cpu(token_masks[0]).sum()),
        "official_tokens",
        int(np_cpu(official_masks[0]).sum()),
    )


if __name__ == "__main__":
    main()
