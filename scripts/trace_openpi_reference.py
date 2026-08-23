#!/usr/bin/env python3
import argparse
import pathlib
import types

import einops
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as model_api
from openpi.models import pi0
from openpi.policies import policy_config
from openpi.shared import nnx_utils
from openpi.training import config


def trace_sample_actions(self, rng, observation, *, noise=None):
    del rng
    observation = model_api.preprocess_observation(None, observation, train=False)
    dt = -0.1
    batch_size = observation.state.shape[0]
    prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    prefix_attn_mask = pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = self.PaliGemma.llm(
        [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
    )

    def step(x_t, time):
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, x_t, jnp.broadcast_to(time, batch_size)
        )
        suffix_attn_mask = pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_to_suffix = einops.repeat(
            prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1]
        )
        full_attn_mask = jnp.concatenate(
            [prefix_to_suffix, suffix_attn_mask], axis=-1
        )
        suffix_positions = (
            jnp.sum(prefix_mask, axis=-1)[:, None]
            + jnp.cumsum(suffix_mask, axis=-1)
            - 1
        )
        (_, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=suffix_positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        velocity = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        x_next = x_t + dt * velocity
        return x_next, velocity

    x_t = noise
    time = jnp.asarray(1.0, dtype=jnp.float32)
    velocities = []
    states = [noise]
    for _ in range(10):
        x_t, velocity = step(x_t, time)
        velocities.append(velocity)
        states.append(x_t)
        time = time + dt
    final = x_t
    velocities = jnp.stack(velocities)
    states = jnp.stack(states)
    return final, velocities, states


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observation", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    bundle = np.load(args.observation)
    raw_input = {
        "observation/image": bundle["base"],
        "observation/wrist_image": bundle["wrist"],
        "observation/state": bundle["state"],
        "prompt": str(bundle["prompt"]),
    }
    train_config = config.get_config("pi05_libero")
    policy = policy_config.create_trained_policy(train_config, args.checkpoint)

    transformed = policy._input_transform(raw_input.copy())
    batched = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], transformed)
    observation = model_api.Observation.from_dict(batched)
    model_observation = model_api.preprocess_observation(
        None,
        observation,
        train=False,
    )
    noise = jnp.asarray(bundle["noise"])[None, ...]
    normalized_actions = policy._sample_actions(
        jax.random.key(0),
        observation,
        noise=noise,
        num_steps=10,
    )
    normalized_actions_np = np.asarray(normalized_actions[0])

    traced = nnx_utils.module_jit(
        types.MethodType(trace_sample_actions, policy._model)
    )
    traced_final, flow_velocity, flow_state = traced(
        jax.random.key(0), observation, noise=noise
    )
    trace_max_abs = float(
        np.max(np.abs(np.asarray(normalized_actions) - np.asarray(traced_final)))
    )
    if trace_max_abs > 2e-5:
        print("OPENPI_TRACE_COMPILATION_DELTA", trace_max_abs)

    outputs = {
        "state": np.asarray(transformed["state"]),
        "actions": normalized_actions_np,
    }
    env_outputs = policy._output_transform(outputs)
    env_actions = np.asarray(env_outputs["actions"])

    direct = policy.infer(raw_input, noise=np.asarray(bundle["noise"]))["actions"]
    if not np.allclose(env_actions, direct, atol=1e-5, rtol=1e-5):
        raise RuntimeError("Direct policy output and traced output disagree")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        state_raw=bundle["state"],
        state_normalized=np.asarray(transformed["state"]),
        prompt=np.asarray(str(bundle["prompt"])),
        tokens=np.asarray(transformed["tokenized_prompt"]),
        token_mask=np.asarray(transformed["tokenized_prompt_mask"]),
        base_model=np.asarray(model_observation.images["base_0_rgb"][0]),
        wrist_model=np.asarray(model_observation.images["left_wrist_0_rgb"][0]),
        empty_model=np.asarray(model_observation.images["right_wrist_0_rgb"][0]),
        image_mask_base=np.asarray(model_observation.image_masks["base_0_rgb"][0]),
        image_mask_wrist=np.asarray(model_observation.image_masks["left_wrist_0_rgb"][0]),
        image_mask_empty=np.asarray(model_observation.image_masks["right_wrist_0_rgb"][0]),
        noise=np.asarray(bundle["noise"]),
        flow_velocity=np.asarray(flow_velocity[:, 0]),
        flow_state=np.asarray(flow_state[:, 0]),
        flow_final=np.asarray(traced_final[0]),
        flow_compilation_max_abs=np.asarray(trace_max_abs),
        action_normalized=normalized_actions_np,
        action_env=env_actions,
        action_executed=env_actions[:5],
        action_horizon=np.asarray(10),
        flow_steps=np.asarray(10),
        flow_dt=np.asarray(-0.1, dtype=np.float32),
        replan_steps=np.asarray(5),
    )
    print(
        "OPENPI_TRACE_OK",
        args.output,
        "tokens",
        int(np.asarray(transformed["tokenized_prompt_mask"]).sum()),
        "action_shape",
        env_actions.shape,
        "executed_shape",
        env_actions[:5].shape,
    )


if __name__ == "__main__":
    main()
