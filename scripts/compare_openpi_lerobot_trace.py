#!/usr/bin/env python3
import argparse
import json
import pathlib

import numpy as np
from safetensors.torch import load_file


def metric(a, b):
    def convert(value):
        array = np.asarray(value)
        if array.dtype.kind == "V" and array.dtype.itemsize == 2:
            bits = array.view(np.uint16).astype(np.uint32) << 16
            array = bits.view(np.float32)
        return array.astype(np.float64)

    a, b = convert(a), convert(b)
    d = a - b
    denom = np.linalg.norm(a.ravel()) * np.linalg.norm(b.ravel())
    return {
        "shape": list(a.shape),
        "mae": float(np.mean(np.abs(d))),
        "rmse": float(np.sqrt(np.mean(d * d))),
        "max_abs": float(np.max(np.abs(d))),
        "cosine": float(np.dot(a.ravel(), b.ravel()) / denom) if denom else None,
    }


def show(m):
    return f"MAE {m['mae']:.3e}, RMSE {m['rmse']:.3e}, max {m['max_abs']:.3e}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--observation", type=pathlib.Path, required=True)
    p.add_argument("--openpi", type=pathlib.Path, required=True)
    p.add_argument("--lerobot", type=pathlib.Path, required=True)
    p.add_argument("--checkpoint", type=pathlib.Path, required=True)
    p.add_argument("--json", type=pathlib.Path, required=True)
    p.add_argument("--markdown", type=pathlib.Path, required=True)
    a = p.parse_args()

    obs, op, lr = np.load(a.observation), np.load(a.openpi), np.load(a.lerobot)
    stats = load_file(a.checkpoint / "policy_postprocessor_step_0_unnormalizer_processor.safetensors")
    q01, q99 = stats["action.q01"].numpy(), stats["action.q99"].numpy()
    checkpoint_config = json.loads((a.checkpoint / "config.json").read_text())

    op_base, op_wrist = op["base_model"], op["wrist_model"]
    if op_base.shape[-1] == 3:
        op_base = np.transpose(op_base, (2, 0, 1))
        op_wrist = np.transpose(op_wrist, (2, 0, 1))

    op_ids = op["tokens"][op["token_mask"].astype(bool)]
    lr_ids = lr["saved_tokens"][lr["saved_token_mask"].astype(bool)]
    exact_unnorm = (lr["matched_action_normalized"][:, :7] + 1) / 2 * (q99 - q01 + 1e-6) + q01

    result = {
        "protocol": {
            "openpi_horizon": int(op["action_horizon"]),
            "openpi_execute": int(op["replan_steps"]),
            "lerobot_saved_horizon": int(lr["saved_chunk_size"]),
            "lerobot_saved_execute": int(lr["saved_replan_steps"]),
            "lerobot_artifact_n_action_steps": int(checkpoint_config["n_action_steps"]),
            "lerobot_matched_horizon": int(lr["matched_chunk_size"]),
            "flow_steps": int(op["flow_steps"]),
            "flow_dt": float(op["flow_dt"]),
            "openpi_trace_compilation_max_abs": float(op["flow_compilation_max_abs"]),
        },
        "observations": {
            "base_image": metric(op_base, lr["base_model"]),
            "wrist_image": metric(op_wrist, lr["wrist_model"]),
            "openpi_masks": [bool(op["image_mask_base"]), bool(op["image_mask_wrist"]), bool(op["image_mask_empty"])],
            "lerobot_masks": [bool(x) for x in lr["image_masks"]],
            "raw_state": [float(x) for x in obs["state"]],
            "normalized_state": metric(op["state_normalized"][:8], lr["state_normalized"]),
        },
        "tokens": {
            "openpi_prompt": str(op["prompt"]),
            "lerobot_saved_prompt": str(lr["saved_prompt"]),
            "openpi_valid_count": int(op["token_mask"].sum()),
            "lerobot_valid_count": int(lr["saved_token_mask"].sum()),
            "valid_ids_equal": bool(op_ids.shape == lr_ids.shape and np.array_equal(op_ids, lr_ids)),
            "openpi_valid_ids": [int(x) for x in op_ids],
            "lerobot_valid_ids": [int(x) for x in lr_ids],
        },
        "normalization": {
            "q01": [float(x) for x in q01],
            "q99": [float(x) for x in q99],
            "openpi_epsilon": 1e-6,
            "lerobot_zero_range_epsilon": 1e-8,
            "action_formula_delta": metric(exact_unnorm, lr["matched_action_env"]),
        },
        "flow_matching": {
            "matched_velocity": metric(op["flow_velocity"], lr["matched_velocity"]),
            "matched_state": metric(op["flow_state"], lr["matched_flow_state"]),
            "matched_final": metric(op["action_normalized"], lr["matched_action_normalized"]),
            "saved_first10": metric(op["action_normalized"], lr["saved_action_normalized"][:10]),
        },
        "executed_actions": {
            "matched_exact_unnorm": metric(op["action_executed"], exact_unnorm[:5]),
            "matched_lerobot_unnorm": metric(op["action_executed"], lr["matched_action_executed"]),
            "saved_conversion": metric(op["action_executed"], lr["saved_action_executed"]),
        },
    }

    md = [
        "# OpenPI vs LeRobot pi05 parity trace",
        "",
        "Same fixed LIBERO Spatial observation and fixed Gaussian noise.",
        "",
        "## Protocol",
        "",
        f"- OpenPI predicts {result['protocol']['openpi_horizon']} and executes {result['protocol']['openpi_execute']}.",
        f"- Saved LeRobot predicts {result['protocol']['lerobot_saved_horizon']}; its artifact default executes {result['protocol']['lerobot_artifact_n_action_steps']}.",
        f"- The previous evaluator override and this parity trace compare the first {result['protocol']['lerobot_saved_execute']} actions.",
        f"- Matched LeRobot predicts {result['protocol']['lerobot_matched_horizon']} with official token IDs and executes 5.",
        "- Both run ten Euler steps with dt=-0.1.",
        "",
        "## Tensor differences",
        "",
        f"- Diagnostic OpenPI unrolled trace versus upstream compiled final max delta: {result['protocol']['openpi_trace_compilation_max_abs']:.6g}.",
        "  Final-action metrics use the upstream compiled output.",
        f"- Base image: {show(result['observations']['base_image'])}.",
        f"- Wrist image: {show(result['observations']['wrist_image'])}.",
        f"- Normalized state: {show(result['observations']['normalized_state'])}.",
        f"- Matched flow velocities: {show(result['flow_matching']['matched_velocity'])}.",
        f"- Matched flow states: {show(result['flow_matching']['matched_state'])}.",
        f"- Matched final normalized actions: {show(result['flow_matching']['matched_final'])}.",
        f"- Matched first five executed actions: {show(result['executed_actions']['matched_exact_unnorm'])}.",
        f"- Saved conversion first five executed actions: {show(result['executed_actions']['saved_conversion'])}.",
        "",
        "## Semantic differences",
        "",
        f"- Valid prompt tokens: OpenPI {result['tokens']['openpi_valid_count']}, LeRobot {result['tokens']['lerobot_valid_count']}; equal IDs: {result['tokens']['valid_ids_equal']}.",
        "- Official pi05_libero uses cleaned task text plus newline and does not feed state to the model.",
        "- Saved LeRobot inserts normalized, zero-padded 32D state values discretized into 256 bins in a Task/State/Action prompt.",
        "- OpenPI adds 1e-6 to every quantile range; LeRobot substitutes 1e-8 only for an exactly zero range.",
        "- Saved LeRobot predicts 50 coupled tokens and defaults to executing 10; OpenPI predicts 10 and executes five.",
        "- The parity trace compares the first five from both chunks to isolate replanning semantics.",
    ]

    a.json.parent.mkdir(parents=True, exist_ok=True)
    a.json.write_text(json.dumps(result, indent=2) + "\n")
    a.markdown.write_text("\n".join(md) + "\n")
    print("PARITY_COMPARE_OK", a.json, a.markdown)


if __name__ == "__main__":
    main()
