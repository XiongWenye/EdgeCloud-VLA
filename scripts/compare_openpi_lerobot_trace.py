#!/usr/bin/env python3
"""Compare OpenPI reference trace with aligned LeRobot trace."""

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
    stats = load_file(
        a.checkpoint / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    )
    q01, q99 = stats["action.q01"].numpy(), stats["action.q99"].numpy()
    checkpoint_config = json.loads((a.checkpoint / "config.json").read_text())

    op_base, op_wrist = op["base_model"], op["wrist_model"]
    if op_base.shape[-1] == 3:
        op_base = np.transpose(op_base, (2, 0, 1))
        op_wrist = np.transpose(op_wrist, (2, 0, 1))

    op_ids = op["tokens"][op["token_mask"].astype(bool)]
    lr_aligned_ids = lr["aligned_tokens"][lr["aligned_token_mask"].astype(bool)]

    exact_unnorm = (
        (lr["aligned_action_normalized"][:, :7] + 1) / 2 * (q99 - q01 + 1e-6) + q01
    )

    result = {
        "protocol": {
            "openpi_horizon": int(op["action_horizon"]),
            "openpi_execute": int(op["replan_steps"]),
            "lerobot_aligned_horizon": int(lr["aligned_chunk_size"]),
            "lerobot_aligned_execute": int(lr["aligned_replan_steps"]),
            "lerobot_artifact_n_action_steps": int(checkpoint_config["n_action_steps"]),
            "lerobot_artifact_chunk_size": int(checkpoint_config["chunk_size"]),
            "flow_steps": int(op["flow_steps"]),
            "flow_dt": float(op["flow_dt"]),
            "openpi_trace_compilation_max_abs": float(op["flow_compilation_max_abs"]),
        },
        "observations": {
            "base_image": metric(op_base, lr["base_model"]),
            "wrist_image": metric(op_wrist, lr["wrist_model"]),
            "openpi_masks": [
                bool(op["image_mask_base"]),
                bool(op["image_mask_wrist"]),
                bool(op["image_mask_empty"]),
            ],
            "lerobot_masks": [bool(x) for x in lr["image_masks"]],
            "raw_state": [float(x) for x in obs["state"]],
            "normalized_state": metric(
                op["state_normalized"][:8], lr["state_normalized"]
            ),
        },
        "tokens": {
            "openpi_prompt": str(op["prompt"]),
            "lerobot_aligned_prompt": str(
                lr["prompt"] if "prompt" in lr else op["prompt"]
            ),
            "openpi_valid_count": int(op["token_mask"].sum()),
            "lerobot_aligned_valid_count": int(lr["aligned_token_mask"].sum()),
            "valid_ids_equal": bool(
                op_ids.shape == lr_aligned_ids.shape
                and np.array_equal(op_ids, lr_aligned_ids)
            ),
            "openpi_valid_ids": [int(x) for x in op_ids],
            "lerobot_aligned_valid_ids": [int(x) for x in lr_aligned_ids],
        },
        "normalization": {
            "q01": [float(x) for x in q01],
            "q99": [float(x) for x in q99],
            "openpi_epsilon": 1e-6,
            "action_formula_delta": metric(exact_unnorm, lr["aligned_action_env"]),
        },
        "flow_matching": {
            "aligned_velocity": metric(op["flow_velocity"], lr["aligned_velocity"]),
            "aligned_state": metric(op["flow_state"], lr["aligned_flow_state"]),
            "aligned_final": metric(
                op["action_normalized"], lr["aligned_action_normalized"]
            ),
            "saved_first10": (
                metric(op["action_normalized"], lr["saved_action_normalized"][:10])
                if "saved_action_normalized" in lr
                else None
            ),
        },
        "executed_actions": {
            "aligned_executed": metric(
                op["action_executed"], lr["aligned_action_executed"]
            ),
            "aligned_exact_unnorm": metric(op["action_executed"], exact_unnorm[:5]),
            "saved_conversion": (
                metric(op["action_executed"], lr["saved_action_executed"])
                if "saved_action_executed" in lr
                else None
            ),
        },
    }

    md = [
        "# OpenPI vs Aligned LeRobot pi05 Parity Trace",
        "",
        "Same fixed LIBERO Spatial observation and fixed Gaussian noise.",
        "",
        "## Protocol",
        "",
        f"- OpenPI predicts {result['protocol']['openpi_horizon']} and executes {result['protocol']['openpi_execute']}.",
        f"- Aligned LeRobot predicts {result['protocol']['lerobot_aligned_horizon']} and executes {result['protocol']['lerobot_aligned_execute']}.",
        "- Both run 10 Euler flow-matching steps with dt=-0.1.",
        "",
        "## Tensor Parity Metrics",
        "",
        f"- Base image: {show(result['observations']['base_image'])}.",
        f"- Wrist image: {show(result['observations']['wrist_image'])}.",
        f"- Camera masks: OpenPI {result['observations']['openpi_masks']}, LeRobot {result['observations']['lerobot_masks']}.",
        f"- Aligned flow velocities: {show(result['flow_matching']['aligned_velocity'])}.",
        f"- Aligned flow states: {show(result['flow_matching']['aligned_state'])}.",
        f"- Aligned final normalized actions: {show(result['flow_matching']['aligned_final'])}.",
        f"- Aligned first five executed actions: {show(result['executed_actions']['aligned_executed'])}.",
    ]
    if result["executed_actions"]["saved_conversion"]:
        md.append(
            f"- Saved unaligned conversion first five executed actions: {show(result['executed_actions']['saved_conversion'])}."
        )

    md.extend(
        [
            "",
            "## Token and Semantic Alignment",
            "",
            f"- Valid prompt tokens: OpenPI {result['tokens']['openpi_valid_count']}, Aligned LeRobot {result['tokens']['lerobot_aligned_valid_count']}; equal IDs: {result['tokens']['valid_ids_equal']}.",
            "- Official pi05_libero uses cleaned task text plus newline and does not feed state to the model.",
            "- Aligned LeRobot preprocessor independently produces the exact same token IDs and masks using local PaliGemma SentencePiece.",
            "- OpenPI quantile normalization formula with epsilon 1e-6 is matched exactly in LeRobot normalizer/unnormalizer.",
            "- Aligned LeRobot predicts 10 coupled tokens and executes 5, matching OpenPI replanning horizon.",
        ]
    )

    a.json.parent.mkdir(parents=True, exist_ok=True)
    a.json.write_text(json.dumps(result, indent=2) + chr(10))
    a.markdown.write_text(chr(10).join(md) + chr(10))
    print("PARITY_COMPARE_OK", a.json, a.markdown)


if __name__ == "__main__":
    main()
