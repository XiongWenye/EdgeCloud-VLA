#!/usr/bin/env python3
"""Aggregate horizon-matched bootstrap and edge-zero V3 diagnostics."""

import json
from pathlib import Path

ROOT = Path("/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA")
DIAG = ROOT / "results/cloudedge_pi05_v3_diagnostics"
CAND = ROOT / "results/cloudedge_pi05_v3_candidates"


def rate(path: Path) -> float:
    data = json.loads(path.read_text())["overall"]
    if int(data["n_episodes"]) != 100:
        raise ValueError(f"Expected 100 episodes: {path}")
    return float(data["pc_success"])


def eval_path(root: Path, delay: int) -> Path:
    return root / f"dmax_{delay}/libero_spatial/seed_7/eval_info.json"


def main() -> None:
    bootstrap = DIAG / "bootstrap_h1"
    edgezero = DIAG / "best_lr2e-6_step625_edgezero"
    edgeon = CAND / "lr2e-6/step_000625"
    rows = []
    for delay in (0, 10):
        base = rate(eval_path(bootstrap, delay))
        on = rate(eval_path(edgeon, delay))
        zero = rate(eval_path(edgezero, delay))
        rows.append(
            {
                "delay_max": delay,
                "bootstrap_h1_success_percent": base,
                "trained_edge_on_success_percent": on,
                "trained_edge_zero_success_percent": zero,
                "training_gain_points": on - base,
                "edge_feature_contribution_points": on - zero,
            }
        )
    result = {
        "protocol": "LIBERO Spatial, seed 7, 100 episodes/point, n_action_steps=1",
        "trained_checkpoint": "v3 lr2e-6 step625",
        "rows": rows,
    }
    json_path = ROOT / "results/cloudedge_pi05_v3_diagnostics_summary.json"
    md_path = ROOT / "results/cloudedge_pi05_v3_diagnostics_summary.md"
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# CloudEdge pi0.5 v3 horizon-matched diagnostics",
        "",
        "LIBERO Spatial, seed 7, 100 episodes/point, action horizon 1.",
        "",
        "| dmax | Bootstrap h1 | Trained edge on | Trained edge zero | Training gain | Edge contribution |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['delay_max']} | {row['bootstrap_h1_success_percent']:.1f}% | "
            f"{row['trained_edge_on_success_percent']:.1f}% | "
            f"{row['trained_edge_zero_success_percent']:.1f}% | "
            f"{row['training_gain_points']:+.1f} pp | "
            f"{row['edge_feature_contribution_points']:+.1f} pp |"
        )
    md_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
