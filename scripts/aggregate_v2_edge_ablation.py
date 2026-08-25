#!/usr/bin/env python3
"""Aggregate seed-matched edge-on versus edge-zero CloudEdge pi0.5 v2 results."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
DELAYS = [0, 10, 20, 40]


def load_rate(root: Path, suite: str, delay: int, seed: int) -> float:
    path = root / f"dmax_{delay}" / suite / f"seed_{seed}" / "eval_info.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    overall = json.loads(path.read_text())["overall"]
    if overall["n_episodes"] != 500:
        raise ValueError(f"{path}: expected 500 episodes, got {overall['n_episodes']}")
    return float(overall["pc_success"])


def main() -> None:
    edge_on_root = ROOT / "results/cloudedge_pi05_v2_full/raw"
    edge_zero_root = ROOT / "results/cloudedge_pi05_v2_edgezero/raw"
    records = []
    for suite in SUITES:
        for delay in DELAYS:
            edge_on = load_rate(edge_on_root, suite, delay, 7)
            edge_zero = load_rate(edge_zero_root, suite, delay, 7)
            records.append(
                {
                    "suite": suite,
                    "delay_max": delay,
                    "seed": 7,
                    "edge_on_success_percent": edge_on,
                    "edge_zero_success_percent": edge_zero,
                    "edge_contribution_points": edge_on - edge_zero,
                }
            )

    macro = []
    for delay in DELAYS:
        rows = [row for row in records if row["delay_max"] == delay]
        edge_on = sum(row["edge_on_success_percent"] for row in rows) / len(rows)
        edge_zero = sum(row["edge_zero_success_percent"] for row in rows) / len(rows)
        macro.append(
            {
                "delay_max": delay,
                "edge_on_success_percent": edge_on,
                "edge_zero_success_percent": edge_zero,
                "edge_contribution_points": edge_on - edge_zero,
            }
        )

    result = {
        "protocol": {
            "checkpoint": "cloudedge_pi05_v2_lr5e-6 step 2500",
            "seed": 7,
            "episodes_per_suite_point": 500,
            "edge_zero_definition": "trained edge action head with a zero edge feature vector",
        },
        "records": records,
        "macro": macro,
    }
    output_json = ROOT / "results/cloudedge_pi05_v2_edge_ablation_summary.json"
    output_md = ROOT / "results/cloudedge_pi05_v2_edge_ablation_summary.md"
    output_json.write_text(json.dumps(result, indent=2) + "\n")

    lines = [
        "# CloudEdge pi0.5 v2 edge-feature ablation",
        "",
        "Winner: lr=5e-6, step=2500. Seed-matched closed-loop evaluation, "
        "500 episodes per suite/point.",
        "",
        "Edge-zero keeps the trained action head and replaces only its edge feature "
        "with zeros. It does not disable the complete residual head.",
        "",
        "| Suite | dmax | Edge on | Edge zero | Edge contribution |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in records:
        lines.append(
            f"| {row['suite']} | {row['delay_max']} | "
            f"{row['edge_on_success_percent']:.1f}% | "
            f"{row['edge_zero_success_percent']:.1f}% | "
            f"{row['edge_contribution_points']:+.1f} pp |"
        )
    lines.extend(
        [
            "",
            "## Four-suite macro average",
            "",
            "| dmax | Edge on | Edge zero | Edge contribution |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in macro:
        lines.append(
            f"| {row['delay_max']} | {row['edge_on_success_percent']:.2f}% | "
            f"{row['edge_zero_success_percent']:.2f}% | "
            f"{row['edge_contribution_points']:+.2f} pp |"
        )
    output_md.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
