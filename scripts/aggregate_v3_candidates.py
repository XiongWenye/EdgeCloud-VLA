#!/usr/bin/env python3
"""Aggregate and gate post-flow CloudEdge pi0.5 v3 candidates."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("results/cloudedge_pi05_v3_candidates")
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("results/cloudedge_pi05_v3_candidates_summary.json"),
    )
    parser.add_argument(
        "--md-output",
        type=Path,
        default=Path("results/cloudedge_pi05_v3_candidates_summary.md"),
    )
    parser.add_argument("--d0-threshold", type=float, default=94.0)
    parser.add_argument("--d10-threshold", type=float, default=46.0)
    args = parser.parse_args()

    records = []
    for path in sorted(
        args.root.glob("*/step_*/dmax_*/libero_spatial/seed_7/eval_info.json")
    ):
        records.append(
            {
                "candidate": path.parents[4].name,
                "step": int(path.parents[3].name.removeprefix("step_")),
                "delay_max": int(path.parents[2].name.removeprefix("dmax_")),
                "success_percent": float(
                    json.loads(path.read_text())["overall"]["pc_success"]
                ),
                "n_episodes": int(
                    json.loads(path.read_text())["overall"]["n_episodes"]
                ),
                "source": str(path),
            }
        )
    labels = ["lr2e-6", "lr5e-6"]
    steps = [625, 1250, 2500]
    delays = [0, 10]
    lookup = {
        (item["candidate"], item["step"], item["delay_max"]): item
        for item in records
    }
    gates = []
    for label in labels:
        for step in steps:
            d0 = lookup.get((label, step, 0))
            d10 = lookup.get((label, step, 10))
            if d0 is None or d10 is None:
                continue
            admitted = (
                d0["success_percent"] >= args.d0_threshold
                and d10["success_percent"] >= args.d10_threshold
            )
            gates.append(
                {
                    "candidate": label,
                    "step": step,
                    "d0_success_percent": d0["success_percent"],
                    "d10_success_percent": d10["success_percent"],
                    "admitted": admitted,
                    "selection_score": d0["success_percent"]
                    + d10["success_percent"],
                }
            )
    admitted = [item for item in gates if item["admitted"]]
    winner = max(admitted, key=lambda item: item["selection_score"]) if admitted else None
    complete = all(
        (label, step, delay) in lookup
        for label in labels
        for step in steps
        for delay in delays
    )
    summary = {
        "complete": complete,
        "protocol": "LIBERO Spatial, seed 7, 100 episodes per point",
        "d0_threshold": args.d0_threshold,
        "d10_threshold": args.d10_threshold,
        "strict_v2_comparison": "d10 must exceed v2 winner's 45%",
        "records": records,
        "candidate_gates": gates,
        "winner": winner,
    }
    args.json_output.write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# CloudEdge pi0.5 v3 candidate gate",
        "",
        "LIBERO Spatial, seed 7, 10 episodes per task (100 episodes per point).",
        "",
        f"Admission: d0 >= {args.d0_threshold:.1f}% and d10 >= {args.d10_threshold:.1f}%.",
        "The d10 gate is a strict improvement over the v2 winner (45%).",
        "",
        "| Candidate | Step | d0 | d10 | Admitted |",
        "|---|---:|---:|---:|:---:|",
    ]
    for label in labels:
        for step in steps:
            item = next(
                (
                    candidate
                    for candidate in gates
                    if candidate["candidate"] == label and candidate["step"] == step
                ),
                None,
            )
            if item is None:
                lines.append(f"| {label} | {step:,} | pending | pending | no |")
            else:
                lines.append(
                    f"| {label} | {step:,} | {item['d0_success_percent']:.1f}% | "
                    f"{item['d10_success_percent']:.1f}% | "
                    f"{'yes' if item['admitted'] else 'no'} |"
                )
    lines.extend(
        [
            "",
            f"Complete: **{complete}**",
            f"Winner: **{winner['candidate'] + ' step ' + str(winner['step']) if winner else 'none'}**",
            "",
        ]
    )
    args.md_output.write_text("\n".join(lines))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
