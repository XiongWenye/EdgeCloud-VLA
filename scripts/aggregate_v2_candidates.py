#!/usr/bin/env python3
"""Aggregate and gate short CloudEdge pi0.5 v2 candidates."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/cloudedge_pi05_v2_candidates"),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("results/cloudedge_pi05_v2_candidates_summary.json"),
    )
    parser.add_argument(
        "--md-output",
        type=Path,
        default=Path("results/cloudedge_pi05_v2_candidates_summary.md"),
    )
    parser.add_argument("--d0-threshold", type=float, default=85.0)
    parser.add_argument("--d10-threshold", type=float, default=30.0)
    args = parser.parse_args()

    records = []
    for path in sorted(args.root.glob("*/step_*/dmax_*/libero_spatial/seed_7/eval_info.json")):
        label = path.parents[4].name
        step = int(path.parents[3].name.removeprefix("step_"))
        delay = int(path.parents[2].name.removeprefix("dmax_"))
        overall = json.loads(path.read_text())["overall"]
        records.append(
            {
                "candidate": label,
                "step": step,
                "delay_max": delay,
                "success_percent": float(overall["pc_success"]),
                "n_episodes": int(overall["n_episodes"]),
                "source": str(path),
            }
        )

    labels = ["lr5e-6", "lr1e-5"]
    steps = [2500, 5000, 10000]
    delays = [0, 10]
    lookup = {
        (record["candidate"], record["step"], record["delay_max"]): record
        for record in records
    }
    candidates = []
    for label in labels:
        for step in steps:
            d0 = lookup.get((label, step, 0))
            d10 = lookup.get((label, step, 10))
            if not d0 or not d10:
                continue
            admitted = (
                d0["success_percent"] >= args.d0_threshold
                and d10["success_percent"] >= args.d10_threshold
            )
            candidates.append(
                {
                    "candidate": label,
                    "step": step,
                    "d0_success_percent": d0["success_percent"],
                    "d10_success_percent": d10["success_percent"],
                    "admitted": admitted,
                    "selection_score": d0["success_percent"] + d10["success_percent"],
                }
            )

    admitted = [candidate for candidate in candidates if candidate["admitted"]]
    winner = max(admitted, key=lambda item: item["selection_score"]) if admitted else None
    complete = all((label, step, delay) in lookup for label in labels for step in steps for delay in delays)
    summary = {
        "complete": complete,
        "d0_threshold": args.d0_threshold,
        "d10_threshold": args.d10_threshold,
        "records": records,
        "candidate_gates": candidates,
        "winner": winner,
    }
    args.json_output.write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# CloudEdge pi0.5 v2 candidate gate",
        "",
        "LIBERO Spatial, seed 7, 10 episodes per task (100 episodes per point).",
        "",
        f"Admission: d0 >= {args.d0_threshold:.1f}% and d10 >= {args.d10_threshold:.1f}%.",
        "",
        "| Candidate | Step | d0 | d10 | Admitted |",
        "|---|---:|---:|---:|:---:|",
    ]
    for label in labels:
        for step in steps:
            item = next(
                (
                    candidate
                    for candidate in candidates
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
