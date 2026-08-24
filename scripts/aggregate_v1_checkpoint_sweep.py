#!/usr/bin/env python3
"""Aggregate the v1 checkpoint sweep into machine-readable and Markdown summaries."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/diagnostics_v1_checkpoint_sweep"),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("results/diagnostics_v1_checkpoint_sweep_summary.json"),
    )
    parser.add_argument(
        "--md-output",
        type=Path,
        default=Path("results/diagnostics_v1_checkpoint_sweep_summary.md"),
    )
    args = parser.parse_args()

    records = []
    for path in sorted(args.root.glob("step_*/dmax_*/libero_spatial/seed_7/eval_info.json")):
        step = int(path.parents[3].name.removeprefix("step_"))
        delay = int(path.parents[2].name.removeprefix("dmax_"))
        data = json.loads(path.read_text())
        overall = data["overall"]
        records.append(
            {
                "step": step,
                "delay_max": delay,
                "success_percent": float(overall["pc_success"]),
                "n_episodes": int(overall["n_episodes"]),
                "eval_seconds": float(overall["eval_s"]),
                "source": str(path),
            }
        )

    steps = sorted({record["step"] for record in records})
    delays = sorted({record["delay_max"] for record in records})
    lookup = {(record["step"], record["delay_max"]): record for record in records}
    complete = all((step, delay) in lookup for step in steps for delay in delays)

    summary = {
        "suite": "libero_spatial",
        "seed": 7,
        "expected_steps": [5000, 20000, 40000, 60000, 80000, 120000],
        "expected_delays": [0, 10],
        "complete": complete
        and steps == [5000, 20000, 40000, 60000, 80000, 120000]
        and delays == [0, 10],
        "records": records,
    }
    args.json_output.write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# v1 checkpoint sweep",
        "",
        "LIBERO Spatial, seed 7, 5 episodes per task (50 episodes per point).",
        "",
        "| Step | d=0 | d=10 | delta (d10-d0) |",
        "|---:|---:|---:|---:|",
    ]
    for step in steps:
        d0 = lookup.get((step, 0))
        d10 = lookup.get((step, 10))
        d0_text = f"{d0['success_percent']:.1f}%" if d0 else "pending"
        d10_text = f"{d10['success_percent']:.1f}%" if d10 else "pending"
        delta_text = (
            f"{d10['success_percent'] - d0['success_percent']:+.1f} pp"
            if d0 and d10
            else "pending"
        )
        lines.append(f"| {step:,} | {d0_text} | {d10_text} | {delta_text} |")
    lines.extend(["", f"Complete: **{summary['complete']}**", ""])
    args.md_output.write_text("\n".join(lines))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
